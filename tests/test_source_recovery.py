import unittest
from app.rag.source_recovery import SourceRecovery


def chunk(text, ident, page=1):
    return {'content': text, 'metadata': {'chunk_id': ident, 'page': page}}


def row(quote, ident=1, source=None, page=1):
    return {'quote': quote, 'chunk_id': ident, 'source_id': source or f'c{ident}', 'page': page, 'line': 1}


def recover(page, evidence, chunks=None):
    topic = [{'title': 'Provas', 'subtopics': [{'title': 'Identificação', 'evidence': evidence}]}]
    SourceRecovery([page], chunks or []).enrich(topic)
    return topic[0]['subtopics'][0]['source_blocks']


class RecoveryTests(unittest.TestCase):
    def test_user_example_restores_clauses_and_deduplicates_chunks(self):
        text = ('11.5 O candidato deverá comparecer com antecedência mínima de 60 minutos do horário fixado para o\n'
                'fechamento do portão de acesso ao local de realização da prova, munido de caneta esferográfica transparente,\n'
                'de tinta azul ou preta, seu documento oficial de identificação com foto e o Cartão de Informação do\n'
                'Candidato.\n11.5.1 São considerados documentos de identidade as carteiras e/ou cédulas de identidade expedidas pelas Secretarias\n'
                'de Segurança Pública e pelos órgãos profissionais competentes.\n11.6 Não será permitido ingresso após o fechamento.')
        rows = [row('11.5 O candidato deverá comparecer com antecedência mínima de 60 minutos do horário fixado para o'),
                row('fechamento do portão de acesso ao local de realização da prova, munido de caneta esferográfica transparente,', 2),
                row('11.5.1 São considerados documentos de identidade as carteiras e/ou cédulas de identidade expedidas pelas Secretarias', 3)]
        blocks = recover(text, rows)
        self.assertEqual([b['item'] for b in blocks], ['11.5', '11.5.1'])
        self.assertEqual(blocks[0]['text'].count('fechamento do portão'), 1)
        self.assertIn('Cartão de Informação do Candidato.', blocks[0]['text'])
        self.assertIn('órgãos profissionais competentes.', blocks[1]['text'])
        self.assertNotIn('11.6', blocks[1]['text'])
        self.assertEqual(blocks[0]['source_ids'], ['c1', 'c2'])

    def test_similar_text_in_different_clauses_is_not_deduplicated(self):
        text = '2.1 Documento obrigatório para inscrição.\n2.2 Documento obrigatório para posse.'
        blocks = recover(text, [row('2.1 Documento obrigatório'), row('2.2 Documento obrigatório', 2)])
        self.assertEqual(len(blocks), 2)

    def test_ambiguous_quote_does_not_guess(self):
        text = '2.1 É obrigatório apresentar documento.\n2.2 É obrigatório apresentar documento.'
        blocks = recover(text, [row('É obrigatório apresentar documento.')])
        self.assertEqual(blocks[0]['context_status'], 'selected_only')
        self.assertFalse(blocks[0]['expanded'])

    def test_unique_chunk_disambiguates_repeated_quote(self):
        text = '2.1 É obrigatório apresentar documento.\n2.2 É obrigatório apresentar documento.'
        blocks = recover(text, [row('É obrigatório apresentar documento.')],
                         [chunk('2.2 É obrigatório apresentar documento.', 1)])
        self.assertEqual(blocks[0]['item'], '2.2')

    def test_non_numbered_paragraph_uses_original_context_only(self):
        text = 'Título\n\nO candidato deve trazer documento\ne caneta transparente.\n\nOutra informação.'
        blocks = recover(text, [row('documento')])
        self.assertEqual(blocks[0]['text'], 'O candidato deve trazer documento e caneta transparente.')
        self.assertNotIn('Outra informação', blocks[0]['text'])

    def test_lists_preserved_and_next_clause_excluded(self):
        text = '16.25 Quem informar dados falsos estará\nsujeito:\na) à exclusão antes do resultado;\nb) à anulação após o resultado.\n16.25.1Detectada falsidade, será assegurada defesa.'
        blocks = recover(text, [row('Quem informar dados falsos estará')])
        self.assertIn('\n\na) à exclusão', blocks[0]['text'])
        self.assertNotIn('16.25.1', blocks[0]['text'])

    def test_confirmed_continuation_keeps_both_page_references(self):
        topics = [{'subtopics': [{'evidence': [row('2.1 Texto que continua'), row('na outra página.', page=2)]}]}]
        SourceRecovery(['2.1 Texto que continua', 'na outra página.'], []).enrich(topics)
        blocks = topics[0]['subtopics'][0]['source_blocks']
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]['pages'], [1, 2])
        self.assertEqual(blocks[0]['item'], '2.1')
        self.assertEqual(blocks[0]['text'], '2.1 Texto que continua na outra página.')

    def test_contact_continuation_restores_beginning_and_excludes_footer_next_item(self):
        pages = ["22.12 O candidato deverá solicitar atualização por e-mail, até a homologação do\nEdital X Página 1 de 2",
                 "certame. Em caso de dúvida, ligue para o atendimento.\n22.12.1 O órgão não se responsabiliza pelo endereço incorreto."]
        topics = [{"subtopics": [{"pages": [2], "evidence": [row("certame. Em caso de dúvida", page=2),
            row("22.12 O candidato", page=1)]}]}]
        SourceRecovery(pages, []).enrich(topics)
        blocks = topics[0]["subtopics"][0]["source_blocks"]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["pages"], [1, 2])
        self.assertTrue(blocks[0]["text"].startswith("22.12 O candidato"))
        self.assertIn("homologação do certame.", blocks[0]["text"])
        self.assertNotIn("Edital X", blocks[0]["text"])
        self.assertNotIn("22.12.1", blocks[0]["text"])

    def test_independent_uppercase_paragraph_is_not_joined_to_previous_page(self):
        topics = [{"subtopics": [{"evidence": [row("2.1 Texto interrompido", page=1), row("Novo assunto.", page=2)]}]}]
        SourceRecovery(["2.1 Texto interrompido", "Novo assunto."], []).enrich(topics)
        blocks = topics[0]["subtopics"][0]["source_blocks"]
        self.assertEqual([b["page"] for b in blocks], [1, 2])
        self.assertTrue(blocks[0]["continuation_pending"])

    def test_missing_page_or_quote_preserves_verified_selection(self):
        blocks = recover('Outro texto.', [row('Regra selecionada.')])
        self.assertEqual(blocks[0]['text'], 'Regra selecionada.')
        self.assertEqual(blocks[0]['context_status'], 'selected_only')

    def test_whitespace_normalization_preserves_source_words(self):
        text = '4.1 Traga documento\n com foto e\n caneta preta.\n4.2 Outra regra.'
        blocks = recover(text, [row('Traga documento com foto')])
        self.assertEqual(blocks[0]['text'], '4.1 Traga documento com foto e caneta preta.')

    def test_very_large_clause_does_not_publish_whole_page(self):
        text = '4.1 ' + ('conteúdo ' * 2000) + 'fim exclusivo.'
        blocks = recover(text, [row('fim exclusivo.')])
        self.assertEqual(blocks[0]['text'], 'fim exclusivo.')


if __name__ == '__main__': unittest.main()
