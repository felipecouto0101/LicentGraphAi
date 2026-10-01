import ast
import html
import unittest
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
from app.rag.source_display import evidence_blocks
from app.rag.topic_organizer import _source_line_records, _source_lines, _validate_extraction


def evidence(line, quote, page=29, source='c1', **kw):
    return dict(line=line, quote=quote, page=page, source_id=source, chunk_id=1, **kw)


class SourceBox:
    open = True
    def __enter__(self): return self
    def __exit__(self, *args): pass


class SourceDisplayTests(unittest.TestCase):
    def test_example_reflows_wrapped_sentences_and_preserves_list(self):
        rows = [evidence(1, '16.25 Quem prestar informação falsa estará'),
                evidence(2, 'sujeito:'), evidence(3, 'a) ao cancelamento da inscrição'),
                evidence(4, 'antes da homologação;'), evidence(5, 'b) à exclusão da lista de aprovados.')]
        blocks = evidence_blocks(rows)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]['text'], '16.25 Quem prestar informação falsa estará sujeito:'
            '\n\na) ao cancelamento da inscrição antes da homologação;\n\nb) à exclusão da lista de aprovados.')

    def test_gaps_pages_and_sources_remain_separate(self):
        rows = [evidence(1, 'Uma regra'), evidence(3, 'Outra regra'),
                evidence(4, 'Em outra página', page=30), evidence(4, 'Outra fonte', source='c2')]
        self.assertEqual(len(evidence_blocks(rows)), 4)
        self.assertNotIn('Outra regra', evidence_blocks(rows)[0]['text'])

    def test_source_line_order_and_duplicate_removal(self):
        rows = [evidence(2, 'com continuação.'), evidence(1, 'Uma frase'), evidence(1, 'Uma frase')]
        self.assertEqual(evidence_blocks(rows)[0]['text'], 'Uma frase com continuação.')

    def test_original_paragraphs_and_midword_splits_are_preserved(self):
        text = 'a' * 170 + '\ncontinuação do parágrafo.\n\nNovo parágrafo.\n1. Outra cláusula.'
        records = _source_line_records(text)
        rows = [evidence(i + 1, record['text'], join_before=record['join_before']) for i, record in enumerate(records)]
        self.assertEqual(evidence_blocks(rows)[0]['text'], 'a' * 170 + ' continuação do parágrafo.'
                         '\n\nNovo parágrafo.\n\n1. Outra cláusula.')
        self.assertEqual(_source_lines(text), [r['text'] for r in records])

    def test_extraction_carries_original_paragraph_separator(self):
        text = 'Primeiro parágrafo.\n\nSegundo parágrafo.'
        batch = [{'id': 'c1', 'text': text, 'page': 29, 'chunk_id': 1, 'section': []}]
        parsed = {'topics': [{'theme': 'Regras', 'title': 'Participação', 'sources': [{'id': 'p1', 'lines': [1, 2]}]}]}
        rows = _validate_extraction(parsed, batch)[0]['evidence']
        self.assertEqual(evidence_blocks(rows)[0]['text'], text)

    def test_legacy_without_line_ids_stays_separate(self):
        self.assertEqual(len(evidence_blocks([{'page': 1, 'quote': 'A'}, {'page': 1, 'quote': 'B'}])), 2)

    def test_frontend_removes_section_labels_and_escapes_pdf_html(self):
        tree = ast.parse((Path(__file__).parents[1] / 'app/streamlit_app.py').read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_subtopic_card')
        calls = []
        st = SimpleNamespace(container=lambda **kw: nullcontext(),
            expander=lambda label, **kw: calls.append(('expander', label)) or SourceBox())
        for name in ('subheader', 'caption', 'markdown'):
            setattr(st, name, lambda text, _name=name, **kw: calls.append((_name, text)))
        namespace = dict(st=st, html=html, evidence_blocks=evidence_blocks, _pages=lambda p: '29')
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'ui', 'exec'), namespace)
        namespace['_subtopic_card']({'title': 'Sanções', 'pages': [29], 'source_titles': ['Visão geral da seção'],
            'evidence': [evidence(1, '<script>texto</script>'), evidence(2, 'continuação')]}, True)
        self.assertEqual(sum(name == 'caption' and 'Fonte:' in text for name, text in calls), 1)
        self.assertFalse(any('Seções vinculadas' in text for _, text in calls))
        self.assertTrue(any(name == 'markdown' and '&lt;script&gt;' in text for name, text in calls))


if __name__ == '__main__': unittest.main()
