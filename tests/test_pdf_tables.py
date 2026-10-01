import unittest
from types import SimpleNamespace
from app.rag.pdf_tables import extract_page_tables, table_html, is_tabular, link_table_continuations
from app.rag.source_recovery import SourceRecovery, _BOUNDARY


class TableTests(unittest.TestCase):
    def test_geometry_retains_merged_cells_and_exact_values(self):
        cells = [(0,10,20,20), (0,20,10,40), (10,20,20,30), (10,30,20,40)]
        table = SimpleNamespace(bbox=(0,10,20,40), cells=cells,
            rows=[SimpleNamespace(cells=[cells[0],None]), SimpleNamespace(cells=[cells[1],cells[2]]),
                  SimpleNamespace(cells=[None,cells[3]])],
            extract=lambda: [['Título',None], ['Cargo','12'], [None,'6']])
        values = [('TABELA 1.1',0), ('Título',12), ('Cargo 12',22), ('6',32), ('Fim da tabela',45)]
        page = SimpleNamespace(find_tables=lambda:[table], extract_text_lines=lambda:[
            {'text':text,'top':top,'bottom':top+4} for text,top in values])
        text='\n'.join(text for text,_ in values)
        result=extract_page_tables(page,text,lambda value,**kw:value)
        self.assertEqual(result[0]['title'],'TABELA 1.1')
        self.assertEqual(result[0]['cells'][0]['colspan'],2)
        self.assertEqual(result[0]['cells'][1]['rowspan'],2)
        self.assertEqual(result[0]['rows'][2][1],'6')
        self.assertEqual(text[result[0]['start']:result[0]['end']].strip(),'Título\nCargo 12\n6')

    def test_html_is_escaped_and_retains_rowspan(self):
        markup=table_html({'row_count':1,'cells':[{'row':0,'col':0,'rowspan':2,'colspan':3,'text':'<script>x</script>'}]})
        self.assertIn('rowspan="2"',markup); self.assertIn('colspan="3"',markup)
        self.assertIn('&lt;script&gt;',markup); self.assertNotIn('<script>',markup)

    def test_medical_cargo_codes_are_not_clause_boundaries(self):
        matches=list(_BOUNDARY.finditer('413.3 - Médico/Cirurgia Geral\n408.1 - Médico/Generalista\n410.1 - Músico/Trompete\n10.2 Conteúdos da prova.'))
        self.assertEqual([m.group('item') for m in matches],['10.2'])

    def test_table_not_flattened_into_clause(self):
        text='10.1 Fases da prova:\nTABELA 10.1\nCARGO FASE\n413.3 - Médico 1ª Objetiva\n10.2 Outra regra.'
        start,end=text.index('CARGO'),text.index('10.2')
        table={'start':start,'end':end,'title':'TABELA 10.1','cells':[],'row_count':0}
        rows=[{'page':1,'quote':q,'source_id':str(i),'chunk_id':i,'line':1} for i,q in enumerate(['10.1 Fases','413.3 - Médico'])]
        topics=[{'subtopics':[{'evidence':rows}]}]
        SourceRecovery([text],[],[[table]]).enrich(topics)
        blocks=topics[0]['subtopics'][0]['source_blocks']
        self.assertEqual(blocks[0]['item'],'10.1'); self.assertNotIn('CARGO',blocks[0]['text'])
        self.assertEqual(blocks[1]['context_status'],'table'); self.assertIsNotNone(blocks[1]['table'])

    def test_repeated_selection_hidden_when_all_occurrences_are_in_tables(self):
        text='Tabela A\nCARGO FASE\n201 - Técnico\nTabela B\nCARGO FASE\n202 - Técnico'
        split=text.index('Tabela B')
        tables=[{'start':0,'end':split,'title':'A','cells':[],'row_count':0},
                {'start':split,'end':len(text),'title':'B','cells':[],'row_count':0}]
        rows=[{'page':1,'quote':q,'source_id':str(i),'chunk_id':i,'line':1}
              for i,q in enumerate(['201 - Técnico','202 - Técnico','CARGO FASE','CARGO FASE'])]
        topics=[{'subtopics':[{'evidence':rows}]}]
        SourceRecovery([text],[],[tables]).enrich(topics)
        blocks=topics[0]['subtopics'][0]['source_blocks']
        self.assertEqual(len(blocks),2)
        self.assertTrue(all(block['context_status']=='table' for block in blocks))

    def test_uncovered_ambiguous_selection_is_retained_once(self):
        text='CARGO FASE\nTexto\nCARGO FASE'
        evidence=[{'page':1,'quote':'CARGO FASE','source_id':str(i),'chunk_id':i,'line':1} for i in range(2)]
        topics=[{'subtopics':[{'evidence':evidence}]}]
        SourceRecovery([text],[]).enrich(topics)
        self.assertEqual(len(topics[0]['subtopics'][0]['source_blocks']),1)
        self.assertEqual(topics[0]['subtopics'][0]['source_blocks'][0]['context_status'],'selected_only')

    def test_numbered_text_inside_table_never_becomes_clause(self):
        text='Dados\n11.1 Nome de modelo\n11.2 Segunda linha\n12.1 Regra após tabela.'
        table={'start':0,'end':text.index('12.1'),'title':'Dados','cells':[],'row_count':0}
        recovery=SourceRecovery([text],[],[[table]])
        self.assertEqual([m.group('item') for m in recovery.boundaries[1]],['12.1'])

    def test_prose_and_clause_layout_are_not_tables(self):
        self.assertFalse(is_tabular([[None, "quando"], ["uma frase longa", None], ["a continuação", None]]))
        self.assertFalse(is_tabular([["um parágrafo"], ["sua continuação"]]))
        self.assertFalse(is_tabular([["22.1", "", "Uma frase longa " * 10], ["22.2", "", "Outro parágrafo " * 10]]))
        self.assertTrue(is_tabular([["Descrição", "Preço"], ["Produto", "100"]]))

    def test_adjacent_aligned_table_grids_link_without_inventing_rows(self):
        a = {"id": "table-0", "title": "TABELA 2.1", "bbox": [30,600,560,780], "page_height": 842,
             "column_count": 9, "rows": [["205", "Cargo A"]]}
        b = {"id": "table-0", "title": "Tabela do PDF", "bbox": [30,36,560,390], "page_height": 842,
             "column_count": 9, "rows": [["206", "Cargo B"]]}
        link_table_continuations([[a], [b]])
        self.assertEqual(a["continuations"], [{"page": 2, "id": "table-0"}])
        self.assertEqual(b["title"], "TABELA 2.1 · continuação")
        self.assertEqual(b["rows"], [["206", "Cargo B"]])

    def test_new_caption_or_different_columns_prevent_table_link(self):
        a = {"id": "a", "title": "TABELA 1", "bbox": [0,90,50,100], "page_height": 100, "column_count": 3}
        b = {"id": "b", "title": "TABELA 2", "bbox": [0,0,50,50], "column_count": 3}
        link_table_continuations([[a], [b]])
        self.assertNotIn("continuations", a)
        b["title"], b["column_count"] = "Tabela do PDF", 2
        link_table_continuations([[a], [b]])
        self.assertNotIn("continuations", a)

    def test_selection_in_table_continuation_includes_original_page(self):
        a = {"id": "a", "start": 0, "end": 6, "title": "Tabela", "cells": [], "row_count": 0,
             "continuations": [{"page": 2, "id": "b"}]}
        b = {"id": "b", "start": 0, "end": 6, "title": "Tabela · continuação", "cells": [], "row_count": 0,
             "continued_from": {"page": 1, "id": "a"}}
        topics = [{"subtopics": [{"pages": [2], "evidence": [
            {"page": 2, "quote": "DadosB", "chunk_id": 2, "source_id": "b", "line": 1}]}]}]
        SourceRecovery(["DadosA", "DadosB"], [], [[a], [b]]).enrich(topics)
        sub = topics[0]["subtopics"][0]
        self.assertEqual([block["page"] for block in sub["source_blocks"]], [1, 2])
        self.assertEqual(sub["pages"], [1, 2])

    def test_page_continuation_table_remains_separate(self):
        tables=[[{'start':0,'end':6,'title':'Tabela original','cells':[],'row_count':0}],
                [{'start':0,'end':6,'title':'Tabela do PDF','cells':[],'row_count':0}]]
        topics=[{'subtopics':[{'evidence':[{'page':p,'quote':'Dados.','chunk_id':p,'source_id':str(p),'line':1} for p in (1,2)]}]}]
        SourceRecovery(['Dados.','Dados.'],[],tables).enrich(topics)
        blocks=topics[0]['subtopics'][0]['source_blocks']
        self.assertEqual([b['page'] for b in blocks],[1,2])
        self.assertFalse(any(b['continuation_pending'] for b in blocks))


if __name__=='__main__':unittest.main()
