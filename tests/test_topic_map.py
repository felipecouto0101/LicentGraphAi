"""Regressões do mapa rápido e das respostas verificadas."""

import importlib
import threading
import types
import unittest
from unittest.mock import patch

from app.rag.topic_map import build_topic_map, all_chunk_ids, merge_similar_themes, split_topic_sections


class FakeEncoder:
    def encode(self, titles):
        vectors = []
        for title in titles:
            if "Habilita" in title or "Qualificação" in title:
                vectors.append([1.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 1.0, 0.0])
        return vectors


class TopicMapTests(unittest.TestCase):
    def test_covers_every_chunk_and_preserves_physical_pages(self):
        chunks = [
            {"content": "Introdução sem numeração", "metadata": {"page": 1, "chunk_id": 0}},
            {"content": "1. HABILITAÇÃO\nDeclaração inicial.", "metadata": {"page": 2, "chunk_id": 1}},
            {"content": "1.1 Qualificação técnica\nAtestado específico.", "metadata": {"page": 3, "chunk_id": 2}},
            {"content": "Texto corrido da qualificação.", "metadata": {"page": 4, "chunk_id": 3}},
            {"content": "2. PROPOSTA\nPreço global.", "metadata": {"page": 5, "chunk_id": 4}},
        ]
        topic_map = build_topic_map(chunks)
        self.assertEqual(all_chunk_ids(topic_map), {0, 1, 2, 3, 4})
        tecnico = next(sub for theme in topic_map for sub in theme["subtopics"]
                       if sub["title"] == "Qualificação técnica")
        self.assertEqual(tecnico["pages"], [3, 4])
        self.assertEqual(tecnico["chunk_ids"], [2, 3])

    def test_sections_before_chunks_preserve_wrapped_headings_and_all_text(self):
        pages = ["Capa\nSumário\n1. DO OBJETO ........ 2\n2. DA PROPOSTA E DOS\nDOCUMENTOS ........ 3",
                 "Cabeçalho\n1. DO OBJETO\n1.1. A contratada deve executar os serviços.\n2. DA PROPOSTA E DOS\nDOCUMENTOS\n2.1 PREÇOS E VALORES\n2.1.1. O preço será global.",
                 "Continuação da proposta.\n3 DOS RECURSOS\nTexto sobre recursos.\nANEXO II- MINUTA DE CONTRATO\nCLÁUSULA PRIMEIRA – DO OBJETO\n1.1. Obra pública."]
        blocks = split_topic_sections(pages)
        self.assertEqual("\n".join(b["content"] for b in blocks).splitlines(),
                         [line for page in pages for line in page.splitlines()])
        chunks = [{"content": b["content"], "metadata": {
            "page": b["page"], "chunk_id": i, "topic_title": b["topic_title"],
            "subtopic_title": b["subtopic_title"]}} for i, b in enumerate(blocks)]
        topics = build_topic_map(chunks)
        self.assertEqual(all_chunk_ids(topics), set(range(len(chunks))))
        self.assertEqual([t["title"] for t in topics], ["Introdução e dados iniciais",
            "Do Objeto", "Da Proposta E Dos Documentos", "Dos Recursos", "Anexo Ii- Minuta De Contrato"])
        self.assertEqual(topics[2]["subtopics"][1]["pages"], [2, 3])
        self.assertEqual(topics[2]["subtopics"][1]["title"], "Preços E Valores")
        self.assertEqual(topics[-1]["subtopics"][1]["title"], "Do Objeto")
        self.assertFalse(any("deve executar" in t["title"] for t in topics))

    def test_reader_preserves_lines_only_when_requested(self):
        from app.rag.pdf_reader import PDFReader
        reader = PDFReader()
        text = "1. DO OBJETO\nTexto da cláusula.\n2. DA PROPOSTA"
        self.assertIn("\n", reader._clean_text(text, preserve_lines=True))
        self.assertNotIn("\n", reader._clean_text(text))

    def test_semantic_grouping_keeps_distinct_subtopics_and_sources(self):
        topics = [
            {"id": "a", "title": "Habilitação", "subtopics": [
                {"id": "s1", "title": "Atestados", "pages": [2], "chunk_ids": [1]}]},
            {"id": "b", "title": "Qualificação", "subtopics": [
                {"id": "s2", "title": "Balanços", "pages": [9], "chunk_ids": [8]}]},
            {"id": "c", "title": "Proposta", "subtopics": [
                {"id": "s3", "title": "Preço", "pages": [11], "chunk_ids": [12]}]},
        ]
        grouped = merge_similar_themes(topics, FakeEncoder())
        self.assertEqual(len(grouped), 2)
        self.assertEqual(all_chunk_ids(grouped), {1, 8, 12})
        self.assertEqual({sub["title"] for sub in grouped[0]["subtopics"]},
                         {"Atestados", "Balanços"})


if __name__ == "__main__":
    unittest.main()
