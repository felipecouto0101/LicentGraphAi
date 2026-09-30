"""Regressões do mapa rápido e das respostas verificadas."""

import importlib
import threading
import types
import unittest
from unittest.mock import patch

from app.rag.topic_map import build_topic_map, all_chunk_ids, merge_similar_themes


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
