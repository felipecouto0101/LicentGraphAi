import unittest
from app.rag.topic_organizer import organize_topic_map, source_catalog
from app.rag.topic_map import all_chunk_ids


class OrganizerTests(unittest.TestCase):
    def setUp(self):
        self.topics = [{"id": "original", "title": "Título formal", "subtopics": [
            {"id": f"s{i}", "title": f"Seção {i}", "pages": [i + 1], "chunk_ids": [i]}
            for i in range(8)]}]
        self.chunks = [{"content": f"Texto do documento {i}", "metadata": {"page": i + 1, "chunk_id": i}}
                       for i in range(8)]

    def reply(self, system, payload):
        if "sections" in payload:
            return {"themes": [{"title": "Regras relacionadas", "subtopics": [
                {"title": "Condições para inscrição", "source_ids": [entry["id"] for entry in payload["sections"]]}]}]}
        return {"themes": [{"title": "Inscrição no programa", "source_ids": [e["id"] for e in payload["themes"]]}]}

    def test_dynamic_themes_merge_sources_across_batches_without_explanations(self):
        requests, progress = [], []
        def invoke(system, payload):
            requests.append((system, payload))
            return self.reply(system, payload)
        result = organize_topic_map(self.topics, self.chunks, invoke, progress=lambda **p: progress.append(p))
        self.assertEqual(len(requests), 3)
        self.assertEqual(result[0]["title"], "Inscrição no programa")
        self.assertEqual(len(result[0]["subtopics"]), 1)
        sub = result[0]["subtopics"][0]
        self.assertEqual(sub["pages"], list(range(1, 9)))
        self.assertEqual(all_chunk_ids(result), set(range(8)))
        self.assertEqual(sub["source_ids"], [f"s{i}" for i in range(8)])
        self.assertNotIn("explanation", sub)
        self.assertEqual(self.topics[0]["title"], "Título formal")
        self.assertEqual(progress[-1]["completed"], progress[-1]["total"])

    def test_missing_or_unknown_source_is_rejected_before_cache(self):
        for ids in (["s0"], ["inventado"]):
            cache = {}
            with self.assertRaises(ValueError):
                organize_topic_map(self.topics, self.chunks, lambda *args: {"themes": [
                    {"title": "Participação", "subtopics": [{"title": "Regras", "source_ids": ids}]}]}, cache=cache)
            self.assertEqual(cache, {})

    def test_retry_reuses_verified_batches(self):
        cache, calls = {}, []
        def fail_second(system, payload):
            calls.append(payload)
            if len(calls) == 2:
                raise RuntimeError("429 sem cota")
            return self.reply(system, payload)
        with self.assertRaises(RuntimeError):
            organize_topic_map(self.topics, self.chunks, fail_second, cache=cache)
        self.assertEqual(len(cache), 1)
        resumed = []
        def resume(system, payload):
            resumed.append(payload)
            return self.reply(system, payload)
        organize_topic_map(self.topics, self.chunks, resume, cache=cache)
        self.assertEqual(len(resumed), 2)
        self.assertEqual(resumed[0]["sections"][0]["id"], "s6")

    def test_duplicate_consolidation_ids_are_rejected(self):
        def invoke(system, payload):
            if "themes" in payload:
                return {"themes": [{"title": "Inscrição", "source_ids": ["0", "0", "1"]}]}
            return self.reply(system, payload)
        with self.assertRaises(ValueError):
            organize_topic_map(self.topics, self.chunks, invoke)

    def test_truncation_splits_batches_and_keeps_verified_parts(self):
        class GroqOutputTruncated(RuntimeError): pass
        sizes = []
        def invoke(system, payload):
            if "sections" in payload:
                sizes.append(len(payload["sections"]))
                if len(payload["sections"]) > 3:
                    raise GroqOutputTruncated()
            return self.reply(system, payload)
        result = organize_topic_map(self.topics, self.chunks, invoke)
        self.assertEqual(sizes, [6, 3, 3, 2])
        self.assertEqual(all_chunk_ids(result), set(range(8)))

    def test_catalog_uses_bounded_samples_and_pages(self):
        catalog = source_catalog(self.topics, self.chunks)
        self.assertEqual(len(catalog), 8)
        self.assertEqual(catalog[3]["samples"][0]["page"], 4)
        self.assertLessEqual(len(catalog[3]["samples"][0]["text"]), 240)


if __name__ == "__main__":
    unittest.main()
