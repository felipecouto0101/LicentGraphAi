import unittest
from app.rag.topic_organizer import organize_topic_map, source_catalog, inspect_annex_references
from app.rag.topic_map import all_chunk_ids


class OrganizerTests(unittest.TestCase):
    def setUp(self):
        self.topics = [{"id": "original", "title": "Título formal", "subtopics": [
            {"id": f"s{i}", "title": f"Seção {i}", "pages": [i + 1], "chunk_ids": [i]}
            for i in range(16)]}]
        self.chunks = [{"content": f"Texto completo do documento {i}", "metadata": {"page": i + 1, "chunk_id": i}}
                       for i in range(16)]

    def reply(self, system, payload):
        if "passages" in payload:
            return {"topics": [{"theme": "Inscrição", "title": "Condições para inscrição", "sources": [
                {"id": e["id"], "lines": [1]} for e in payload["passages"]]}]}
        return {"themes": [{"title": "Inscrição no programa", "subtopics": [
            {"title": "Condições para inscrição", "source_ids": [e["id"] for e in payload["topics"]]}]}]}

    def test_dynamic_themes_merge_sources_without_explanations(self):
        requests, progress = [], []
        def invoke(system, payload):
            requests.append(payload)
            return self.reply(system, payload)
        result = organize_topic_map(self.topics, self.chunks, invoke, progress=lambda **p: progress.append(p))
        self.assertEqual(len(requests), 3)
        self.assertEqual(result[0]["title"], "Inscrição no programa")
        sub = result[0]["subtopics"][0]
        self.assertEqual(sub["pages"], list(range(1, 17)))
        self.assertEqual(all_chunk_ids(result), set(range(16)))
        self.assertEqual(len(sub["evidence"]), 16)
        self.assertNotIn("explanation", sub)
        self.assertEqual(self.topics[0]["title"], "Título formal")
        self.assertEqual(progress[-1]["completed"], progress[-1]["total"])

    def test_missing_or_invalid_lines_rejected_without_saving_invalid_result(self):
        for source in ({"id": "inventado", "lines": [1]},
                       {"id": "c0-0", "lines": [999]},
                       {"id": "c0-0", "lines": [1]}):
            cache = {}
            with self.assertRaises(ValueError):
                organize_topic_map(self.topics, self.chunks, lambda *args: {"topics": [
                    {"theme": "Participação", "title": "Inscrição", "sources": [source]}]}, cache=cache)
            if source["id"] == "c0-0" and source["lines"] == [1]:
                self.assertTrue(cache)  # Parte válida pode ser preservada antes da omissão.
            else:
                self.assertEqual(cache, {})

    def test_retry_reuses_verified_batches_and_changed_text_invalidates_cache(self):
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
        self.assertEqual(resumed[0]["passages"][0]["lines"][0]["text"], "Texto completo do documento 8")
        self.chunks[0]["content"] += " Informação adicional."
        resumed.clear()
        organize_topic_map(self.topics, self.chunks, resume, cache=cache)
        self.assertEqual(resumed[0]["passages"][0]["lines"][0]["text"], "Texto completo do documento 0 Informação adicional.")

    def test_duplicate_grouping_ids_rejected(self):
        def invoke(system, payload):
            if "topics" in payload:
                return {"themes": [{"title": "Inscrição", "subtopics": [
                    {"title": "Condições", "source_ids": ["0", "0"]}]}]}
            return self.reply(system, payload)
        with self.assertRaises(ValueError):
            organize_topic_map(self.topics, self.chunks, invoke)

    def test_truncation_splits_batches_and_keeps_verified_parts(self):
        class GeminiOutputTruncated(RuntimeError): pass
        sizes = []
        def invoke(system, payload):
            if "passages" in payload:
                sizes.append(len(payload["passages"]))
                if len(payload["passages"]) > 4:
                    raise GeminiOutputTruncated()
            return self.reply(system, payload)
        result = organize_topic_map(self.topics, self.chunks, invoke)
        self.assertEqual(sizes, [8, 4, 4, 8, 4, 4])
        self.assertEqual(all_chunk_ids(result), set(range(16)))

    def test_catalog_transmits_entire_text_including_middle_and_long_chunks(self):
        text = "Início. " + "Texto intermediário. " * 300 + "Nome social no final."
        self.chunks[0]["content"] = text
        catalog = source_catalog(self.topics, self.chunks)
        self.assertEqual("".join(e["text"] for e in catalog if e["chunk_id"] == 0), text)
        self.assertTrue(any("Nome social" in e["text"] for e in catalog))
        self.assertTrue(all(len(e["text"]) <= 2400 for e in catalog))

    def test_subtopic_pages_do_not_inherit_entire_section(self):
        chunks = [{"content": "Pagamento da taxa por boleto bancário.", "metadata": {"page": 2, "chunk_id": 0}},
                  {"content": "A utilização de nome social poderá ser solicitada.", "metadata": {"page": 12, "chunk_id": 1}}]
        structural = [{"title": "Inscrições", "subtopics": [{"id": "s", "title": "Visão geral", "pages": [2, 12], "chunk_ids": [0, 1]}]}]
        def invoke(system, payload):
            if "passages" in payload:
                return {"topics": [{"theme": "Inscrições", "title": "Pagamento" if e["page"] == 2 else "Nome social",
                    "sources": [{"id": e["id"], "lines": [1]}]} for e in payload["passages"]]}
            return {"themes": [{"title": "Inscrições", "subtopics": [
                {"title": e["title"], "source_ids": [e["id"]]} for e in payload["topics"]]}]}
        result = organize_topic_map(structural, chunks, invoke)
        by_title = {s["title"]: s for s in result[0]["subtopics"]}
        self.assertEqual(by_title["Nome social"]["pages"], [12])
        self.assertEqual(by_title["Pagamento"]["pages"], [2])

    def test_synonyms_consolidate_and_preserve_separate_evidence(self):
        def invoke(system, payload):
            if "passages" in payload:
                return {"topics": [{"theme": "Inscrição", "title": "Pagamento da taxa" if i % 2 else "Taxa de inscrição",
                    "sources": [{"id": e["id"], "lines": [1]}]} for i,e in enumerate(payload["passages"])]}
            return {"themes": [{"title": "Inscrições", "subtopics": [
                {"title": "Taxa e pagamento", "source_ids": [e["id"] for e in payload["topics"]]}]}]}
        result = organize_topic_map(self.topics, self.chunks, invoke)
        self.assertEqual(len(result[0]["subtopics"]), 1)
        self.assertEqual(all_chunk_ids(result), set(range(16)))

    def test_very_short_passage_remains_covered(self):
        self.chunks[0]["content"] = "FGTS;"
        result = organize_topic_map(self.topics, self.chunks, self.reply)
        self.assertIn(0, all_chunk_ids(result))

    def test_final_theme_consolidation_keeps_all_evidence(self):
        def invoke(system, payload):
            if "passages" in payload:
                return {"topics": [{"theme": "Grupo " + e["lines"][0]["text"], "title": "Assunto " + e["lines"][0]["text"],
                    "sources": [{"id": e["id"], "lines": [1]}]} for e in payload["passages"]]}
            if all(e["theme"] == "Temas do documento" for e in payload["topics"]):
                return {"themes": [{"title": "Assuntos relacionados", "subtopics": [
                    {"title": "Grupo reunido", "source_ids": [e["id"] for e in payload["topics"]]}]}]}
            return {"themes": [{"title": "Grupo de origem " + payload["topics"][0]["id"], "subtopics": [
                {"title": e["title"], "source_ids": [e["id"]]} for e in payload["topics"]]}]}
        result = organize_topic_map(self.topics, self.chunks, invoke)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]["subtopics"]), 16)
        self.assertEqual(all_chunk_ids(result), set(range(16)))

    def test_invalid_line_can_be_corrected_without_repeating_valid_batches(self):
        calls = []
        def invoke(system, payload):
            calls.append(payload)
            result = self.reply(system, payload)
            if len(calls) == 1:
                result["topics"][0]["sources"][0]["lines"] = [999]
            return result
        result = organize_topic_map(self.topics, self.chunks, invoke)
        self.assertIn("validation_error", calls[1])
        self.assertEqual(all_chunk_ids(result), set(range(16)))
        for theme in result:
            for sub in theme["subtopics"]:
                for evidence in sub["evidence"]:
                    self.assertIn(evidence["quote"], self.chunks[evidence["chunk_id"]]["content"])

    def test_single_truncated_passage_splits_text_and_preserves_sources(self):
        class GeminiOutputTruncated(RuntimeError): pass
        self.chunks = [{"content": "Conteúdo integral do edital. " * 20,
                        "metadata": {"page": 1, "chunk_id": 0}}]
        def invoke(system, payload):
            if "passages" in payload and sum(len(line["text"]) for line in payload["passages"][0]["lines"]) > 320:
                raise GeminiOutputTruncated()
            return self.reply(system, payload)
        result = organize_topic_map(self.topics, self.chunks, invoke)
        self.assertEqual(all_chunk_ids(result), {0})
        sub = result[0]["subtopics"][0]
        self.assertEqual(len(sub["source_ids"]), 2)
        self.assertTrue(all(e["quote"] in self.chunks[0]["content"] for e in sub["evidence"]))

    def test_partial_subtopics_published_before_later_failure(self):
        partial, calls = [], []
        def invoke(system, payload):
            calls.append(payload)
            if len(calls) == 2:
                raise RuntimeError("Cota indisponível")
            self.assertEqual(payload["passages"][0]["id"], "p1")
            self.assertEqual(payload["passages"][0]["lines"][0]["number"], 1)
            return self.reply(system, payload)
        with self.assertRaises(RuntimeError):
            organize_topic_map(self.topics, self.chunks, invoke, on_partial=partial.append)
        self.assertTrue(partial)
        self.assertEqual(all_chunk_ids(partial[-1]), set(range(8)))
        self.assertEqual(partial[-1][0]["subtopics"][0]["title"], "Condições para inscrição")

    def test_numeric_strings_and_duplicate_lines_normalize_without_guessing(self):
        def invoke(system, payload):
            result = self.reply(system, payload)
            if "passages" in payload:
                for source in result["topics"][0]["sources"]:
                    source["lines"] = ["1", 1]
            return result
        result = organize_topic_map(self.topics, self.chunks, invoke)
        self.assertEqual(all_chunk_ids(result), set(range(16)))
        self.assertEqual(len(result[0]["subtopics"][0]["evidence"]), 16)

    def test_annex_mentions_are_not_treated_as_present_annexes(self):
        chunks = [{"content": "Ver Anexos I e II deste edital. Cronograma no Anexo III.", "metadata": {"page": 1}},
                  {"content": "ANEXO 2 – CONTEÚDO PROGRAMÁTICO\nMatérias da prova.", "metadata": {"page": 4}}]
        result = inspect_annex_references(chunks)
        self.assertEqual({r["label"]: r["status"] for r in result},
                         {"Anexo I": "not_located", "Anexo II": "located", "Anexo III": "not_located"})


if __name__ == "__main__":
    unittest.main()
