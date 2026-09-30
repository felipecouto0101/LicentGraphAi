"""API do mapa sem Groq antecipada e citações verificadas no chat."""

import importlib.util
import json
import numpy  # Mantém o módulo carregado durante mocks de sys.modules.
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_api():
    fastapi = types.ModuleType("fastapi")
    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code, self.detail = status_code, detail
            super().__init__(detail)
    class FastAPI:
        def __init__(self, **kw): pass
        def add_middleware(self, *args, **kw): pass
        def get(self, *args, **kw): return lambda fn: fn
        def post(self, *args, **kw): return lambda fn: fn
    fastapi.FastAPI = FastAPI
    fastapi.UploadFile = type("UploadFile", (), {})
    fastapi.File = lambda *args, **kw: None
    fastapi.HTTPException = HTTPException
    fastapi.BackgroundTasks = type("BackgroundTasks", (), {})
    cors = types.ModuleType("fastapi.middleware.cors")
    cors.CORSMiddleware = type("CORSMiddleware", (), {})
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda: None
    workflow = types.ModuleType("app.rag.langgraph_workflow")
    workflow.is_mock_mode = lambda: False
    workflow.validate_analysis_configuration = lambda: None
    workflow.run_licit_graph_pipeline = lambda *a, **kw: None
    classes = {"fastapi": fastapi, "fastapi.middleware": types.ModuleType("fastapi.middleware"),
               "fastapi.middleware.cors": cors, "dotenv": dotenv,
               "app.rag.langgraph_workflow": workflow}
    with patch.dict(sys.modules, classes):
        spec = importlib.util.spec_from_file_location("topic_api_test", Path("app/api/main.py"))
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
    return api, classes


class APITests(unittest.TestCase):
    def setUp(self):
        self.api, self.modules = load_api()
        self.stack = patch.dict(sys.modules, self.modules)
        self.stack.start()
        self.addCleanup(self.stack.stop)
        self.topic = {"id": "tema-1", "title": "Habilitação", "subtopics": [{
            "id": "subtema-1", "title": "Documentos", "pages": [4], "chunk_ids": [1]}]}
        self.chunk = {"content": "O licitante deve apresentar atestado de capacidade técnica.",
                      "metadata": {"page": 4, "chunk_id": 1}}

    def test_preparation_uses_ai_organization_without_embeddings(self):
        node1 = types.ModuleType("app.rag.node_1_reader_chunker")
        node1.Node1ReaderChunker = lambda **kwargs: types.SimpleNamespace(
            process_pdf=lambda path: {"chunks": [self.chunk], "page_count": 4})
        node3 = types.ModuleType("app.rag.node_3_analyzer")
        class Analyzer:
            calls = []
            def __init__(self, **kw): pass
            def _invoke_with_rotation(self, messages):
                self.calls.append(json.loads(messages[1].content))
                if "passages" in self.calls[-1]:
                    entry = self.calls[-1]["passages"][0]
                    output = {"topics": [{"theme": "Documentação", "title": "Capacidade técnica",
                        "sources": [{"id": entry["id"], "lines": [1]}]}]}
                else:
                    output = {"themes": [{"title": "Documentos para participar", "subtopics": [
                        {"title": "Capacidade técnica", "source_ids": ["0"]}]}]}
                return types.SimpleNamespace(content=json.dumps(output))
            _parse_llm_json = staticmethod(json.loads)
        node3.Node3RequirementAnalyzer = Analyzer
        messages = types.ModuleType("langchain_core.messages")
        messages.SystemMessage = messages.HumanMessage = lambda content: types.SimpleNamespace(content=content)
        with patch.dict(sys.modules, {"app.rag.node_1_reader_chunker": node1,
                "app.rag.node_3_analyzer": node3, "langchain_core": types.ModuleType("langchain_core"),
                "langchain_core.messages": messages}):
            self.api._jobs["job"] = {"status": "queued", "progress": {}, "result": None}
            self.api._run_topic_job("job", "edital.pdf")
        result = self.api._jobs["job"]["result"]
        self.assertEqual(result["map_version"], 4)
        self.assertEqual(result["organization_status"], "done")
        self.assertEqual(result["topic_map"][0]["title"], "Documentos para participar")
        self.assertEqual(result["topic_map"][0]["subtopics"][0]["chunk_ids"], [1])
        self.assertEqual(len(Analyzer.calls), 2)
        self.assertEqual(self.api._jobs["job"]["progress"]["completed"], 1)
        self.assertFalse(hasattr(self.api, "ask_about_edital"))

    def test_gemini_upload_organizes_without_groq_and_publishes_activity(self):
        node1 = types.ModuleType("app.rag.node_1_reader_chunker")
        node1.Node1ReaderChunker = lambda **kw: types.SimpleNamespace(
            process_pdf=lambda path: {"chunks": [self.chunk], "page_count": 4})
        events = []
        class Gemini:
            def __init__(self, progress): self.progress = progress
            def invoke(self, system, payload):
                self.progress(activity="waiting", activity_message="Aguardando Gemini", wait_until=123)
                events.append(self_api._jobs["job"]["progress"].copy())
                if "passages" in payload:
                    return {"topics": [{"theme": "Documentos", "title": "Capacidade técnica",
                        "sources": [{"id": "p1", "lines": [1]}]}]}
                return {"themes": [{"title": "Documentos", "subtopics": [
                    {"title": "Capacidade técnica", "source_ids": ["0"]}]}]}
        self_api = self.api
        with patch.dict(sys.modules, {"app.rag.node_1_reader_chunker": node1}), patch.object(
                self.api, "provider_name", return_value="gemini"), patch.object(
                self.api, "validate_topic_configuration"), patch.object(
                self.api, "GeminiTopicClient", Gemini), patch.object(
                self.api, "validate_analysis_configuration", side_effect=AssertionError("Groq should not run")):
            self.api._jobs["job"] = {"status": "queued", "progress": {}, "result": None}
            self.api._run_topic_job("job", "edital.pdf")
        self.assertEqual(self.api._jobs["job"]["result"]["organization_status"], "done")
        self.assertEqual(events[0]["activity"], "waiting")
        self.assertEqual(events[0]["wait_until"], 123)

    def test_organization_failure_keeps_structural_index(self):
        node1 = types.ModuleType("app.rag.node_1_reader_chunker")
        node1.Node1ReaderChunker = lambda **kwargs: types.SimpleNamespace(
            process_pdf=lambda path: {"chunks": [self.chunk], "page_count": 4})
        with patch.dict(sys.modules, {"app.rag.node_1_reader_chunker": node1}), patch.object(
                self.api, "validate_analysis_configuration", side_effect=ValueError("Cota indisponível")):
            self.api._jobs["job"] = {"status": "queued", "progress": {}, "result": None}
            self.api._run_topic_job("job", "edital.pdf")
        result = self.api._jobs["job"]["result"]
        self.assertEqual(self.api._jobs["job"]["status"], "done")
        self.assertEqual(result["organization_status"], "error")
        self.assertIn("Cota indisponível", result["organization_error"])
        self.assertEqual(result["topic_map"][0]["subtopics"][0]["chunk_ids"], [1])

    def test_grouping_failure_displays_validated_subtopics_as_partial_map(self):
        node1 = types.ModuleType("app.rag.node_1_reader_chunker")
        node1.Node1ReaderChunker = lambda **kwargs: types.SimpleNamespace(
            process_pdf=lambda path: {"chunks": [self.chunk], "page_count": 4})
        node3 = types.ModuleType("app.rag.node_3_analyzer")
        class Analyzer:
            def __init__(self, **kw): pass
            def _invoke_with_rotation(self, messages):
                payload = json.loads(messages[1].content)
                if "passages" not in payload:
                    raise RuntimeError("Consolidação interrompida")
                return types.SimpleNamespace(content=json.dumps({"topics": [{
                    "theme": "Documentação", "title": "Capacidade técnica", "sources": [
                        {"id": payload["passages"][0]["id"], "lines": [1]}]}]}))
            _parse_llm_json = staticmethod(json.loads)
        node3.Node3RequirementAnalyzer = Analyzer
        messages = types.ModuleType("langchain_core.messages")
        messages.SystemMessage = messages.HumanMessage = lambda content: types.SimpleNamespace(content=content)
        with patch.dict(sys.modules, {"app.rag.node_1_reader_chunker": node1,
                "app.rag.node_3_analyzer": node3, "langchain_core": types.ModuleType("langchain_core"),
                "langchain_core.messages": messages}):
            self.api._jobs["job"] = {"status": "queued", "progress": {}, "result": None}
            self.api._run_topic_job("job", "edital.pdf")
        result = self.api._jobs["job"]["result"]
        self.assertEqual(result["organization_status"], "partial")
        self.assertEqual(result["topic_map"][0]["subtopics"][0]["title"], "Capacidade técnica")
        self.assertEqual(result["topic_map"][0]["subtopics"][0]["pages"], [4])
        self.assertIn("Consolidação interrompida", result["organization_error"])

    def test_retry_refuses_concurrent_organization(self):
        self.api._jobs["job"] = {"status": "running", "result": {}}
        self.api._documents["job"] = {}
        with self.assertRaises(self.api.HTTPException) as error:
            self.api.retry_map_organization("job")
        self.assertEqual(error.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main()
