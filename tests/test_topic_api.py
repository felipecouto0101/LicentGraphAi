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

    def test_index_does_not_call_llm(self):
        node1 = types.ModuleType("app.rag.node_1_reader_chunker")
        node1.Node1ReaderChunker = lambda **kwargs: types.SimpleNamespace(
            process_pdf=lambda path: {"chunks": [self.chunk]})
        node2 = types.ModuleType("app.rag.node_2_embeddings")
        class Indexer:
            embeddings = types.SimpleNamespace(encode=lambda self, titles: [[1., 0.] for _ in titles])
            def process_chunks(self, chunks, **kwargs): return {"collection_id": "collection"}
        node2.Node2EmbeddingGenerator = Indexer
        with patch.dict(sys.modules, {"app.rag.node_1_reader_chunker": node1,
                                      "app.rag.node_2_embeddings": node2}):
            self.api._jobs["job"] = {"status": "queued", "progress": {}, "result": None}
            self.api._run_topic_job("job", "edital.pdf")
        self.assertEqual(self.api._jobs["job"]["status"], "done")
        self.assertEqual(self.api._jobs["job"]["result"]["chunks_count"], 1)
        self.assertEqual(self.api._jobs["job"]["result"]["map_version"], 2)
        self.assertEqual(self.api._documents["job"]["topics"][0]["subtopics"][0]["chunk_ids"], [1])
        self.assertIsNone(self.api._documents["job"]["llm"])

    def test_chat_rejects_wrong_page_and_returns_verified_quote(self):
        node3 = types.ModuleType("app.rag.node_3_analyzer")
        class Analyzer:
            output = None
            def __init__(self, **kw): pass
            def _invoke_with_rotation(self, messages):
                return types.SimpleNamespace(content=json.dumps(self.output))
            _parse_llm_json = staticmethod(json.loads)
        node3.Node3RequirementAnalyzer = Analyzer
        messages = types.ModuleType("langchain_core.messages")
        messages.SystemMessage = messages.HumanMessage = lambda content: types.SimpleNamespace(content=content)
        llc = types.ModuleType("langchain_core")
        self.api._documents["job"] = {"chunks": [self.chunk], "topics": [self.topic],
                                      "collection": "collection", "lock": threading.Lock(), "llm": None}
        q = self.api.TopicQuestion(topic_id="subtema-1", question="Qual atestado é necessário?")
        with patch.dict(sys.modules, {"app.rag.node_3_analyzer": node3,
                                      "langchain_core": llc, "langchain_core.messages": messages}):
            Analyzer.output = {"found": True, "answer": "Apresente atestado de capacidade técnica.",
                               "citations": [{"page": 99, "quote": "atestado de capacidade técnica"}]}
            with self.assertRaises(self.api.HTTPException) as error:
                self.api.ask_about_edital("job", q)
            self.assertEqual(error.exception.status_code, 502)
            Analyzer.output["citations"][0]["page"] = 4
            result = self.api.ask_about_edital("job", q)
            self.assertTrue(result["found"])
            self.assertEqual(result["citations"][0]["page"], 4)

    def test_ai_names_preserve_ids_and_original_titles(self):
        node3 = types.ModuleType("app.rag.node_3_analyzer")
        class Analyzer:
            output = None
            def __init__(self, **kw): pass
            def _invoke_with_rotation(self, messages):
                return types.SimpleNamespace(content=json.dumps(self.output))
            _parse_llm_json = staticmethod(json.loads)
        node3.Node3RequirementAnalyzer = Analyzer
        messages = types.ModuleType("langchain_core.messages")
        messages.SystemMessage = messages.HumanMessage = lambda content: types.SimpleNamespace(content=content)
        session = {"chunks": [self.chunk], "topics": [self.topic],
                   "collection": "collection", "lock": threading.Lock(), "llm": None}
        self.api._documents["job"] = session
        self.api._jobs["job"] = {"result": {"topic_map": session["topics"]}}
        with patch.dict(sys.modules, {"app.rag.node_3_analyzer": node3,
                                      "langchain_core": types.ModuleType("langchain_core"),
                                      "langchain_core.messages": messages}):
            Analyzer.output = {"theme_label": "Participação na licitação", "subtopics": []}
            with self.assertRaises(self.api.HTTPException) as error:
                self.api.improve_topic_names("job", self.api.TopicRename(theme_id="tema-1"))
            self.assertEqual(error.exception.status_code, 502)
            self.assertNotIn("display_title", self.topic)
            Analyzer.output["subtopics"] = [{"id": "subtema-1", "label": "O que apresentar"}]
            result = self.api.improve_topic_names("job", self.api.TopicRename(theme_id="tema-1"))
            self.assertEqual(result["topic_map"][0]["display_title"], "Participação na licitação")
            self.assertEqual(self.topic["title"], "Habilitação")
            self.assertEqual(self.topic["subtopics"][0]["id"], "subtema-1")
            self.assertEqual(self.topic["subtopics"][0]["pages"], [4])


if __name__ == "__main__":
    unittest.main()
