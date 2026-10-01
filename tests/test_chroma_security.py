import ast
from pathlib import Path
import types

import chromadb
import numpy as np
import pytest

from app.rag.chroma_security import create_local_chroma_client, get_local_collection


def test_local_backend_overrides_unsafe_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("CHROMA_API_IMPL", "chromadb.api.fastapi.FastAPI")
    monkeypatch.setenv("CHROMA_SERVER_HOST", "untrusted.example")
    monkeypatch.setenv("CHROMA_SERVER_AUTHZ_PROVIDER", "chromadb.auth.simple_rbac_authz.SimpleRBACAuthorizationProvider")
    client = create_local_chroma_client(str(tmp_path))
    settings = client.get_settings()
    assert settings.chroma_api_impl == "chromadb.api.rust.RustBindingsAPI"
    assert settings.chroma_server_host is None
    assert settings.chroma_server_authz_provider is None
    assert not settings.anonymized_telemetry
    assert type(client._server).__name__ == "RustBindingsAPI"


def test_real_chroma_persistence_search_and_no_embedding_loader(tmp_path, monkeypatch):
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    monkeypatch.setattr(DefaultEmbeddingFunction, "__call__", lambda *a, **kw: pytest.fail("Unexpected Chroma embedding loader"))
    client = create_local_chroma_client(str(tmp_path))
    col = client.create_collection("edital", embedding_function=None, metadata={"hnsw:space": "cosine"})
    col.add(ids=["prazo", "documento"], documents=["Prazo", "Documento"],
            embeddings=[[1., 0.], [0., 1.]], metadatas=[{"page": 7}, {"page": 2}])
    reopened = get_local_collection(create_local_chroma_client(str(tmp_path)), "edital")
    assert reopened._collection._embedding_function is None
    found = reopened.query(query_embeddings=[[1., 0.]], n_results=1)
    assert found["ids"] == [["prazo"]]
    assert found["metadatas"] == [[{"page": 7}]]
    with pytest.raises(ValueError):
        reopened.query(query_texts=["Do not load a model"])


def test_old_default_collection_remains_readable(tmp_path):
    client = create_local_chroma_client(str(tmp_path))
    client.create_collection("legacy")  # Old app defaults; never invokes the EF.
    assert get_local_collection(client, "legacy")._collection._embedding_function is None


def test_persisted_remote_model_configuration_is_rejected():
    class Client:
        def get_collection(self, **kwargs):
            assert kwargs["embedding_function"] is None
            return types.SimpleNamespace(_model=types.SimpleNamespace(configuration_json={
                "embedding_function": {"type": "known", "name": "sentence_transformer",
                                       "config": {"model_name": "attacker/model", "trust_remote_code": True}}}))
    with pytest.raises(ValueError, match="unsupported persisted"):
        get_local_collection(Client(), "unsafe")


def test_unreviewed_version_is_rejected(monkeypatch):
    monkeypatch.setattr(chromadb, "__version__", "1.5.10")
    with pytest.raises(RuntimeError, match="version changed"):
        create_local_chroma_client()


def test_backend_mismatch_is_rejected(monkeypatch):
    monkeypatch.setattr(chromadb, "EphemeralClient", lambda **kw: types.SimpleNamespace(_server=object()))
    with pytest.raises(RuntimeError, match="local Rust"):
        create_local_chroma_client()


def test_application_has_no_other_chroma_constructor_or_server_import():
    root = Path(__file__).parents[1] / "app"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                assert not any(m.startswith(("chromadb.server", "chromadb.auth")) for m in modules), path
            if path.name != "chroma_security.py" and isinstance(node, ast.Call):
                func = node.func
                constructors = {"Client", "PersistentClient", "EphemeralClient", "HttpClient", "AsyncHttpClient", "CloudClient"}
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "chromadb":
                    assert func.attr not in constructors, path
                if isinstance(func, ast.Name):
                    imported = {a.asname or a.name for n in ast.walk(tree)
                                if isinstance(n, ast.ImportFrom) and n.module == "chromadb"
                                for a in n.names if a.name in constructors}
                    assert func.id not in imported, path


def test_node2_uses_real_chroma_without_model_download(monkeypatch, tmp_path):
    import importlib
    import sys
    class Encoder:
        def __init__(self, *a, **kw):
            pass
        def get_embedding_dimension(self):
            return 2
        def encode(self, texts, **kw):
            return np.asarray([[1., 0.] if "prazo" in text else [0., 1.] for text in texts])
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=Encoder))
    module = importlib.import_module("app.rag.node_2_embeddings")
    monkeypatch.setattr(module, "SentenceTransformer", Encoder)
    monkeypatch.setattr(module.Node2EmbeddingGenerator, "_model_cache", {})
    node = module.Node2EmbeddingGenerator(model_name="test")
    try:
        result = node.process_chunks([{"content": "prazo", "metadata": {"page": 5}}], persist_directory=str(tmp_path))
        found = node.search_similar("prazo", collection_name=result["collection_id"], persist_directory=str(tmp_path))
        assert found["documents"] == [["prazo"]]
        assert found["metadatas"][0][0]["page"] == 5
        assert get_local_collection(node.chroma_client, result["collection_id"])._collection._embedding_function is None
    finally:
        sys.modules.pop("app.rag.node_2_embeddings", None)
