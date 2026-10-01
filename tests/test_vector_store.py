import importlib
import sqlite3
import sys
import types

import numpy as np
import pytest

from app.rag.vector_store import CollectionNotFoundError, LocalVectorClient


def collection(client, name="edital"):
    return client.create_collection(name, {"dimension": 2, "model": "test"})


def insert(target, ids=("a", "b"), vectors=((1, 0), (0, 1)), upsert=False):
    method = target.upsert if upsert else target.add
    method(ids=list(ids), documents=[f"texto {i}" for i in ids],
           embeddings=list(vectors), metadatas=[{"page": i + 1} for i in range(len(ids))])


def test_persistence_ranking_and_sources(tmp_path):
    insert(collection(LocalVectorClient(str(tmp_path))))
    reopened = LocalVectorClient(str(tmp_path)).get_collection("edital")
    result = reopened.query(query_embeddings=[[0, 3], [4, 0]], n_results=8)
    assert result["ids"] == [["b", "a"], ["a", "b"]]
    assert result["documents"][0] == ["texto b", "texto a"]
    assert result["metadatas"][0] == [{"page": 2}, {"page": 1}]
    assert result["distances"][0] == pytest.approx([0, 1])


def test_isolation_and_parameterized_names():
    client = LocalVectorClient()
    insert(collection(client, "edital'; DROP TABLE vectors;--"))
    other = collection(client, "outro")
    assert other.count() == 0
    assert other.query(query_embeddings=[[1, 0]])["documents"] == [[]]
    assert client.get_collection("edital'; DROP TABLE vectors;--").count() == 2


def test_upsert_and_atomic_duplicate_rejection():
    target = collection(LocalVectorClient())
    insert(target, ids=("a",), vectors=((1, 0),))
    with pytest.raises(sqlite3.IntegrityError):
        insert(target, ids=("new", "a"))
    assert target.count() == 1  # The earlier record in the failed transaction rolled back.
    insert(target, ids=("a",), vectors=((0, 1),), upsert=True)
    assert target.count() == 1
    assert target.query(query_embeddings=[[0, 1]])["distances"] == [[0.0]]


@pytest.mark.parametrize("vector", [[1], [1, 2, 3], [0, 0], [float("nan"), 1], [float("inf"), 1]])
def test_invalid_vectors_do_not_change_collection(vector):
    target = collection(LocalVectorClient())
    with pytest.raises(ValueError):
        insert(target, ids=("a",), vectors=(vector,))
    assert target.count() == 0


def test_invalid_query_and_record_lengths():
    target = collection(LocalVectorClient())
    with pytest.raises(ValueError):
        target.add(ids=["a"], documents=[], embeddings=[], metadatas=[])
    with pytest.raises(ValueError):
        target.query(query_embeddings=[[1, 0]], n_results=0)
    with pytest.raises(ValueError):
        target.query(query_embeddings=[[1, 0, 0]])


def test_legacy_index_is_preserved_and_rebuild_is_explicit(tmp_path):
    legacy = tmp_path / "chroma.sqlite3"
    legacy.write_bytes(b"legacy index")
    client = LocalVectorClient(str(tmp_path))
    with pytest.raises(CollectionNotFoundError, match="Reenvie o PDF"):
        client.get_collection("old")
    assert legacy.read_bytes() == b"legacy index"
    insert(collection(client))
    assert client.get_collection("edital").count() == 2


@pytest.fixture
def node_module(monkeypatch):
    class Encoder:
        def __init__(self, *args, **kwargs):
            pass

        def get_embedding_dimension(self):
            return 2

        def encode(self, texts, **kwargs):
            return np.asarray([[1, 0] if "prazo" in text else [0, 1] for text in texts])

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=Encoder))
    module = importlib.import_module("app.rag.node_2_embeddings")
    monkeypatch.setattr(module, "SentenceTransformer", Encoder)
    monkeypatch.setattr(module.Node2EmbeddingGenerator, "_model_cache", {})
    yield module
    # Do not let the offline encoder escape into real integration tests.
    sys.modules.pop("app.rag.node_2_embeddings", None)


def test_node2_store_search_and_restart_contract(node_module, tmp_path):
    node = node_module.Node2EmbeddingGenerator(model_name="test")
    chunks = [{"content": "prazo de entrega", "metadata": {"page": 7, "chunk_id": 0}},
              {"content": "documentos", "metadata": {"page": 2, "chunk_id": 1}}]
    result = node.process_chunks(chunks, persist_directory=str(tmp_path))
    assert result["total_embeddings"] == 2
    fresh = node_module.Node2EmbeddingGenerator(model_name="test")
    found = fresh.search_similar("prazo", result["collection_id"], persist_directory=str(tmp_path))
    assert found["documents"][0][0] == "prazo de entrega"
    assert found["metadatas"][0][0]["page"] == 7
    # Legacy API delegates to the new backend; IDs remain stable across clients.
    fresh.store_in_chromadb(result["chunks_with_embeddings"], persist_directory=str(tmp_path), upsert=True)
    assert fresh.chroma_client.get_collection(result["collection_id"]).count() == 2


def test_node2_rejects_model_mismatch(node_module, tmp_path):
    node = node_module.Node2EmbeddingGenerator(model_name="original")
    node.process_chunks([{"content": "prazo"}], persist_directory=str(tmp_path))
    other = node_module.Node2EmbeddingGenerator(model_name="different")
    with pytest.raises(ValueError, match="outro modelo"):
        other.search_similar("prazo", persist_directory=str(tmp_path))
    with pytest.raises(ValueError, match="outro modelo"):
        other.store_vectors([{"content": "prazo", "embedding": [1, 0]}], persist_directory=str(tmp_path))
