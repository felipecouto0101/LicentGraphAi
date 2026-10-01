"""Local persistent vectors: SQLite records and exact NumPy cosine search.

No model loading or server endpoints are part of this storage layer. Collection
names and document IDs are always SQL parameters, never executable SQL.
"""

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from threading import RLock

import numpy as np


class CollectionNotFoundError(LookupError):
    pass


class LocalVectorClient:
    def __init__(self, path: str | None = None):
        self.path = str(Path(path).resolve()) if path else None
        self._lock = RLock()
        self._memory = None
        if self.path:
            Path(self.path).mkdir(parents=True, exist_ok=True)
            self._database = str(Path(self.path) / "vectors.sqlite3")
        else:
            self._database = ":memory:"
            self._memory = sqlite3.connect(":memory:", check_same_thread=False)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS collections (
                    name TEXT PRIMARY KEY, metadata TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS vectors (
                    collection TEXT NOT NULL REFERENCES collections(name),
                    id TEXT NOT NULL, document TEXT NOT NULL,
                    embedding TEXT NOT NULL, metadata TEXT NOT NULL,
                    PRIMARY KEY (collection, id)
                );
            """)

    @contextmanager
    def _connect(self):
        # Each disk operation closes its handle, including on Windows. The
        # memory connection stays alive and is serialized across worker threads.
        with self._lock:
            db = self._memory or sqlite3.connect(self._database, timeout=30)
            try:
                db.execute("PRAGMA foreign_keys = ON")
                with db:
                    yield db
            finally:
                if self._memory is None:
                    db.close()

    def get_collection(self, name: str):
        with self._connect() as db:
            row = db.execute("SELECT metadata FROM collections WHERE name = ?", (name,)).fetchone()
        if row is None:
            legacy = self.path and (Path(self.path) / "chroma.sqlite3").exists()
            hint = " Reenvie o PDF para reconstruir o índice antigo; os arquivos Chroma foram preservados." if legacy else ""
            raise CollectionNotFoundError(f"Coleção vetorial '{name}' não encontrada.{hint}")
        return LocalVectorCollection(self, name, json.loads(row[0]))

    def create_collection(self, name: str, metadata: dict):
        if not name:
            raise ValueError("Collection name must not be empty")
        with self._connect() as db:
            db.execute("INSERT INTO collections VALUES (?, ?)", (name, json.dumps(metadata)))
        return LocalVectorCollection(self, name, metadata)


class LocalVectorCollection:
    def __init__(self, client: LocalVectorClient, name: str, metadata: dict):
        self.client, self.name, self.metadata = client, name, metadata

    def count(self) -> int:
        with self.client._connect() as db:
            return db.execute("SELECT COUNT(*) FROM vectors WHERE collection = ?", (self.name,)).fetchone()[0]

    def _validate_vector(self, vector):
        array = np.asarray(vector, dtype=np.float64)
        dimension = self.metadata["dimension"]
        if array.ndim != 1 or len(array) != dimension or not np.isfinite(array).all():
            raise ValueError(f"Expected a finite vector with dimension {dimension}")
        norm = np.linalg.norm(array)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Vector norm must be finite and positive")
        return array

    def _write(self, ids, documents, embeddings, metadatas, *, upsert):
        if not len(ids) == len(documents) == len(embeddings) == len(metadatas):
            raise ValueError("Vector records have inconsistent lengths")
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate IDs in the same batch")
        records = [
            (self.name, record_id, document,
             json.dumps(self._validate_vector(vector).tolist(), allow_nan=False),
             json.dumps(metadata, ensure_ascii=False, allow_nan=False))
            for record_id, document, vector, metadata in zip(ids, documents, embeddings, metadatas)
        ]
        with self.client._connect() as db:
            if upsert:
                db.executemany("""
                    INSERT INTO vectors VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(collection, id) DO UPDATE SET
                        document=excluded.document, embedding=excluded.embedding,
                        metadata=excluded.metadata
                """, records)
            else:
                db.executemany("INSERT INTO vectors VALUES (?, ?, ?, ?, ?)", records)

    def add(self, *, ids, documents, embeddings, metadatas):
        self._write(ids, documents, embeddings, metadatas, upsert=False)

    def upsert(self, *, ids, documents, embeddings, metadatas):
        self._write(ids, documents, embeddings, metadatas, upsert=True)

    def query(self, *, query_embeddings, n_results=3):
        if not isinstance(n_results, int) or isinstance(n_results, bool) or n_results < 1:
            raise ValueError("n_results must be a positive integer")
        queries = [self._validate_vector(vector) for vector in query_embeddings]
        with self.client._connect() as db:
            rows = db.execute(
                "SELECT id, document, embedding, metadata FROM vectors WHERE collection = ? ORDER BY id",
                (self.name,),
            ).fetchall()
        result = {key: [] for key in ("ids", "documents", "metadatas", "distances")}
        if rows:
            matrix = np.asarray([json.loads(row[2]) for row in rows], dtype=np.float64)
            # Normalize separately to avoid overflow in products of norms.
            matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        for query in queries:
            if rows:
                scores = np.clip(matrix @ (query / np.linalg.norm(query)), -1, 1)
                indices = np.argsort(-scores, kind="stable")[:n_results]
            else:
                indices = []
            result["ids"].append([rows[i][0] for i in indices])
            result["documents"].append([rows[i][1] for i in indices])
            result["metadatas"].append([json.loads(rows[i][3]) for i in indices])
            result["distances"].append([float(1 - scores[i]) for i in indices])
        return result
