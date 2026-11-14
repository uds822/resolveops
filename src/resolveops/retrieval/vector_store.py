from __future__ import annotations

import datetime
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False
    QdrantClient = None  # type: ignore
    qmodels = None  # type: ignore

from resolveops.ingestion.chunking import Chunk
from resolveops.retrieval.embeddings import EmbeddingModel

SENTINEL_POINT_ID = "00000000-0000-0000-0000-000000000000"


def chunk_id_to_uuid(chunk_id: str) -> str:
    """Map an arbitrary chunk ID string to a deterministic UUIDv5 string for Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"urn:resolveops:chunk:{chunk_id}"))


class SearchResult(BaseModel):
    """A retrieved chunk paired with its relevance score."""
    model_config = ConfigDict(extra="forbid")

    chunk: Chunk
    score: float


class VectorStore(ABC):
    """Abstract interface for all vector stores."""

    @abstractmethod
    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Embed and index chunks. Returns count of newly added chunks."""
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """Search for the most relevant chunks given a text query."""
        ...

    @abstractmethod
    def delete_chunks(self, chunk_ids: Sequence[str]) -> int:
        """Remove specific chunks by ID. Returns count of removed chunks."""
        ...

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Remove all chunks associated with a document ID. Returns count of removed chunks."""
        ...

    @abstractmethod
    def replace_document(self, document_id: str, chunks: Sequence[Chunk]) -> tuple[int, int]:
        """Atomically replace chunks for document_id. Returns (deleted_count, added_count)."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return total number of indexed chunks."""
        ...

    @abstractmethod
    def get_generation_id(self) -> str | None:
        """Return current generation ID of the vector store, or None if unstamped."""
        ...

    @abstractmethod
    def set_generation_id(self, generation_id: str) -> None:
        """Stamp the generation ID into the vector store."""
        ...


class InMemoryVectorStore(VectorStore):
    """
    Fast, lightweight in-memory vector store using NumPy cosine similarity.
    Ideal for local development, testing, and bounded pilots.
    """

    def __init__(self, embedding_model: EmbeddingModel, generation_id: str | None = None):
        self.embedding_model = embedding_model
        self.generation_id = generation_id
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, np.ndarray] = {}

    def get_generation_id(self) -> str | None:
        return self.generation_id

    def set_generation_id(self, generation_id: str) -> None:
        self.generation_id = generation_id

    def snapshot_document(self, document_id: str):
        return [(chunk, self._vectors[cid].copy()) for cid, chunk in self._chunks.items()
                if chunk.document_id == document_id]

    def restore_document(self, document_id: str, snapshot) -> None:
        self.delete_document(document_id)
        for chunk, vector in snapshot:
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.id] = vector

    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        """
        Embed and store chunks with strict validation and batch-level deduplication.
        
        Enforces:
        1. Skips chunks where indexable is False (quarantined bad tables).
        2. Deduplicates within the incoming batch (idempotent across batch duplicates).
        3. Skips chunks already present in the store (idempotent across store state).
        4. Validates embedding return count, dimensionality, and finiteness.
        """
        seen_in_batch: set[str] = set()
        new_chunks: list[Chunk] = []

        for chunk in chunks:
            if not chunk.indexable:
                continue
            if chunk.id in self._chunks or chunk.id in seen_in_batch:
                continue
            seen_in_batch.add(chunk.id)
            new_chunks.append(chunk)

        if not new_chunks:
            return 0

        # Batch embed the texts for efficiency
        texts = [chunk.content for chunk in new_chunks]
        embeddings = self.embedding_model.embed_documents(texts)

        # Strict validation: embedding count must strictly match chunk count
        if len(embeddings) != len(new_chunks):
            raise ValueError(
                f"Embedding count mismatch: model returned {len(embeddings)} vectors for {len(new_chunks)} chunks"
            )

        processed_vectors: list[np.ndarray] = []
        for idx, (chunk, vector) in enumerate(zip(new_chunks, embeddings)):
            arr = np.asarray(vector, dtype=np.float32)
            if arr.ndim != 1 or arr.size == 0:
                raise ValueError(
                    f"Vector for chunk '{chunk.id}' at index {idx} has invalid shape {arr.shape}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"Vector for chunk '{chunk.id}' at index {idx} contains non-finite values (NaN or Inf)"
                )
            processed_vectors.append(arr)

        # Dimensionality check across the batch
        expected_dim = processed_vectors[0].shape[0]
        for idx, vec in enumerate(processed_vectors):
            if vec.shape[0] != expected_dim:
                raise ValueError(
                    f"Vector dimension mismatch at index {idx}: expected {expected_dim}, got {vec.shape[0]}"
                )

        # Check against existing collection dimension
        if self._vectors:
            existing_dim = next(iter(self._vectors.values())).shape[0]
            if expected_dim != existing_dim:
                raise ValueError(
                    f"Vector dimension mismatch with collection: collection expects dimension {existing_dim}, "
                    f"but incoming batch has dimension {expected_dim}"
                )

        # Atomically mutate state after all validations pass
        for chunk, vector in zip(new_chunks, processed_vectors):
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.id] = vector

        return len(new_chunks)

    def delete_chunks(self, chunk_ids: Sequence[str]) -> int:
        """Remove specific chunks by ID. Returns count of removed chunks."""
        removed = 0
        for cid in chunk_ids:
            if cid in self._chunks:
                del self._chunks[cid]
                self._vectors.pop(cid, None)
                removed += 1
        return removed

    def delete_document(self, document_id: str) -> int:
        """Remove all chunks belonging to a document ID. Returns count of removed chunks."""
        matching_ids = [
            cid for cid, chunk in self._chunks.items()
            if chunk.document_id == document_id
        ]
        return self.delete_chunks(matching_ids)

    def replace_document(self, document_id: str, chunks: Sequence[Chunk]) -> tuple[int, int]:
        """Atomically replace chunks for document_id. Validates new chunks before removing old ones."""
        seen_in_batch: set[str] = set()
        new_chunks: list[Chunk] = []

        for chunk in chunks:
            if not chunk.indexable:
                continue
            if chunk.id in seen_in_batch:
                continue
            seen_in_batch.add(chunk.id)
            new_chunks.append(chunk)

        if not new_chunks:
            deleted = self.delete_document(document_id)
            return deleted, 0

        # Batch embed and validate new chunks first without modifying state
        texts = [chunk.content for chunk in new_chunks]
        embeddings = self.embedding_model.embed_documents(texts)

        if len(embeddings) != len(new_chunks):
            raise ValueError(
                f"Embedding count mismatch: model returned {len(embeddings)} vectors for {len(new_chunks)} chunks"
            )

        processed_vectors: list[np.ndarray] = []
        for idx, (chunk, vector) in enumerate(zip(new_chunks, embeddings)):
            arr = np.asarray(vector, dtype=np.float32)
            if arr.ndim != 1 or arr.size == 0:
                raise ValueError(
                    f"Vector for chunk '{chunk.id}' at index {idx} has invalid shape {arr.shape}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"Vector for chunk '{chunk.id}' at index {idx} contains non-finite values (NaN or Inf)"
                )
            processed_vectors.append(arr)

        expected_dim = processed_vectors[0].shape[0]
        for idx, vec in enumerate(processed_vectors):
            if vec.shape[0] != expected_dim:
                raise ValueError(
                    f"Vector dimension mismatch at index {idx}: expected {expected_dim}, got {vec.shape[0]}"
                )

        # Check dimension against surviving collection vectors
        surviving_vectors = {
            cid: vec for cid, vec in self._vectors.items()
            if self._chunks[cid].document_id != document_id
        }
        if surviving_vectors:
            existing_dim = next(iter(surviving_vectors.values())).shape[0]
            if expected_dim != existing_dim:
                raise ValueError(
                    f"Vector dimension mismatch with collection: collection expects dimension {existing_dim}, "
                    f"but incoming batch has dimension {expected_dim}"
                )

        # New chunks are 100% validated; now perform the atomic swap
        deleted = self.delete_document(document_id)
        for chunk, vector in zip(new_chunks, processed_vectors):
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.id] = vector

        return deleted, len(new_chunks)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """
        Search for top_k chunks closest to query using cosine similarity.
        """
        if not self._chunks:
            return []

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        # 1. Embed query
        query_vec = np.array(
            self.embedding_model.embed_query(query),
            dtype=np.float32,
        )
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        # 2. Build doc matrix and calculate cosine similarity
        chunk_ids = list(self._chunks.keys())
        doc_matrix = np.stack([self._vectors[cid] for cid in chunk_ids])
        doc_norms = np.linalg.norm(doc_matrix, axis=1)

        # Avoid division by zero
        valid = doc_norms > 0
        scores = np.zeros(len(chunk_ids), dtype=np.float32)
        scores[valid] = np.dot(doc_matrix[valid], query_vec) / (doc_norms[valid] * query_norm)

        # 3. Sort by highest score first
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            SearchResult(
                chunk=self._chunks[chunk_ids[idx]],
                score=round(float(scores[idx]), 4),
            )
            for idx in top_indices
        ]

    def count(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)


class QdrantVectorStore(VectorStore):
    """
    Persistent, production-grade vector store backed by Qdrant.
    Supports:
    - Embedded local disk persistence (path="./data/qdrant_db")
    - In-memory ephemeral mode (client=QdrantClient(":memory:"))
    - Remote Qdrant server / cloud (url="http://localhost:6333", api_key=...)
    - Cross-store generation ID tracking via a reserved sentinel point
    - Document deletion and atomic document replacement
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        collection_name: str = "resolveops_chunks",
        client: QdrantClient | None = None,
        path: str | Path | None = None,
        url: str | None = None,
        api_key: str | None = None,
        vector_dimension: int | None = None,
        generation_id: str | None = None,
    ):
        if not HAS_QDRANT:
            raise ImportError(
                "qdrant-client is required to use QdrantVectorStore. "
                "Install it with `pip install qdrant-client`."
            )

        self.embedding_model = embedding_model
        self.collection_name = collection_name

        if client is not None:
            self.client = client
        elif url is not None:
            self.client = QdrantClient(url=url, api_key=api_key)
        elif path is not None:
            db_path = Path(path).resolve()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(db_path))
        else:
            default_path = Path("./data/qdrant_db").resolve()
            default_path.parent.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(default_path))

        # Determine vector dimensionality
        if vector_dimension is not None:
            self.vector_dimension = vector_dimension
        else:
            probe = self.embedding_model.embed_query("dimension_probe")
            self.vector_dimension = len(probe)

        # Initialize collection if it doesn't exist
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_dimension,
                    distance=qmodels.Distance.COSINE,
                ),
            )
        else:
            info = self.client.get_collection(self.collection_name)
            col_size = info.config.params.vectors.size
            if col_size != self.vector_dimension:
                raise ValueError(
                    f"Qdrant collection '{self.collection_name}' expects vector dimension {col_size}, "
                    f"but embedding model provides {self.vector_dimension}"
                )

        if generation_id is not None:
            self.set_generation_id(generation_id)

    def get_generation_id(self) -> str | None:
        """Retrieve collection generation ID from the sentinel point."""
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[SENTINEL_POINT_ID],
            )
            if points and points[0].payload:
                return points[0].payload.get("generation_id")
        except Exception:
            pass
        return None

    def snapshot_document(self, document_id: str):
        condition = qmodels.Filter(must=[qmodels.FieldCondition(
            key="document_id", match=qmodels.MatchValue(value=document_id))])
        points = []
        offset = None
        while True:
            batch, offset = self.client.scroll(collection_name=self.collection_name,
                scroll_filter=condition, limit=256, offset=offset,
                with_payload=True, with_vectors=True)
            points.extend(batch)
            if offset is None:
                return points

    def restore_document(self, document_id: str, snapshot) -> None:
        self.delete_document(document_id)
        if snapshot:
            self.client.upsert(collection_name=self.collection_name, points=[
                qmodels.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in snapshot
            ], wait=True)

    def set_generation_id(self, generation_id: str) -> None:
        """Persist collection generation ID into the sentinel point."""
        sentinel_point = qmodels.PointStruct(
            id=SENTINEL_POINT_ID,
            vector=[0.0] * self.vector_dimension,
            payload={
                "_is_system": True,
                "generation_id": generation_id,
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "schema_version": 1,
            },
        )
        self.client.upsert(
            collection_name=self.collection_name,
            points=[sentinel_point],
        )

    def count(self) -> int:
        """Return total count of active user chunks, excluding system sentinel points."""
        res = self.client.count(
            collection_name=self.collection_name,
            count_filter=qmodels.Filter(
                must_not=[
                    qmodels.FieldCondition(
                        key="_is_system",
                        match=qmodels.MatchValue(value=True),
                    )
                ]
            ),
            exact=True,
        )
        return res.count

    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        """
        Embed and store chunks in Qdrant with deduplication, validation, and batching.
        """
        seen_in_batch: set[str] = set()
        new_chunks: list[Chunk] = []

        for chunk in chunks:
            if not chunk.indexable:
                continue
            if chunk.id in seen_in_batch:
                continue
            seen_in_batch.add(chunk.id)
            new_chunks.append(chunk)

        if not new_chunks:
            return 0

        # Check existing IDs in Qdrant to ensure idempotency across store state
        chunk_uuids = [chunk_id_to_uuid(c.id) for c in new_chunks]
        existing_pts = self.client.retrieve(
            collection_name=self.collection_name,
            ids=chunk_uuids,
        )
        existing_uuids = {str(p.id) for p in existing_pts}

        chunks_to_add: list[Chunk] = []
        for c, u in zip(new_chunks, chunk_uuids):
            if u not in existing_uuids:
                chunks_to_add.append(c)

        if not chunks_to_add:
            return 0

        # Embed texts
        texts = [c.content for c in chunks_to_add]
        embeddings = self.embedding_model.embed_documents(texts)

        if len(embeddings) != len(chunks_to_add):
            raise ValueError(
                f"Embedding count mismatch: model returned {len(embeddings)} vectors for {len(chunks_to_add)} chunks"
            )

        processed_vectors: list[list[float]] = []
        for idx, (chunk, vector) in enumerate(zip(chunks_to_add, embeddings)):
            arr = np.asarray(vector, dtype=np.float32)
            if arr.ndim != 1 or arr.size == 0:
                raise ValueError(
                    f"Vector for chunk '{chunk.id}' at index {idx} has invalid shape {arr.shape}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"Vector for chunk '{chunk.id}' at index {idx} contains non-finite values (NaN or Inf)"
                )
            if arr.shape[0] != self.vector_dimension:
                raise ValueError(
                    f"Vector dimension mismatch at index {idx}: expected {self.vector_dimension}, got {arr.shape[0]}"
                )
            processed_vectors.append(arr.tolist())

        points = [
            qmodels.PointStruct(
                id=chunk_id_to_uuid(chunk.id),
                vector=vec,
                payload={
                    "_is_system": False,
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "chunk": chunk.model_dump(mode="json"),
                },
            )
            for chunk, vec in zip(chunks_to_add, processed_vectors)
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return len(chunks_to_add)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """Search for top_k chunks closest to query using cosine similarity."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not query or not query.strip():
            return []

        if self.count() == 0:
            return []

        query_vec = self.embedding_model.embed_query(query)
        filter_non_system = qmodels.Filter(
            must_not=[
                qmodels.FieldCondition(
                    key="_is_system",
                    match=qmodels.MatchValue(value=True),
                )
            ]
        )
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            query_filter=filter_non_system,
            limit=top_k,
        )

        results: list[SearchResult] = []
        for pt in response.points:
            chunk_data = pt.payload.get("chunk")
            if chunk_data:
                chunk = Chunk.model_validate(chunk_data)
                results.append(
                    SearchResult(
                        chunk=chunk,
                        score=round(float(pt.score), 4),
                    )
                )
        return results

    def delete_chunks(self, chunk_ids: Sequence[str]) -> int:
        """Remove specific chunks by ID."""
        if not chunk_ids:
            return 0
        point_uuids = [chunk_id_to_uuid(cid) for cid in chunk_ids]
        existing = self.client.retrieve(
            collection_name=self.collection_name,
            ids=point_uuids,
        )
        count = len(existing)
        if count > 0:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.PointIdsList(points=point_uuids),
            )
        return count

    def delete_document(self, document_id: str) -> int:
        """Remove all chunks belonging to document_id."""
        filter_cond = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="document_id",
                    match=qmodels.MatchValue(value=document_id),
                )
            ]
        )
        matching = self.client.count(
            collection_name=self.collection_name,
            count_filter=filter_cond,
            exact=True,
        )
        if matching.count > 0:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.FilterSelector(filter=filter_cond),
            )
        return matching.count

    def replace_document(self, document_id: str, chunks: Sequence[Chunk]) -> tuple[int, int]:
        """Atomically replace chunks for document_id in Qdrant after embedding validation."""
        seen_in_batch: set[str] = set()
        new_chunks: list[Chunk] = []

        for chunk in chunks:
            if not chunk.indexable:
                continue
            if chunk.id in seen_in_batch:
                continue
            seen_in_batch.add(chunk.id)
            new_chunks.append(chunk)

        if not new_chunks:
            deleted = self.delete_document(document_id)
            return deleted, 0

        # Validate embeddings first before mutating store
        texts = [chunk.content for chunk in new_chunks]
        embeddings = self.embedding_model.embed_documents(texts)

        if len(embeddings) != len(new_chunks):
            raise ValueError(
                f"Embedding count mismatch: model returned {len(embeddings)} vectors for {len(new_chunks)} chunks"
            )

        processed_vectors: list[list[float]] = []
        for idx, (chunk, vector) in enumerate(zip(new_chunks, embeddings)):
            arr = np.asarray(vector, dtype=np.float32)
            if arr.ndim != 1 or arr.size == 0:
                raise ValueError(
                    f"Vector for chunk '{chunk.id}' at index {idx} has invalid shape {arr.shape}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"Vector for chunk '{chunk.id}' at index {idx} contains non-finite values (NaN or Inf)"
                )
            if arr.shape[0] != self.vector_dimension:
                raise ValueError(
                    f"Vector dimension mismatch at index {idx}: expected {self.vector_dimension}, got {arr.shape[0]}"
                )
            processed_vectors.append(arr.tolist())

        # Once validated, execute atomic swap
        deleted = self.delete_document(document_id)

        points = [
            qmodels.PointStruct(
                id=chunk_id_to_uuid(chunk.id),
                vector=vec,
                payload={
                    "_is_system": False,
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "chunk": chunk.model_dump(mode="json"),
                },
            )
            for chunk, vec in zip(new_chunks, processed_vectors)
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return deleted, len(new_chunks)
