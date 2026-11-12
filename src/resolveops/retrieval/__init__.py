"""ResolveOps retrieval module: Dense, Sparse (BM25), Hybrid (RRF), and Cross-Encoder Reranker."""

from resolveops.retrieval.bm25 import BM25Index, BM25SearchResult, technical_tokenize
from resolveops.retrieval.embeddings import (
    EmbeddingModel,
    LocalEmbeddingModel,
    RemoteEmbeddingModel,
    make_embedding_model,
)
from resolveops.retrieval.hybrid import HybridRetriever, HybridSearchResult
from resolveops.retrieval.reranker import (
    MockReranker,
    NVIDIAReranker,
    RerankResult,
    Reranker,
    make_reranker,
)
from resolveops.retrieval.vector_store import (
    InMemoryVectorStore,
    SearchResult,
    VectorStore,
)

__all__ = [
    "EmbeddingModel",
    "LocalEmbeddingModel",
    "RemoteEmbeddingModel",
    "make_embedding_model",
    "VectorStore",
    "InMemoryVectorStore",
    "SearchResult",
    "BM25Index",
    "BM25SearchResult",
    "technical_tokenize",
    "HybridRetriever",
    "HybridSearchResult",
    "Reranker",
    "NVIDIAReranker",
    "MockReranker",
    "RerankResult",
    "make_reranker",
]

