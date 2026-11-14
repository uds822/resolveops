"""Demonstration of Stage 1 (Hybrid RRF) vs Stage 2 (Cross-Encoder Reranker)."""

from resolveops.retrieval.bm25 import BM25Index
from resolveops.retrieval.embeddings import make_embedding_model
from resolveops.retrieval.hybrid import HybridRetriever
from resolveops.retrieval.reranker import make_reranker
from resolveops.retrieval.vector_store import QdrantVectorStore


def main() -> None:
    query = "What is the maximum item size in DynamoDB?"
    print(f"\n=======================================================")
    print(f"QUERY: \"{query}\"")
    print(f"=======================================================\n")

    # 1. Initialize Hybrid Search
    print("[1/3] Loading Hybrid Retriever (Nemotron-1B + BM25)...")
    embedder = make_embedding_model("nvidia:nvidia/nemotron-3-embed-1b")
    vs = QdrantVectorStore(embedder, path="./data/qdrant_db")
    bm25 = BM25Index.load_snapshot("./data/bm25_index.json")
    retriever = HybridRetriever(vs, bm25)

    # 2. Retrieve Top-10 Candidates (Stage 1)
    print("[2/3] Retrieving Top-10 candidates via Hybrid RRF...")
    candidates = retriever.search(query, top_k=10)

    print("\n--- [STAGE 1: HYBRID RRF CANDIDATES (Top 3)] ---")
    for i, r in enumerate(candidates[:3], 1):
        filename = r.chunk.source.replace("\\", "/").split("/")[-1]
        print(f"Rank #{i} | RRF Score: {r.score:.5f} | Dense Rank: {r.dense_rank} | BM25 Rank: {r.bm25_rank}")
        print(f"  Source: {filename} (Page {r.chunk.page_numbers})")
        print(f"  Content: {r.chunk.content[:140]}...\n")

    # 3. Rerank Candidates with Cross-Encoder (Stage 2)
    print("[3/3] Reranking Top-10 candidates via nemotron-3.5-rerank-8b (Cross-Attention)...")
    reranker = make_reranker("nvidia:nvidia/nemotron-3.5-rerank-8b")
    reranked = reranker.rerank(query, [c.chunk for c in candidates], top_k=3)

    print("\n--- [STAGE 2: CROSS-ENCODER RERANKED (Top 3)] ---")
    for r in reranked:
        filename = r.chunk.source.replace("\\", "/").split("/")[-1]
        print(
            f"Reranked #{r.reranked_rank} (Was Stage 1 Rank #{r.initial_rank}) | "
            f"Logit: {r.logit:+.4f} | Prob: {r.score:.4f}"
        )
        print(f"  Source: {filename} (Page {r.chunk.page_numbers})")
        print(f"  Content: {r.chunk.content[:140]}...\n")

    vs.client.close()
    print("Done! Closed Qdrant cleanly.\n")


if __name__ == "__main__":
    main()

