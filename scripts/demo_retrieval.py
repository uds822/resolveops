import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resolveops.ingestion.chunking import Chunk
from resolveops.ingestion.models import SourceType
from resolveops.retrieval.embeddings import make_embedding_model
from resolveops.retrieval.vector_store import InMemoryVectorStore

print("\n" + "=" * 65)
print("🚀 STEP 1: INITIALIZING LOCAL EMBEDDING MODEL (MiniLM)...")
print("=" * 65)

embedding_model = make_embedding_model("all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(embedding_model)
print("✅ Embedding model & Vector Store initialized successfully!\n")

# Create 3 realistic sample chunks
sample_chunks = [
    Chunk(
        id="chunk_limits_01",
        document_id="doc_dynamo_dg",
        source="dynamodb-dg.pdf",
        source_type=SourceType.PDF,
        document_title="Amazon DynamoDB Developer Guide",
        kind="text",
        content="Amazon DynamoDB > Quotas > Item Sizes\n\nThe maximum item size in DynamoDB is 400 KB, which includes both attribute names and attribute values.",
        block_ids=["b_01"],
        context_block_ids=["h_limits"],
        heading_context=["Amazon DynamoDB", "Quotas", "Item Sizes"],
        page_numbers=[58],
        indexable=True,
    ),
    Chunk(
        id="chunk_partition_02",
        document_id="doc_dynamo_dg",
        source="dynamodb-dg.pdf",
        source_type=SourceType.PDF,
        document_title="Amazon DynamoDB Developer Guide",
        kind="text",
        content="Amazon DynamoDB > Core Components > Partition Keys\n\nThe partition key of an item is also known as its hash attribute. The maximum length of a partition key value is 2048 bytes.",
        block_ids=["b_02"],
        context_block_ids=["h_keys"],
        heading_context=["Amazon DynamoDB", "Core Components", "Partition Keys"],
        page_numbers=[42],
        indexable=True,
    ),
    Chunk(
        id="chunk_bad_table_03",
        document_id="doc_dynamo_dg",
        source="dynamodb-dg.pdf",
        source_type=SourceType.PDF,
        document_title="Amazon DynamoDB Developer Guide",
        kind="table",
        content="Corrupted table content with missing columns",
        block_ids=["b_table_err"],
        context_block_ids=[],
        heading_context=["Corrupted"],
        page_numbers=[99],
        indexable=False,  # <--- QUARANTINED!
    ),
]

print("=" * 65)
print("📦 STEP 2: INDEXING CHUNKS INTO VECTOR STORE...")
print("=" * 65)
print(f"Total chunks submitted: {len(sample_chunks)}")

indexed_count = vector_store.add_chunks(sample_chunks)
print(f"✅ Successfully indexed: {indexed_count} chunk(s)")
print(f"🛡️  Quarantined / Skipped: {len(sample_chunks) - indexed_count} chunk(s) (Safety rule verified!)\n")

# Test Queries
queries = [
    "What is the maximum storage limit for a single record?",
    "How large can a primary partition key be?",
]

print("=" * 65)
print("🔍 STEP 3: PERFORMING SIMILARITY SEARCH QUERIES...")
print("=" * 65)

for query in queries:
    print(f"\n❓ USER QUERY: '{query}'")
    print("-" * 65)
    
    results = vector_store.search(query, top_k=2)
    
    for rank, res in enumerate(results, 1):
        chunk = res.chunk
        print(f"  🏆 Result #{rank} (Cosine Similarity Score: {res.score:.4f})")
        print(f"     📄 Source: {chunk.source} | Page: {chunk.page_numbers}")
        print(f"     🏷️  Breadcrumbs: {' > '.join(chunk.heading_context)}")
        print(f"     📝 Content: {chunk.content.splitlines()[-1]}")
        print()

print("=" * 65)
print("🎉 END-TO-END RETRIEVAL DEMO COMPLETE!")
print("=" * 65 + "\n")