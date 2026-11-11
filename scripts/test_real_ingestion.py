import json
import sys
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resolveops.ingestion.pipeline import ingest_document
from resolveops.retrieval.embeddings import make_embedding_model
from resolveops.retrieval.vector_store import InMemoryVectorStore

pdf_path = "data/raw/aws/pdf/dynamodb-dg.pdf"

print("\n" + "=" * 70)
print("🚀 STEP 1: INITIALIZING LOCAL EMBEDDING MODEL & VECTOR STORE...")
print("=" * 70)

embedding_model = make_embedding_model("all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(embedding_model)
print("✅ Local MiniLM model and In-Memory Vector Store ready!\n")

print("=" * 70)
print(f"📄 STEP 2: INGESTING REAL AWS PDF ({pdf_path}, Pages 57-58)...")
print("   Engine: Docling (Deep-Learning Layout & TableFormer)")
print("=" * 70)

started = time.monotonic()
result = ingest_document(
    file_path=pdf_path,
    vector_store=vector_store,
    page_range=(57, 58),
    pdf_engine="docling",
    max_chars=1800,
)

print(f"\n📋 INGESTION RECEIPT:")
print("-" * 70)
print(f"  • Status:             {result.status.upper()}")
print(f"  • Document ID:        {result.document_id}")
print(f"  • Pages Processed:    {result.pages_processed}")
print(f"  • Total Blocks:       {result.total_blocks}")
print(f"  • Total Chunks:       {result.total_chunks}")
print(f"  • Indexed Chunks:     {result.indexed_chunks}")
print(f"  • Quarantined Chunks: {result.quarantined_chunks}")
print(f"  • Pipeline Time:      {result.elapsed_seconds}s")
if result.error:
    print(f"  • Error:              {result.error}")
print("-" * 70 + "\n")

if result.status == "failed":
    print("❌ Ingestion failed! Check error details above.")
    sys.exit(1)

print("=" * 70)
print("🧩 STEP 3: INSPECTING STORED CHUNKS IN VECTOR DATABASE...")
print("=" * 70)

table_chunks = [
    c for c in vector_store._chunks.values()
    if c.kind == "table"
]
text_chunks = [
    c for c in vector_store._chunks.values()
    if c.kind == "text"
]

print(f"✅ Stored Text Chunks:  {len(text_chunks)}")
print(f"✅ Stored Table Chunks: {len(table_chunks)}")

if table_chunks:
    table = table_chunks[0]
    print("\n🔍 PREVIEW OF STORED TABLE CHUNK (Page 58):")
    print(f"  • Breadcrumbs: {' > '.join(table.heading_context)}")
    print(f"  • Page:        {table.page_numbers}")
    print(f"  • Table Cells: {len(table.table_cells or [])} rich cells preserved")
    print("  • Markdown Content Preview:")
    for line in table.content.splitlines()[:5]:
        print(f"    {line}")
    print("    ...")

print("\n" + "=" * 70)
print("🔍 STEP 4: REAL SEMANTIC SEARCH QUERIES ON INGESTED PDF...")
print("=" * 70)

test_queries = [
    "What dependencies were updated in version 3.3.2?",
    "When was PartiQL support added to DynamoDB local?",
    "How does DynamoDB local help developers test applications?",
]

for query in test_queries:
    print(f"\n❓ USER QUERY: '{query}'")
    print("-" * 70)
    
    results = vector_store.search(query, top_k=2)
    
    if not results:
        print("  ⚠️ No matching chunks found.")
        continue

    for rank, res in enumerate(results, 1):
        chunk = res.chunk
        first_line = chunk.content.splitlines()[0]
        snippet = chunk.content.splitlines()[1:3]
        print(f"  🏆 Result #{rank} [Cosine Similarity Score: {res.score:.4f}]")
        print(f"     📄 Source: {chunk.source} | Page: {chunk.page_numbers} | Kind: {chunk.kind.upper()}")
        print(f"     🏷️  Context: {' > '.join(chunk.heading_context)}")
        print(f"     📝 Preview: {first_line}")
        for s in snippet:
            if s.strip():
                print(f"        {s.strip()[:100]}...")
        print()

print("=" * 70)
print("🎉 REAL-DOCUMENT QUALITY TEST COMPLETE!")
print("=" * 70 + "\n")

