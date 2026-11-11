import os
import sys
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resolveops.retrieval.embeddings import make_embedding_model

print("\n" + "=" * 70)
print("🌐 TESTING CLOUD EMBEDDING CONNECTIONS...")
print("=" * 70)

test_query = "What is the maximum item size in Amazon DynamoDB?"
test_doc = "Amazon DynamoDB limits maximum item size to 400 KB including attribute names."

# 1. Test NVIDIA NIM with the current Nemotron model
nv_key = os.environ.get("NVIDIA_API_KEY", "").strip()
if nv_key:
    # We test Nemotron 3.5 / 3 Embed from your NVIDIA catalog
    candidate_models = ["nvidia/nemotron-3.5-embed-1b", "nvidia/nemotron-3-embed-1b"]
    success = False
    for model_name in candidate_models:
        print(f"\n🟢 Testing NVIDIA NIM: 'nvidia:{model_name}'...")
        try:
            t0 = time.monotonic()
            nv_model = make_embedding_model(f"nvidia:{model_name}")
            q_vec = nv_model.embed_query(test_query)
            d_vec = nv_model.embed_documents([test_doc])[0]
            elapsed = round(time.monotonic() - t0, 3)
            print(f"  ✅ SUCCESS! Generated embeddings in {elapsed}s")
            print(f"  • Model:       {nv_model.model_name}")
            print(f"  • Dimensions:  {len(q_vec)} (high resolution)")
            print(f"  • Query vec:   {[round(x, 4) for x in q_vec[:4]]}...")
            success = True
            break
        except Exception as exc:
            print(f"  ❌ Error for {model_name}: {exc}")
    if not success:
        print("  ⚠️ Could not connect to tested Nemotron models. Check endpoint name in build.nvidia.com.")
else:
    print("\n⚠️ [1/2] NVIDIA_API_KEY not found in .env.")

# 2. Test OpenRouter
or_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
if or_key:
    print("\n🟢 Testing OpenRouter API ('openrouter:openai/text-embedding-3-small')...")
    try:
        t0 = time.monotonic()
        or_model = make_embedding_model("openrouter:openai/text-embedding-3-small")
        q_vec = or_model.embed_query(test_query)
        d_vec = or_model.embed_documents([test_doc])[0]
        elapsed = round(time.monotonic() - t0, 3)
        print(f"  ✅ SUCCESS! Generated embeddings in {elapsed}s")
        print(f"  • Model:       {or_model.model_name}")
        print(f"  • Dimensions:  {len(q_vec)} dimensions")
        print(f"  • Query vec:   {[round(x, 4) for x in q_vec[:4]]}...")
    except Exception as exc:
        print(f"  ❌ OpenRouter Error: {exc}")
else:
    print("\n⚠️ [2/2] OPENROUTER_API_KEY not found in .env.")

print("\n" + "=" * 70)
print("🎉 CLOUD CONNECTION TEST COMPLETE!")
print("=" * 70 + "\n")

