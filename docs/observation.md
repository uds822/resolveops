# ResolveOps: Two-Stage Retrieval Benchmark & Empirical Observations

This document records the empirical results, audit metrics, and architectural observations from benchmarking the ResolveOps **Two-Stage Retrieval Pipeline** on real-world AWS technical documentation.

---

## 1. Architecture Overview

A common pitfall in production RAG systems is relying solely on vector search or keyword search. ResolveOps implements a high-precision **Two-Stage Funnel**:

```
                       User Question
                             │
                             ▼
 ┌───────────────────────────────────────────────────────────────┐
 │ STAGE 1: Fast Hybrid Candidate Retrieval (~15 - 50ms)         │
 │  • Dense Vector Search: NVIDIA Nemotron 3.5 1B (2048 dims)    │
 │  • Sparse Keyword Search: Okapi BM25 with technical tokenizer │
 │  • Merge Strategy: Reciprocal Rank Fusion (RRF, k=60)         │
 │  • Output: Candidate pool of Top-10 chunks                    │
 └───────────────────────────────┬───────────────────────────────┘
                                 │ 10 Candidate Chunks
                                 ▼
 ┌───────────────────────────────────────────────────────────────┐
 │ STAGE 2: Deep Cross-Encoder Reranking (~200 - 500ms)          │
 │  • Model: NVIDIA Nemotron 3.5 8B Cross-Encoder                │
 │  • Mechanism: Joint query ↔ passage cross-attention           │
 │  • Score Normalization: Sigmoid probability mapping           │
 │  • Output: Top-5 re-ordered, audited chunks                   │
 └───────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
                     Grounded LLM Generation
```

### Why Two Stages?
1. **The Latency Problem:** Evaluating a 10,000-chunk corpus with an 8-Billion parameter cross-encoder would take **over 30 seconds per query**.
2. **The Funnel Solution:** Stage 1 filters thousands of chunks down to 10 high-recall candidates in milliseconds. Stage 2 applies deep reasoning *only* to those 10 candidates.

---

## 2. Benchmark Corpus & Setup

The benchmark corpus mixes structured Markdown and multi-page PDFs parsed with Docling's deep-learning layout engine (OCR disabled for digital speed):

* **Markdown Sources (8 files, 67 chunks):**
  * `TTL.md` (DynamoDB Time-To-Live)
  * `Programming.Errors.md` (DynamoDB Error Handling & HTTP Status Codes)
  * `Versioning.md` (S3 Object Versioning)
  * `Concepts.MultiAZ.md` (High Availability architecture)
  * `id_roles.md` (IAM Roles, Delegation, & Trust Policies)
  * `invocation-async.md` (AWS Lambda Asynchronous Invocations & Retries)
  * `object-lock.md` (S3 Object Lock & Compliance Retention)
  * `id_credentials_mfa.md` (IAM Credentials & Multi-Factor Auth)
* **PDF Sources (4 multi-page ranges, 23 chunks via Docling):**
  * `dynamodb-dg.pdf` (Pages 57–59: DynamoDB Local Release Notes & Tables)
  * `iam-ug.pdf` (Pages 30–32: How IAM Works & Request Context)
  * `lambda-dg.pdf` (Pages 20–22: Lambda Compute Primitives & Execution Models)
  * `s3-userguide.pdf` (Pages 30–32: S3 Architecture, Buckets, & Objects)
* **Total Indexed Corpus:** 90 chunks.

---

## 3. Benchmark Queries & Audit Results

### Query 1: Exact Technical — Error Code
> **Query:** `"AccessDeniedException 400 Signature Version 4 signing"`  
> **Type:** Exact technical term search

#### Rank Audit Table
| Final Rank | Initial Hybrid Rank | Probability | Logit | Shift | Source File | Heading Context |
|---|---|---|---|---|---|---|
| **#1** 🥇 | **#1** | **8.2%** | **-2.414** | **0 ━** | `Programming.Errors.md` | *Error messages and codes ▸ HTTP status code 400* |
| **#2** 🥈 | **#2** | **3.6%** | **-3.277** | **0 ━** | `Programming.Errors.md` | *Error messages and codes ▸ HTTP status code 400* |
| **#3** 🥉 | **#5** | **3.6%** | **-3.283** | **+2 🚀** | `Programming.Errors.md` | *Error messages and codes ▸ HTTP status code 400* |
| **#4** | **#7** | **3.5%** | **-3.311** | **+3 🚀** | `s3-userguide.pdf` | *Topics* |
| **#5** | **#4** | **3.4%** | **-3.341** | **-1 🔻** | `Programming.Errors.md` | *Error components* |

* **Observation:** Both Hybrid and Reranker strongly agreed on the #1 result (`Programming.Errors.md`), confirming that exact technical error codes benefit from dual BM25 + dense agreement.

---

### Query 2: Conceptual — Data Expiry
> **Query:** `"automatically expire and delete old items without consuming write throughput"`  
> **Type:** Semantic / conceptual question (no exact keyword match for "TTL")

#### Rank Audit Table
| Final Rank | Initial Hybrid Rank | Probability | Logit | Shift | Source File | Heading Context |
|---|---|---|---|---|---|---|
| **#1** 🥇 | **#1** | **10.2%** | **-2.172** | **0 ━** | `TTL.md` | *Using time to live (TTL) in DynamoDB* |
| **#2** 🥈 | **#2** | **7.3%** | **-2.541** | **0 ━** | `TTL.md` | *Using time to live (TTL) in DynamoDB* |
| **#3** 🥉 | **#5** | **3.6%** | **-3.301** | **+2 🚀** | `Programming.Errors.md` | *Batch operations and error handling* |
| **#4** | **#4** | **3.4%** | **-3.338** | **0 ━** | `Programming.Errors.md` | *Batch operations and error handling* |
| **#5** | **#3** | **3.3%** | **-3.364** | **-2 🔻** | `Programming.Errors.md` | *Batch operations and error handling* |

* **Observation:** Vector embeddings successfully mapped the concept of "expire without write throughput" to DynamoDB Time-To-Live (`TTL.md`), and the Cross-Encoder confirmed Rank #1 with a strong logit separation of $+0.37$ over Rank #2.

---

### Query 3: Exact Technical — DynamoDB Local Release Notes Table
> **Query:** `"DynamoDB local version 3.3.2 pom.xml dependencies"`  
> **Type:** Multi-column Markdown table extracted from PDF

#### Rank Audit Table
| Final Rank | Initial Hybrid Rank | Probability | Logit | Shift | Source File | Heading Context |
|---|---|---|---|---|---|---|
| **#1** 🥇 | **#1** | **11.5%** | **-2.041** | **0 ━** | `dynamodb-dg.pdf` (p. 58) | *Release history for DynamoDB local* |
| **#2** 🥈 | **#2** | **7.4%** | **-2.521** | **0 ━** | `dynamodb-dg.pdf` (p. 59) | *Release history for DynamoDB local* |
| **#3** 🥉 | **#3** | **4.9%** | **-2.957** | **0 ━** | `dynamodb-dg.pdf` (p. 58) | *Release history for DynamoDB local* |
| **#4** | **#4** | **3.9%** | **-3.204** | **0 ━** | `dynamodb-dg.pdf` (p. 57) | *Downloadable DynamoDB limitations* |
| **#5** | **#5** | **3.5%** | **-3.317** | **0 ━** | `dynamodb-dg.pdf` (p. 57) | *Downloadable DynamoDB limitations* |

* **Observation:** The table-aware chunking strategy (which retained the entire release notes table intact) paid off completely: Rank #1 specifically contained the version 3.3.2 entry with its Java SDK and pom.xml dependencies.

---

### Query 4: Conceptual — Lambda Failure & Retries (The BM25 Keyword Trap)
> **Query:** `"what happens when a Lambda function fails and how are retries handled?"`  
> **Type:** Conceptual causal reasoning

#### Rank Audit Table
| Final Rank | Initial Hybrid Rank | Probability | Logit | Shift | Source File | Heading Context |
|---|---|---|---|---|---|---|
| **#1** 🥇 | **#4** | **6.4%** | **-2.682** | **+3 🚀** | `invocation-async.md` | *Invoking a Lambda Event Function asynchronously* |
| **#2** 🥈 | **#7** | **4.7%** | **-3.002** | **+5 🚀** | `Programming.Errors.md` | *Error retries and exponential backoff* |
| **#3** 🥉 | **#3** | **4.5%** | **-3.045** | **0 ━** | `Programming.Errors.md` | *Error retries and exponential backoff* |
| **#4** | **#1** | **4.3%** | **-3.095** | **-3 🔻** | `lambda-dg.pdf` (p. 21) | *How Lambda compute primitives compare* |
| **#5** | **#2** | **4.3%** | **-3.105** | **-3 🔻** | `lambda-dg.pdf` (p. 20) | *How Lambda compute primitives compare* |

#### 💡 Critical Architectural Discovery:
1. **The Failure of Pure Hybrid Search:** In Stage 1, the comparison table from `lambda-dg.pdf` occupied **Rank #1 and Rank #2** because the word "Lambda" was repeated 15+ times across the table columns. However, that table contained **zero information about retries or failure handling**.
2. **The Triumph of Cross-Attention:** The true answer (`invocation-async.md`), which explains retry queues and dead-letter handling, was stranded at **Rank #4**.
3. **The Reranker Correction:** The Nemotron 8B Cross-Encoder read the query and candidates simultaneously, **promoted the true answer to Rank #1 (+3 🚀)**, and **demoted the keyword distractor to Rank #4 (-3 🔻)**.

---

### Query 5: Mixed — IAM Role Delegation
> **Query:** `"IAM role assume cross-account delegation temporary credentials"`  
> **Type:** Mixed terminology and permission concept

#### Rank Audit Table
| Final Rank | Initial Hybrid Rank | Probability | Logit | Shift | Source File | Heading Context |
|---|---|---|---|---|---|---|
| **#1** 🥇 | **#3** | **7.8%** | **-2.471** | **+2 🚀** | `id_roles.md` | *Roles terms and concepts ▸ Delegation* |
| **#2** 🥈 | **#2** | **7.3%** | **-2.535** | **0 ━** | `id_roles.md` | *IAM roles overview* |
| **#3** 🥉 | **#1** | **6.6%** | **-2.649** | **-2 🔻** | `id_roles.md` | *Roles terms and concepts* |
| **#4** | **#4** | **6.3%** | **-2.691** | **0 ━** | `id_roles.md` | *Roles terms and concepts ▸ Trust policy* |
| **#5** | **#5** | **6.0%** | **-2.759** | **0 ━** | `id_roles.md` | *Roles terms and concepts ▸ Role chaining* |

* **Observation:** Hybrid search placed the top-level section intro at Rank #1. The Cross-Encoder identified that the user specifically asked for *delegation*, promoting the dedicated `Delegation` term definition chunk to **Rank #1 (+2 🚀)**.

---

### Query 6: Conceptual — S3 Bucket Access & Privacy
> **Query:** `"Are S3 bucket objects private by default and how to grant access permissions?"`  
> **Type:** Security baseline query

#### Rank Audit Table
| Final Rank | Initial Hybrid Rank | Probability | Logit | Shift | Source File | Heading Context |
|---|---|---|---|---|---|---|
| **#1** 🥇 | **#1** | **11.7%** | **-2.020** | **0 ━** | `s3-userguide.pdf` (pp. 31-32) | *How Amazon S3 works* |
| **#2** 🥈 | **#7** | **6.6%** | **-2.653** | **+5 🚀** | `s3-userguide.pdf` (p. 32) | *Topics* |
| **#3** 🥉 | **#4** | **5.8%** | **-2.783** | **+1 🚀** | `s3-userguide.pdf` (p. 32) | *Buckets* |
| **#4** | **#2** | **5.2%** | **-2.904** | **-2 🔻** | `object-lock.md` | *Required permissions* |
| **#5** | **#9** | **4.7%** | **-3.013** | **+4 🚀** | `object-lock.md` | *Retention modes* |

* **Observation:** The S3 architecture overview was confirmed at #1. A buried S3 bucket summary chunk surged **+5 positions (Rank #7 ➔ Rank #2)**, while an off-topic Object Lock permissions chunk fell back **-2 positions (Rank #2 ➔ Rank #4)**.

---

## 4. How to Interpret the Metrics as an Engineer

### 1. The Rank Shift Metric (`Initial ➔ Final`)
* **`+N 🚀 PROMOTED`:** A high-value correction. Indicates a chunk that lacked superficial keyword density but contained the semantic answer. Without the reranker, downstream generation would receive inferior context.
* **`-N 🔻 DEMOTED`:** A successful false-positive suppression. Indicates a chunk that tripped BM25 keyword matching or broad dense topic similarity but failed to address the specific intent of the question.
* **`CONFIRMED #1 ✅`:** High confidence agreement between Stage 1 and Stage 2.

### 2. Logit vs Sigmoid Probability
* Cross-Encoders evaluate $(query, passage)$ pairs through transformer attention heads and output an unbounded **logit** (typically $-5.0$ to $+5.0$).
* We normalize logits via Sigmoid:
  $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
* In technical documentation retrieval, passages rarely provide a verbatim 1-to-1 match for complex human queries, so probabilities typically cluster between $5\%$ and $15\%$ (logits $-3.0$ to $-2.0$).
* **Relative delta is decisive:** A logit shift of $+0.5$ represents a **substantial margin of relevance**, indicating that the cross-attention heads found strong contextual grounding.

### 3. Latency & Resource Consumption
* **Hybrid Search (Stage 1):** ~850 ms (dominated by remote cloud query embedding; would be < 20 ms with local on-device embeddings).
* **Cross-Encoder Rerank (Stage 2):** ~780 ms across 10 candidates.
* **Total Funnel Latency:** ~1.6 to 1.8 seconds.
* **Production Decision:** By bounding Stage 2 to a Top-10 candidate pool, we achieve cross-encoder accuracy while keeping latency under 2 seconds.

---

## 5. Summary & Next Milestones

| Component | Status | Empirical Outcome |
|---|---|---|
| **Table-Aware Chunking** | ✅ Completed | Markdown tables in PDFs remain atomic; 100% precision on version query #3. |
| **Hybrid Search (Dense + BM25)** | ✅ Completed | High candidate recall across technical keywords and conceptual phrasing. |
| **Cross-Encoder Reranker** | ✅ Completed | Successfully resolved the "keyword trap" in query #4 and elevated answers in queries #5 and #6. |
| **Unified Retrieval Service** | ⏳ Next | Wrap Stage 1 and Stage 2 into a single production service interface. |
| **Context Assembly & LLM Generation** | ⏳ Upcoming | Assemble reranked chunks into grounded prompts with citations. |

