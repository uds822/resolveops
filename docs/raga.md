# ResolveOps RAGAS Evaluation Plan

Status: planning only. No evaluation code, packages, cloud services or runtime changes are implemented by this document.

## 1. Purpose in simple terms

RAGAS helps evaluate how well ResolveOps retrieves evidence and answers questions. It is an evaluation framework, not an answer database and not a replacement for the RAG pipeline.

The golden dataset is a collection of questions with human-reviewed answers and source evidence. It acts as a repeatable test paper: run the same questions before and after a change to see what improved or regressed.

A user question does not have to match this dataset. Every live question follows the normal document retrieval and answer generation path. If the documents contain sufficient evidence, ResolveOps should answer; otherwise, it should acknowledge missing information or ask for clarification.

| Component | Purpose | When it runs |
| --- | --- | --- |
| RAG pipeline | Retrieve documents and generate an answer | During a user request |
| Runtime citation verifier | Check citation markers and assess support for cited claims | During a user request, when enabled |
| Golden dataset | Provide reviewed test questions, answers and evidence | During evaluation |
| RAGAS | Score selected aspects of pipeline outputs | Offline before releases, after changes, or on sampled production results |

Initial decision: include RAGAS as a separate development/evaluation dependency. Do not add the full evaluation suite to every live request or make application startup depend on it.

## 2. Example of an evaluation

Use this fictional document to understand the process:

> Failed jobs may be retried twice. If both retries fail, notify the administrator.

Golden example:

- Question: How many times may I retry a failed job?
- Reference answer: At most twice.
- Expected evidence: The sentence specifying two retries, with its document version and location.

ResolveOps might answer "Retry five times." It addresses the question, so relevance alone may look good. However, the answer contradicts the retrieved document and reference answer, so faithfulness and correctness should expose the failure.

Another user might ask "Who should I notify after retries fail?" even though this question is not in the dataset. ResolveOps should still answer using the document. Its recorded question, answer and retrieved context can later be evaluated without an existing golden match. Reference-based metrics require a reviewed reference to be added first.

Automated judges can make mistakes. Scores are diagnostic evidence, not guarantees of truth.

## 3. Prerequisite: close citation-verification gaps

The earlier offline review passed 120 tests, with four live-provider tests excluded, but targeted probes found gaps. Confirm their current status before collecting a release-quality baseline:

- [ ] A citation after punctuation stays associated with the correct sentence.
- [ ] An answer mixing factual claims with a refusal does not lose all citations or bypass auditing.
- [ ] Claims beyond the verification budget remain explicitly unaudited; a partial audit is not labeled fully verified.
- [ ] Uncited factual claims are distinguished from supported cited claims.
- [ ] Malformed judge fields and exceptions produce an explicit unavailable/failed audit state, not a successful validation result.
- [ ] Judge failures are not presented as proven unsupported claims or as perfect faithfulness.
- [ ] Verification deadlines are enforced; usage accounting includes the judge call separately from answer generation.
- [ ] The UI distinguishes mock results, partial checks, unsupported claims and unavailable verification.

Existing areas to integrate with later: `service.py`, `generation/context_builder.py`, `generation/grounding_verifier.py`, the retrieval adapters, and the API response models. Reuse the actual pipeline instead of building a second evaluation-only retrieval implementation.

Dataset preparation can proceed while these issues are fixed. Label any earlier scores exploratory rather than release evidence. This checklist does not authorize code changes as part of this planning task.

## 4. Build the golden dataset

### Initial scope

Start with 20 reviewed examples to validate the evaluation workflow. Expand to 60 before the first meaningful comparison; grow toward 100 as real failure cases emerge.

Suggested composition for the 60-example version:

| Category | Count | What it tests |
| --- | ---: | --- |
| Direct factual questions | 15 | Ordinary retrieval and answer generation |
| Troubleshooting and procedures | 10 | Correct conditions and ordered steps |
| Tables, limits, units and numbers | 10 | Table extraction and precise facts |
| Similar terminology across services | 10 | Retrieval discrimination and reranking |
| Multi-passage questions | 5 | Combining required evidence |
| Unanswerable questions | 5 | Appropriate abstention |
| Ambiguous or conflicting evidence | 5 | Clarification and honest uncertainty |
| Total | 60 | |

Use documents already indexed in ResolveOps. Do not invent cloud limits from memory. Every factual reference answer must be checked against a specific corpus version.

### Fields to prepare

| Field | Purpose |
| --- | --- |
| Example ID | Stable identity across runs |
| Question | Natural user phrasing |
| Reference answer | Reviewed answer or expected refusal/clarification behavior |
| Essential facts | Facts needed for a correct and complete answer |
| Expected evidence | Document IDs, version/hash, page/section and exact supporting passage |
| Category and difficulty | Breakdown of performance by question type |
| Answerability | Answerable, unanswerable, ambiguous, or conflicting |
| Accepted alternatives | Equivalent wording or valid interpretations |
| Reviewer and review date | Provenance of the reference labels |
| Split | Development or held-out evaluation |

Reference evidence should be anchored to source passages, not only chunk IDs. Chunk IDs may change during chunking experiments; map the evidence to each candidate index version without rewriting the expected facts.

### Review and split rules

- Review every reference against the source. AI-generated questions may be drafts, but they do not become golden labels automatically.
- Divide the 60 examples into 40 development and 20 held-out examples, stratified by category where possible.
- Keep near-duplicate questions and paraphrases in the same split. Do not tune on held-out answers or repeatedly select changes based on held-out results.
- Keep expected answers out of the generator's prompt and out of the searchable corpus; they are evaluation inputs only.
- Version dataset changes. When a source changes, review affected labels instead of silently comparing incompatible runs.
- Track human corrections and disagreements. For a solo project, perform a second review pass; obtain an independent review of a subset if practical.

Exit criterion: each example has a reviewed expected behavior and evidence, with a declared corpus version and split.

## 5. Metrics to include

RAGAS provides metrics for retrieval and generation, including faithfulness, answer relevance and context metrics. Pin a supported version during implementation and verify its required fields and adapters; metric names and interfaces can differ by release. See the [official metrics catalog](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/).

| Measure | Question it answers | Reference needed? | Planned use |
| --- | --- | --- | --- |
| Faithfulness | Are the answer's claims supported by the context supplied to the generator? | No reviewed answer required | Core RAGAS metric |
| Answer relevance | Does the answer address the user's question? | No reviewed answer required | Core RAGAS metric |
| Reference-based context precision | Are relevant retrieved passages ranked ahead of irrelevant ones? | Yes, for the selected reference-based variant | Core retrieval evaluation |
| Reference-based context recall | Does retrieved context cover the required reference information? | Yes | Core retrieval evaluation |
| Answer correctness | Are the response's facts consistent with the reference? | Yes | Human rubric initially; evaluate an appropriate automated metric later |
| Evidence recall@k | Did the top-k results include the labeled required evidence? | Labeled source evidence | Deterministic/project-specific retrieval check |
| Citation support precision | Do cited claim/source pairs actually support the attached claims? | Reviewed support labels or calibrated judging | Separate citation evaluation |
| Citation coverage | Are factual claims accompanied by supporting citations? | Claim-level review/judging | Separate citation evaluation |
| Abstention/clarification correctness | Does the system recognize missing or ambiguous evidence? | Expected behavior | Separate rubric |
| Latency, tokens and cost | What does each answer and evaluation cost? | No | Operational measurements |

Context precision is not citation precision. Retrieving a relevant chunk does not prove that a particular claim correctly cites it. Also, faithfulness to retrieved evidence does not establish that the evidence is current or correct.

The runtime `faithfulness_score` currently describes its verifier's audited claims. Store it separately from RAGAS scores; do not assume the denominators or definitions match.

Do not penalize an appropriate refusal merely because a generic answer metric expects a factual response. Report answerable and unanswerable examples separately. Record undefined scores as not applicable, and evaluator errors as failed evaluations, never automatically as zero or one.

Exit criterion: every metric has a defined input, denominator, applicability rule and failure representation.

## 6. Offline evaluation workflow

1. Select a fixed code revision, configuration, corpus/index generation and dataset version.
2. Run each selected question through the same application pipeline used for ordinary queries.
3. Capture the original question, generated answer, actual context, citations, timing, model identities and usage.
4. Score applicable metrics using the recorded outputs and reviewed references where needed.
5. Inspect low scores, evaluator errors and a sample of apparently good answers.
6. Produce a per-example report plus aggregate results and category breakdowns.
7. Compare a candidate change against the baseline on the same examples and corpus.

Capture three retrieval stages separately where available: candidates before reranking, ranked results, and the bounded excerpts actually included in the generation prompt. Use the final prompt excerpts for answer faithfulness. Use the declared retrieval stage for retrieval metrics. Do not let an evaluator see a full document when the generator only saw a truncated passage.

Run metadata should include:

- Code revision; dataset and corpus fingerprints; index generation.
- Chunking, overlap, top-k, reranking and prompt settings.
- Embedding, answer-generation and evaluation models, including revisions where available.
- RAGAS/dependency versions and evaluation prompts/configuration.
- Cache state, timestamps, hardware/region where relevant, and retry counts.
- Generation, runtime audit and offline evaluation usage/cost as separate totals.

Persist completed records so a failed judging run can resume without regenerating every answer. Never reuse cached scores when the example, context, answer, metric or judge configuration has changed.

## 7. Calibration, experiments and release decisions

### Calibration

- Hand-label the initial 20 examples before treating automated scores as trustworthy.
- Include deliberate wrong answers, unsupported additions, correct refusals, missing citations and misleading numeric units.
- Compare judge results with human labels and document false positives/negatives.
- Prefer an evaluator independent of the answer model when feasible. If the same model is used, disclose that limitation and strengthen manual review.
- Treat documents and model responses as untrusted evaluation data; embedded instructions must not override the judge rubric.
- Repeat judging on a small fixed subset to estimate variability. Even low-temperature judging need not be perfectly repeatable.

### First comparisons

Change one variable at a time:

1. Hybrid retrieval with versus without reranking.
2. Candidate and final context sizes.
3. Current chunking versus one justified alternative, such as bounded prose overlap or neighbor retrieval.
4. Prompt changes that improve evidence use and abstention.

Use development data to select changes, then run the held-out set for a release assessment. Report per-category failures and paired changes, not just an overall average. Small datasets support limited conclusions; show sample counts and avoid declaring tiny score differences meaningful without reviewing variability.

### Initial release policy

- Required examples must have completed pipeline records; evaluation coverage and failures must be visible.
- Known critical cases must pass human-reviewed checks: wrong numerical limits, unsupported operational instructions, and misleading verification badges.
- No newly discovered serious regression may be hidden by a higher average.
- Define numerical regression tolerances after baseline calibration, before evaluating a candidate change. Do not choose thresholds afterward to make a release pass.
- Maintain a manual release decision until automated judges and thresholds are sufficiently calibrated.

## 8. Production use and unfamiliar questions

The live application does not look for a golden-dataset match. It answers from the authorized indexed documents.

Optional production evaluation happens afterward:

1. Sample a small, budgeted set of eligible conversation turns.
2. Preserve the actual context used at answer time, not a later rerun against changed documents.
3. Redact sensitive content and follow user/tenant data rules before sending records to an evaluation provider.
4. Run metrics that do not require references, such as faithfulness and relevance.
5. Send uncertain/failed cases for human review. Add reviewed examples to the development regression set when useful.
6. Run reference-based correctness/recall only when appropriate reference labels exist.

Sampling is asynchronous and must not delay a user response. These sampled scores describe sampled traffic, not every production answer. Publish sampling volume and selection method; a handpicked set of good answers is not a representative production score.

## 9. Cost and deployment boundaries

- Begin with a five-example evaluation smoke run; measure cost before running 20 or 60 examples.
- Set a per-run monetary/token ceiling, concurrency limit, timeout, bounded retries and cancellation behavior.
- Estimate evaluation cost separately from the live RAG calls and runtime citation audit. Some metrics require multiple model calls or embeddings.
- Use existing GCP credits only if the actual evaluation service/model is eligible. External provider charges are separate unless explicitly covered.
- Use saved response records to avoid unnecessary regeneration during metric development.
- Keep offline evaluator dependencies out of the production image unless a separately justified deployment needs them.
- Run deterministic fixtures in routine CI. Trigger paid model evaluations on demand or for selected releases within the budget.
- A mock judge validates integration, not answer quality; label mock reports clearly.

## 10. Proposed artifacts and implementation order

These are future deliverables; this plan does not create them.

| Artifact | Purpose |
| --- | --- |
| `evals/datasets/` | Versioned reviewed examples and label provenance |
| `evals/configs/` | Reproducible pipeline and judge configurations |
| Offline evaluation runner | Capture application outputs, score metrics and resume failed runs |
| Per-example result records | Answers, evidence, scores, errors and resource usage |
| Sanitized comparison report | Baseline versus candidate, category results and failure examples |
| Dataset/metric guide | Definitions, review rules and applicability |

Recommended sequence:

1. Resolve citation-verifier prerequisites and review index/model consistency.
2. Prepare and review 20 examples with a documented corpus snapshot.
3. Choose and pin the RAGAS version, judge and applicable metrics.
4. Implement the offline adapter and a five-example smoke evaluation in a future task.
5. Calibrate on 20 examples and expand to the 60-example split.
6. Run one controlled retrieval improvement experiment and publish the comparison.
7. Deploy the application with evaluation separate from live requests.
8. Add optional production sampling; extend the dataset for conversations and voice later.

Voice evaluation will also need speech-specific measures such as transcription accuracy, end-of-turn latency and interruption behavior. Text RAGAS scores alone do not validate the voice experience.

## 11. Completion criteria and resume evidence

- [ ] Reviewed dataset with source evidence, answerability labels and held-out examples.
- [ ] Reproducible baseline with declared metric definitions and complete error accounting.
- [ ] Actual prompt context captured accurately; golden answers never supplied to the generator.
- [ ] Human calibration documented, including evaluator mistakes.
- [ ] At least one controlled before/after experiment with quality, latency and cost results.
- [ ] Live queries work independently of the golden dataset and evaluation service.
- [ ] Optional production sampling respects privacy and explicit cost limits.
- [ ] Published claims identify corpus size, example count, split, models and limitations.

Resume template, to fill only after measurement:

> Built and deployed ResolveOps with an offline RAGAS evaluation workflow over N reviewed technical questions; improved [named metric] from X to Y while tracking citation support, p95 latency and cost per answer.

Do not describe a faithfulness score as overall accuracy, claim perfect hallucination prevention, or imply that held-out performance guarantees correctness for every user question.

Related plans: [production roadmap](production-scale-roadmap.md), [claim grounding](claim_grounding_plan.md), and [voice integration](voice-agent-plan.md). This document refines the older grounding plan's evaluation section by separating retrieval metrics, citation metrics and reference-based evaluation.
