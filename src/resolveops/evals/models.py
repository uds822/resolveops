"""Pydantic data models for RAGAS evaluation examples, metrics, and reports."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class EvalExample(BaseModel):
    """A golden benchmark evaluation case."""

    model_config = ConfigDict(extra="ignore")

    example_id: str = Field(description="Unique stable identifier")
    question: str = Field(description="Natural user phrasing of the technical question")
    reference_answer: str = Field(description="Human-reviewed ground-truth answer or expected refusal")
    essential_facts: list[str] = Field(default_factory=list, description="Key atomic statements needed for full recall")
    expected_evidence: dict[str, Any] = Field(default_factory=dict, description="Source file and excerpt where evidence resides")
    category: str = Field(default="direct_factual", description="Category: direct_factual, procedures, tables_limits, unanswerable, etc.")
    answerability: str = Field(default="answerable", description="'answerable' or 'unanswerable'")
    split: str = Field(default="dev", description="'dev' or 'held_out'")
    accepted_alternatives: list[str] = Field(default_factory=list, description="Acceptable equivalent answers")


class RagasMetrics(BaseModel):
    """Calculated RAGAS metrics for a single query."""

    model_config = ConfigDict(extra="forbid")

    context_precision: float | None = Field(default=None, description="MAP@K of retrieved chunks against expected evidence (0.0 - 1.0)")
    context_recall: float | None = Field(default=None, description="Ratio of essential facts covered by retrieved context (0.0 - 1.0)")
    faithfulness: float | None = Field(default=None, description="Ratio of answer claims entailed by retrieved context (0.0 - 1.0)")
    answer_relevance: float | None = Field(default=None, description="Semantic relevance of answer to the user query (0.0 - 1.0)")
    abstention_correctness: float | None = Field(default=None, description="Correct refusal indicator for unanswerable queries (0.0 or 1.0)")


class EvalResult(BaseModel):
    """Full execution and scoring profile for an evaluated example."""

    model_config = ConfigDict(extra="ignore")

    example_id: str
    question: str
    reference_answer: str
    generated_answer: str
    category: str
    answerability: str
    split: str
    retrieved_sources: list[str] = Field(default_factory=list)
    cited_sources: list[str] = Field(default_factory=list)
    metrics: RagasMetrics
    latency_ms: float = 0.0
    tokens_used: int = 0
    audit_passed: bool = True
    error: str | None = None
    notes: list[str] = Field(default_factory=list)


class CategorySummary(BaseModel):
    """Aggregate metric scores grouped by question category."""

    model_config = ConfigDict(extra="forbid")

    category: str
    count: int
    mean_context_precision: float | None = None
    mean_context_recall: float | None = None
    mean_faithfulness: float | None = None
    mean_answer_relevance: float | None = None
    abstention_rate: float | None = None


class RagasReport(BaseModel):
    """Complete summary report of an offline evaluation benchmark run."""

    model_config = ConfigDict(extra="ignore")

    run_id: str
    timestamp: str
    dataset_file: str
    total_examples: int
    answerable_count: int
    unanswerable_count: int
    mean_context_precision: float
    mean_context_recall: float
    mean_faithfulness: float
    mean_answer_relevance: float
    abstention_accuracy: float
    mean_latency_ms: float
    category_summaries: list[CategorySummary] = Field(default_factory=list)
    results: list[EvalResult] = Field(default_factory=list)
    release_gate_passed: bool = True
    gate_notes: list[str] = Field(default_factory=list)

