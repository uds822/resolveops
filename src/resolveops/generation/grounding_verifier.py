"""Semantic claim-level grounding and citation entailment verification."""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field

from resolveops.generation.context_builder import AssembledPrompt, CitationReference
from resolveops.generation.llm import LLMClient


class ClaimStatus(str, Enum):
    """Grounding status of an individual citation claim."""

    SUPPORTED = "supported"          # Source explicitly confirms or unambiguously entails claim
    UNSUPPORTED = "unsupported"      # Source does not contain claim, contradicts it, or claim exaggerates
    PARTIAL = "partial"              # Source partially supports, minor unverified extrapolation
    UNVERIFIED = "unverified"        # Verification was skipped, timed out, or unparseable


class ClaimVerification(BaseModel):
    """Detailed audit verdict for a single sentence claim tied to a citation."""

    model_config = ConfigDict(extra="forbid")

    claim_id: int = Field(description="1-based sequence identifier for the audited claim")
    sentence: str = Field(description="The specific claim sentence carrying the citation marker")
    cited_marker: int = Field(description="The citation index marker [i] referenced in the sentence")
    source_filename: str = Field(description="Base filename of the cited document")
    status: ClaimStatus = Field(description="Verification verdict: supported, unsupported, partial, or unverified")
    evidence_quote: str | None = Field(default=None, description="Exact or closely matching quote from source confirming claim")
    explanation: str = Field(default="", description="Concise rationale explaining the verdict")


class GroundingAuditReport(BaseModel):
    """Aggregated semantic grounding audit report with quantitative faithfulness score."""

    model_config = ConfigDict(extra="forbid")

    total_claims: int = Field(default=0, description="Total number of cited claims audited")
    supported_claims: int = Field(default=0, description="Count of claims verified as supported")
    unsupported_claims: int = Field(default=0, description="Count of claims lacking source support or contradicted")
    faithfulness_score: float = Field(default=1.0, description="Ratio of supported claims (0.0 to 1.0)")
    claims: list[ClaimVerification] = Field(default_factory=list, description="List of per-claim verification verdicts")
    unsupported_details: list[ClaimVerification] = Field(default_factory=list, description="Subset of claims flagged as unsupported")
    audit_status: str = Field(default="verified", description="verified, skipped, failed, or no_claims")
    latency_ms: float = Field(default=0.0, description="Latency of the batch audit verification in milliseconds")


def extract_cited_claims(
    answer: str,
    citations: Sequence[CitationReference],
) -> list[tuple[int, str, int, CitationReference]]:
    """Deconstruct generated answer into sentence claims that carry citation markers [i].

    Returns:
        List of tuples: (claim_id, clean_sentence, marker_int, citation_reference)
    """
    if not answer or not citations:
        return []

    cit_map = {c.index: c for c in citations}
    claims: list[tuple[int, str, int, CitationReference]] = []
    claim_counter = 1

    # Collapse fenced code blocks with trailing citations (e.g. ```xml\n...\n```[1]) into one line
    # so the verifier audits the actual code snippet instead of an isolated "```[1]" fence.
    normalized_answer = re.sub(
        r"```[a-zA-Z0-9_\-]*\n([\s\S]*?)```(\s*(?:\[\d+\]\s*)+)",
        lambda m: " ".join(m.group(1).split()) + " " + m.group(2).strip(),
        answer,
    )

    # Split answer into lines/blocks (preserving bullet points and lists)
    lines = [ln.strip() for ln in normalized_answer.splitlines() if ln.strip()]

    for line in lines:
        # Split line into sentences while preserving sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", line)
        for sent in sentences:
            sent_clean = sent.strip()
            if not sent_clean:
                continue

            markers = [int(m) for m in re.findall(r"\[(\d+)\]", sent_clean)]
            for marker in markers:
                if marker in cit_map:
                    claims.append((claim_counter, sent_clean, marker, cit_map[marker]))
                    claim_counter += 1

    return claims


VERIFICATION_SYSTEM_PROMPT = """You are an expert RAG Fact-Checking and Grounding Auditor.
Your mission is to rigorously verify whether technical claim statements are factually supported by their cited source excerpts.

Evaluation Standards:
1. "supported": The cited source excerpt directly confirms or unambiguously entails the claim.
2. "unsupported": The cited source excerpt does NOT contain this information, contradicts the claim, or the claim makes an exaggerated or ungrounded assertion.

You MUST respond with a strict JSON array containing an evaluation object for each claim:
[
  {
    "claim_id": 1,
    "status": "supported" or "unsupported",
    "evidence_quote": "exact quote from source confirming the claim" (or null if unsupported),
    "explanation": "concise 1-sentence rationale"
  }
]
Do NOT include any markdown code blocks, conversational introductions, or commentary. Output pure JSON."""


class GroundingVerifier(ABC):
    """Abstract interface for citation entailment verifiers."""

    @abstractmethod
    def verify_claims(
        self,
        answer: str,
        citations: Sequence[CitationReference],
    ) -> GroundingAuditReport:
        """Verify semantic entailment between all cited sentences in answer and their sources."""
        ...


class LLMGroundingVerifier(GroundingVerifier):
    """Production Tier 2 claim-level verifier using Strategy A (Single Batch LLM Call)."""

    def __init__(
        self,
        llm_client: LLMClient,
        max_claims_budget: int = 10,
        timeout: float = 30.0,
    ) -> None:
        self.llm_client = llm_client
        self.max_claims_budget = max_claims_budget
        self.timeout = timeout

    def verify_claims(
        self,
        answer: str,
        citations: Sequence[CitationReference],
    ) -> GroundingAuditReport:
        t0 = time.monotonic()
        claims_meta = extract_cited_claims(answer, citations)

        if not claims_meta:
            return GroundingAuditReport(
                total_claims=0,
                supported_claims=0,
                unsupported_claims=0,
                faithfulness_score=1.0,
                claims=[],
                unsupported_details=[],
                audit_status="no_claims",
                latency_ms=0.0,
            )

        # Cap at maximum claims budget to prevent latency runaway
        audited_meta = claims_meta[: self.max_claims_budget]

        # Assemble Strategy A: Single Batch Verification Prompt
        claim_blocks: list[str] = []
        for claim_id, sentence, marker, cit in audited_meta:
            source_text = cit.content or cit.snippet or "(No excerpt provided)"
            block = (
                f"[Claim {claim_id}]\n"
                f"Sentence: \"{sentence}\"\n"
                f"Cited Source: [{marker}] {cit.source}\n"
                f"Source Excerpt:\n\"{source_text}\""
            )
            claim_blocks.append(block)

        user_content = (
            "Verify the following claims against their respective cited source excerpts:\n\n"
            + "\n\n".join(claim_blocks)
        )

        assembled_prompt = AssembledPrompt(
            system_prompt=VERIFICATION_SYSTEM_PROMPT,
            user_prompt=user_content,
            citations=[],
        )

        try:
            gen_res = self.llm_client.generate(
                prompt=assembled_prompt,
                max_tokens=2048,
                temperature=0.0,
            )
            raw_output = gen_res.answer.strip()
        except Exception as exc:
            latency_ms = (time.monotonic() - t0) * 1000
            return GroundingAuditReport(
                total_claims=len(audited_meta),
                supported_claims=0,
                unsupported_claims=len(audited_meta),
                faithfulness_score=0.0,
                claims=[
                    ClaimVerification(
                        claim_id=cid,
                        sentence=s,
                        cited_marker=m,
                        source_filename=c.source,
                        status=ClaimStatus.UNVERIFIED,
                        explanation=f"LLM verification call failed: {exc}",
                    )
                    for cid, s, m, c in audited_meta
                ],
                unsupported_details=[],
                audit_status=f"error: {str(exc)}",
                latency_ms=round(latency_ms, 2),
            )

        latency_ms = (time.monotonic() - t0) * 1000

        # Parse JSON output robustly
        cleaned_json = raw_output
        if "```" in cleaned_json:
            cleaned_json = re.sub(r"^```(?:json)?\s*", "", cleaned_json, flags=re.MULTILINE)
            cleaned_json = re.sub(r"```\s*$", "", cleaned_json, flags=re.MULTILINE)
        cleaned_json = cleaned_json.strip()

        # Locate JSON array
        json_match = re.search(r"\[\s*\{.*\}\s*\]", cleaned_json, flags=re.DOTALL)
        if json_match:
            cleaned_json = json_match.group(0)

        try:
            verdicts = json.loads(cleaned_json)
            if not isinstance(verdicts, list):
                verdicts = [verdicts]
        except Exception:
            # Fallback if parsing fails
            return GroundingAuditReport(
                total_claims=len(audited_meta),
                supported_claims=0,
                unsupported_claims=len(audited_meta),
                faithfulness_score=0.0,
                claims=[
                    ClaimVerification(
                        claim_id=cid,
                        sentence=s,
                        cited_marker=m,
                        source_filename=c.source,
                        status=ClaimStatus.UNVERIFIED,
                        explanation=f"Malformed judge JSON response: {raw_output[:120]}...",
                    )
                    for cid, s, m, c in audited_meta
                ],
                unsupported_details=[],
                audit_status="parse_failed",
                latency_ms=round(latency_ms, 2),
            )

        verdict_by_id: dict[int, dict[str, Any]] = {}
        for item in verdicts:
            if isinstance(item, dict) and "claim_id" in item:
                verdict_by_id[int(item["claim_id"])] = item

        # Build final claim verifications mapped back to audited claims
        claim_verifications: list[ClaimVerification] = []
        supported_count = 0
        unsupported_count = 0
        unsupported_details: list[ClaimVerification] = []

        for cid, sent, marker, cit in audited_meta:
            v_data = verdict_by_id.get(cid)
            if v_data:
                status_raw = str(v_data.get("status", "unsupported")).strip().lower()
                status = ClaimStatus.SUPPORTED if status_raw == "supported" else ClaimStatus.UNSUPPORTED
                quote = v_data.get("evidence_quote")
                expl = str(v_data.get("explanation", ""))
            else:
                status = ClaimStatus.UNVERIFIED
                quote = None
                expl = "Missing in judge response."

            if status == ClaimStatus.SUPPORTED:
                supported_count += 1
            else:
                unsupported_count += 1

            cv = ClaimVerification(
                claim_id=cid,
                sentence=sent,
                cited_marker=marker,
                source_filename=cit.source,
                status=status,
                evidence_quote=quote,
                explanation=expl,
            )
            claim_verifications.append(cv)
            if status != ClaimStatus.SUPPORTED:
                unsupported_details.append(cv)

        total = len(claim_verifications)
        faithfulness = round(supported_count / total, 2) if total > 0 else 1.0

        return GroundingAuditReport(
            total_claims=total,
            supported_claims=supported_count,
            unsupported_claims=unsupported_count,
            faithfulness_score=faithfulness,
            claims=claim_verifications,
            unsupported_details=unsupported_details,
            audit_status="verified",
            latency_ms=round(latency_ms, 2),
        )


class MockGroundingVerifier(GroundingVerifier):
    """Deterministic offline verifier for unit tests and local CI pipelines."""

    def __init__(
        self,
        default_status: ClaimStatus = ClaimStatus.SUPPORTED,
        unsupported_keywords: Sequence[str] | None = None,
    ) -> None:
        self.default_status = default_status
        self.unsupported_keywords = [k.lower() for k in (unsupported_keywords or ["unsupported", "50 mb", "fake", "hallucinated"])]

    def verify_claims(
        self,
        answer: str,
        citations: Sequence[CitationReference],
    ) -> GroundingAuditReport:
        t0 = time.monotonic()
        claims_meta = extract_cited_claims(answer, citations)

        if not claims_meta:
            return GroundingAuditReport(
                total_claims=0,
                supported_claims=0,
                unsupported_claims=0,
                faithfulness_score=1.0,
                claims=[],
                unsupported_details=[],
                audit_status="no_claims",
                latency_ms=0.0,
            )

        verifications: list[ClaimVerification] = []
        supported_count = 0
        unsupported_count = 0
        unsupported_details: list[ClaimVerification] = []

        for cid, sentence, marker, cit in claims_meta:
            sent_lower = sentence.lower()
            is_unsupported = any(kw in sent_lower for kw in self.unsupported_keywords)

            if is_unsupported or self.default_status == ClaimStatus.UNSUPPORTED:
                status = ClaimStatus.UNSUPPORTED
                quote = None
                expl = "Claim contradicted or not mentioned in mock source."
                unsupported_count += 1
            else:
                status = ClaimStatus.SUPPORTED
                quote = f"Mock evidence supporting claim {cid}"
                expl = "Directly supported by source."
                supported_count += 1

            cv = ClaimVerification(
                claim_id=cid,
                sentence=sentence,
                cited_marker=marker,
                source_filename=cit.source,
                status=status,
                evidence_quote=quote,
                explanation=expl,
            )
            verifications.append(cv)
            if status != ClaimStatus.SUPPORTED:
                unsupported_details.append(cv)

        total = len(verifications)
        faithfulness = round(supported_count / total, 2) if total > 0 else 1.0
        latency_ms = (time.monotonic() - t0) * 1000

        return GroundingAuditReport(
            total_claims=total,
            supported_claims=supported_count,
            unsupported_claims=unsupported_count,
            faithfulness_score=faithfulness,
            claims=verifications,
            unsupported_details=unsupported_details,
            audit_status="verified",
            latency_ms=round(latency_ms, 2),
        )


def make_grounding_verifier(
    verifier_spec: str = "mock",
    llm_client: LLMClient | None = None,
) -> GroundingVerifier:
    """Factory creating appropriate GroundingVerifier instance."""
    if verifier_spec == "mock":
        return MockGroundingVerifier()

    if verifier_spec == "llm":
        if llm_client is None:
            raise ValueError("llm_client is required when verifier_spec='llm'")
        return LLMGroundingVerifier(llm_client=llm_client)

    raise ValueError(f"Unsupported verifier spec: '{verifier_spec}'. Expected 'mock' or 'llm'.")

