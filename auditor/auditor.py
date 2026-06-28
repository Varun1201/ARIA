from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from auditor.nli_scorer import nli_scorer
from auditor.groq_judge import groq_judge
from auditor.citation_tracer import citation_tracer
from config import settings


@dataclass
class AuditResult:
    query_id: str
    faithfulness_score: float
    relevance_score: float
    contradiction_score: float
    hallucination_score: float        # final combined score (higher = more likely hallucination)
    is_hallucination_flagged: bool
    groq_verdict: Optional[str]       # "faithful" | "hallucination" | "unknown" | None
    groq_confidence: Optional[float]
    groq_reason: Optional[str]
    problematic_claims: list[str]
    citations: list[dict]


class RAGAuditor:
    """
    Orchestrates the full audit pipeline for every RAG response.

    Flow:
    1. NLI scorer → faithfulness, relevance, contradiction scores
    2. If hallucination_score > threshold → escalate to Groq judge
    3. Citation tracer → map sentences to source chunks
    4. Return AuditResult
    """

    async def audit(
        self,
        query_id: str,
        query: str,
        answer: str,
        chunks: list[str],
        chunk_ids: list[str],
    ) -> AuditResult:
        logger.info(f"Auditing query {query_id}")

        # ── Step 1: NLI Scoring ───────────────────────────────────────────────
        faithfulness = nli_scorer.score_faithfulness(answer, chunks)
        relevance = nli_scorer.score_relevance(query, chunks)
        contradiction = nli_scorer.score_contradiction(answer, chunks)

        # Combined hallucination score:
        # Low faithfulness + high contradiction = likely hallucination
        hallucination_score = (1 - faithfulness) * 0.6 + contradiction * 0.4

        logger.info(
            f"NLI scores — faithfulness: {faithfulness:.3f}, "
            f"relevance: {relevance:.3f}, "
            f"contradiction: {contradiction:.3f}, "
            f"hallucination_score: {hallucination_score:.3f}"
        )

        # ── Step 2: Groq Escalation ───────────────────────────────────────────
        groq_verdict = None
        groq_confidence = None
        groq_reason = None
        problematic_claims = []
        is_flagged = hallucination_score > settings.hallucination_flag_threshold

        if is_flagged:
            logger.warning(f"Hallucination suspected for {query_id}, escalating to Groq judge")
            verdict = await groq_judge.judge(query=query, answer=answer, chunks=chunks)
            groq_verdict = verdict.get("verdict")
            groq_confidence = verdict.get("confidence")
            groq_reason = verdict.get("reason")
            problematic_claims = verdict.get("problematic_claims", [])

            # Override flag based on Groq's verdict
            if groq_verdict == "faithful":
                is_flagged = False
                logger.info(f"Groq judge cleared {query_id} as faithful")
            else:
                logger.warning(f"Groq judge confirmed hallucination in {query_id}: {groq_reason}")

        # ── Step 3: Citation Tracing ──────────────────────────────────────────
        citations = citation_tracer.trace(answer, chunks, chunk_ids)

        return AuditResult(
            query_id=query_id,
            faithfulness_score=round(faithfulness, 4),
            relevance_score=round(relevance, 4),
            contradiction_score=round(contradiction, 4),
            hallucination_score=round(hallucination_score, 4),
            is_hallucination_flagged=is_flagged,
            groq_verdict=groq_verdict,
            groq_confidence=groq_confidence,
            groq_reason=groq_reason,
            problematic_claims=problematic_claims,
            citations=citations,
        )


# Singleton
auditor = RAGAuditor()