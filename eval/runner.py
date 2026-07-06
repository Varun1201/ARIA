"""
ARIA Evaluation Runner
Runs the ground truth eval dataset through ARIA and measures
whether faithfulness scores correlate with actual answer correctness.
"""

import asyncio
import json
import httpx
from datetime import datetime
from loguru import logger
from eval.dataset import EVAL_DATASET


ARIA_BASE = "http://localhost:8000"


async def run_single_eval(client: httpx.AsyncClient, item: dict) -> dict:
    """Run a single eval item through ARIA and return results."""
    try:
        response = await client.post(
            f"{ARIA_BASE}/query/",
            json={"query": item["question"], "top_k": 4},
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()

        aria_answer = result.get("answer", "")
        audit = result.get("audit", {})

        # Simple correctness check — does the answer contain key concepts
        # from the expected answer? (keyword overlap as proxy for correctness)
        expected_keywords = set(item["expected_answer"].lower().split())
        answer_keywords = set(aria_answer.lower().split())
        # Filter to meaningful words (>4 chars)
        expected_meaningful = {w for w in expected_keywords if len(w) > 4}
        answer_meaningful = {w for w in answer_keywords if len(w) > 4}
        if expected_meaningful:
            keyword_overlap = len(expected_meaningful & answer_meaningful) / len(expected_meaningful)
        else:
            keyword_overlap = 0.0

        # Binary correctness: did ARIA refuse to answer?
        refused = "cannot answer" in aria_answer.lower() or "not found" in aria_answer.lower()

        return {
            "id": item["id"],
            "question": item["question"],
            "expected_answer": item["expected_answer"][:200] + "...",
            "aria_answer": aria_answer[:300] + "..." if len(aria_answer) > 300 else aria_answer,
            "source_paper": item["source_paper"],
            "topic": item["topic"],
            "difficulty": item["difficulty"],
            "faithfulness_score": audit.get("faithfulness_score", 0),
            "relevance_score": audit.get("relevance_score", 0),
            "hallucination_score": audit.get("hallucination_score", 0),
            "is_hallucination_flagged": audit.get("is_hallucination_flagged", False),
            "keyword_overlap": round(keyword_overlap, 3),
            "refused": refused,
            "latency_ms": result.get("latency_ms", 0),
            "status": "ok",
        }

    except Exception as e:
        logger.error(f"Eval failed for {item['id']}: {e}")
        return {
            "id": item["id"],
            "question": item["question"],
            "topic": item["topic"],
            "difficulty": item["difficulty"],
            "status": "error",
            "error": str(e),
            "faithfulness_score": 0,
            "keyword_overlap": 0,
            "refused": False,
        }


async def run_eval(output_file: str = "eval_results.json") -> dict:
    """Run the full eval dataset through ARIA."""
    logger.info(f"Starting eval run — {len(EVAL_DATASET)} questions")
    start = datetime.utcnow()

    results = []
    async with httpx.AsyncClient() as client:
        # Check ARIA is running
        try:
            health = await client.get(f"{ARIA_BASE}/health", timeout=5)
            health.raise_for_status()
            logger.info("ARIA is running — starting eval")
        except Exception:
            logger.error("ARIA is not running at localhost:8000 — start it first")
            return {}

        for i, item in enumerate(EVAL_DATASET):
            logger.info(f"[{i+1}/{len(EVAL_DATASET)}] {item['id']}: {item['question'][:60]}...")
            result = await run_single_eval(client, item)
            results.append(result)
            await asyncio.sleep(1)  # Rate limiting

    # ── Compute aggregate metrics ─────────────────────────────────────────────
    ok_results = [r for r in results if r["status"] == "ok"]
    refused = [r for r in ok_results if r["refused"]]
    flagged = [r for r in ok_results if r["is_hallucination_flagged"]]

    avg_faithfulness = sum(r["faithfulness_score"] for r in ok_results) / max(len(ok_results), 1)
    avg_relevance = sum(r["relevance_score"] for r in ok_results) / max(len(ok_results), 1)
    avg_hallucination = sum(r["hallucination_score"] for r in ok_results) / max(len(ok_results), 1)
    avg_keyword_overlap = sum(r["keyword_overlap"] for r in ok_results) / max(len(ok_results), 1)

    # Score correlation: do higher faithfulness scores → higher keyword overlap?
    if len(ok_results) > 1:
        faith_scores = [r["faithfulness_score"] for r in ok_results]
        kw_scores = [r["keyword_overlap"] for r in ok_results]
        correlation = _pearson_correlation(faith_scores, kw_scores)
    else:
        correlation = 0.0

    # Per-topic breakdown
    topics = {}
    for r in ok_results:
        t = r["topic"]
        if t not in topics:
            topics[t] = {"count": 0, "avg_faithfulness": 0, "avg_keyword_overlap": 0, "refused": 0}
        topics[t]["count"] += 1
        topics[t]["avg_faithfulness"] += r["faithfulness_score"]
        topics[t]["avg_keyword_overlap"] += r["keyword_overlap"]
        if r["refused"]:
            topics[t]["refused"] += 1
    for t in topics:
        n = topics[t]["count"]
        topics[t]["avg_faithfulness"] = round(topics[t]["avg_faithfulness"] / n, 3)
        topics[t]["avg_keyword_overlap"] = round(topics[t]["avg_keyword_overlap"] / n, 3)

    # Per-difficulty breakdown
    difficulties = {}
    for r in ok_results:
        d = r["difficulty"]
        if d not in difficulties:
            difficulties[d] = {"count": 0, "avg_faithfulness": 0, "refused": 0}
        difficulties[d]["count"] += 1
        difficulties[d]["avg_faithfulness"] += r["faithfulness_score"]
        if r["refused"]:
            difficulties[d]["refused"] += 1
    for d in difficulties:
        n = difficulties[d]["count"]
        difficulties[d]["avg_faithfulness"] = round(difficulties[d]["avg_faithfulness"] / n, 3)

    elapsed = (datetime.utcnow() - start).seconds

    report = {
        "run_timestamp": start.isoformat(),
        "elapsed_seconds": elapsed,
        "summary": {
            "total_questions": len(EVAL_DATASET),
            "successful": len(ok_results),
            "errors": len(results) - len(ok_results),
            "refused_to_answer": len(refused),
            "hallucination_flagged": len(flagged),
            "avg_faithfulness_score": round(avg_faithfulness, 3),
            "avg_relevance_score": round(avg_relevance, 3),
            "avg_hallucination_score": round(avg_hallucination, 3),
            "avg_keyword_overlap": round(avg_keyword_overlap, 3),
            "faithfulness_keyword_correlation": round(correlation, 3),
            "interpretation": _interpret_correlation(correlation),
        },
        "by_topic": topics,
        "by_difficulty": difficulties,
        "results": results,
    }

    # Save to file
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Eval complete — results saved to {output_file}")
    _print_summary(report)
    return report


def _pearson_correlation(x: list, y: list) -> float:
    """Compute Pearson correlation coefficient between two lists."""
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    den_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def _interpret_correlation(r: float) -> str:
    if r >= 0.7:
        return "Strong positive correlation — ARIA faithfulness scores reliably predict answer quality"
    elif r >= 0.4:
        return "Moderate positive correlation — ARIA faithfulness scores are a useful quality signal"
    elif r >= 0.2:
        return "Weak positive correlation — faithfulness scores have limited predictive value"
    else:
        return "No meaningful correlation — faithfulness scores need calibration"


def _print_summary(report: dict):
    s = report["summary"]
    print("\n" + "="*60)
    print("ARIA EVALUATION REPORT")
    print("="*60)
    print(f"Questions:          {s['total_questions']}")
    print(f"Successful:         {s['successful']}")
    print(f"Refused to answer:  {s['refused_to_answer']}")
    print(f"Hallucination flag: {s['hallucination_flagged']}")
    print(f"Avg faithfulness:   {s['avg_faithfulness_score']:.3f}")
    print(f"Avg relevance:      {s['avg_relevance_score']:.3f}")
    print(f"Avg keyword overlap:{s['avg_keyword_overlap']:.3f}")
    print(f"Score correlation:  {s['faithfulness_keyword_correlation']:.3f}")
    print(f"Interpretation:     {s['interpretation']}")
    print("\nBy topic:")
    for topic, stats in report["by_topic"].items():
        print(f"  {topic:20s} — faithfulness: {stats['avg_faithfulness']:.3f}, overlap: {stats['avg_keyword_overlap']:.3f}, refused: {stats['refused']}/{stats['count']}")
    print("\nBy difficulty:")
    for diff, stats in report["by_difficulty"].items():
        print(f"  {diff:10s} — faithfulness: {stats['avg_faithfulness']:.3f}, refused: {stats['refused']}/{stats['count']}")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(run_eval())