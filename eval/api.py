import asyncio
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
from loguru import logger
import json
import os

router = APIRouter(prefix="/eval", tags=["Evaluation"])

# Track running eval state
_eval_running = False
_eval_results = None


class EvalRequest(BaseModel):
    output_file: str = "eval_results.json"
    subset: Optional[str] = None  # topic filter: "rag", "llm_foundation", etc.


@router.post("/run")
async def run_evaluation(request: EvalRequest, background_tasks: BackgroundTasks):
    """
    Run the full evaluation dataset through ARIA.
    Executes in background — poll /eval/status for progress.
    """
    global _eval_running
    if _eval_running:
        raise HTTPException(status_code=409, detail="Eval is already running — poll /eval/status")

    background_tasks.add_task(
        _run_eval_background,
        output_file=request.output_file,
        subset=request.subset,
    )

    return {
        "message": "Evaluation started in background",
        "output_file": request.output_file,
        "total_questions": _get_question_count(request.subset),
        "poll": "/eval/status",
    }


@router.get("/status")
async def eval_status():
    """Check if eval is running and get latest results."""
    global _eval_running, _eval_results
    return {
        "running": _eval_running,
        "has_results": _eval_results is not None,
        "summary": _eval_results.get("summary") if _eval_results else None,
    }


@router.get("/results")
async def get_results():
    """Get the full results from the last eval run."""
    global _eval_results

    # Try memory first
    if _eval_results:
        return _eval_results

    # Try loading from file
    if os.path.exists("eval_results.json"):
        with open("eval_results.json") as f:
            return json.load(f)

    raise HTTPException(status_code=404, detail="No eval results found. Run /eval/run first.")


@router.get("/results/summary")
async def get_summary():
    """Get just the summary metrics from the last eval run."""
    global _eval_results

    data = _eval_results
    if not data and os.path.exists("eval_results.json"):
        with open("eval_results.json") as f:
            data = json.load(f)

    if not data:
        raise HTTPException(status_code=404, detail="No eval results found. Run /eval/run first.")

    return {
        "summary": data["summary"],
        "by_topic": data["by_topic"],
        "by_difficulty": data["by_difficulty"],
        "run_timestamp": data["run_timestamp"],
        "elapsed_seconds": data["elapsed_seconds"],
    }


@router.get("/dataset")
async def get_dataset():
    """Preview the eval dataset without running it."""
    from eval.dataset import EVAL_DATASET
    return {
        "total": len(EVAL_DATASET),
        "by_topic": _count_by_key(EVAL_DATASET, "topic"),
        "by_difficulty": _count_by_key(EVAL_DATASET, "difficulty"),
        "questions": [
            {
                "id": q["id"],
                "question": q["question"],
                "topic": q["topic"],
                "difficulty": q["difficulty"],
                "source_paper": q["source_paper"],
            }
            for q in EVAL_DATASET
        ],
    }


# ── Background task ───────────────────────────────────────────────────────────

async def _run_eval_background(output_file: str, subset: Optional[str]):
    global _eval_running, _eval_results
    _eval_running = True
    try:
        from eval.runner import run_eval
        from eval.dataset import EVAL_DATASET

        # Filter by subset if specified
        if subset:
            filtered = [q for q in EVAL_DATASET if q["topic"] == subset]
            if not filtered:
                logger.warning(f"No questions found for topic: {subset}")
                return

            # Temporarily patch the dataset
            import eval.runner as runner_module
            original = runner_module.EVAL_DATASET
            runner_module.EVAL_DATASET = filtered
            results = await run_eval(output_file)
            runner_module.EVAL_DATASET = original
        else:
            results = await run_eval(output_file)

        _eval_results = results
        logger.info("Eval complete")

    except Exception as e:
        logger.error(f"Eval failed: {e}")
    finally:
        _eval_running = False


def _get_question_count(subset: Optional[str]) -> int:
    from eval.dataset import EVAL_DATASET
    if subset:
        return sum(1 for q in EVAL_DATASET if q["topic"] == subset)
    return len(EVAL_DATASET)


def _count_by_key(items: list, key: str) -> dict:
    counts = {}
    for item in items:
        v = item.get(key, "unknown")
        counts[v] = counts.get(v, 0) + 1
    return counts