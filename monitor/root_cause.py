import asyncio
import json
from loguru import logger
from groq import Groq
from config import settings

groq_client = Groq(api_key=settings.groq_api_key)

ROOT_CAUSE_PROMPT = """You are an expert MLOps engineer analyzing a RAG (Retrieval Augmented Generation) pipeline failure.

PIPELINE METRICS:
{metrics}

ANOMALY DETECTED:
- Type: {anomaly_type}
- Severity: {severity}
- Description: {description}

RECENT QUERY LOGS (last 5):
{query_logs}

DOCUMENT STATS:
{doc_stats}

Based on this data, analyze the root cause and recommend a remediation action.

Respond ONLY with this exact JSON format:
{{
    "root_cause": "one clear sentence explaining why this happened",
    "contributing_factors": ["factor 1", "factor 2"],
    "recommended_action": "re_chunk" | "purge_low_quality" | "rebuild_index" | "alert_only",
    "action_reasoning": "why this action will fix the problem",
    "risk_level": "low" | "medium" | "high",
    "estimated_impact": "what will improve after remediation"
}}

Choose recommended_action from ONLY these options:
- re_chunk: re-process specific documents with better chunking
- purge_low_quality: remove chunks with low embedding quality scores
- rebuild_index: completely rebuild the vector index
- alert_only: no automated fix, just flag for human review"""


class RootCauseAnalyzer:
    """Uses Groq LLM to diagnose why a pipeline anomaly occurred."""

    async def analyze(
        self,
        anomaly_type: str,
        severity: str,
        description: str,
        metrics: dict,
        query_logs: list[dict],
        doc_stats: dict,
    ) -> dict:
        """
        Returns diagnosis dict with root_cause, recommended_action, risk_level etc.
        """
        # Format context for the prompt
        metrics_str = json.dumps(metrics, indent=2)
        query_logs_str = json.dumps(query_logs[-5:], indent=2) if query_logs else "No recent queries"
        doc_stats_str = json.dumps(doc_stats, indent=2)

        prompt = ROOT_CAUSE_PROMPT.format(
            metrics=metrics_str,
            anomaly_type=anomaly_type,
            severity=severity,
            description=description,
            query_logs=query_logs_str,
            doc_stats=doc_stats_str,
        )

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: groq_client.chat.completions.create(
                    model=settings.groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=512,
                )
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            diagnosis = json.loads(raw)
            logger.info(f"Root cause: {diagnosis['root_cause']}")
            logger.info(f"Recommended action: {diagnosis['recommended_action']} (risk: {diagnosis['risk_level']})")
            return diagnosis

        except Exception as e:
            logger.error(f"Root cause analysis failed: {e}")
            return {
                "root_cause": f"Analysis failed: {str(e)}",
                "contributing_factors": [],
                "recommended_action": "alert_only",
                "action_reasoning": "Defaulting to alert_only due to analysis failure",
                "risk_level": "low",
                "estimated_impact": "Unknown",
            }


# Singleton
root_cause_analyzer = RootCauseAnalyzer()