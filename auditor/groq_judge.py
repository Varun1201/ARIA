import asyncio
from loguru import logger
from groq import Groq
from config import settings

groq_client = Groq(api_key=settings.groq_api_key)

JUDGE_PROMPT = """You are a strict factual auditor. Your job is to determine if an AI-generated answer is faithful to the provided source chunks.

SOURCE CHUNKS:
{chunks}

AI ANSWER:
{answer}

QUESTION:
{query}

Carefully analyze whether the answer:
1. Makes claims NOT supported by the source chunks
2. Contradicts information in the source chunks
3. Fabricates specific facts, numbers, names, or dates not in the sources

Respond in this exact JSON format:
{{
    "verdict": "faithful" or "hallucination",
    "confidence": 0.0 to 1.0,
    "reason": "one sentence explanation",
    "problematic_claims": ["list of specific claims that are unsupported or false, empty if faithful"]
}}

Respond ONLY with the JSON object, no other text."""


class GroqJudge:
    """
    LLM-as-judge for hallucination verification.
    Only called when NLI flags a potential hallucination.
    """

    async def judge(
        self,
        query: str,
        answer: str,
        chunks: list[str],
    ) -> dict:
        """
        Returns a verdict dict with keys:
        verdict, confidence, reason, problematic_claims
        """
        chunks_text = "\n\n".join(
            f"[Chunk {i+1}]: {chunk}" for i, chunk in enumerate(chunks)
        )
        prompt = JUDGE_PROMPT.format(
            chunks=chunks_text,
            answer=answer,
            query=query,
        )

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: groq_client.chat.completions.create(
                    model=settings.groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,  # deterministic for judging
                    max_tokens=512,
                )
            )
            raw = response.choices[0].message.content.strip()

            import json
            # Strip markdown fences if present
            raw = raw.replace("```json", "").replace("```", "").strip()
            verdict = json.loads(raw)
            logger.info(f"Groq judge verdict: {verdict['verdict']} (confidence: {verdict['confidence']})")
            return verdict

        except Exception as e:
            logger.error(f"Groq judge failed: {e}")
            return {
                "verdict": "unknown",
                "confidence": 0.0,
                "reason": f"Judge failed: {str(e)}",
                "problematic_claims": [],
            }


# Singleton
groq_judge = GroqJudge()