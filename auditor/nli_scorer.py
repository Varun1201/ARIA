import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger


NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


class NLIScorer:
    """
    Local NLI scorer using DeBERTa-v3.
    Labels: CONTRADICTION=0, ENTAILMENT=1, NEUTRAL=2
    For faithfulness: premise=chunk, hypothesis=answer sentence
    For relevance:    premise=chunk, hypothesis=query
    """

    def __init__(self):
        logger.info(f"Loading NLI model: {NLI_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"NLI model ready on {self.device}")

    def _predict(self, premises: list[str], hypotheses: list[str]) -> list[dict]:
        """Run NLI on batched premise-hypothesis pairs."""
        inputs = self.tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu()

        results = []
        for p in probs:
            results.append({
                "contradiction": p[0].item(),
                "entailment": p[1].item(),
                "neutral": p[2].item(),
            })
        return results

    def score_faithfulness(self, answer: str, chunks: list[str]) -> float:
        """
        Faithfulness: does the answer follow from the retrieved chunks?
        Split answer into sentences, check each against all chunks,
        take max entailment per sentence, average across sentences.
        """
        sentences = self._split_sentences(answer)
        if not sentences or not chunks:
            return 0.0

        sentence_scores = []
        for sentence in sentences:
            if len(sentence.split()) < 4:
                continue
            premises = chunks
            hypotheses = [sentence] * len(chunks)
            results = self._predict(premises, hypotheses)
            max_entailment = max(r["entailment"] for r in results)
            sentence_scores.append(max_entailment)

        if not sentence_scores:
            return 0.0
        return sum(sentence_scores) / len(sentence_scores)

    def score_relevance(self, query: str, chunks: list[str]) -> float:
        """
        Relevance: are retrieved chunks relevant to the query?
        Check entailment of query against each chunk, return average.
        """
        if not chunks:
            return 0.0

        premises = chunks
        hypotheses = [query] * len(chunks)
        results = self._predict(premises, hypotheses)
        scores = [r["entailment"] + r["neutral"] * 0.3 for r in results]
        return sum(scores) / len(scores)

    def score_contradiction(self, answer: str, chunks: list[str]) -> float:
        """
        Contradiction score: does the answer contradict the source chunks?
        High contradiction = likely hallucination.
        """
        sentences = self._split_sentences(answer)
        if not sentences or not chunks:
            return 0.0

        contradiction_scores = []
        for sentence in sentences:
            if len(sentence.split()) < 4:
                continue
            premises = chunks
            hypotheses = [sentence] * len(chunks)
            results = self._predict(premises, hypotheses)
            max_contradiction = max(r["contradiction"] for r in results)
            contradiction_scores.append(max_contradiction)

        if not contradiction_scores:
            return 0.0
        return sum(contradiction_scores) / len(contradiction_scores)

    def _split_sentences(self, text: str) -> list[str]:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]


# Singleton
nli_scorer = NLIScorer()