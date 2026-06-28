from dataclasses import dataclass
from typing import Optional
from loguru import logger
from config import settings


@dataclass
class AnomalySignal:
    anomaly_type: str        # score_drop | hallucination_spike | ingestion_failure | volume_anomaly
    severity: str            # low | medium | high | critical
    description: str
    metric_value: float
    threshold_value: float
    extra: Optional[dict] = None


class AnomalyDetector:
    """
    Stateless anomaly detector — takes a window of metrics and returns signals.
    Designed to be called periodically by the monitor worker.
    """

    def detect_score_drop(self, scores: list[float], metric_name: str, threshold: float) -> Optional[AnomalySignal]:
        """Detect if rolling average of a score has dropped below threshold."""
        if len(scores) < 5:
            return None  # Not enough data

        recent = scores[-10:]  # Last 10 queries
        avg = sum(recent) / len(recent)

        if avg < threshold:
            drop_pct = (threshold - avg) / threshold * 100
            severity = "critical" if drop_pct > 30 else "high" if drop_pct > 15 else "medium"
            return AnomalySignal(
                anomaly_type="score_drop",
                severity=severity,
                description=f"{metric_name} rolling average dropped to {avg:.3f} (threshold: {threshold})",
                metric_value=avg,
                threshold_value=threshold,
                extra={"metric": metric_name, "drop_pct": round(drop_pct, 2), "window_size": len(recent)},
            )
        return None

    def detect_hallucination_spike(self, hallucination_flags: list[bool]) -> Optional[AnomalySignal]:
        """Detect sudden spike in hallucination rate."""
        if len(hallucination_flags) < 10:
            return None

        recent = hallucination_flags[-20:]
        rate = sum(recent) / len(recent)
        threshold = 0.25  # Alert if >25% of recent queries are flagged

        if rate > threshold:
            severity = "critical" if rate > 0.5 else "high" if rate > 0.35 else "medium"
            return AnomalySignal(
                anomaly_type="hallucination_spike",
                severity=severity,
                description=f"Hallucination flag rate spiked to {rate:.1%} in last {len(recent)} queries",
                metric_value=rate,
                threshold_value=threshold,
                extra={"flagged_count": sum(recent), "window_size": len(recent)},
            )
        return None

    def detect_ingestion_failures(self, statuses: list[str]) -> Optional[AnomalySignal]:
        """Detect high rate of ingestion failures."""
        if len(statuses) < 3:
            return None

        recent = statuses[-10:]
        failure_rate = sum(1 for s in recent if s == "failed") / len(recent)
        threshold = 0.3  # Alert if >30% of recent ingestions failed

        if failure_rate > threshold:
            severity = "critical" if failure_rate > 0.6 else "high"
            return AnomalySignal(
                anomaly_type="ingestion_failure",
                severity=severity,
                description=f"Ingestion failure rate is {failure_rate:.1%} in last {len(recent)} documents",
                metric_value=failure_rate,
                threshold_value=threshold,
                extra={"failed_count": sum(1 for s in recent if s == "failed"), "window_size": len(recent)},
            )
        return None

    def detect_latency_spike(self, latencies_ms: list[int]) -> Optional[AnomalySignal]:
        """Detect abnormal query latency increase."""
        if len(latencies_ms) < 10:
            return None

        baseline = sorted(latencies_ms)[len(latencies_ms) // 2]  # median
        recent_avg = sum(latencies_ms[-5:]) / 5
        threshold_multiplier = 3.0  # Alert if 3x the median

        if baseline > 0 and recent_avg > baseline * threshold_multiplier:
            return AnomalySignal(
                anomaly_type="latency_spike",
                severity="medium",
                description=f"Query latency spiked to {recent_avg:.0f}ms (baseline: {baseline:.0f}ms)",
                metric_value=recent_avg,
                threshold_value=baseline * threshold_multiplier,
                extra={"baseline_ms": baseline, "recent_avg_ms": round(recent_avg)},
            )
        return None

    def run_all(
        self,
        faithfulness_scores: list[float],
        relevance_scores: list[float],
        hallucination_flags: list[bool],
        ingestion_statuses: list[str],
        latencies_ms: list[int],
    ) -> list[AnomalySignal]:
        """Run all detectors and return list of triggered anomalies."""
        signals = []

        checks = [
            self.detect_score_drop(faithfulness_scores, "faithfulness", settings.faithfulness_threshold),
            self.detect_score_drop(relevance_scores, "relevance", settings.relevance_threshold),
            self.detect_hallucination_spike(hallucination_flags),
            self.detect_ingestion_failures(ingestion_statuses),
            self.detect_latency_spike(latencies_ms),
        ]

        for signal in checks:
            if signal is not None:
                logger.warning(f"Anomaly detected: [{signal.severity}] {signal.description}")
                signals.append(signal)

        if not signals:
            logger.debug("Anomaly check passed — all metrics normal")

        return signals


# Singleton
anomaly_detector = AnomalyDetector()