from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class EpisodeMetrics:
    detection_rate: float
    response_time: float
    false_positive_actions: float
    service_availability: float
    cumulative_reward: float


def summarize_metrics(info_metrics: Dict[str, float], steps: int, max_steps: int) -> EpisodeMetrics:
    return EpisodeMetrics(
        detection_rate=float(info_metrics.get("detection_rate", 0.0)),
        response_time=float(steps / max(1, max_steps)),
        false_positive_actions=float(max(0.0, 1.0 - info_metrics.get("detection_rate", 0.0))),
        service_availability=float(info_metrics.get("service_availability", 0.0)),
        cumulative_reward=float(info_metrics.get("cumulative_reward", 0.0)),
    )
