from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .models import CyberState
from .tasks import TaskConfig


@dataclass
class TaskGrade:
    score: float
    breakdown: Dict[str, float]


class TaskGrader:
    """Deterministic graders for easy/medium/hard scenarios."""

    def score(self, task: TaskConfig, final_state: CyberState, trajectory_len: int) -> TaskGrade:
        total_hosts = len(task.hosts)
        compromised = final_state.compromised_count
        isolated = final_state.isolated_count
        incidents = len(final_state.incidents)
        detected = final_state.detected_attacks

        prevention = max(0.0, 1.0 - (compromised / max(1, total_hosts)))
        detection = min(1.0, detected / max(1, incidents)) if incidents else 1.0
        response_time = max(0.0, 1.0 - (trajectory_len / max(1, task.max_steps)))
        collateral = max(0.0, 1.0 - (isolated / max(1, total_hosts)))

        # Weighted composition tuned per task complexity.
        if task.task_id == "easy":
            weights = {"prevention": 0.45, "detection": 0.25, "response": 0.2, "collateral": 0.1}
        elif task.task_id == "medium":
            weights = {"prevention": 0.4, "detection": 0.25, "response": 0.15, "collateral": 0.2}
        else:
            weights = {"prevention": 0.35, "detection": 0.25, "response": 0.15, "collateral": 0.25}

        breakdown = {
            "prevention": prevention,
            "detection": detection,
            "response_time": response_time,
            "collateral_damage_control": collateral,
        }
        score = (
            weights["prevention"] * prevention
            + weights["detection"] * detection
            + weights["response"] * response_time
            + weights["collateral"] * collateral
        )
        return TaskGrade(score=float(max(0.0, min(1.0, score))), breakdown=breakdown)


def aggregate_grades(grades: List[TaskGrade]) -> float:
    if not grades:
        return 0.0
    return float(sum(g.score for g in grades) / len(grades))
