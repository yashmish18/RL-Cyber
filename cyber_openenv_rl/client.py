from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CyberAction, CyberObservation


class CyberEnv(EnvClient[CyberAction, CyberObservation, State]):
    """OpenEnv client for the cyber defense environment."""

    def _step_payload(self, action: CyberAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[CyberObservation]:
        obs_data = payload.get("observation", {})
        observation = CyberObservation(
            task_id=obs_data.get("task_id", "unknown"),
            host_compromise=obs_data.get("host_compromise", {}),
            host_isolation=obs_data.get("host_isolation", {}),
            service_status=obs_data.get("service_status", {}),
            ids_alerts=obs_data.get("ids_alerts", []),
            traffic_anomaly_score=obs_data.get("traffic_anomaly_score", 0.0),
            active_incidents=obs_data.get("active_incidents", []),
            step_budget_remaining=obs_data.get("step_budget_remaining", 0),
            available_defender_actions=obs_data.get("available_defender_actions", []),
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(**payload)
