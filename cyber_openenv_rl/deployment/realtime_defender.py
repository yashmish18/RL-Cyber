from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from stable_baselines3 import DQN, PPO

from ..core.policies import scripted_defender_policy
from ..core.simulator import DEFENDER_ACTIONS
from ..models import CyberAction, CyberObservation, RewardSignal
from ..rl.gym_env import DefenderGymEnv
from .guardrails import apply_guardrails
from .telemetry_adapter import TelemetryAdapter
from ..connectors.base import BaseConnector


@dataclass
class InferenceResult:
    proposed_action: Dict[str, Any]
    final_action: Dict[str, Any]
    confidence: float
    policy_scores: List[float]
    override_reason: str
    guardrail_reason: str


class RealtimeDefender:
    """Runs a trained defender policy against real-time telemetry snapshots."""

    def __init__(
        self,
        model_path: Path,
        algorithm: str,
        task_id: str,
        human_approval_required: bool = True,
        confidence_threshold: float = 0.52,
    ) -> None:
        self.task_id = task_id
        self.algorithm = algorithm
        self.human_approval_required = human_approval_required
        self.confidence_threshold = confidence_threshold
        self.env = DefenderGymEnv(task_id=task_id, seed=777)
        if algorithm == "ppo":
            self.model = PPO.load(str(model_path))
        elif algorithm == "dqn":
            self.model = DQN.load(str(model_path))
        else:
            raise ValueError("algorithm must be ppo or dqn")

    def _build_observation(self, telemetry: Dict[str, Any]) -> CyberObservation:
        snap = TelemetryAdapter.from_json(telemetry, known_hosts=self.env.hosts)
        return CyberObservation(
            task_id=self.task_id,
            host_compromise=snap.host_compromise,
            host_isolation=snap.host_isolation,
            service_status=snap.service_status,
            ids_alerts=snap.ids_alerts,
            traffic_anomaly_score=snap.traffic_anomaly_score,
            active_incidents=snap.active_incidents,
            reward_signal=RewardSignal(),
            step_budget_remaining=999,
            available_defender_actions=DEFENDER_ACTIONS,
            reward=0.0,
            done=False,
            metadata={"source": "realtime_telemetry"},
        )

    def _vectorize(self, obs: CyberObservation) -> np.ndarray:
        return self.env._vectorize(obs)  # reuse training feature encoder

    def _policy_scores(self, vec: np.ndarray) -> List[float]:
        obs_t = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        if self.algorithm == "dqn":
            with torch.no_grad():
                q_values = self.model.q_net(obs_t).squeeze(0)
                probs = torch.softmax(q_values, dim=0)
            return probs.detach().cpu().tolist()

        # PPO policy distribution
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_t)
            probs = dist.distribution.probs.squeeze(0)
        return probs.detach().cpu().tolist()

    def _pick_action(self, vec: np.ndarray) -> tuple[CyberAction, float, List[float]]:
        scores = self._policy_scores(vec)
        idx = int(np.argmax(scores))
        confidence = float(max(scores)) if scores else 0.0
        proposed = self.env._decode_action(idx)
        return proposed, confidence, scores

    def _hybrid_override(
        self, obs: CyberObservation, proposed: CyberAction, confidence: float
    ) -> tuple[CyberAction, str]:
        if confidence < self.confidence_threshold:
            fallback = scripted_defender_policy(obs)
            return fallback, "low_confidence_fallback"

        high_risk = obs.traffic_anomaly_score >= 0.8 or len(obs.active_incidents) > 0
        if high_risk and proposed.action_type == "ignore":
            for host, compromised in obs.host_compromise.items():
                if compromised:
                    return CyberAction(
                        actor="defender",
                        action_type="scan_host",
                        target_host=host,
                        source_ip="10.0.0.45",
                    ), "high_risk_override"
            first_host = next(iter(obs.host_compromise.keys()))
            return CyberAction(
                actor="defender",
                action_type="scan_host",
                target_host=first_host,
                source_ip="10.0.0.45",
            ), "high_risk_override"

        return proposed, "none"

    def infer(self, telemetry: Dict[str, Any]) -> InferenceResult:
        obs = self._build_observation(telemetry)
        vec = self._vectorize(obs)
        proposed, confidence, scores = self._pick_action(vec)
        adjusted, override_reason = self._hybrid_override(obs, proposed, confidence)

        decision = apply_guardrails(
            proposed=adjusted,
            known_hosts=self.env.hosts,
            human_approval_required=self.human_approval_required,
        )
        return InferenceResult(
            proposed_action=proposed.model_dump(),
            final_action=decision.safe_action.model_dump(),
            confidence=confidence,
            policy_scores=scores,
            override_reason=override_reason,
            guardrail_reason=decision.reason,
        )

    async def infer_live(self, connector: BaseConnector) -> InferenceResult:
        """Fetch live telemetry and run inference with potential human-approval pause."""
        obs = await connector.get_observation()
        vec = self._vectorize(obs)
        proposed, confidence, scores = self._pick_action(vec)
        adjusted, override_reason = self._hybrid_override(obs, proposed, confidence)

        final_action = adjusted
        guardrail_reason = "allowed"

        # If high risk and human approval is required, pause and wait for the dashboard
        high_risk_actions = {"isolate_node", "block_ip", "patch_service"}
        if self.human_approval_required and adjusted.action_type in high_risk_actions:
            import uuid
            import asyncio
            import httpx
            
            action_id = str(uuid.uuid4())
            proposal = {
                "action_id": action_id,
                "proposed_action": adjusted.model_dump(),
                "confidence": confidence,
                "observation": obs.model_dump(exclude={"metadata"}),
                "status": "pending"
            }
            
            # Post to approval API (assuming local server for research purposes)
            try:
                async with httpx.AsyncClient() as client:
                    await client.post("http://localhost:8000/api/approval/propose", json=proposal)
                    
                    # Poll for decision
                    max_retries = 60  # 1 minute timeout
                    for _ in range(max_retries):
                        await asyncio.sleep(1)
                        resp = await client.get(f"http://localhost:8000/api/approval/status/{action_id}")
                        if resp.status_code == 200:
                            data = resp.json()
                            if data.get("status") == "decided":
                                decision = data.get("decision")
                                if decision == "approve":
                                    guardrail_reason = "human_approved"
                                elif decision == "reject":
                                    final_action = CyberAction(actor="defender", action_type="ignore", target_host=adjusted.target_host)
                                    guardrail_reason = "human_rejected"
                                elif decision == "modify":
                                    final_action = CyberAction(**data.get("modified_action"))
                                    guardrail_reason = "human_modified"
                                break
                    else:
                        # Timeout: apply safe guardrails
                        decision = apply_guardrails(adjusted, self.env.hosts, True)
                        final_action = decision.safe_action
                        guardrail_reason = "approval_timeout; " + decision.reason
            except Exception as e:
                # API failure: apply safe guardrails
                decision = apply_guardrails(adjusted, self.env.hosts, True)
                final_action = decision.safe_action
                guardrail_reason = f"api_error ({e}); " + decision.reason

        return InferenceResult(
            proposed_action=proposed.model_dump(),
            final_action=final_action.model_dump(),
            confidence=confidence,
            policy_scores=scores,
            override_reason=override_reason,
            guardrail_reason=guardrail_reason,
        )


def run_realtime_file(
    model_path: Path,
    algorithm: str,
    task_id: str,
    telemetry_file: Path,
    output_file: Path,
    human_approval_required: bool = True,
    confidence_threshold: float = 0.52,
) -> Dict[str, Any]:
    service = RealtimeDefender(
        model_path=model_path,
        algorithm=algorithm,
        task_id=task_id,
        human_approval_required=human_approval_required,
        confidence_threshold=confidence_threshold,
    )
    telemetry = json.loads(telemetry_file.read_text(encoding="utf-8"))
    result = service.infer(telemetry)
    payload = {
        "task_id": task_id,
        "algorithm": algorithm,
        "human_approval_required": human_approval_required,
        "confidence_threshold": confidence_threshold,
        "result": {
            "proposed_action": result.proposed_action,
            "final_action": result.final_action,
            "confidence": result.confidence,
            "policy_scores": result.policy_scores,
            "override_reason": result.override_reason,
            "guardrail_reason": result.guardrail_reason,
        },
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
