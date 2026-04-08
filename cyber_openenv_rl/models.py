from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, computed_field


DefenderActionName = Literal[
    "block_ip",
    "isolate_node",
    "scan_host",
    "patch_service",
    "restore_backup",
    "ignore",
]
AttackerActionName = Literal[
    "lateral_move",
    "credential_stuff",
    "malware_drop",
    "recon",
    "idle",
]


class CyberAction(Action):
    """Unified action model for attacker or defender steps."""

    actor: Literal["defender", "attacker"] = Field(
        ..., description="Actor taking the action"
    )
    action_type: DefenderActionName | AttackerActionName = Field(
        ..., description="Action operation"
    )
    target_host: str = Field(..., description="Host identifier")
    target_service: Optional[str] = Field(
        default=None, description="Target service for patching/attacks"
    )
    source_ip: Optional[str] = Field(
        default=None, description="IP used for attacker/defender decisions"
    )


class Incident(BaseModel):
    host: str
    attack_type: str
    severity: float = Field(ge=0.0, le=1.0)
    detected: bool = False


class RewardSignal(BaseModel):
    """Typed reward model for deterministic grading and trajectory analysis."""

    base: float = 0.0
    prevention: float = 0.0
    detection: float = 0.0
    containment: float = 0.0
    spread_penalty: float = 0.0
    collateral_penalty: float = 0.0
    loop_penalty: float = 0.0

    @computed_field(return_type=float)
    @property
    def total(self) -> float:
        return float(
            self.base
            + self.prevention
            + self.detection
            + self.containment
            + self.spread_penalty
            + self.collateral_penalty
            + self.loop_penalty
        )


class CyberObservation(Observation):
    """Partially observable SOC view."""

    task_id: str = Field(..., description="Task identifier")
    host_compromise: Dict[str, bool] = Field(default_factory=dict)
    host_isolation: Dict[str, bool] = Field(default_factory=dict)
    service_status: Dict[str, Dict[str, bool]] = Field(default_factory=dict)
    ids_alerts: List[str] = Field(default_factory=list)
    traffic_anomaly_score: float = Field(0.0, ge=0.0, le=1.0)
    active_incidents: List[Incident] = Field(default_factory=list)
    reward_signal: RewardSignal = Field(default_factory=RewardSignal)
    step_budget_remaining: int = Field(0, ge=0)
    available_defender_actions: List[DefenderActionName] = Field(default_factory=list)


class CyberState(State):
    """Full internal environment state."""

    task_id: str = Field(default="")
    current_seed: int = Field(default=0)
    max_steps: int = Field(default=50, ge=1)
    compromised_hosts: Dict[str, bool] = Field(default_factory=dict)
    isolated_hosts: Dict[str, bool] = Field(default_factory=dict)
    blocked_ips: List[str] = Field(default_factory=list)
    services_vulnerable: Dict[str, Dict[str, bool]] = Field(default_factory=dict)
    ids_alerts: List[str] = Field(default_factory=list)
    incidents: List[Incident] = Field(default_factory=list)
    attack_graph_edges: List[tuple[str, str]] = Field(default_factory=list)
    cooldowns: Dict[str, int] = Field(default_factory=dict)
    action_history: List[CyberAction] = Field(default_factory=list)
    cumulative_reward: float = Field(default=0.0)
    detected_attacks: int = Field(default=0, ge=0)
    prevented_attacks: int = Field(default=0, ge=0)
    collateral_damage: int = Field(default=0, ge=0)

    @computed_field(return_type=int)
    @property
    def compromised_count(self) -> int:
        return sum(1 for compromised in self.compromised_hosts.values() if compromised)

    @computed_field(return_type=int)
    @property
    def isolated_count(self) -> int:
        return sum(1 for isolated in self.isolated_hosts.values() if isolated)


class StepInfo(BaseModel):
    """Gym-style step info payload for analytics."""

    attacker_action: Dict[str, Any]
    defender_action: Dict[str, Any]
    reward_breakdown: RewardSignal
    metrics: Dict[str, float]
    terminal_reason: Optional[str] = None
    terminal_reason: Optional[str] = None
