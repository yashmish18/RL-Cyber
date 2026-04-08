from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    difficulty: str
    description: str
    max_steps: int
    seed: int
    hosts: List[str]
    services: Dict[str, List[str]]
    initial_compromised: List[str]
    attacker_entry_host: str
    attack_script: List[dict]
    objective: str


TASKS: Dict[str, TaskConfig] = {
    "easy": TaskConfig(
        task_id="easy",
        difficulty="easy",
        description="Single-host brute-force triage with fast containment.",
        max_steps=20,
        seed=101,
        hosts=["web-1", "db-1"],
        services={"web-1": ["ssh", "http"], "db-1": ["postgres"]},
        initial_compromised=[],
        attacker_entry_host="web-1",
        attack_script=[
            {"action_type": "recon", "target_host": "web-1", "source_ip": "10.0.0.45"},
            {"action_type": "credential_stuff", "target_host": "web-1", "source_ip": "10.0.0.45"},
            {"action_type": "malware_drop", "target_host": "web-1", "source_ip": "10.0.0.45"},
            {"action_type": "lateral_move", "target_host": "db-1", "source_ip": "10.0.0.45"},
        ],
        objective="Detect and contain brute-force entry before lateral movement.",
    ),
    "medium": TaskConfig(
        task_id="medium",
        difficulty="medium",
        description="Subnet lateral movement requiring selective isolation and patching.",
        max_steps=30,
        seed=202,
        hosts=["jump-1", "app-1", "app-2", "db-1"],
        services={
            "jump-1": ["ssh"],
            "app-1": ["ssh", "http"],
            "app-2": ["ssh", "http"],
            "db-1": ["postgres"],
        },
        initial_compromised=["jump-1"],
        attacker_entry_host="jump-1",
        attack_script=[
            {"action_type": "recon", "target_host": "app-1", "source_ip": "172.16.2.20"},
            {"action_type": "lateral_move", "target_host": "app-1", "source_ip": "172.16.2.20"},
            {"action_type": "credential_stuff", "target_host": "app-2", "source_ip": "172.16.2.20"},
            {"action_type": "lateral_move", "target_host": "db-1", "source_ip": "172.16.2.20"},
            {"action_type": "malware_drop", "target_host": "db-1", "source_ip": "172.16.2.20"},
        ],
        objective="Prevent compromise spread to data tier while preserving service uptime.",
    ),
    "hard": TaskConfig(
        task_id="hard",
        difficulty="hard",
        description="Multi-stage campaign with decoy signals and constrained action budget.",
        max_steps=40,
        seed=303,
        hosts=["edge-1", "edge-2", "api-1", "api-2", "mq-1", "db-1"],
        services={
            "edge-1": ["ssh", "http"],
            "edge-2": ["ssh", "http"],
            "api-1": ["http", "grpc"],
            "api-2": ["http", "grpc"],
            "mq-1": ["amqp"],
            "db-1": ["postgres"],
        },
        initial_compromised=["edge-1"],
        attacker_entry_host="edge-1",
        attack_script=[
            {"action_type": "recon", "target_host": "edge-2", "source_ip": "185.19.88.90"},
            {"action_type": "credential_stuff", "target_host": "edge-2", "source_ip": "185.19.88.90"},
            {"action_type": "lateral_move", "target_host": "api-1", "source_ip": "185.19.88.90"},
            {"action_type": "malware_drop", "target_host": "mq-1", "source_ip": "185.19.88.90"},
            {"action_type": "lateral_move", "target_host": "db-1", "source_ip": "185.19.88.90"},
            {"action_type": "credential_stuff", "target_host": "api-2", "source_ip": "185.19.88.90"},
            {"action_type": "malware_drop", "target_host": "api-2", "source_ip": "185.19.88.90"},
        ],
        objective="Contain advanced campaign and protect crown-jewel data services.",
    ),
}


def get_task(task_id: str) -> TaskConfig:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id}. Valid: {list(TASKS)}")
    return TASKS[task_id]
