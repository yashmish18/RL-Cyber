from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..models import CyberAction


DEFENSIVE_ALLOWLIST = {
    "block_ip",
    "isolate_node",
    "scan_host",
    "patch_service",
    "ignore",
}


@dataclass
class GuardrailDecision:
    allowed: bool
    reason: str
    safe_action: CyberAction


def apply_guardrails(
    proposed: CyberAction,
    known_hosts: Iterable[str],
    human_approval_required: bool,
) -> GuardrailDecision:
    hosts = set(known_hosts)

    if proposed.actor != "defender":
        safe = CyberAction(
            actor="defender",
            action_type="ignore",
            target_host=next(iter(hosts)) if hosts else "unknown-host",
            source_ip="0.0.0.0",
        )
        return GuardrailDecision(False, "non-defender action blocked", safe)

    if proposed.action_type not in DEFENSIVE_ALLOWLIST:
        safe = proposed.model_copy(update={"action_type": "ignore"})
        return GuardrailDecision(False, "action type not allowlisted", safe)

    if proposed.target_host not in hosts:
        safe = proposed.model_copy(update={"action_type": "ignore", "target_host": next(iter(hosts)) if hosts else proposed.target_host})
        return GuardrailDecision(False, "unknown target host blocked", safe)

    if human_approval_required and proposed.action_type in {"isolate_node", "block_ip"}:
        safe = proposed.model_copy(update={"action_type": "scan_host"})
        return GuardrailDecision(False, "requires human approval; downgraded to scan_host", safe)

    return GuardrailDecision(True, "allowed", proposed)
