from cyber_openenv_rl.deployment.guardrails import apply_guardrails
from cyber_openenv_rl.models import CyberAction


def test_guardrails_block_attacker_actor() -> None:
    proposed = CyberAction(
        actor="attacker",
        action_type="recon",
        target_host="edge-1",
        source_ip="1.2.3.4",
    )
    d = apply_guardrails(proposed, known_hosts=["edge-1"], human_approval_required=True)
    assert not d.allowed
    assert d.safe_action.actor == "defender"


def test_guardrails_downgrade_disruptive_action_without_approval() -> None:
    proposed = CyberAction(
        actor="defender",
        action_type="isolate_node",
        target_host="edge-1",
        source_ip="10.0.0.45",
    )
    d = apply_guardrails(proposed, known_hosts=["edge-1"], human_approval_required=True)
    assert not d.allowed
    assert d.safe_action.action_type == "scan_host"
