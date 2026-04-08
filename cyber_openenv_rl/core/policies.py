from __future__ import annotations

from ..models import CyberAction, CyberObservation


def scripted_defender_policy(obs: CyberObservation) -> CyberAction:
    # Prioritize hosts with detected compromise.
    for host, compromised in obs.host_compromise.items():
        if compromised and not obs.host_isolation.get(host, False):
            return CyberAction(
                actor="defender",
                action_type="isolate_node",
                target_host=host,
                source_ip="10.0.0.45",
            )

    # Investigate latest alert host if present.
    for alert in reversed(obs.ids_alerts):
        for host in obs.host_compromise.keys():
            if host in alert:
                return CyberAction(
                    actor="defender",
                    action_type="scan_host",
                    target_host=host,
                    source_ip="10.0.0.45",
                )

    # Fallback: patch first service on most exposed host.
    first_host = next(iter(obs.service_status.keys()))
    services = list(obs.service_status.get(first_host, {}).keys())
    if not services:
        return CyberAction(
            actor="defender",
            action_type="scan_host",
            target_host=first_host,
            source_ip="10.0.0.45",
        )
    first_service = services[0]
    return CyberAction(
        actor="defender",
        action_type="patch_service",
        target_host=first_host,
        target_service=first_service,
        source_ip="10.0.0.45",
    )
