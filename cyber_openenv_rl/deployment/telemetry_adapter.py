from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ..models import Incident


@dataclass
class TelemetrySnapshot:
    host_compromise: Dict[str, bool]
    host_isolation: Dict[str, bool]
    service_status: Dict[str, Dict[str, bool]]
    ids_alerts: List[str]
    traffic_anomaly_score: float
    active_incidents: List[Incident]


class TelemetryAdapter:
    """Maps SIEM-like JSON telemetry into environment-style snapshot."""

    @staticmethod
    def from_json(payload: Dict[str, Any], known_hosts: List[str]) -> TelemetrySnapshot:
        compromise = {h: False for h in known_hosts}
        isolation = {h: False for h in known_hosts}
        service_status = {h: {} for h in known_hosts}

        for event in payload.get("events", []):
            host = event.get("host")
            if host not in compromise:
                continue
            evt = event.get("type", "")
            if evt in {"compromise", "malware", "lateral_move"}:
                compromise[host] = True
            if evt == "isolated":
                isolation[host] = True
            if evt == "service_status":
                svc = event.get("service", "unknown")
                service_status[host][svc] = bool(event.get("healthy", True))

        alerts = [str(a) for a in payload.get("alerts", [])][-20:]
        anomaly = float(payload.get("anomaly_score", 0.0))

        incidents = []
        for item in payload.get("incidents", []):
            host = item.get("host")
            if host not in compromise:
                continue
            incidents.append(
                Incident(
                    host=host,
                    attack_type=str(item.get("attack_type", "unknown")),
                    severity=float(item.get("severity", 0.5)),
                    detected=bool(item.get("detected", False)),
                )
            )

        return TelemetrySnapshot(
            host_compromise=compromise,
            host_isolation=isolation,
            service_status=service_status,
            ids_alerts=alerts,
            traffic_anomaly_score=max(0.0, min(1.0, anomaly)),
            active_incidents=incidents,
        )
