from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ..models import CyberAction, CyberObservation, CyberState, Incident, RewardSignal, StepInfo
from ..tasks import TaskConfig, get_task


DEFENDER_ACTIONS = ["block_ip", "isolate_node", "scan_host", "patch_service", "restore_backup", "ignore"]
ATTACKER_ACTIONS = ["lateral_move", "credential_stuff", "malware_drop", "recon", "privesc", "exfiltrate", "idle"]

HOST_CRITICALITY = {
    "db": 5.0,
    "domain_controller": 10.0,
    "web": 2.0,
    "admin": 3.0,
    "user": 1.0
}


@dataclass
class SimulatorConfig:
    task_id: str = "easy"
    seed: int | None = None
    threat_profile: dict[str, float] | None = None


class CyberSimulator:
    """Deterministic multi-agent simulator with Gym-style step API."""

    def __init__(self, config: SimulatorConfig):
        self.task_config = get_task(config.task_id)
        self.seed = self.task_config.seed if config.seed is None else config.seed
        self.rng = np.random.default_rng(self.seed)
        self.threat_profile = config.threat_profile or {}
        self._state = CyberState(task_id=self.task_config.task_id)
        self._done = False
        self._last_info: StepInfo | None = None
        self.reset(seed=self.seed)

    def reset(self, seed: int | None = None) -> CyberObservation:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        self._done = False
        compromised = {h: False for h in self.task_config.hosts}
        for h in self.task_config.initial_compromised:
            compromised[h] = True

        vulnerable = {
            host: {svc: True for svc in services}
            for host, services in self.task_config.services.items()
        }

        self._state = CyberState(
            task_id=self.task_config.task_id,
            current_seed=self.seed,
            step_count=0,
            max_steps=self.task_config.max_steps,
            compromised_hosts=compromised,
            isolated_hosts={h: False for h in self.task_config.hosts},
            blocked_ips=[],
            services_vulnerable=vulnerable,
            ids_alerts=[],
            incidents=[],
            attack_graph_edges=self._build_attack_edges(),
            cooldowns={},
            action_history=[],
            cumulative_reward=0.0,
        )
        return self._build_observation(
            reward=0.0, done=False, reward_signal=RewardSignal()
        )

    def state(self) -> CyberState:
        """Method-form state accessor for Gym/OpenEnv compatibility docs."""
        return self._state

    def current_observation(self) -> CyberObservation:
        return self._build_observation(
            reward=0.0, done=self._done, reward_signal=RewardSignal()
        )

    def step(
        self,
        defender_action: CyberAction,
        attacker_action: CyberAction | None = None,
    ) -> tuple[CyberObservation, float, bool, Dict[str, Any]]:
        if defender_action.actor != "defender":
            raise ValueError("defender_action.actor must be 'defender'")
        if self._done:
            return self._build_observation(
                0.0, True, reward_signal=RewardSignal()
            ), 0.0, True, {
                "terminal_reason": "episode_already_done"
            }

        self._state.step_count += 1

        if attacker_action is None:
            attacker_action = self.scripted_attacker_action(self._state.step_count - 1)

        reward_breakdown = RewardSignal()

        self._apply_attacker_action(attacker_action, reward_breakdown)
        self._apply_defender_action(defender_action, reward_breakdown)

        reward = reward_breakdown.total
        self._state.cumulative_reward += reward
        self._state.action_history.extend([attacker_action, defender_action])

        terminal_reason = self._terminal_reason()
        self._done = terminal_reason is not None

        obs = self._build_observation(
            reward=reward, done=self._done, reward_signal=reward_breakdown
        )
        metrics = self._compute_metrics()
        info = StepInfo(
            attacker_action=attacker_action.model_dump(),
            defender_action=defender_action.model_dump(),
            reward_breakdown=reward_breakdown,
            metrics=metrics,
            terminal_reason=terminal_reason,
        )
        self._last_info = info
        return obs, reward, self._done, info.model_dump()

    def scripted_attacker_action(self, idx: int) -> CyberAction:
        script = self.task_config.attack_script
        if idx < len(script):
            action = script[idx]
        else:
            live_hosts = [h for h, isolated in self._state.isolated_hosts.items() if not isolated]
            target = live_hosts[idx % len(live_hosts)] if live_hosts else self.task_config.hosts[0]
            action = {
                "action_type": "recon",
                "target_host": target,
                "source_ip": "198.51.100.25",
            }

        return CyberAction(actor="attacker", **action)

    def _build_attack_edges(self) -> list[tuple[str, str]]:
        hosts = self.task_config.hosts
        return [(hosts[i], hosts[i + 1]) for i in range(len(hosts) - 1)]

    def _apply_attacker_action(self, action: CyberAction, rb: RewardSignal) -> None:
        target = action.target_host
        source_ip = action.source_ip or "0.0.0.0"
        
        # Survival bonus for this step
        rb.base += 0.05 
        
        if source_ip in self._state.blocked_ips:
            rb.prevention += 2.0  # Increased from 1.5
            return
        if self._state.isolated_hosts.get(target, False):
            rb.prevention += 1.5  # Increased from 1.0
            return

        noisy = self.rng.uniform(0.0, 1.0)

        if action.action_type == "recon":
            if noisy > 0.6:
                self._state.ids_alerts.append(f"Recon on {target} from {source_ip}")
            rb.base -= 0.1
            return

        if action.action_type == "credential_stuff":
            threshold = self.threat_profile.get("credential_stuff_success_threshold", 0.35)
            success = noisy > threshold
            if success:
                self._state.compromised_hosts[target] = True
                self._state.incidents.append(
                    Incident(host=target, attack_type="credential_stuff", severity=0.6)
                )
                penalty = 3.0 if "db" in target else 1.7
                rb.spread_penalty -= penalty
            else:
                rb.prevention += 0.5
            return

        if action.action_type == "malware_drop":
            if self._state.compromised_hosts.get(target, False):
                self._state.incidents.append(
                    Incident(host=target, attack_type="malware_drop", severity=0.8)
                )
                rb.spread_penalty -= 2.2
            else:
                rb.prevention += 0.5
            return

        if action.action_type == "lateral_move":
            if not self._state.compromised_hosts.get(target, False):
                # Lateral move is harder if the source host isn't compromised
                threshold = self.threat_profile.get("lateral_move_success_threshold", 0.45)
                self._state.compromised_hosts[target] = noisy > threshold
            
            if self._state.compromised_hosts[target]:
                self._state.incidents.append(
                    Incident(host=target, attack_type="lateral_move", severity=0.7)
                )
                critic = self._get_host_criticality(target)
                rb.spread_penalty -= (2.0 * critic)
            else:
                rb.prevention += 1.0
            return

        if action.action_type == "privesc":
            if self._state.compromised_hosts.get(target, False):
                # If already compromised, gain "root" (higher penalty if not caught)
                rb.spread_penalty -= 1.5
                self._state.incidents.append(
                    Incident(host=target, attack_type="privesc", severity=0.9)
                )
            return

        if action.action_type == "exfiltrate":
            if self._state.compromised_hosts.get(target, False) and "db" in target:
                rb.spread_penalty -= 10.0  # MAJOR PENALTY
                self._state.incidents.append(
                    Incident(host=target, attack_type="exfiltration", severity=1.0)
                )
            return

    def _get_host_criticality(self, host: str) -> float:
        for key, val in HOST_CRITICALITY.items():
            if key in host.lower():
                return val
        return 1.0

    def _apply_defender_action(self, action: CyberAction, rb: RewardSignal) -> None:
        host = action.target_host

        if action.action_type == "ignore":
            rb.loop_penalty -= 0.6
            return

        if action.action_type == "block_ip":
            ip = action.source_ip or "0.0.0.0"
            if ip not in self._state.blocked_ips:
                self._state.blocked_ips.append(ip)
                rb.prevention += 1.2
            else:
                rb.loop_penalty -= 0.4
            return

        if action.action_type == "scan_host":
            compromised = self._state.compromised_hosts.get(host, False)
            if compromised:
                rb.detection += 1.1
                self._state.detected_attacks += 1
                self._state.ids_alerts.append(f"Confirmed compromise on {host}")
                for incident in self._state.incidents:
                    if incident.host == host:
                        incident.detected = True
            else:
                rb.collateral_penalty -= 0.2
            return

        if action.action_type == "isolate_node":
            if not self._state.isolated_hosts.get(host, False):
                self._state.isolated_hosts[host] = True
                rb.containment += 1.5
                if self._state.compromised_hosts.get(host, False):
                    rb.containment += 1.0
                    self._state.prevented_attacks += 1
            else:
                rb.loop_penalty -= 0.3
            return

        if action.action_type == "patch_service":
            service = action.target_service
            if service and service in self._state.services_vulnerable.get(host, {}):
                if self._state.services_vulnerable[host][service]:
                    self._state.services_vulnerable[host][service] = False
                    rb.prevention += 0.9
                else:
                    rb.loop_penalty -= 0.2
            else:
                rb.collateral_penalty -= 0.3
            return

        if action.action_type == "restore_backup":
            if self._state.compromised_hosts.get(host, False):
                self._state.compromised_hosts[host] = False
                rb.containment += 2.0
            else:
                rb.loop_penalty -= 0.5
            return

    def _build_observation(
        self, reward: float, done: bool, reward_signal: RewardSignal
    ) -> CyberObservation:
        anomaly = float(min(1.0, len(self._state.incidents) * 0.18 + self.rng.uniform(0.0, 0.08)))
        incidents = self._state.incidents[-10:]
        return CyberObservation(
            task_id=self.task_config.task_id,
            host_compromise=self._state.compromised_hosts,
            host_isolation=self._state.isolated_hosts,
            service_status={
                h: {svc: (not vuln) for svc, vuln in v.items()}
                for h, v in self._state.services_vulnerable.items()
            },
            ids_alerts=self._state.ids_alerts[-10:],
            traffic_anomaly_score=anomaly,
            active_incidents=incidents,
            reward_signal=reward_signal,
            step_budget_remaining=max(0, self.task_config.max_steps - self._state.step_count),
            available_defender_actions=DEFENDER_ACTIONS,
            reward=reward,
            done=done,
            metadata={
                "seed": self.seed,
                "step": self._state.step_count,
            },
        )

    def _terminal_reason(self) -> str | None:
        if self._state.step_count >= self.task_config.max_steps:
            return "max_steps"
        
        # Don't end too early on minor compromises, but be strict on data
        total_hosts = len(self.task_config.hosts)
        comp_count = sum(1 for h, c in self._state.compromised_hosts.items() if c)
        
        if comp_count >= max(2, total_hosts - 1):
            return "catastrophic_compromise"
            
        data_hosts = [h for h in self.task_config.hosts if "db" in h]
        if data_hosts and any(self._state.compromised_hosts.get(h, False) for h in data_hosts):
            # If a DB is compromised, it's game over immediately
            return "crown_jewel_compromised"

        # If exfiltration happened, it's also terminal failure
        if any(inc.attack_type == "exfiltration" for inc in self._state.incidents):
            return "data_exfiltrated"
            
        return None

    def _compute_metrics(self) -> Dict[str, float]:
        total_hosts = len(self.task_config.hosts)
        compromised = self._state.compromised_count
        isolated = self._state.isolated_count
        healthy = total_hosts - compromised
        return {
            "detection_rate": float(self._state.detected_attacks / max(1, len(self._state.incidents))),
            "service_availability": float(max(0.0, 1.0 - (isolated / max(1, total_hosts)))),
            "compromise_ratio": float(compromised / max(1, total_hosts)),
            "healthy_ratio": float(healthy / max(1, total_hosts)),
            "response_time_norm": float(self._state.step_count / max(1, self.task_config.max_steps)),
            "cumulative_reward": float(self._state.cumulative_reward),
        }

    @property
    def done(self) -> bool:
        return self._done

    @property
    def last_info(self) -> StepInfo | None:
        return self._last_info
