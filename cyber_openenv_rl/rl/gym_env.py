from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core.simulator import CyberSimulator, DEFENDER_ACTIONS, SimulatorConfig
from ..models import CyberAction, CyberObservation


class DefenderGymEnv(gym.Env):
    """Gym wrapper for training defender against scripted or callable attackers."""

    metadata = {"render_modes": []}

    def __init__(
        self, task_id: str = "easy", seed: int = 101, threat_profile: dict | None = None
    ):
        super().__init__()
        self.task_id = task_id
        self.seed = seed
        self.threat_profile = threat_profile or {}
        self.sim = CyberSimulator(
            SimulatorConfig(task_id=task_id, seed=seed, threat_profile=self.threat_profile)
        )
        self.hosts = self.sim.task_config.hosts
        self.services = sorted({s for svcs in self.sim.task_config.services.values() for s in svcs})

        self.action_space = spaces.Discrete(len(DEFENDER_ACTIONS) * len(self.hosts) * max(1, len(self.services)))
        vec_len = len(self.hosts) * 3 + len(self.services) * len(self.hosts) + 2
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(vec_len,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs = self.sim.reset(seed=seed or self.seed)
        return self._vectorize(obs), {}

    def step(self, action_idx: int):
        action = self._decode_action(action_idx)
        obs, reward, done, info = self.sim.step(action)
        terminated = done
        truncated = False
        return self._vectorize(obs), float(reward), terminated, truncated, info

    def _decode_action(self, idx: int) -> CyberAction:
        idx = int(idx) % self.action_space.n
        svc_count = max(1, len(self.services))
        host_idx = (idx // svc_count) % len(self.hosts)
        action_idx = idx // (len(self.hosts) * svc_count)
        action_type = DEFENDER_ACTIONS[action_idx % len(DEFENDER_ACTIONS)]
        service = self.services[idx % svc_count] if self.services else None
        if action_type != "patch_service":
            service = None
        target_host = self.hosts[host_idx]

        source_ip = "10.0.0.45"
        return CyberAction(
            actor="defender",
            action_type=action_type,
            target_host=target_host,
            target_service=service,
            source_ip=source_ip,
        )

    def _vectorize(self, obs: CyberObservation) -> np.ndarray:
        v = []
        for h in self.hosts:
            v.append(1.0 if obs.host_compromise.get(h, False) else 0.0)
            v.append(1.0 if obs.host_isolation.get(h, False) else 0.0)
            alert_score = 1.0 if any(h in alert for alert in obs.ids_alerts) else 0.0
            v.append(alert_score)

        for h in self.hosts:
            for s in self.services:
                v.append(1.0 if obs.service_status.get(h, {}).get(s, False) else 0.0)

        v.append(float(obs.traffic_anomaly_score))
        denom = max(1, self.sim.task_config.max_steps)
        v.append(float(obs.step_budget_remaining / denom))
        return np.array(v, dtype=np.float32)
