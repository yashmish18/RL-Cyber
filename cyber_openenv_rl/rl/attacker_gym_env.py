from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core.policies import scripted_defender_policy
from ..core.simulator import ATTACKER_ACTIONS, CyberSimulator, SimulatorConfig
from ..models import CyberAction


class AttackerGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, task_id: str = "hard", seed: int = 303):
        super().__init__()
        self.sim = CyberSimulator(SimulatorConfig(task_id=task_id, seed=seed))
        self.hosts = self.sim.task_config.hosts
        self.action_space = spaces.Discrete(len(ATTACKER_ACTIONS) * len(self.hosts))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(len(self.hosts) * 2 + 2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs = self.sim.reset(seed=seed)
        return self._vec(obs), {}

    def step(self, action_idx: int):
        attacker_action = self._decode_action(action_idx)
        defender_action = scripted_defender_policy(self.sim.current_observation())
        obs, reward, done, info = self.sim.step(defender_action=defender_action, attacker_action=attacker_action)
        # Attacker objective is inverse defender reward
        attacker_reward = -float(reward)
        return self._vec(obs), attacker_reward, done, False, info

    def _decode_action(self, idx: int) -> CyberAction:
        host = self.hosts[idx % len(self.hosts)]
        action_type = ATTACKER_ACTIONS[(idx // len(self.hosts)) % len(ATTACKER_ACTIONS)]
        return CyberAction(
            actor="attacker",
            action_type=action_type,
            target_host=host,
            source_ip="185.19.88.90",
        )

    def _vec(self, obs):
        v = []
        for h in self.hosts:
            v.append(1.0 if obs.host_compromise.get(h, False) else 0.0)
            v.append(1.0 if obs.host_isolation.get(h, False) else 0.0)
        v.append(obs.traffic_anomaly_score)
        v.append(obs.step_budget_remaining / max(1, self.sim.task_config.max_steps))
        return np.array(v, dtype=np.float32)
