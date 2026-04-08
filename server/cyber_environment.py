import numpy as np
from uuid import uuid4
from pathlib import Path
from stable_baselines3 import PPO, DQN

from openenv.core.env_server.interfaces import Environment

from cyber_openenv_rl.core.simulator import CyberSimulator, SimulatorConfig
from cyber_openenv_rl.models import CyberAction, CyberObservation, CyberState


class CyberEnvironment(Environment[CyberAction, CyberObservation, CyberState]):
    """OpenEnv server environment backed by deterministic cyber simulator."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task_id: str = "easy", seed: int = 101):
        super().__init__()
        self.task_id = task_id
        self.seed = seed
        self._episode_id = str(uuid4())
        self.sim = CyberSimulator(SimulatorConfig(task_id=task_id, seed=seed))
        
        # Try to load the best model
        self.model = None
        model_path = Path("outputs/models/best/best_model.zip")
        if not model_path.exists():
             # Fallback to a common path
             model_path = Path(f"outputs/models/defender_ppo_{task_id}.zip")
             
        if model_path.exists():
            print(f"Server: Loading AI Agent from {model_path}")
            try:
                self.model = PPO.load(str(model_path))
            except Exception:
                try:
                    self.model = DQN.load(str(model_path))
                except Exception as e:
                    print(f"Server: Failed to load model: {e}")

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs,
    ) -> CyberObservation:
        if "task_id" in kwargs:
            self.task_id = kwargs["task_id"]
            self.sim = CyberSimulator(SimulatorConfig(task_id=self.task_id, seed=seed or self.seed))

        self._episode_id = episode_id or str(uuid4())
        obs = self.sim.reset(seed=seed or self.seed)
        obs.metadata.update({"episode_id": self._episode_id})
        return obs

    def step(
        self,
        action: CyberAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> CyberObservation:
        if action.actor != "defender":
            raise ValueError("OpenEnv step expects defender action; attacker is scripted by default")

        attacker_action = None
        attacker_payload = kwargs.get("attacker_action")
        if attacker_payload:
            attacker_action = CyberAction(**attacker_payload)

        obs, reward, done, info = self.sim.step(action, attacker_action=attacker_action)
        obs.reward = reward
        obs.done = done
        obs.metadata.update(info)
        obs.metadata["episode_id"] = self._episode_id
        
        # Add AI suggestion for next step
        if self.model:
            # Simple vectorization for prediction (matching gym_env logic)
            vec_obs = self._vectorize_for_model(obs)
            ai_action_idx, _ = self.model.predict(vec_obs, deterministic=True)
            info["ai_suggestion"] = int(ai_action_idx)
            
        return obs

    def _vectorize_for_model(self, obs: CyberObservation) -> np.ndarray:
        # Match the logic in gym_env.py
        v = []
        hosts = self.sim.task_config.hosts
        services = sorted({s for svcs in self.sim.task_config.services.values() for s in svcs})
        
        for h in hosts:
            v.append(1.0 if obs.host_compromise.get(h, False) else 0.0)
            v.append(1.0 if obs.host_isolation.get(h, False) else 0.0)
            alert_score = 1.0 if any(h in alert for alert in obs.ids_alerts) else 0.0
            v.append(alert_score)

        for h in hosts:
            for s in services:
                v.append(1.0 if obs.service_status.get(h, {}).get(s, False) else 0.0)

        v.append(float(obs.traffic_anomaly_score))
        denom = max(1, self.sim.task_config.max_steps)
        v.append(float(obs.step_budget_remaining / denom))
        return np.array([v], dtype=np.float32)

    @property
    def state(self) -> CyberState:
        st = self.sim.state().model_copy()
        st.episode_id = self._episode_id
        return st
