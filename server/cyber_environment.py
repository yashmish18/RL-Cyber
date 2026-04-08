from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..core.simulator import CyberSimulator, SimulatorConfig
    from ..models import CyberAction, CyberObservation, CyberState
except ImportError:  # pragma: no cover
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
        return obs

    @property
    def state(self) -> CyberState:
        st = self.sim.state().model_copy()
        st.episode_id = self._episode_id
        return st
