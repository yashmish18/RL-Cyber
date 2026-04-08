from cyber_openenv_rl.core.simulator import CyberSimulator, SimulatorConfig
from cyber_openenv_rl.models import CyberAction


def _run(seed: int) -> tuple[float, int, int]:
    sim = CyberSimulator(SimulatorConfig(task_id="medium", seed=seed))
    sim.reset(seed=seed)
    reward_sum = 0.0
    while not sim.done:
        action = CyberAction(
            actor="defender",
            action_type="scan_host",
            target_host="app-1",
            source_ip="10.0.0.45",
        )
        _, reward, _, _ = sim.step(action)
        reward_sum += reward
    st = sim.state()
    return (reward_sum, st.compromised_count, st.step_count)


def test_same_seed_is_deterministic() -> None:
    first = _run(202)
    second = _run(202)
    assert first == second
