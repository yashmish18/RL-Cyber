from cyber_openenv_rl.core.simulator import CyberSimulator, SimulatorConfig
from cyber_openenv_rl.models import CyberAction
from server.cyber_environment import CyberEnvironment


def test_transition_and_reward_signal() -> None:
    sim = CyberSimulator(SimulatorConfig(task_id="easy", seed=101))
    sim.reset(seed=101)

    action = CyberAction(
        actor="defender",
        action_type="scan_host",
        target_host="web-1",
        source_ip="10.0.0.45",
    )
    obs, reward, done, info = sim.step(action)

    assert isinstance(reward, float)
    assert obs.task_id == "easy"
    assert "reward_breakdown" in info
    assert done in (True, False)


def test_loop_penalty_on_ignore() -> None:
    sim = CyberSimulator(SimulatorConfig(task_id="easy", seed=101))
    sim.reset(seed=101)
    ignore = CyberAction(
        actor="defender",
        action_type="ignore",
        target_host="web-1",
        source_ip="10.0.0.45",
    )
    _, reward, _, info = sim.step(ignore)
    assert reward <= 0.0
    assert info["reward_breakdown"]["loop_penalty"] < 0


def test_server_rejects_non_defender_step() -> None:
    env = CyberEnvironment(task_id="easy", seed=101)
    env.reset(seed=101)
    with_raised = False
    try:
        env.step(
            CyberAction(
                actor="attacker",
                action_type="recon",
                target_host="web-1",
                source_ip="10.0.0.45",
            )
        )
    except ValueError:
        with_raised = True
    assert with_raised


def test_simulator_rejects_non_defender_action() -> None:
    sim = CyberSimulator(SimulatorConfig(task_id="easy", seed=101))
    sim.reset(seed=101)
    raised = False
    try:
        sim.step(
            CyberAction(
                actor="attacker",
                action_type="recon",
                target_host="web-1",
                source_ip="1.2.3.4",
            )
        )
    except ValueError:
        raised = True
    assert raised
