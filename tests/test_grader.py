from cyber_openenv_rl.grading import TaskGrader
from cyber_openenv_rl.models import CyberState
from cyber_openenv_rl.tasks import get_task


def test_grader_bounds_and_shape() -> None:
    grader = TaskGrader()
    task = get_task("hard")
    state = CyberState(
        task_id="hard",
        max_steps=40,
        compromised_hosts={h: False for h in task.hosts},
        isolated_hosts={h: False for h in task.hosts},
    )
    grade = grader.score(task, state, trajectory_len=10)
    assert 0.0 <= grade.score <= 1.0
    assert set(grade.breakdown.keys()) == {
        "prevention",
        "detection",
        "response_time",
        "collateral_damage_control",
    }
