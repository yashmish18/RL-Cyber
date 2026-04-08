from __future__ import annotations

import json
from pathlib import Path

from ..core.policies import scripted_defender_policy
from ..core.simulator import CyberSimulator, SimulatorConfig
from ..grading import TaskGrade, TaskGrader, aggregate_grades
from ..tasks import TASKS


def run(output_path: Path) -> dict:
    grader = TaskGrader()
    task_results = []
    grades = []

    for task_id, config in TASKS.items():
        sim = CyberSimulator(SimulatorConfig(task_id=task_id, seed=config.seed))
        obs = sim.reset(seed=config.seed)
        steps = 0
        last_info = {}
        while not sim.done:
            action = scripted_defender_policy(obs)
            obs, _, _, last_info = sim.step(action)
            steps += 1

        grade = grader.score(config, sim.state(), steps)
        grades.append(TaskGrade(score=grade.score, breakdown=grade.breakdown))
        task_results.append(
            {
                "task_id": task_id,
                "score": grade.score,
                "steps": steps,
                "terminal_reason": last_info.get("terminal_reason"),
                "breakdown": grade.breakdown,
                "metrics": last_info.get("metrics", {}),
            }
        )

    payload = {
        "baseline": "scripted_defender",
        "tasks": task_results,
        "aggregate_score": aggregate_grades(grades),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    p = run(Path("outputs/evals/scripted_baseline.json"))
    print(json.dumps(p, indent=2))
