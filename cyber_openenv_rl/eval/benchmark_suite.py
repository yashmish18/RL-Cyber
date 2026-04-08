from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from stable_baselines3 import DQN, PPO

from ..grading import TaskGrade, TaskGrader, aggregate_grades
from ..rl.gym_env import DefenderGymEnv
from ..rl.train_defender import train
from ..tasks import TASKS


@dataclass
class EvalResult:
    algorithm: str
    task: str
    seed: int
    score: float
    mean_reward: float


def evaluate_model(model, task: str, seed: int, episodes: int = 5) -> EvalResult:
    env = DefenderGymEnv(task_id=task, seed=seed)
    grader = TaskGrader()
    rewards = []
    grades: List[TaskGrade] = []

    for _ in range(episodes):
        obs, _ = env.reset(seed=seed)
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = bool(terminated or truncated)
            total += float(reward)
        rewards.append(total)
        sim_state = env.sim.state()
        grades.append(grader.score(env.sim.task_config, sim_state, sim_state.step_count))

    return EvalResult(
        algorithm="",
        task=task,
        seed=seed,
        score=aggregate_grades(grades),
        mean_reward=float(sum(rewards) / max(1, len(rewards))),
    )


def run_benchmark(
    algorithms: List[str],
    seeds: List[int],
    timesteps: int,
    output_path: Path,
    train_models: bool,
) -> Dict:
    rows = []

    for algorithm in algorithms:
        for task in ["easy", "medium", "hard"]:
            for seed in seeds:
                model_path = (
                    output_path.parent
                    / "models"
                    / f"{algorithm}_{task}_seed{seed}.zip"
                )
                if train_models or not model_path.exists():
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    trained_path = train(
                        algorithm=algorithm,
                        task_id=task,
                        total_timesteps=timesteps,
                        output_dir=model_path.parent,
                        log_dir=output_path.parent / "logs" / f"{algorithm}_{task}_seed{seed}",
                        seed=seed,
                    )
                    if trained_path != model_path:
                        model_path.write_bytes(trained_path.read_bytes())

                if algorithm == "ppo":
                    model = PPO.load(str(model_path))
                else:
                    model = DQN.load(str(model_path))

                result = evaluate_model(model, task=task, seed=seed)
                result.algorithm = algorithm
                rows.append(result)

    by_algo = {}
    for algo in algorithms:
        algo_rows = [r for r in rows if r.algorithm == algo]
        by_algo[algo] = {
            "mean_score": float(sum(r.score for r in algo_rows) / max(1, len(algo_rows))),
            "mean_reward": float(sum(r.mean_reward for r in algo_rows) / max(1, len(algo_rows))),
        }

    leaderboard = sorted(
        by_algo.items(), key=lambda kv: kv[1]["mean_score"], reverse=True
    )

    payload = {
        "algorithms": algorithms,
        "seeds": seeds,
        "timesteps": timesteps,
        "rows": [
            {
                "algorithm": r.algorithm,
                "task": r.task,
                "seed": r.seed,
                "score": r.score,
                "mean_reward": r.mean_reward,
            }
            for r in rows
        ],
        "summary": by_algo,
        "leaderboard": leaderboard,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Benchmark Report",
        "",
        f"Timesteps per run: {timesteps}",
        f"Seeds: {seeds}",
        "",
        "## Leaderboard",
        "",
        "| Rank | Algorithm | Mean Score | Mean Reward |",
        "|---|---|---:|---:|",
    ]
    for i, (algo, stats) in enumerate(leaderboard, start=1):
        md_lines.append(
            f"| {i} | {algo} | {stats['mean_score']:.4f} | {stats['mean_reward']:.4f} |"
        )

    md_lines.extend([
        "",
        "## Per-Run Results",
        "",
        "| Algorithm | Task | Seed | Score | Mean Reward |",
        "|---|---|---:|---:|---:|",
    ])
    for r in rows:
        md_lines.append(
            f"| {r.algorithm} | {r.task} | {r.seed} | {r.score:.4f} | {r.mean_reward:.4f} |"
        )

    (output_path.parent / "benchmark_report.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed benchmark suite for defender agents")
    parser.add_argument("--algorithms", default="ppo,dqn")
    parser.add_argument("--seeds", default="42,1337,2026")
    parser.add_argument("--timesteps", type=int, default=3000)
    parser.add_argument("--output", default="outputs/evals/benchmark_results.json")
    parser.add_argument("--train", action="store_true", help="Train models before evaluation")
    args = parser.parse_args()

    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    payload = run_benchmark(
        algorithms=algorithms,
        seeds=seeds,
        timesteps=args.timesteps,
        output_path=Path(args.output),
        train_models=args.train,
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
