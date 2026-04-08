from __future__ import annotations

import argparse
from pathlib import Path

from .train_defender import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Curriculum training easy -> medium -> hard")
    parser.add_argument("--algorithm", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--timesteps-per-task", type=int, default=6000)
    parser.add_argument("--output-dir", default="outputs/models/curriculum")
    parser.add_argument("--log-dir", default="outputs/logs/curriculum")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    args = parser.parse_args()

    output = Path(args.output_dir)
    logs = Path(args.log_dir)
    checkpoints = []

    for i, task in enumerate(["easy", "medium", "hard"]):
        ckpt = train(
            algorithm=args.algorithm,
            task_id=task,
            total_timesteps=args.timesteps_per_task,
            output_dir=output / task,
            log_dir=logs / task,
            seed=args.seed + i,
            device=args.device,
        )
        checkpoints.append(str(ckpt))
        print(f"task={task} model={ckpt}")

    print("curriculum_complete=true")
    print("checkpoints=")
    for ckpt in checkpoints:
        print(ckpt)


if __name__ == "__main__":
    main()
