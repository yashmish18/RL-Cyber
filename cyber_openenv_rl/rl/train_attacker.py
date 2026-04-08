from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from .attacker_gym_env import AttackerGymEnv


def train(task_id: str, total_timesteps: int, output_dir: Path, device: str = "cpu") -> Path:
    env = AttackerGymEnv(task_id=task_id)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=42,
        n_steps=2048,
        batch_size=64,
        learning_rate=1e-4,
        gamma=0.99,
        device=device,
    )
    model.learn(total_timesteps=total_timesteps)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"attacker_ppo_{task_id}.zip"
    model.save(str(path))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train optional attacker PPO model")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="hard")
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--output-dir", default="outputs/models")
    parser.add_argument("--device", default="cpu", help="auto|cpu|cuda")
    args = parser.parse_args()
    model_path = train(args.task, args.timesteps, Path(args.output_dir), device=args.device)
    print(f"saved_model={model_path}")


if __name__ == "__main__":
    main()
