from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ..data.real_dataset import build_threat_profile, download_nsl_kdd, save_profile
from .gym_env import DefenderGymEnv


def _make_env(task: str, seed: int, threat_profile: dict):
    def _factory():
        return Monitor(DefenderGymEnv(task_id=task, seed=seed, threat_profile=threat_profile))

    return _factory


def train_for_wall_time(
    algorithm: str,
    task: str,
    hours: float,
    chunk_timesteps: int,
    output_dir: Path,
    seed: int,
    threat_profile: dict,
    device: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = DummyVecEnv([_make_env(task, seed, threat_profile)])

    if algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=1,
            device=device,
            n_steps=max(128, min(1024, chunk_timesteps)),
        )
    elif algorithm == "dqn":
        model = DQN(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=1,
            device=device,
            learning_starts=min(500, max(20, chunk_timesteps // 5)),
        )
    else:
        raise ValueError("algorithm must be ppo or dqn")

    start = time.time()
    duration = hours * 3600.0
    round_id = 0
    while (time.time() - start) < duration:
        round_id += 1
        model.learn(total_timesteps=chunk_timesteps, reset_num_timesteps=False)
        ckpt = output_dir / f"{algorithm}_{task}_round{round_id}.zip"
        model.save(str(ckpt))

    final_model = output_dir / f"{algorithm}_{task}_realdata_final.zip"
    model.save(str(final_model))

    meta = {
        "algorithm": algorithm,
        "task": task,
        "seed": seed,
        "hours_requested": hours,
        "chunk_timesteps": chunk_timesteps,
        "elapsed_seconds": time.time() - start,
        "final_model": str(final_model),
        "threat_profile": threat_profile,
    }
    (output_dir / f"{algorithm}_{task}_realdata_training_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return final_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train defender on simulator calibrated with real intrusion dataset")
    parser.add_argument("--algorithm", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="hard")
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--chunk-timesteps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-path", default="data/real/nsl_kdd/KDDTrain+.txt")
    parser.add_argument("--profile-out", default="outputs/evals/real_data_threat_profile.json")
    parser.add_argument("--output-dir", default="outputs/models/real_data")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    ds_path = download_nsl_kdd(Path(args.dataset_path))
    profile = build_threat_profile(ds_path)
    save_profile(profile, Path(args.profile_out))

    model_path = train_for_wall_time(
        algorithm=args.algorithm,
        task=args.task,
        hours=args.hours,
        chunk_timesteps=args.chunk_timesteps,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        threat_profile=profile,
        device=args.device,
    )
    print(f"final_model={model_path}")


if __name__ == "__main__":
    main()
