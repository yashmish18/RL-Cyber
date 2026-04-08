from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .gym_env import DefenderGymEnv

try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401

    _HAS_TENSORBOARD = True
except Exception:
    _HAS_TENSORBOARD = False


def _make_env(task_id: str, seed: int, threat_profile: dict | None = None):
    def _factory():
        env = DefenderGymEnv(task_id=task_id, seed=seed, threat_profile=threat_profile)
        return Monitor(env)

    return _factory


def _evaluate_policy(
    model, task_id: str, episodes: int = 5, threat_profile: dict | None = None
) -> dict:
    env = DefenderGymEnv(task_id=task_id, seed=404, threat_profile=threat_profile)
    totals = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = bool(terminated or truncated)
            total_reward += float(reward)
        totals.append(total_reward)
    return {
        "task": task_id,
        "episodes": episodes,
        "mean_reward": float(sum(totals) / max(1, len(totals))),
        "min_reward": float(min(totals)),
        "max_reward": float(max(totals)),
    }


def train(
    algorithm: str,
    task_id: str,
    total_timesteps: int,
    output_dir: Path,
    log_dir: Path,
    seed: int = 42,
    threat_profile: dict | None = None,
    device: str = "auto",
) -> Path:
    env = DummyVecEnv([_make_env(task_id=task_id, seed=seed, threat_profile=threat_profile)])
    eval_env = DummyVecEnv(
        [_make_env(task_id=task_id, seed=seed + 57, threat_profile=threat_profile)]
    )
    ppo_n_steps = max(64, min(512, total_timesteps))
    dqn_learning_starts = min(100, max(10, total_timesteps // 8))
    if algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            n_steps=ppo_n_steps,
            device=device,
            tensorboard_log=str(log_dir) if _HAS_TENSORBOARD else None,
        )
    elif algorithm == "dqn":
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            learning_starts=dqn_learning_starts,
            device=device,
            tensorboard_log=str(log_dir) if _HAS_TENSORBOARD else None,
        )
    else:
        raise ValueError("algorithm must be one of: ppo, dqn")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1000, total_timesteps // 10),
        save_path=str(output_dir / "checkpoints"),
        name_prefix=f"{algorithm}_{task_id}",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=max(500, total_timesteps // 8),
        deterministic=True,
        render=False,
    )
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb, eval_cb])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"defender_{algorithm}_{task_id}.zip"
    model.save(str(path))

    summary = _evaluate_policy(
        model, task_id=task_id, episodes=5, threat_profile=threat_profile
    )
    summary["seed"] = int(seed)
    summary["total_timesteps"] = int(total_timesteps)
    summary["algorithm"] = algorithm
    (output_dir / f"summary_{algorithm}_{task_id}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train defender PPO/DQN models")
    parser.add_argument("--algorithm", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--timesteps", type=int, default=8000)
    parser.add_argument("--output-dir", default="outputs/models")
    parser.add_argument("--log-dir", default="outputs/logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    args = parser.parse_args()

    model_path = train(
        args.algorithm,
        args.task,
        args.timesteps,
        Path(args.output_dir),
        Path(args.log_dir),
        seed=args.seed,
        device=args.device,
    )
    print(f"saved_model={model_path}")


if __name__ == "__main__":
    main()
