from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from .gym_env import DefenderGymEnv

def test(
    algorithm: str,
    model_path: Path,
    stats_path: Path | None = None,
    task_id: str = "easy",
    episodes: int = 10,
    seed: int = 999
):
    print(f"\n[TESTING] {algorithm.upper()} model on task: {task_id}")
    print(f"Location: {model_path}")
    
    # 1. Create Environment
    def make_env():
        env = DefenderGymEnv(task_id=task_id, seed=seed)
        return Monitor(env)

    env = DummyVecEnv([make_env])

    # 2. Load Normalization Stats if provided
    if stats_path and stats_path.exists():
        print(f"Stats: Loading normalization stats from {stats_path}")
        env = VecNormalize.load(str(stats_path), env)
        # VERY IMPORTANT: Disable training mode for normalization
        env.training = False
        env.norm_reward = False 
    else:
        print("⚠️ Warning: No normalization stats found. Evaluation might be inaccurate.")

    # 3. Load Model
    if algorithm == "ppo":
        model = PPO.load(model_path, env=env)
    elif algorithm == "dqn":
        model = DQN.load(model_path, env=env)
    else:
        raise ValueError("algorithm must be ppo or dqn")

    # 4. Evaluation Loop
    all_rewards = []
    all_lengths = []
    db_compromised_count = 0

    for i in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
            
            # Check for terminal reason in info
            if done:
                reason = info[0].get("terminal_reason", "unknown")
                if reason == "crown_jewel_compromised":
                    db_compromised_count += 1
        
        all_rewards.append(total_reward)
        all_lengths.append(steps)
        print(f"Episode {i+1}: Reward = {total_reward:.2f}, Steps = {steps}")

    # 5. Summary
    print("\n" + "="*30)
    print("TEST RESULTS SUMMARY")
    print("="*30)
    print(f"Mean Reward:        {np.mean(all_rewards):.2f} (+/- {np.std(all_rewards):.2f})")
    print(f"Mean Episode Len:   {np.mean(all_lengths):.2f}")
    print(f"DB Compromise Rate: {db_compromised_count/episodes * 100:.1f}%")
    print("="*30)

def main():
    parser = argparse.ArgumentParser(description="Test a trained RL model")
    parser.add_argument("--algorithm", choices=["ppo", "dqn"], required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--stats-path", type=str, help="Path to .pkl normalization stats")
    parser.add_argument("--task", default="easy")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=999)
    
    args = parser.parse_args()
    
    test(
        algorithm=args.algorithm,
        model_path=Path(args.model_path),
        stats_path=Path(args.stats_path) if args.stats_path else None,
        task_id=args.task,
        episodes=args.episodes,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
