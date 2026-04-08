from pathlib import Path

from cyber_openenv_rl.rl.train_defender import train


def test_smoke_train_ppo_and_dqn(tmp_path: Path) -> None:
    p1 = train(
        "ppo",
        "easy",
        total_timesteps=64,
        output_dir=tmp_path / "ppo",
        log_dir=tmp_path / "logs" / "ppo",
    )
    p2 = train(
        "dqn",
        "easy",
        total_timesteps=64,
        output_dir=tmp_path / "dqn",
        log_dir=tmp_path / "logs" / "dqn",
    )
    assert p1.exists()
    assert p2.exists()
