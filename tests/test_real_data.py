from pathlib import Path

from cyber_openenv_rl.data.real_dataset import build_threat_profile


def test_build_threat_profile_from_sample(tmp_path: Path) -> None:
    p = tmp_path / "sample.txt"
    p.write_text(
        "0,tcp,http,SF,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,normal,20\n"
        "0,tcp,http,SF,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,guess_passwd,20\n",
        encoding="utf-8",
    )
    profile = build_threat_profile(p)
    assert "credential_stuff_success_threshold" in profile
    assert "lateral_move_success_threshold" in profile
