from pathlib import Path

from cyber_openenv_rl.deployment.realtime_defender import run_realtime_file
from cyber_openenv_rl.rl.train_defender import train


def test_realtime_inference_outputs_confidence(tmp_path: Path) -> None:
    model_path = train(
        algorithm="ppo",
        task_id="easy",
        total_timesteps=64,
        output_dir=tmp_path / "models",
        log_dir=tmp_path / "logs",
        seed=42,
    )

    telemetry = tmp_path / "telemetry.json"
    telemetry.write_text(
        '{"anomaly_score":0.9,"events":[{"type":"compromise","host":"web-1"}],"alerts":["a"],"incidents":[{"host":"web-1","attack_type":"x","severity":0.8,"detected":false}]}'
    )
    out = tmp_path / "out.json"
    payload = run_realtime_file(
        model_path=model_path,
        algorithm="ppo",
        task_id="easy",
        telemetry_file=telemetry,
        output_file=out,
    )
    assert out.exists()
    assert "confidence" in payload["result"]
