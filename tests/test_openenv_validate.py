import subprocess
from pathlib import Path


def test_openenv_validate_passes() -> None:
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        ["openenv", "validate", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
