from __future__ import annotations

import argparse
import json
from pathlib import Path

from .realtime_defender import run_realtime_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run realtime defensive inference from telemetry JSON")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--algorithm", choices=["ppo", "dqn"], required=True)
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="hard")
    parser.add_argument("--telemetry", required=True)
    parser.add_argument("--output", default="outputs/evals/realtime_inference.json")
    parser.add_argument("--no-human-approval", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.52)
    args = parser.parse_args()

    payload = run_realtime_file(
        model_path=Path(args.model_path),
        algorithm=args.algorithm,
        task_id=args.task,
        telemetry_file=Path(args.telemetry),
        output_file=Path(args.output),
        human_approval_required=not args.no_human_approval,
        confidence_threshold=args.confidence_threshold,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
