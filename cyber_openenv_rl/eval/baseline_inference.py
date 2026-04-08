from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

from ..core.policies import scripted_defender_policy
from ..core.simulator import CyberSimulator, SimulatorConfig
from ..grading import TaskGrade, TaskGrader, aggregate_grades
from ..models import CyberAction, CyberObservation
from ..tasks import TASKS, get_task
from .metrics import summarize_metrics


def _resolve_api_key() -> str:
    key = os.getenv("HF_TOKEN")
    if not key:
        raise RuntimeError("Missing credentials: set HF_TOKEN")
    return key


def _build_prompt(obs: CyberObservation) -> str:
    return (
        "You are a SOC defender. Choose exactly one action in compact JSON with keys: "
        "action_type,target_host,target_service,source_ip. "
        f"Available actions: {obs.available_defender_actions}. "
        f"Compromised hosts: {json.dumps(obs.host_compromise)}. "
        f"Isolated hosts: {json.dumps(obs.host_isolation)}. "
        f"Service status: {json.dumps(obs.service_status)}. "
        f"IDS alerts: {json.dumps(obs.ids_alerts)}. "
        f"Traffic anomaly: {obs.traffic_anomaly_score}."
    )


def choose_action_with_openai(
    client: OpenAI,
    model: str,
    obs: CyberObservation,
    deterministic_source_ip: str,
    seed: int,
) -> CyberAction:
    response = client.responses.create(
        model=model,
        temperature=0,
        top_p=1,
        seed=seed,
        input=[
            {"role": "system", "content": "Return only valid JSON. No markdown."},
            {"role": "user", "content": _build_prompt(obs)},
        ],
    )
    text = (response.output_text or "").strip()
    try:
        payload = json.loads(text)
    except Exception:
        payload = None

    if payload is None:
        # Robust extraction in case model wraps JSON in prose.
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                payload = json.loads(match.group(0))
            except Exception:
                payload = None

    try:
        if payload is None:
            raise ValueError("missing valid JSON payload")
        action = CyberAction(
            actor="defender",
            action_type=payload.get("action_type", "ignore"),
            target_host=payload.get("target_host")
            or next(iter(obs.host_compromise.keys())),
            target_service=payload.get("target_service"),
            source_ip=payload.get("source_ip") or deterministic_source_ip,
        )
        return action
    except Exception:
        fallback = scripted_defender_policy(obs)
        fallback.source_ip = deterministic_source_ip
        return fallback


def evaluate_task(task_id: str, client: OpenAI, model: str, seed: int) -> Dict:
    task = get_task(task_id)
    sim = CyberSimulator(SimulatorConfig(task_id=task_id, seed=seed))
    grader = TaskGrader()

    obs = sim.reset(seed=seed)
    steps = 0
    final_info: Dict = {}

    while not sim.done:
        step_seed = seed + steps
        action = choose_action_with_openai(
            client,
            model,
            obs,
            deterministic_source_ip="10.0.0.45",
            seed=step_seed,
        )
        obs, _, _, final_info = sim.step(action)
        steps += 1

    grade: TaskGrade = grader.score(task, sim.state(), steps)
    metrics = summarize_metrics(final_info.get("metrics", {}), steps, task.max_steps)

    return {
        "task_id": task_id,
        "difficulty": task.difficulty,
        "seed": seed,
        "steps": steps,
        "score": grade.score,
        "grade_breakdown": grade.breakdown,
        "metrics": metrics.__dict__,
        "terminal_reason": final_info.get("terminal_reason"),
    }


def run_baseline(model: str, output_path: Path) -> Dict:
    client = OpenAI(api_key=_resolve_api_key())
    results = [
        evaluate_task("easy", client, model, TASKS["easy"].seed),
        evaluate_task("medium", client, model, TASKS["medium"].seed),
        evaluate_task("hard", client, model, TASKS["hard"].seed),
    ]

    grades = [TaskGrade(score=r["score"], breakdown=r["grade_breakdown"]) for r in results]
    payload = {
        "model": model,
        "deterministic": True,
        "credential_source": "HF_TOKEN",
        "tasks": results,
        "aggregate_score": aggregate_grades(grades),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenAI baseline inference across all tasks")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--output", default="outputs/evals/baseline_scores.json")
    args = parser.parse_args()

    payload = run_baseline(args.model, Path(args.output))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
