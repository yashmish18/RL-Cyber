---
title: Cyber OpenEnv RL
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# Cyber OpenEnv RL: Multi-Agent Autonomous Cyber Defense

Research-grade OpenEnv environment for autonomous cyber defense using reinforcement learning. This project models SOC workflows (alert triage, containment, patching) as a sequential decision process with attacker-defender interaction.

## Why This Environment

- Models realistic SOC tasks instead of game-like toy domains.
- Supports deterministic graded tasks (easy -> medium -> hard) with reproducible seeds.
- Supports multi-agent interaction: scripted attacker/defender and optional RL attacker.
- Works with OpenEnv server/client APIs and Hugging Face Space container deployment.
- Includes a sim-to-real defensive deployment path with guardrails and human approval gates.

## Architecture

- `CyberSimulator` (core): deterministic cyber world transitions and reward shaping.
- `CyberEnvironment` (OpenEnv server): exposes `reset`, `step`, `state` endpoints.
- `DefenderGymEnv` / `AttackerGymEnv`: RL wrappers for PPO/DQN training.
- `TaskGrader`: deterministic score in `[0.0, 1.0]` with transparent decomposition.

## Observation / Action / State Spaces

### Observation (`CyberObservation`)
- `host_compromise`: host -> compromised bool
- `host_isolation`: host -> isolated bool
- `service_status`: host -> service patched/healthy bool
- `ids_alerts`: simulated SIEM/IDS alert strings
- `traffic_anomaly_score`: float `[0, 1]`
- `active_incidents`: typed incident list
- `reward_signal`: typed reward decomposition (`RewardSignal`)
- `step_budget_remaining`: remaining episode budget

### Action (`CyberAction`)
- `actor`: `defender` or `attacker`
- `action_type`:
  - defender: `block_ip`, `isolate_node`, `scan_host`, `patch_service`, `ignore`
  - attacker: `lateral_move`, `credential_stuff`, `malware_drop`, `recon`, `idle`
- `target_host`, optional `target_service`, optional `source_ip`

### State (`CyberState`)
- Full internal world state including attack graph edges, cooldowns, action history, cumulative reward, detection/prevention counters.

## Tasks (Deterministic + Graded)

1. Easy (`seed=101`): single-host intrusion triage and containment.
2. Medium (`seed=202`): lateral movement in subnet with patch/isolation tradeoffs.
3. Hard (`seed=303`): multi-stage campaign with decoy signals and constrained budget.

Each task has:
- fixed scenario config and seed,
- objective contract,
- deterministic grader returning score in `[0.0, 1.0]`.

## Reward Function (Incremental)

Reward includes:
- positive: prevention, detection, containment.
- negative: spread penalties, collateral penalties, loop/no-op penalties.

This provides feedback throughout trajectories, not only terminal reward.
Reward is typed and auditable via `RewardSignal` in both observation and step info.

## Setup

```bash
cd C:\Users\yashm\Documents\Playground\cyber-openenv-rl
python -m pip install -e .[dev]
```

## Run and Validate

### OpenEnv structure validation

```bash
openenv validate .
```

### Start local server

```bash
python -m server.app --port 8000
```

### Runtime API validation

```bash
openenv validate --url http://localhost:8000
```

## RL Training

### Defender PPO

```bash
python -m cyber_openenv_rl.rl.train_defender --algorithm ppo --task medium --timesteps 10000 --device cuda
```

### Defender DQN

```bash
python -m cyber_openenv_rl.rl.train_defender --algorithm dqn --task medium --timesteps 10000 --device cuda
```

### Optional attacker PPO

```bash
python -m cyber_openenv_rl.rl.train_attacker --task hard --timesteps 12000
```

### Curriculum Training (easy -> medium -> hard)

```bash
python -m cyber_openenv_rl.rl.train_curriculum --algorithm ppo --timesteps-per-task 8000 --device cuda
```

Training outputs include:
- checkpoints (`outputs/models/**/checkpoints`)
- best-model snapshots (`outputs/models/**/best`)
- eval logs (`outputs/models/**/eval_logs`)
- deterministic summary JSON per run (`summary_*.json`)

## Benchmarking and Leaderboard

Run multi-seed benchmark suite (deterministic evaluation) and generate JSON + Markdown reports:

```bash
python -m cyber_openenv_rl.eval.benchmark_suite --algorithms ppo,dqn --seeds 42,1337,2026 --timesteps 3000 --output outputs/evals/benchmark_results.json --train
```

Artifacts:
- `outputs/evals/benchmark_results.json`
- `outputs/evals/benchmark_report.md`

## Baseline Inference (OpenAI API Client)

Reads credentials from `HF_TOKEN` (required by hackathon spec).

```bash
set HF_TOKEN=your_api_key_here
python -m cyber_openenv_rl.eval.baseline_inference --model gpt-4o-mini
```

Output is written to `outputs/evals/baseline_scores.json` with per-task and aggregate scores.
The baseline uses fixed task seeds and deterministic request settings (`temperature=0`, `top_p=1`, request seed).

## Real Deployment Path (Defensive-Only)

This project is designed for real-world **defense** deployment in stages:

1. Train in simulation (curriculum)
2. Validate with deterministic tasks and graders
3. Run in shadow mode on real telemetry (no automatic enforcement)
4. Assisted mode (recommendations + human approval)
5. Limited auto-remediation for allowlisted actions only

### Are we training in a virtual environment?

Yes. Training is done in the simulated cyber environment (`CyberSimulator` + Gym wrappers), which is required for safe RL exploration.

### Can I use it outside the simulator?

Yes, for defensive workflows. Use the real-time deployment path:
- ingest live telemetry JSON from SIEM/EDR pipelines,
- run policy inference,
- enforce guardrails and human approval,
- execute only allowlisted defensive actions.

Current implementation supports real-time defensive recommendations with confidence scores and guardrail enforcement. You can integrate your own execution connector for production remediation actions.

### 1) Train a production candidate

```bash
python -m cyber_openenv_rl.rl.train_curriculum --algorithm ppo --timesteps-per-task 15000 --device cuda
```

### 1b) Real-data calibrated training (3-hour run)

Uses NSL-KDD intrusion dataset to calibrate simulator threat profile, then trains for wall-clock duration.

```bash
python -m cyber_openenv_rl.rl.train_real_data --algorithm ppo --task hard --hours 3 --chunk-timesteps 12000 --seed 42 --device cuda --output-dir outputs/models/real_data
```

If CUDA is unavailable in your PyTorch install, the script will raise an error so you can fix your CUDA setup first.

### 2) Run real-time defensive inference on telemetry

```bash
python -m cyber_openenv_rl.deployment.run_realtime_defense ^
  --model-path outputs/models/curriculum/hard/defender_ppo_hard.zip ^
  --algorithm ppo ^
  --task hard ^
  --telemetry data/sample_telemetry/incident_001.json ^
  --output outputs/evals/realtime_inference.json ^
  --confidence-threshold 0.55
```

### 3) Enforce policy guardrails

Guardrails are configured in:
- `configs/production_policy.yaml`

Guardrails block non-defensive/offensive behavior and can require approval for disruptive actions (`block_ip`, `isolate_node`).

## Important Safety and Reality Notes

- This is a **real research/deployment framework**, but not a drop-in replacement for enterprise EDR/SIEM.
- It is intentionally defensive-only; offensive real-world attack automation is not provided.
- “State-of-the-art” is benchmark-dependent. To claim SOTA, benchmark this against strong baselines on public datasets and report statistical significance.

## Codebase Index

To regenerate the project-wide index:

```bash
python tools/generate_codebase_index.py
```

Generated file:
- `CODEBASE_INDEX.md`

## Baseline Scores (Current Reproducible Scripted Baseline)

Generated from fixed-seed scripted defender (`python -m cyber_openenv_rl.eval.scripted_baseline`):

| Task | Score |
|---|---:|
| easy | 0.5050 |
| medium | 0.3350 |
| hard | 0.5729 |
| aggregate | 0.4710 |

## Tests

```bash
pytest -q
```

Test coverage includes:
- transition/reward behavior,
- invalid action handling,
- determinism,
- grader bounds,
- PPO/DQN smoke training,
- `openenv validate` command pass.

## Docker / HF Space

### Build

```bash
docker build -t cyber-openenv-rl -f server/Dockerfile .
```

### Run

```bash
docker run --rm -p 8000:8000 cyber-openenv-rl
```

Tag your Space with `openenv` and use `openenv push` for deployment.
