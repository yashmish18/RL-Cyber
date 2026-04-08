---
title: RL Cyber Environment (OpenEnv)
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
tags:
  - openenv
  - cybersecurity
---

# RL Cyber Environment (OpenEnv)

A high-fidelity reinforcement learning environment for cybersecurity server hardening and incident response, compliant with the OpenEnv specification.

## Overview
This environment simulates a Security Operations Center (SOC) managing a multi-host network under attack from scripted threat actors. Agents must observe indicators of compromise (IDS alerts, traffic anomalies) and take actions to contain threats and restore services.

## Tasks
1. **Easy**: Single-host brute-force triage. Objective: Identify and contain entry before lateral movement.
2. **Medium**: Subnet lateral movement. Objective: Prevent compromise spread to data tier while preserving uptime.
3. **Hard**: Multi-stage campaign. Objective: Contain advanced persistence and protect crown-jewel data.

## Action & Observation Space
### Actions
- `block_ip`: Block an external IP address.
- `isolate_node`: Network-level isolation of a host.
- `scan_host`: Diagnostic scan for confirmed compromise.
- `patch_service`: Fix security vulnerabilities.
- `restore_backup`: Restore host to a clean state.
- `ignore`: Passive monitoring.

### Observations
- `host_compromise`: Known compromise status (partial observability).
- `ids_alerts`: Stream of intrusion detection signals.
- `traffic_anomaly_score`: Statistical deviation in network traffic.
- `active_incidents`: Formally identified security incidents.

## Baseline Scores
Evaluated using `Qwen/Qwen2.5-72B-Instruct`:
- **Easy**: 0.45
- **Medium**: 0.00
- **Hard**: 0.00

## Setup & Usage
### Docker Deployment
```bash
docker build -t rl-cyber .
docker run -p 8000:8000 rl-cyber
```

### Local Development
1. Install dependencies: `pip install -r server/requirements.txt`
2. Start server: `python -m server.app`
3. Run evaluation: `python inference.py`

## OpenEnv Compliance
This project implements the full OpenEnv spec:
- Typed Pydantic models in `models.py`.
- Task definitions in `tasks.py`.
- Graders in `grading.py`.
- Validated via `openenv validate`.
