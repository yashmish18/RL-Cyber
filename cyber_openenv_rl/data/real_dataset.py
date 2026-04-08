from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict

import requests

# Public NSL-KDD mirror (intrusion detection benchmark dataset)
NSL_KDD_TRAIN_URL = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain+.txt"


def download_nsl_kdd(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest

    response = requests.get(NSL_KDD_TRAIN_URL, timeout=60)
    response.raise_for_status()
    dest.write_text(response.text, encoding="utf-8")
    return dest


def build_threat_profile(dataset_path: Path) -> Dict[str, float]:
    """Build simulator threat profile from real intrusion data stats."""
    total = 0
    attacks = 0
    brute_force = 0
    lateralish = 0

    with dataset_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            total += 1
            label = row[-2].strip().lower() if len(row) >= 2 else "normal"
            if label != "normal":
                attacks += 1
            if label in {"guess_passwd", "ftp_write", "imap", "phf", "warezmaster"}:
                brute_force += 1
            if label in {"ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"}:
                lateralish += 1

    if total == 0:
        return {
            "credential_stuff_success_threshold": 0.35,
            "lateral_move_success_threshold": 0.25,
        }

    attack_rate = attacks / total
    brute_rate = brute_force / max(1, attacks)
    lateral_rate = lateralish / max(1, attacks)

    # Convert observed rates to conservative simulator thresholds
    cred_threshold = max(0.10, min(0.70, 0.55 - (attack_rate * 0.40) - (brute_rate * 0.15)))
    lat_threshold = max(0.08, min(0.65, 0.50 - (attack_rate * 0.35) - (lateral_rate * 0.20)))

    return {
        "credential_stuff_success_threshold": float(cred_threshold),
        "lateral_move_success_threshold": float(lat_threshold),
        "stats": {
            "total_rows": total,
            "attack_rows": attacks,
            "attack_rate": attack_rate,
            "brute_force_attack_rate": brute_rate,
            "lateral_attack_rate": lateral_rate,
        },
    }


def save_profile(profile: Dict, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return output
