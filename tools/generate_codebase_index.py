from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "CODEBASE_INDEX.md"

INCLUDE_SUFFIX = {".py", ".md", ".toml", ".yaml", ".yml", ".json"}
IGNORE_PARTS = {".git", ".pytest_cache", "__pycache__", "openenv_cyber_openenv_rl.egg-info"}


def should_include(path: Path) -> bool:
    if any(part in IGNORE_PARTS for part in path.parts):
        return False
    if path.is_dir():
        return False
    if path.suffix.lower() not in INCLUDE_SUFFIX:
        return False
    return True


def section(path: Path) -> str:
    rel = path.relative_to(ROOT)
    if rel.parts[0] in {"cyber_openenv_rl", "server", "tests", "configs", "data", "tools"}:
        return rel.parts[0]
    return "root"


def main() -> None:
    files = sorted([p for p in ROOT.rglob("*") if should_include(p)])
    buckets = {}
    for p in files:
        buckets.setdefault(section(p), []).append(p)

    lines = [
        "# Codebase Index",
        "",
        "Auto-generated file map for quick repository navigation.",
        "",
    ]

    for key in sorted(buckets.keys()):
        lines.append(f"## {key}")
        lines.append("")
        for p in buckets[key]:
            rel = p.relative_to(ROOT)
            try:
                line_count = sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore"))
            except Exception:
                line_count = -1
            line_text = f"- `{rel.as_posix()}`"
            if line_count >= 0:
                line_text += f" ({line_count} lines)"
            lines.append(line_text)
        lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
