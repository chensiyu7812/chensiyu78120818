"""One-off: machine-readable inventory of the dirty working tree before any
git recovery / .gitignore changes are made (2026-08-05).

Purpose: record what exists RIGHT NOW -- path, size, sha256 (for small /
code-like files), aggregate stats (for large directories), and a category
tag -- so nothing gets silently hidden by a later .gitignore edit, and so
later commits can be checked against this list. Read-only: never writes
inside the repo except the two output files it's asked to produce.

Run from the repo root (one level above the `project/` directory).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../metacom_v33_pm_v1_5_repair
OUT_JSON = REPO_ROOT / "project/docs/GIT_RECOVERY_INVENTORY_20260805.json"
OUT_MD = REPO_ROOT / "project/docs/GIT_RECOVERY_INVENTORY_20260805_SUMMARY_ZH.md"

# Individual files at or under this size get a sha256; larger individual
# files inside "unknown"/"generated_artifact"/"private_output" dirs are
# skipped (size-only) to keep this script fast and avoid hashing gigabytes.
HASH_SIZE_LIMIT_BYTES = 5 * 1024 * 1024

PRIVATE_NAME_MARKERS = (
    "private",
    "blind",
    "condition_mapping",
    "physical_attempt_ledger",
    "raw_api_calls",
    "raw_seeker_calls",
    "raw_judge_calls",
)


def sh(*args: str) -> str:
    return subprocess.run(
        args, cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def category_of(rel_path: str, is_dir: bool) -> str:
    p = rel_path
    if p.startswith("project/src/metacom_pm/") and p.endswith(".py"):
        return "source_code"
    if p.startswith("project/scripts/") and p.endswith(".py"):
        return "source_code"
    if p.startswith("project/tests/") and p.endswith(".py"):
        return "test"
    if p.startswith("project/data/pm_v1_5_contracts/") and p.endswith(".json"):
        return "contract"
    if p.startswith("project/docs/") and p.endswith(".md"):
        return "doc"
    if p == "project/README_CN.md":
        return "doc"
    if p.startswith("project/data/strategy/"):
        return "curated_fixture"
    if p.startswith("project/configs/"):
        return "source_code"
    if p.startswith("project/data/external/"):
        return "curated_fixture"
    if any(marker in p.lower() for marker in PRIVATE_NAME_MARKERS):
        return "private_output"
    if p.startswith("project/outputs/") or (p.startswith("project/data/") and is_dir):
        return "generated_artifact"
    return "unknown"


def walk_dir_aggregate(abs_dir: Path) -> dict:
    total_size = 0
    file_count = 0
    top_level = sorted(
        (entry.name + ("/" if entry.is_dir() else "")) for entry in abs_dir.iterdir()
    )
    private_flagged = False
    for f in abs_dir.rglob("*"):
        if f.is_file():
            file_count += 1
            try:
                total_size += f.stat().st_size
            except OSError:
                pass
            if any(marker in f.name.lower() for marker in PRIVATE_NAME_MARKERS):
                private_flagged = True
    return {
        "total_size_bytes": total_size,
        "file_count": file_count,
        "top_level_listing": top_level,
        "contains_private_marker_filenames": private_flagged,
    }


def main() -> None:
    head_sha = sh("git", "rev-parse", "HEAD").strip()
    branch = sh("git", "branch", "--show-current").strip()
    porcelain = sh("git", "status", "--porcelain").splitlines()

    entries = []
    for line in porcelain:
        if not line.strip():
            continue
        status = line[:2]
        rel_path = line[3:]
        abs_path = REPO_ROOT / rel_path
        is_dir = abs_path.is_dir()
        entry: dict = {
            "path": rel_path,
            "git_status": status.strip(),
            "is_dir": is_dir,
            "category": category_of(rel_path, is_dir),
        }
        if is_dir:
            entry.update(walk_dir_aggregate(abs_path))
        else:
            try:
                size = abs_path.stat().st_size
            except OSError:
                size = None
            entry["size_bytes"] = size
            if size is not None and size <= HASH_SIZE_LIMIT_BYTES:
                entry["sha256"] = sha256_file(abs_path)
            else:
                entry["sha256"] = None
                entry["sha256_skipped_reason"] = "file exceeds hash size limit"
        entries.append(entry)

    by_category: dict[str, list[str]] = {}
    for e in entries:
        by_category.setdefault(e["category"], []).append(e["path"])

    manifest = {
        "protocol": "pm-v1.5-git-recovery-inventory-v1",
        "generated_at": "2026-08-05",
        "head_sha": head_sha,
        "branch": branch,
        "note": (
            "Snapshot of `git status --porcelain` taken BEFORE any .gitignore "
            "changes, so nothing was hidden before being catalogued here. "
            "Directories get aggregate stats (size/file_count/top_level_listing), "
            "not per-file hashes -- individual files inside them are not "
            "separately catalogued by this manifest."
        ),
        "entry_count": len(entries),
        "category_counts": {k: len(v) for k, v in sorted(by_category.items())},
        "entries": entries,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# Git 恢复前库存清单（2026-08-05，只读快照）",
        "",
        f"- HEAD: `{head_sha}`",
        f"- 分支: `{branch}`",
        f"- 总条目数: {len(entries)}",
        "",
        "## 按类别统计",
        "",
        "| 类别 | 条目数 |",
        "|---|---:|",
    ]
    for cat, paths in sorted(by_category.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"| {cat} | {len(paths)} |")
    lines.append("")
    lines.append("完整逐条清单见同目录 `GIT_RECOVERY_INVENTORY_20260805.json`。")
    OUT_MD.write_text("\n".join(lines) + "\n")

    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(json.dumps(manifest["category_counts"], indent=2))


if __name__ == "__main__":
    main()
