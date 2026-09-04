#!/usr/bin/env python3
"""Seal the V5.2 confirmation runner and implementation before API use."""

from __future__ import annotations

import json
from pathlib import Path

from metacom_pm.io import read_json, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[2]
PLAN_DIR = ROOT / "outputs/pm_v1_5_v5_2_confirmation_plan_v1"


def main() -> None:
    manifest = read_json(PLAN_DIR / "freeze_manifest.json")
    if (
        manifest.get("status")
        != "READY_FOR_SINGLE_V5_2_CONTENT_DISJOINT_CONFIRMATION_EXECUTION"
    ):
        raise RuntimeError("confirmation plan is not frozen")
    call_path = PLAN_DIR / "call_plan_private.jsonl"
    if sha256_file(call_path) != manifest["call_plan_sha256"]:
        raise RuntimeError("confirmation call plan changed")
    paths = (
        ROOT / "scripts/v1_5/27h_freeze_v5_2_content_disjoint_confirmation_v1_5.py",
        ROOT / "scripts/v1_5/27i_materialize_v5_2_confirmation_plan_v1_5.py",
        ROOT / "scripts/v1_5/27j_run_v5_2_confirmation_generation_v1_5.py",
        ROOT / "src/metacom_pm/v1_5_v5_2_atomic_memory.py",
        ROOT / "src/metacom_pm/v1_5_v5_2_locked_composer.py",
        ROOT / "src/metacom_pm/v1_5_itt_policy.py",
    )
    seal = {
        "protocol": "pm-v1.5-v5.2-content-disjoint-confirmation-execution-seal-v1",
        "status": "SEALED_BEFORE_CONFIRMATION_EXECUTION",
        "call_plan_sha256": manifest["call_plan_sha256"],
        "generator": manifest["generator"],
        "temperature": manifest["temperature"],
        "implementation_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in paths
        },
        "human_generated_confirmation_or_external_outcomes_read": 0,
        "post_execution_method_change_allowed": False,
    }
    path = PLAN_DIR / "execution_seal.json"
    if path.exists():
        if read_json(path) != seal:
            raise RuntimeError("execution seal would change")
    else:
        write_json(path, seal)
    print(json.dumps({"status": seal["status"], "api_calls": 0}))


if __name__ == "__main__":
    main()
