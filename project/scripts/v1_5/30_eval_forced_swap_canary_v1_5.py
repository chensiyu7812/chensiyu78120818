#!/usr/bin/env python3
"""PM-v1.5 defaults for the shared forced-swap evaluator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION = ROOT / "scripts" / "30_eval_pm_v2_forced_swap.py"


def _default(argv: list[str], flag: str, *values: Path) -> list[str]:
    if flag in argv:
        return argv
    return [*argv, flag, *(str(value) for value in values)]


def main() -> None:
    argv = list(sys.argv)
    argv = _default(argv, "--pm-v2-config", ROOT / "configs" / "pm_v1_5.yaml")
    argv = _default(argv, "--freeze", ROOT / "outputs" / "pm_v1_5_study_freeze.json")
    argv = _default(
        argv,
        "--turn-paths",
        ROOT / "outputs" / "evoemo_pm_v1_5" / "turns.jsonl",
        ROOT / "outputs" / "evoemo_pm_v1_5_cost_matched_fixed" / "turns.jsonl",
    )
    argv = _default(
        argv,
        "--generation-attestations",
        ROOT / "outputs" / "evoemo_pm_v1_5" / "artifact_attestation.json",
        ROOT
        / "outputs"
        / "evoemo_pm_v1_5_cost_matched_fixed"
        / "artifact_attestation.json",
    )
    argv = _default(
        argv, "--out-dir", ROOT / "outputs" / "pm_v1_5_forced_swap_canary"
    )
    spec = importlib.util.spec_from_file_location(
        "pm_v1_5_forced_swap_implementation", IMPLEMENTATION
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load forced-swap implementation: {IMPLEMENTATION}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous = sys.argv
    try:
        sys.argv = argv
        module.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
