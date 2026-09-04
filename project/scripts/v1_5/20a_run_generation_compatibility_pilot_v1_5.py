#!/usr/bin/env python3
"""PM-v1.5 entry point for the one-call generation compatibility pilot.

The paid implementation remains scripts/20a_run_pm_v2_generation_compatibility_pilot.py;
that implementation is version-neutral at the artifact-contract layer.  This
driver only supplies isolated PM-v1.5 defaults, so a V1.5 run cannot silently
reuse a V2.2 config, seed set, or output directory.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION = ROOT / "scripts" / "20a_run_pm_v2_generation_compatibility_pilot.py"


def _with_default(argv: list[str], flag: str, value: Path) -> list[str]:
    if flag in argv:
        return argv
    return [*argv, flag, str(value)]


def main() -> None:
    argv = list(sys.argv)
    if "--run" in argv and "--overwrite" in argv:
        raise RuntimeError(
            "paid PM-v1.5 generation-pilot runs prohibit --overwrite"
        )
    argv = _with_default(argv, "--pm-v2-config", ROOT / "configs" / "pm_v1_5.yaml")
    argv = _with_default(
        argv,
        "--seed-dialogues",
        ROOT / "data" / "pm_v2" / "train_seed_dialogues_v1_5.jsonl",
    )
    argv = _with_default(
        argv,
        "--out-dir",
        ROOT / "outputs" / "pm_v1_5_generation_compatibility_pilot",
    )
    spec = importlib.util.spec_from_file_location(
        "pm_v1_5_generation_compatibility_implementation", IMPLEMENTATION
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load generation-pilot implementation: {IMPLEMENTATION}")
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
