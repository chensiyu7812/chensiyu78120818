#!/usr/bin/env python3
"""Run the paid, dual-family semantic/fallback gate on all 468 real states."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = ROOT / "scripts" / "v1_5_run_automated_semantic_review.py"


def _default(argv: list[str], flag: str, value: str | Path) -> list[str]:
    return argv if flag in argv else [*argv, flag, str(value)]


def main() -> None:
    argv = _default(list(sys.argv), "--review-scope", "actual_468")
    argv = _default(
        argv,
        "--out-dir",
        ROOT / "outputs" / "pm_v1_5_actual_corpus_semantic_review",
    )
    spec = importlib.util.spec_from_file_location(
        "pm_v1_5_actual_corpus_review_implementation", IMPLEMENTATION
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load semantic-review implementation")
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
