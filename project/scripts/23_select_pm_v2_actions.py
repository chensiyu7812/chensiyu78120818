#!/usr/bin/env python3
"""Deprecated unsafe PM-v2 selection-only entry point.

Selection over caller-supplied runtime rows cannot prove that the rows, model,
retrieval contract, and action gates belong to the frozen external study.  The
replacement is the no-API dry-run of ``24_run_pm_v2_evoemo.py``; it constructs
the exact frozen states, evaluates every decision, enforces all preflight gates,
and writes the immutable generation call plan before any client is created.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deprecated: use scripts/24_run_pm_v2_evoemo.py --dry-run."
    )
    parser.parse_args()
    raise RuntimeError(
        "scripts/23_select_pm_v2_actions.py is intentionally disabled because "
        "caller-supplied runtime states are not bound to the PM-v2 study freeze. "
        "Run scripts/24_run_pm_v2_evoemo.py --dry-run with the frozen checkpoint, "
        "condition, tracks, pricing, and budget limits instead; it performs the "
        "same no-API action/OOD preflight on the exact external call matrix."
    )


if __name__ == "__main__":
    main()
