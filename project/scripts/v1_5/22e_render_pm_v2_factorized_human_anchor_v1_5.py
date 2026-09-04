#!/usr/bin/env python3
"""Render the factorized weak-learning human anchor as local HTML."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from metacom_pm.io import iter_jsonl


ROOT = Path(__file__).resolve().parents[2]
SHARED_RENDERER = (
    ROOT / "scripts" / "v1_5" / "21t_render_low_budget_judge_human_packet_v1_5.py"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    items = [dict(row) for row in iter_jsonl(args.packet)]
    if len(items) != 32 or len(
        {str(row["blind_item_id"]) for row in items}
    ) != 32:
        raise RuntimeError("factorized human anchor must contain 32 unique items")
    spec = importlib.util.spec_from_file_location(
        "pm_v1_5_shared_human_renderer", SHARED_RENDERER
    )
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("could not load shared human packet renderer")
    spec.loader.exec_module(module)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(module.render(items), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
