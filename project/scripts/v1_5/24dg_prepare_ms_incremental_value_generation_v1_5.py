#!/usr/bin/env python3
"""Freeze the 80-call MS incremental-value generation plan."""

from __future__ import annotations

import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-ms-incremental-value-generation-plan-v1"
BLUEPRINT_PROTOCOL = "pm-v1.5-ms-incremental-value-blueprint-v1"
BLUEPRINT_STATUS = "PASS_READY_TO_PREPARE_80_MS_INCREMENTAL_VALUE_CALLS"
STATUS = "FROZEN_READY_FOR_80_MS_INCREMENTAL_VALUE_CALLS"


def main() -> None:
    legacy = runpy.run_path(
        str(ROOT / "scripts/v1_5/24cu_prepare_d3_ms_replacement_generation_v1_5.py")
    )
    prepare = legacy["prepare"]
    globals_ = prepare.__globals__
    globals_.update(
        {
            "PROTOCOL": PROTOCOL,
            "BLUEPRINT_PROTOCOL": BLUEPRINT_PROTOCOL,
            "BLUEPRINT_STATUS": BLUEPRINT_STATUS,
            "STATUS": STATUS,
        }
    )
    report = prepare(
        blueprint_dir=ROOT / "outputs/pm_v1_5_ms_incremental_value_step0_v1",
        strategy_cards_path=ROOT / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl",
        pm_config_path=ROOT / "configs/pm_v1_5.yaml",
        experiment_config_path=ROOT / "configs/experiment.yaml",
        out_dir=ROOT / "outputs/pm_v1_5_ms_incremental_value_generation_v1",
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "protocol",
                    "status",
                    "pairs",
                    "planned_calls",
                    "cost_upper_bound_usd",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
