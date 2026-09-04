#!/usr/bin/env python3
"""Materialize a zero-API V5.3 six-baseline fairness scaffold.

With ``--states-jsonl`` the input rows must match ``FrozenResponseState``.
Without it, a deterministic four-state fixture is used to test the planner and
write an independently inspectable pre-outcome report.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from metacom_pm.v1_5_v5_3_response_baselines import (  # noqa: E402
    BaselineFreeze,
    FrozenResponseState,
    build_response_baseline_plan,
)


PROTOCOL = "pm-v1.5-v5.3-response-baseline-fairness-scaffold-v1"
DEFAULT_OUTPUT = ROOT / "outputs/pm_v1_5_v5_3_response_baseline_fairness_scaffold_v1"


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _fixture() -> list[FrozenResponseState]:
    common = {
        "stratum": "fixture_external_replication",
        "eligible_bits": {"MP": True, "MS": True, "ME": True, "RS": True},
        "candidate_ids": {
            "MP": "mem_mp_fixture",
            "MS": "mem_ms_fixture",
            "ME": "mem_me_fixture",
            "RS": "strat_rs_fixture",
        },
        "incremental_tokens": {"MP": 20, "MS": 30, "ME": 40, "RS": 25},
        "transparent_rule_bits": {
            "MP": False,
            "MS": True,
            "ME": False,
            "RS": True,
        },
    }
    learned = (
        {"MP": True, "MS": False, "ME": False, "RS": True},
        {"MP": False, "MS": True, "ME": True, "RS": False},
        {"MP": False, "MS": False, "ME": True, "RS": True},
        {"MP": True, "MS": True, "ME": False, "RS": False},
    )
    result = [
        FrozenResponseState(
            state_id=f"fixture_state_{index}",
            candidate_snapshot_sha256=f"candidate-snapshot-{index}",
            step2_shared_surface_sha256=f"step2-surface-{index}",
            learned_pm_bits=bits,
            **common,
        )
        for index, bits in enumerate(learned, start=1)
    ]
    # A deliberately different cost cell with learned=off demonstrates that
    # logical arms are retained even when off/learned/matched-random alias.
    result.append(
        FrozenResponseState(
            state_id="fixture_state_5_alias",
            stratum="fixture_external_replication",
            candidate_snapshot_sha256="candidate-snapshot-5",
            step2_shared_surface_sha256="step2-surface-5",
            eligible_bits=common["eligible_bits"],
            candidate_ids=common["candidate_ids"],
            incremental_tokens={"MP": 21, "MS": 31, "ME": 41, "RS": 26},
            transparent_rule_bits=common["transparent_rule_bits"],
            learned_pm_bits={
                "MP": False,
                "MS": False,
                "ME": False,
                "RS": False,
            },
        )
    )
    return result


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cost-matched-action",
        action="append",
        default=[],
        metavar="STRATUM=ACTION",
        help="Repeat for each stratum; fixture defaults to M0+RS.",
    )
    args = parser.parse_args()
    states = (
        [FrozenResponseState(**row) for row in _jsonl(args.states_jsonl)]
        if args.states_jsonl
        else _fixture()
    )
    cost_actions: dict[str, str] = {}
    for item in args.cost_matched_action:
        if "=" not in item:
            raise ValueError("--cost-matched-action must be STRATUM=ACTION")
        stratum, action = item.split("=", 1)
        cost_actions[stratum] = action
    if not cost_actions and not args.states_jsonl:
        cost_actions = {"fixture_external_replication": "M0+RS"}
    freeze = BaselineFreeze(
        protocol=PROTOCOL,
        seed_labels=("seed-a", "seed-b"),
        cost_matched_fixed_action_by_stratum=cost_actions,
        matched_random_seed=PROTOCOL + ":matched-random",
    )
    plan = build_response_baseline_plan(states, freeze=freeze)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "logical_policy_bindings.jsonl", plan["logical_bindings"])
    _write_jsonl(args.output_dir / "physical_call_plan.jsonl", plan["physical_calls"])
    (args.output_dir / "report.json").write_text(
        json.dumps(plan["report"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "freeze_config.json").write_text(
        json.dumps(asdict(freeze), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plan["report"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
