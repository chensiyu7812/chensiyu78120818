#!/usr/bin/env python3
"""Prepare the train-only low-budget judge qualification packet (zero API)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from metacom_pm.io import (
    canonical_json,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.pm_v2_contracts import PMV2State
from metacom_pm.pm_v2_judging import (
    build_response_messages,
    build_risk_messages,
)
from metacom_pm.v1_5_judge_qualification import (
    GPT_ANCHOR_REGIMES,
    PAIRWISE_ORDER_VARIANTS,
    build_combined_messages,
    build_pairwise_messages,
    qualification_contract_record,
    reverse_pair,
    select_qualification_pairs,
)
from metacom_pm.v1_5_judge_qualification_analysis import (
    require_tracked_human_anchor,
)


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_TRAIN_STATES = 216


def _read_exact_prefix(
    path: Path, *, expected_rows: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read only a frozen train prefix, never later split rows."""

    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    raw_lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for index in range(expected_rows):
            line = handle.readline()
            if not line:
                raise RuntimeError(
                    f"{path} ended before frozen train prefix row {index}"
                )
            if not line.strip():
                raise RuntimeError(
                    f"{path} has a blank row inside its frozen train prefix"
                )
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"{path} train-prefix row {index} is not an object"
                )
            rows.append(dict(value))
            raw_lines.append(line)
    return rows, {
        "rows_consumed": len(rows),
        "serialized_prefix_sha256": sha256_text("".join(raw_lines)),
        "canonical_content_sha256": sha256_text(canonical_json(rows)),
    }


def _endpoint_contract(path: Path) -> dict[str, Any]:
    record = read_json(path)
    if record.get("protocol") != "pm-v1.5-low-budget-judge-endpoints-v2":
        raise RuntimeError("unexpected judge qualification endpoint contract")
    request = dict(record.get("request_contract") or {})
    if int(request.get("maximum_physical_attempts_per_logical_call", 0)) != 2:
        raise RuntimeError("qualification must retain the frozen two-attempt ceiling")
    return {
        **record,
        "file_sha256": sha256_file(path),
        "content_sha256": sha256_text(canonical_json(record)),
    }


def _candidate_b_absolute_item(
    pair: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "visible_state": dict(pair["visible_state"]),
        "authorized_user_context": str(pair["authorized_user_context"]),
        "candidate_b": dict(pair["candidate_b"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--states",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate/"
        "pm_v2_states.jsonl",
    )
    parser.add_argument(
        "--evaluator-contexts",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate/"
        "evaluator_contexts.jsonl",
    )
    parser.add_argument(
        "--action-outcomes",
        type=Path,
        default=ROOT
        / "outputs/"
        "pm_v1_5_longitudinal_action_sweep_v8_19_2_continuation_v2_dry_run/"
        "action_outcomes.jsonl",
    )
    parser.add_argument(
        "--endpoint-contract",
        type=Path,
        default=ROOT / "configs/pm_v1_5_judge_qualification_v2.json",
    )
    parser.add_argument(
        "--human-anchor-binding",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/low_budget_judge_human_anchor_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_low_budget_judge_qualification_packet_v2",
    )
    args = parser.parse_args()

    # The canonical development files are frozen in train/calibration/
    # internal-test order.  Read the exact train prefixes and stop before any
    # later-split row.  In particular, no calibration/internal-test outcome
    # text is parsed or hashed by this preparation stage.
    train_states, state_prefix = _read_exact_prefix(
        args.states, expected_rows=EXPECTED_TRAIN_STATES
    )
    if any(str(row.get("split")) != "train" for row in train_states):
        raise RuntimeError("frozen state prefix is not exclusively train")
    train_state_ids = {str(row["state_id"]) for row in train_states}
    if len(train_state_ids) != EXPECTED_TRAIN_STATES:
        raise RuntimeError("frozen train state IDs are not unique")

    train_evaluators, evaluator_prefix = _read_exact_prefix(
        args.evaluator_contexts, expected_rows=EXPECTED_TRAIN_STATES
    )
    if {str(row["state_id"]) for row in train_evaluators} != train_state_ids:
        raise RuntimeError("evaluator train prefix does not match train states")

    expected_train_outcomes = sum(
        len(list(row["allowed_actions"])) for row in train_states
    )
    train_outcomes, outcome_prefix = _read_exact_prefix(
        args.action_outcomes, expected_rows=expected_train_outcomes
    )
    outcome_keys = {
        (str(row["state_id"]), str(row["action_id"]))
        for row in train_outcomes
    }
    expected_outcome_keys = {
        (str(state["state_id"]), str(action_id))
        for state in train_states
        for action_id in state["allowed_actions"]
    }
    if len(outcome_keys) != len(train_outcomes):
        raise RuntimeError("train outcome prefix has duplicate state/action keys")
    if outcome_keys != expected_outcome_keys:
        raise RuntimeError("train outcome prefix coverage drifted")
    pairs = select_qualification_pairs(
        state_rows=train_states,
        evaluator_rows=train_evaluators,
        outcome_rows=train_outcomes,
    )
    state_models = {
        str(row["state_id"]): PMV2State.model_validate(row, strict=False)
        for row in train_states
    }

    ordered_pairwise: list[dict[str, Any]] = []
    for pair in pairs:
        for order in PAIRWISE_ORDER_VARIANTS:
            ordered = dict(pair) if order == 0 else reverse_pair(pair)
            ordered_pairwise.append(
                {
                    "pair_id": str(pair["pair_id"]),
                    "state_id": str(pair["state_id"]),
                    "user_id": str(pair["user_id"]),
                    "regime": str(pair["regime"]),
                    "order_variant": order,
                    "proxy_expected_winner": (
                        str(pair["proxy_expected_winner"])
                        if order == 0
                        else {
                            "A": "B",
                            "B": "A",
                            "unknown": "unknown",
                        }[str(pair["proxy_expected_winner"])]
                    ),
                    "messages": build_pairwise_messages(ordered),
                }
            )

    by_regime: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        by_regime.setdefault(str(pair["regime"]), pair)
    if len(by_regime) != 9:
        raise RuntimeError("split-equivalence anchors must cover all nine regimes")
    equivalence_rows: list[dict[str, Any]] = []
    for regime in sorted(by_regime):
        pair = by_regime[regime]
        absolute = _candidate_b_absolute_item(pair)
        state = state_models[str(pair["state_id"])]
        selected_context = canonical_json(
            pair["candidate_b"]["selected_context"]
        )
        response = str(pair["candidate_b"]["response"])
        equivalence_rows.append(
            {
                "equivalence_id": "eq_" + str(pair["pair_id"])[3:],
                "pair_id": str(pair["pair_id"]),
                "state_id": str(pair["state_id"]),
                "user_id": str(pair["user_id"]),
                "regime": regime,
                "combined_messages": build_combined_messages(absolute),
                "quality_messages": build_response_messages(
                    state=state,
                    authorized_user_context=str(
                        pair["authorized_user_context"]
                    ),
                    candidate_response=response,
                ),
                "risk_messages": build_risk_messages(
                    state=state,
                    authorized_user_context=str(
                        pair["authorized_user_context"]
                    ),
                    selected_context=selected_context,
                    candidate_response=response,
                ),
            }
        )

    gpt_anchor_pair_ids = {
        str(by_regime[regime]["pair_id"])
        for regime in GPT_ANCHOR_REGIMES
    }
    human_packet: list[dict[str, Any]] = []
    human_template: list[dict[str, Any]] = []
    for pair in pairs:
        # One deterministic order per item, balanced 6/6 without carrying
        # hidden expected directions into the human packet.
        index = len(human_packet)
        ordered = dict(pair) if index % 2 == 0 else reverse_pair(pair)
        blind_id = "human_" + sha256_text(str(pair["pair_id"]))[:16]
        human_packet.append(
            {
                "blind_item_id": blind_id,
                "visible_state": dict(ordered["visible_state"]),
                "authorized_user_context": str(
                    ordered["authorized_user_context"]
                ),
                "candidate_a": dict(ordered["candidate_a"]),
                "candidate_b": dict(ordered["candidate_b"]),
            }
        )
        human_template.append(
            {
                "blind_item_id": blind_id,
                "overall_preference": "",
                "support_quality_preference": "",
                "evidence_handling_preference": "",
                "safety_preference": "",
                "confidence": "",
                "notes": "",
            }
        )

    endpoint_contract = _endpoint_contract(args.endpoint_contract)
    human_anchor = require_tracked_human_anchor(
        root=ROOT,
        binding_path=args.human_anchor_binding,
        pairs=pairs,
        human_packet=human_packet,
    )
    source_lineage = {
        "states_path": str(args.states.relative_to(ROOT)),
        "states_train_prefix": state_prefix,
        "evaluator_contexts_path": str(
            args.evaluator_contexts.relative_to(ROOT)
        ),
        "evaluator_contexts_train_prefix": evaluator_prefix,
        "action_outcomes_path": str(
            args.action_outcomes.relative_to(ROOT)
        ),
        "action_outcomes_train_prefix": outcome_prefix,
        "train_state_count": len(train_states),
        "train_outcome_count": len(train_outcomes),
        "selected_pairs_sha256": sha256_text(canonical_json(pairs)),
        "ordered_pairwise_sha256": sha256_text(
            canonical_json(ordered_pairwise)
        ),
        "split_equivalence_sha256": sha256_text(
            canonical_json(equivalence_rows)
        ),
        "human_blind_packet_sha256": sha256_text(
            canonical_json(human_packet)
        ),
        "gpt_anchor_pair_ids_sha256": sha256_text(
            canonical_json(sorted(gpt_anchor_pair_ids))
        ),
    }
    contract = qualification_contract_record(
        source_lineage=source_lineage,
        endpoint_contract=endpoint_contract,
        human_anchor=human_anchor,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "qualification_pairs.jsonl", pairs)
    write_jsonl(
        args.out_dir / "ordered_pairwise_prompts.jsonl",
        ordered_pairwise,
    )
    write_jsonl(
        args.out_dir / "split_equivalence_prompts.jsonl",
        equivalence_rows,
    )
    write_jsonl(args.out_dir / "human_blind_packet.jsonl", human_packet)
    write_jsonl(
        args.out_dir / "human_annotation_template.jsonl",
        human_template,
    )
    write_json(
        args.out_dir / "gpt_anchor_pair_ids.json",
        {"pair_ids": sorted(gpt_anchor_pair_ids)},
    )
    write_json(args.out_dir / "qualification_contract.json", contract)
    write_json(
        args.out_dir / "preparation_summary.json",
        {
            "status": "COMPLETE_ZERO_API",
            "protocol": contract["protocol"],
            "contract_sha256": contract["contract_sha256"],
            "train_pairs": len(pairs),
            "ordered_pairwise_items": len(ordered_pairwise),
            "split_equivalence_items": len(equivalence_rows),
            "gpt_anchor_pairs": len(gpt_anchor_pair_ids),
            "human_blind_items": len(human_packet),
            "api_calls_made": 0,
            "training_labels_created": False,
            "internal_test_or_external_outcomes_consumed": False,
        },
    )
    print(
        canonical_json(
            {
                "status": "COMPLETE_ZERO_API",
                "contract_sha256": contract["contract_sha256"],
                "out_dir": str(args.out_dir),
            }
        )
    )


if __name__ == "__main__":
    main()
