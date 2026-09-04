#!/usr/bin/env python3
"""Prepare fresh dual-domain role-decomposed judge qualification data (zero API)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import (
    canonical_json,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.v1_5_role_decomposed_judge_qualification import (
    PROTOCOL,
    build_measurement_items,
    contract_record,
    human_anchor_mapping,
    human_anchor_items,
    select_esconv_pairs,
    select_longitudinal_pairs,
    validate_endpoint_contract,
)


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_LONGITUDINAL_TRAIN_STATES = 216
EXPECTED_LONGITUDINAL_TRAIN_OUTCOMES = 3456
EXPECTED_ESCONV_TRAIN_STATES = 318
EXPECTED_ESCONV_TRAIN_OUTCOMES = 636


def _read_exact_rows(path: Path, count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    raw: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for index in range(count):
            line = handle.readline()
            if not line or not line.strip():
                raise RuntimeError(f"{path} ended before required row {index}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path} row {index} is not an object")
            rows.append(dict(value))
            raw.append(line)
    return rows, {
        "path": str(path.relative_to(ROOT)),
        "rows": len(rows),
        # Do not hash the whole canonical multi-split file: the bytes after
        # this prefix contain calibration/internal-test rows.  Binding the
        # exact serialized and canonical train prefix is sufficient and keeps
        # the sealed suffix unread.
        "consumed_bytes_sha256": sha256_text("".join(raw)),
        "canonical_content_sha256": sha256_text(canonical_json(rows)),
    }


def _read_all(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [
        dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows, {
        "path": str(path.relative_to(ROOT)),
        "rows": len(rows),
        "file_sha256": sha256_file(path),
        "canonical_content_sha256": sha256_text(canonical_json(rows)),
    }


def _endpoint_contract(path: Path) -> dict[str, Any]:
    record = read_json(path)
    validate_endpoint_contract(record)
    return {
        **record,
        "file_sha256": sha256_file(path),
        "content_sha256": sha256_text(canonical_json(record)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--longitudinal-states",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate/"
        "pm_v2_states.jsonl",
    )
    parser.add_argument(
        "--longitudinal-evaluators",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate/"
        "evaluator_contexts.jsonl",
    )
    parser.add_argument(
        "--longitudinal-outcomes",
        type=Path,
        default=ROOT
        / "outputs/"
        "pm_v1_5_longitudinal_action_sweep_v8_19_2_continuation_v2_dry_run/"
        "action_outcomes.jsonl",
    )
    parser.add_argument(
        "--longitudinal-exclusion-pairs",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_low_budget_judge_qualification_v2_full_schema_v3_packet/"
        "qualification_pairs.jsonl",
    )
    parser.add_argument(
        "--esconv-states",
        type=Path,
        default=ROOT / "data/esconv_auxiliary_v1_5/train/pm_v2_states.jsonl",
    )
    parser.add_argument(
        "--esconv-runtime",
        type=Path,
        default=ROOT / "data/esconv_auxiliary_v1_5/train/runtime_states.jsonl",
    )
    parser.add_argument(
        "--esconv-outcomes",
        type=Path,
        default=ROOT
        / "outputs/esconv_auxiliary_generation_v1_5_full_train/action_outcomes.jsonl",
    )
    parser.add_argument(
        "--esconv-exclusion-selection",
        type=Path,
        default=ROOT
        / "outputs/esconv_auxiliary_pairwise_pilot_v1_5_prep_schema_fix_candidate/"
        "selection.jsonl",
    )
    parser.add_argument(
        "--endpoint-contract",
        type=Path,
        default=ROOT
        / "configs/pm_v1_5_role_decomposed_judge_qualification_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_role_decomposed_judge_qualification_v1_packet",
    )
    args = parser.parse_args()

    long_states, long_states_lineage = _read_exact_rows(
        args.longitudinal_states, EXPECTED_LONGITUDINAL_TRAIN_STATES
    )
    long_evaluators, long_evaluators_lineage = _read_exact_rows(
        args.longitudinal_evaluators, EXPECTED_LONGITUDINAL_TRAIN_STATES
    )
    long_outcomes, long_outcomes_lineage = _read_exact_rows(
        args.longitudinal_outcomes, EXPECTED_LONGITUDINAL_TRAIN_OUTCOMES
    )
    if any(str(row.get("split")) != "train" for row in long_states):
        raise RuntimeError("longitudinal prefix is not train-only")
    long_exclusions, long_exclusions_lineage = _read_all(
        args.longitudinal_exclusion_pairs
    )
    if len(long_exclusions) != 12:
        raise RuntimeError("expected exactly 12 prior longitudinal exclusions")
    long_excluded_states = {str(row["state_id"]) for row in long_exclusions}
    long_excluded_users = {str(row["user_id"]) for row in long_exclusions}

    esconv_states, esconv_states_lineage = _read_all(args.esconv_states)
    esconv_runtime, esconv_runtime_lineage = _read_all(args.esconv_runtime)
    esconv_outcomes, esconv_outcomes_lineage = _read_all(args.esconv_outcomes)
    if len(esconv_states) != EXPECTED_ESCONV_TRAIN_STATES:
        raise RuntimeError("ESConv train state count drifted")
    if len(esconv_runtime) != EXPECTED_ESCONV_TRAIN_STATES:
        raise RuntimeError("ESConv train runtime count drifted")
    if len(esconv_outcomes) != EXPECTED_ESCONV_TRAIN_OUTCOMES:
        raise RuntimeError("ESConv train outcome count drifted")
    if any(str(row.get("split")) != "train" for row in esconv_states):
        raise RuntimeError("ESConv state file is not train-only")
    esconv_exclusions, esconv_exclusions_lineage = _read_all(
        args.esconv_exclusion_selection
    )
    if len(esconv_exclusions) != 24:
        raise RuntimeError("expected exactly 24 prior ESConv exclusions")
    esconv_excluded_states = {str(row["state_id"]) for row in esconv_exclusions}

    pairs = select_longitudinal_pairs(
        state_rows=long_states,
        evaluator_rows=long_evaluators,
        outcome_rows=long_outcomes,
        excluded_state_ids=long_excluded_states,
        excluded_user_ids=long_excluded_users,
    ) + select_esconv_pairs(
        state_rows=esconv_states,
        runtime_rows=esconv_runtime,
        outcome_rows=esconv_outcomes,
        excluded_state_ids=esconv_excluded_states,
    )
    quality_items, audit_items = build_measurement_items(pairs)
    human_packet, human_template = human_anchor_items(pairs)
    human_mapping = human_anchor_mapping(pairs)

    source_lineage = {
        "longitudinal": {
            "states": long_states_lineage,
            "evaluator_contexts": long_evaluators_lineage,
            "action_outcomes": long_outcomes_lineage,
            "prior_packet_exclusion": long_exclusions_lineage,
            "excluded_state_ids_sha256": sha256_text(
                canonical_json(sorted(long_excluded_states))
            ),
            "excluded_user_ids_sha256": sha256_text(
                canonical_json(sorted(long_excluded_users))
            ),
        },
        "esconv_auxiliary": {
            "states": esconv_states_lineage,
            "runtime_states": esconv_runtime_lineage,
            "action_outcomes": esconv_outcomes_lineage,
            "prior_packet_exclusion": esconv_exclusions_lineage,
            "excluded_state_ids_sha256": sha256_text(
                canonical_json(sorted(esconv_excluded_states))
            ),
            "protected_evaluation_fields_opened": False,
        },
        "selected_pairs_sha256": sha256_text(canonical_json(pairs)),
        "quality_items_sha256": sha256_text(canonical_json(quality_items)),
        "audit_items_sha256": sha256_text(canonical_json(audit_items)),
        "human_packet_sha256": sha256_text(canonical_json(human_packet)),
        "human_mapping_sha256": sha256_text(canonical_json(human_mapping)),
    }
    contract = contract_record(
        source_lineage=source_lineage,
        endpoint_contract=_endpoint_contract(args.endpoint_contract),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "selected_pairs_internal.jsonl", pairs)
    write_jsonl(args.out_dir / "quality_ordered_items_internal.jsonl", quality_items)
    write_jsonl(args.out_dir / "evidence_risk_items_internal.jsonl", audit_items)
    write_jsonl(args.out_dir / "human_blind_packet.jsonl", human_packet)
    write_jsonl(
        args.out_dir / "human_annotation_template.jsonl", human_template
    )
    write_jsonl(
        args.out_dir / "human_anchor_mapping_internal.jsonl", human_mapping
    )
    write_json(args.out_dir / "qualification_contract.json", contract)
    write_json(
        args.out_dir / "preparation_summary.json",
        {
            "status": "COMPLETE_ZERO_API_AWAITING_HUMAN_ANCHOR",
            "protocol": PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "pairs": len(pairs),
            "quality_ordered_items": len(quality_items),
            "evidence_risk_items": len(audit_items),
            "human_blind_items": len(human_packet),
            "human_mapping_items": len(human_mapping),
            "api_calls_made": 0,
            "api_clients_created": 0,
            "training_labels_created": False,
            "internal_test_or_external_outcomes_consumed": False,
        },
    )
    print(
        canonical_json(
            {
                "status": "COMPLETE_ZERO_API_AWAITING_HUMAN_ANCHOR",
                "contract_sha256": contract["contract_sha256"],
                "out_dir": str(args.out_dir),
            }
        )
    )


if __name__ == "__main__":
    main()
