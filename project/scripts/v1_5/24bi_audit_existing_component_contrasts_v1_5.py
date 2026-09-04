#!/usr/bin/env python3
"""Audit which historical memory contrasts are reusable for V1.5.

This is a zero-API, outcome-blind structural audit.  It never turns historical
evaluator ``needed_memory_sources`` fields or judge scores into labels.  It
only checks whether two already-generated responses form a clean one-bit
component treatment under the same state and frozen generator.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import chain, combinations
import json
from pathlib import Path
from typing import Any, Iterable

from metacom_pm.contracts import (
    ALL_ACTION_IDS,
    MemorySource,
    StrategyMode,
    canonical_action_id,
)
from metacom_pm.io import (
    iter_jsonl,
    sha256_file,
    sha256_text,
    stable_hex,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-existing-component-contrast-reuse-audit-v1"
COMPONENTS = tuple(MemorySource)


def _powerset(values: Iterable[MemorySource]) -> list[frozenset[MemorySource]]:
    ordered = tuple(values)
    return [
        frozenset(items)
        for size in range(len(ordered) + 1)
        for items in combinations(ordered, size)
    ]


def _memory_sources_by_card(path: Path) -> dict[str, dict[str, MemorySource]]:
    result: dict[str, dict[str, MemorySource]] = {}
    for row in iter_jsonl(path):
        card_id = str(row["card_id"])
        result[card_id] = {
            str(item["memory_id"]): MemorySource(str(item["source"]))
            for item in row["items"]
        }
    return result


def _uniform(rows: list[dict[str, Any]], getter) -> set[Any]:
    return {getter(row) for row in rows}


def _contrast(
    *,
    component: MemorySource,
    background: frozenset[MemorySource],
    control: dict[str, Any],
    treatment: dict[str, Any],
    memory_sources: dict[str, MemorySource],
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    expected_control = canonical_action_id(background, StrategyMode.R0)
    expected_treatment = canonical_action_id(
        set(background) | {component}, StrategyMode.R0
    )
    if control["requested_action_id"] != expected_control:
        failures.append("control_action_mismatch")
    if treatment["requested_action_id"] != expected_treatment:
        failures.append("treatment_action_mismatch")
    for side_name, row in (("control", control), ("treatment", treatment)):
        if not (
            row["requested_action_id"]
            == row["effective_action_id"]
            == row["realized_action_id"]
        ):
            failures.append(f"{side_name}_action_not_realized")
        if row.get("selected_strategy_ids"):
            failures.append(f"{side_name}_strategy_not_empty")
    for key in ("state_id", "user_id", "card_id", "model_name"):
        if control[key] != treatment[key]:
            failures.append(f"{key}_differs")
    control_treatment = control["provenance"].get(
        "supporter_generation_treatment_sha256"
    )
    treatment_treatment = treatment["provenance"].get(
        "supporter_generation_treatment_sha256"
    )
    if control_treatment != treatment_treatment:
        failures.append("supporter_generation_treatment_differs")

    control_ids = set(map(str, control.get("selected_memory_ids", [])))
    treatment_ids = set(map(str, treatment.get("selected_memory_ids", [])))
    removed = control_ids - treatment_ids
    added = treatment_ids - control_ids
    if removed:
        failures.append("background_memory_changed_or_removed")
    if not added:
        failures.append("component_treatment_added_no_memory")
    unknown = (control_ids | treatment_ids) - set(memory_sources)
    if unknown:
        failures.append("selected_memory_missing_from_backend")
    if any(memory_sources.get(value) is not component for value in added):
        failures.append("added_memory_not_from_component")
    if any(memory_sources.get(value) not in background for value in control_ids):
        failures.append("control_memory_outside_background")
    if control["prompt_hash"] == treatment["prompt_hash"]:
        failures.append("prompt_did_not_change")

    row = {
        "protocol": PROTOCOL,
        "contrast_id": "component_contrast_"
        + stable_hex(
            PROTOCOL,
            str(control["state_id"]),
            component.value,
            expected_control,
            expected_treatment,
            n=24,
        ),
        "state_id": str(control["state_id"]),
        "user_id": str(control["user_id"]),
        "card_id": str(control["card_id"]),
        "component": component.value,
        "strategy_background": "R0",
        "other_memory_background": sorted(value.value for value in background),
        "control_action": expected_control,
        "treatment_action": expected_treatment,
        "control_response_sha256": sha256_text(str(control["response"])),
        "treatment_response_sha256": sha256_text(str(treatment["response"])),
        "responses_exactly_equal": (
            str(control["response"]) == str(treatment["response"])
        ),
        "control_prompt_sha256": str(control["prompt_hash"]),
        "treatment_prompt_sha256": str(treatment["prompt_hash"]),
        "control_selected_memory_ids": sorted(control_ids),
        "treatment_selected_memory_ids": sorted(treatment_ids),
        "added_component_memory_ids": sorted(added),
        "control_total_input_tokens": int(
            control.get("cost", {}).get("total_input_tokens", 0)
        ),
        "treatment_total_input_tokens": int(
            treatment.get("cost", {}).get("total_input_tokens", 0)
        ),
        "control_output_tokens": int(
            control.get("cost", {}).get("output_tokens", 0)
        ),
        "treatment_output_tokens": int(
            treatment.get("cost", {}).get("output_tokens", 0)
        ),
        "structurally_reusable": not failures,
        "failure_reasons": sorted(set(failures)),
    }
    return row, failures


def run_audit(
    *,
    outcomes_path: Path,
    states_path: Path,
    backend_path: Path,
    current_rs_plan_path: Path,
    current_bank_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    outcomes = [dict(row) for row in iter_jsonl(outcomes_path)]
    states = {str(row["state_id"]): dict(row) for row in iter_jsonl(states_path)}
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_keys: list[tuple[str, str]] = []
    for row in outcomes:
        key = (str(row["state_id"]), str(row["requested_action_id"]))
        if key in by_key:
            duplicate_keys.append(key)
        by_key[key] = row

    expected_keys = {
        (state_id, action_id)
        for state_id in states
        for action_id in ALL_ACTION_IDS
    }
    observed_keys = set(by_key)
    current_rs_plan = json.loads(
        current_rs_plan_path.read_text(encoding="utf-8")
    )
    expected_treatment_sha = str(
        current_rs_plan["supporter_generation_treatment_sha256"]
    )
    expected_model = str(current_rs_plan["generator_identity"]["model"])
    treatment_shas = _uniform(
        outcomes,
        lambda row: row["provenance"].get(
            "supporter_generation_treatment_sha256"
        ),
    )
    models = _uniform(outcomes, lambda row: row["model_name"])
    backend_shas = _uniform(
        outcomes, lambda row: row["provenance"].get("backend_sha256")
    )
    strategy_bank_shas = _uniform(
        outcomes, lambda row: row["provenance"].get("strategy_bank_sha256")
    )

    source_map_by_card = _memory_sources_by_card(backend_path)
    contrast_rows: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    for state_id, state in states.items():
        card_id = str(state["card_id"])
        memory_sources = source_map_by_card.get(card_id, {})
        for component in COMPONENTS:
            others = [value for value in COMPONENTS if value is not component]
            for background in _powerset(others):
                control_action = canonical_action_id(
                    background, StrategyMode.R0
                )
                treatment_action = canonical_action_id(
                    set(background) | {component}, StrategyMode.R0
                )
                control = by_key.get((state_id, control_action))
                treatment = by_key.get((state_id, treatment_action))
                if control is None or treatment is None:
                    failure_counts["missing_action_outcome"] += 1
                    continue
                row, failures = _contrast(
                    component=component,
                    background=background,
                    control=control,
                    treatment=treatment,
                    memory_sources=memory_sources,
                )
                contrast_rows.append(row)
                failure_counts.update(set(failures))

    reusable = [row for row in contrast_rows if row["structurally_reusable"]]
    exact_equal = sum(row["responses_exactly_equal"] for row in reusable)
    matrix_complete = (
        not duplicate_keys
        and expected_keys == observed_keys
        and len(outcomes) == len(expected_keys)
    )
    stack_checks = {
        "action_matrix_complete_468_x_16": matrix_complete,
        "supporter_treatment_uniform": treatment_shas
        == {expected_treatment_sha},
        "generator_model_uniform": models == {expected_model},
        "memory_backend_file_matches_outcome_binding": backend_shas
        == {sha256_file(backend_path)},
        "historical_strategy_bank_differs_from_current_bank": (
            strategy_bank_shas != {sha256_file(current_bank_path)}
        ),
        "R0_contrasts_inject_no_strategy": all(
            not by_key[(row["state_id"], row["control_action"])].get(
                "selected_strategy_ids"
            )
            and not by_key[(row["state_id"], row["treatment_action"])].get(
                "selected_strategy_ids"
            )
            for row in reusable
        ),
    }
    formal_reuse_pass = (
        all(
            stack_checks[key]
            for key in (
                "action_matrix_complete_468_x_16",
                "supporter_treatment_uniform",
                "generator_model_uniform",
                "memory_backend_file_matches_outcome_binding",
                "R0_contrasts_inject_no_strategy",
            )
        )
        and len(reusable) == 468 * 3 * 4
        and not failure_counts
    )

    by_component = Counter(row["component"] for row in reusable)
    by_background = Counter(
        (
            row["component"],
            ",".join(row["other_memory_background"]) or "M0",
        )
        for row in reusable
    )
    report = {
        "protocol": PROTOCOL,
        "status": (
            "R0_MEMORY_CONTRASTS_STRUCTURALLY_REUSABLE"
            if formal_reuse_pass
            else "REUSE_AUDIT_FAILED"
        ),
        "api_calls_made": 0,
        "training_labels_created": False,
        "human_quality_labels_read": False,
        "needed_memory_sources_read_as_labels": False,
        "outcomes": len(outcomes),
        "states": len(states),
        "users": len({str(row["user_id"]) for row in states.values()}),
        "actions": len({str(row["requested_action_id"]) for row in outcomes}),
        "expected_clean_R0_memory_contrasts": 468 * 3 * 4,
        "observed_contrasts": len(contrast_rows),
        "reusable_contrasts": len(reusable),
        "exact_equal_response_contrasts": exact_equal,
        "failure_counts": dict(sorted(failure_counts.items())),
        "reusable_by_component": dict(sorted(by_component.items())),
        "reusable_by_component_and_background": {
            f"{component}|{background}": count
            for (component, background), count in sorted(by_background.items())
        },
        "stack_checks": stack_checks,
        "bindings": {
            "supporter_generation_treatment_sha256": expected_treatment_sha,
            "generator_model": expected_model,
            "memory_backend_sha256": sha256_file(backend_path),
            "historical_strategy_bank_sha256_values": sorted(
                value for value in strategy_bank_shas if value
            ),
            "current_strategy_bank_sha256": sha256_file(current_bank_path),
        },
        "reuse_boundary": {
            "approved": (
                "R0 memory-component contrasts only, subject to later blind "
                "quality and positive-arm risk review"
            ),
            "not_approved": (
                "historical RS arms, historical judge labels, "
                "needed_memory_sources, or applicability-proxy labels"
            ),
            "why_old_strategy_bank_mismatch_is_nonblocking_here": (
                "Every approved contrast is R0 on both sides and injects no "
                "strategy item."
            ),
        },
        "inputs": {
            "outcomes": str(outcomes_path.relative_to(ROOT)),
            "states": str(states_path.relative_to(ROOT)),
            "memory_backend": str(backend_path.relative_to(ROOT)),
            "current_rs_plan": str(current_rs_plan_path.relative_to(ROOT)),
            "current_strategy_bank": str(current_bank_path.relative_to(ROOT)),
        },
    }
    return report, reusable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_longitudinal_action_sweep_v8_19_2_continuation_v2_dry_run/action_outcomes.jsonl",
    )
    parser.add_argument(
        "--states",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate/pm_v2_states.jsonl",
    )
    parser.add_argument(
        "--memory-backend",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate/memory_backend.jsonl",
    )
    parser.add_argument(
        "--current-rs-plan",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_paired_effect_v1/plan_report.json",
    )
    parser.add_argument(
        "--current-bank",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_component_contrast_reuse_audit_v1",
    )
    args = parser.parse_args()
    report, rows = run_audit(
        outcomes_path=args.outcomes,
        states_path=args.states,
        backend_path=args.memory_backend,
        current_rs_plan_path=args.current_rs_plan,
        current_bank_path=args.current_bank,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "reusable_R0_memory_contrasts.jsonl"
    report_path = args.out_dir / "audit_report.json"
    write_jsonl(rows_path, rows)
    report["outputs"] = {
        rows_path.name: sha256_file(rows_path),
    }
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "reusable_contrasts": report["reusable_contrasts"],
                "exact_equal_response_contrasts": report[
                    "exact_equal_response_contrasts"
                ],
                "out_dir": str(args.out_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
