#!/usr/bin/env python3
"""Audit MS routing evidence without treating response preference as PM gold."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from metacom_pm.contracts import MemoryBackendRecord, MemorySource, parse_action_id
from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl
from metacom_pm.text import normalize_space


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-ms-routing-retrieval-execution-audit-v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def build(
    *,
    blueprint_dir: Path,
    plan_dir: Path,
    execution_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    contrasts = _rows(blueprint_dir / "d3_ms_replacement_contrasts.jsonl")
    design = _rows(blueprint_dir / "private_design_strata.jsonl")
    calls = _rows(plan_dir / "call_plan.jsonl")
    outcomes = _rows(execution_dir / "generation_outcomes.jsonl")
    backends = {
        record.card_id: {item.memory_id: item for item in record.items}
        for record in (
            MemoryBackendRecord.model_validate(row)
            for row in iter_jsonl(blueprint_dir / "memory_backend.jsonl")
        )
    }
    states = {
        str(row["state_id"]): row
        for row in _rows(blueprint_dir / "runtime_states.jsonl")
    }
    contrast_by_slot = {
        str(row["contrast_slot_id"]): row for row in contrasts
    }
    design_by_slot = {
        str(row["contrast_slot_id"]): row for row in design
    }
    outcome_by_call = {str(row["call_id"]): row for row in outcomes}
    pair_calls: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for call in calls:
        pair_calls[str(call["pair_id"])][str(call["arm"])] = call

    row_audits: list[dict[str, Any]] = []
    for pair_id, arms in sorted(pair_calls.items()):
        control = arms["control"]
        treatment = arms["treatment"]
        contrast = contrast_by_slot[str(control["contrast_slot_id"])]
        stratum = design_by_slot[str(control["contrast_slot_id"])]
        state = states[str(control["state_id"])]
        catalog = backends[str(state["card_id"])]
        control_sources, control_strategy = parse_action_id(
            str(control["action_id"])
        )
        treatment_sources, treatment_strategy = parse_action_id(
            str(treatment["action_id"])
        )
        treatment_ms_ids = [
            memory_id
            for memory_id in treatment["selected_memory_ids"]
            if catalog[str(memory_id)].source is MemorySource.MS
        ]
        control_items = [
            catalog[str(memory_id)]
            for memory_id in control["selected_memory_ids"]
        ]
        cross_resource_exact_redundancy = any(
            (
                normalize_space(catalog[str(memory_id)].text).lower()
                in normalize_space(other.text).lower()
                or normalize_space(other.text).lower()
                in normalize_space(catalog[str(memory_id)].text).lower()
            )
            for memory_id in treatment_ms_ids
            for other in control_items
        )
        treatment_prompt = "\n".join(
            str(message.get("content") or "")
            for message in treatment["messages"]
        )
        control_prompt = "\n".join(
            str(message.get("content") or "")
            for message in control["messages"]
        )
        prompt_contains_every_selected_ms = all(
            catalog[str(memory_id)].text in treatment_prompt
            for memory_id in treatment_ms_ids
        )
        # A deliberately redundant negative may repeat the same words in the
        # visible current message.  Contamination therefore means an injected
        # MS record marker, not mere occurrence of the candidate text.
        control_contains_injected_ms = "- [MS;" in control_prompt
        grounding = contrast["component_candidate_observation"]["grounding"]
        incremental = contrast["component_candidate_observation"][
            "incremental_value"
        ]
        control_outcome = outcome_by_call.get(str(control["call_id"]))
        treatment_outcome = outcome_by_call.get(str(treatment["call_id"]))
        checks = {
            "same_state": control["state_id"] == treatment["state_id"],
            "same_user": control["user_id"] == treatment["user_id"],
            "same_seed": (
                control["generation"]["seed"]
                == treatment["generation"]["seed"]
            ),
            "only_ms_source_added": (
                set(treatment_sources) - set(control_sources)
                == {MemorySource.MS}
                and not (set(control_sources) - set(treatment_sources))
                and control_strategy == treatment_strategy
            ),
            "treatment_selected_ms": bool(treatment_ms_ids),
            "control_selected_no_ms": all(
                catalog[str(memory_id)].source is not MemorySource.MS
                for memory_id in control["selected_memory_ids"]
            ),
            "selected_ms_present_in_treatment_prompt": (
                prompt_contains_every_selected_ms
            ),
            "selected_ms_absent_from_control_prompt": (
                not control_contains_injected_ms
            ),
            "candidate_passed_hard_grounding": (
                grounding["candidate_present"]
                and not grounding["deterministic_hard_off"]
            ),
            "both_responses_completed": (
                control_outcome is not None
                and treatment_outcome is not None
                and control_outcome["normalized_finish_reason"] == "complete"
                and treatment_outcome["normalized_finish_reason"] == "complete"
            ),
        }
        pre_background_target = int(
            incremental["candidate_incremental_alignment_score"]
        )
        routing_target = int(
            pre_background_target and not cross_resource_exact_redundancy
        )
        row_audits.append(
            {
                "protocol": PROTOCOL,
                "pair_id": pair_id,
                "pair_role": control["pair_role"],
                "state_id": control["state_id"],
                "user_id": control["user_id"],
                "design_cell_private_not_pm_input": stratum["design_cell"],
                "pre_background_opportunity_target": pre_background_target,
                "cross_resource_exact_redundancy": (
                    cross_resource_exact_redundancy
                ),
                "routing_opportunity_target": routing_target,
                "topically_relevant_candidate": bool(
                    grounding["candidate_present"]
                ),
                "incrementally_usable_candidate": bool(
                    incremental["candidate_incremental_alignment_score"]
                ),
                "selected_ms_count": len(treatment_ms_ids),
                "checks": checks,
                "all_machine_checks_pass": all(checks.values()),
                "generator_uptake_status": (
                    "NOT_INFERRED_FROM_TEXT_DIFFERENCE_OR_MENTION"
                ),
                "quality_preference_status": "NOT_USED_AS_ROUTING_GOLD",
            }
        )

    state_rows: dict[str, dict[str, Any]] = {}
    for row in row_audits:
        state_rows.setdefault(row["state_id"], row)
    state_values = list(state_rows.values())
    opportunity_counts = Counter(
        row["routing_opportunity_target"] for row in state_values
    )
    pre_background_counts = Counter(
        row["pre_background_opportunity_target"] for row in state_values
    )
    pair_role_counts = Counter(row["pair_role"] for row in row_audits)
    total_usage = Counter()
    for row in outcomes:
        for key, value in dict(row.get("usage") or {}).items():
            total_usage[key] += int(value)
    checks = {
        "32_independent_states": len(state_values) == 32,
        "16_on_16_off_pre_background_targets": pre_background_counts
        == Counter({0: 16, 1: 16}),
        "background_redundancy_removed_from_on_targets": all(
            not row["cross_resource_exact_redundancy"]
            or row["routing_opportunity_target"] == 0
            for row in state_values
        ),
        "40_pairs_80_calls": len(row_audits) == 40 and len(calls) == 80,
        "32_primary_8_repeat_pairs": pair_role_counts
        == Counter({"primary": 32, "outcome_blind_repeat": 8}),
        "80_completed_outcomes": len(outcomes) == 80,
        "all_pair_machine_checks_pass": all(
            row["all_machine_checks_pass"] for row in row_audits
        ),
    }
    status = (
        "PASS_RETRIEVAL_AND_EXECUTION_ROUTING_NOT_EVALUATED_HERE"
        if all(checks.values())
        else "FAIL_MACHINE_CHAIN_AUDIT"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "pair_audit.jsonl", row_audits)
    report = {
        "protocol": PROTOCOL,
        "status": status,
        "construct_separation": {
            "pm_routing": "NOT_EVALUATED_IN_THIS_AUDIT_SEE_SEPARATE_ROUTER_FIT",
            "retrieval": "MACHINE_QUALIFIED_FOR_THE_FROZEN_CONSTRUCTED_STATES",
            "execution": "MACHINE_QUALIFIED",
            "generator_grounded_use": "NOT_YET_SEMANTICALLY_ADJUDICATED",
            "reply_quality": "NOT_USED_AS_PM_ROUTING_GOLD",
        },
        "independent_states": len(state_values),
        "routing_opportunity_targets": {
            str(key): value for key, value in sorted(opportunity_counts.items())
        },
        "pre_background_opportunity_targets": {
            str(key): value
            for key, value in sorted(pre_background_counts.items())
        },
        "states_with_cross_resource_exact_redundancy": sum(
            int(row["cross_resource_exact_redundancy"])
            for row in state_values
        ),
        "topically_relevant_states": sum(
            int(row["topically_relevant_candidate"]) for row in state_values
        ),
        "incrementally_usable_states": sum(
            int(row["incrementally_usable_candidate"]) for row in state_values
        ),
        "pairs": len(row_audits),
        "calls": len(calls),
        "completed_outcomes": len(outcomes),
        "realized_usage": dict(sorted(total_usage.items())),
        "checks": checks,
        "interpretation": [
            "The audit establishes that the frozen retriever and executor delivered the intended MS contrast.",
            "It does not establish that PM learned the routing target because no PM prediction has been fitted here.",
            "It does not infer generator uptake from a memory mention or from treatment/control wording differences.",
            "Single-pair response preference remains downstream system evidence, not Step-1 correctness gold.",
        ],
    }
    write_json(out_dir / "report_evidence.json", report)
    write_json(
        out_dir / "freeze_manifest.json",
        {
            "protocol": PROTOCOL,
            "status": status,
            "report_evidence_sha256": sha256_file(
                out_dir / "report_evidence.json"
            ),
            "pair_audit_sha256": sha256_file(out_dir / "pair_audit.jsonl"),
            "measurement_layers_contract_sha256": sha256_file(
                ROOT
                / "data/pm_v1_5_contracts/resource_routing_measurement_layers_v1.json"
            ),
        },
    )
    return report


def main() -> None:
    out_dir = ROOT / "outputs/pm_v1_5_ms_routing_retrieval_execution_audit_v1"
    report = build(
        blueprint_dir=ROOT / "outputs/pm_v1_5_ms_incremental_value_step0_v1",
        plan_dir=ROOT / "outputs/pm_v1_5_ms_incremental_value_generation_v1",
        execution_dir=ROOT
        / "outputs/pm_v1_5_ms_incremental_value_generation_v1_execution",
        out_dir=out_dir,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "protocol",
                    "status",
                    "independent_states",
                    "routing_opportunity_targets",
                    "pairs",
                    "calls",
                    "completed_outcomes",
                    "checks",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
