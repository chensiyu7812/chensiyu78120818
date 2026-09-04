#!/usr/bin/env python3
"""Run two independent diagnostic audits of generator resource use."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
from typing import Any

from metacom_pm.api import make_client, require_reported_usage
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.io import (
    append_jsonl,
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    stable_hex,
    utc_now,
    write_json,
    write_jsonl,
)
from metacom_pm.prompts import common_context
from metacom_pm.contracts import RuntimeState
from metacom_pm.v1_5_generator_alignment_audit import (
    GeneratorResourceUseJudgment,
    generator_resource_use_messages,
    validate_judgment_contract,
    validate_literal_excerpts,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-generator-alignment-two-judge-diagnostic-v2"
PLAN_PROTOCOL = "pm-v1.5-generator-alignment-qualification-plan-v1"
EXECUTION_PROTOCOL = "pm-v1.5-generator-alignment-qualification-execution-v1"
ENDPOINTS = ("training_judge_openai_mini", "training_judge_gemini_flash_lite")
EXPECTED_RESPONSES = 64
EXPECTED_CALLS = EXPECTED_RESPONSES * len(ENDPOINTS)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _resource_surface(row: dict[str, Any]) -> str:
    lines: list[str] = []
    for item in row["selected_memory_items"]:
        lines.append(f"[{item['source']}] {item['text']}")
    for card in row["selected_strategy_cards"]:
        for key in ("strategy_family", "support_move", "when_to_use", "when_not_to_use"):
            value = card.get(key)
            if value:
                lines.append(f"{key}: {value}")
    if not lines:
        raise RuntimeError(f"{row['state_id']}: selected resource is empty")
    return "\n".join(lines)


def _endpoint_record(name: str, endpoint: Any) -> dict[str, Any]:
    return {
        "name": name,
        "base_url": endpoint.base_url,
        "model": endpoint.model,
        "family": endpoint.family or endpoint.model,
        "transport": endpoint.transport,
    }


def _aggregate(
    rows: list[dict[str, Any]], outcomes: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_call: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["audit_contract_valid"]:
            by_call[str(row["generation_call_id"])].append(row)
    outcome_by_call = {str(row["call_id"]): row for row in outcomes}
    consensus_rows: list[dict[str, Any]] = []
    for call_id, generation in sorted(outcome_by_call.items()):
        audits = by_call.get(call_id, [])
        labels = [str(row["judgment"]["resource_use_label"]) for row in audits]
        exact = len(audits) == len(ENDPOINTS) and len(set(labels)) == 1
        consensus_rows.append(
            {
                "generation_call_id": call_id,
                "state_id": generation["state_id"],
                "component": generation["component"],
                "resource_subtype": generation["resource_subtype"],
                "prompt_variant": generation["prompt_variant"],
                "exact_two_judge_agreement": exact,
                "consensus_label": labels[0] if exact else "unresolved",
                "judge_labels": labels,
            }
        )

    cells: list[dict[str, Any]] = []
    for component in ("RS", "MP", "MS", "ME"):
        for variant in ("legacy_generic", "source_matched"):
            cell = [
                row
                for row in consensus_rows
                if row["component"] == component and row["prompt_variant"] == variant
            ]
            counts = Counter(str(row["consensus_label"]) for row in cell)
            n = len(cell)
            functional = counts["grounded_functional_use"]
            unresolved = counts["unresolved"]
            cells.append(
                {
                    "component": component,
                    "prompt_variant": variant,
                    "n": n,
                    "exact_agreement_n": n - unresolved,
                    "label_counts": dict(sorted(counts.items())),
                    "functional_use_lower_bound": functional / n,
                    "functional_use_upper_bound": (functional + unresolved) / n,
                    "confirmed_misuse_rate": counts["misuse"] / n,
                    "safe_nonuse_rate": counts["correct_nonuse"] / n,
                }
            )

    labels_by_judge = {
        endpoint: Counter(
            str(row["judgment"]["resource_use_label"])
            for row in rows
            if row["judge_endpoint"] == endpoint and row["literal_excerpts_valid"]
        )
        for endpoint in ENDPOINTS
    }
    exact_n = sum(int(row["exact_two_judge_agreement"]) for row in consensus_rows)
    return {
        "protocol": PROTOCOL,
        "status": "COMPLETE_DIAGNOSTIC_NOT_TRAINING_GOLD",
        "response_count": len(outcomes),
        "audit_call_count": len(rows),
        "literal_valid_count": sum(int(row["literal_excerpts_valid"]) for row in rows),
        "audit_contract_valid_count": sum(int(row["audit_contract_valid"]) for row in rows),
        "two_judge_exact_agreement_n": exact_n,
        "two_judge_exact_agreement_rate": exact_n / len(consensus_rows),
        "judge_label_counts": {
            key: dict(sorted(value.items())) for key, value in labels_by_judge.items()
        },
        "component_prompt_cells": cells,
        "interpretation_contract": {
            "automated_audit_is_training_gold": False,
            "unresolved_counts_as_functional_in_lower_bound": False,
            "matched_prompt_repair_gate": "functional-use lower bound >= 0.60 and confirmed misuse <= 0.10",
            "routing_or_retrieval_rejudged": False,
            "general_response_quality_rejudged": False,
        },
    }, consensus_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir", type=Path,
        default=ROOT / "outputs/pm_v1_5_generator_alignment_qualification_v1",
    )
    parser.add_argument(
        "--generation-dir", type=Path,
        default=ROOT / "outputs/pm_v1_5_generator_alignment_qualification_v1_execution",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "outputs/pm_v1_5_generator_alignment_audit_v2",
    )
    parser.add_argument(
        "--experiment-config", type=Path, default=ROOT / "configs/experiment.yaml"
    )
    parser.add_argument("--max-new-calls", type=int, default=EXPECTED_CALLS)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    plan_report = read_json(args.plan_dir / "generation_plan_report.json")
    execution_summary = read_json(args.generation_dir / "execution_summary.json")
    resources = {str(row["state_id"]): row for row in _rows(args.plan_dir / "private_selected_resources.jsonl")}
    outcomes = _rows(args.generation_dir / "generation_outcomes.jsonl")
    experiment = load_config(args.experiment_config)
    endpoints = {
        name: endpoint_from_config(experiment, name) for name in ENDPOINTS
    }
    lineage_ok = (
        plan_report.get("protocol") == PLAN_PROTOCOL
        and execution_summary.get("protocol") == EXECUTION_PROTOCOL
        and execution_summary.get("status") == "COMPLETE"
        and len(outcomes) == EXPECTED_RESPONSES
        and len({row["call_id"] for row in outcomes}) == EXPECTED_RESPONSES
        and set(resources) == {str(row["state_id"]) for row in outcomes}
        and all(row.get("normalized_finish_reason") == "complete" for row in outcomes)
    )

    plan: list[dict[str, Any]] = []
    for outcome in outcomes:
        selected = resources[str(outcome["state_id"])]
        state = RuntimeState.model_validate(selected["runtime_state"])
        resource = _resource_surface(selected)
        for endpoint_name in ENDPOINTS:
            messages = generator_resource_use_messages(
                component=str(outcome["component"]),
                resource_subtype=str(outcome["resource_subtype"]),
                visible_context=common_context(state),
                selected_resource=resource,
                response=str(outcome["response"]),
            )
            plan.append(
                {
                    "protocol": PROTOCOL,
                    "audit_call_id": "genalign_audit_" + stable_hex(
                        PROTOCOL, outcome["call_id"], endpoint_name, n=24
                    ),
                    "generation_call_id": outcome["call_id"],
                    "state_id": outcome["state_id"],
                    "component": outcome["component"],
                    "resource_subtype": outcome["resource_subtype"],
                    "prompt_variant_private_not_judge_input": outcome["prompt_variant"],
                    "judge_endpoint": endpoint_name,
                    "judge_identity": _endpoint_record(endpoint_name, endpoints[endpoint_name]),
                    "messages": messages,
                    "messages_sha256": sha256_text(canonical_json(messages)),
                    "response": outcome["response"],
                    "selected_resource": resource,
                }
            )
    plan_ok = (
        lineage_ok
        and len(plan) == EXPECTED_CALLS
        and len({row["audit_call_id"] for row in plan}) == EXPECTED_CALLS
        and all("prompt_variant" not in canonical_json(row["messages"]) for row in plan)
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.out_dir / "audit_call_plan.jsonl"
    write_jsonl(plan_path, plan)
    outcomes_path = args.out_dir / "audit_outcomes.jsonl"
    existing_rows = _rows(outcomes_path) if outcomes_path.is_file() else []
    existing = {str(row["audit_call_id"]): row for row in existing_rows}
    expected_ids = {str(row["audit_call_id"]) for row in plan}
    existing_ok = (
        len(existing) == len(existing_rows)
        and set(existing) <= expected_ids
        and all(row.get("protocol") == PROTOCOL for row in existing_rows)
    )
    missing_keys = [
        endpoints[name].api_key_env
        for name in ENDPOINTS
        if not os.environ.get(endpoints[name].api_key_env, "")
    ]
    pending = [row for row in plan if str(row["audit_call_id"]) not in existing]
    preflight = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_EXECUTION" if plan_ok and existing_ok and not missing_keys else "BLOCKED",
        "lineage_ok": lineage_ok,
        "plan_ok": plan_ok,
        "existing_outputs_ok": existing_ok,
        "planned_calls": len(plan),
        "completed_calls": len(existing),
        "remaining_calls": len(pending),
        "missing_api_key_environment_variables": sorted(set(missing_keys)),
        "run_requested": args.run,
    }
    write_json(args.out_dir / "audit_preflight.json", preflight)
    if not args.run:
        print(preflight)
        return
    if not plan_ok or not existing_ok or missing_keys:
        raise RuntimeError(canonical_json(preflight))
    if len(pending) > args.max_new_calls:
        raise RuntimeError(
            f"{len(pending)} calls remain but --max-new-calls={args.max_new_calls}"
        )

    clients = {name: make_client(endpoint) for name, endpoint in endpoints.items()}
    for row in pending:
        result, parsed = clients[row["judge_endpoint"]].chat(
            row["messages"],
            temperature=0.0,
            max_tokens=700,
            seed=int(stable_hex(PROTOCOL, row["audit_call_id"], n=8), 16)
            % (2**31 - 1),
            response_schema=GeneratorResourceUseJudgment,
            retries=3,
        )
        if parsed is None:
            raise RuntimeError("structured audit returned no parsed judgment")
        usage = require_reported_usage(result.usage, stage="generator_alignment_audit")
        literal_valid = validate_literal_excerpts(
            judgment=parsed,
            response=row["response"],
            resource=row["selected_resource"],
        )
        expected_function = {
            "RS": "support_technique",
            "MS": "session_continuity",
            "ME": "event_or_outcome",
            "MP": (
                "style_or_pacing"
                if row["resource_subtype"] == "MP_PREFERENCE"
                else "profile_context"
            ),
        }[row["component"]]
        function_valid = validate_judgment_contract(
            judgment=parsed, expected_function=expected_function
        )
        saved = {
            "protocol": PROTOCOL,
            "audit_call_id": row["audit_call_id"],
            "generation_call_id": row["generation_call_id"],
            "state_id": row["state_id"],
            "component": row["component"],
            "resource_subtype": row["resource_subtype"],
            "prompt_variant_private_not_judge_input": row[
                "prompt_variant_private_not_judge_input"
            ],
            "judge_endpoint": row["judge_endpoint"],
            "judge_identity": row["judge_identity"],
            "messages_sha256": row["messages_sha256"],
            "judgment": parsed.model_dump(mode="json"),
            "literal_excerpts_valid": literal_valid,
            "judgment_cross_fields_valid": function_valid,
            "audit_contract_valid": literal_valid and function_valid,
            "normalized_finish_reason": result.normalized_finish_reason,
            "provider_finish_reason": result.provider_finish_reason,
            "usage": usage,
            "completed_at": utc_now(),
        }
        append_jsonl(outcomes_path, saved)
        existing[str(row["audit_call_id"])] = saved

    final_rows = [existing[row["audit_call_id"]] for row in plan]
    report, consensus = _aggregate(final_rows, outcomes)
    report["inputs"] = {
        "plan_freeze_manifest_sha256": sha256_file(args.plan_dir / "freeze_manifest.json"),
        "generation_outcomes_sha256": sha256_file(args.generation_dir / "generation_outcomes.jsonl"),
        "audit_call_plan_sha256": sha256_file(plan_path),
    }
    write_jsonl(args.out_dir / "consensus_diagnostic.jsonl", consensus)
    write_json(args.out_dir / "audit_report.json", report)
    print(
        {
            "protocol": PROTOCOL,
            "status": report["status"],
            "audit_call_count": report["audit_call_count"],
            "two_judge_exact_agreement_rate": report["two_judge_exact_agreement_rate"],
        }
    )


if __name__ == "__main__":
    main()
