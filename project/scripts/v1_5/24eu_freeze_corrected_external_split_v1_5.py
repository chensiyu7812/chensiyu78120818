#!/usr/bin/env python3
"""Freeze the corrected external development/qualification/lockbox split.

The stopped V2 execution exposed every EvoEmo state belonging to users p1-p6
and no state belonging to p7-p18.  Split membership is therefore frozen at
the *user* grain, never at the remaining-call grain:

* p1-p6: diagnostic/development only;
* p7-p12: one-shot external qualification;
* p13-p18: sealed final lockbox;
* ESConv: corrected same-state rerun after the objectively empty RS treatment.

No response quality, risk, preference, or judge result is read.  A companion
validator can partition a subsequently rebuilt call plan, but only after every
requested resource is materially present in both the private resource field
and the generator messages.  This prevents the empty-RS bug from silently
reaching another paid execution.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.text import normalize_space


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5b-corrected-external-user-split-freeze-v1"
DIAGNOSTIC_USERS = tuple(f"p{i}" for i in range(1, 7))
QUALIFICATION_USERS = tuple(f"p{i}" for i in range(7, 13))
LOCKBOX_USERS = tuple(f"p{i}" for i in range(13, 19))
PARTITION_FILES = {
    "evoemo_diagnostic": "evoemo_diagnostic_panel_private.jsonl",
    "evoemo_qualification": "evoemo_qualification_panel_private.jsonl",
    "evoemo_lockbox": "evoemo_lockbox_panel_private.jsonl",
    "esconv_corrected_rerun": "esconv_corrected_rerun_panel_private.jsonl",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _partition_name(panel: dict[str, Any]) -> str:
    if panel["domain"] == "ESConv":
        return "esconv_corrected_rerun"
    user_id = str(panel["user_id_private_analysis_only"])
    if user_id in DIAGNOSTIC_USERS:
        return "evoemo_diagnostic"
    if user_id in QUALIFICATION_USERS:
        return "evoemo_qualification"
    if user_id in LOCKBOX_USERS:
        return "evoemo_lockbox"
    raise RuntimeError(f"unexpected EvoEmo user outside frozen p1-p18: {user_id}")


def _empty_requested_resources(call: dict[str, Any]) -> list[str]:
    selected = dict(call.get("selected_resources_private") or {})
    return [
        component
        for component in call.get("requested_components") or []
        if not normalize_space(str(selected.get(component) or ""))
    ]


def _resources_absent_from_messages(call: dict[str, Any]) -> list[str]:
    selected = dict(call.get("selected_resources_private") or {})
    message_text = normalize_space(
        "\n".join(str(row.get("content") or "") for row in call.get("messages") or [])
    )
    return [
        component
        for component in call.get("requested_components") or []
        if normalize_space(str(selected.get(component) or "")) not in message_text
    ]


def build_split(*, root: Path = ROOT) -> dict[str, Any]:
    """Freeze user-grain membership from the stopped V2 execution."""

    panel_dir = root / "outputs/pm_v1_5b_final_external_panel_v2"
    plan_dir = root / "outputs/pm_v1_5b_final_system_generation_v2_candidate"
    execution_dir = root / "outputs/pm_v1_5b_final_system_generation_v2_execution"
    esconv_path = panel_dir / "esconv_panel_private.jsonl"
    evoemo_path = panel_dir / "evoemo_panel_private.jsonl"
    panel_report_path = panel_dir / "panel_freeze_report.json"
    plan_path = plan_dir / "call_plan_private.jsonl"
    outcomes_path = execution_dir / "generation_outcomes_private.jsonl"

    if read_json(panel_report_path)["status"] != "PASS_POST_COMPATIBILITY_FINAL_V2_FROZEN":
        raise RuntimeError("V2 external panel is not a valid frozen source")
    esconv = _rows(esconv_path)
    evoemo = _rows(evoemo_path)
    plan = _rows(plan_path)
    outcomes = _rows(outcomes_path)
    panels = [*esconv, *evoemo]
    by_panel = {str(row["panel_id"]): row for row in panels}
    if len(by_panel) != len(panels):
        raise RuntimeError("source panel_id values are not unique")
    plan_by_call = {str(row["call_id"]): row for row in plan}
    if len(plan_by_call) != len(plan):
        raise RuntimeError("source call_id values are not unique")
    outcome_by_call = {str(row["call_id"]): row for row in outcomes}
    if len(outcome_by_call) != len(outcomes):
        raise RuntimeError("saved outcome call_id values are not unique")
    if not set(outcome_by_call).issubset(plan_by_call):
        raise RuntimeError("an outcome does not belong to the frozen V2 plan")

    exposure_calls: Counter[str] = Counter()
    exposure_panels: dict[str, set[str]] = defaultdict(set)
    for outcome in outcomes:
        panel_id = str(outcome["panel_id"])
        panel = by_panel.get(panel_id)
        if panel is None:
            raise RuntimeError(f"outcome references unknown panel: {panel_id}")
        if str(outcome["state_id"]) != str(panel["state_id"]):
            raise RuntimeError("outcome state_id disagrees with frozen panel")
        if outcome["domain"] == "EvoEmo":
            user_id = str(panel["user_id_private_analysis_only"])
            exposure_calls[user_id] += 1
            exposure_panels[user_id].add(panel_id)

    partitions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for panel in panels:
        partitions[_partition_name(panel)].append(panel)
    out_dir = root / "outputs/pm_v1_5b_corrected_external_split_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}
    for name, filename in PARTITION_FILES.items():
        path = out_dir / filename
        write_jsonl(path, partitions[name])
        output_paths[name] = path

    all_evo_users = {str(row["user_id_private_analysis_only"]) for row in evoemo}
    diagnostic_panel_ids = {str(row["panel_id"]) for row in partitions["evoemo_diagnostic"]}
    qualification_panel_ids = {str(row["panel_id"]) for row in partitions["evoemo_qualification"]}
    lockbox_panel_ids = {str(row["panel_id"]) for row in partitions["evoemo_lockbox"]}
    exposed_evo_panel_ids = {
        str(row["panel_id"]) for row in outcomes if row["domain"] == "EvoEmo"
    }
    exposed_esconv_panel_ids = {
        str(row["panel_id"]) for row in outcomes if row["domain"] == "ESConv"
    }
    empty_resource_counts = Counter(
        component for call in plan for component in _empty_requested_resources(call)
    )

    checks = {
        "source_panel_exactly_122_esconv_and_204_evoemo_states": len(esconv) == 122
        and len(evoemo) == 204,
        "all_18_expected_evoemo_users_present": all_evo_users
        == set((*DIAGNOSTIC_USERS, *QUALIFICATION_USERS, *LOCKBOX_USERS)),
        "user_grain_partition_is_66_60_78_states": [
            len(partitions["evoemo_diagnostic"]),
            len(partitions["evoemo_qualification"]),
            len(partitions["evoemo_lockbox"]),
        ]
        == [66, 60, 78],
        "evoemo_partitions_are_disjoint_and_exhaustive": not (
            diagnostic_panel_ids & qualification_panel_ids
            or diagnostic_panel_ids & lockbox_panel_ids
            or qualification_panel_ids & lockbox_panel_ids
        )
        and diagnostic_panel_ids | qualification_panel_ids | lockbox_panel_ids
        == {str(row["panel_id"]) for row in evoemo},
        "only_p1_p6_have_any_saved_v2_outcome": set(exposure_calls)
        == set(DIAGNOSTIC_USERS),
        "every_p1_p6_state_is_development_contaminated": exposed_evo_panel_ids
        == diagnostic_panel_ids,
        "p7_p12_qualification_has_zero_saved_v2_outcomes": not bool(
            exposed_evo_panel_ids & qualification_panel_ids
        ),
        "p13_p18_lockbox_has_zero_saved_v2_outcomes": not bool(
            exposed_evo_panel_ids & lockbox_panel_ids
        ),
        "all_esconv_panels_were_exposed_under_objectively_broken_treatment": exposed_esconv_panel_ids
        == {str(row["panel_id"]) for row in esconv},
        "v2_empty_rs_treatment_bug_detected": empty_resource_counts["RS"] > 0,
        "split_did_not_read_quality_risk_preference_or_judge": True,
    }
    report = {
        "protocol": PROTOCOL,
        "status": "PASS_CORRECTED_EXTERNAL_USER_SPLIT_FROZEN"
        if all(checks.values())
        else "FAIL_CORRECTED_EXTERNAL_SPLIT",
        "grain": "independent_user_for_EvoEmo; independent_dialogue_for_ESConv",
        "partitions": {
            "evoemo_diagnostic": {
                "users": list(DIAGNOSTIC_USERS),
                "states": len(partitions["evoemo_diagnostic"]),
                "saved_v2_calls": sum(exposure_calls[user] for user in DIAGNOSTIC_USERS),
                "role": "development_and_root_cause_only_not_final_evidence",
            },
            "evoemo_qualification": {
                "users": list(QUALIFICATION_USERS),
                "states": len(partitions["evoemo_qualification"]),
                "saved_v2_calls": sum(exposure_calls[user] for user in QUALIFICATION_USERS),
                "role": "one_shot_external_qualification_before_lockbox",
            },
            "evoemo_lockbox": {
                "users": list(LOCKBOX_USERS),
                "states": len(partitions["evoemo_lockbox"]),
                "saved_v2_calls": sum(exposure_calls[user] for user in LOCKBOX_USERS),
                "role": "sealed_final_confirmatory_evidence",
            },
            "esconv_corrected_rerun": {
                "states": len(partitions["esconv_corrected_rerun"]),
                "independent_dialogues": len(
                    {
                        row["dialogue_id_private_analysis_only"]
                        for row in partitions["esconv_corrected_rerun"]
                    }
                ),
                "role": "same_state_corrected_rerun_after_empty_RS_treatment_bug",
            },
        },
        "v2_diagnostic_closeout": {
            "saved_outcomes": len(outcomes),
            "domain_calls": dict(sorted(Counter(row["domain"] for row in outcomes).items())),
            "evoemo_user_calls": dict(sorted(exposure_calls.items())),
            "empty_requested_resource_counts_in_v2_plan": dict(sorted(empty_resource_counts.items())),
            "quality_risk_preference_or_judge_used_for_partition": False,
            "v2_outputs_are_diagnostic_not_formal_effect_evidence": True,
        },
        "execution_contract": {
            "corrected_plan_must_pass_nonempty_resource_materialization": True,
            "qualification_must_precede_lockbox": True,
            "lockbox_may_not_be_used_to_modify_router_retriever_prompt_generator_or_guard": True,
            "if_stack_changes_after_qualification_it_must_be_reported_and_requalified_before_lockbox": True,
            "esconv_rerun_reason": "objective treatment materialization bug; not outcome-driven tuning",
        },
        "checks": checks,
        "inputs": {
            str(path.relative_to(root)): sha256_file(path)
            for path in (esconv_path, evoemo_path, panel_report_path, plan_path, outcomes_path)
        },
        "outputs": {
            str(path.relative_to(root)): sha256_file(path)
            for path in output_paths.values()
        },
    }
    write_json(out_dir / "external_split_freeze_report.json", report)
    return report


def validate_and_partition_corrected_plan(
    *, corrected_plan_path: Path, output_dir: Path, root: Path = ROOT
) -> dict[str, Any]:
    """Validate treatment materialization, then partition a corrected plan."""

    split_dir = root / "outputs/pm_v1_5b_corrected_external_split_v1"
    split_report_path = split_dir / "external_split_freeze_report.json"
    if not split_report_path.exists():
        build_split(root=root)
    split_report = read_json(split_report_path)
    if split_report["status"] != "PASS_CORRECTED_EXTERNAL_USER_SPLIT_FROZEN":
        raise RuntimeError("corrected external membership is not frozen")

    panels: dict[str, tuple[str, str, str]] = {}
    for partition, filename in PARTITION_FILES.items():
        for row in _rows(split_dir / filename):
            panel_id = str(row["panel_id"])
            panels[panel_id] = (partition, str(row["state_id"]), str(row["domain"]))
    calls = _rows(corrected_plan_path)
    if not calls:
        raise RuntimeError("corrected call plan is empty")
    if len({str(row["call_id"]) for row in calls}) != len(calls):
        raise RuntimeError("corrected call_id values are not unique")
    unknown = {str(row["panel_id"]) for row in calls} - set(panels)
    if unknown:
        raise RuntimeError(f"corrected plan contains unknown panels: {sorted(unknown)[:3]}")
    missing_panels = set(panels) - {str(row["panel_id"]) for row in calls}
    if missing_panels:
        raise RuntimeError(f"corrected plan omits frozen panels: {len(missing_panels)}")

    empty: list[tuple[str, str]] = []
    absent: list[tuple[str, str]] = []
    outcome_dependent: list[str] = []
    missing_execution_plan: list[str] = []
    prompt_hash_mismatch: list[str] = []
    partitions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        call_id = str(call["call_id"])
        partition, state_id, domain = panels[str(call["panel_id"])]
        if str(call["state_id"]) != state_id or str(call["domain"]) != domain:
            raise RuntimeError(f"corrected call disagrees with frozen panel: {call_id}")
        empty.extend((call_id, component) for component in _empty_requested_resources(call))
        absent.extend((call_id, component) for component in _resources_absent_from_messages(call))
        if bool(call.get("selection_or_prompt_uses_external_response_quality_risk_or_judge")):
            outcome_dependent.append(call_id)
        requested = set(call.get("requested_components") or [])
        if requested - set(call.get("execution_plans") or {}):
            missing_execution_plan.append(call_id)
        if str(call.get("messages_sha256")) != sha256_text(
            canonical_json(call.get("messages") or [])
        ):
            prompt_hash_mismatch.append(call_id)
        partitions[partition].append(call)
    if empty:
        raise RuntimeError(f"corrected plan still has empty requested resources: {empty[:3]}")
    if absent:
        raise RuntimeError(f"requested resource body is absent from messages: {absent[:3]}")
    if outcome_dependent:
        raise RuntimeError(f"corrected plan depends on external outcomes: {outcome_dependent[:3]}")
    if missing_execution_plan:
        raise RuntimeError(
            f"requested component lacks an execution plan: {missing_execution_plan[:3]}"
        )
    if prompt_hash_mismatch:
        raise RuntimeError(f"corrected prompt hash mismatch: {prompt_hash_mismatch[:3]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for partition in PARTITION_FILES:
        path = output_dir / f"{partition}_call_plan_private.jsonl"
        write_jsonl(path, partitions[partition])
        paths[partition] = path
    checks = {
        "all_326_frozen_panels_represented": len(
            {str(row["panel_id"]) for row in calls}
        )
        == 326,
        "all_requested_resource_bodies_nonempty": not empty,
        "all_requested_resource_bodies_present_in_messages": not absent,
        "all_requested_components_have_execution_plans": not missing_execution_plan,
        "all_prompt_hashes_match_corrected_messages": not prompt_hash_mismatch,
        "no_external_response_outcome_used_for_selection_or_prompt": all(
            not bool(row.get("selection_or_prompt_uses_external_response_quality_risk_or_judge"))
            for row in calls
        ),
        "partition_membership_matches_frozen_user_grain_split": sum(
            len(rows) for rows in partitions.values()
        )
        == len(calls),
    }
    report = {
        "protocol": PROTOCOL,
        "status": "PASS_CORRECTED_CALL_PLAN_MATERIALIZED_AND_PARTITIONED"
        if all(checks.values())
        else "FAIL_CORRECTED_CALL_PLAN_MATERIALIZATION",
        "corrected_plan_calls": len(calls),
        "partition_calls": {
            name: len(partitions[name]) for name in PARTITION_FILES
        },
        "checks": checks,
        "corrected_plan_sha256": sha256_file(corrected_plan_path),
        "membership_freeze_sha256": sha256_file(split_report_path),
        "outputs": {
            str(path): sha256_file(path) for path in paths.values()
        },
    }
    write_json(output_dir / "corrected_call_plan_partition_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrected-plan", type=Path)
    parser.add_argument("--partition-out", type=Path)
    args = parser.parse_args()
    report = build_split()
    result: dict[str, Any] = {"split": report}
    if args.corrected_plan:
        if not args.partition_out:
            parser.error("--partition-out is required with --corrected-plan")
        result["corrected_plan"] = validate_and_partition_corrected_plan(
            corrected_plan_path=args.corrected_plan,
            output_dir=args.partition_out,
        )
    print(
        {
            key: {"protocol": value["protocol"], "status": value["status"]}
            for key, value in result.items()
        }
    )


if __name__ == "__main__":
    main()
