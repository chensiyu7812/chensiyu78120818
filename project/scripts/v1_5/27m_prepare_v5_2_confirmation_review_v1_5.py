#!/usr/bin/env python3
"""Prepare the single frozen V5.2 confirmation quality and risk panels.

The primary panel covers every generated ON/OFF pair and every generated arm.
An independent overlap is selected only from protocol-hashed counterfactual
group identities.  No response content, label, policy choice, or outcome is
used to choose overlap rows or A/B positions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    stable_hex,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-v5.2-content-disjoint-confirmation-review-v1"
QUALITY_PROTOCOL = "pm-v1.5-v5.2-confirmation-quality-blind-v1"
RISK_PROTOCOL = "pm-v1.5-v5.2-confirmation-grounding-risk-v1"
COMPONENTS = ("MP", "MS", "ME", "RS")
QUALITY_HELPER = ROOT / "scripts/v1_5/25zy_prepare_v5_full_fit_outcome_review_v1_5.py"
RISK_HELPER = ROOT / "scripts/v1_5/26i_prepare_v5_1_confirmation_review_v1_5.py"


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load review helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def visible_conversation(call: dict[str, Any]) -> str:
    user = [str(item["content"]) for item in call["messages"] if item["role"] == "user"]
    prefix = "Visible current conversation:"
    if len(user) != 1 or not user[0].startswith(prefix):
        raise RuntimeError(f"invalid visible prompt: {call['call_id']}")
    return user[0][len(prefix) :].strip()


def authorization(call: dict[str, Any]) -> str:
    if call["arm"] == "OFF":
        return (
            "Only the visible current conversation was authorized. "
            "No user-specific memory or strategy resource was authorized."
        )
    plan = dict(call["composition_plan"])
    parts: list[str] = []
    for clause in plan["locked_clauses"]:
        parts.append(
            "Backend-locked authorized content (must remain source-grounded): "
            + str(clause["text"])
        )
    preference = str(plan["response_preference"]).strip()
    if preference:
        parts.append("Authorized active response-format preference: " + preference)
    strategy = str(plan["strategy_instruction"]).strip()
    if strategy:
        parts.append("Authorized atomic support instruction: " + strategy)
    if not parts:
        raise RuntimeError(f"ON call lacks a public authorization surface: {call['call_id']}")
    text = "\n".join(parts)
    if any(marker in text for marker in ("mem_", "strategy_v", "card_")):
        raise RuntimeError(f"opaque resource ID leaked into review surface: {call['call_id']}")
    return text


def repeated_signature_counts(records: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, int]:
    counts = Counter(canonical_json([record[field] for field in fields]) for record in records)
    repeated_groups = sum(count > 1 for count in counts.values())
    repeated_rows = sum(count for count in counts.values() if count > 1)
    return {"groups": repeated_groups, "rows": repeated_rows}


def deduplicate(
    records: list[dict[str, Any]],
    *,
    id_field: str,
    signature: Callable[[dict[str, Any]], str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Return one public-surface representative plus an all-row propagation map."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[signature(record)].append(record)
    representatives: list[dict[str, Any]] = []
    representative_by_id: dict[str, str] = {}
    for members in grouped.values():
        representative = min(members, key=lambda row: str(row[id_field]))
        representatives.append(representative)
        for member in members:
            representative_by_id[str(member[id_field])] = str(representative[id_field])
    representatives.sort(key=lambda row: str(row[id_field]))
    return representatives, representative_by_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_2_confirmation_plan_v1",
    )
    parser.add_argument(
        "--execution-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_2_confirmation_execution_v1",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "data/pm_v1_5_contracts/v5_2_content_disjoint_confirmation_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_2_confirmation_review_v1_candidate",
    )
    args = parser.parse_args()

    manifest = read_json(args.plan_dir / "freeze_manifest.json")
    summary = read_json(args.execution_dir / "execution_summary.json")
    contract = read_json(args.contract)
    if manifest["status"] != "READY_FOR_SINGLE_V5_2_CONTENT_DISJOINT_CONFIRMATION_EXECUTION":
        raise RuntimeError("V5.2 confirmation plan is not frozen")
    if summary["status"] != "COMPLETE_AWAITING_SINGLE_CONFIRMATION_OUTCOME_PANEL":
        raise RuntimeError("V5.2 confirmation execution is incomplete")
    if summary["completed_calls"] != 512 or summary["fallback_calls"] or summary["invalid_itt_rows"]:
        raise RuntimeError("V5.2 confirmation contains missing, fallback, or invalid rows")
    if contract["measurement"]["human_review"].split(";", 1)[0] != (
        "one complete primary quality/risk panel"
    ):
        raise RuntimeError("unexpected frozen human-review contract")

    calls = {str(row["call_id"]): row for row in rows(args.plan_dir / "call_plan_private.jsonl")}
    outcomes = {
        str(row["call_id"]): row
        for row in rows(args.execution_dir / "outcomes_ordered_private.jsonl")
    }
    bindings = {
        str(row["state_id"]): row
        for row in rows(args.plan_dir / "state_policy_bindings_private.jsonl")
    }
    if len(calls) != 512 or set(calls) != set(outcomes) or len(bindings) != 128:
        raise RuntimeError("confirmation plan, outcome, or policy-binding identity mismatch")

    paired: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    for call_id, call in calls.items():
        paired[(str(call["state_id"]), int(call["seed"]))][str(call["arm"])] = call_id
    if len(paired) != 256 or any(set(arms) != {"ON", "OFF"} for arms in paired.values()):
        raise RuntimeError("expected 256 complete same-state same-seed ON/OFF pairs")

    by_component: dict[str, list[tuple[str, int]]] = {component: [] for component in COMPONENTS}
    for state_seed, arms in paired.items():
        component = str(calls[arms["ON"]]["target_component"])
        by_component[component].append(state_seed)
    on_as_a: set[tuple[str, int]] = set()
    for component, values in by_component.items():
        ordered = sorted(
            values,
            key=lambda value: stable_hex(
                PROTOCOL, "position", component, value[0], str(value[1]), n=32
            ),
        )
        if len(ordered) != 64:
            raise RuntimeError(f"expected 64 quality pairs for {component}")
        on_as_a.update(ordered[:32])

    quality_rows: list[dict[str, Any]] = []
    quality_key: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    risk_key: list[dict[str, Any]] = []
    for state_seed, arms in sorted(paired.items()):
        on_id, off_id = arms["ON"], arms["OFF"]
        on_call, off_call = calls[on_id], calls[off_id]
        conversation = visible_conversation(on_call)
        if conversation != visible_conversation(off_call):
            raise RuntimeError(f"visible conversation differs inside pair: {state_seed}")
        component = str(on_call["target_component"])
        group_id = str(on_call["counterfactual_group_id_private_analysis_only"])
        if group_id != str(off_call["counterfactual_group_id_private_analysis_only"]):
            raise RuntimeError(f"counterfactual group differs inside pair: {state_seed}")
        first, second = (
            (on_id, off_id) if state_seed in on_as_a else (off_id, on_id)
        )
        blind_id = "v52cq_" + stable_hex(
            PROTOCOL, "quality", state_seed[0], str(state_seed[1]), n=24
        )
        quality_rows.append(
            {
                "protocol": QUALITY_PROTOCOL,
                "blind_item_id": blind_id,
                "visible_conversation": conversation,
                "response_a": str(outcomes[first]["final_response"]),
                "response_b": str(outcomes[second]["final_response"]),
            }
        )
        quality_key.append(
            {
                "protocol": PROTOCOL,
                "blind_item_id": blind_id,
                "state_id": state_seed[0],
                "seed": state_seed[1],
                "component": component,
                "counterfactual_group_id": group_id,
                "response_a_call_id": first,
                "response_a_arm": str(calls[first]["arm"]),
                "response_b_call_id": second,
                "response_b_arm": str(calls[second]["arm"]),
                "policy_decisions": bindings[state_seed[0]]["policy_decisions"],
            }
        )
        for arm, call_id in (("ON", on_id), ("OFF", off_id)):
            risk_id = "v52cr_" + stable_hex(PROTOCOL, "risk", call_id, n=24)
            risk_rows.append(
                {
                    "protocol": RISK_PROTOCOL,
                    "risk_item_id": risk_id,
                    "visible_conversation": conversation,
                    "authorized_evidence_and_instruction": authorization(calls[call_id]),
                    "candidate_response": str(outcomes[call_id]["final_response"]),
                }
            )
            risk_key.append(
                {
                    "protocol": PROTOCOL,
                    "risk_item_id": risk_id,
                    "call_id": call_id,
                    "state_id": state_seed[0],
                    "seed": state_seed[1],
                    "component": component,
                    "arm": arm,
                    "counterfactual_group_id": group_id,
                    "policy_decisions": bindings[state_seed[0]]["policy_decisions"],
                }
            )

    quality_manual, quality_representative = deduplicate(
        quality_rows,
        id_field="blind_item_id",
        signature=lambda row: canonical_json(
            [row["visible_conversation"], row["response_a"], row["response_b"]]
        ),
    )
    risk_manual, risk_representative = deduplicate(
        risk_rows,
        id_field="risk_item_id",
        signature=lambda row: canonical_json(
            [
                row["visible_conversation"],
                row["authorized_evidence_and_instruction"],
                row["candidate_response"],
            ]
        ),
    )
    for row in quality_key:
        row["manual_representative_id"] = quality_representative[str(row["blind_item_id"])]
    for row in risk_key:
        row["manual_representative_id"] = risk_representative[str(row["risk_item_id"])]

    groups_by_component: dict[str, set[str]] = {component: set() for component in COMPONENTS}
    for row in quality_key:
        groups_by_component[str(row["component"])].add(str(row["counterfactual_group_id"]))
    if any(len(groups) != 16 for groups in groups_by_component.values()):
        raise RuntimeError("expected 16 confirmation counterfactual groups per component")

    # Thirteen of 64 groups (20.3125%) are selected without reading responses.
    overlap_groups: set[str] = set()
    remaining_groups: list[tuple[str, str]] = []
    for component, groups in groups_by_component.items():
        ordered = sorted(
            groups,
            key=lambda group: stable_hex(PROTOCOL, "overlap", component, group, n=32),
        )
        overlap_groups.update(ordered[:3])
        remaining_groups.extend((component, group) for group in ordered[3:])
    extra_component, extra_group = min(
        remaining_groups,
        key=lambda item: stable_hex(PROTOCOL, "overlap-extra", item[0], item[1], n=32),
    )
    del extra_component
    overlap_groups.add(extra_group)
    if len(overlap_groups) != 13:
        raise RuntimeError("protocol-hashed overlap must contain exactly 13 groups")

    quality_group = {
        str(row["blind_item_id"]): str(row["counterfactual_group_id"])
        for row in quality_key
    }
    risk_group = {
        str(row["risk_item_id"]): str(row["counterfactual_group_id"])
        for row in risk_key
    }
    overlap_quality_underlying = [
        row for row in quality_rows if quality_group[str(row["blind_item_id"])] in overlap_groups
    ]
    overlap_risk_underlying = [
        row for row in risk_rows if risk_group[str(row["risk_item_id"])] in overlap_groups
    ]
    if len(overlap_quality_underlying) != 52 or len(overlap_risk_underlying) != 104:
        raise RuntimeError("unexpected underlying 20% overlap size")
    quality_manual_by_id = {str(row["blind_item_id"]): row for row in quality_manual}
    risk_manual_by_id = {str(row["risk_item_id"]): row for row in risk_manual}
    overlap_quality_ids = {
        quality_representative[str(row["blind_item_id"])] for row in overlap_quality_underlying
    }
    overlap_risk_ids = {
        risk_representative[str(row["risk_item_id"])] for row in overlap_risk_underlying
    }
    overlap_quality = [quality_manual_by_id[item_id] for item_id in sorted(overlap_quality_ids)]
    overlap_risk = [risk_manual_by_id[item_id] for item_id in sorted(overlap_risk_ids)]

    quality_helper = load_module(QUALITY_HELPER, "v52_confirmation_quality_helper")
    risk_helper = load_module(RISK_HELPER, "v52_confirmation_risk_helper")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "primary_quality_packet.jsonl", quality_manual)
    write_jsonl(args.out_dir / "primary_risk_packet.jsonl", risk_manual)
    write_jsonl(args.out_dir / "overlap_quality_packet.jsonl", overlap_quality)
    write_jsonl(args.out_dir / "overlap_risk_packet.jsonl", overlap_risk)
    write_jsonl(args.out_dir / "private_quality_key.jsonl", quality_key)
    write_jsonl(args.out_dir / "private_risk_key.jsonl", risk_key)
    write_json(
        args.out_dir / "private_overlap_groups.json",
        {
            "protocol": PROTOCOL,
            "selection": "protocol-hashed counterfactual groups; response and label blind",
            "group_count": 13,
            "groups": sorted(overlap_groups),
        },
    )

    def write_html(
        path: Path,
        template: str,
        protocol: str,
        items: list[dict[str, Any]],
        role: str,
    ) -> None:
        payload = {
            "manifest": {"protocol": PROTOCOL, "panel_role": role, "items": len(items)},
            "items": items,
        }
        path.write_text(
            template.replace("__DATA__", quality_helper._safe_script_json(payload)).replace(
                "__PROTOCOL__", protocol
            ),
            encoding="utf-8",
        )

    write_html(
        args.out_dir / "human_quality_primary.html",
        quality_helper.QUALITY_HTML,
        QUALITY_PROTOCOL,
        quality_manual,
        "primary",
    )
    write_html(
        args.out_dir / "human_risk_primary.html",
        risk_helper.RISK_HTML,
        RISK_PROTOCOL,
        risk_manual,
        "primary",
    )
    write_html(
        args.out_dir / "human_quality_overlap.html",
        quality_helper.QUALITY_HTML,
        QUALITY_PROTOCOL,
        overlap_quality,
        "independent_overlap",
    )
    write_html(
        args.out_dir / "human_risk_overlap.html",
        risk_helper.RISK_HTML,
        RISK_PROTOCOL,
        overlap_risk,
        "independent_overlap",
    )

    report = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_SINGLE_V5_2_CONFIRMATION_PRIMARY_AND_INDEPENDENT_REVIEW",
        "underlying": {"quality_pairs": 256, "risk_arm_responses": 512},
        "primary": {"quality": len(quality_manual), "risk": len(risk_manual)},
        "independent_overlap": {
            "counterfactual_groups": len(overlap_groups),
            "underlying_quality": len(overlap_quality_underlying),
            "underlying_risk": len(overlap_risk_underlying),
            "quality": len(overlap_quality),
            "risk": len(overlap_risk),
        },
        "quality_position_distribution": dict(
            Counter(row["response_a_arm"] for row in quality_key)
        ),
        "risk_arm_distribution": dict(Counter(row["arm"] for row in risk_key)),
        "exact_repeat_audit": {
            "quality": repeated_signature_counts(
                quality_rows, ("visible_conversation", "response_a", "response_b")
            ),
            "risk": repeated_signature_counts(
                risk_rows,
                (
                    "visible_conversation",
                    "authorized_evidence_and_instruction",
                    "candidate_response",
                ),
            ),
        },
        "exact_public_surface_deduplication": {
            "quality_rows_propagated": len(quality_rows) - len(quality_manual),
            "risk_rows_propagated": len(risk_rows) - len(risk_manual),
            "underlying_statistical_rows_retained": True,
        },
        "quality_blind_to_component_arm_resource_and_policy": True,
        "overlap_selection_response_label_and_policy_blind": True,
        "function_review_required": False,
        "reason_function_not_repeated": (
            "Step2 functional-use is a mechanism diagnostic already measured on FIT; "
            "the frozen confirmation estimand uses quality, material risk, and cost."
        ),
        "inputs_sha256": {
            "contract": sha256_file(args.contract),
            "plan_manifest": sha256_file(args.plan_dir / "freeze_manifest.json"),
            "call_plan": sha256_file(args.plan_dir / "call_plan_private.jsonl"),
            "policy_bindings": sha256_file(
                args.plan_dir / "state_policy_bindings_private.jsonl"
            ),
            "execution_summary": sha256_file(args.execution_dir / "execution_summary.json"),
            "outcomes": sha256_file(args.execution_dir / "outcomes_ordered_private.jsonl"),
        },
        "method_policy_or_response_changed_after_confirmation": False,
    }
    write_json(args.out_dir / "review_manifest.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
