#!/usr/bin/env python3
"""Prepare the frozen T5 stratified human quality and grounding-risk panels.

The sampling design and overlap flags were frozen before generation.  This
script only joins those bindings to completed outcomes.  Byte-identical
responses are recorded as automatic quality ties; they are never sent to a
human merely to inflate the panel size.  Quality remains blind to policy and
resource identity, while risk sees only the evidence authorized for that
specific generated response.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
from typing import Any

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
PROTOCOL = "pm-v1.5-v5.1-t5-human-review-packet-v1"
QUALITY_PROTOCOL = "pm-v1.5-v5.1-t5-human-quality-blind-v1"
RISK_PROTOCOL = "pm-v1.5-v5.1-t5-human-grounding-risk-v1"
PLAN_DIR = ROOT / "outputs/pm_v1_5_v5_1_t5_plan_v1"
EXECUTION_DIR = ROOT / "outputs/pm_v1_5_v5_1_t5_execution_v1"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_1_t5_human_review_v1_candidate"
QUALITY_HELPER = ROOT / "scripts/v1_5/25zy_prepare_v5_full_fit_outcome_review_v1_5.py"
RISK_HELPER = ROOT / "scripts/v1_5/26i_prepare_v5_1_confirmation_review_v1_5.py"


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_script_json(value: Any) -> str:
    return (
        canonical_json(value)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def visible_conversation(call: dict[str, Any]) -> str:
    user = [str(message["content"]) for message in call["messages"] if message["role"] == "user"]
    if len(user) != 1 or not user[0].startswith("Visible current conversation:\n"):
        raise RuntimeError(f"unexpected visible prompt: {call['call_id']}")
    return user[0].split("Visible current conversation:\n", 1)[1].strip()


def authorization(call: dict[str, Any]) -> str:
    system = [str(message["content"]) for message in call["messages"] if message["role"] == "system"]
    if len(system) != 1:
        raise RuntimeError(f"unexpected system prompt: {call['call_id']}")
    text = system[0]
    marker = "\n\nResource 1\n"
    if marker not in text:
        return (
            "Authorized evidence: visible current conversation only. "
            "No prior user-specific memory or strategy resource was authorized for this reply."
        )
    public = text.split(marker, 1)[1].strip()
    for opaque in ("mem_", "strat_", "card_"):
        if opaque in public:
            raise RuntimeError(f"opaque resource ID leaked into review evidence: {call['call_id']}")
    return public


def call_key(state_id: str, seed_hex: str, action: str) -> tuple[str, str, str]:
    return state_id, seed_hex, action


def packet(items: list[dict[str, Any]], protocol: str, role: str) -> dict[str, Any]:
    return {
        "manifest": {
            "protocol": protocol,
            "panel_role": role,
            "items": len(items),
            "sampling_frozen_before_generation": True,
        },
        "items": items,
    }


def build() -> dict[str, Any]:
    manifest = read_json(PLAN_DIR / "freeze_manifest.json")
    summary = read_json(EXECUTION_DIR / "execution_summary.json")
    if manifest.get("status") != "READY_FOR_SINGLE_T5_GENERATION":
        raise RuntimeError("T5 plan is not frozen")
    if summary.get("status") != "COMPLETE_AWAITING_FROZEN_AUTOMATIC_JUDGE_AND_HUMAN_EVALUATION":
        raise RuntimeError("T5 generation is incomplete")
    if summary.get("call_plan_sha256") != manifest.get("call_plan_sha256"):
        raise RuntimeError("execution does not bind the frozen T5 call plan")

    calls = rows(PLAN_DIR / "call_plan_private.jsonl")
    outcomes = rows(EXECUTION_DIR / "outcomes_ordered_private.jsonl")
    design = rows(PLAN_DIR / "human_review_design_private.jsonl")
    call_by_key_map = {
        call_key(str(row["state_id"]), str(row["seed_hex"]), str(row["requested_action_id"])): row
        for row in calls
    }
    outcome_by_call = {str(row["call_id"]): row for row in outcomes}
    if len(call_by_key_map) != len(calls) or len(outcome_by_call) != len(outcomes):
        raise RuntimeError("T5 call or outcome identity is not unique")
    if len(design) != int(manifest["human_final"]["quality_pairs"]):
        raise RuntimeError("human design count differs from frozen manifest")

    # Assign position from design identifiers only, exactly balanced inside
    # each partition x comparator stratum (apart from an unavoidable odd one).
    position_strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in design:
        position_strata[(str(row["partition"]), str(row["policy_b"]))].append(row)
    learned_as_a_ids: set[str] = set()
    for stratum, selected in position_strata.items():
        ordered = sorted(
            selected,
            key=lambda row: stable_hex(
                PROTOCOL, "human-position", *stratum, str(row["comparison_id"]), n=24
            ),
        )
        learned_as_a_ids.update(
            str(row["comparison_id"]) for row in ordered[: (len(ordered) + 1) // 2]
        )

    quality_primary: list[dict[str, Any]] = []
    quality_overlap: list[dict[str, Any]] = []
    quality_key: list[dict[str, Any]] = []
    automatic_ties: list[dict[str, Any]] = []
    primary_risk_calls: set[str] = set()
    overlap_risk_calls: set[str] = set()

    for row in design:
        state_id = str(row["state_id"])
        seed_hex = str(row["seed_hex"])
        call_learned = call_by_key_map[call_key(state_id, seed_hex, str(row["action_a"]))]
        call_comparator = call_by_key_map[call_key(state_id, seed_hex, str(row["action_b"]))]
        out_learned = outcome_by_call[str(call_learned["call_id"])]
        out_comparator = outcome_by_call[str(call_comparator["call_id"])]
        response_learned = str(out_learned["final_response"])
        response_comparator = str(out_comparator["final_response"])
        blind_id = "t5hq_" + stable_hex(PROTOCOL, str(row["comparison_id"]), n=24)
        learned_as_a = str(row["comparison_id"]) in learned_as_a_ids
        first_call, second_call = (
            (call_learned, call_comparator) if learned_as_a else (call_comparator, call_learned)
        )
        first_response, second_response = (
            (response_learned, response_comparator) if learned_as_a else (response_comparator, response_learned)
        )
        identical = response_learned == response_comparator
        key_row = {
            "protocol": PROTOCOL,
            "blind_item_id": blind_id,
            "comparison_id": row["comparison_id"],
            "domain": row["domain"],
            "partition": row["partition"],
            "state_id": state_id,
            "group_id_private_analysis_only": row["group_id_private_analysis_only"],
            "policy_a_presented": row["policy_a"] if learned_as_a else row["policy_b"],
            "policy_b_presented": row["policy_b"] if learned_as_a else row["policy_a"],
            "call_id_a": first_call["call_id"],
            "call_id_b": second_call["call_id"],
            "response_byte_identical": identical,
            "primary_full_comparison": row["primary_full_comparison"],
            "secondary_stratified_quarter_sample": row["secondary_stratified_quarter_sample"],
            "independent_second_reviewer_overlap": row["independent_second_reviewer_overlap"],
        }
        quality_key.append(key_row)
        # Risk is a per-response construct, not a pairwise quality construct.
        # Keep every unique response in the frozen human design even when two
        # policy aliases share a byte-identical output and quality is auto-tied.
        primary_risk_calls.update((str(call_learned["call_id"]), str(call_comparator["call_id"])))
        if bool(row["independent_second_reviewer_overlap"]):
            overlap_risk_calls.update((str(call_learned["call_id"]), str(call_comparator["call_id"])))
        if identical:
            automatic_ties.append({
                "protocol": QUALITY_PROTOCOL,
                "blind_item_id": blind_id,
                "quality_preference": "tie",
                "decisive_criterion": "materially_equivalent",
                "quality_notes": "Byte-identical responses; automatic exact-identity tie.",
                "annotator_id": "deterministic_exact_identity",
            })
            continue
        item = {
            "protocol": QUALITY_PROTOCOL,
            "blind_item_id": blind_id,
            "visible_conversation": visible_conversation(call_learned),
            "response_a": first_response,
            "response_b": second_response,
        }
        quality_primary.append(item)
        if bool(row["independent_second_reviewer_overlap"]):
            quality_overlap.append(item)

    def risk_items(call_ids: set[str], protocol: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        items: list[dict[str, Any]] = []
        keys: list[dict[str, Any]] = []
        calls_by_id = {str(row["call_id"]): row for row in calls}
        for call_id in sorted(call_ids):
            call = calls_by_id[call_id]
            outcome = outcome_by_call[call_id]
            risk_id = "t5hr_" + stable_hex(protocol, call_id, n=24)
            items.append({
                "protocol": protocol,
                "risk_item_id": risk_id,
                "visible_conversation": visible_conversation(call),
                "authorized_evidence_and_instruction": authorization(call),
                "candidate_response": str(outcome["final_response"]),
            })
            keys.append({
                "protocol": PROTOCOL,
                "risk_item_id": risk_id,
                "call_id": call_id,
                "domain": call["domain"],
                "partition": call["partition"],
                "state_id": call["state_id"],
                "group_id_private_analysis_only": call["group_id_private_analysis_only"],
                "seed_hex": call["seed_hex"],
                "requested_action_id": call["requested_action_id"],
                "realized_action_id": outcome["realized_action_id"],
                "fallback_used": outcome["fallback_used"],
            })
        return items, keys

    risk_primary, risk_key_primary = risk_items(primary_risk_calls, RISK_PROTOCOL)
    overlap_risk_protocol = RISK_PROTOCOL + "-independent-overlap"
    risk_overlap, risk_key_overlap = risk_items(overlap_risk_calls, overlap_risk_protocol)

    quality_helper = load_module(QUALITY_HELPER, "t5_quality_helper")
    risk_helper = load_module(RISK_HELPER, "t5_risk_helper")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {
        "quality_primary": packet(quality_primary, QUALITY_PROTOCOL, "primary"),
        "quality_overlap": packet(quality_overlap, QUALITY_PROTOCOL, "independent_overlap"),
        "risk_primary": packet(risk_primary, RISK_PROTOCOL, "primary"),
        "risk_overlap": packet(risk_overlap, overlap_risk_protocol, "independent_overlap"),
    }
    for name, value in artifacts.items():
        write_json(OUT_DIR / f"{name}_packet.json", value)
    write_jsonl(OUT_DIR / "private_quality_key.jsonl", quality_key)
    write_jsonl(OUT_DIR / "private_risk_primary_key.jsonl", risk_key_primary)
    write_jsonl(OUT_DIR / "private_risk_overlap_key.jsonl", risk_key_overlap)
    write_jsonl(OUT_DIR / "automatic_exact_identity_ties.jsonl", automatic_ties)
    for role in ("primary", "overlap"):
        q_packet = artifacts[f"quality_{role}"]
        r_packet = artifacts[f"risk_{role}"]
        (OUT_DIR / f"human_quality_{role}.html").write_text(
            quality_helper.QUALITY_HTML.replace("__DATA__", safe_script_json(q_packet)).replace("__PROTOCOL__", QUALITY_PROTOCOL),
            encoding="utf-8",
        )
        risk_protocol = RISK_PROTOCOL if role == "primary" else overlap_risk_protocol
        (OUT_DIR / f"human_grounding_risk_{role}.html").write_text(
            risk_helper.RISK_HTML.replace("__DATA__", safe_script_json(r_packet)).replace("__PROTOCOL__", risk_protocol),
            encoding="utf-8",
        )

    report = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_FROZEN_STRATIFIED_HUMAN_FINAL",
        "designed_quality_comparisons": len(design),
        "automatic_exact_identity_ties": len(automatic_ties),
        "manual_quality_primary": len(quality_primary),
        "manual_quality_independent_overlap": len(quality_overlap),
        "manual_risk_unique_responses_primary": len(risk_primary),
        "manual_risk_unique_responses_independent_overlap": len(risk_overlap),
        "manual_quality_by_partition": dict(Counter(str(row["partition"]) for row in quality_key if not row["response_byte_identical"])),
        "automatic_ties_by_partition": dict(Counter(str(row["partition"]) for row in quality_key if row["response_byte_identical"])),
        "quality_position_distribution": dict(
            Counter(str(row["policy_a_presented"]) for row in quality_key)
        ),
        "quality_position_assignment": "response-blind exact balance within partition x comparator from frozen comparison IDs",
        "construct_separation": {
            "quality": "blind A/B adoption difference; no resource or policy identity",
            "risk": "single response with only its authorized evidence and boundary visible",
            "cost": "objective provider usage; not human-rated",
        },
        "inputs_sha256": {
            "freeze_manifest": sha256_file(PLAN_DIR / "freeze_manifest.json"),
            "human_design": sha256_file(PLAN_DIR / "human_review_design_private.jsonl"),
            "call_plan": sha256_file(PLAN_DIR / "call_plan_private.jsonl"),
            "execution_summary": sha256_file(EXECUTION_DIR / "execution_summary.json"),
            "outcomes": sha256_file(EXECUTION_DIR / "outcomes_ordered_private.jsonl"),
        },
        "outcome_used_to_change_sampling_or_method": False,
    }
    write_json(OUT_DIR / "review_manifest.json", report)
    return report


def main() -> None:
    print(json.dumps(build(), ensure_ascii=False))


if __name__ == "__main__":
    main()
