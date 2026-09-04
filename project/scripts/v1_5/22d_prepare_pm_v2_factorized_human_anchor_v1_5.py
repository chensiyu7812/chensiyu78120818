#!/usr/bin/env python3
"""Prepare a 32-item train-only human anchor for factorized weak learning."""

from __future__ import annotations

import argparse
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
from metacom_pm.pm_v2_contracts import PMV2Split
from metacom_pm.pm_v2_data import load_states
from metacom_pm.v1_5_factorized_weak_learning import (
    FACTORIZED_SIGNAL_AUDIT_PROTOCOL,
    select_human_anchor_rows,
)


ROOT = Path(__file__).resolve().parents[2]
ANCHOR_PROTOCOL = "pm-v1.5-factorized-small-sample-human-anchor-v1"
CANONICAL_ACTION_PAIR = {
    "MP": ("M0+R0", "MP+R0"),
    "MS": ("M0+R0", "MS+R0"),
    "ME": ("M0+R0", "ME+R0"),
    "RS": ("M0+R0", "M0+RS"),
}


def _require_learning_contract(path: Path) -> dict[str, Any]:
    contract = read_json(path)
    expected = str(contract.pop("contract_sha256"))
    if sha256_text(canonical_json(contract)) != expected:
        raise RuntimeError("factorized weak-learning contract self-hash mismatch")
    contract["contract_sha256"] = expected
    if (
        contract.get("protocol")
        != "pm-v1.5-factorized-small-sample-weak-learning-v1"
        or contract.get("scope", {}).get("internal_test_outcomes_opened")
        is not False
    ):
        raise RuntimeError("factorized weak-learning contract is incompatible")
    return contract


def _selected_context(outcome: dict[str, Any]) -> dict[str, Any]:
    memory = []
    for row in outcome.get("memory_view") or []:
        if isinstance(row, dict):
            memory.append(str(row.get("text") or ""))
        else:
            memory.append(str(row))
    strategy = []
    for row in outcome.get("strategy_view") or []:
        if isinstance(row, dict):
            strategy.append(
                {
                    key: str(row[key])
                    for key in (
                        "retrieval_text",
                        "guidance_text",
                        "example_response",
                    )
                    if row.get(key)
                }
            )
        else:
            strategy.append(str(row))
    return {
        "memory": memory,
        "strategy": strategy,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-audit", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--evaluator-contexts", type=Path, required=True)
    parser.add_argument("--action-outcomes", type=Path, required=True)
    parser.add_argument(
        "--learning-contract",
        type=Path,
        default=ROOT
        / "data"
        / "pm_v1_5_contracts"
        / "factorized_small_sample_weak_learning_v1.json",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4311)
    args = parser.parse_args()

    learning_contract = _require_learning_contract(args.learning_contract)
    audit = read_json(args.signal_audit)
    if (
        audit.get("protocol") != FACTORIZED_SIGNAL_AUDIT_PROTOCOL
        or audit.get("status") != "COMPLETE_ZERO_API_TRAIN_ONLY_DIAGNOSTIC"
        or audit.get("internal_or_external_outcomes_opened") is not False
    ):
        raise RuntimeError("factorized signal audit is not eligible")
    selected = select_human_anchor_rows(
        audit["longitudinal_synthetic"]["effect_rows"],
        per_component_per_stratum=2,
        seed=args.seed,
    )
    selected_ids = {str(row["state_id"]) for row in selected}

    states = {
        state.state_id: state
        for state in load_states(args.states)
        if state.split is PMV2Split.TRAIN and state.state_id in selected_ids
    }
    if set(states) != selected_ids:
        raise RuntimeError("selected human-anchor states are not exact train rows")
    contexts = {
        str(row["state_id"]): row
        for row in iter_jsonl(args.evaluator_contexts)
        if str(row.get("state_id")) in selected_ids
    }
    if set(contexts) != selected_ids:
        raise RuntimeError("human-anchor evaluator contexts are incomplete")
    outcomes: dict[tuple[str, str], dict[str, Any]] = {}
    required_outcomes = {
        (str(row["state_id"]), action)
        for row in selected
        for action in CANONICAL_ACTION_PAIR[str(row["component"])]
    }
    for row in iter_jsonl(args.action_outcomes):
        key = (str(row.get("state_id")), str(row.get("action_id")))
        if key not in required_outcomes:
            continue
        if key in outcomes:
            raise RuntimeError(f"duplicate human-anchor outcome: {key}")
        outcomes[key] = row
    if set(outcomes) != required_outcomes:
        raise RuntimeError("human-anchor outcomes are incomplete")

    packet: list[dict[str, Any]] = []
    template: list[dict[str, Any]] = []
    private_mapping: list[dict[str, Any]] = []
    for row in selected:
        state_id = str(row["state_id"])
        component = str(row["component"])
        control_action, treatment_action = CANONICAL_ACTION_PAIR[component]
        control = outcomes[(state_id, control_action)]
        treatment = outcomes[(state_id, treatment_action)]
        if (
            control.get("prompt_equivalence_id")
            and control.get("prompt_equivalence_id")
            == treatment.get("prompt_equivalence_id")
        ):
            raise RuntimeError("human-anchor pair has identical prompt outcomes")
        order_hash = sha256_text(
            f"{ANCHOR_PROTOCOL}|{args.seed}|{state_id}|{component}"
        )
        treatment_is_a = int(order_hash[:2], 16) % 2 == 0
        candidates = (
            (treatment, control) if treatment_is_a else (control, treatment)
        )
        blind_id = "human_fwl_" + sha256_text(
            f"{ANCHOR_PROTOCOL}|{state_id}|{component}"
        )[:16]
        state = states[state_id]
        context = contexts[state_id]
        packet.append(
            {
                "blind_item_id": blind_id,
                "visible_state": {
                    "current_user_text": state.current_user_text,
                    "recent_dialogue": [
                        turn.model_dump(mode="json")
                        for turn in state.current_session_history
                    ],
                    "current_session_summary": state.current_session_summary,
                },
                "authorized_user_context": context["authorized_user_context"],
                "candidate_a": {
                    "response": candidates[0]["response"],
                    "selected_context": _selected_context(candidates[0]),
                },
                "candidate_b": {
                    "response": candidates[1]["response"],
                    "selected_context": _selected_context(candidates[1]),
                },
            }
        )
        template.append(
            {
                "blind_item_id": blind_id,
                "overall_preference": None,
                "support_quality_preference": None,
                "evidence_handling_preference": None,
                "safety_preference": None,
                "confidence": None,
                "notes": "",
            }
        )
        private_mapping.append(
            {
                "blind_item_id": blind_id,
                "state_id": state_id,
                "user_id": state.user_id,
                "component": component,
                "judge_consensus_stratum": row["consensus"],
                "structural_target": row["structural_target"],
                "family_effects": row["family_effects"],
                "candidate_a_action": (
                    treatment_action if treatment_is_a else control_action
                ),
                "candidate_b_action": (
                    control_action if treatment_is_a else treatment_action
                ),
                "treatment_candidate": "A" if treatment_is_a else "B",
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    packet_path = args.out_dir / "human_blind_packet.jsonl"
    template_path = args.out_dir / "human_annotation_template.jsonl"
    mapping_path = args.out_dir / "private_mapping.jsonl"
    write_jsonl(packet_path, packet)
    write_jsonl(template_path, template)
    write_jsonl(mapping_path, private_mapping)
    binding = {
        "protocol": ANCHOR_PROTOCOL,
        "learning_contract_sha256": learning_contract["contract_sha256"],
        "signal_audit_file_sha256": sha256_file(args.signal_audit),
        "states_sha256": sha256_file(args.states),
        "evaluator_contexts_sha256": sha256_file(args.evaluator_contexts),
        "action_outcomes_sha256": sha256_file(args.action_outcomes),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "seed": args.seed,
        "packet_rows": len(packet),
        "packet_sha256": sha256_file(packet_path),
        "annotation_template_sha256": sha256_file(template_path),
        "private_mapping_sha256": sha256_file(mapping_path),
        "automatic_gold_role": False,
        "internal_or_external_outcomes_opened": False,
    }
    summary = {
        **binding,
        "binding_sha256": sha256_text(canonical_json(binding)),
    }
    write_json(args.out_dir / "summary.json", summary)
    print(summary["binding_sha256"])


if __name__ == "__main__":
    main()
