#!/usr/bin/env python3
"""Build the single bounded 160-pair D3 quality-only blind review packet."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
from typing import Any

from metacom_pm.contracts import RuntimeState
from metacom_pm.io import canonical_json, iter_jsonl, read_json, sha256_file, sha256_text, stable_hex, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-d3-human-quality-blind-v1"
EXECUTION_PROTOCOL = "pm-v1.5-d3-generation-execution-v1"
STATUS = "READY_FOR_SINGLE_BOUNDED_160_PAIR_D3_QUALITY_REVIEW"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _template() -> str:
    path = ROOT / "scripts/v1_5/24bm_prepare_component_effect_blind_review_v1_5.py"
    spec = importlib.util.spec_from_file_location("_d3_html", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load blind-review template")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        str(module.HTML_TEMPLATE)
        .replace("PM V1.5 四组件效应盲评", "PM V1.5 D3 最终质量盲评")
        .replace("PM V1.5 四组件同状态盲评", "PM V1.5 D3 最终同状态质量盲评")
        .replace("pm_v1_5_four_component_effect_human_quality_blind_v1", "pm_v1_5_d3_human_quality_blind_v1")
        .replace("four_component_effect_quality_annotations.jsonl", "d3_quality_annotations.jsonl")
    )


def build(*, execution_dir: Path, blueprint_dir: Path, out_dir: Path) -> dict[str, Any]:
    summary = read_json(execution_dir / "execution_summary.json")
    outcomes = _rows(execution_dir / "generation_outcomes.jsonl")
    if (
        summary.get("protocol") != EXECUTION_PROTOCOL
        or summary.get("status") != "COMPLETE"
        or summary.get("completed_calls") != 320
        or len(outcomes) != 320
    ):
        raise RuntimeError("D3 generation is incomplete")
    pair_arms: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in outcomes:
        pair_arms[str(row["pair_id"])][str(row["arm"])] = row
    if len(pair_arms) != 160 or any(set(arms) != {"control", "treatment"} for arms in pair_arms.values()):
        raise RuntimeError("D3 response pairs are incomplete")
    states = {
        state.state_id: state
        for state in (
            RuntimeState.model_validate(row)
            for row in iter_jsonl(blueprint_dir / "runtime_states.jsonl")
        )
    }

    # Position balance is exact inside component × primary/repeat strata.
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for pair_id, arms in pair_arms.items():
        row = arms["control"]
        grouped[(str(row["component"]), str(row["pair_role"]))].append(pair_id)
    treatment_as_a: set[str] = set()
    for key, pair_ids in grouped.items():
        ordered = sorted(pair_ids, key=lambda value: stable_hex(PROTOCOL, "position", *key, value, n=32))
        if len(ordered) % 2:
            raise RuntimeError(f"cannot exactly balance A/B positions: {key}")
        treatment_as_a.update(ordered[: len(ordered) // 2])

    public: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    for pair_id, arms in pair_arms.items():
        control = arms["control"]
        a_role, b_role = ("treatment", "control") if pair_id in treatment_as_a else ("control", "treatment")
        state = states[str(control["state_id"])]
        blind_id = "d3_blind_" + stable_hex(PROTOCOL, pair_id, n=24)
        public.append(
            {
                "protocol": PROTOCOL,
                "blind_item_id": blind_id,
                "current_session_summary": state.current_session_summary,
                "recent_dialogue": [
                    {"role": turn.role, "content": turn.content}
                    for turn in state.current_session_history
                ],
                "current_user_text": state.current_user_text,
                "response_a": str(arms[a_role]["response"]),
                "response_b": str(arms[b_role]["response"]),
            }
        )
        private.append(
            {
                "protocol": PROTOCOL,
                "blind_item_id": blind_id,
                "pair_id": pair_id,
                "pair_role": control["pair_role"],
                "contrast_slot_id": control["contrast_slot_id"],
                "component": control["component"],
                "state_id": control["state_id"],
                "user_id": control["user_id"],
                "a_role": a_role,
                "b_role": b_role,
                "a_response_sha256": sha256_text(str(arms[a_role]["response"])),
                "b_response_sha256": sha256_text(str(arms[b_role]["response"])),
                "effect_label": "UNKNOWN_BEFORE_HUMAN_ANNOTATION",
            }
        )
    public.sort(key=lambda row: stable_hex(PROTOCOL, "display-order", row["blind_item_id"], n=32))
    private_by_id = {row["blind_item_id"]: row for row in private}
    private = [private_by_id[row["blind_item_id"]] for row in public]

    position_counts = Counter(
        (row["component"], row["pair_role"], row["a_role"])
        for row in private
    )
    public_keys = set().union(*(row.keys() for row in public))
    forbidden_public = {
        "pair_id", "pair_role", "contrast_slot_id", "component", "state_id", "user_id",
        "a_role", "b_role", "action_id", "selected_memory_ids", "selected_strategy_card_id",
    }
    checks = {
        "160_public_items": len(public) == 160,
        "160_private_keys": len(private) == 160,
        "unique_blind_ids": len({row["blind_item_id"] for row in public}) == 160,
        "components_40_each": Counter(row["component"] for row in private) == Counter({"RS": 40, "MP": 40, "MS": 40, "ME": 40}),
        "primary_repeat_128_32": Counter(row["pair_role"] for row in private) == Counter({"primary": 128, "outcome_blind_repeat": 32}),
        "exact_position_balance_per_stratum": all(
            position_counts[(component, role, "control")] == position_counts[(component, role, "treatment")]
            for component in ("RS", "MP", "MS", "ME")
            for role in ("primary", "outcome_blind_repeat")
        ),
        "private_fields_absent_from_public": not bool(public_keys & forbidden_public),
        "responses_nonempty_and_distinct": all(
            row["response_a"].strip() and row["response_b"].strip() and row["response_a"] != row["response_b"]
            for row in public
        ),
        "effect_labels_unknown": all(row["effect_label"].startswith("UNKNOWN") for row in private),
    }
    if not all(checks.values()):
        raise RuntimeError(f"D3 blind packet checks failed: {checks}")

    manifest = {
        "protocol": PROTOCOL,
        "status": STATUS,
        "unique_pairs": 160,
        "primary_pairs": 128,
        "outcome_blind_repeat_pairs": 32,
        "component_counts": {"RS": 40, "MP": 40, "MS": 40, "ME": 40},
        "quality_rule": (
            "Choose A or B only when it is materially better under the visible dialogue. "
            "Use tie for merely stylistic or slight preference; uncertain only when the "
            "visible evidence cannot support a reliable comparison."
        ),
        "decisive_criteria": [
            "visible_context_fidelity",
            "request_and_dialogue_fit",
            "emotional_understanding",
            "immediate_helpfulness",
            "clarity_naturalness",
            "materially_equivalent",
            "uncertain",
        ],
        "risk_not_scored_here": True,
        "component_arm_repeat_identity_hidden": True,
        "additional_second_reviewer_required": False,
        "checks": checks,
        "effect_labels_created": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    packet_path = out_dir / "human_blind_packet.jsonl"
    key_path = out_dir / "private_blind_key.jsonl"
    blank_path = out_dir / "blank_quality_annotations.jsonl"
    html_path = out_dir / "human_blind_review.html"
    write_jsonl(packet_path, public)
    write_jsonl(key_path, private)
    write_jsonl(
        blank_path,
        [
            {
                "protocol": PROTOCOL,
                "blind_item_id": row["blind_item_id"],
                "quality_preference": None,
                "decisive_criterion": None,
                "quality_notes": "",
                "annotator_id": "",
            }
            for row in public
        ],
    )
    html_path.write_text(
        _template().replace("__DATA__", canonical_json({"manifest": manifest, "items": public})),
        encoding="utf-8",
    )
    manifest["outputs"] = {
        path.name: sha256_file(path)
        for path in (packet_path, key_path, blank_path, html_path)
    }
    manifest["generation_outcomes_sha256"] = sha256_file(
        execution_dir / "generation_outcomes.jsonl"
    )
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-dir", type=Path, default=ROOT / "outputs/pm_v1_5_d3_generation_v1_execution")
    parser.add_argument("--blueprint-dir", type=Path, default=ROOT / "outputs/pm_v1_5_d3_step0_blueprint_v1")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/pm_v1_5_d3_blind_review_v1")
    args = parser.parse_args()
    manifest = build(execution_dir=args.execution_dir, blueprint_dir=args.blueprint_dir, out_dir=args.out_dir)
    print(json.dumps({key: manifest[key] for key in ("protocol", "status", "unique_pairs", "primary_pairs", "outcome_blind_repeat_pairs")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
