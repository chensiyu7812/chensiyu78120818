#!/usr/bin/env python3
"""Build primary-72 and independent-overlap-32 D2 blind review packets."""

from __future__ import annotations

import argparse
import importlib.util
from collections import Counter
from pathlib import Path
from typing import Any

from metacom_pm.contracts import RuntimeState
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    stable_hex,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-d2-measurement-human-quality-blind-v1"
EXECUTION_PROTOCOL = "pm-v1.5-d2-measurement-generation-execution-v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _template() -> str:
    path = (
        ROOT
        / "scripts/v1_5/24bm_prepare_component_effect_blind_review_v1_5.py"
    )
    spec = importlib.util.spec_from_file_location("_d2_html", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load blind-review template")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(module.HTML_TEMPLATE).replace(
        "pm_v1_5_four_component_effect_human_quality_blind_v1",
        "pm_v1_5_d2_measurement_human_quality_blind_v1",
    )


def _write_packet(
    *,
    role: str,
    pair_ids: list[str],
    pair_arms: dict[str, dict[str, dict[str, Any]]],
    states: dict[str, RuntimeState],
    out_dir: Path,
) -> dict[str, Any]:
    public: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        arms = pair_arms[pair_id]
        control = arms["control"]
        treatment = arms["treatment"]
        component = str(control["component"])
        swap = (
            int(
                stable_hex(
                    PROTOCOL, role, component, pair_id, n=8
                ),
                16,
            )
            % 2
            == 1
        )
        a_role, b_role = (
            ("treatment", "control")
            if swap
            else ("control", "treatment")
        )
        state = states[str(control["state_id"])]
        blind_id = "d2_blind_" + stable_hex(
            PROTOCOL, role, pair_id, n=24
        )
        public.append(
            {
                "protocol": PROTOCOL,
                "blind_item_id": blind_id,
                "current_session_summary": (
                    state.current_session_summary
                ),
                "recent_dialogue": [
                    {"role": turn.role, "content": turn.content}
                    for turn in state.current_session_history
                ],
                "current_user_text": state.current_user_text,
                "response_a": arms[a_role]["response"],
                "response_b": arms[b_role]["response"],
            }
        )
        private.append(
            {
                "protocol": PROTOCOL,
                "reviewer_role": role,
                "blind_item_id": blind_id,
                "pair_id": pair_id,
                "d2_state_id": control["d2_state_id"],
                "replicate_index": control["replicate_index"],
                "component": component,
                "state_id": control["state_id"],
                "user_id": control["user_id"],
                "a_role": a_role,
                "b_role": b_role,
                "a_response_sha256": sha256_text(
                    arms[a_role]["response"]
                ),
                "b_response_sha256": sha256_text(
                    arms[b_role]["response"]
                ),
                "effect_label": "UNKNOWN_BEFORE_HUMAN_ANNOTATION",
            }
        )
    public.sort(
        key=lambda row: stable_hex(
            PROTOCOL, role, "order", row["blind_item_id"], n=32
        )
    )
    manifest = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_BOUNDED_D2_BLIND_QUALITY_REVIEW",
        "reviewer_role": role,
        "unique_pairs": len(pair_ids),
        "component_counts": dict(
            sorted(
                Counter(
                    pair_arms[pair_id]["control"]["component"]
                    for pair_id in pair_ids
                ).items()
            )
        ),
        "quality_rule": (
            "Choose A or B only when it is materially better under the "
            "visible dialogue. Use tie for merely stylistic or slight "
            "preference; uncertain only when the visible evidence cannot "
            "support a reliable comparison."
        ),
        "decisive_criteria": [
            "visible_context_fidelity",
            "request_and_dialogue_fit",
            "emotional_understanding",
            "immediate_helpfulness",
            "clarity_naturalness",
            "materially_equivalent",
        ],
        "risk_not_scored_here": True,
        "component_and_arm_identity_hidden": True,
        "effect_labels_created": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    packet_path = out_dir / "human_blind_packet.jsonl"
    key_path = out_dir / "private_blind_key.jsonl"
    blank_path = out_dir / "blank_quality_annotations.jsonl"
    html_path = out_dir / "human_blind_review.html"
    manifest_path = out_dir / "manifest.json"
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
        _template().replace(
            "__DATA__",
            canonical_json({"manifest": manifest, "items": public}),
        ),
        encoding="utf-8",
    )
    manifest["outputs"] = {
        path.name: sha256_file(path)
        for path in (packet_path, key_path, blank_path, html_path)
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execution-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_d2_measurement_generation_v1_execution",
    )
    parser.add_argument(
        "--runtime-states",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_evo_style_synthetic_memory_v1_candidate/"
        "runtime_states.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_d2_measurement_blind_v1",
    )
    args = parser.parse_args()

    summary = read_json(args.execution_dir / "execution_summary.json")
    outcomes = _rows(args.execution_dir / "generation_outcomes.jsonl")
    if (
        summary.get("protocol") != EXECUTION_PROTOCOL
        or summary.get("status") != "COMPLETE"
        or summary.get("completed_calls") != 144
        or len(outcomes) != 144
    ):
        raise RuntimeError("D2 generation is incomplete")
    pair_arms: dict[str, dict[str, dict[str, Any]]] = {}
    for row in outcomes:
        pair_arms.setdefault(str(row["pair_id"]), {})[
            str(row["arm"])
        ] = row
    if len(pair_arms) != 72 or any(
        set(arms) != {"control", "treatment"}
        for arms in pair_arms.values()
    ):
        raise RuntimeError("D2 pairs are incomplete")
    states = {
        state.state_id: state
        for state in (
            RuntimeState.model_validate(row)
            for row in iter_jsonl(args.runtime_states)
        )
    }
    primary_ids = sorted(pair_arms)
    overlap_ids: list[str] = []
    for component in ("RS", "MP", "MS", "ME"):
        component_ids = sorted(
            (
                pair_id
                for pair_id, arms in pair_arms.items()
                if arms["control"]["component"] == component
            ),
            key=lambda pair_id: stable_hex(
                PROTOCOL, "second-review-overlap", pair_id, n=32
            ),
        )
        overlap_ids.extend(component_ids[:8])
    primary = _write_packet(
        role="primary",
        pair_ids=primary_ids,
        pair_arms=pair_arms,
        states=states,
        out_dir=args.out_dir / "primary_72",
    )
    secondary = _write_packet(
        role="independent_overlap",
        pair_ids=overlap_ids,
        pair_arms=pair_arms,
        states=states,
        out_dir=args.out_dir / "independent_overlap_32",
    )
    report = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_PRIMARY_72_AND_INDEPENDENT_OVERLAP_32",
        "api_calls_made": 0,
        "primary": primary,
        "independent_overlap": secondary,
        "shared_pair_count": len(overlap_ids),
        "shared_pairs_per_component": 8,
        "effect_labels_created": False,
        "generation_outcomes_sha256": sha256_file(
            args.execution_dir / "generation_outcomes.jsonl"
        ),
    }
    write_json(args.out_dir / "report.json", report)
    print(
        {
            "protocol": PROTOCOL,
            "status": report["status"],
            "primary_decisions": primary["unique_pairs"],
            "independent_overlap_decisions": (
                secondary["unique_pairs"]
            ),
        }
    )


if __name__ == "__main__":
    main()
