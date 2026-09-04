#!/usr/bin/env python3
"""Build the one blind packet: 64 RS + 192 memory on one repaired stack."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
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
PROTOCOL = (
    "pm-v1.5-transport-repaired-four-component-human-quality-blind-v1"
)
REPEAT_COUNT = 52


def _load_html_template() -> str:
    path = (
        ROOT
        / "scripts/v1_5/24bm_prepare_component_effect_blind_review_v1_5.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_legacy_blind_template", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load bounded blind-review template")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(module.HTML_TEMPLATE).replace(
        "pm_v1_5_four_component_effect_human_quality_blind_v1",
        "pm_v1_5_transport_repaired_four_component_quality_blind_v1",
    )


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _states(path: Path) -> dict[str, RuntimeState]:
    return {
        state.state_id: state
        for state in (
            RuntimeState.model_validate(row) for row in iter_jsonl(path)
        )
    }


def _public_item(
    *,
    state: RuntimeState,
    blind_item_id: str,
    response_a: str,
    response_b: str,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "blind_item_id": blind_item_id,
        "current_session_summary": state.current_session_summary,
        "recent_dialogue": [
            {"role": turn.role, "content": turn.content}
            for turn in state.current_session_history
        ],
        "current_user_text": state.current_user_text,
        "response_a": response_a,
        "response_b": response_b,
    }


def _realized_arms(
    *,
    registry_path: Path,
    outcomes_path: Path,
    component_filter: set[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    outcomes = {
        (str(row["contrast_slot_id"]), str(row["arm"])): dict(row)
        for row in iter_jsonl(outcomes_path)
        if str(row["component"]) in component_filter
    }
    arms: dict[tuple[str, str], dict[str, Any]] = {}
    for row in iter_jsonl(registry_path):
        if str(row["component"]) not in component_filter:
            continue
        key = (str(row["contrast_slot_id"]), str(row["arm"]))
        response = row.get("response")
        if response is None:
            response = outcomes[key]["response"]
        enriched = dict(row)
        enriched["response"] = str(response)
        enriched["response_sha256"] = sha256_text(str(response))
        arms[key] = enriched
    return arms


def build_packet(
    *,
    rs_blueprint_dir: Path,
    rs_plan_dir: Path,
    rs_execution_dir: Path,
    repaired_blueprint_dir: Path,
    repaired_plan_dir: Path,
    repaired_execution_dir: Path,
    repaired_runtime_states: Path,
    out_dir: Path,
) -> dict[str, Any]:
    rs_summary = read_json(rs_execution_dir / "generation_summary.json")
    repaired_summary = read_json(
        repaired_execution_dir / "generation_summary.json"
    )
    if (
        rs_summary.get("status") != "COMPLETE"
        or rs_summary.get("completed_calls") != 128
    ):
        raise RuntimeError("repaired RS execution is incomplete")
    if (
        repaired_summary.get("status") != "COMPLETE"
        or repaired_summary.get("completed_calls") != 384
    ):
        raise RuntimeError("transport-repaired execution is incomplete")

    contrasts: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(
        rs_blueprint_dir / "rs_contrast_blueprint.jsonl"
    ):
        contrasts[str(row["contrast_slot_id"])] = dict(row)
    for row in iter_jsonl(
        repaired_blueprint_dir / "memory_contrast_blueprint.jsonl"
    ):
        contrasts[str(row["contrast_slot_id"])] = dict(row)
    counts = Counter(str(row["component"]) for row in contrasts.values())
    if counts != Counter({"RS": 64, "MP": 64, "MS": 64, "ME": 64}):
        raise RuntimeError(f"unexpected replacement contrast counts: {counts}")
    if len(contrasts) != 256:
        raise RuntimeError("replacement contrasts are not unique")

    arms = _realized_arms(
        registry_path=rs_plan_dir / "response_arm_registry.jsonl",
        outcomes_path=rs_execution_dir / "generation_outcomes.jsonl",
        component_filter={"RS"},
    )
    repaired_arms = _realized_arms(
        registry_path=repaired_plan_dir / "response_arm_registry.jsonl",
        outcomes_path=repaired_execution_dir / "generation_outcomes.jsonl",
        component_filter={"MP", "MS", "ME"},
    )
    overlap = set(arms) & set(repaired_arms)
    if overlap:
        raise RuntimeError("old and repaired arm identities overlap")
    arms.update(repaired_arms)
    if len(arms) != 512:
        raise RuntimeError("expected 512 realized replacement arms")

    states = _states(repaired_runtime_states)

    repeat_slots = set(
        sorted(
            contrasts,
            key=lambda value: stable_hex(
                PROTOCOL, "reliability-repeat", value, n=32
            ),
        )[:REPEAT_COUNT]
    )
    primary_swap_by_slot: dict[str, bool] = {}
    for component in ("RS", "MP", "MS", "ME"):
        component_slots = sorted(
            (
                slot_id
                for slot_id, row in contrasts.items()
                if row["component"] == component
            ),
            key=lambda value: stable_hex(
                PROTOCOL,
                "position-balance",
                component,
                value,
                n=32,
            ),
        )
        if len(component_slots) != 64:
            raise RuntimeError(
                f"expected 64 contrasts for {component}"
            )
        primary_swap_by_slot.update(
            {
                slot_id: index < len(component_slots) // 2
                for index, slot_id in enumerate(component_slots)
            }
        )
    presentations: list[dict[str, Any]] = []
    private_key: list[dict[str, Any]] = []
    for slot_id, contrast in contrasts.items():
        primary_swap = primary_swap_by_slot[slot_id]
        presentations_for_slot = (
            ("primary", "repeat")
            if slot_id in repeat_slots
            else ("primary",)
        )
        for presentation in presentations_for_slot:
            blind_id = "transport_blind_" + stable_hex(
                PROTOCOL, slot_id, presentation, n=24
            )
            swap = (
                not primary_swap
                if presentation == "repeat"
                else primary_swap
            )
            role_a, role_b = (
                ("treatment", "control")
                if swap
                else ("control", "treatment")
            )
            state = states[str(contrast["state_id"])]
            presentations.append(
                _public_item(
                    state=state,
                    blind_item_id=blind_id,
                    response_a=arms[(slot_id, role_a)]["response"],
                    response_b=arms[(slot_id, role_b)]["response"],
                )
            )
            private_key.append(
                {
                    "protocol": PROTOCOL,
                    "blind_item_id": blind_id,
                    "contrast_slot_id": slot_id,
                    "repeat_group_id": "repeat_group_"
                    + stable_hex(PROTOCOL, slot_id, n=24),
                    "is_reliability_repeat": presentation == "repeat",
                    "component": contrast["component"],
                    "split": contrast["split"],
                    "state_id": contrast["state_id"],
                    "control_action": contrast["control_action"],
                    "treatment_action": contrast["treatment_action"],
                    "a_role": role_a,
                    "b_role": role_b,
                    "a_response_sha256": arms[(slot_id, role_a)][
                        "response_sha256"
                    ],
                    "b_response_sha256": arms[(slot_id, role_b)][
                        "response_sha256"
                    ],
                    "effect_label": "UNKNOWN_BEFORE_HUMAN_ANNOTATION",
                }
            )
    presentations.sort(
        key=lambda row: stable_hex(
            PROTOCOL,
            "presentation-order",
            row["blind_item_id"],
            n=32,
        )
    )
    if len(presentations) != 256 + REPEAT_COUNT:
        raise RuntimeError("unexpected presentation count")

    manifest = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_ONE_FINAL_BOUNDED_BLIND_QUALITY_REVIEW",
        "unique_contrasts": 256,
        "component_counts": dict(sorted(counts.items())),
        "all_components_on_transport_repaired_stack": True,
        "legacy_component_pairs_included": False,
        "reliability_repeats": REPEAT_COUNT,
        "primary_position_balance": {
            "control_as_a_per_component": 32,
            "treatment_as_a_per_component": 32,
        },
        "reliability_repeat_position_rule": (
            "Every repeated contrast reverses A/B roles."
        ),
        "review_presentations": len(presentations),
        "quality_rule": (
            "Choose A/B only for a material quality difference under visible "
            "dialogue; slight stylistic preference is tie."
        ),
        "tie_policy_after_unblinding": (
            "tie maps to component off because there is no material gain to "
            "justify added cost"
        ),
        "risk_stage": (
            "Only component-on quality winners receive the separate minimal "
            "material-risk review."
        ),
        "effect_labels_created": False,
        "inputs": {
            "repaired_rs_blueprint": str(
                (
                    rs_blueprint_dir / "rs_contrast_blueprint.jsonl"
                ).relative_to(ROOT)
            ),
            "repaired_memory_blueprint": str(
                (
                    repaired_blueprint_dir
                    / "memory_contrast_blueprint.jsonl"
                ).relative_to(ROOT)
            ),
            "repaired_rs_generation_outcomes": str(
                (
                    rs_execution_dir / "generation_outcomes.jsonl"
                ).relative_to(ROOT)
            ),
            "repaired_memory_generation_outcomes": str(
                (
                    repaired_execution_dir / "generation_outcomes.jsonl"
                ).relative_to(ROOT)
            ),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    packet_path = out_dir / "human_blind_packet.jsonl"
    key_path = out_dir / "private_blind_key.jsonl"
    annotations_path = out_dir / "blank_quality_annotations.jsonl"
    html_path = out_dir / "human_blind_review.html"
    manifest_path = out_dir / "manifest.json"
    write_jsonl(packet_path, presentations)
    write_jsonl(key_path, private_key)
    write_jsonl(
        annotations_path,
        [
            {
                "protocol": PROTOCOL,
                "blind_item_id": row["blind_item_id"],
                "quality_preference": None,
                "decisive_criterion": None,
                "quality_notes": "",
                "annotator_id": "",
            }
            for row in presentations
        ],
    )
    write_json(manifest_path, manifest)
    html_data = {"manifest": manifest, "items": presentations}
    html_path.write_text(
        _load_html_template().replace(
            "__DATA__", canonical_json(html_data)
        ),
        encoding="utf-8",
    )
    manifest["outputs"] = {
        path.name: sha256_file(path)
        for path in (
            packet_path,
            key_path,
            annotations_path,
            html_path,
        )
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rs-blueprint-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_rs_contrast_blueprint_v1",
    )
    parser.add_argument(
        "--rs-plan-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_rs_generation_v1",
    )
    parser.add_argument(
        "--rs-execution-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_rs_generation_v1_execution",
    )
    parser.add_argument(
        "--repaired-blueprint-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_memory_contrast_blueprint_v1",
    )
    parser.add_argument(
        "--repaired-plan-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_memory_generation_v1",
    )
    parser.add_argument(
        "--repaired-execution-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_memory_generation_v1_execution",
    )
    parser.add_argument(
        "--repaired-runtime-states",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_evo_style_synthetic_memory_v1_candidate/"
        "runtime_states.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_four_component_blind_v1",
    )
    args = parser.parse_args()
    manifest = build_packet(
        rs_blueprint_dir=args.rs_blueprint_dir,
        rs_plan_dir=args.rs_plan_dir,
        rs_execution_dir=args.rs_execution_dir,
        repaired_blueprint_dir=args.repaired_blueprint_dir,
        repaired_plan_dir=args.repaired_plan_dir,
        repaired_execution_dir=args.repaired_execution_dir,
        repaired_runtime_states=args.repaired_runtime_states,
        out_dir=args.out_dir,
    )
    print(
        {
            "protocol": PROTOCOL,
            "status": manifest["status"],
            "unique_contrasts": manifest["unique_contrasts"],
            "review_presentations": manifest["review_presentations"],
            "human_review": str(
                args.out_dir / "human_blind_review.html"
            ),
        }
    )


if __name__ == "__main__":
    main()
