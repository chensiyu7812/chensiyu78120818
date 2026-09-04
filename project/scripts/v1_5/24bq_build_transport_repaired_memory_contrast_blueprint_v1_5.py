#!/usr/bin/env python3
"""Build 192 outcome-blind MP/MS/ME contrasts on the repaired backend."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Iterable

from metacom_pm.contracts import (
    MemorySource,
    StrategyMode,
    canonical_action_id,
    parse_action_id,
)
from metacom_pm.io import (
    iter_jsonl,
    read_json,
    sha256_file,
    stable_hex,
    write_json,
    write_jsonl,
)
from metacom_pm.text import normalize_space
from metacom_pm.v1_5_strategy_rag_repair import (
    effect_study_observable_flags,
    effect_study_rank_applicable_cards,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-transport-repaired-memory-contrast-blueprint-v1"
COMPONENTS = ("MP", "MS", "ME")
SPLIT_QUOTAS = {"train": 4, "calibration": 2, "internal_test": 2}
TARGET_QUANTILES = {
    "train": (0.10, 0.35, 0.65, 0.90),
    "calibration": (0.25, 0.75),
    "internal_test": (0.25, 0.75),
}
MAX_GROUPS_PER_USER_BY_SPLIT = {
    "train": 6,
    "calibration": 6,
    "internal_test": 4,
}


def _powerset(values: Iterable[MemorySource]) -> list[frozenset[MemorySource]]:
    ordered = tuple(values)
    return [
        frozenset(items)
        for size in range(len(ordered) + 1)
        for items in combinations(ordered, size)
    ]


def _backgrounds(component: str) -> list[tuple[str, str]]:
    source = MemorySource(component)
    others = tuple(value for value in MemorySource if value is not source)
    return [
        (
            canonical_action_id(background, strategy),
            canonical_action_id(set(background) | {source}, strategy),
        )
        for strategy in StrategyMode
        for background in _powerset(others)
    ]


def _visible_dialogue(state: dict[str, Any]) -> list[dict[str, str]]:
    dialogue = [
        {
            "speaker": (
                "seeker" if str(turn["role"]) == "user" else "supporter"
            ),
            "content": normalize_space(turn["content"]),
        }
        for turn in state["current_session_history"]
        if normalize_space(turn.get("content", ""))
    ]
    current = normalize_space(state["current_user_text"])
    dialogue.append({"speaker": "seeker", "content": current})
    return dialogue


def _strategy_query(
    dialogue: list[dict[str, str]], flags: dict[str, Any]
) -> str:
    cues = [
        key.replace("_", " ")
        for key, value in flags.items()
        if isinstance(value, bool)
        and value
        and key
        not in {
            "substantive",
            "pure_phatic",
            "routine_closing",
            "active_high_stakes",
            "explicit_stop",
            "ordinary_rag_hard_off",
            "structural_eligibility_only_not_benefit",
        }
    ]
    seeker = [
        row["content"] for row in dialogue if row["speaker"] == "seeker"
    ]
    return (
        "Choose one safe topic-agnostic emotional-support technique. "
        f"Observable cues: {', '.join(cues) or 'none'}. "
        f"Recent seeker context: {' '.join(seeker[-3:])}"
    )


def _strategy_candidate(
    state: dict[str, Any], cards: list[dict[str, Any]]
) -> dict[str, Any] | None:
    dialogue = _visible_dialogue(state)
    seeker = [
        row["content"] for row in dialogue if row["speaker"] == "seeker"
    ]
    flags = effect_study_observable_flags(
        current_user_text=seeker[-1],
        recent_user_text=" ".join(seeker[-3:]),
        visible_dialogue=dialogue,
    )
    query = _strategy_query(dialogue, flags)
    ranked = effect_study_rank_applicable_cards(
        query=query,
        current_user_text=seeker[-1],
        recent_user_text=" ".join(seeker[-3:]),
        visible_dialogue=dialogue,
        cards=cards,
    )
    if not ranked:
        return None
    top = ranked[0]
    return {
        "card_id": top["card_id"],
        "core_submove_id": top["core_submove_id"],
        "strategy_family": top["strategy_family"],
        "execution_profile": top["execution_profile"],
        "compatibility_tier": top["compatibility_tier"],
        "retrieval_score": float(top["score"]),
        "applicability_reasons": top["applicability_reasons"],
        "observable_flags": flags,
        "query": query,
    }


def build_blueprint(
    *,
    backend_dir: Path,
    strategy_cards_path: Path,
    old_blueprint_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    backend_report = read_json(backend_dir / "report.json")
    if backend_report.get("status") != (
        "READY_FOR_OUTCOME_BLIND_192_MEMORY_CONTRAST_BLUEPRINT"
    ):
        raise RuntimeError("transport-repaired backend is not qualified")
    states = [
        dict(row) for row in iter_jsonl(backend_dir / "pm_v2_states.jsonl")
    ]
    states_by_id = {str(row["state_id"]): row for row in states}
    retrieval = {
        (str(row["state_id"]), str(row["source"])): dict(row)
        for row in iter_jsonl(backend_dir / "retrieval_audit.jsonl")
    }
    cards = [dict(row) for row in iter_jsonl(strategy_cards_path)]
    strategy_by_state = {
        state_id: _strategy_candidate(state, cards)
        for state_id, state in states_by_id.items()
    }

    slot_requests: list[dict[str, Any]] = []
    for component in COMPONENTS:
        for control_action, treatment_action in _backgrounds(component):
            _, strategy = parse_action_id(control_action)
            for split, count in SPLIT_QUOTAS.items():
                for ordinal in range(count):
                    slot_requests.append(
                        {
                            "component": component,
                            "control_action": control_action,
                            "treatment_action": treatment_action,
                            "split": split,
                            "ordinal": ordinal,
                            "target_quantile": TARGET_QUANTILES[split][
                                ordinal
                            ],
                            "requires_rs": strategy is StrategyMode.RS,
                        }
                    )
    slot_requests.sort(
        key=lambda row: (
            not row["requires_rs"],
            row["component"],
            row["control_action"],
            row["split"],
            row["ordinal"],
        )
    )

    used_states: set[str] = set()
    user_counts: Counter[tuple[str, str]] = Counter()
    selected: list[dict[str, Any]] = []
    for slot in slot_requests:
        component = str(slot["component"])
        candidates = [
            state
            for state in states
            if str(state["split"]) == slot["split"]
            and str(state["state_id"]) not in used_states
            and user_counts[
                (str(state["user_id"]), str(slot["split"]))
            ]
            < MAX_GROUPS_PER_USER_BY_SPLIT[str(slot["split"])]
            and retrieval[
                (str(state["state_id"]), component)
            ]["selected_count"]
            > 0
            and (
                not slot["requires_rs"]
                or strategy_by_state[str(state["state_id"])] is not None
            )
        ]
        if not candidates:
            raise RuntimeError(f"no candidate for slot {slot}")
        candidates.sort(
            key=lambda state: (
                float(
                    retrieval[
                        (str(state["state_id"]), component)
                    ]["top1_lexical_score"]
                ),
                stable_hex(PROTOCOL, str(state["state_id"]), n=24),
            )
        )
        target_index = int(
            round(float(slot["target_quantile"]) * (len(candidates) - 1))
        )
        state = candidates[target_index]
        state_id = str(state["state_id"])
        user_id = str(state["user_id"])
        used_states.add(state_id)
        user_counts[(user_id, str(slot["split"]))] += 1

        action_memory_ids: dict[str, list[str]] = {}
        for action in (
            str(slot["control_action"]),
            str(slot["treatment_action"]),
        ):
            sources, _ = parse_action_id(action)
            ids = [
                memory_id
                for source in sorted(sources, key=lambda value: value.value)
                for memory_id in retrieval[
                    (state_id, source.value)
                ]["selected_memory_ids"]
            ]
            realized_sources = {
                source
                for source in sources
                if retrieval[(state_id, source.value)]["selected_count"] > 0
            }
            if realized_sources != set(sources):
                raise RuntimeError(
                    f"requested action would collapse: {state_id}/{action}"
                )
            action_memory_ids[action] = ids

        selected.append(
            {
                "protocol": PROTOCOL,
                "contrast_slot_id": "transport_component_slot_"
                + stable_hex(
                    PROTOCOL,
                    component,
                    str(slot["control_action"]),
                    str(slot["split"]),
                    str(slot["ordinal"]),
                    n=24,
                ),
                "component": component,
                "split": str(slot["split"]),
                "background_action": str(slot["control_action"]),
                "control_action": str(slot["control_action"]),
                "treatment_action": str(slot["treatment_action"]),
                "state_id": state_id,
                "card_id": str(state["card_id"]),
                "user_id": user_id,
                "semantic_family_diagnostic_only": str(
                    state["semantic_family"]
                ),
                "session_index": int(state["session_index"]),
                "selection_target_quantile": float(
                    slot["target_quantile"]
                ),
                "component_candidate_observation": {
                    key: retrieval[(state_id, component)][key]
                    for key in (
                        "catalog_count",
                        "selected_count",
                        "top1_lexical_score",
                        "top2_lexical_score",
                        "top1_top2_margin",
                    )
                },
                "selected_memory_ids_by_action_generation_only": (
                    action_memory_ids
                ),
                "current_strategy_candidate": (
                    strategy_by_state[state_id]
                    if slot["requires_rs"]
                    else None
                ),
                "control_generation_status": (
                    "GENERATE_ON_TRANSPORT_REPAIRED_BACKEND"
                ),
                "treatment_generation_status": (
                    "GENERATE_ON_TRANSPORT_REPAIRED_BACKEND"
                ),
                "effect_label": "UNKNOWN_BEFORE_BLIND_HUMAN_REVIEW",
                "material_risk": "UNKNOWN_UNLESS_COMPONENT_ON_WINS",
            }
        )

    old_rows = [
        dict(row) for row in iter_jsonl(old_blueprint_path)
    ]
    retained_rs = [row for row in old_rows if row["component"] == "RS"]
    if len(retained_rs) != 64:
        raise RuntimeError("the retained formal RS set is not 64 contrasts")

    by_component = Counter(row["component"] for row in selected)
    by_split = Counter(row["split"] for row in selected)
    by_background = Counter(
        (row["component"], row["background_action"]) for row in selected
    )
    report = {
        "protocol": PROTOCOL,
        "status": "READY_192_MEMORY_PAIRS_ZERO_API_RETAIN_64_RS",
        "api_calls_made": 0,
        "human_labels_read": False,
        "effect_labels_created": False,
        "memory_contrasts": len(selected),
        "retained_formal_rs_contrasts": len(retained_rs),
        "replacement_total_contrasts": len(selected) + len(retained_rs),
        "new_response_calls_required": 2 * len(selected),
        "distinct_memory_states": len(used_states),
        "distinct_memory_users": len(
            {row["user_id"] for row in selected}
        ),
        "component_counts": dict(sorted(by_component.items())),
        "split_counts": dict(sorted(by_split.items())),
        "background_counts": {
            f"{component}|{background}": count
            for (component, background), count in sorted(
                by_background.items()
            )
        },
        "selection": {
            "outcome_blind": True,
            "observable_axis": "component_top1_lexical_score",
            "quantiles_by_split": {
                key: list(value) for key, value in TARGET_QUANTILES.items()
            },
            "purpose": (
                "cover low-to-high candidate relevance without selecting on "
                "quality, risk, generated response, or old needed-source labels"
            ),
        },
        "checks": {
            "64_per_memory_component": set(by_component.values()) == {64},
            "8_per_component_background": set(
                by_background.values()
            )
            == {8},
            "split_quotas_96_48_48": by_split
            == Counter(
                {"train": 96, "calibration": 48, "internal_test": 48}
            ),
            "one_state_per_memory_contrast": len(used_states) == 192,
            "all_labels_unknown": all(
                row["effect_label"]
                == "UNKNOWN_BEFORE_BLIND_HUMAN_REVIEW"
                for row in selected
            ),
            "all_rs_backgrounds_have_current_candidate": all(
                row["current_strategy_candidate"] is not None
                for row in selected
                if row["background_action"].endswith("+RS")
            ),
            "retains_exactly_64_existing_rs": len(retained_rs) == 64,
        },
        "claim_boundary": {
            "formal": (
                "64 retained RS contrasts plus 192 newly generated memory "
                "contrasts under the repaired construction/retrieval stack"
            ),
            "legacy_memory_pairs": (
                "diagnostic only; not concatenated with these formal pairs"
            ),
        },
        "inputs": {
            "backend_report": str(
                (backend_dir / "report.json").relative_to(ROOT)
            ),
            "backend_report_sha256": sha256_file(
                backend_dir / "report.json"
            ),
            "strategy_cards": str(
                strategy_cards_path.relative_to(ROOT)
            ),
            "strategy_cards_sha256": sha256_file(strategy_cards_path),
            "retained_rs_blueprint": str(
                old_blueprint_path.relative_to(ROOT)
            ),
            "retained_rs_blueprint_sha256": sha256_file(
                old_blueprint_path
            ),
        },
    }
    if not all(report["checks"].values()):
        report["status"] = "BLOCKED_BY_BLUEPRINT_CHECK"
    return report, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend-dir",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_evo_style_synthetic_memory_v1_candidate",
    )
    parser.add_argument(
        "--strategy-cards",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v4_final_v1/"
        "strategy_cards_v4_final.jsonl",
    )
    parser.add_argument(
        "--old-blueprint",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_four_component_contrast_blueprint_v1/"
        "contrast_blueprint.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_memory_contrast_blueprint_v1",
    )
    args = parser.parse_args()
    report, rows = build_blueprint(
        backend_dir=args.backend_dir,
        strategy_cards_path=args.strategy_cards,
        old_blueprint_path=args.old_blueprint,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "memory_contrast_blueprint.jsonl"
    write_jsonl(rows_path, rows)
    report["outputs"] = {
        rows_path.name: sha256_file(rows_path)
    }
    write_json(args.out_dir / "blueprint_report.json", report)
    print(
        json.dumps(
            {
                "protocol": report["protocol"],
                "status": report["status"],
                "memory_contrasts": report["memory_contrasts"],
                "retained_rs": report["retained_formal_rs_contrasts"],
                "new_response_calls": report[
                    "new_response_calls_required"
                ],
                "out_dir": str(args.out_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
