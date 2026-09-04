#!/usr/bin/env python3
"""Build the outcome-blind 256-pair four-component contrast blueprint."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
    sha256_file,
    sha256_text,
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
PROTOCOL = "pm-v1.5-four-component-clean-contrast-blueprint-v1"
COMPONENTS = ("MP", "MS", "ME", "RS")
SPLIT_QUOTAS = {"train": 4, "calibration": 2, "internal_test": 2}
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
    if component == "RS":
        return [
            (
                canonical_action_id(sources, StrategyMode.R0),
                canonical_action_id(sources, StrategyMode.RS),
            )
            for sources in _powerset(tuple(MemorySource))
        ]
    source = MemorySource(component)
    others = tuple(value for value in MemorySource if value is not source)
    result: list[tuple[str, str]] = []
    for strategy in StrategyMode:
        for background in _powerset(others):
            result.append(
                (
                    canonical_action_id(background, strategy),
                    canonical_action_id(
                        set(background) | {source}, strategy
                    ),
                )
            )
    return result


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
    if (
        not dialogue
        or dialogue[-1]["speaker"] != "seeker"
        or normalize_space(dialogue[-1]["content"]).casefold()
        != current.casefold()
    ):
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
    if not seeker:
        return None
    flags = effect_study_observable_flags(
        current_user_text=seeker[-1],
        recent_user_text=" ".join(seeker[-3:]),
        visible_dialogue=dialogue,
    )
    ranked = effect_study_rank_applicable_cards(
        query=_strategy_query(dialogue, flags),
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
        "applicability_reasons": top["applicability_reasons"],
        "observable_flags": flags,
        "query": _strategy_query(dialogue, flags),
    }


def build_blueprint(
    *,
    states_path: Path,
    outcomes_path: Path,
    reusable_contrasts_path: Path,
    cards_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    states = [dict(row) for row in iter_jsonl(states_path)]
    states_by_id = {str(row["state_id"]): row for row in states}
    outcomes = {
        (str(row["state_id"]), str(row["requested_action_id"])): dict(row)
        for row in iter_jsonl(outcomes_path)
    }
    reusable = {
        (
            str(row["state_id"]),
            str(row["component"]),
            str(row["control_action"]),
            str(row["treatment_action"]),
        ): dict(row)
        for row in iter_jsonl(reusable_contrasts_path)
    }
    cards = [dict(row) for row in iter_jsonl(cards_path)]
    candidate_by_state = {
        str(state["state_id"]): _strategy_candidate(state, cards)
        for state in states
    }

    slot_requests: list[dict[str, Any]] = []
    for component in COMPONENTS:
        for control_action, treatment_action in _backgrounds(component):
            _, control_strategy = parse_action_id(control_action)
            _, treatment_strategy = parse_action_id(treatment_action)
            requires_current_rs = (
                control_strategy is StrategyMode.RS
                or treatment_strategy is StrategyMode.RS
            )
            for split, count in SPLIT_QUOTAS.items():
                for ordinal in range(count):
                    slot_requests.append(
                        {
                            "component": component,
                            "control_action": control_action,
                            "treatment_action": treatment_action,
                            "split": split,
                            "ordinal": ordinal,
                            "requires_current_rs": requires_current_rs,
                        }
                    )
    # Allocate scarcer current-RS-eligible states first.
    slot_requests.sort(
        key=lambda row: (
            not row["requires_current_rs"],
            row["component"],
            row["control_action"],
            row["split"],
            row["ordinal"],
        )
    )

    used_states: set[str] = set()
    user_counts: Counter[tuple[str, str]] = Counter()
    selected: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for slot in slot_requests:
        candidates = [
            state
            for state in states
            if str(state["split"]) == slot["split"]
            and str(state["state_id"]) not in used_states
            and (
                not slot["requires_current_rs"]
                or candidate_by_state[str(state["state_id"])] is not None
            )
            and user_counts[(str(state["user_id"]), slot["split"])]
            < MAX_GROUPS_PER_USER_BY_SPLIT[slot["split"]]
        ]
        candidates.sort(
            key=lambda state: stable_hex(
                PROTOCOL,
                slot["component"],
                slot["control_action"],
                slot["treatment_action"],
                slot["split"],
                str(slot["ordinal"]),
                str(state["user_id"]),
                str(state["state_id"]),
                n=32,
            )
        )
        if not candidates:
            failures.append(dict(slot))
            continue
        state = candidates[0]
        state_id = str(state["state_id"])
        user_id = str(state["user_id"])
        used_states.add(state_id)
        user_counts[(user_id, slot["split"])] += 1

        component = str(slot["component"])
        control_action = str(slot["control_action"])
        treatment_action = str(slot["treatment_action"])
        _, control_strategy = parse_action_id(control_action)
        _, treatment_strategy = parse_action_id(treatment_action)
        old_reuse = reusable.get(
            (state_id, component, control_action, treatment_action)
        )
        if old_reuse is not None:
            reuse_class = "FULL_PAIR_REUSE_R0_MEMORY_CONTRAST"
            control_status = "REUSE_EXISTING_OUTCOME"
            treatment_status = "REUSE_EXISTING_OUTCOME"
        elif component == "RS":
            reuse_class = "REUSE_R0_CONTROL_GENERATE_CURRENT_RS_TREATMENT"
            control_status = "REUSE_EXISTING_OUTCOME"
            treatment_status = "GENERATE_WITH_CURRENT_RS_TOP1"
        else:
            reuse_class = "GENERATE_BOTH_CURRENT_RS_BACKGROUND_ARMS"
            control_status = "GENERATE_WITH_CURRENT_RS_TOP1"
            treatment_status = "GENERATE_WITH_CURRENT_RS_TOP1"

        control_old = outcomes.get((state_id, control_action))
        treatment_old = outcomes.get((state_id, treatment_action))
        strategy_candidate = (
            candidate_by_state[state_id]
            if (
                control_strategy is StrategyMode.RS
                or treatment_strategy is StrategyMode.RS
            )
            else None
        )
        selected.append(
            {
                "protocol": PROTOCOL,
                "contrast_slot_id": "component_slot_"
                + stable_hex(
                    PROTOCOL,
                    component,
                    control_action,
                    treatment_action,
                    slot["split"],
                    str(slot["ordinal"]),
                    n=24,
                ),
                "component": component,
                "split": str(slot["split"]),
                "background_action": control_action,
                "control_action": control_action,
                "treatment_action": treatment_action,
                "state_id": state_id,
                "user_id": user_id,
                "card_id": str(state["card_id"]),
                "semantic_family_diagnostic_only": str(
                    state["semantic_family"]
                ),
                "session_index": int(state["session_index"]),
                "reuse_class": reuse_class,
                "control_generation_status": control_status,
                "treatment_generation_status": treatment_status,
                "historical_structural_contrast_id": (
                    None if old_reuse is None else old_reuse["contrast_id"]
                ),
                "historical_control_response_sha256": (
                    None
                    if control_old is None
                    else sha256_text(str(control_old["response"]))
                ),
                "historical_treatment_response_sha256": (
                    None
                    if treatment_old is None
                    else sha256_text(str(treatment_old["response"]))
                ),
                "current_strategy_candidate": strategy_candidate,
                "effect_label": "UNKNOWN_BEFORE_BLIND_HUMAN_REVIEW",
                "material_risk": "UNKNOWN_UNLESS_COMPONENT_ON_WINS",
            }
        )

    if failures:
        raise RuntimeError(
            f"could not allocate {len(failures)} blueprint slots: "
            + json.dumps(failures[:3], ensure_ascii=False)
        )
    if len(selected) != 256 or len(used_states) != 256:
        raise RuntimeError("blueprint must contain 256 distinct states")

    by_component = Counter(row["component"] for row in selected)
    by_component_background = Counter(
        (row["component"], row["background_action"]) for row in selected
    )
    by_split = Counter(row["split"] for row in selected)
    by_reuse = Counter(row["reuse_class"] for row in selected)
    new_arm_keys = {
        (row["state_id"], action)
        for row in selected
        for action, status in (
            (row["control_action"], row["control_generation_status"]),
            (row["treatment_action"], row["treatment_generation_status"]),
        )
        if status.startswith("GENERATE_")
    }
    report = {
        "protocol": PROTOCOL,
        "status": "READY_256_CONTRAST_BLUEPRINT_ZERO_NEW_API",
        "api_calls_made": 0,
        "human_labels_read": False,
        "effect_labels_created": False,
        "contrast_groups": len(selected),
        "distinct_states": len(used_states),
        "distinct_users": len({row["user_id"] for row in selected}),
        "component_counts": dict(sorted(by_component.items())),
        "split_counts": dict(sorted(by_split.items())),
        "reuse_counts": dict(sorted(by_reuse.items())),
        "new_generation_response_calls": len(new_arm_keys),
        "human_quality_decisions_after_generation": len(selected),
        "component_background_counts": {
            f"{component}|{background}": count
            for (component, background), count in sorted(
                by_component_background.items()
            )
        },
        "checks": {
            "64_contrasts_per_component": set(by_component.values()) == {64},
            "8_contrasts_per_component_background": set(
                by_component_background.values()
            )
            == {8},
            "split_quotas_128_64_64": by_split
            == Counter({"train": 128, "calibration": 64, "internal_test": 64}),
            "one_distinct_state_per_contrast": len(used_states) == 256,
            "all_RS_backgrounds_have_current_candidate": all(
                row["current_strategy_candidate"] is not None
                for row in selected
                if row["control_action"].endswith("+RS")
                or row["treatment_action"].endswith("+RS")
            ),
            "all_labels_unknown": all(
                row["effect_label"]
                == "UNKNOWN_BEFORE_BLIND_HUMAN_REVIEW"
                for row in selected
            ),
        },
        "generation_scope": {
            "fully_reused_pairs": by_reuse[
                "FULL_PAIR_REUSE_R0_MEMORY_CONTRAST"
            ],
            "half_reused_RS_pairs": by_reuse[
                "REUSE_R0_CONTROL_GENERATE_CURRENT_RS_TREATMENT"
            ],
            "fully_new_RS_background_pairs": by_reuse[
                "GENERATE_BOTH_CURRENT_RS_BACKGROUND_ARMS"
            ],
            "unique_new_response_calls": len(new_arm_keys),
            "API_execution_authorized_by_this_blueprint": False,
        },
        "inputs": {
            "states": str(states_path.relative_to(ROOT)),
            "historical_outcomes": str(outcomes_path.relative_to(ROOT)),
            "reusable_contrast_audit": str(
                reusable_contrasts_path.relative_to(ROOT)
            ),
            "current_strategy_bank": str(cards_path.relative_to(ROOT)),
        },
    }
    return report, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--states",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate/pm_v2_states.jsonl",
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_longitudinal_action_sweep_v8_19_2_continuation_v2_dry_run/action_outcomes.jsonl",
    )
    parser.add_argument(
        "--reuse-audit",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_component_contrast_reuse_audit_v1/reusable_R0_memory_contrasts.jsonl",
    )
    parser.add_argument(
        "--cards",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_four_component_contrast_blueprint_v1",
    )
    args = parser.parse_args()
    report, rows = build_blueprint(
        states_path=args.states,
        outcomes_path=args.outcomes,
        reusable_contrasts_path=args.reuse_audit,
        cards_path=args.cards,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "contrast_blueprint.jsonl"
    report_path = args.out_dir / "blueprint_report.json"
    write_jsonl(rows_path, rows)
    report["outputs"] = {rows_path.name: sha256_file(rows_path)}
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "contrast_groups": report["contrast_groups"],
                "reuse_counts": report["reuse_counts"],
                "new_generation_response_calls": report[
                    "new_generation_response_calls"
                ],
                "out_dir": str(args.out_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
