#!/usr/bin/env python3
"""Freeze the 32-state D2 label-reproducibility blueprint.

The selection is deliberately completed before any new D2 response is
generated.  Existing RS/MP/ME states are sampled by their already-observed
three-way direction solely to measure repeatability.  MS uses fresh states
that were absent from all 256 first-fit contrasts and whose selected MS items
come only from supplied strictly-prior session summaries.
"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
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
from metacom_pm.pm_v2_data import load_bundles
from metacom_pm.text import normalize_space
from metacom_pm.v1_5_memory_transport import (
    compile_bounded_memory,
    synthetic_basic_info,
    synthetic_prior_session,
)
from metacom_pm.v1_5_strategy_rag_repair import (
    effect_study_observable_flags,
    effect_study_rank_applicable_cards,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-d2-measurement-blueprint-v1"
RECOVERY_PROTOCOL = "pm-v1.5-first-four-component-fit-root-cause-recovery-v1"
SUBDOMAIN_PROTOCOL = "pm-v1.5-internal-superdomain-external-subdomain-contract-v1"
MS_AUDIT_PROTOCOL = "pm-v1.5-ms-construct-and-external-subdomain-audit-v1"
EXISTING_COMPONENTS = ("RS", "MP", "ME")
NEW_PAIR_SEEDS = (20260741, 20260742)
FRESH_MS_PAIR_SEEDS = (20260741, 20260742, 20260743)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _powerset(
    values: Iterable[MemorySource],
) -> list[frozenset[MemorySource]]:
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
    dialogue.append(
        {
            "speaker": "seeker",
            "content": normalize_space(state["current_user_text"]),
        }
    )
    return dialogue


def _strategy_candidate(
    state: dict[str, Any],
    cards: list[dict[str, Any]],
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
    query = (
        "Choose one safe topic-agnostic emotional-support technique. "
        f"Observable cues: {', '.join(cues) or 'none'}. "
        f"Recent seeker context: {' '.join(seeker[-3:])}"
    )
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


def _ms_origin(
    bundles_path: Path,
) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for bundle in load_bundles(bundles_path):
        ordered = sorted(bundle.cases, key=lambda case: case.session_index)
        session_rows = [synthetic_prior_session(case) for case in ordered]
        items, _ = compile_bounded_memory(
            {
                "id": bundle.user_id,
                "basic_info": synthetic_basic_info(bundle),
                "dialog_history": session_rows,
            }
        )
        for item in items:
            if item.source is not MemorySource.MS:
                continue
            case = ordered[int(item.created_session) - 1]
            result[(bundle.user_id, item.memory_id)] = (
                "supplied_session_summary"
                if case.session_summary.strip()
                else "fallback_current_user_text"
            )
    return result


def _choose_existing(
    labels: list[dict[str, Any]],
    blueprint_by_slot: dict[str, dict[str, Any]],
    states_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for component in EXISTING_COMPONENTS:
        component_rows = [
            row
            for row in labels
            if row["component"] == component
            and row["split"] in {"train", "calibration"}
            and not normalize_space(
                states_by_id[str(row["state_id"])].get(
                    "current_session_summary", ""
                )
            )
        ]
        used_users: set[str] = set()
        used_backgrounds: set[str] = set()
        for target_y in (1, 0):
            candidates = sorted(
                (
                    row
                    for row in component_rows
                    if int(row["target_y"]) == target_y
                ),
                key=lambda row: (
                    str(row["user_id"]) in used_users,
                    str(row["control_action"]) in used_backgrounds,
                    stable_hex(
                        PROTOCOL,
                        component,
                        str(target_y),
                        str(row["contrast_slot_id"]),
                        n=24,
                    ),
                ),
            )
            chosen: list[dict[str, Any]] = []
            for row in candidates:
                if str(row["user_id"]) in {
                    str(value["user_id"]) for value in chosen
                }:
                    continue
                chosen.append(row)
                if len(chosen) == 4:
                    break
            if len(chosen) != 4:
                raise RuntimeError(
                    f"cannot select four unique {component}/{target_y} rows"
                )
            for row in chosen:
                original = blueprint_by_slot[str(row["contrast_slot_id"])]
                used_users.add(str(row["user_id"]))
                used_backgrounds.add(str(row["control_action"]))
                selected.append(
                    {
                        "protocol": PROTOCOL,
                        "d2_state_id": "d2_state_"
                        + stable_hex(
                            PROTOCOL,
                            str(row["contrast_slot_id"]),
                            n=24,
                        ),
                        "component": component,
                        "state_origin": "existing_first_fit_pair",
                        "state_id": row["state_id"],
                        "user_id": row["user_id"],
                        "split": row["split"],
                        "control_action": row["control_action"],
                        "treatment_action": row["treatment_action"],
                        "selected_memory_ids_by_action_generation_only": (
                            original[
                                "selected_memory_ids_by_action_generation_only"
                            ]
                        ),
                        "current_strategy_candidate": original.get(
                            "current_strategy_candidate"
                        ),
                        "historical_pair_reference": {
                            "contrast_slot_id": row["contrast_slot_id"],
                            "direction": (
                                "component_on"
                                if int(row["target_y"]) == 1
                                else "nonpositive"
                            ),
                            "realized_pair_used_only_for_reproducibility": True,
                        },
                        "new_pair_seeds": list(NEW_PAIR_SEEDS),
                        "new_pair_count": len(NEW_PAIR_SEEDS),
                        "effect_label": (
                            "UNKNOWN_FOR_EACH_NEW_PAIR_BEFORE_BLIND_REVIEW"
                        ),
                    }
                )
    return selected


def _choose_fresh_ms(
    *,
    states: list[dict[str, Any]],
    retrieval: dict[tuple[str, str], dict[str, Any]],
    excluded_state_ids: set[str],
    origin: dict[tuple[str, str], str],
    cards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    eligible = [
        state
        for state in states
        if str(state["state_id"]) not in excluded_state_ids
        and not normalize_space(state.get("current_session_summary", ""))
        and retrieval[(str(state["state_id"]), "MS")]["selected_count"] > 0
        and all(
            origin.get((str(state["user_id"]), str(memory_id)))
            == "supplied_session_summary"
            for memory_id in retrieval[
                (str(state["state_id"]), "MS")
            ]["selected_memory_ids"]
        )
    ]
    result: list[dict[str, Any]] = []
    used_states: set[str] = set()
    used_users: set[str] = set()
    for control_action, treatment_action in _backgrounds("MS"):
        sources, strategy = parse_action_id(control_action)
        candidates: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
        for state in eligible:
            state_id = str(state["state_id"])
            user_id = str(state["user_id"])
            if state_id in used_states or user_id in used_users:
                continue
            if any(
                retrieval[(state_id, source.value)]["selected_count"] <= 0
                for source in sources
            ):
                continue
            strategy_candidate = (
                _strategy_candidate(state, cards)
                if strategy is StrategyMode.RS
                else None
            )
            if (
                strategy is StrategyMode.RS
                and strategy_candidate is None
            ):
                continue
            candidates.append((state, strategy_candidate))
        candidates.sort(
            key=lambda pair: stable_hex(
                PROTOCOL,
                control_action,
                str(pair[0]["state_id"]),
                n=24,
            )
        )
        if not candidates:
            raise RuntimeError(
                f"no fresh qualified MS state for {control_action}"
            )
        state, strategy_candidate = candidates[0]
        state_id = str(state["state_id"])
        user_id = str(state["user_id"])
        used_states.add(state_id)
        used_users.add(user_id)
        ids_by_action: dict[str, list[str]] = {}
        for action in (control_action, treatment_action):
            requested, _ = parse_action_id(action)
            ids_by_action[action] = [
                str(memory_id)
                for source in sorted(
                    requested, key=lambda value: value.value
                )
                for memory_id in retrieval[
                    (state_id, source.value)
                ]["selected_memory_ids"]
            ]
        result.append(
            {
                "protocol": PROTOCOL,
                "d2_state_id": "d2_state_"
                + stable_hex(PROTOCOL, state_id, "fresh_ms", n=24),
                "component": "MS",
                "state_origin": "fresh_qualified_ms_no_prior_effect_outcome",
                "state_id": state_id,
                "user_id": user_id,
                "split": state["split"],
                "control_action": control_action,
                "treatment_action": treatment_action,
                "selected_memory_ids_by_action_generation_only": (
                    ids_by_action
                ),
                "current_strategy_candidate": strategy_candidate,
                "historical_pair_reference": None,
                "new_pair_seeds": list(FRESH_MS_PAIR_SEEDS),
                "new_pair_count": len(FRESH_MS_PAIR_SEEDS),
                "ms_qualification": {
                    "subtype": "MS_SESSION",
                    "selected_ms_items": list(
                        retrieval[(state_id, "MS")][
                            "selected_memory_ids"
                        ]
                    ),
                    "all_selected_items_use_supplied_summary": True,
                    "last_message_fallback_used": False,
                    "current_session_summary": "empty",
                    "state_absent_from_all_first_fit_contrasts": True,
                },
                "effect_label": (
                    "UNKNOWN_FOR_EACH_NEW_PAIR_BEFORE_BLIND_REVIEW"
                ),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_final_effect_labels_v1/"
        "component_effect_labels.jsonl",
    )
    parser.add_argument(
        "--memory-blueprint",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_memory_contrast_blueprint_v1/"
        "memory_contrast_blueprint.jsonl",
    )
    parser.add_argument(
        "--rs-blueprint",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_rs_contrast_blueprint_v1/"
        "rs_contrast_blueprint.jsonl",
    )
    parser.add_argument(
        "--backend-dir",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_evo_style_synthetic_memory_v1_candidate",
    )
    parser.add_argument(
        "--bundles",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate/"
        "pm_v2_bundles.jsonl",
    )
    parser.add_argument(
        "--strategy-cards",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v4_final_v1/"
        "strategy_cards_v4_final.jsonl",
    )
    parser.add_argument(
        "--recovery-contract",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/"
        "first_fit_root_cause_recovery_v1.json",
    )
    parser.add_argument(
        "--subdomain-contract",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/"
        "internal_superdomain_external_subdomain_v1.json",
    )
    parser.add_argument(
        "--ms-audit",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_ms_external_subdomain_audit_v1/"
        "ms_external_subdomain_audit.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_d2_measurement_blueprint_v1",
    )
    args = parser.parse_args()

    recovery = read_json(args.recovery_contract)
    subdomain = read_json(args.subdomain_contract)
    ms_audit = read_json(args.ms_audit)
    if recovery.get("protocol") != RECOVERY_PROTOCOL:
        raise RuntimeError("wrong recovery contract")
    if subdomain.get("protocol") != SUBDOMAIN_PROTOCOL:
        raise RuntimeError("wrong subdomain contract")
    if ms_audit.get("protocol") != MS_AUDIT_PROTOCOL:
        raise RuntimeError("wrong MS audit")

    labels = _rows(args.labels)
    existing_state_ids = {str(row["state_id"]) for row in labels}
    blueprint_by_slot = {
        str(row["contrast_slot_id"]): row
        for row in (
            _rows(args.memory_blueprint) + _rows(args.rs_blueprint)
        )
    }
    states = _rows(args.backend_dir / "pm_v2_states.jsonl")
    states_by_id = {
        str(state["state_id"]): state for state in states
    }
    retrieval = {
        (str(row["state_id"]), str(row["source"])): row
        for row in _rows(args.backend_dir / "retrieval_audit.jsonl")
    }
    cards = _rows(args.strategy_cards)
    origin = _ms_origin(args.bundles)

    existing = _choose_existing(
        labels,
        blueprint_by_slot,
        states_by_id,
    )
    fresh_ms = _choose_fresh_ms(
        states=states,
        retrieval=retrieval,
        excluded_state_ids=existing_state_ids,
        origin=origin,
        cards=cards,
    )
    rows = existing + fresh_ms
    pair_count = sum(int(row["new_pair_count"]) for row in rows)
    response_calls = 2 * pair_count
    component_counts = Counter(str(row["component"]) for row in rows)
    direction_counts = Counter(
        (
            str(row["component"]),
            str(
                (row.get("historical_pair_reference") or {}).get(
                    "direction", "fresh_no_historical_direction"
                )
            ),
        )
        for row in rows
    )
    checks = {
        "32_states": len(rows) == 32,
        "8_states_per_component": component_counts
        == Counter({"RS": 8, "MP": 8, "ME": 8, "MS": 8}),
        "24_existing_RS_MP_ME": len(existing) == 24,
        "existing_four_on_four_nonpositive_per_component": all(
            direction_counts[(component, "component_on")] == 4
            and direction_counts[(component, "nonpositive")] == 4
            for component in EXISTING_COMPONENTS
        ),
        "existing_current_session_summary_empty": all(
            not normalize_space(
                states_by_id[str(row["state_id"])].get(
                    "current_session_summary", ""
                )
            )
            for row in existing
        ),
        "8_fresh_MS": len(fresh_ms) == 8,
        "fresh_MS_all_first_fit_state_disjoint": all(
            str(row["state_id"]) not in existing_state_ids
            for row in fresh_ms
        ),
        "fresh_MS_all_supplied_summary_no_fallback": all(
            row["ms_qualification"][
                "all_selected_items_use_supplied_summary"
            ]
            and not row["ms_qualification"][
                "last_message_fallback_used"
            ]
            for row in fresh_ms
        ),
        "fresh_MS_all_eight_backgrounds": {
            str(row["control_action"]) for row in fresh_ms
        }
        == {control for control, _ in _backgrounds("MS")},
        "72_new_pairs": pair_count == 72,
        "144_new_response_calls": response_calls == 144,
        "all_new_effect_labels_unknown": all(
            row["effect_label"].startswith("UNKNOWN_") for row in rows
        ),
    }
    status = (
        "FROZEN_READY_FOR_144_D2_RESPONSE_CALLS"
        if all(checks.values())
        else "BLOCKED_BY_D2_BLUEPRINT_CHECK"
    )
    report = {
        "protocol": PROTOCOL,
        "status": status,
        "api_calls_made": 0,
        "new_human_decisions_made": 0,
        "external_outcomes_read": False,
        "new_quality_or_risk_outcomes_read": False,
        "selection_role": (
            "measurement reliability audit, not confirmation-set selection "
            "and not direct final model promotion"
        ),
        "states": len(rows),
        "component_counts": dict(sorted(component_counts.items())),
        "new_pairs": pair_count,
        "new_response_calls": response_calls,
        "primary_reviewer_pair_decisions": pair_count,
        "second_reviewer_fixed_overlap_decisions": 32,
        "total_human_decisions": pair_count + 32,
        "distinct_users": len({str(row["user_id"]) for row in rows}),
        "existing_direction_counts": {
            f"{component}|{direction}": count
            for (component, direction), count in sorted(
                direction_counts.items()
            )
        },
        "checks": checks,
        "inputs": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                args.labels,
                args.memory_blueprint,
                args.rs_blueprint,
                args.backend_dir / "pm_v2_states.jsonl",
                args.backend_dir / "retrieval_audit.jsonl",
                args.bundles,
                args.strategy_cards,
                args.recovery_contract,
                args.subdomain_contract,
                args.ms_audit,
            )
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    blueprint_path = args.out_dir / "d2_measurement_blueprint.jsonl"
    report_path = args.out_dir / "blueprint_report.json"
    write_jsonl(blueprint_path, rows)
    write_json(report_path, report)
    write_json(
        args.out_dir / "freeze_manifest.json",
        {
            "protocol": PROTOCOL,
            "status": status,
            "blueprint_sha256": sha256_file(blueprint_path),
            "report_sha256": sha256_file(report_path),
        },
    )
    print(
        {
            "protocol": PROTOCOL,
            "status": status,
            "states": len(rows),
            "new_pairs": pair_count,
            "new_response_calls": response_calls,
            "distinct_users": report["distinct_users"],
        }
    )


if __name__ == "__main__":
    main()
