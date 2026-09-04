#!/usr/bin/env python3
"""Freeze an outcome-blind ESConv panel for the first learned PM_RS check.

The script performs two separate jobs:

1. audit candidate discovery and PM actions over every frozen ESConv test state;
2. select a small family-balanced paired-response panel without using outcomes.

The panel is a mechanism/qualification sample, not a population-prevalence
estimate. Population action coverage remains available in the full audit.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import RuntimeState
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    sha256_file,
    sha256_text,
    stable_hex,
    write_json,
    write_jsonl,
)
from metacom_pm.prompts import common_context
from metacom_pm.text import estimate_tokens, normalize_space
from metacom_pm.v1_5_strategy_rag_repair import (
    EFFECT_STUDY_PROTOCOL,
    effect_study_observable_flags,
    effect_study_rank_applicable_cards,
)
from metacom_pm.v1_5_strategy_rag_v4 import EXPECTED_FAMILIES


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-rs-esconv-frozen-validation-plan-v1"
PROMPT_PROTOCOL = "pm-v1.5-rs-final-bank-transparent-top1-prompt-v1"
TARGET_PER_FAMILY = 8
EXPECTED_TEST_STATES = 2112
EXPECTED_TEST_DIALOGUES = 169


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _visible_dialogue(state: RuntimeState) -> list[dict[str, str]]:
    rows = [
        {
            "speaker": "seeker" if turn.role == "user" else "supporter",
            "content": normalize_space(turn.content),
        }
        for turn in state.current_session_history
        if normalize_space(turn.content)
    ]
    rows.append(
        {
            "speaker": "seeker",
            "content": normalize_space(state.current_user_text),
        }
    )
    return rows


def _recent_seeker_text(dialogue: list[dict[str, str]]) -> str:
    return " ".join(
        row["content"]
        for row in dialogue
        if row["speaker"] == "seeker"
    )[-4000:]


def _query(dialogue: list[dict[str, str]], flags: dict[str, Any]) -> str:
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
        row["content"]
        for row in dialogue
        if row["speaker"] == "seeker"
    ]
    return (
        "Choose one safe topic-agnostic emotional-support technique. "
        f"Observable cues: {', '.join(cues) or 'none'}. "
        f"Recent seeker context: {' '.join(seeker[-3:])}"
    )


def _messages(
    *,
    state: RuntimeState,
    system_prompt: str,
    card: dict[str, Any] | None,
) -> list[dict[str, str]]:
    sections = [common_context(state)]
    if card is not None:
        sections.append(
            "Potential emotional-support technique. Treat it as the primary "
            "move, not the whole reply. Use it only when it fits; otherwise "
            "ignore it. Ground every factual, emotional, and temporal claim "
            "in the visible dialogue. Ask at most one question OR give at "
            "most one suggestion, never both or a list. You may add one short "
            "natural continuation when needed. Compose new wording and never "
            "mention this guidance:\n"
            f"- {card['prompt_guidance']}"
        )
    sections.append("Write only the counselor's next response.")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


def _policy_for_family(
    family: str,
    final_fit: dict[str, Any],
) -> tuple[float, str]:
    coefficient = dict(
        final_fit["coefficient_for_RS_on_by_family"]
    )
    if family not in coefficient:
        raise RuntimeError(f"PM_RS has no frozen coefficient for {family}")
    logit = float(final_fit["intercept_for_RS_on"]) + float(
        coefficient[family]
    )
    probability = 1.0 / (1.0 + math.exp(-logit))
    action = (
        "M0+RS"
        if probability >= float(final_fit["threshold"])
        else "M0+R0"
    )
    return probability, action


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--states",
        type=Path,
        default=ROOT / "data/esconv_test_v1_5/runtime_states.jsonl",
    )
    parser.add_argument(
        "--test-build-report",
        type=Path,
        default=ROOT / "data/esconv_test_v1_5/build_report.json",
    )
    parser.add_argument(
        "--bank",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v4_final_v1/"
        "strategy_cards_v4_final.jsonl",
    )
    parser.add_argument(
        "--bank-provenance",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v4_final_v1/"
        "source_provenance_private.jsonl",
    )
    parser.add_argument(
        "--pm-report",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_candidate_aware_pm_rs_development_v1/"
        "candidate_aware_pm_rs_report.json",
    )
    parser.add_argument(
        "--training-labels",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_effect_final_labels_v1/"
        "pm_rs_effect_labels.jsonl",
    )
    parser.add_argument(
        "--construct",
        type=Path,
        default=ROOT / "data/pm_v1_5_contracts/rs_open_close_construct_v2.json",
    )
    parser.add_argument(
        "--pm-config", type=Path, default=ROOT / "configs/pm_v1_5.yaml"
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=ROOT / "configs/experiment.yaml",
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--target-per-family", type=int, default=TARGET_PER_FAMILY
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_esconv_frozen_validation_v1",
    )
    args = parser.parse_args()

    cards = _rows(args.bank)
    card_by_id = {str(row["card_id"]): row for row in cards}
    states = [
        RuntimeState.model_validate(row) for row in iter_jsonl(args.states)
    ]
    build_report = json.loads(
        args.test_build_report.read_text(encoding="utf-8")
    )
    pm_report = json.loads(args.pm_report.read_text(encoding="utf-8"))
    construct = json.loads(args.construct.read_text(encoding="utf-8"))
    training_labels = _rows(args.training_labels)
    bank_sources = {
        str(row["source_dialogue_id"])
        for row in iter_jsonl(args.bank_provenance)
    }

    if (
        len(cards) != 80
        or len(card_by_id) != 80
        or any(row["raw_source_response_exposed_to_generator"] for row in cards)
    ):
        raise RuntimeError("the final 80-card source-free Bank is required")
    if (
        pm_report["status"]
        != "MINIMUM_DEVELOPMENT_SIGNAL_LEARNED_FORMAL_GATE_NOT_PASSED"
        or not pm_report["gates"]["minimum_development_signal_passed"]
        or pm_report["gates"]["external_or_fresh_holdout_gate_passed"]
    ):
        raise RuntimeError("PM_RS development report is not eligible to freeze")
    if (
        construct["protocol"] != "pm-v1.5-rs-open-close-construct-v2"
        or construct["decision_timing"]["PM_RS"]
        != (
            "runs after candidate discovery but before card injection and "
            "response generation"
        )
    ):
        raise RuntimeError("the corrected RS decision timing is required")
    if (
        len(states) != EXPECTED_TEST_STATES
        or len({state.user_id for state in states})
        != EXPECTED_TEST_DIALOGUES
        or any(state.split != "esconv_test" for state in states)
        or build_report.get("test_turns") != EXPECTED_TEST_STATES
        or build_report.get("test_dialogues") != EXPECTED_TEST_DIALOGUES
    ):
        raise RuntimeError("ESConv test state universe drifted")
    if sha256_file(args.training_labels) != pm_report["lineage"][
        "labels_sha256"
    ]:
        raise RuntimeError("PM_RS training labels drifted after model fit")

    training_users = {str(row["user_id"]) for row in training_labels}
    test_users = {state.user_id for state in states}
    if training_users & test_users:
        raise RuntimeError("PM_RS training dialogues overlap ESConv test")
    if bank_sources & test_users:
        raise RuntimeError("Strategy Bank source dialogues overlap ESConv test")

    final_fit = dict(pm_report["final_fit_for_frozen_unseen_evaluation"])
    coverage_rows: list[dict[str, Any]] = []
    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        dialogue = _visible_dialogue(state)
        recent = _recent_seeker_text(dialogue)
        flags = effect_study_observable_flags(
            current_user_text=state.current_user_text,
            recent_user_text=recent,
            visible_dialogue=dialogue,
        )
        ranked = effect_study_rank_applicable_cards(
            query=_query(dialogue, flags),
            current_user_text=state.current_user_text,
            recent_user_text=recent,
            visible_dialogue=dialogue,
            cards=cards,
        )
        top = ranked[0] if ranked else None
        if top is None:
            probability = None
            pm_action = "M0+R0"
            pm_reason = (
                "deterministic_hard_off"
                if flags["ordinary_rag_hard_off"]
                else "no_realized_eligible_candidate"
            )
        else:
            probability, pm_action = _policy_for_family(
                str(top["strategy_family"]), final_fit
            )
            pm_reason = "frozen_candidate_family_logistic"
        row = {
            "protocol": PROTOCOL,
            "state_id": state.state_id,
            "user_id": state.user_id,
            "turn_index": state.provenance.get("turn_index"),
            "visible_dialogue_sha256": sha256_text(
                canonical_json(dialogue)
            ),
            "ordinary_rag_hard_off": flags["ordinary_rag_hard_off"],
            "ordinary_rag_hard_off_reasons": flags[
                "ordinary_rag_hard_off_reasons"
            ],
            "selected_card_id": None if top is None else top["card_id"],
            "selected_core_submove_id": (
                None if top is None else top["core_submove_id"]
            ),
            "selected_strategy_family": (
                None if top is None else top["strategy_family"]
            ),
            "selected_execution_profile": (
                None if top is None else top["execution_profile"]
            ),
            "retrieval_score_diagnostic_only": (
                None if top is None else top["score"]
            ),
            "pm_rs_probability": probability,
            "frozen_pm_action": pm_action,
            "pm_action_reason": pm_reason,
            "outcome_used_for_selection": False,
        }
        coverage_rows.append(row)
        if top is not None:
            eligible[str(top["strategy_family"])].append(
                {
                    "state": state,
                    "dialogue": dialogue,
                    "flags": flags,
                    "top": top,
                    "coverage": row,
                }
            )

    target = int(args.target_per_family)
    if target < 1:
        raise RuntimeError("--target-per-family must be positive")
    chosen: list[dict[str, Any]] = []
    used_users: set[str] = set()
    for family in EXPECTED_FAMILIES:
        candidates = sorted(
            eligible[family],
            key=lambda row: stable_hex(
                PROTOCOL,
                args.seed,
                family,
                row["state"].state_id,
                row["top"]["card_id"],
                n=40,
            ),
        )
        for row in candidates:
            if row["state"].user_id in used_users:
                continue
            chosen.append(row)
            used_users.add(row["state"].user_id)
            if sum(
                item["top"]["strategy_family"] == family
                for item in chosen
            ) == target:
                break
        observed = sum(
            item["top"]["strategy_family"] == family for item in chosen
        )
        if observed != target:
            raise RuntimeError(
                f"family {family}: selected {observed}/{target}"
            )

    pm_config = load_config(args.pm_config)
    experiment = load_config(args.experiment_config)
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(
        experiment, generation.generator_endpoint
    )
    generator_identity = {
        "base_url": endpoint.base_url,
        "model": endpoint.model,
        "family": endpoint.family,
        "transport": endpoint.transport,
    }

    selected_rows: list[dict[str, Any]] = []
    call_plan: list[dict[str, Any]] = []
    for row in sorted(
        chosen,
        key=lambda item: (
            EXPECTED_FAMILIES.index(
                str(item["top"]["strategy_family"])
            ),
            item["state"].user_id,
        ),
    ):
        state: RuntimeState = row["state"]
        top = row["top"]
        card = card_by_id[str(top["card_id"])]
        pair_id = "rs_external_pair_" + stable_hex(
            PROTOCOL,
            state.state_id,
            card["card_id"],
            generation.digest(),
            args.seed,
            n=24,
        )
        selected_rows.append(
            {
                "protocol": PROTOCOL,
                "pair_id": pair_id,
                "state_id": state.state_id,
                "user_id": state.user_id,
                "source_dialogue_id": state.user_id,
                "source_turn_index": state.provenance.get("turn_index"),
                "current_session_summary": state.current_session_summary,
                "current_user_text": state.current_user_text,
                "visible_dialogue": row["dialogue"],
                "observable_flags": row["flags"],
                "selected_card_id": card["card_id"],
                "selected_core_submove_id": card["core_submove_id"],
                "selected_strategy_family": card["strategy_family"],
                "selected_execution_profile": card["execution_profile"],
                "retrieval_score_diagnostic_only": top["score"],
                "pm_rs_probability": row["coverage"]["pm_rs_probability"],
                "frozen_pm_action": row["coverage"]["frozen_pm_action"],
                "outcome_label": "UNKNOWN_FROZEN_BEFORE_GENERATION",
            }
        )
        for arm, arm_card in (("R0", None), ("RS", card)):
            messages = _messages(
                state=state,
                system_prompt=generation.system_prompt,
                card=arm_card,
            )
            call_plan.append(
                {
                    "protocol": PROTOCOL,
                    "prompt_protocol": PROMPT_PROTOCOL,
                    "pair_id": pair_id,
                    "state_id": state.state_id,
                    "user_id": state.user_id,
                    "arm": arm,
                    "action_id": "M0+R0" if arm == "R0" else "M0+RS",
                    "selected_strategy_card_id": (
                        None if arm_card is None else arm_card["card_id"]
                    ),
                    "selected_strategy_family": (
                        None
                        if arm_card is None
                        else arm_card["strategy_family"]
                    ),
                    "messages": messages,
                    "prompt_sha256": sha256_text(
                        canonical_json(messages)
                    ),
                    "estimated_input_tokens": estimate_tokens(
                        canonical_json(messages)
                    ),
                    "generation": {**generation.payload(), "seed": args.seed},
                    "generator_identity": generator_identity,
                    "api_call_made": False,
                }
            )

    parity_ok = True
    for selected in selected_rows:
        calls = [
            row
            for row in call_plan
            if row["pair_id"] == selected["pair_id"]
        ]
        r0 = next(row for row in calls if row["arm"] == "R0")
        rs = next(row for row in calls if row["arm"] == "RS")
        r0_base = r0["messages"][1]["content"].split(
            "\n\nWrite only the counselor's next response."
        )[0]
        rs_base = rs["messages"][1]["content"].split(
            "\n\nPotential emotional-support technique.", 1
        )[0]
        parity_ok &= (
            len(calls) == 2
            and r0["generation"] == rs["generation"]
            and r0["generator_identity"] == rs["generator_identity"]
            and r0["messages"][0] == rs["messages"][0]
            and r0_base == rs_base
        )

    family_counts = Counter(
        row["selected_strategy_family"] for row in selected_rows
    )
    full_candidate_counts = Counter(
        str(row["selected_strategy_family"])
        if row["selected_strategy_family"] is not None
        else "NO_CANDIDATE"
        for row in coverage_rows
    )
    full_action_counts = Counter(
        str(row["frozen_pm_action"]) for row in coverage_rows
    )
    panel_action_counts = Counter(
        str(row["frozen_pm_action"]) for row in selected_rows
    )
    expected_panel = target * len(EXPECTED_FAMILIES)
    ready = (
        len(selected_rows) == expected_panel
        and len(used_users) == expected_panel
        and all(family_counts[family] == target for family in EXPECTED_FAMILIES)
        and len(call_plan) == 2 * expected_panel
        and parity_ok
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "full_coverage_audit.jsonl", coverage_rows)
    write_jsonl(args.out_dir / "selected_states.jsonl", selected_rows)
    write_jsonl(args.out_dir / "call_plan.jsonl", call_plan)
    report = {
        "protocol": PROTOCOL,
        "status": (
            "FROZEN_READY_FOR_PAIRED_GENERATION"
            if ready
            else "FROZEN_VALIDATION_PLAN_NOT_READY"
        ),
        "research_role": (
            "First frozen unseen-group check of the development-selected "
            "coarse-family PM_RS; not a pristine benchmark claim because "
            "historical V1/V1.5 work previously used the ESConv benchmark."
        ),
        "full_test_universe": {
            "states": len(states),
            "independent_dialogues": len(test_users),
            "candidate_counts": dict(sorted(full_candidate_counts.items())),
            "frozen_pm_action_counts": dict(sorted(full_action_counts.items())),
            "outcomes_used": False,
        },
        "paired_panel": {
            "states": len(selected_rows),
            "independent_dialogues": len(used_users),
            "target_per_family": target,
            "family_counts": dict(sorted(family_counts.items())),
            "frozen_pm_action_counts": dict(
                sorted(panel_action_counts.items())
            ),
            "planned_generation_calls": len(call_plan),
            "selection": (
                "family-balanced, one state per dialogue, deterministic hash; "
                "no response, judge, human preference, or external outcome"
            ),
            "estimand": (
                "family-balanced mechanism qualification; do not report as "
                "the natural ESConv population prevalence without weighting"
            ),
        },
        "frozen_policy": {
            "threshold": final_fit["threshold"],
            "intercept_for_RS_on": final_fit["intercept_for_RS_on"],
            "coefficient_for_RS_on_by_family": final_fit[
                "coefficient_for_RS_on_by_family"
            ],
            "family_decisions": {
                family: {
                    "probability": _policy_for_family(family, final_fit)[0],
                    "action": _policy_for_family(family, final_fit)[1],
                }
                for family in EXPECTED_FAMILIES
            },
        },
        "generator_identity": generator_identity,
        "supporter_generation_treatment_sha256": generation.digest(),
        "checks": {
            "final_bank_80_cards": len(cards) == 80,
            "raw_source_response_absent": all(
                not row["raw_source_response_exposed_to_generator"]
                for row in cards
            ),
            "pm_training_dialogue_disjoint_from_esconv_test": not (
                training_users & test_users
            ),
            "bank_source_dialogue_disjoint_from_esconv_test": not (
                bank_sources & test_users
            ),
            "one_state_per_panel_dialogue": len(selected_rows)
            == len(used_users),
            "same_stack_except_strategy_section": parity_ok,
            "outcome_blind_selection": all(
                not row["outcome_used_for_selection"]
                for row in coverage_rows
            )
            and all(
                row["outcome_label"]
                == "UNKNOWN_FROZEN_BEFORE_GENERATION"
                for row in selected_rows
            ),
            "formal_train_gate_not_rewritten": not final_fit[
                "formal_train_only_gate_passed"
            ],
        },
        "next_fixed_steps": [
            "generate both R0 and RS arms once for all frozen panel states",
            "perform one quality-only blind human comparison per pair",
            "risk-review only human-adjudicated RS material wins",
            "compare frozen PM with always-off, always-on, and deterministic "
            "scope/candidate baselines without retuning on this panel",
        ],
        "inputs": {
            "states": _relative(args.states),
            "states_sha256": sha256_file(args.states),
            "test_build_report": _relative(args.test_build_report),
            "test_build_report_sha256": sha256_file(args.test_build_report),
            "bank": _relative(args.bank),
            "bank_sha256": sha256_file(args.bank),
            "bank_provenance": _relative(args.bank_provenance),
            "bank_provenance_sha256": sha256_file(args.bank_provenance),
            "pm_report": _relative(args.pm_report),
            "pm_report_sha256": sha256_file(args.pm_report),
            "training_labels": _relative(args.training_labels),
            "training_labels_sha256": sha256_file(args.training_labels),
            "construct": _relative(args.construct),
            "construct_sha256": sha256_file(args.construct),
        },
    }
    write_json(args.out_dir / "plan_report.json", report)
    manifest = {
        "protocol": PROTOCOL,
        "status": report["status"],
        "outputs": {
            name: sha256_file(args.out_dir / name)
            for name in (
                "full_coverage_audit.jsonl",
                "selected_states.jsonl",
                "call_plan.jsonl",
                "plan_report.json",
            )
        },
    }
    write_json(args.out_dir / "freeze_manifest.json", manifest)
    print(
        {
            "protocol": PROTOCOL,
            "status": report["status"],
            "test_states": len(states),
            "panel_states": len(selected_rows),
            "calls": len(call_plan),
            "family_counts": dict(sorted(family_counts.items())),
            "panel_actions": dict(sorted(panel_action_counts.items())),
        }
    )


if __name__ == "__main__":
    main()
