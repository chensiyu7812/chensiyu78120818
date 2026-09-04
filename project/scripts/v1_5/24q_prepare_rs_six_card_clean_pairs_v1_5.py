#!/usr/bin/env python3
"""Prepare 32 leakage-controlled R0/RS pairs for the frozen six-card Bank.

Selection uses only pre-response ESConv-train dialogue, the corrected
fail-closed six-card runtime, and a protocol-derived deterministic order.  The
hidden next supporter response and its native strategy label are never used.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import RuntimeState, StrategyCard
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
from metacom_pm.prompts import generation_messages
from metacom_pm.text import estimate_tokens
from metacom_pm.v1_5_strategy_rag_runtime import (
    PROTOCOL as RUNTIME_PROTOCOL,
    QualifiedStrategyRAG,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-rs-six-card-clean-pair-plan-v1"
SELECTION_SEED = "pm-v1.5-rs-six-card-clean-pair-selection-v1"
TARGET_BY_MOVE = {
    "AM01_invite_open_expression": 5,
    "AM02_ask_one_focused_clarification": 5,
    "AM04_tentative_paraphrase_check": 7,
    "AM05_grounded_validation": 7,
    "AM10_offer_one_optional_micro_step": 8,
}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _g1_source_sets() -> tuple[set[str], set[str]]:
    all_exposed: set[str] = set()
    for path in sorted(
        (ROOT / "outputs").glob("pm_v1_5_strategy_g1_*/**/*lineage*.jsonl")
    ):
        for row in _rows(path):
            dialogue_id = row.get("source_dialogue_id")
            if dialogue_id is None:
                dialogue_id = (row.get("private_candidate_lineage") or {}).get(
                    "source_dialogue_id"
                )
            if dialogue_id:
                all_exposed.add(str(dialogue_id))
    accepted_path = (
        ROOT
        / "outputs/pm_v1_5_strategy_g1_stage1_aggregation_v1"
        / "combined_private_weak_sources.jsonl"
    )
    accepted = {
        str(row["source_dialogue_id"]) for row in _rows(accepted_path)
    }
    return all_exposed, accepted


def _prior_development_dialogues() -> tuple[set[str], list[str]]:
    paths = sorted(
        (ROOT / "outputs").glob(
            "pm_v1_5_strategy_rag_v4_direct_effect_v*/selected_states.jsonl"
        )
    )
    paths.extend(
        [
            ROOT
            / "outputs/pm_v1_5_rs_natural_retrieval_review_v1_candidate"
            / "private_selection_audit.jsonl",
            ROOT
            / "outputs/pm_v1_5_rs_natural_retrieval_fit_holdout_v2_candidate"
            / "private_selection_audit.jsonl",
            ROOT
            / "outputs/pm_v1_5_rs_natural_retrieval_fit_holdout_v3_final_candidate"
            / "private_selection_audit.jsonl",
        ]
    )
    seen: set[str] = set()
    used_paths: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        used_paths.append(str(path.relative_to(ROOT)))
        for row in _rows(path):
            dialogue_id = row.get("source_dialogue_id") or row.get("user_id")
            if dialogue_id:
                seen.add(str(dialogue_id))
    return seen, used_paths


def _visible_state(
    row: dict[str, Any],
    *,
    inventory_template: dict[str, Any],
) -> RuntimeState:
    dialogue = list(row.get("recent_dialogue") or [])
    if not dialogue or str(dialogue[-1].get("speaker")) != "seeker":
        raise ValueError("clean universe state must end on a seeker turn")
    current = str(dialogue[-1]["content"]).strip()
    history = [
        {
            "role": (
                "user" if str(turn.get("speaker")) == "seeker" else "assistant"
            ),
            "content": str(turn.get("content") or "").strip(),
        }
        for turn in dialogue[:-1]
    ]
    dialogue_id = str(row["source_dialogue_id"])
    turn_index = int(row["source_turn_index"])
    state_id = "state_" + stable_hex(
        PROTOCOL, dialogue_id, turn_index, current, n=24
    )
    payload = {
        "state_id": state_id,
        "card_id": "card_" + stable_hex(PROTOCOL, state_id, n=24),
        "user_id": dialogue_id,
        "split": "esconv_auxiliary_train",
        "semantic_family": "rs_six_card_component_effect",
        "current_user_text": current,
        "current_session_history": history,
        "current_session_summary": "",
        "session_index": 1,
        "inventory": deepcopy(inventory_template),
        "allowed_actions": ["M0+R0", "M0+RS"],
        "provenance": {
            "protocol": PROTOCOL,
            "source_dialogue_id": dialogue_id,
            "source_turn_index": turn_index,
            "selection_uses_hidden_next_response": False,
            "selection_uses_native_strategy_label": False,
        },
    }
    return RuntimeState.model_validate(payload)


def _current_runtime_is_conservative(
    universe_by_key: dict[tuple[str, int], dict[str, Any]],
    rag: QualifiedStrategyRAG,
    old_results_path: Path,
) -> dict[str, Any]:
    transitions: Counter[str] = Counter()
    violations: list[dict[str, Any]] = []
    for row in _rows(old_results_path):
        dialogue_id = str(row["source_dialogue_id"])
        turn_index = int(str(row["query_state_id"]).rsplit("_", 1)[-1])
        source = universe_by_key.get((dialogue_id, turn_index))
        if source is None:
            violations.append(
                {
                    "dialogue_id": dialogue_id,
                    "turn_index": turn_index,
                    "reason": "missing_source_state",
                }
            )
            continue
        decision = rag.retrieve(source["recent_dialogue"])
        current = (
            decision.selected_cards[0].strategy_label
            if decision.selected_cards
            else None
        )
        previous = row.get("lexical_top1")
        transitions[f"{previous or 'OFF'} -> {current or 'OFF'}"] += 1
        if current is not None and current != previous:
            violations.append(
                {
                    "dialogue_id": dialogue_id,
                    "turn_index": turn_index,
                    "previous": previous,
                    "current": current,
                    "reason": "current_runtime_added_or_changed_an_open",
                }
            )
        if current is None and previous not in {
            None,
            "AM14_supportive_transition",
        }:
            violations.append(
                {
                    "dialogue_id": dialogue_id,
                    "turn_index": turn_index,
                    "previous": previous,
                    "current": current,
                    "reason": "unexpected_nontransition_close",
                }
            )
    return {
        "comparison_rows": sum(transitions.values()),
        "transition_counts": dict(sorted(transitions.items())),
        "violations": violations,
        "passed": not violations,
        "interpretation": (
            "The corrected runtime only turns former AM14 routine "
            "stop/thanks opportunities off; it adds or changes no open."
        ),
    }


def _balanced_select(
    candidates: Iterable[dict[str, Any]],
    *,
    target_by_move: dict[str, int],
) -> list[dict[str, Any]]:
    by_move: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_move[str(row["selected_move_id"])].append(row)
    for move in by_move:
        by_move[move].sort(
            key=lambda row: stable_hex(
                SELECTION_SEED,
                move,
                row["source_dialogue_id"],
                row["source_turn_index"],
                n=32,
            )
        )

    chosen: list[dict[str, Any]] = []
    chosen_dialogues: set[str] = set()
    # Allocate scarce moves first so the global one-dialogue rule cannot erase
    # their already small support.
    order = sorted(
        target_by_move,
        key=lambda move: (
            len({row["source_dialogue_id"] for row in by_move[move]}),
            move,
        ),
    )
    for move in order:
        target = target_by_move[move]
        for row in by_move[move]:
            dialogue_id = str(row["source_dialogue_id"])
            if dialogue_id in chosen_dialogues:
                continue
            chosen.append(row)
            chosen_dialogues.add(dialogue_id)
            if sum(x["selected_move_id"] == move for x in chosen) >= target:
                break
        actual = sum(x["selected_move_id"] == move for x in chosen)
        if actual != target:
            raise RuntimeError(
                f"cannot allocate {target} independent dialogues for {move}; "
                f"allocated {actual}"
            )
    return sorted(
        chosen,
        key=lambda row: (
            row["selected_move_id"],
            row["source_dialogue_id"],
            row["source_turn_index"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_esconv_train_strategy_inductive_audit_v1"
        / "clean_train_strategy_universe.jsonl",
    )
    parser.add_argument(
        "--bank",
        type=Path,
        default=ROOT / "data/strategy/strategy_cards_v1_5_minimal.jsonl",
    )
    parser.add_argument(
        "--runtime-template",
        type=Path,
        default=ROOT
        / "data/esconv_auxiliary_v1_5_visible_v2_candidate/train"
        / "runtime_states.jsonl",
    )
    parser.add_argument(
        "--formal-esconv-test",
        type=Path,
        default=ROOT
        / "data/esconv_test_v1_5_visible_v2_candidate/runtime_states.jsonl",
    )
    parser.add_argument(
        "--old-g2-results",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_g2_real_qualification_v1"
        / "real_query_results.jsonl",
    )
    parser.add_argument(
        "--pm-config", type=Path, default=ROOT / "configs/pm_v1_5.yaml"
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=ROOT / "configs/experiment.yaml",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_rs_six_card_clean_pair_v1",
    )
    parser.add_argument(
        "--exclude-selected-states",
        nargs="*",
        type=Path,
        default=[],
        help=(
            "Additional selected_states.jsonl files whose dialogue groups "
            "must be excluded, e.g. the first six-card wave."
        ),
    )
    parser.add_argument(
        "--target-by-move",
        nargs="*",
        default=[],
        metavar="MOVE=COUNT",
        help=(
            "Optional per-move allocation override. Omitted moves receive "
            "zero; without this option the frozen first-wave allocation is "
            "used."
        ),
    )
    args = parser.parse_args()
    target_by_move = dict(TARGET_BY_MOVE)
    if args.target_by_move:
        target_by_move = {}
        for item in args.target_by_move:
            move, separator, raw_count = str(item).partition("=")
            if (
                not separator
                or move not in TARGET_BY_MOVE
                or not raw_count.isdigit()
                or int(raw_count) < 1
            ):
                raise ValueError(f"invalid --target-by-move item: {item!r}")
            target_by_move[move] = int(raw_count)

    universe = _rows(args.universe)
    universe_by_key = {
        (str(row["source_dialogue_id"]), int(row["source_turn_index"])): row
        for row in universe
    }
    cards = [StrategyCard.model_validate(row) for row in _rows(args.bank)]
    rag = QualifiedStrategyRAG(cards)
    all_g1_exposed, accepted_g1_sources = _g1_source_sets()
    prior_development, prior_paths = _prior_development_dialogues()
    for path in args.exclude_selected_states:
        if not path.is_file():
            raise FileNotFoundError(path)
        prior_paths.append(str(path.resolve().relative_to(ROOT)))
        for row in _rows(path):
            dialogue_id = row.get("source_dialogue_id") or row.get("user_id")
            if dialogue_id:
                prior_development.add(str(dialogue_id))
    formal_test_dialogues = {
        str(row["user_id"]) for row in _rows(args.formal_esconv_test)
    }
    blocked = accepted_g1_sources | prior_development | formal_test_dialogues

    template = next(iter_jsonl(args.runtime_template))
    inventory_template = dict(template["inventory"])
    candidates: list[dict[str, Any]] = []
    strict_candidates: list[dict[str, Any]] = []
    for source in universe:
        dialogue_id = str(source["source_dialogue_id"])
        recent_dialogue = list(source.get("recent_dialogue") or [])
        # Consecutive supporter turns occur in raw ESConv. They are valid for
        # corpus coding but are not a PM decision state because no new seeker
        # message precedes the target response. Exclude them instead of
        # rewinding or altering visible chronology.
        if (
            not recent_dialogue
            or str(recent_dialogue[-1].get("speaker")) != "seeker"
        ):
            continue
        decision = rag.retrieve(recent_dialogue)
        if not decision.selected_cards:
            continue
        row = {
            "source_dialogue_id": dialogue_id,
            "source_turn_index": int(source["source_turn_index"]),
            "problem_type_audit_only": str(source.get("problem_type") or ""),
            "emotion_type_audit_only": str(source.get("emotion_type") or ""),
            "selected_move_id": decision.selected_cards[0].strategy_label,
            "selected_strategy_id": decision.selected_cards[0].strategy_id,
            "retrieval_score": float(decision.selected_score or 0.0),
            "observable_flags": decision.observable_flags,
            "eligible_move_ids": list(decision.eligible_move_ids),
        }
        if dialogue_id not in (all_g1_exposed | prior_development | formal_test_dialogues):
            strict_candidates.append(row)
        if dialogue_id not in blocked:
            candidates.append(row)

    selected = _balanced_select(
        candidates, target_by_move=target_by_move
    )
    pm_config = load_config(args.pm_config)
    experiment = load_config(args.experiment_config)
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, generation.generator_endpoint)
    generator_identity = {
        "base_url": endpoint.base_url,
        "model": endpoint.model,
        "family": endpoint.family,
        "transport": endpoint.transport,
    }
    card_by_id = {card.strategy_id: card for card in cards}
    runtime_states: list[dict[str, Any]] = []
    selected_states: list[dict[str, Any]] = []
    call_plan: list[dict[str, Any]] = []
    for index, selected_row in enumerate(selected):
        source = universe_by_key[
            (
                str(selected_row["source_dialogue_id"]),
                int(selected_row["source_turn_index"]),
            )
        ]
        state = _visible_state(source, inventory_template=inventory_template)
        runtime_states.append(state.model_dump(mode="json"))
        card = card_by_id[str(selected_row["selected_strategy_id"])]
        pair_id = "rs6_pair_" + stable_hex(
            PROTOCOL,
            state.state_id,
            card.strategy_id,
            generation.digest(),
            n=24,
        )
        guidance_token_estimate = estimate_tokens(card.guidance_text)
        features = {
            "retrieval_score": float(selected_row["retrieval_score"]),
            "eligible_move_count": len(selected_row["eligible_move_ids"]),
            "history_turn_count": len(state.current_session_history),
            "current_user_token_estimate": estimate_tokens(
                state.current_user_text
            ),
            "guidance_token_estimate": guidance_token_estimate,
            "current_user_has_question_mark": (
                "?" in state.current_user_text
            ),
            **{
                f"flag__{key}": bool(value)
                for key, value in selected_row["observable_flags"].items()
            },
            **{
                f"move__{move}": move == selected_row["selected_move_id"]
                for move in TARGET_BY_MOVE
            },
        }
        selected_states.append(
            {
                "protocol": PROTOCOL,
                "pair_id": pair_id,
                "state_id": state.state_id,
                "user_id": state.user_id,
                "source_dialogue_id": selected_row["source_dialogue_id"],
                "source_turn_index": selected_row["source_turn_index"],
                "selected_strategy_card_id": card.strategy_id,
                "selected_strategy_family": card.strategy_label,
                "retrieval_score": selected_row["retrieval_score"],
                "observable_flags": selected_row["observable_flags"],
                "eligible_move_ids": selected_row["eligible_move_ids"],
                "transparent_pm_features": features,
            }
        )
        pair_seed = 1701 + index
        for arm, strategies in (("R0", []), ("RS", [card])):
            messages = generation_messages(
                state,
                [],
                strategies,
                system_prompt=generation.system_prompt,
            )
            call_plan.append(
                {
                    "protocol": PROTOCOL,
                    "pair_id": pair_id,
                    "state_id": state.state_id,
                    "user_id": state.user_id,
                    "arm": arm,
                    "action_id": "M0+R0" if arm == "R0" else "M0+RS",
                    "selected_strategy_card_id": (
                        None if arm == "R0" else card.strategy_id
                    ),
                    "selected_strategy_family": (
                        None if arm == "R0" else card.strategy_label
                    ),
                    "messages": messages,
                    "prompt_sha256": sha256_text(canonical_json(messages)),
                    "estimated_input_tokens": estimate_tokens(
                        canonical_json(messages)
                    ),
                    "generation": {
                        **generation.payload(),
                        "seed": pair_seed,
                    },
                    "generator_identity": generator_identity,
                    "api_call_made": False,
                }
            )

    calls_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in call_plan:
        calls_by_pair[str(row["pair_id"])].append(row)
    parity_ok = True
    for rows in calls_by_pair.values():
        if {str(row["arm"]) for row in rows} != {"R0", "RS"}:
            parity_ok = False
            continue
        r0 = next(row for row in rows if row["arm"] == "R0")
        rs = next(row for row in rows if row["arm"] == "RS")
        parity_ok &= (
            r0["generation"] == rs["generation"]
            and r0["generator_identity"] == rs["generator_identity"]
            and r0["messages"][0] == rs["messages"][0]
            and r0["messages"][1]["content"].split(
                "\n\nWrite only the counselor's next response."
            )[0]
            == rs["messages"][1]["content"].split(
                "\n\nPotential emotional-support guidance.", 1
            )[0]
        )

    conservative = _current_runtime_is_conservative(
        universe_by_key, rag, args.old_g2_results
    )
    selected_dialogues = {
        str(row["source_dialogue_id"]) for row in selected_states
    }
    move_counts = Counter(
        str(row["selected_strategy_family"]) for row in selected_states
    )
    ready = (
        len(selected_states) == 32
        and len(selected_dialogues) == 32
        and move_counts == Counter(target_by_move)
        and not (selected_dialogues & blocked)
        and parity_ok
        and conservative["passed"]
    )
    report = {
        "protocol": PROTOCOL,
        "status": (
            "READY_FOR_NO_PM_DIRECT_EFFECT_GENERATION"
            if ready
            else "BLOCKED_BY_SELECTION_OR_LINEAGE"
        ),
        "scope": (
            "train-only RS component-effect labels under the fixed six-card "
            "Bank; not an external effect estimate"
        ),
        "planned_pairs": len(selected_states),
        "planned_generation_calls": len(call_plan),
        "independent_dialogue_groups": len(selected_dialogues),
        "target_by_move": target_by_move,
        "selected_by_move": dict(sorted(move_counts.items())),
        "eligible_pool": {
            "accepted_final_source_disjoint_states": len(candidates),
            "accepted_final_source_disjoint_dialogues": len(
                {row["source_dialogue_id"] for row in candidates}
            ),
            "strict_all_g1_exposure_disjoint_states": len(strict_candidates),
            "strict_all_g1_exposure_disjoint_dialogues": len(
                {row["source_dialogue_id"] for row in strict_candidates}
            ),
            "why_primary_uses_accepted_source_disjoint": (
                "All-G1 exposure disjointness leaves too few independent "
                "opportunity dialogues for 32 pairs. Rejected weak candidates "
                "did not contribute text or examples to the final "
                "definition-only Bank."
            ),
        },
        "exclusions": {
            "accepted_g1_source_dialogues": len(accepted_g1_sources),
            "all_g1_exposed_dialogues_audit_only": len(all_g1_exposed),
            "prior_rs_development_dialogues": len(prior_development),
            "formal_esconv_test_dialogues": len(formal_test_dialogues),
            "prior_development_paths": prior_paths,
        },
        "baai_policy": {
            "used_for_selection_or_features": False,
            "reason": (
                "Post-hoc human-preference reranking diagnostic underperformed "
                "the lexical Top-1 baseline and rescued none of four holdout "
                "failures."
            ),
            "diagnostic_report": (
                "outputs/pm_v1_5_rs_bge_top3_reranking_diagnostic_v1/"
                "bge_reranking_report.json"
            ),
        },
        "runtime": {
            "protocol": RUNTIME_PROTOCOL,
            "path": "src/metacom_pm/v1_5_strategy_rag_runtime.py",
            "sha256": sha256_file(
                ROOT / "src/metacom_pm/v1_5_strategy_rag_runtime.py"
            ),
            "conservative_amendment_check": conservative,
        },
        "bank": {
            "path": str(args.bank.relative_to(ROOT)),
            "sha256": sha256_file(args.bank),
            "card_count": len(cards),
            "source_response_text_injected": False,
        },
        "selection": {
            "seed_protocol": SELECTION_SEED,
            "uses_hidden_next_supporter_response": False,
            "uses_native_strategy_label": False,
            "uses_formal_test_outcomes": False,
            "global_one_dialogue_per_pair": True,
        },
        "generator_identity": generator_identity,
        "supporter_generation_treatment_sha256": generation.digest(),
        "checks": {
            "exactly_32_pairs": len(selected_states) == 32,
            "exactly_32_independent_dialogues": len(selected_dialogues) == 32,
            "target_move_distribution_met": move_counts
            == Counter(target_by_move),
            "zero_blocked_dialogue_overlap": not (
                selected_dialogues & blocked
            ),
            "same_stack_except_strategy_section": parity_ok,
            "same_seed_within_each_pair": all(
                len({row["generation"]["seed"] for row in rows}) == 1
                for rows in calls_by_pair.values()
            ),
            "corrected_runtime_is_monotonic_fail_closed": conservative[
                "passed"
            ],
        },
        "next_gate": (
            "Generate 64 responses, conduct one blind material "
            "quality+risk review, and train only if both RS-on and R0 classes "
            "contain at least eight independent dialogue groups."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "plan_report.json", report)
    write_jsonl(args.out_dir / "runtime_states.jsonl", runtime_states)
    write_jsonl(args.out_dir / "selected_states.jsonl", selected_states)
    write_jsonl(args.out_dir / "call_plan.jsonl", call_plan)
    write_jsonl(args.out_dir / "selection_audit.jsonl", candidates)
    print(report)


if __name__ == "__main__":
    main()
