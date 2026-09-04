#!/usr/bin/env python3
"""Prepare fresh same-stack R0/RS pairs under the corrected RS construct."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import json
from pathlib import Path
from typing import Any

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import (
    DialogueTurn,
    MemorySource,
    RuntimeState,
    SourceCatalog,
)
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
PROTOCOL = "pm-v1.5-rs-paired-effect-plan-v1"
PROMPT_PROTOCOL = "pm-v1.5-rs-final-bank-transparent-top1-prompt-v1"
TARGETS = {
    "Question": 7,
    "Providing Suggestions": 7,
    "Affirmation and Reassurance": 6,
    "Restatement or Paraphrasing": 6,
    "Reflection of feelings": 6,
}


def _seeker_texts(dialogue: list[dict[str, str]]) -> list[str]:
    return [
        turn["content"]
        for turn in dialogue
        if turn["speaker"] == "seeker" and turn["content"]
    ]


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
    recent = " ".join(_seeker_texts(dialogue)[-3:])
    return (
        "Choose one safe topic-agnostic emotional-support technique. "
        f"Observable cues: {', '.join(cues) or 'none'}. "
        f"Recent seeker context: {recent}"
    )


def _state(source: dict[str, Any], dialogue: list[dict[str, str]]) -> RuntimeState:
    latest = dialogue[-1]["content"]
    history = [
        DialogueTurn(
            role="user" if row["speaker"] == "seeker" else "assistant",
            content=row["content"],
        )
        for row in dialogue[:-1]
    ]
    state_id = "rs_effect_state_" + stable_hex(
        PROTOCOL,
        source["source_dialogue_id"],
        source["source_turn_index"],
        sha256_text(canonical_json(dialogue)),
        n=24,
    )
    inventory = {
        source_key: SourceCatalog(
            available=False,
            count=0,
            estimated_tokens=0,
            catalog_fingerprint=[],
            semantic_query_similarity=0.0,
            semantic_representation_valid=False,
        )
        for source_key in MemorySource
    }
    return RuntimeState(
        state_id=state_id,
        card_id="card_" + stable_hex(PROTOCOL, state_id, n=24),
        user_id=str(source["source_dialogue_id"]),
        split="development",
        semantic_family=str(source.get("problem_type") or "unknown"),
        current_user_text=latest,
        current_session_history=history,
        current_session_summary="",
        session_index=max(1, int(source["source_turn_index"]) + 1),
        inventory=inventory,
        allowed_actions=["M0+R0", "M0+RS"],
        provenance={
            "source_dialogue_id": str(source["source_dialogue_id"]),
            "source_turn_index": int(source["source_turn_index"]),
            "visible_dialogue_sha256": sha256_text(
                canonical_json(dialogue)
            ),
        },
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


def _collect_exclusions(
    *,
    h1_packet: Path,
    h1_private: Path,
    relink_candidates: Path,
    h2_states: Path,
    bank_provenance: Path,
    prior_selected_glob: str,
) -> tuple[set[str], dict[str, int], list[str]]:
    h1 = json.loads(h1_packet.read_text(encoding="utf-8"))
    groups: dict[str, set[str]] = {
        "h1_source_examples": {
            str(example["source_dialogue_id"])
            for item in h1["card_items"]
            for example in item["source_examples"]
        },
        "h1_query_dialogues": {
            str(row["source_dialogue_id"]) for row in iter_jsonl(h1_private)
        },
        "source_relink_candidates": {
            str(row["source_dialogue_id"])
            for row in iter_jsonl(relink_candidates)
        },
        "consumed_h2_dialogues": {
            str(row["source_dialogue_id"]) for row in iter_jsonl(h2_states)
        },
        "final_bank_source_dialogues": {
            str(row["source_dialogue_id"])
            for row in iter_jsonl(bank_provenance)
        },
    }
    prior_paths = sorted(glob.glob(prior_selected_glob))
    prior: set[str] = set()
    for path_text in prior_paths:
        for row in iter_jsonl(Path(path_text)):
            value = row.get("source_dialogue_id") or row.get("user_id")
            if isinstance(value, str) and value.startswith("esconv_"):
                prior.add(value)
    groups["prior_rs_pair_dialogues"] = prior
    union = set().union(*groups.values())
    counts = {key: len(value) for key, value in groups.items()}
    counts["union"] = len(union)
    return union, counts, prior_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_esconv_train_strategy_inductive_audit_v1/"
        "clean_train_strategy_universe.jsonl",
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
        "--construct",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/rs_open_close_construct_v2.json",
    )
    parser.add_argument(
        "--h1-packet",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_h1_complete_rag_review_v1_candidate/"
        "h1_review_packet.json",
    )
    parser.add_argument(
        "--h1-private",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_h1_complete_rag_review_v1_candidate/"
        "private_retrieval_audit.jsonl",
    )
    parser.add_argument(
        "--relink-candidates",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_h1_source_relink_review_v1_candidate/"
        "private_candidate_scores.jsonl",
    )
    parser.add_argument(
        "--h2-states",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_h2_retrieval_state_freeze_v1/"
        "h2_states_private.jsonl",
    )
    parser.add_argument(
        "--prior-selected-glob",
        default=str(ROOT / "outputs/pm_v1_5_*/selected_states.jsonl"),
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
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_rs_paired_effect_v1",
    )
    args = parser.parse_args()

    contract = json.loads(args.construct.read_text(encoding="utf-8"))
    cards = [dict(row) for row in iter_jsonl(args.bank)]
    if (
        contract["protocol"] != "pm-v1.5-rs-open-close-construct-v2"
        or len(cards) != 80
        or any(row["raw_source_response_exposed_to_generator"] for row in cards)
    ):
        raise RuntimeError("corrected construct or final Bank is invalid")
    card_by_id = {str(row["card_id"]): row for row in cards}
    excluded, exclusion_counts, prior_paths = _collect_exclusions(
        h1_packet=args.h1_packet,
        h1_private=args.h1_private,
        relink_candidates=args.relink_candidates,
        h2_states=args.h2_states,
        bank_provenance=args.bank_provenance,
        prior_selected_glob=args.prior_selected_glob,
    )

    seen: set[tuple[str, str]] = set()
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    audit: list[dict[str, Any]] = []
    for source in iter_jsonl(args.universe):
        dialogue_id = str(source["source_dialogue_id"])
        dialogue = [
            {
                "speaker": str(turn["speaker"]),
                "content": normalize_space(turn.get("content", "")),
            }
            for turn in source["recent_dialogue"]
            if str(turn.get("speaker")) in {"seeker", "supporter"}
            and normalize_space(turn.get("content", ""))
        ]
        if (
            dialogue_id in excluded
            or not dialogue
            or dialogue[-1]["speaker"] != "seeker"
        ):
            continue
        digest = sha256_text(canonical_json(dialogue))
        key = (dialogue_id, digest)
        if key in seen:
            continue
        seen.add(key)
        seekers = _seeker_texts(dialogue)
        latest = seekers[-1]
        recent = " ".join(seekers[-3:])
        flags = effect_study_observable_flags(
            current_user_text=latest,
            recent_user_text=recent,
            visible_dialogue=dialogue,
        )
        query = _query(dialogue, flags)
        ranked = effect_study_rank_applicable_cards(
            query=query,
            current_user_text=latest,
            recent_user_text=recent,
            visible_dialogue=dialogue,
            cards=cards,
        )
        top = ranked[0] if ranked else None
        row = {
            "source": dict(source),
            "dialogue": dialogue,
            "visible_dialogue_sha256": digest,
            "flags": flags,
            "query": query,
            "ranked_top3": ranked[:3],
            "selected": top,
        }
        audit.append(
            {
                "source_dialogue_id": dialogue_id,
                "source_turn_index": int(source["source_turn_index"]),
                "problem_type": str(source.get("problem_type") or "unknown"),
                "visible_dialogue_sha256": digest,
                "ordinary_rag_hard_off": flags["ordinary_rag_hard_off"],
                "ordinary_rag_hard_off_reasons": flags[
                    "ordinary_rag_hard_off_reasons"
                ],
                "selected_card_id": None if top is None else top["card_id"],
                "selected_family": (
                    None if top is None else top["strategy_family"]
                ),
                "selected_core_submove_id": (
                    None if top is None else top["core_submove_id"]
                ),
                "top3_card_ids": [value["card_id"] for value in ranked[:3]],
            }
        )
        if top is not None:
            pools[str(top["strategy_family"])].append(row)

    chosen: list[dict[str, Any]] = []
    used_dialogues: set[str] = set()
    for family in EXPECTED_FAMILIES:
        target = TARGETS[family]
        problem_counts: Counter[str] = Counter()
        family_rows = sorted(
            pools[family],
            key=lambda row: stable_hex(
                PROTOCOL,
                family,
                row["source"]["source_dialogue_id"],
                row["visible_dialogue_sha256"],
                n=32,
            ),
        )
        for row in family_rows:
            dialogue_id = str(row["source"]["source_dialogue_id"])
            problem = str(row["source"].get("problem_type") or "unknown")
            if dialogue_id in used_dialogues or problem_counts[problem] >= 3:
                continue
            chosen.append(row)
            used_dialogues.add(dialogue_id)
            problem_counts[problem] += 1
            if sum(
                item["selected"]["strategy_family"] == family
                for item in chosen
            ) == target:
                break
        observed = sum(
            item["selected"]["strategy_family"] == family for item in chosen
        )
        if observed != target:
            raise RuntimeError(f"family {family}: selected {observed}/{target}")

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
        key=lambda value: (
            EXPECTED_FAMILIES.index(
                str(value["selected"]["strategy_family"])
            ),
            str(value["source"]["source_dialogue_id"]),
        ),
    ):
        source = row["source"]
        state = _state(source, row["dialogue"])
        card = card_by_id[str(row["selected"]["card_id"])]
        pair_id = "rs_effect_pair_" + stable_hex(
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
                "source_dialogue_id": source["source_dialogue_id"],
                "source_turn_index": source["source_turn_index"],
                "problem_type": source.get("problem_type") or "unknown",
                "current_user_text": state.current_user_text,
                "visible_dialogue": row["dialogue"],
                "observable_flags": row["flags"],
                "selected_card_id": card["card_id"],
                "selected_core_submove_id": card["core_submove_id"],
                "selected_strategy_family": card["strategy_family"],
                "selected_execution_profile": card["execution_profile"],
                "retrieval_score": row["selected"]["score"],
                "compatibility_tier": row["selected"][
                    "compatibility_tier"
                ],
                "outcome_label": "UNKNOWN_PENDING_PAIRED_RESPONSE_MEASUREMENT",
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
        if len(calls) != 2:
            parity_ok = False
            continue
        r0 = next(row for row in calls if row["arm"] == "R0")
        rs = next(row for row in calls if row["arm"] == "RS")
        r0_base = r0["messages"][1]["content"].split(
            "\n\nWrite only the counselor's next response."
        )[0]
        rs_base = rs["messages"][1]["content"].split(
            "\n\nPotential emotional-support technique.", 1
        )[0]
        parity_ok &= (
            r0["generation"] == rs["generation"]
            and r0["generator_identity"] == rs["generator_identity"]
            and r0["messages"][0] == rs["messages"][0]
            and r0_base == rs_base
        )

    family_counts = Counter(
        str(row["selected_strategy_family"]) for row in selected_rows
    )
    ready = (
        len(selected_rows) == sum(TARGETS.values())
        and len({row["source_dialogue_id"] for row in selected_rows})
        == len(selected_rows)
        and family_counts == Counter(TARGETS)
        and parity_ok
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "selected_states.jsonl", selected_rows)
    write_jsonl(args.out_dir / "call_plan.jsonl", call_plan)
    write_jsonl(args.out_dir / "retrieval_audit.jsonl", audit)
    report = {
        "protocol": PROTOCOL,
        "status": (
            "READY_FOR_NO_PM_DIRECT_EFFECT_GENERATION"
            if ready
            else "PAIR_PLAN_NOT_READY"
        ),
        "scope": "fresh_esconv_train_same_stack_rs_component_effect",
        "construct_protocol": contract["protocol"],
        "effect_study_protocol": EFFECT_STUDY_PROTOCOL,
        "api_calls_made": 0,
        "selected_pair_count": len(selected_rows),
        "selected_independent_user_groups": len(
            {row["source_dialogue_id"] for row in selected_rows}
        ),
        "planned_generation_calls": len(call_plan),
        "selected_family_counts": dict(sorted(family_counts.items())),
        "family_targets": TARGETS,
        "candidate_pool_counts": {
            family: len(pools[family]) for family in EXPECTED_FAMILIES
        },
        "excluded_dialogue_counts": exclusion_counts,
        "prior_selected_state_paths": prior_paths,
        "ranker": "transparent_compatibility_tier_then_lexical",
        "bge_used": False,
        "runtime_retrieve_top_k_diagnostic": 3,
        "runtime_inject_top_k": 1,
        "generator_identity": generator_identity,
        "supporter_generation_treatment_sha256": generation.digest(),
        "checks": {
            "final_bank_80_cards": len(cards) == 80,
            "raw_source_response_absent": all(
                not row["raw_source_response_exposed_to_generator"]
                for row in cards
            ),
            "h1_h2_bank_and_prior_pair_dialogues_excluded": not (
                {row["source_dialogue_id"] for row in selected_rows}
                & excluded
            ),
            "one_state_per_dialogue": len(selected_rows)
            == len({row["source_dialogue_id"] for row in selected_rows}),
            "same_stack_except_strategy_section": parity_ok,
            "one_transparent_top1_card_per_rs_arm": all(
                row["selected_strategy_card_id"] is not None
                for row in call_plan
                if row["arm"] == "RS"
            ),
            "no_outcome_or_h2_label_used_for_selection": True,
            "all_effect_labels_unknown_before_generation": all(
                row["outcome_label"]
                == "UNKNOWN_PENDING_PAIRED_RESPONSE_MEASUREMENT"
                for row in selected_rows
            ),
        },
        "inputs": {
            "universe": str(args.universe.relative_to(ROOT)),
            "universe_sha256": sha256_file(args.universe),
            "bank": str(args.bank.relative_to(ROOT)),
            "bank_sha256": sha256_file(args.bank),
            "construct": str(args.construct.relative_to(ROOT)),
            "construct_sha256": sha256_file(args.construct),
        },
        "interpretation": (
            "The 32 states are selected before any new response outcome by "
            "fixed transparent Top-1 family coverage. Retrieval suitability "
            "does not create an open label. Only later same-state R0/RS "
            "quality, atomic risk, and cost measurement can create effect "
            "labels for the RS head."
        ),
    }
    write_json(args.out_dir / "plan_report.json", report)
    manifest = {
        "protocol": PROTOCOL,
        "status": report["status"],
        "outputs": {
            name: sha256_file(args.out_dir / name)
            for name in (
                "selected_states.jsonl",
                "call_plan.jsonl",
                "retrieval_audit.jsonl",
                "plan_report.json",
            )
        },
    }
    write_json(args.out_dir / "manifest.json", manifest)
    print(
        {
            "protocol": PROTOCOL,
            "status": report["status"],
            "pairs": len(selected_rows),
            "calls": len(call_plan),
            "families": dict(sorted(family_counts.items())),
        }
    )


if __name__ == "__main__":
    main()
