#!/usr/bin/env python3
"""Prepare a no-PM matched R0/RS direct-effect pilot for Strategy RAG V4."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
from metacom_pm.text import estimate_tokens, lexical_score, normalize_space
from metacom_pm.v1_5_strategy_rag_v4 import (
    EXPECTED_FAMILIES,
    V4_PROTOCOL,
    eligible_families,
    observable_opportunity_flags,
    selected_execution_profile,
    validate_v4_candidate_cards,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-strategy-rag-v4-direct-effect-plan-v4"
PROMPT_PROTOCOL = "pm-v1.5-strategy-rag-v4-top1-technique-prompt-v3"
DEFAULT_NLI = Path(
    "/home/tokkio/.cache/huggingface/hub/"
    "models--cross-encoder--nli-deberta-v3-base/snapshots/"
    "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
)


def _dialogue_text(state: RuntimeState) -> str:
    history = list(state.current_session_history)
    def role_name(turn: Any) -> str:
        value = getattr(turn, "role", "")
        return str(getattr(value, "value", value))

    if (
        history
        and role_name(history[-1]) == "user"
        and normalize_space(history[-1].content).casefold()
        == normalize_space(state.current_user_text).casefold()
    ):
        history = history[:-1]
    lines = [
        f"{'User' if role_name(turn) == 'user' else 'Supporter'}: "
        f"{normalize_space(turn.content)}"
        for turn in history[-8:]
    ]
    lines.append(f"User: {normalize_space(state.current_user_text)}")
    return "\n".join(lines)


def _recent_user_text(state: RuntimeState) -> str:
    rows = [
        normalize_space(turn.content)
        for turn in state.current_session_history
        if str(getattr(getattr(turn, "role", ""), "value", turn.role)) == "user"
    ]
    rows.append(normalize_space(state.current_user_text))
    return " ".join(rows[-3:])


def _nli_scores(
    *,
    pairs: list[tuple[str, str]],
    model_path: Path,
    batch_size: int,
) -> list[float]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    )
    # In this workstation's Conda runtime torch enumerates the A6000 as
    # cuda:0 even though nvidia-smi displays it second.  Prefer the device with
    # the greatest total memory instead of assuming system display order.
    if torch.cuda.is_available():
        device_index = max(
            range(torch.cuda.device_count()),
            key=lambda index: torch.cuda.get_device_properties(index).total_memory,
        )
        device = torch.device(f"cuda:{device_index}")
    else:
        device = torch.device("cpu")
    model.to(device).eval()
    entailment_id = int(model.config.label2id["entailment"])
    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            encoded = tokenizer(
                [row[0] for row in batch_pairs],
                [row[1] for row in batch_pairs],
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            probabilities = torch.softmax(model(**encoded).logits, dim=-1)
            scores.extend(
                float(value)
                for value in probabilities[:, entailment_id].cpu().tolist()
            )
    return scores


def _emotion_terms(text: str) -> set[str]:
    terms = set(
        re.findall(
            r"\b(?:sad|angry|upset|hurt|afraid|scared|anxious|worried|"
            r"nervous|overwhelmed|exhausted|tired|lonely|frustrated|"
            r"confused|embarrassed|ashamed|guilty|disappointed|stressed|"
            r"depressed|discouraged|lost|hopeless)\b",
            text,
            flags=re.I,
        )
    )
    for negated in re.findall(
        r"\b(?:not|never|no longer)\s+"
        r"(sad|angry|upset|hurt|afraid|scared|anxious|worried|"
        r"nervous|overwhelmed|exhausted|tired|lonely|frustrated|"
        r"confused|embarrassed|ashamed|guilty|disappointed|stressed|"
        r"depressed|discouraged|lost|hopeless)\b",
        text,
        flags=re.I,
    ):
        terms.discard(negated.casefold())
    return terms


def _transparent_route(
    flags: dict[str, bool],
    current_user_text: str,
    recent_user_text: str,
    *,
    allow_weak_source_fail: bool,
) -> tuple[str, str] | None:
    """Return a source-supported card route justified by visible cues only."""

    text = normalize_space(current_user_text)
    context = normalize_space(recent_user_text)
    lowered = text.casefold()
    if flags["advice_welcome"]:
        if re.search(
            r"\b(?:say|tell|talk|speak|conversation|communicat|message)\b",
            text,
            re.I,
        ):
            submove = "suggestion_communication_opening"
        elif re.search(
            r"\b(?:start|first|begin|job|application|apply)\b", text, re.I
        ):
            submove = "suggestion_break_down_first_step"
        elif re.search(
            r"\b(?:doctor|therap|counsel|professional|specialist)\b",
            text,
            re.I,
        ):
            submove = "suggestion_consider_qualified_support"
        elif re.search(
            r"\b(?:cope|feel better|mood|sad|depress|motivat|hobby)\b",
            text,
            re.I,
        ):
            submove = "suggestion_small_experiment"
        elif re.search(
            r"\b(?:friend|family|relative|partner|well[- ]?wisher|"
            r"someone|support person)\b",
            context,
            re.I,
        ):
            submove = "suggestion_reach_trusted_support"
        elif re.search(
            r"\b(?:breakup|broke up|split|sad|lonely|panic|anxious|"
            r"overwhelmed|settle|calm|right now)\b",
            context,
            re.I,
        ):
            submove = "suggestion_brief_pause_or_grounding"
        elif re.search(
            r"\b(?:too many|everything at once|which first|prioriti[sz])\b",
            context,
            re.I,
        ):
            submove = "suggestion_prioritize_one_focus"
        elif re.search(
            r"\b(?:routine|sleep|workspace|room|environment|concentrat)\b",
            context,
            re.I,
        ):
            submove = "suggestion_adjust_environment"
        else:
            submove = "suggestion_single_microstep"
        return "Providing Suggestions", submove

    if flags["listen_only"]:
        return (
            "Restatement or Paraphrasing",
            "restatement_boundary_preference",
        )

    if flags["effort_or_progress_visible"]:
        if re.search(
            r"\b(?:managed|started|finished|finally|improved|"
            r"made progress|first step)\b",
            text,
            re.I,
        ):
            submove = "affirmation_acknowledge_progress"
        elif re.search(
            r"\b(?:choice|choose|decid|figure out|my decision)\b",
            text,
            re.I,
        ):
            submove = "affirmation_support_agency"
        else:
            submove = "affirmation_recognize_persistence"
        return "Affirmation and Reassurance", submove

    if flags["uncertainty_or_multi_concern"]:
        if re.search(
            r"\b(?:not sure|don't know|dont know)\s+how\b.+"
            r"\b(?:help|work|matter|change|make a difference)\b",
            text,
            re.I,
        ):
            # This asks for an explanation. A technique-only Question card
            # would add another probe instead of supplying the needed content.
            return None
        if re.search(r"\b(?:mean|meaning|understand)\b", text, re.I):
            submove = "question_clarify_meaning"
        elif re.search(
            r"\b(?:feel|emotion|sad|angry|afraid|anxious|worried)\b",
            text,
            re.I,
        ):
            submove = "question_clarify_feeling"
        else:
            submove = "question_clarify_goal"
        return "Question", submove

    emotion_terms = _emotion_terms(lowered)
    if flags["emotion_visible"]:
        if re.search(
            r"\b(?:still|always|all day|every day|for (?:days|weeks|months)|"
            r"keeps?|recurring|won't go away|can't make it go away)\b",
            text,
            re.I,
        ):
            return (
                "Reflection of feelings",
                "reflection_lingering_feeling",
            )
        if len(emotion_terms) >= 2:
            return "Reflection of feelings", "reflection_mixed_feelings"
        if re.search(
            r"\b(?:love|care about|important|matters? to me|means a lot|"
            r"miss|breaks? my heart)\b",
            text,
            re.I,
        ):
            return "Reflection of feelings", "reflection_value_tension"
        if allow_weak_source_fail:
            if re.search(
                r"\b(?:this|that|it)\s+(?:is|feels|was)\s+"
                r"(?:painful|difficult|hard)\b",
                text,
                re.I,
            ):
                return (
                    "Reflection of feelings",
                    "reflection_emotional_load",
                )
            return "Reflection of feelings", "reflection_explicit_emotion"
        return (
            "Affirmation and Reassurance",
            "affirmation_validate_reaction",
        )

    # Restatement is useful only when a visible object of paraphrase exists.
    if re.search(
        r"(?:\bpart of me\b.+\b(?:but|while|and)\b.+\bpart of me\b|"
        r"\bon the one hand\b.+\bon the other\b|"
        r"\b(?:want|hope|feel|would|could)\b.+\b(?:but|though|yet)\b"
        r".+\b(?:want|hope|feel|would|could)\b)",
        text,
        re.I,
    ):
        return (
            "Restatement or Paraphrasing",
            "restatement_two_sides",
        )
    if re.search(
        r"\b(?:i\s+(?:want|hope|need|plan|am trying)|"
        r"my\s+goal)\b",
        text,
        re.I,
    ):
        return (
            "Restatement or Paraphrasing",
            "restatement_stated_goal",
        )
    if re.search(
        r"\b(?:can't|cannot|unable|hard to|not the same|affect|impact|"
        r"prevent)\b",
        text,
        re.I,
    ):
        return (
            "Restatement or Paraphrasing",
            "restatement_current_impact",
        )
    if re.search(r"\b(?:most|main|priority|first concern)\b", text, re.I):
        return (
            "Restatement or Paraphrasing",
            "restatement_current_priority",
        )
    return None


def _transparent_query(
    *,
    family: str,
    submove_id: str,
    flags: dict[str, bool],
    current_user_text: str,
) -> str:
    text = normalize_space(current_user_text)
    if family == "Providing Suggestions":
        scope = (
            "one optional concrete microstep first manageable step"
            if flags["low_burden"]
            else "small optional suggestions directions linked to stated goal"
        )
        return f"Advice is explicitly welcome. {scope}. User: {text}"
    if family == "Affirmation and Reassurance":
        return (
            "Acknowledge a specific visible effort, progress, persistence, "
            f"or action without generic praise. User: {text}"
        )
    if family == "Question":
        return (
            "Ask one focused clarification about the current goal, priority, "
            f"meaning, or missing context. User expresses uncertainty: {text}"
        )
    if family == "Reflection of feelings":
        return (
            "Reflect the explicit emotion, uncertainty, or emotional load "
            f"proportionately without advice. User: {text}"
        )
    tension = (
        "Restate the two visible sides or central tension."
        if re.search(r"\b(?:but|though|yet|part of me)\b", text, re.I)
        else "Restate the central concern or current impact accurately."
    )
    return f"{tension} Target atomic move: {submove_id}. User: {text}"


def _messages(
    *,
    state: RuntimeState,
    system_prompt: str,
    card: dict[str, Any] | None,
) -> list[dict[str, str]]:
    visible = state.model_copy(update={"current_session_summary": ""})
    sections = [common_context(visible)]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-states",
        type=Path,
        default=(
            ROOT
            / "data/esconv_auxiliary_v1_5_visible_v2_candidate/train"
            / "runtime_states.jsonl"
        ),
    )
    parser.add_argument(
        "--cards",
        type=Path,
        default=(
            ROOT
            / "outputs/pm_v1_5_strategy_rag_v4_candidate_v1"
            / "strategy_cards_v4_candidate.jsonl"
        ),
    )
    parser.add_argument(
        "--pm-config", type=Path, default=ROOT / "configs/pm_v1_5.yaml"
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=ROOT / "configs/experiment.yaml",
    )
    parser.add_argument("--nli-model", type=Path, default=DEFAULT_NLI)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--ranker",
        choices=("transparent_rule", "transparent_lexical", "nli"),
        default="transparent_rule",
    )
    parser.add_argument("--minimum-score", type=float, default=0.02)
    parser.add_argument("--minimum-margin", type=float, default=0.0)
    parser.add_argument("--states-per-family", type=int, default=4)
    parser.add_argument(
        "--minimum-per-family",
        type=int,
        default=None,
        help=(
            "Override the development-plan readiness minimum for every "
            "included family. This changes readiness reporting, not sampling."
        ),
    )
    parser.add_argument(
        "--include-families",
        nargs="+",
        choices=EXPECTED_FAMILIES,
        default=list(EXPECTED_FAMILIES),
        help=(
            "Restrict a targeted treatment-qualification plan to these "
            "strategy families."
        ),
    )
    parser.add_argument(
        "--allow-weak-source-fail",
        action="store_true",
        help="Allow cards that did not pass the provisional weak-source gate.",
    )
    parser.add_argument(
        "--exclude-selected-states",
        nargs="*",
        type=Path,
        default=[],
        help=(
            "Prior selected_states.jsonl files whose user groups must be "
            "excluded from this plan."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_strategy_rag_v4_direct_effect_v4",
    )
    args = parser.parse_args()
    if args.minimum_per_family is not None and args.minimum_per_family < 1:
        raise ValueError("--minimum-per-family must be positive")
    included_families = [
        family for family in EXPECTED_FAMILIES if family in args.include_families
    ]
    excluded_user_ids = {
        str(row["user_id"])
        for path in args.exclude_selected_states
        for row in iter_jsonl(path)
    }

    states = [
        RuntimeState.model_validate(row)
        for row in iter_jsonl(args.runtime_states)
    ]
    cards = validate_v4_candidate_cards(
        [dict(row) for row in iter_jsonl(args.cards)]
    )
    if not args.allow_weak_source_fail:
        active_cards = [
            row
            for row in cards
            if row["source_support"]["provisional_source_support_pass"]
        ]
    else:
        active_cards = cards

    unique_states: list[RuntimeState] = []
    seen_visible: set[tuple[str, str]] = set()
    phatic_controls: list[dict[str, Any]] = []
    opportunity_rows: list[dict[str, Any]] = []
    pair_specs: list[tuple[int, str, str]] = []
    for state in states:
        if state.user_id in excluded_user_ids:
            continue
        visible_key = (state.user_id, sha256_text(_dialogue_text(state)))
        if visible_key in seen_visible:
            continue
        seen_visible.add(visible_key)
        flags = observable_opportunity_flags(
            current_user_text=state.current_user_text,
            recent_user_text=_recent_user_text(state),
        )
        families = eligible_families(flags)
        if flags["pure_phatic"] or flags["explicit_stop"]:
            phatic_controls.append(
                {
                    "state_id": state.state_id,
                    "user_id": state.user_id,
                    "current_user_text": state.current_user_text,
                    "flags": flags,
                    "eligible_families": list(families),
                    "expected_action": "M0+R0",
                }
            )
        if not families:
            continue
        route = _transparent_route(
            flags,
            state.current_user_text,
            _recent_user_text(state),
            allow_weak_source_fail=args.allow_weak_source_fail,
        )
        if route is None:
            continue
        target_family, target_submove_id = route
        if target_family not in families:
            raise RuntimeError(
                "transparent route escaped the broad safety gate: "
                f"{state.state_id}/{target_family}"
            )
        if target_family not in included_families:
            continue
        state_index = len(unique_states)
        unique_states.append(state)
        profile = selected_execution_profile(flags)
        row = {
            "state_index": state_index,
            "state_id": state.state_id,
            "user_id": state.user_id,
            "dialogue_text": _dialogue_text(state),
            "current_user_text": state.current_user_text,
            "flags": flags,
            "execution_profile": profile,
            "eligible_families": list(families),
            "target_family": target_family,
            "target_submove_id": target_submove_id,
            "transparent_query": _transparent_query(
                family=target_family,
                submove_id=target_submove_id,
                flags=flags,
                current_user_text=state.current_user_text,
            ),
            "candidate_scores": [],
        }
        opportunity_rows.append(row)
        for card in active_cards:
            if (
                card["strategy_family"] == target_family
                and card["execution_profile"] == profile
                and (
                    args.ranker != "transparent_rule"
                    or card["core_submove_id"] == target_submove_id
                )
            ):
                pair_specs.append(
                    (
                        state_index,
                        str(card["card_id"]),
                        str(card["when_to_use"]),
                    )
                )

    state_by_index = {index: state for index, state in enumerate(unique_states)}
    opportunity_by_index = {
        int(row["state_index"]): row for row in opportunity_rows
    }
    card_by_id = {str(row["card_id"]): row for row in active_cards}
    if args.ranker == "nli":
        nli_pairs = [
            (_dialogue_text(state_by_index[index]), hypothesis)
            for index, _, hypothesis in pair_specs
        ]
        scores = _nli_scores(
            pairs=nli_pairs,
            model_path=args.nli_model,
            batch_size=args.batch_size,
        )
    elif args.ranker == "transparent_lexical":
        scores = [
            lexical_score(
                str(opportunity_by_index[index]["transparent_query"]),
                str(card_by_id[card_id]["retrieval_text"]),
            )
            for index, card_id, _ in pair_specs
        ]
    else:
        scores = [1.0 for _ in pair_specs]
    if len(scores) != len(pair_specs):
        raise RuntimeError("NLI score count differs from candidate pair count")
    for (state_index, card_id, _), score in zip(
        pair_specs, scores, strict=True
    ):
        card = card_by_id[card_id]
        opportunity_by_index[state_index]["candidate_scores"].append(
            {
                "card_id": card_id,
                "core_submove_id": card["core_submove_id"],
                "strategy_family": card["strategy_family"],
                "execution_profile": card["execution_profile"],
                "score": round(float(score), 8),
            }
        )

    retrieval_rows: list[dict[str, Any]] = []
    for row in opportunity_rows:
        candidates = sorted(
            row["candidate_scores"],
            key=lambda item: (item["score"], item["card_id"]),
            reverse=True,
        )
        top = candidates[0] if candidates else None
        second_score = (
            float(candidates[1]["score"])
            if len(candidates) > 1
            else 0.0
        )
        top_score = float(top["score"]) if top else 0.0
        margin = top_score - second_score
        accepted = (
            top is not None
            and top_score >= args.minimum_score
            and margin >= args.minimum_margin
        )
        retrieval_rows.append(
            {
                **{key: value for key, value in row.items() if key != "candidate_scores"},
                "ranked_within_target_family": candidates,
                "selected": top if accepted else None,
                "top_score": round(top_score, 8),
                "top_vs_second_card_margin": round(margin, 8),
                "accepted": accepted,
                "rejection_reason": (
                    None
                    if accepted
                    else (
                        "below_score"
                        if top_score < args.minimum_score
                        else "below_card_margin"
                    )
                ),
            }
        )

    selected_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in retrieval_rows:
        if row["accepted"]:
            selected_by_family[str(row["selected"]["strategy_family"])].append(
                row
            )
    chosen: list[dict[str, Any]] = []
    chosen_users: set[str] = set()
    # Allocate scarce families first. This is a deterministic greedy
    # group-disjoint matching, not five independent within-family samples.
    family_allocation_order = sorted(
        included_families,
        key=lambda family: (
            len(
                {
                    str(row["user_id"])
                    for row in selected_by_family[family]
                }
            ),
            EXPECTED_FAMILIES.index(family),
        ),
    )
    for family in family_allocation_order:
        family_user_count = 0
        family_rows = sorted(
            selected_by_family[family],
            key=lambda row: (
                float(row["top_score"]),
                float(row["top_vs_second_card_margin"]),
                str(row["state_id"]),
            ),
            reverse=True,
        )
        for row in family_rows:
            user_id = str(row["user_id"])
            if user_id in chosen_users:
                continue
            chosen.append(row)
            chosen_users.add(user_id)
            family_user_count += 1
            if family_user_count >= args.states_per_family:
                break
    chosen.sort(
        key=lambda row: (
            EXPECTED_FAMILIES.index(str(row["selected"]["strategy_family"])),
            str(row["user_id"]),
            str(row["state_id"]),
        )
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
    call_plan: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for row in chosen:
        state = state_by_index[int(row["state_index"])]
        selected_card = card_by_id[str(row["selected"]["card_id"])]
        pair_id = "rag_v4_pair_" + stable_hex(
            PROTOCOL,
            state.state_id,
            selected_card["card_id"],
            generation.digest(),
            args.seed,
            n=24,
        )
        selected_rows.append(
            {
                "pair_id": pair_id,
                "state_id": state.state_id,
                "user_id": state.user_id,
                "current_user_text": state.current_user_text,
                "observable_flags": row["flags"],
                "eligible_families": row["eligible_families"],
                "selected_card_id": selected_card["card_id"],
                "selected_core_submove_id": selected_card["core_submove_id"],
                "selected_strategy_family": selected_card["strategy_family"],
                "selected_execution_profile": selected_card[
                    "execution_profile"
                ],
                "selected_provisional_source_support_pass": bool(
                    selected_card["source_support"][
                        "provisional_source_support_pass"
                    ]
                ),
                "retrieval_score": row["top_score"],
                "top_vs_second_card_margin": row[
                    "top_vs_second_card_margin"
                ],
            }
        )
        for arm, card in (("R0", None), ("RS", selected_card)):
            messages = _messages(
                state=state,
                system_prompt=generation.system_prompt,
                card=card,
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
                        None if card is None else card["card_id"]
                    ),
                    "selected_strategy_family": (
                        None if card is None else card["strategy_family"]
                    ),
                    "messages": messages,
                    "prompt_sha256": sha256_text(canonical_json(messages)),
                    "estimated_input_tokens": estimate_tokens(
                        canonical_json(messages)
                    ),
                    "generation": {
                        **generation.payload(),
                        "seed": args.seed,
                    },
                    "generator_identity": generator_identity,
                    "api_call_made": False,
                }
            )

    family_counts = Counter(
        str(row["selected_strategy_family"]) for row in selected_rows
    )
    parity_ok = True
    for selected in selected_rows:
        pair_calls = [
            row for row in call_plan if row["pair_id"] == selected["pair_id"]
        ]
        if len(pair_calls) != 2:
            parity_ok = False
            continue
        r0 = next(row for row in pair_calls if row["arm"] == "R0")
        rs = next(row for row in pair_calls if row["arm"] == "RS")
        parity_ok &= (
            r0["generation"] == rs["generation"]
            and r0["generator_identity"] == rs["generator_identity"]
            and r0["messages"][0] == rs["messages"][0]
            and r0["messages"][1]["content"].split(
                "\n\nWrite only the counselor's next response."
            )[0]
            == rs["messages"][1]["content"].split(
                "\n\nPotential emotional-support technique.", 1
            )[0]
        )
    minimum_by_family = {
        family: min(
            (
                args.minimum_per_family
                if args.minimum_per_family is not None
                else (2 if family == "Question" else 3)
            ),
            args.states_per_family,
        )
        for family in included_families
    }
    family_minimum_met = all(
        family_counts.get(family, 0) >= minimum_by_family[family]
        for family in included_families
    )
    ready = bool(selected_rows) and parity_ok and family_minimum_met
    report = {
        "protocol": PROTOCOL,
        "status": (
            "READY_FOR_NO_PM_DIRECT_EFFECT_GENERATION"
            if ready
            else "RETRIEVAL_AUDIT_INCOMPLETE"
        ),
        "scope": "train_only_development_treatment_qualification",
        "qualification_grain": "overall_RS_component_effect",
        "family_level_claim_authorized": False,
        "formal_rs_claim_authorized": False,
        "api_calls_made": 0,
        "runtime_states_path": str(args.runtime_states),
        "runtime_states_sha256": sha256_file(args.runtime_states),
        "candidate_cards_path": str(args.cards),
        "candidate_cards_sha256": sha256_file(args.cards),
        "candidate_card_count": len(cards),
        "active_card_count": len(active_cards),
        "active_provisional_source_supported_card_count": sum(
            bool(row["source_support"]["provisional_source_support_pass"])
            for row in active_cards
        ),
        "allow_weak_source_fail": args.allow_weak_source_fail,
        "unique_visible_states": len(seen_visible),
        "states_with_broad_safe_opportunity": len(opportunity_rows),
        "accepted_retrieval_states": sum(
            bool(row["accepted"]) for row in retrieval_rows
        ),
        "selected_pair_count": len(selected_rows),
        "selected_independent_user_groups": len(
            {str(row["user_id"]) for row in selected_rows}
        ),
        "planned_generation_calls": len(call_plan),
        "selected_family_counts": dict(sorted(family_counts.items())),
        "selected_provisional_source_support_counts": dict(
            sorted(
                Counter(
                    "pass"
                    if row["selected_provisional_source_support_pass"]
                    else "fail"
                    for row in selected_rows
                ).items()
            )
        ),
        "phatic_or_stop_negative_controls": len(phatic_controls),
        "ranker": args.ranker,
        "nli_model": str(args.nli_model) if args.ranker == "nli" else None,
        "minimum_score": args.minimum_score,
        "minimum_card_margin": args.minimum_margin,
        "states_per_family_target": args.states_per_family,
        "included_strategy_families": included_families,
        "excluded_prior_user_group_count": len(excluded_user_ids),
        "excluded_selected_state_paths": [
            str(path) for path in args.exclude_selected_states
        ],
        "minimum_required_per_family": minimum_by_family,
        "family_allocation_order": family_allocation_order,
        "generator_identity": generator_identity,
        "supporter_generation_treatment_sha256": generation.digest(),
        "checks": {
            "raw_source_response_absent": True,
            "test_validation_evoemo_outcome_unused": True,
            "phatic_and_routine_stop_have_zero_rs_opportunity": all(
                not row["eligible_families"] for row in phatic_controls
            ),
            "one_top1_card_per_rs_arm": all(
                row["selected_strategy_card_id"] is not None
                for row in call_plan
                if row["arm"] == "RS"
            ),
            "same_stack_except_strategy_section": parity_ok,
            "minimum_independent_dialogues_per_family": family_minimum_met,
            "global_user_group_uniqueness": len(selected_rows)
            == len({str(row["user_id"]) for row in selected_rows}),
            "no_prior_selected_user_group_overlap": not (
                {str(row["user_id"]) for row in selected_rows}
                & excluded_user_ids
            ),
        },
        "interpretation": (
            "This targeted plan estimates the end-to-end effect of the actual "
            "filter+retriever+Top-1 compact prompt treatment without a PM. It "
            "does not estimate an oracle-card effect or qualify omitted "
            "families. Cards with provisional source-support failure remain "
            "development hypotheses and cannot be promoted by retrieval "
            "balance alone."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "plan_report.json", report)
    write_jsonl(args.out_dir / "retrieval_audit.jsonl", retrieval_rows)
    write_jsonl(args.out_dir / "phatic_negative_controls.jsonl", phatic_controls)
    write_jsonl(args.out_dir / "selected_states.jsonl", selected_rows)
    write_jsonl(args.out_dir / "call_plan.jsonl", call_plan)
    print(report)


if __name__ == "__main__":
    main()
