"""Minimal, auditable RS clean-pair pilot for PM-v1.5.

The pilot deliberately avoids the legacy 16-action pseudo-oracle.  It selects
train-only states containing an explicit, model-visible interaction boundary,
then prepares one R0 and one RS generation request with the same generator,
seed, visible context, and output contract.  The RS arm adds exactly one
technique-only Strategy Bank V2 card; raw supporter examples never enter the
prompt.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import re
from typing import Any, Literal

from .contracts import RuntimeState
from .generation_contract import SupporterGenerationContract
from .io import canonical_json, iter_jsonl, sha256_file, sha256_text, stable_hex
from .prompts import common_context
from .text import estimate_tokens, lexical_score, normalize_space
from .v1_5_strategy_bank import StrategyBankV2Card


RS_PILOT_PROTOCOL = "pm-v1.5-minimum-rs-clean-pair-pilot-v1"
RS_RETRIEVER_PROTOCOL = "pm-v1.5-eligible-subset-lexical-rs-v1"
RS_PROMPT_PROTOCOL = "pm-v1.5-technique-only-rs-prompt-v1"
BoundaryCue = Literal["advice_welcome", "listen_only"]


_ADVICE_PATTERNS = (
    r"\bany tips\b",
    r"\bneed some guidance\b",
    r"\bplease offer one gentle,?\s+low-pressure idea\b",
    r"\bwould you help me think of a single,?\s+low-pressure next step\b",
    r"\bcould you offer one gentle suggestion\b",
    r"\bi think i could use one manageable idea\b",
    r"\bif you have one small,?\s+low-pressure idea\b",
    r"\bcould you give me one small suggestion\b",
)
_LISTEN_PATTERNS = (
    r"\bneed to feel heard before thinking about solutions\b",
    r"\brather be understood for a moment before (?:we )?talk about possible steps\b",
    r"\bhelp me put this into words first;\s*i do not want advice\b",
    r"\blisten while i talk through how this feels;\s*i do not want suggestions\b",
    r"\bnot looking for ideas right now;\s*i need a little room to talk it through\b",
)
_ADVICE_RE = re.compile("|".join(_ADVICE_PATTERNS), flags=re.IGNORECASE)
_LISTEN_RE = re.compile("|".join(_LISTEN_PATTERNS), flags=re.IGNORECASE)
_CONSTRAINED_ADVICE_RE = re.compile(
    "|".join(_ADVICE_PATTERNS[2:]),
    flags=re.IGNORECASE,
)

_ELIGIBLE_FAMILIES: dict[BoundaryCue, tuple[str, ...]] = {
    "advice_welcome": ("Providing Suggestions",),
    "listen_only": (
        "Restatement or Paraphrasing",
        "Reflection of feelings",
        "Affirmation and Reassurance",
    ),
}


def explicit_boundary_cue(text: str) -> BoundaryCue | None:
    """Return the preregistered explicit boundary visible in one user turn."""

    normalized = normalize_space(text)
    advice = bool(_ADVICE_RE.search(normalized))
    listen = bool(_LISTEN_RE.search(normalized))
    if advice and listen:
        raise ValueError("one state contains conflicting explicit boundary cues")
    if advice:
        return "advice_welcome"
    if listen:
        return "listen_only"
    return None


def explicit_risk_boundary(text: str) -> dict[str, Any]:
    """Return the observable boundary used by the risk denominator.

    A request for generic tips or guidance makes advice welcome, but does not
    itself impose a narrow interaction constraint.  Listen-only language and
    requests explicitly limited to one/small/low-pressure support do.
    """

    normalized = normalize_space(text)
    cue = explicit_boundary_cue(normalized)
    if cue == "listen_only":
        match = _LISTEN_RE.search(normalized)
        return {
            "applicable": True,
            "cue": cue,
            "evidence_quote": match.group(0) if match else normalized,
            "reason": "explicit_listen_first_or_no_advice_boundary",
        }
    if cue == "advice_welcome":
        match = _CONSTRAINED_ADVICE_RE.search(normalized)
        if match:
            return {
                "applicable": True,
                "cue": cue,
                "evidence_quote": match.group(0),
                "reason": "explicitly_limited_advice_or_step_boundary",
            }
        return {
            "applicable": False,
            "cue": cue,
            "evidence_quote": None,
            "reason": "generic_advice_request_without_narrow_boundary",
        }
    return {
        "applicable": False,
        "cue": None,
        "evidence_quote": None,
        "reason": "no_preregistered_explicit_boundary",
    }


def _load_runtime_states(path: str) -> list[RuntimeState]:
    return [RuntimeState.model_validate(row) for row in iter_jsonl(path)]


def _load_cards(path: str) -> list[StrategyBankV2Card]:
    cards = [StrategyBankV2Card.model_validate(row) for row in iter_jsonl(path)]
    families = [card.strategy_family for card in cards]
    if len(cards) != 5 or len(set(families)) != 5:
        raise ValueError("the minimum RS pilot requires exactly five technique families")
    if any(card.content_scope != "technique_only" for card in cards):
        raise ValueError("the minimum RS pilot forbids domain-information cards")
    return cards


def select_explicit_boundary_states(
    states: Sequence[RuntimeState],
) -> list[tuple[RuntimeState, BoundaryCue]]:
    """Select all train-only explicit-boundary states, outcome-blind."""

    selected: list[tuple[RuntimeState, BoundaryCue]] = []
    for state in states:
        if state.split != "train":
            continue
        cue = explicit_boundary_cue(state.current_user_text)
        if cue is not None:
            selected.append((state, cue))
    selected.sort(key=lambda row: (row[0].user_id, row[0].state_id))
    if len({state.state_id for state, _ in selected}) != len(selected):
        raise ValueError("selected RS pilot states repeat a state_id")
    cue_groups: dict[str, set[str]] = defaultdict(set)
    for state, cue in selected:
        cue_groups[cue].add(state.user_id)
    if any(len(cue_groups[cue]) < 8 for cue in _ELIGIBLE_FAMILIES):
        raise ValueError("each explicit boundary requires at least eight user groups")
    return selected


def select_technique_card(
    *,
    query: str,
    cue: BoundaryCue,
    cards: Sequence[StrategyBankV2Card],
) -> tuple[StrategyBankV2Card, dict[str, Any]]:
    """Filter by the explicit boundary, then rank the safe subset lexically."""

    eligible_families = set(_ELIGIBLE_FAMILIES[cue])
    eligible = [card for card in cards if card.strategy_family in eligible_families]
    if not eligible:
        raise ValueError(f"no eligible Strategy Bank V2 card for cue={cue}")
    scored = [
        (float(lexical_score(query, card.retrieval_text)), card)
        for card in eligible
    ]
    score, card = sorted(
        scored,
        key=lambda row: (row[0], row[1].card_id),
        reverse=True,
    )[0]
    return card, {
        "protocol": RS_RETRIEVER_PROTOCOL,
        "cue": cue,
        "eligible_families": sorted(eligible_families),
        "eligible_card_count": len(eligible),
        "selected_card_id": card.card_id,
        "selected_strategy_family": card.strategy_family,
        "selected_lexical_score": score,
        "tie_break": "lexical_score_then_card_id_desc",
        "uses_judge_reward_or_outcome": False,
    }


def _messages(
    *,
    state: RuntimeState,
    system_prompt: str,
    strategy_card: StrategyBankV2Card | None,
) -> list[dict[str, str]]:
    # These historical synthetic states contain a generated session summary.
    # Even when the text appears reconstructible from the visible dialogue, its
    # production lineage is not strong enough for the new MVP claim.  Omit it
    # in both arms so the pilot uses only verbatim visible turns.
    visible_state = state.model_copy(update={"current_session_summary": ""})
    sections = [common_context(visible_state)]
    if strategy_card is not None:
        sections.append(
            "Potential emotional-support technique. Use only when fitting:\n"
            f"- {strategy_card.prompt_guidance}"
        )
    sections.append("Write only the counselor's next response.")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


def build_minimum_rs_clean_pair_plan(
    *,
    runtime_states_path: str,
    strategy_cards_path: str,
    strategy_lineage_path: str,
    selected_seed_sources_path: str,
    generation_contract: SupporterGenerationContract,
    generator_identity: Mapping[str, Any],
    seed: int = 4311,
    human_bank_review_passed: bool = False,
) -> dict[str, Any]:
    """Prepare the exact zero-API plan and its scientific stop status."""

    states = _load_runtime_states(runtime_states_path)
    cards = _load_cards(strategy_cards_path)
    strategy_source_ids = {
        str(row["source_dialogue_id"])
        for row in iter_jsonl(strategy_lineage_path)
    }
    development_seed_ids = {
        str(row["dialogue_id"])
        for row in iter_jsonl(selected_seed_sources_path)
    }
    source_overlap = strategy_source_ids & development_seed_ids
    selected = select_explicit_boundary_states(states)
    call_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for state, cue in selected:
        query = normalize_space(
            "\n".join(
                [
                    *(turn.content for turn in state.current_session_history),
                    state.current_user_text,
                ]
            )
        )
        card, retrieval = select_technique_card(
            query=query,
            cue=cue,
            cards=cards,
        )
        pair_id = "rs_pair_" + stable_hex(
            RS_PILOT_PROTOCOL,
            state.state_id,
            generation_contract.digest(),
            seed,
            n=24,
        )
        selection_rows.append(
            {
                "pair_id": pair_id,
                "state_id": state.state_id,
                "card_id": state.card_id,
                "user_id": state.user_id,
                "boundary_cue": cue,
                "current_user_text": state.current_user_text,
                "selected_strategy_card_id": card.card_id,
                "selected_strategy_family": card.strategy_family,
                "retrieval": retrieval,
            }
        )
        for arm, action_id, strategy_card in (
            ("R0", "M0+R0", None),
            ("RS", "M0+RS", card),
        ):
            messages = _messages(
                state=state,
                system_prompt=generation_contract.system_prompt,
                strategy_card=strategy_card,
            )
            prompt_sha = sha256_text(canonical_json(messages))
            call_rows.append(
                {
                    "protocol": RS_PILOT_PROTOCOL,
                    "prompt_protocol": RS_PROMPT_PROTOCOL,
                    "pair_id": pair_id,
                    "state_id": state.state_id,
                    "card_id": state.card_id,
                    "user_id": state.user_id,
                    "boundary_cue": cue,
                    "arm": arm,
                    "action_id": action_id,
                    "selected_strategy_card_id": (
                        strategy_card.card_id if strategy_card is not None else None
                    ),
                    "selected_strategy_family": (
                        strategy_card.strategy_family
                        if strategy_card is not None
                        else None
                    ),
                    "messages": messages,
                    "prompt_sha256": prompt_sha,
                    "estimated_input_tokens": estimate_tokens(
                        canonical_json(messages)
                    ),
                    "generation": {
                        **generation_contract.payload(),
                        "seed": seed,
                    },
                    "generator_identity": dict(generator_identity),
                    "api_call_made": False,
                }
            )

    pair_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in call_rows:
        pair_rows[str(row["pair_id"])].append(row)
    parity_errors: list[str] = []
    for pair_id, rows in pair_rows.items():
        if len(rows) != 2 or {row["arm"] for row in rows} != {"R0", "RS"}:
            parity_errors.append(f"{pair_id}:arm_coverage")
            continue
        r0 = next(row for row in rows if row["arm"] == "R0")
        rs = next(row for row in rows if row["arm"] == "RS")
        if r0["generation"] != rs["generation"]:
            parity_errors.append(f"{pair_id}:generation_parameters")
        if r0["generator_identity"] != rs["generator_identity"]:
            parity_errors.append(f"{pair_id}:generator_identity")
        if r0["messages"][0] != rs["messages"][0]:
            parity_errors.append(f"{pair_id}:system_prompt")
        r0_user = str(r0["messages"][1]["content"])
        rs_user = str(rs["messages"][1]["content"])
        strategy_marker = "\n\nPotential emotional-support technique."
        if strategy_marker not in rs_user:
            parity_errors.append(f"{pair_id}:missing_rs_section")
        else:
            rs_without_strategy = (
                rs_user.split(strategy_marker, 1)[0]
                + "\n\nWrite only the counselor's next response."
            )
            if rs_without_strategy != r0_user:
                parity_errors.append(f"{pair_id}:visible_context_or_instruction")

    cue_counts = Counter(str(row["boundary_cue"]) for row in selection_rows)
    cue_group_counts = {
        cue: len(
            {
                str(row["user_id"])
                for row in selection_rows
                if row["boundary_cue"] == cue
            }
        )
        for cue in sorted(cue_counts)
    }
    family_counts = Counter(
        str(row["selected_strategy_family"]) for row in selection_rows
    )
    raw_examples_exposed = any(
        "Example style" in str(row["messages"][1]["content"])
        for row in call_rows
    )
    situation_exposed = any(
        bool(state.current_session_summary.strip())
        and state.provenance.get("dialogue_level_situation_exposed_to_pm") is True
        for state, _ in selected
    )
    scientific_checks = {
        "train_only": all(state.split == "train" for state, _ in selected),
        "minimum_eight_groups_per_cue": all(
            value >= 8 for value in cue_group_counts.values()
        ),
        "exact_two_arms_per_state": all(
            len(rows) == 2 for rows in pair_rows.values()
        ),
        "same_stack_except_strategy_section": not parity_errors,
        "raw_supporter_examples_absent": not raw_examples_exposed,
        "generated_session_summary_omitted": all(
            "Current-session summary:\n(none)"
            in str(row["messages"][1]["content"])
            for row in call_rows
        ),
        "dataset_situation_not_declared_exposed": not situation_exposed,
        "judge_reward_and_action_outcomes_unused": True,
        "strategy_sources_disjoint_from_development_seeds": not source_overlap,
    }
    data_ready = all(scientific_checks.values())
    generation_ready = data_ready and human_bank_review_passed
    report = {
        "protocol": RS_PILOT_PROTOCOL,
        "status": (
            "READY_FOR_TRAIN_ONLY_GENERATION"
            if generation_ready
            else (
                "BLOCKED_ONLY_ON_FIVE_CARD_HUMAN_REVIEW"
                if data_ready
                else "BLOCKED_BY_PLAN_INTEGRITY"
            )
        ),
        "scope": "train_only_development_pilot",
        "api_calls_made": 0,
        "runtime_states_path": runtime_states_path,
        "runtime_states_sha256": sha256_file(runtime_states_path),
        "strategy_cards_path": strategy_cards_path,
        "strategy_cards_sha256": sha256_file(strategy_cards_path),
        "strategy_lineage_path": strategy_lineage_path,
        "strategy_lineage_sha256": sha256_file(strategy_lineage_path),
        "selected_seed_sources_path": selected_seed_sources_path,
        "selected_seed_sources_sha256": sha256_file(selected_seed_sources_path),
        "strategy_development_seed_source_overlap_count": len(source_overlap),
        "supporter_generation_treatment_sha256": generation_contract.digest(),
        "generator_identity": dict(generator_identity),
        "seed": seed,
        "selected_states": len(selection_rows),
        "independent_user_groups": len(
            {str(row["user_id"]) for row in selection_rows}
        ),
        "planned_logical_calls": len(call_rows),
        "boundary_cue_counts": dict(sorted(cue_counts.items())),
        "boundary_cue_group_counts": cue_group_counts,
        "selected_strategy_family_counts": dict(sorted(family_counts.items())),
        "estimated_total_input_tokens": sum(
            int(row["estimated_input_tokens"]) for row in call_rows
        ),
        "maximum_output_tokens": (
            len(call_rows) * generation_contract.max_output_tokens
        ),
        "human_bank_review_passed": human_bank_review_passed,
        "scientific_checks": scientific_checks,
        "parity_errors": parity_errors,
        "claim_boundary": (
            "This is a controlled train-only treatment-uptake pilot. It cannot "
            "establish PM learnability, final quality, risk, cost, or external "
            "generalization until outcomes are generated, judged, and evaluated "
            "with user-group-held-out procedures."
        ),
    }
    report["plan_sha256"] = sha256_text(canonical_json(call_rows))
    report["report_sha256"] = sha256_text(canonical_json(report))
    return {
        "report": report,
        "selected_states": selection_rows,
        "call_plan": call_rows,
    }


def validate_minimum_rs_execution_inputs(
    *,
    report: Mapping[str, Any],
    call_rows: Sequence[Mapping[str, Any]],
    generation_contract: SupporterGenerationContract,
    generator_identity: Mapping[str, Any],
    human_review_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Fail closed before a generation client or API key is accessed."""

    errors: list[str] = []
    if report.get("protocol") != RS_PILOT_PROTOCOL:
        errors.append("unexpected_plan_protocol")
    report_core = dict(report)
    expected_report_sha = str(report_core.pop("report_sha256", ""))
    if expected_report_sha != sha256_text(canonical_json(report_core)):
        errors.append("invalid_report_sha256")
    if report.get("plan_sha256") != sha256_text(canonical_json(call_rows)):
        errors.append("call_plan_sha256_mismatch")
    if int(report.get("planned_logical_calls", -1)) != len(call_rows):
        errors.append("planned_logical_call_count_mismatch")
    logical_keys = [
        (str(row.get("pair_id") or ""), str(row.get("arm") or ""))
        for row in call_rows
    ]
    if (
        any(not pair_id or arm not in {"R0", "RS"} for pair_id, arm in logical_keys)
        or len(logical_keys) != len(set(logical_keys))
    ):
        errors.append("invalid_or_duplicate_pair_arm")
    if report.get("supporter_generation_treatment_sha256") != (
        generation_contract.digest()
    ):
        errors.append("supporter_generation_treatment_mismatch")
    if dict(report.get("generator_identity") or {}) != dict(generator_identity):
        errors.append("generator_identity_mismatch")
    if report.get("runtime_states_sha256") != sha256_file(
        str(report.get("runtime_states_path") or "")
    ):
        errors.append("runtime_states_file_mismatch")
    if report.get("strategy_cards_sha256") != sha256_file(
        str(report.get("strategy_cards_path") or "")
    ):
        errors.append("strategy_cards_file_mismatch")
    if report.get("strategy_lineage_sha256") != sha256_file(
        str(report.get("strategy_lineage_path") or "")
    ):
        errors.append("strategy_lineage_file_mismatch")
    if report.get("selected_seed_sources_sha256") != sha256_file(
        str(report.get("selected_seed_sources_path") or "")
    ):
        errors.append("selected_seed_sources_file_mismatch")

    human_ok = bool(
        human_review_binding
        and str(human_review_binding.get("status") or "").startswith(
            "HUMAN_REVIEW_PASS"
        )
        and int(human_review_binding.get("review_count", -1)) == 5
        and int(human_review_binding.get("approved_count", -1)) == 5
    )
    if report.get("human_bank_review_passed") is not human_ok:
        errors.append("human_review_status_or_binding_mismatch")
    ready = (
        not errors
        and human_ok
        and report.get("status") == "READY_FOR_TRAIN_ONLY_GENERATION"
    )
    return {
        "protocol": "pm-v1.5-minimum-rs-execution-preflight-v1",
        "status": (
            "READY_FOR_TRAIN_ONLY_GENERATION"
            if ready
            else (
                "BLOCKED_ONLY_ON_FIVE_CARD_HUMAN_REVIEW"
                if not errors and not human_ok
                else "BLOCKED_BY_EXECUTION_INPUT_INTEGRITY"
            )
        ),
        "ready": ready,
        "human_review_passed": human_ok,
        "logical_calls": len(call_rows),
        "errors": errors,
        "api_calls_made": 0,
        "one_shot_execution_required": False,
        "resume_policy": "reuse matching completed pair_id+arm; run only missing rows",
    }
