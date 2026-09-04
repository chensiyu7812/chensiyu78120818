from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import require_artifact_attestation
from .contracts import MemoryBackendRecord, StrategyCard
from .io import canonical_json, iter_jsonl, read_json, sha256_file, sha256_text
from .pm_v2_data import (
    GENERATION_CASE_FIELDS,
    GeneratedUserBundle,
    load_bundles,
    load_evaluator_context_index,
    load_states,
)
from .pm_v2_generation_pilot import (
    build_generation_compatibility_contract,
    require_generation_compatibility_attestation,
)
from .pm_v2_generation_review_v8 import REVIEW_QUESTIONS_EN
from .retrieval import StrategyRetriever, context_query
from .v1_5_automated_semantic_review import (
    build_control_manifest,
    build_judge_endpoint_descriptors,
)
from .v1_5_semantic_review_diagnostic import (
    single_field_diagnostic_messages,
)
from .config import endpoint_from_config, load_config


ACTUAL_CORPUS_REVIEW_PROTOCOL = "pm-v1.5-actual-468-structured-qa-v3"
ACTUAL_CORPUS_CONTROL_PROTOCOL = "pm-v1.5-actual-corpus-semantic-controls-v3"
ACTUAL_CORPUS_REVIEW_STAGE = "pm_v1_5_actual_corpus_semantic_review"

# The original formal audit asked two LLMs to score twelve dependent labels in
# one response and rejected a state whenever either LLM emitted one zero.  The
# V4 calibration run showed that this is not a calibrated measurement
# instrument.  V3 therefore separates facts that code owns from the two atomic
# semantic claims that genuinely require language understanding.  The other
# construction labels remain visible, hashed metadata; their causal value is
# measured later by the blinded action sweep rather than asserted by an LLM.
ACTUAL_DETERMINISTIC_FIELDS = (
    "source_type_match",
    "dialogue_temporal_order_match",
    "memory_age_design_match",
    "surface_naturalness_match",
)
ACTUAL_SEMANTIC_FIELDS = (
    "context_grounding_match",
    "advice_readiness_match",
)
ACTUAL_DERIVED_OR_CONSTRUCTION_FIELDS = (
    "semantic_family_match",
    "regime_match",
    "memory_sources_marginal_value_match",
    "memory_item_utility_match",
    "strategy_resource_need_match",
    "strategy_item_utility_match",
)
ACTUAL_PANEL_POLICY = "unanimous-reject-disagreement-retained-v1"
ACTUAL_CITATION_POLICY = "report-only-not-outcome-v1"

ADVICE_READINESS_OPERATIONAL_DEFINITIONS = {
    "listen_only": (
        "The user explicitly asks to be heard, understood, or allowed to talk, "
        "and declines suggestions or decisions for this turn."
    ),
    "explore_first": (
        "The user wants help naming, understanding, or untangling the situation "
        "before choosing an action; the exact words 'ready' or 'explore' are not required."
    ),
    "light_suggestion": (
        "The user asks for or welcomes one gentle, low-pressure idea or next step, "
        "without requesting a detailed plan."
    ),
    "structured_plan": (
        "The user explicitly asks for concrete ordered steps, a detailed plan, or "
        "help implementing a chosen course of action."
    ),
    "ambiguous": (
        "The visible dialogue provides no reliable signal about the amount of "
        "directiveness the user wants in this turn."
    ),
}

ACTUAL_REVIEW_QUESTIONS_EN = {
    **REVIEW_QUESTIONS_EN,
    "memory_age_design_match": (
        "Are each displayed current-session index, created-session index, and "
        "derived age arithmetically consistent, causal, and free of label leakage?"
    ),
    "strategy_item_utility_match": (
        "Is every actually retrieved Strategy card individually appropriate for "
        "this dialogue rather than clearly irrelevant, overly directive, or harmful?"
    ),
    "advice_readiness_match": (
        "Is the displayed candidate directiveness level (listen-only, explore-first, "
        "light suggestion, structured planning, or ambiguous) supported by the dialogue?"
    ),
}


def _surface_fallback_report(
    *,
    bundles_path: str | Path,
    split_by_user: Mapping[str, str],
    maximum_fallback_rate_by_split: Mapping[str, float],
) -> dict[str, Any]:
    counts = {
        split: {"cases": 0, "fallback_cases": 0}
        for split in ("train", "calibration", "internal_test")
    }
    for bundle in load_bundles(bundles_path):
        split = str(split_by_user[bundle.user_id])
        selection = bundle.provenance.get("surface_selection") or {}
        fallback_cases = list(selection.get("fallback_cases") or [])
        counts[split]["cases"] += len(bundle.cases)
        counts[split]["fallback_cases"] += len(fallback_cases)
    checks = {}
    for split, row in counts.items():
        rate = row["fallback_cases"] / max(row["cases"], 1)
        row["fallback_rate"] = rate
        row["provider_surface_rate"] = 1.0 - rate
        row["maximum_fallback_rate"] = float(maximum_fallback_rate_by_split[split])
        checks[split] = rate <= float(maximum_fallback_rate_by_split[split])
    return {
        "protocol": "pm-v1.5-provider-surface-fallback-gate-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "splits": counts,
        "checks": checks,
    }


def _advice_readiness_candidate(context: Mapping[str, Any]) -> str:
    return str(context.get("advice_readiness_target") or "ambiguous")


def _strategy_resource_candidate(context: Mapping[str, Any]) -> str:
    value = str(context.get("strategy_resource_target") or "ambiguous")
    return "uncertain" if value == "ambiguous" else value


def build_generation_pilot_review_items(
    *,
    pilot_attestation_path: str | Path,
    experiment_config_path: str | Path,
    pm_v1_5_config_path: str | Path,
    strategy_bank_path: str | Path,
    strategy_top_k: int,
    strategy_min_score: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Render the exact paid nine-case pilot with the current 12-field rubric.

    The legacy two-reviewer packet combines Strategy-resource value with advice
    readiness and therefore cannot audit the counterbalanced V1.5 factorial.
    This adapter reuses the same payload/rubric as the later 468-state audit and
    binds every row to the paid pilot attestation and bundle hashes.
    """

    attestation_path = Path(pilot_attestation_path).resolve()
    attestation = read_json(attestation_path)
    contract = (attestation.get("parameters") or {}).get(
        "compatibility_contract"
    )
    if not isinstance(contract, dict):
        raise RuntimeError("pilot semantic review lacks compatibility contract")
    experiment_config = load_config(experiment_config_path)
    pm_config = load_config(pm_v1_5_config_path)
    generation = pm_config.get("data_generation")
    api_cost = pm_config.get("api_cost_planning")
    if not isinstance(generation, dict) or not isinstance(api_cost, dict):
        raise RuntimeError("pilot semantic review lacks generation config")
    endpoint = endpoint_from_config(
        experiment_config, str(generation.get("generator_endpoint") or "")
    )
    pricing = generation.get("pricing_usd_per_mtok")
    if not isinstance(pricing, dict):
        raise RuntimeError("pilot semantic review lacks generation pricing")
    current_contract = build_generation_compatibility_contract(
        project_root=Path(__file__).resolve().parents[2],
        experiment_config_path=experiment_config_path,
        pm_v2_config_path=pm_v1_5_config_path,
        seed_dialogues_path=str(contract.get("seed_dialogues_path") or ""),
        endpoint=endpoint,
        base_generation_seed=int(generation["base_seed"]),
        full_user_count=sum(
            int(generation[key])
            for key in ("train_users", "calibration_users", "internal_test_users")
        ),
        input_token_safety_factor=float(api_cost["input_token_safety_factor"]),
        fail_on_reported_input_overrun=bool(
            api_cost["fail_on_reported_input_overrun"]
        ),
        input_usd_per_mtok=float(pricing["input"]),
        output_usd_per_mtok=float(pricing["output"]),
    )
    if current_contract != contract:
        raise RuntimeError(
            "paid generation pilot differs from the current scoped generation "
            "config/prompt/schema/code contract"
        )
    verification = require_generation_compatibility_attestation(
        attestation_path, expected_contract=current_contract
    )
    bundle_path = attestation_path.parent / "pilot_bundle.json"
    bundle = GeneratedUserBundle.model_validate(read_json(bundle_path))
    if len(bundle.cases) != len(GENERATION_CASE_FIELDS):
        raise RuntimeError("pilot semantic review requires the exact nine cases")

    cards = [
        StrategyCard.model_validate(row) for row in iter_jsonl(strategy_bank_path)
    ]
    if not cards:
        raise RuntimeError("pilot semantic review requires the frozen Strategy Bank")
    strategy = StrategyRetriever(
        cards, top_k=int(strategy_top_k), minimum_score=float(strategy_min_score)
    )
    items: list[dict[str, Any]] = []
    for (case_field, expected_regime), case in zip(
        GENERATION_CASE_FIELDS, bundle.cases, strict=True
    ):
        if case.regime is not expected_regime:
            raise RuntimeError("pilot semantic review case order/regime drift")
        query = context_query(
            case.current_user_text,
            [turn.model_dump(mode="json") for turn in case.recent_dialogue],
            case.session_summary,
        )
        retrieved_cards = strategy.retrieve(query)
        memory_evidence = [
            {
                "memory_id": memory.memory_id,
                "source": memory.source.value,
                "utility": memory.item_utility,
                "created_session": memory.created_session,
                "age": case.session_index - memory.created_session,
                "stale": memory.stale,
                "conflict": memory.conflicts_with_current_state,
                "text": memory.text,
            }
            for memory in (
                *case.profile_memories,
                *case.summary_memories,
                *case.event_memories,
            )
        ]
        payload = {
            "semantic_family": case.semantic_family,
            "regime": case.regime.value,
            "needed_memory_sources": [
                source.value for source in case.materially_useful_memory_sources
            ],
            "advice_readiness": case.advice_readiness_target,
            "split": "compatibility_pilot",
            "session_index": case.session_index,
            "history": [
                {"role": turn.role, "content": turn.content}
                for turn in case.recent_dialogue
            ],
            "current_user_text": case.current_user_text,
            "session_summary": case.session_summary,
            "authorized_user_context": case.authorized_user_context,
            "coverage_rationale": case.coverage_rationale,
            "memory_evidence": memory_evidence,
            "strategy_resource_candidate": _strategy_resource_candidate(
                {"strategy_resource_target": case.strategy_resource_target}
            ),
            "strategy_cards": [
                {
                    "strategy_id": card.strategy_id,
                    "strategy_label": card.strategy_label,
                    "guidance_text": card.guidance_text,
                    "example_response": card.example_response,
                    "source_dialogue_id": card.source_dialogue_id,
                }
                for card in retrieved_cards
            ],
        }
        items.append(
            {
                "kind": "real",
                "real_source": "paid_compatibility_pilot",
                "item_id": f"paid_pilot_{case_field}",
                "split": "compatibility_pilot",
                "regime": case.regime.value,
                "payload": payload,
                "text": _render_actual_payload(payload),
            }
        )
    return items, {
        "status": "PASS",
        "pilot_attestation_path": str(attestation_path),
        "pilot_attestation_sha256": sha256_file(attestation_path),
        "pilot_bundle_path": str(bundle_path),
        "pilot_bundle_sha256": sha256_file(bundle_path),
        "pilot_contract_sha256": str(contract["contract_sha256"]),
        "generation_config_projection_sha256": str(
            contract["generation_config_projection_sha256"]
        ),
        "pilot_verification_attestation_sha256": verification[
            "attestation_sha256"
        ],
        "item_count": len(items),
    }


def _render_actual_payload(payload: Mapping[str, Any]) -> str:
    lines = [
        f"Candidate semantic family: {payload['semantic_family']}",
        f"Candidate regime: {payload['regime']}",
        "Candidate materially-useful memory sources: "
        + repr(payload["needed_memory_sources"]),
        f"Candidate advice readiness: {payload['advice_readiness']}",
        f"Data split: {payload['split']}",
        f"Current session index: {payload['session_index']}",
        "",
        "Dialogue before current turn:",
    ]
    for index, turn in enumerate(payload["history"]):
        lines.append(f"  {index}: {turn['role']}: {turn['content']}")
    lines.extend(
        [
            f"Current user message: {payload['current_user_text']}",
            f"Session summary: {payload['session_summary']}",
            f"Authorized user context: {payload['authorized_user_context']}",
            f"Coverage rationale: {payload['coverage_rationale']}",
            "",
            "Memory evidence:",
        ]
    )
    for index, memory in enumerate(payload["memory_evidence"]):
        lines.append(
            "  - row={row}, source={source}, utility={utility}, "
            "created_session={created}, current_session={current}, age={age}, "
            "stale={stale}, conflict={conflict}: {text}".format(
                row=index,
                source=memory["source"],
                utility=memory["utility"],
                created=memory["created_session"],
                current=payload["session_index"],
                age=memory["age"],
                stale=memory["stale"],
                conflict=memory["conflict"],
                text=memory["text"],
            )
        )
    lines.extend(
        [
            "",
            "Strategy resource candidate: "
            + str(payload["strategy_resource_candidate"]),
            "Retrieved Strategy cards (each is a candidate for actual use):",
        ]
    )
    for card in payload["strategy_cards"]:
        lines.append(
            f"  - {card['strategy_label']}: {card['guidance_text']} | "
            f"example={card['example_response']} | source={card['source_dialogue_id']}"
        )
    return "\n".join(lines)


_EXPLICIT_GENERATION_MARKERS = (
    "as the generated user",
    "resource-condition example",
    "data-generation example",
    "regime label",
    "utility label",
)


def _normalized_surface_text(value: Any) -> str:
    return " ".join(str(value).split()).casefold()


def evaluate_actual_deterministic_payload(
    *, item_id: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate the four formal facts whose truth is owned by code/data lineage."""

    memories = list(payload.get("memory_evidence") or [])
    memory_ids = [str(row.get("memory_id") or "") for row in memories]
    valid_sources = all(row.get("source") in {"MP", "MS", "ME"} for row in memories)
    source_type_match = (
        valid_sources
        and all(memory_ids)
        and len(memory_ids) == len(set(memory_ids))
    )

    history = list(payload.get("history") or [])
    roles_alternate = all(
        isinstance(row, Mapping)
        and row.get("role") == ("user" if index % 2 == 0 else "assistant")
        for index, row in enumerate(history)
    )
    history_ends_assistant = bool(history) and history[-1].get("role") == "assistant"
    current = _normalized_surface_text(payload.get("current_user_text") or "")
    current_not_copied = bool(current) and all(
        _normalized_surface_text(row.get("content") or "") != current
        for row in history
        if isinstance(row, Mapping)
    )
    dialogue_temporal_order_match = (
        roles_alternate and history_ends_assistant and current_not_copied
    )

    session_index = int(payload.get("session_index") or -1)
    memory_age_match = all(
        int(row.get("created_session", -1)) <= session_index
        and int(row.get("age", -1))
        == session_index - int(row.get("created_session", -1))
        for row in memories
    )

    visible_surface = " ".join(
        _normalized_surface_text(value)
        for value in (
            payload.get("current_user_text") or "",
            payload.get("history") or [],
            payload.get("session_summary") or "",
            payload.get("authorized_user_context") or "",
        )
    )
    observed_markers = [
        marker for marker in _EXPLICIT_GENERATION_MARKERS if marker in visible_surface
    ]
    surface_naturalness_match = not observed_markers

    fields = {
        "source_type_match": source_type_match,
        "dialogue_temporal_order_match": dialogue_temporal_order_match,
        "memory_age_design_match": memory_age_match,
        "surface_naturalness_match": surface_naturalness_match,
    }
    return {
        "item_id": str(item_id),
        "status": "PASS" if all(fields.values()) else "FAIL",
        "fields": fields,
        "details": {
            "memory_count": len(memories),
            "memory_ids_unique_and_nonempty": bool(memory_ids)
            and all(memory_ids)
            and len(memory_ids) == len(set(memory_ids)),
            "registered_sources_valid": valid_sources,
            "history_roles_alternate_from_user": roles_alternate,
            "history_ends_with_assistant": history_ends_assistant,
            "current_text_absent_from_history": current_not_copied,
            "observed_generation_markers": observed_markers,
        },
    }


def _actual_semantic_claim_and_evidence(
    payload: Mapping[str, Any], *, field: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    visible_evidence = {
        "history": list(payload["history"]),
        "current_user_text": str(payload["current_user_text"]),
        "session_summary": str(payload["session_summary"]),
    }
    if field == "context_grounding_match":
        summary_text = str(payload["session_summary"])
        context_text = str(payload["authorized_user_context"])
        # Root cause of a real incident: the old claim used a self-
        # referential meta-description ("both candidate context fields are
        # supported") instead of stating the verification target directly.
        # One judge family systematically misread "candidate context fields"
        # as literal field names that must appear in the evidence text,
        # rejecting well-grounded contexts on that basis alone (verified
        # against 65 real disagreements showing this exact misreading; one
        # further disagreement showed a related but distinct empty-summary-
        # as-missing-evidence misreading, also fixed by the N/A handling
        # below -- see data/pm_v1_5_contracts/context_grounding_defect_
        # classification_v1.jsonl for the full, frozen per-state audit).
        # The claim below states plainly which TEXT is being
        # verified and instructs the judge to evaluate facts, not field
        # names. A structurally absent (empty) session_summary is marked
        # N/A rather than posed as something to "support" -- an empty field
        # is not missing evidence and must never be scored as unsupported.
        if summary_text.strip():
            claim = {
                "verification_instruction": (
                    "session_summary_text and authorized_user_context_text "
                    "below are each a claim about the user that must be "
                    "fully entailed by the evidence. Judge the stated facts "
                    "in the text, never the names of any fields."
                ),
                "session_summary_text": summary_text,
                "authorized_user_context_text": context_text,
            }
        else:
            claim = {
                "verification_instruction": (
                    "session_summary_text is structurally absent (N/A) for "
                    "this case -- it requires no verification and its "
                    "absence must never be treated as missing or "
                    "unsupported evidence. Only authorized_user_context_text "
                    "below must be fully entailed by the evidence. Judge the "
                    "stated facts in the text, never the names of any "
                    "fields."
                ),
                "session_summary_text": "N/A",
                "authorized_user_context_text": context_text,
            }
        return (
            claim,
            {
                "history": visible_evidence["history"],
                "current_user_text": visible_evidence["current_user_text"],
            },
        )
    if field == "advice_readiness_match":
        label = str(payload["advice_readiness"])
        if label not in ADVICE_READINESS_OPERATIONAL_DEFINITIONS:
            raise RuntimeError(f"unknown actual advice-readiness label: {label}")
        return (
            {
                "proposed_advice_readiness": label,
                "operational_definition": ADVICE_READINESS_OPERATIONAL_DEFINITIONS[
                    label
                ],
            },
            visible_evidence,
        )
    raise RuntimeError(f"unsupported actual semantic field: {field}")


def build_actual_semantic_packet(
    *, item_id: str, payload: Mapping[str, Any], field: str, expected_verdict: str
) -> dict[str, Any]:
    if field not in ACTUAL_SEMANTIC_FIELDS:
        raise RuntimeError(f"actual semantic packet field is not frozen: {field}")
    if expected_verdict not in {"supported", "not_supported"}:
        raise RuntimeError("actual semantic packet has invalid expected verdict")
    candidate_claim, allowed_evidence = _actual_semantic_claim_and_evidence(
        payload, field=field
    )
    messages = single_field_diagnostic_messages(
        field=field,
        candidate_claim=candidate_claim,
        allowed_evidence=allowed_evidence,
    )
    packet_id = f"{item_id}::{field}"
    packet = {
        "kind": "real" if expected_verdict == "supported" else "control",
        "item_id": packet_id,
        "case_item_id": str(item_id),
        "field": field,
        "candidate_claim": candidate_claim,
        "allowed_evidence": allowed_evidence,
        "minimum_evidence_citations": (
            2 if field == "context_grounding_match" else 1
        ),
        # Frozen outside the prompt. It is used only to score negative controls
        # and citation telemetry; serializing messages is checked for leakage.
        "expected_verdict": expected_verdict,
        "messages": messages,
    }
    packet["packet_sha256"] = sha256_text(
        canonical_json(
            {
                key: value
                for key, value in packet.items()
                if key not in {"expected_verdict", "packet_sha256"}
            }
        )
    )
    return packet


def _different(values: Sequence[Any], current: Any) -> Any:
    for value in values:
        if value != current:
            return copy.deepcopy(value)
    raise RuntimeError("actual control construction requires a different donor value")


def _actual_control_eligible(item: Mapping[str, Any], field: str) -> bool:
    payload = item["payload"]
    if field in {
        "memory_item_utility_match",
        "source_type_match",
        "memory_age_design_match",
    }:
        return bool(payload["memory_evidence"])
    if field == "dialogue_temporal_order_match":
        return any(turn["role"] == "user" for turn in payload["history"])
    if field == "strategy_resource_need_match":
        return payload["strategy_resource_candidate"] in {"use", "skip"}
    if field == "strategy_item_utility_match":
        return bool(payload["strategy_cards"]) and payload["advice_readiness"] in {
            "listen_only",
            "explore_first",
        }
    if field == "advice_readiness_match":
        return payload["advice_readiness"] != "ambiguous"
    return True


def _corrupt_actual_payload(
    item: Mapping[str, Any],
    *,
    field: str,
    donors: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = copy.deepcopy(item["payload"])
    if field == "semantic_family_match":
        value = _different(
            [row["payload"]["semantic_family"] for row in donors],
            payload["semantic_family"],
        )
        payload["semantic_family"] = value
        override = {"semantic_family": value}
    elif field == "regime_match":
        opposites = {
            "context_only": "profile_needed",
            "profile_needed": "summary_needed",
            "summary_needed": "event_needed",
            "event_needed": "profile_needed",
            "multi_source_needed": "context_only",
            "memory_harmful": "profile_needed",
            "strategy_helpful": "strategy_harmful",
            "strategy_harmful": "strategy_helpful",
            "ambiguous": "event_needed",
        }
        value = opposites[str(payload["regime"])]
        payload["regime"] = value
        override = {"regime": value}
    elif field == "memory_sources_marginal_value_match":
        value = [] if payload["needed_memory_sources"] else ["MP"]
        payload["needed_memory_sources"] = value
        override = {"needed_memory_sources": value}
    elif field == "memory_item_utility_match":
        memory = payload["memory_evidence"][0]
        memory["utility"] = (
            "irrelevant" if memory["utility"] == "helpful" else "helpful"
        )
        override = {"memory_id": memory["memory_id"], "utility": memory["utility"]}
    elif field == "source_type_match":
        memory = payload["memory_evidence"][0]
        memory["source"] = {"MP": "MS", "MS": "ME", "ME": "MP"}[
            memory["source"]
        ]
        override = {"memory_id": memory["memory_id"], "source": memory["source"]}
    elif field == "dialogue_temporal_order_match":
        turn = next(row for row in payload["history"] if row["role"] == "user")
        turn["content"] = payload["current_user_text"]
        override = {"prior_user_turn": "copied_current_user_message"}
    elif field == "context_grounding_match":
        # The donor's session_summary must be non-empty: a structurally
        # absent (empty) summary is N/A, not evidence to be judged
        # "supported"/"not_supported" against (see
        # _actual_semantic_claim_and_evidence's N/A handling) -- a donor
        # summary that happens to be empty is not a real negative control,
        # since the fixed audit contract correctly never treats an empty
        # summary as unsupported evidence.
        value = next(
            row["payload"]["session_summary"]
            for row in donors
            if row["payload"]["semantic_family"] != payload["semantic_family"]
            and row["payload"]["session_summary"] != payload["session_summary"]
            and row["payload"]["session_summary"].strip()
        )
        payload["session_summary"] = value
        override = {"session_summary": value}
    elif field == "memory_age_design_match":
        memory = payload["memory_evidence"][0]
        memory["age"] = int(memory["age"]) + 1
        override = {"memory_id": memory["memory_id"], "age": memory["age"]}
    elif field == "strategy_resource_need_match":
        value = "skip" if payload["strategy_resource_candidate"] == "use" else "use"
        payload["strategy_resource_candidate"] = value
        override = {"strategy_resource_candidate": value}
    elif field == "strategy_item_utility_match":
        card = payload["strategy_cards"][0]
        card.update(
            {
                "strategy_label": "Immediate directive planning",
                "guidance_text": (
                    "Move straight into a rigid detailed plan even when the user "
                    "is not ready for advice or asks only to be heard."
                ),
                "example_response": (
                    "Here is the plan you must follow now; we do not need to spend "
                    "time on how this feels."
                ),
            }
        )
        override = {"strategy_id": card["strategy_id"], "card": "premature_plan"}
    elif field == "advice_readiness_match":
        current = payload["advice_readiness"]
        value = {
            "listen_only": "structured_plan",
            "explore_first": "structured_plan",
            "light_suggestion": "listen_only",
            "structured_plan": "listen_only",
        }[current]
        payload["advice_readiness"] = value
        override = {"advice_readiness": value}
    elif field == "surface_naturalness_match":
        payload["current_user_text"] = (
            "As the generated user for this resource-condition example, "
            + payload["current_user_text"]
        )
        override = {"current_user_text": payload["current_user_text"]}
    else:
        raise RuntimeError(f"unsupported actual control field: {field}")
    return payload, override


def build_actual_corpus_controls(
    items: Sequence[Mapping[str, Any]],
    *,
    control_seed: int,
    required_control_fields: Sequence[str],
    controls_per_field: int,
) -> list[dict[str, Any]]:
    if list(required_control_fields) != list(ACTUAL_SEMANTIC_FIELDS):
        raise RuntimeError(
            "actual controls must cover exactly the two frozen semantic fields"
        )
    if int(controls_per_field) != 2:
        raise RuntimeError("actual controls_per_field must equal 2")
    controls: list[dict[str, Any]] = []
    for field_index, field in enumerate(required_control_fields):
        candidates = [row for row in items if _actual_control_eligible(row, field)]
        if len(candidates) < int(controls_per_field):
            raise RuntimeError(f"insufficient eligible actual cases for {field}")
        random.Random(int(control_seed) + field_index * 1009).shuffle(candidates)
        for replica, item in enumerate(candidates[: int(controls_per_field)], start=1):
            corrupted, override = _corrupt_actual_payload(
                item, field=field, donors=items
            )
            control_id = f"{item['item_id']}__control_{field}_{replica}"
            packet = build_actual_semantic_packet(
                item_id=control_id,
                payload=corrupted,
                field=field,
                expected_verdict="not_supported",
            )
            controls.append(
                {
                    "item_id": packet["item_id"],
                    "case_item_id": item["item_id"],
                    "corrupted_field": field,
                    "rating_field": field,
                    "override": override,
                    "case_text": canonical_json(
                        {
                            "candidate_claim": packet["candidate_claim"],
                            "allowed_evidence": packet["allowed_evidence"],
                        }
                    ),
                    "semantic_packet": packet,
                }
            )
    return controls


def build_actual_corpus_review_items(
    *,
    states_path: str | Path,
    evaluator_contexts_path: str | Path,
    backend_path: str | Path,
    bundles_path: str | Path,
    strategy_bank_path: str | Path,
    strategy_top_k: int,
    strategy_min_score: float,
    maximum_fallback_rate_by_split: Mapping[str, float],
    control_seed: int,
    required_control_fields: Sequence[str],
    controls_per_field: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    states = load_states(states_path)
    if len(states) != 468:
        raise RuntimeError("actual-corpus semantic review requires all 468 states")
    evaluator = load_evaluator_context_index(
        evaluator_contexts_path, states=states, require_exact=True
    )
    backend_by_card = {}
    for row in iter_jsonl(backend_path):
        backend = MemoryBackendRecord.model_validate(row)
        if backend.card_id in backend_by_card:
            raise RuntimeError("actual-corpus backend repeats card_id")
        backend_by_card[backend.card_id] = backend
    cards = [StrategyCard.model_validate(row) for row in iter_jsonl(strategy_bank_path)]
    if not cards:
        raise RuntimeError("actual-corpus review requires the frozen Strategy Bank")
    strategy = StrategyRetriever(
        cards, top_k=int(strategy_top_k), minimum_score=float(strategy_min_score)
    )
    split_by_user = {state.user_id: state.split.value for state in states}
    fallback = _surface_fallback_report(
        bundles_path=bundles_path,
        split_by_user=split_by_user,
        maximum_fallback_rate_by_split=maximum_fallback_rate_by_split,
    )
    if fallback["status"] != "PASS":
        raise RuntimeError("actual corpus failed the provider-surface fallback gate")
    items = []
    deterministic_audits: list[dict[str, Any]] = []
    for state in sorted(states, key=lambda value: value.state_id):
        context = evaluator.by_state[state.state_id]
        backend = backend_by_card.get(state.card_id)
        if backend is None:
            raise RuntimeError(f"actual-corpus backend missing {state.card_id}")
        annotations = {
            str(row["memory_id"]): row for row in context["memory_annotations"]
        }
        query = context_query(
            state.current_user_text,
            [turn.model_dump(mode="json") for turn in state.current_session_history],
            state.current_session_summary,
        )
        retrieved_cards = strategy.retrieve(query)
        memory_evidence = []
        for memory in backend.items:
            annotation = annotations[memory.memory_id]
            memory_evidence.append(
                {
                    "memory_id": memory.memory_id,
                    "source": memory.source.value,
                    "utility": annotation["item_utility"],
                    "created_session": memory.created_session,
                    "age": state.session_index - memory.created_session,
                    "stale": annotation["stale"],
                    "conflict": annotation["conflicts_with_current_state"],
                    "text": memory.text,
                }
            )
        regime = str(context["regime"])
        payload = {
            "semantic_family": state.semantic_family,
            "regime": regime,
            "needed_memory_sources": list(context["needed_memory_sources"]),
            "advice_readiness": _advice_readiness_candidate(context),
            "split": state.split.value,
            "session_index": state.session_index,
            "history": [
                {"role": turn.role, "content": turn.content}
                for turn in state.current_session_history
            ],
            "current_user_text": state.current_user_text,
            "session_summary": state.current_session_summary,
            "authorized_user_context": context["authorized_user_context"],
            "coverage_rationale": context["coverage_rationale"],
            "memory_evidence": memory_evidence,
            "strategy_resource_candidate": _strategy_resource_candidate(context),
            "strategy_cards": [
                {
                    "strategy_id": card.strategy_id,
                    "strategy_label": card.strategy_label,
                    "guidance_text": card.guidance_text,
                    "example_response": card.example_response,
                    "source_dialogue_id": card.source_dialogue_id,
                }
                for card in retrieved_cards
            ],
        }
        deterministic_audit = evaluate_actual_deterministic_payload(
            item_id=state.state_id, payload=payload
        )
        deterministic_audits.append(deterministic_audit)
        items.append(
            {
                "kind": "real",
                "item_id": state.state_id,
                "split": state.split.value,
                "regime": regime,
                "payload": payload,
                "text": _render_actual_payload(payload),
                "deterministic_audit": deterministic_audit,
                "semantic_packets": [
                    build_actual_semantic_packet(
                        item_id=state.state_id,
                        payload=payload,
                        field=field,
                        expected_verdict="supported",
                    )
                    for field in ACTUAL_SEMANTIC_FIELDS
                ],
            }
        )
    controls = build_actual_corpus_controls(
        items,
        control_seed=control_seed,
        required_control_fields=required_control_fields,
        controls_per_field=controls_per_field,
    )
    control_manifest = build_control_manifest(controls)
    deterministic_failures = [
        row for row in deterministic_audits if row["status"] != "PASS"
    ]
    report = {
        "protocol": ACTUAL_CORPUS_REVIEW_PROTOCOL,
        "control_protocol": ACTUAL_CORPUS_CONTROL_PROTOCOL,
        "deterministic_fields": list(ACTUAL_DETERMINISTIC_FIELDS),
        "semantic_fields": list(ACTUAL_SEMANTIC_FIELDS),
        "derived_or_construction_fields": list(
            ACTUAL_DERIVED_OR_CONSTRUCTION_FIELDS
        ),
        "panel_policy": ACTUAL_PANEL_POLICY,
        "citation_policy": ACTUAL_CITATION_POLICY,
        "required_control_fields": list(required_control_fields),
        "controls_per_field": int(controls_per_field),
        "n_controls": len(controls),
        "control_matrix_sha256": sha256_text(canonical_json(control_manifest)),
        "states": len(items),
        "semantic_packets": len(items) * len(ACTUAL_SEMANTIC_FIELDS),
        "deterministic_gate": {
            "status": "PASS" if not deterministic_failures else "FAIL",
            "n_states": len(deterministic_audits),
            "n_failures": len(deterministic_failures),
            "field_pass_counts": {
                field: sum(bool(row["fields"][field]) for row in deterministic_audits)
                for field in ACTUAL_DETERMINISTIC_FIELDS
            },
            "failures": deterministic_failures,
        },
        "construction_metadata_policy": (
            "hashed-and-schema-validated-not-claimed-as-independent-semantic-gold; "
            "resource utility is measured by the blinded action sweep"
        ),
        "states_by_split": {
            split: sum(item["split"] == split for item in items)
            for split in ("train", "calibration", "internal_test")
        },
        "fallback_gate": fallback,
        "input_hashes": {
            "states": sha256_file(states_path),
            "evaluator_contexts": sha256_file(evaluator_contexts_path),
            "backend": sha256_file(backend_path),
            "bundles": sha256_file(bundles_path),
            "strategy_bank": sha256_file(strategy_bank_path),
        },
    }
    for item in items:
        item.pop("payload", None)
    return items, controls, report


def aggregate_actual_corpus_gate(
    *,
    state_items: Sequence[Mapping[str, Any]],
    real_case_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    control_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    controls: Sequence[Mapping[str, Any]],
    judge_family_names: Sequence[str],
    corpus_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate the formal two-judge panel without treating disagreement as gold.

    Real items are rejected only when both independent families return
    ``not_supported``. A split verdict is retained and disclosed. Negative
    controls are fatal only when both families incorrectly return ``supported``.
    Citation integrity is counted separately and never changes the outcome.
    """

    families = list(judge_family_names)
    if len(families) != 2 or len(set(families)) != 2:
        raise RuntimeError("actual corpus panel requires exactly two judge families")
    state_ids = [str(row["item_id"]) for row in state_items]
    if len(state_ids) != 468 or len(set(state_ids)) != 468:
        raise RuntimeError("actual corpus aggregation requires 468 unique states")
    real_packets = {
        str(packet["item_id"]): packet
        for state in state_items
        for packet in state.get("semantic_packets") or []
    }
    control_packets = {
        str(row["semantic_packet"]["item_id"]): row["semantic_packet"]
        for row in controls
    }
    expected_real_packets = 468 * len(ACTUAL_SEMANTIC_FIELDS)
    contract_errors: list[str] = []
    if len(real_packets) != expected_real_packets:
        contract_errors.append(
            f"expected {expected_real_packets} unique real packets, found {len(real_packets)}"
        )
    if set(real_case_results) != set(real_packets):
        contract_errors.append("real judgment packet IDs differ from frozen packets")
    if set(control_results) != set(control_packets):
        contract_errors.append("control judgment packet IDs differ from frozen controls")
    if len(controls) != 2 * len(ACTUAL_SEMANTIC_FIELDS):
        contract_errors.append("semantic controls must equal two per semantic field")
    for packet_id, by_family in [
        *real_case_results.items(),
        *control_results.items(),
    ]:
        if set(by_family) != set(families):
            contract_errors.append(f"{packet_id} lacks the exact two-family panel")

    real_passes: list[dict[str, Any]] = []
    real_disagreements: list[dict[str, Any]] = []
    real_rejections: list[dict[str, Any]] = []
    citation_rows: list[dict[str, Any]] = []
    for packet_id, packet in sorted(real_packets.items()):
        by_family = real_case_results.get(packet_id) or {}
        verdicts = {
            family: str((by_family.get(family) or {}).get("verdict") or "")
            for family in families
        }
        row = {
            "packet_id": packet_id,
            "state_id": str(packet["case_item_id"]),
            "field": str(packet["field"]),
            "verdicts": verdicts,
        }
        values = list(verdicts.values())
        if values == ["supported", "supported"]:
            real_passes.append(row)
        elif values == ["not_supported", "not_supported"]:
            real_rejections.append(row)
        elif set(values) == {"supported", "not_supported"}:
            real_disagreements.append(row)
        else:
            contract_errors.append(f"{packet_id} has invalid semantic verdicts")
        for family in families:
            judgment = by_family.get(family) or {}
            citation_rows.append(
                {
                    "kind": "real",
                    "packet_id": packet_id,
                    "family": family,
                    "citation_valid": bool(judgment.get("citation_valid")),
                    "citation_errors": list(judgment.get("citation_errors") or []),
                }
            )

    control_catches: list[dict[str, Any]] = []
    control_disagreements: list[dict[str, Any]] = []
    control_misses: list[dict[str, Any]] = []
    for packet_id, packet in sorted(control_packets.items()):
        by_family = control_results.get(packet_id) or {}
        verdicts = {
            family: str((by_family.get(family) or {}).get("verdict") or "")
            for family in families
        }
        row = {
            "item_id": packet_id,
            "field": str(packet["field"]),
            "verdicts": verdicts,
        }
        values = list(verdicts.values())
        if values == ["not_supported", "not_supported"]:
            control_catches.append(row)
        elif values == ["supported", "supported"]:
            control_misses.append(row)
        elif set(values) == {"supported", "not_supported"}:
            control_disagreements.append(row)
        else:
            contract_errors.append(f"{packet_id} has invalid control verdicts")
        for family in families:
            judgment = by_family.get(family) or {}
            citation_rows.append(
                {
                    "kind": "control",
                    "packet_id": packet_id,
                    "family": family,
                    "citation_valid": bool(judgment.get("citation_valid")),
                    "citation_errors": list(judgment.get("citation_errors") or []),
                }
            )

    deterministic_gate = dict(corpus_audit.get("deterministic_gate") or {})
    status = (
        "PASS"
        if not contract_errors
        and deterministic_gate.get("status") == "PASS"
        and not real_rejections
        and not control_misses
        else "FAIL"
    )
    control_manifest = build_control_manifest(controls)
    return {
        "protocol": ACTUAL_CORPUS_REVIEW_PROTOCOL,
        "status": status,
        "human_calibration_performed": False,
        "judge_families": families,
        "panel_policy": ACTUAL_PANEL_POLICY,
        "citation_policy": ACTUAL_CITATION_POLICY,
        "deterministic_fields": list(ACTUAL_DETERMINISTIC_FIELDS),
        "semantic_fields": list(ACTUAL_SEMANTIC_FIELDS),
        "derived_or_construction_fields": list(
            ACTUAL_DERIVED_OR_CONSTRUCTION_FIELDS
        ),
        "control_protocol": ACTUAL_CORPUS_CONTROL_PROTOCOL,
        "required_control_fields": list(ACTUAL_SEMANTIC_FIELDS),
        "controls_per_field": 2,
        "control_field_counts": {
            field: sum(row.get("rating_field") == field for row in controls)
            for field in ACTUAL_SEMANTIC_FIELDS
        },
        "control_contract_errors": contract_errors,
        "control_manifest": control_manifest,
        "control_matrix_sha256": sha256_text(canonical_json(control_manifest)),
        "n_real_cases": len(state_ids),
        "n_real_semantic_packets": len(real_packets),
        "n_controls": len(controls),
        "real_case_panel_passes": real_passes,
        "real_case_disagreements": real_disagreements,
        "real_case_rejections": real_rejections,
        "control_catches": control_catches,
        "control_disagreements": control_disagreements,
        "control_misses": control_misses,
        "deterministic_gate": deterministic_gate,
        "citation_audit": {
            "policy": ACTUAL_CITATION_POLICY,
            "n_observations": len(citation_rows),
            "n_valid": sum(bool(row["citation_valid"]) for row in citation_rows),
            "integrity_rate": (
                sum(bool(row["citation_valid"]) for row in citation_rows)
                / max(len(citation_rows), 1)
            ),
            "defects": [row for row in citation_rows if not row["citation_valid"]],
        },
        "gate_reason": (
            "deterministic facts passed; no real semantic packet was unanimously "
            "rejected; no negative control was unanimously missed; split judgments "
            "are retained and reported"
            if status == "PASS"
            else "see deterministic_gate / control_contract_errors / "
            "real_case_rejections / control_misses"
        ),
    }


def require_actual_corpus_semantic_review_pass(
    report_path: str | Path,
    attestation_path: str | Path,
    *,
    expected_experiment_config_path: str | Path,
    expected_states_path: str | Path,
    expected_evaluator_contexts_path: str | Path,
    expected_backend_path: str | Path,
    expected_strategy_bank_path: str | Path,
    expected_pm_config_path: str | Path,
) -> dict[str, Any]:
    verification = require_artifact_attestation(
        attestation_path,
        required_stage=ACTUAL_CORPUS_REVIEW_STAGE,
        required_output_paths={"gate_report": report_path},
    )
    report = read_json(report_path)
    attestation = read_json(attestation_path)
    expected_inputs = {
        "experiment_config": expected_experiment_config_path,
        "states": expected_states_path,
        "evaluator_contexts": expected_evaluator_contexts_path,
        "memory_backend": expected_backend_path,
        "strategy_bank": expected_strategy_bank_path,
        "pm_v1_5_config": expected_pm_config_path,
    }
    for logical_name, expected_path in expected_inputs.items():
        record = (attestation.get("inputs") or {}).get(logical_name)
        if not isinstance(record, dict) or record.get("sha256") != sha256_file(
            expected_path
        ):
            raise RuntimeError(
                "actual semantic-review attestation does not bind current "
                + logical_name
            )
    corpus_hashes = (report.get("corpus_audit") or {}).get("input_hashes") or {}
    required_outputs = {
        "real_case_judgments",
        "control_judgments",
        "controls",
        "gate_report",
        "physical_attempt_ledger",
    }
    output_records = attestation.get("outputs") or {}
    control_manifest = report.get("control_manifest") or []
    controls_record = output_records.get("controls") or {}
    controls_path = Path(str(controls_record.get("path") or ""))
    pm_config = load_config(expected_pm_config_path)
    expected_endpoint_names = list(
        (pm_config.get("actual_corpus_semantic_audit") or {}).get(
            "judge_endpoints"
        )
        or []
    )
    expected_descriptors = build_judge_endpoint_descriptors(
        load_config(expected_experiment_config_path), expected_endpoint_names
    )
    if (
        report.get("protocol") != ACTUAL_CORPUS_REVIEW_PROTOCOL
        or report.get("status") != "PASS"
        or int(report.get("n_real_cases") or 0) != 468
        or int(report.get("n_real_semantic_packets") or 0)
        != 468 * len(ACTUAL_SEMANTIC_FIELDS)
        or report.get("control_protocol") != ACTUAL_CORPUS_CONTROL_PROTOCOL
        or report.get("panel_policy") != ACTUAL_PANEL_POLICY
        or report.get("citation_policy") != ACTUAL_CITATION_POLICY
        or list(report.get("deterministic_fields") or [])
        != list(ACTUAL_DETERMINISTIC_FIELDS)
        or list(report.get("semantic_fields") or [])
        != list(ACTUAL_SEMANTIC_FIELDS)
        or list(report.get("derived_or_construction_fields") or [])
        != list(ACTUAL_DERIVED_OR_CONSTRUCTION_FIELDS)
        or list(report.get("required_control_fields") or [])
        != list(ACTUAL_SEMANTIC_FIELDS)
        or int(report.get("controls_per_field") or 0) != 2
        or int(report.get("n_controls") or 0)
        != 2 * len(ACTUAL_SEMANTIC_FIELDS)
        or report.get("control_field_counts")
        != {field: 2 for field in ACTUAL_SEMANTIC_FIELDS}
        or list(report.get("control_contract_errors") or [])
        or len(str(report.get("control_matrix_sha256") or "")) != 64
        or len(report.get("control_catches") or [])
        + len(report.get("control_disagreements") or [])
        != 2 * len(ACTUAL_SEMANTIC_FIELDS)
        or list(report.get("control_misses") or [])
        or list(report.get("real_case_rejections") or [])
        or (report.get("deterministic_gate") or {}).get("status") != "PASS"
        or not required_outputs <= set(output_records)
        or len(control_manifest) != 2 * len(ACTUAL_SEMANTIC_FIELDS)
        or sha256_text(canonical_json(control_manifest))
        != report.get("control_matrix_sha256")
        or not controls_path.is_file()
        or read_json(controls_path) != control_manifest
        or (report.get("corpus_audit") or {}).get("fallback_gate", {}).get("status")
        != "PASS"
        or corpus_hashes.get("states") != sha256_file(expected_states_path)
        or corpus_hashes.get("evaluator_contexts")
        != sha256_file(expected_evaluator_contexts_path)
        or corpus_hashes.get("backend") != sha256_file(expected_backend_path)
        or corpus_hashes.get("strategy_bank")
        != sha256_file(expected_strategy_bank_path)
        or (report.get("corpus_audit") or {}).get("control_matrix_sha256")
        != report.get("control_matrix_sha256")
        or report.get("judge_endpoint_descriptors") != expected_descriptors
        or report.get("judge_families") != expected_endpoint_names
        or (attestation.get("parameters") or {}).get(
            "judge_endpoint_descriptors"
        )
        != expected_descriptors
        or (attestation.get("parameters") or {}).get("panel_policy")
        != ACTUAL_PANEL_POLICY
        or (attestation.get("parameters") or {}).get("citation_policy")
        != ACTUAL_CITATION_POLICY
    ):
        raise RuntimeError("actual 468-state semantic/fallback gate did not PASS")
    return {
        "status": "PASS",
        "report": report,
        "report_sha256": sha256_file(report_path),
        "attestation_sha256": verification["attestation_sha256"],
    }
