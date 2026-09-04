#!/usr/bin/env python3
"""Materialize the outcome-blind V5.1 T5 evaluation surfaces.

This stage is deliberately response-free.  It replays the frozen compiler,
query builders, exact Rank-1 retrievers, transparent feature builders and four
logistic heads on three final domains:

* SEALED_INTERNAL_TEST: the untouched 128-state controlled test;
* ESConv: the frozen one-dialogue-one-state panel, with memory unavailable;
* EvoEmo: p7-p12 qualification and p13-p18 lockbox users, each with only that
  user's complete causal prior history.

The EvoEmo target is a ``subsequent_topic`` after ``dialog_history``.  Hence
the complete dialogue history is legitimately prior.  Evaluator-only topic
annotations, related-session labels, reference answers and future outcomes are
never compiled into memory or exposed to the PM/generator.

No generated response, human label, LLM judgment, sealed outcome or external
outcome is read here.  The resulting surface is the only legal input to the
subsequent frozen call-plan builder.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping

from metacom_pm.contracts import MemorySource
from metacom_pm.evoemo import load_evoemo
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.retrieval import source_specific_memory_queries
from metacom_pm.v1_5_candidate_discovery import (
    FINAL_TYPED_CANDIDATE_DISCOVERY_PROTOCOL,
    discover_final_typed_memory_candidates,
)
from metacom_pm.v1_5_final_candidate_contract import (
    FINAL_FEATURE_CONTRACT_PROTOCOL,
    FINAL_FEATURE_NAMES,
    CandidateSubtype,
    IndependentFeatureRecord,
    exact_rank1_memory_surface,
    exact_rank1_strategy_surface,
    final_model_feature_projection,
    final_rs_model_features,
)
from metacom_pm.v1_5_memory_opportunity_features import (
    SOURCE_SPECIFIC_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL,
    build_source_specific_memory_opportunity_observation,
)
from metacom_pm.v1_5_memory_transport import (
    BOUNDED_MEMORY_COMPILER_PROTOCOL,
    compile_bounded_memory_with_metadata,
)
from metacom_pm.v1_5_strategy_rag_repair import (
    PRE_PM_CANDIDATE_RANK_PROTOCOL,
    effect_study_observable_flags,
    rank_pre_pm_strategy_candidates,
)
from metacom_pm.v1_5_typed_resource_adapter import TypedResourceCandidate
from metacom_pm.v1_5_v3_candidate_materialization import _rs_dialogue, _rs_query
from metacom_pm.v1_5b_policy_runtime import COMPONENTS


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5.1-t5-surface-materialization-v1"
CONTRACT = ROOT / "data/pm_v1_5_contracts/v5_1_system_pareto_confirmation_v1.json"
FIT_FEATURES = ROOT / "outputs/pm_v1_5_v5_fit_training_surface_v1/fit_feature_rows_private.jsonl"
RAW_STATES = ROOT / "data/pm_v1_5_v3_visible_states_v2/private/raw_states.jsonl"
BLUEPRINT = ROOT / "data/pm_v1_5_v3_effect_blueprint_v1/private/construction_blueprint.jsonl"
CARDS = ROOT / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl"
ESCONV_PANEL = ROOT / "outputs/pm_v1_5b_corrected_external_split_v1/esconv_corrected_rerun_panel_private.jsonl"
EVO_QUALIFICATION_PANEL = ROOT / "outputs/pm_v1_5b_corrected_external_split_v1/evoemo_qualification_panel_private.jsonl"
EVO_LOCKBOX_PANEL = ROOT / "outputs/pm_v1_5b_corrected_external_split_v1/evoemo_lockbox_panel_private.jsonl"
EVO_RAW = ROOT / "data/external/evo_emo.json"
EXTERNAL_SPLIT_REPORT = ROOT / "outputs/pm_v1_5b_corrected_external_split_v1/external_split_freeze_report.json"


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def learned_probability(
    *, component: str, features: Mapping[str, float], contract: Mapping[str, Any]
) -> float:
    head = contract["frozen_policy"]["heads"][component]
    names = list(head["feature_names"])
    if set(features) != set(names):
        raise RuntimeError(f"{component} T5 feature schema differs from frozen head")
    total = float(head["intercept"])
    for name, mean, scale, coefficient in zip(
        names,
        head["scaler_mean"],
        head["scaler_scale"],
        head["coefficients"],
        strict=True,
    ):
        divisor = float(scale) or 1.0
        total += ((float(features[name]) - float(mean)) / divisor) * float(coefficient)
    return sigmoid(total)


def transparent_rule(component: str, f: Mapping[str, float]) -> bool:
    if component == "MP":
        preference = f["candidate_preference_scope_fit"] >= 0.5
        profile = (
            f["candidate_profile_relevance"] >= 0.5
            and f["candidate_profile_entity_scope_fit"] >= 0.5
        )
        return bool(
            f["candidate_incremental_information"] >= 0.5
            and f["candidate_current_scope_conflict"] < 0.5
            and (preference or profile)
        )
    if component == "MS":
        return bool(
            f["candidate_incremental_information"] >= 0.5
            and f["candidate_prior_issue_marked_resolved"] < 0.5
            and f["candidate_current_goal_fit"] >= 0.5
            and (
                f["candidate_prior_outcome_or_distinction"] >= 0.5
                or f["candidate_specific_issue_or_distinction"] >= 0.5
            )
        )
    if component == "ME":
        return bool(
            f["candidate_incremental_information"] >= 0.5
            and f["candidate_current_goal_fit"] >= 0.5
            and f["candidate_contains_action"] >= 0.5
            and (
                f["candidate_contains_result"] >= 0.5
                or f["candidate_contains_mechanism"] >= 0.5
            )
        )
    return all(
        f[name] >= 0.5
        for name in (
            "candidate_mode_fit",
            "candidate_goal_fit",
            "candidate_burden_fit",
            "candidate_boundary_fit",
            "candidate_nonredundancy",
        )
    )


def visible_state(panel_row: Mapping[str, Any]) -> dict[str, Any]:
    runtime = dict(panel_row["runtime_state"])
    return {
        "state_id": str(runtime["state_id"]),
        "user_id": str(runtime["user_id"]),
        "visible_dialogue": [
            {"role": str(turn["role"]), "content": str(turn["content"])}
            for turn in runtime["current_session_history"]
        ],
        "current_user_text": str(runtime["current_user_text"]),
        "current_session_index": int(runtime["session_index"]),
    }


_ACTION_RE = re.compile(
    r"\b(?:tried|used|chose|decided|asked|called|wrote|walked|went|practiced|"
    r"paused|breathed|talked|scheduled|limited|stopped|started)\b",
    re.IGNORECASE,
)
_RESULT_RE = re.compile(
    r"\b(?:helped|worked|eased|reduced|improved|made .{0,60} easier|"
    r"felt (?:better|calmer|safer)|did not help|didn't help|made .{0,60} worse)\b",
    re.IGNORECASE,
)
_MECHANISM_RE = re.compile(
    r"\b(?:because|by |so that|which (?:helped|made|allowed)|"
    r"the part that|what helped was|worked because)\b",
    re.IGNORECASE,
)


def split_reusable_event(text: str) -> tuple[str, str, str] | None:
    """Deterministically split natural ME text without inventing a paraphrase.

    Only literal substrings from the selected Rank-1 episode are returned.  A
    failure makes the candidate structurally non-executable and therefore OFF;
    it is never repaired with an LLM or an evaluator label.
    """

    value = " ".join(text.split())
    action = _ACTION_RE.search(value)
    result = _RESULT_RE.search(value)
    mechanism = _MECHANISM_RE.search(value)
    boundary = result or mechanism
    if action is None or boundary is None or boundary.start() <= action.start():
        return None
    action_text = value[: boundary.start()].strip(" ,;:-")
    outcome_text = value[boundary.start() :].strip(" ,;:-")
    if not action_text or not outcome_text:
        return None
    mechanism_text = outcome_text if result is None and mechanism is not None else ""
    return action_text, outcome_text, mechanism_text


def typed_memory_candidate(
    *, component: str, surface: Mapping[str, Any], owner_id: str
) -> tuple[TypedResourceCandidate | None, str | None]:
    if not surface["candidate_present"]:
        return None, "CANDIDATE_ABSENT"
    raw_hint = surface["compiler_subtype_hint"]
    hint = str(getattr(raw_hint, "value", raw_hint))
    text = str(surface["candidate_text"]).strip()
    common = {
        "component": component,
        "resource_id": str(surface["candidate_id"]),
        "candidate_version": str(surface["candidate_text_sha256"]),
        "owner_id": owner_id,
    }
    if component == "MP" and hint == CandidateSubtype.MP_PREFERENCE.value:
        return TypedResourceCandidate(
            **common,
            subtype="MP_PREFERENCE",
            source_kind="profile",
            preference=text.split(":", 1)[-1].strip(),
        ), None
    if component == "MP" and hint == CandidateSubtype.MP_PROFILE.value:
        return TypedResourceCandidate(
            **common,
            subtype="MP_PROFILE",
            source_kind="profile",
            profile_fact=text.split(":", 1)[-1].strip(),
        ), None
    if component == "MS" and hint == CandidateSubtype.MS_SESSION.value:
        return TypedResourceCandidate(
            **common,
            subtype="MS_SESSION_OBSERVATION",
            source_kind="session",
            strictly_prior=True,
            age_sessions=int(surface["candidate_age_sessions"]),
            prior_observation=text,
        ), None
    if component == "ME" and hint == CandidateSubtype.ME_REUSABLE_OUTCOME.value:
        parsed = split_reusable_event(text)
        if parsed is None:
            return None, "ME_LITERAL_ACTION_OUTCOME_SPLIT_FAILED"
        past_action, observed_outcome, mechanism = parsed
        return TypedResourceCandidate(
            **common,
            subtype="ME_REUSABLE_OUTCOME",
            source_kind="event",
            strictly_prior=True,
            age_sessions=int(surface["candidate_age_sessions"]),
            past_action=past_action,
            observed_outcome=observed_outcome,
            mechanism=mechanism,
        ), None
    return None, f"UNSUPPORTED_TYPED_SUBTYPE:{hint}"


def subtype_value(surface: Mapping[str, Any]) -> str:
    raw = surface["compiler_subtype_hint"]
    return str(getattr(raw, "value", raw))


def typed_strategy_candidate(
    *, surface: Mapping[str, Any], card: Mapping[str, Any]
) -> TypedResourceCandidate:
    return TypedResourceCandidate(
        component="RS",
        subtype="RS_ATOMIC_MOVE",
        resource_id=str(surface["candidate_id"]),
        candidate_version=str(surface["candidate_text_sha256"]),
        source_kind="strategy",
        support_move=str(card["support_move"]),
        when_to_use=str(card["when_to_use"]),
        when_not_to_use=str(card["when_not_to_use"]),
    )


def empirical_support_contract(
    fit_rows: list[dict[str, Any]], contract: Mapping[str, Any]
) -> dict[str, Any]:
    by_component: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in fit_rows:
        by_component[str(row["component"])].append(
            {str(k): float(v) for k, v in row["model_features"].items()}
        )
    result: dict[str, Any] = {}
    for component in COMPONENTS:
        head = contract["frozen_policy"]["heads"][component]
        active = {
            name
            for name, coefficient in zip(
                head["feature_names"], head["coefficients"], strict=True
            )
            if abs(float(coefficient)) > 1e-12
        }
        result[component] = {
            "active_features": sorted(active),
            "empirical_ranges": {
                name: [
                    min(row[name] for row in by_component[component]),
                    max(row[name] for row in by_component[component]),
                ]
                for name in FINAL_FEATURE_NAMES[component]
            },
        }
    return result


def support_diagnostics(
    *, component: str, features: Mapping[str, float], support: Mapping[str, Any]
) -> dict[str, bool]:
    # All final transparent factors have a preregistered bounded semantic scale.
    contract_scale = all(float(value) in {0.0, 0.5, 1.0} for value in features.values())
    ranges = support[component]["empirical_ranges"]
    active = set(support[component]["active_features"])
    empirical_all = all(
        float(ranges[name][0]) <= float(value) <= float(ranges[name][1])
        for name, value in features.items()
    )
    empirical_active = all(
        float(ranges[name][0]) <= float(features[name]) <= float(ranges[name][1])
        for name in active
    )
    return {
        "contract_scale_in_support": contract_scale,
        "empirical_all_feature_range_in_support": empirical_all,
        "empirical_nonzero_coefficient_range_in_support": empirical_active,
    }


def materialize_memory_components(
    *, state: Mapping[str, Any], user: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    items, _session_docs, metadata = compile_bounded_memory_with_metadata(dict(user))
    session_index = int(state["current_session_index"])
    if any(int(item.created_session) >= session_index for item in items):
        raise RuntimeError(f"current/future item entered memory: {state['state_id']}")
    queries = source_specific_memory_queries(
        str(state["current_user_text"]), list(state["visible_dialogue"]), ""
    )
    discoveries = discover_final_typed_memory_candidates(
        queries=queries,
        items=items,
        source_metadata=metadata,
        session_index=session_index,
    )
    result: dict[str, dict[str, Any]] = {}
    for component in ("MP", "MS", "ME"):
        source = MemorySource(component)
        candidate = discoveries[source]
        surface = asdict(
            exact_rank1_memory_surface(
                state_id=str(state["state_id"]),
                candidate=candidate,
                source_metadata=metadata,
                current_session_index=session_index,
                compiler_protocol=BOUNDED_MEMORY_COMPILER_PROTOCOL,
            )
        )
        observation = build_source_specific_memory_opportunity_observation(
            candidate=candidate,
            catalog_items=items,
            catalog_user_id=str(user["id"]),
            current_user_id=str(state["user_id"]),
            current_session_index=session_index,
            current_user_text=str(state["current_user_text"]),
            visible_dialogue=list(state["visible_dialogue"]),
            source_metadata=metadata,
            background_action="M0+R0",
        )
        features = final_model_feature_projection(
            component=component, source_features=observation["model_features"]
        )
        typed, typed_failure = typed_memory_candidate(
            component=component,
            surface=surface,
            owner_id=str(state["user_id"]),
        )
        result[component] = {
            "surface": surface,
            "model_features": features,
            "deterministic_hard_off": bool(observation["deterministic_hard_off"]),
            "hard_off_reasons": list(observation["hard_off_reasons"]),
            "typed_candidate": asdict(typed) if typed is not None else None,
            "typed_execution_failure": typed_failure,
            "catalog_count": sum(item.source is source for item in items),
            "retrieved_count": len(candidate.selected_items),
            "candidate_discovery_protocol": FINAL_TYPED_CANDIDATE_DISCOVERY_PROTOCOL,
        }
    audit = {
        "compiled_item_count": len(items),
        "catalog_counts": {
            component: sum(item.source is MemorySource(component) for item in items)
            for component in ("MP", "MS", "ME")
        },
        "strictly_prior": all(int(item.created_session) < session_index for item in items),
        "owner_id": str(user["id"]),
        "memory_compiler_protocol": BOUNDED_MEMORY_COMPILER_PROTOCOL,
    }
    return result, audit


def materialize_rs(
    *, state: Mapping[str, Any], cards: list[dict[str, Any]], cards_by_id: Mapping[str, Any]
) -> dict[str, Any]:
    dialogue = _rs_dialogue(state)
    seekers = [row["content"] for row in dialogue if row["speaker"] == "seeker"]
    recent = " ".join(" ".join(seekers[-3:]).split())
    flags = effect_study_observable_flags(
        current_user_text=str(state["current_user_text"]),
        recent_user_text=recent,
        visible_dialogue=dialogue,
    )
    ranked = rank_pre_pm_strategy_candidates(
        query=_rs_query(dialogue, flags),
        current_user_text=str(state["current_user_text"]),
        recent_user_text=recent,
        visible_dialogue=dialogue,
        cards=cards,
    )
    surface = asdict(
        exact_rank1_strategy_surface(
            state_id=str(state["state_id"]),
            ranked_cards=ranked,
            cards_by_id=cards_by_id,
            compiler_protocol=PRE_PM_CANDIDATE_RANK_PROTOCOL,
        )
    )
    card = cards_by_id.get(str(surface["candidate_id"])) if surface["candidate_present"] else None
    features = final_rs_model_features(
        current_user_text=str(state["current_user_text"]),
        visible_dialogue=list(state["visible_dialogue"]),
        observable_flags=flags,
        ranked_card={"card_id": surface["candidate_id"]} if card is not None else None,
        bank_card=card,
    )
    typed = typed_strategy_candidate(surface=surface, card=card) if card is not None else None
    return {
        "surface": surface,
        "model_features": features,
        "deterministic_hard_off": not bool(ranked),
        "hard_off_reasons": [] if ranked else ["NO_APPLICABLE_STRATEGY_CARD"],
        "typed_candidate": asdict(typed) if typed is not None else None,
        "typed_execution_failure": None if typed is not None else "CANDIDATE_ABSENT",
        "catalog_count": len(cards),
        "retrieved_count": len(ranked),
        "candidate_discovery_protocol": PRE_PM_CANDIDATE_RANK_PROTOCOL,
        "observable_flags_private_audit_only": flags,
    }


def materialize_domain_state(
    *,
    domain: str,
    partition: str,
    state: Mapping[str, Any],
    user: Mapping[str, Any] | None,
    group_id: str,
    cards: list[dict[str, Any]],
    cards_by_id: Mapping[str, Any],
    contract: Mapping[str, Any],
    support: Mapping[str, Any],
    target_component: str | None = None,
) -> dict[str, Any]:
    components: dict[str, dict[str, Any]] = {}
    memory_audit: dict[str, Any]
    if user is None:
        memory_audit = {
            "structurally_unavailable": True,
            "reason": "ESCONV_HAS_NO_CAUSAL_CROSS_SESSION_USER_MEMORY",
        }
        for component in ("MP", "MS", "ME"):
            features = {name: 0.0 for name in FINAL_FEATURE_NAMES[component]}
            components[component] = {
                "surface": {
                    "state_id": str(state["state_id"]),
                    "component": component,
                    "candidate_present": False,
                    "candidate_id": None,
                    "candidate_text": None,
                    "candidate_text_sha256": None,
                    "candidate_age_sessions": None,
                    "compiler_subtype_hint": CandidateSubtype.CANDIDATE_ABSENT.value,
                    "selected_rank": None,
                    "compiler_protocol": BOUNDED_MEMORY_COMPILER_PROTOCOL,
                },
                "model_features": features,
                "deterministic_hard_off": True,
                "hard_off_reasons": ["MEMORY_STRUCTURALLY_UNAVAILABLE_IN_ESCONV"],
                "typed_candidate": None,
                "typed_execution_failure": "CANDIDATE_ABSENT",
                "catalog_count": 0,
                "retrieved_count": 0,
                "candidate_discovery_protocol": FINAL_TYPED_CANDIDATE_DISCOVERY_PROTOCOL,
            }
    else:
        components, memory_audit = materialize_memory_components(state=state, user=user)
    components["RS"] = materialize_rs(state=state, cards=cards, cards_by_id=cards_by_id)

    for component, item in components.items():
        features = {str(k): float(v) for k, v in item["model_features"].items()}
        IndependentFeatureRecord(
            protocol=FINAL_FEATURE_CONTRACT_PROTOCOL,
            state_id=str(state["state_id"]),
            component=component,  # type: ignore[arg-type]
            feature_builder_protocol=(
                SOURCE_SPECIFIC_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL
                if component != "RS"
                else "pm-v1.5-final-rs-transparent-features-v1"
            ),
            model_features=features,
        )
        diagnostics = support_diagnostics(
            component=component, features=features, support=support
        )
        probability = learned_probability(
            component=component, features=features, contract=contract
        )
        threshold = float(contract["frozen_policy"]["heads"][component]["threshold"])
        executable = bool(
            item["surface"]["candidate_present"]
            and not item["deterministic_hard_off"]
            and item["typed_candidate"] is not None
            and diagnostics["contract_scale_in_support"]
        )
        item.update(
            {
                "model_features": features,
                "support_diagnostics": diagnostics,
                "structurally_executable": executable,
                "learned_probability": probability,
                "learned_threshold": threshold,
                "learned_requested_on_before_feasibility": probability >= threshold,
                "transparent_rule_requested_on_before_feasibility": transparent_rule(
                    component, features
                ),
                "response_or_outcome_read": False,
            }
        )

    return {
        "protocol": PROTOCOL,
        "domain": domain,
        "partition": partition,
        "state_id": str(state["state_id"]),
        "group_id_private_analysis_only": group_id,
        "user_id_private_analysis_only": str(state["user_id"]),
        "target_component_private_analysis_only": target_component,
        "visible_dialogue": list(state["visible_dialogue"]),
        "current_user_text": str(state["current_user_text"]),
        "current_session_index_private_analysis_only": int(state["current_session_index"]),
        "components": components,
        "memory_transport_audit_private_only": memory_audit,
        "outcomes_read": False,
        "evaluator_only_fields_read_by_pm_or_generator": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_1_t5_surfaces_v1",
    )
    args = parser.parse_args()
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V5.1 requires {FORMAL_PYTHON}; got {sys.executable}")

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract.get("confirmation_or_external_outcomes_read") is not False:
        raise RuntimeError("V5.1 frozen policy contract is not outcome-blind")
    fit_features = rows(FIT_FEATURES)
    support = empirical_support_contract(fit_features, contract)
    cards = rows(CARDS)
    cards_by_id = {str(card["card_id"]): card for card in cards}
    if len(cards) != 80 or len(cards_by_id) != 80:
        raise RuntimeError("T5 requires the frozen 80-card Strategy Bank")

    blueprint = {str(row["blueprint_row_id"]): row for row in rows(BLUEPRINT)}
    sealed_states = [row for row in rows(RAW_STATES) if row["split"] == "SEALED_INTERNAL_TEST"]
    if len(sealed_states) != 128:
        raise RuntimeError("sealed internal test must contain 128 states")

    evo_users = {str(user["id"]): user for user in load_evoemo(EVO_RAW)}
    qualification_rows = rows(EVO_QUALIFICATION_PANEL)
    lockbox_rows = rows(EVO_LOCKBOX_PANEL)
    esconv_rows = rows(ESCONV_PANEL)

    output_rows: list[dict[str, Any]] = []
    for raw in sealed_states:
        state = {
            "state_id": str(raw["state_id"]),
            "user_id": str(raw["user_id"]),
            "visible_dialogue": list(raw["visible_dialogue"]),
            "current_user_text": str(raw["current_user_text"]),
            "current_session_index": int(raw["current_session_index"]),
        }
        source = blueprint[str(raw["state_id"])]
        output_rows.append(
            materialize_domain_state(
                domain="SEALED_INTERNAL",
                partition="sealed_internal_test",
                state=state,
                user=raw["user"],
                group_id=str(source["counterfactual_group_id"]),
                cards=cards,
                cards_by_id=cards_by_id,
                contract=contract,
                support=support,
                target_component=str(source["target_component"]),
            )
        )
    for panel in esconv_rows:
        state = visible_state(panel)
        output_rows.append(
            materialize_domain_state(
                domain="ESCONV",
                partition="esconv_corrected_test_panel",
                state=state,
                user=None,
                group_id=str(panel["dialogue_id_private_analysis_only"]),
                cards=cards,
                cards_by_id=cards_by_id,
                contract=contract,
                support=support,
            )
        )
    for partition, panel_rows in (
        ("evoemo_qualification_p7_p12", qualification_rows),
        ("evoemo_lockbox_p13_p18", lockbox_rows),
    ):
        for panel in panel_rows:
            state = visible_state(panel)
            user_id = str(panel["user_id_private_analysis_only"])
            user = evo_users[user_id]
            expected_index = len(user.get("dialog_history") or []) + 1
            if int(state["current_session_index"]) != expected_index:
                raise RuntimeError(
                    f"EvoEmo target is not after complete prior history: {state['state_id']}"
                )
            output_rows.append(
                materialize_domain_state(
                    domain="EVOEMO",
                    partition=partition,
                    state=state,
                    user=user,
                    group_id=user_id,
                    cards=cards,
                    cards_by_id=cards_by_id,
                    contract=contract,
                    support=support,
                )
            )

    expected_partition_counts = {
        "sealed_internal_test": 128,
        "esconv_corrected_test_panel": 122,
        "evoemo_qualification_p7_p12": 60,
        "evoemo_lockbox_p13_p18": 78,
    }
    observed_partitions = Counter(row["partition"] for row in output_rows)
    if observed_partitions != Counter(expected_partition_counts):
        raise RuntimeError(f"T5 partition shape mismatch: {observed_partitions}")
    if len({row["state_id"] for row in output_rows}) != len(output_rows):
        raise RuntimeError("T5 state IDs are not unique")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    surface_path = args.out_dir / "t5_state_surfaces_private.jsonl"
    write_jsonl(surface_path, output_rows)

    coverage: dict[str, Any] = {}
    for partition in expected_partition_counts:
        selected = [row for row in output_rows if row["partition"] == partition]
        per_component: dict[str, Any] = {}
        for component in COMPONENTS:
            items = [row["components"][component] for row in selected]
            per_component[component] = {
                "states": len(items),
                "candidate_present": sum(bool(item["surface"]["candidate_present"]) for item in items),
                "structurally_executable": sum(bool(item["structurally_executable"]) for item in items),
                "contract_scale_in_support": sum(bool(item["support_diagnostics"]["contract_scale_in_support"]) for item in items),
                "empirical_all_feature_range_in_support": sum(bool(item["support_diagnostics"]["empirical_all_feature_range_in_support"]) for item in items),
                "empirical_nonzero_coefficient_range_in_support": sum(bool(item["support_diagnostics"]["empirical_nonzero_coefficient_range_in_support"]) for item in items),
                "learned_requested_on_before_feasibility": sum(bool(item["learned_requested_on_before_feasibility"]) for item in items),
                "learned_realizable_on": sum(bool(item["learned_requested_on_before_feasibility"] and item["structurally_executable"]) for item in items),
                "transparent_realizable_on": sum(bool(item["transparent_rule_requested_on_before_feasibility"] and item["structurally_executable"]) for item in items),
                "typed_failure_reasons": dict(Counter(str(item["typed_execution_failure"]) for item in items if item["typed_execution_failure"])),
                "candidate_subtypes": dict(Counter(subtype_value(item["surface"]) for item in items)),
            }
        coverage[partition] = {
            "states": len(selected),
            "independent_groups": len({row["group_id_private_analysis_only"] for row in selected}),
            "components": per_component,
        }

    qualification = coverage["evoemo_qualification_p7_p12"]["components"]
    evo_memory_qualification = {
        component: bool(
            qualification[component]["candidate_present"] >= 20
            and qualification[component]["structurally_executable"] >= 20
        )
        for component in ("MP", "MS", "ME")
    }
    checks = {
        "expected_388_states": len(output_rows) == 388,
        "partition_counts_exact": observed_partitions == Counter(expected_partition_counts),
        "sealed_64_counterfactual_groups": coverage["sealed_internal_test"]["independent_groups"] == 64,
        "esconv_122_independent_dialogues": coverage["esconv_corrected_test_panel"]["independent_groups"] == 122,
        "evo_qualification_6_users": coverage["evoemo_qualification_p7_p12"]["independent_groups"] == 6,
        "evo_lockbox_6_users": coverage["evoemo_lockbox_p13_p18"]["independent_groups"] == 6,
        "esconv_memory_always_unavailable": all(
            coverage["esconv_corrected_test_panel"]["components"][component]["candidate_present"] == 0
            for component in ("MP", "MS", "ME")
        ),
        "evo_complete_history_is_strictly_prior": all(
            row["memory_transport_audit_private_only"].get("strictly_prior") is True
            for row in output_rows if row["domain"] == "EVOEMO"
        ),
        "evo_owner_isolation": all(
            row["memory_transport_audit_private_only"].get("owner_id")
            == row["user_id_private_analysis_only"]
            for row in output_rows if row["domain"] == "EVOEMO"
        ),
        "all_components_use_frozen_feature_schema": all(
            set(row["components"][component]["model_features"])
            == set(FINAL_FEATURE_NAMES[component])
            for row in output_rows for component in COMPONENTS
        ),
        "all_features_on_frozen_half_step_scale": all(
            row["components"][component]["support_diagnostics"]["contract_scale_in_support"]
            for row in output_rows for component in COMPONENTS
        ),
        "no_response_human_judge_or_external_outcome_read": all(
            row["outcomes_read"] is False for row in output_rows
        ),
    }
    report = {
        "protocol": PROTOCOL,
        "status": "PASS_READY_FOR_T5_CALL_PLAN" if all(checks.values()) else "FAIL_T5_SURFACE_GATE",
        "states": len(output_rows),
        "partition_counts": dict(sorted(observed_partitions.items())),
        "coverage": coverage,
        "evoemo_static_memory_qualification": {
            "minimum_candidate_present_and_executable_states_per_component": 20,
            "component_pass": evo_memory_qualification,
            "all_memory_components_pass": all(evo_memory_qualification.values()),
            "lockbox_opening_uses_response_or_human_outcome": False,
        },
        "support_interpretation": {
            "routing_gate": "typed subtype + structural hard gates + frozen 0/0.5/1 feature contract",
            "empirical_fit_range": "reported as transport diagnostic, not silently clipped or used to tune external routes",
            "external_component_claim_minimum": "at least 20 candidate-present structurally executable states; inference clustered by user",
        },
        "evoemo_temporal_contract": {
            "target": "subsequent_topic after complete dialog_history",
            "complete_dialog_history_is_valid_prior_memory": True,
            "forbidden_from_pm_generator": [
                "subsequent_topics other than current visible track",
                "related_sessions evaluator labels",
                "observations and evaluator summaries",
                "reference answers and generated outcomes",
            ],
        },
        "checks": checks,
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                CONTRACT,
                FIT_FEATURES,
                RAW_STATES,
                BLUEPRINT,
                CARDS,
                ESCONV_PANEL,
                EVO_QUALIFICATION_PANEL,
                EVO_LOCKBOX_PANEL,
                EVO_RAW,
                EXTERNAL_SPLIT_REPORT,
            )
        },
        "implementation_sha256": {
            "surface_builder": sha256_file(Path(__file__).resolve()),
            "memory_compiler": sha256_file(ROOT / "src/metacom_pm/v1_5_memory_transport.py"),
            "memory_retriever": sha256_file(ROOT / "src/metacom_pm/v1_5_candidate_discovery.py"),
            "strategy_ranker": sha256_file(ROOT / "src/metacom_pm/v1_5_strategy_rag_repair.py"),
            "feature_contract": sha256_file(ROOT / "src/metacom_pm/v1_5_final_candidate_contract.py"),
            "typed_executor": sha256_file(ROOT / "src/metacom_pm/v1_5_typed_resource_adapter.py"),
        },
        "t5_state_surfaces_sha256": sha256_file(surface_path),
        "api_calls": 0,
        "human_or_llm_judgments_read": 0,
        "external_response_outcomes_read": 0,
    }
    write_json(args.out_dir / "surface_gate_report.json", report)
    print(
        {
            "protocol": PROTOCOL,
            "status": report["status"],
            "states": report["states"],
            "evoemo_static_memory_qualification": report["evoemo_static_memory_qualification"],
        }
    )


if __name__ == "__main__":
    main()
