from __future__ import annotations

import json
import importlib.util
import inspect
import os
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from pydantic import ValidationError
import yaml

from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.api import (
    CallResult,
    Endpoint,
    RetryableProviderError,
    StructuredOutputValidationError,
)
from metacom_pm import pm_v2_data as pm_v2_data_module
from metacom_pm.contracts import (
    DialogueTurn,
    MemoryItem,
    MemorySource,
    RuntimeState,
    StrategyCard,
)
from metacom_pm.attempt_ledger import PersistentAttemptLedger
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.evoemo import (
    _catalog,
    build_evo_memory,
    load_evoemo,
    make_evo_runtime_state,
)
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.pm_v2_contracts import (
    ActionLabel,
    CompositeSpec,
    PMV2Split,
    PMV2State,
    ResourceNeedRegime,
    ResponseDimensions,
    RiskDimensions,
)
from metacom_pm.pm_v2_audit import _regime_pass
from metacom_pm.pm_v2_data import (
    DATA_GENERATION_CONTRACT_VERSION,
    DETERMINISTIC_MEMORY_BLUEPRINT_PROTOCOL,
    GENERATION_CASE_FIELDS,
    GENERATION_FAMILY_TOPICS,
    GENERATED_MEMORY_DRAFT_MAX_CHARS,
    MIN_UNIQUE_CURRENT_TEXTS_PER_NINE_CASE_BUNDLE,
    READINESS_SURFACE_PROTOCOL,
    GenerationDraftCompilationError,
    GeneratedBundleDraft,
    GeneratedMemory,
    GeneratedSurfaceOnlyCaseDraft,
    GeneratedStateCase,
    GeneratedUserBundle,
    advice_readiness_target_for_case,
    audit_current_user_text_diversity,
    audit_cross_split_near_duplicates,
    bind_bundle_to_generation_run,
    build_evaluator_context_index,
    build_deployable_catalog_statistics,
    case_to_evaluator_context,
    case_to_memory_backend,
    case_to_state,
    compile_surface_only_user_bundle,
    compile_generation_draft,
    compiler_surface_from_provider,
    deterministic_memory_blueprint_preflight,
    evaluator_context_payload_sha256,
    generate_user_bundle,
    generation_case_family_assignments,
    generation_case_messages,
    generation_distractor_family_assignments,
    generation_messages,
    lint_generation_surface_case,
    observable_state_design,
    readiness_surface_clause_for_case,
    require_bundle_generation_binding,
    runtime_to_pmv2_state,
    state_to_v1_runtime,
    surface_generation_contract_hash,
    validate_generation_shortcut_controls,
    validate_split_manifests,
    write_development_dataset,
)
from metacom_pm.pm_v2_features import PMV2FeatureBuilder
from metacom_pm.bounded_retry import RETRY_CONTRACT_PROTOCOL, retry_ledger_summary
from metacom_pm.pm_v1_5_semantic import (
    FrozenSemanticEncoderSpec,
    FrozenTransformerSemanticEncoder,
    SemanticEncoderBinding,
    prepare_visible_semantic_state,
    visible_dialogue_state_text,
)
from metacom_pm.pm_v1_5_required_hit import validate_required_hit_preflight
from metacom_pm.pm_v2_judging import prompt_contract_hash
from metacom_pm.pm_v2_generation_pilot import (
    CALIBRATION_SEMANTIC_FAMILIES,
    GENERATION_PILOT_CONTRACT_VERSION,
    GENERATION_PILOT_FAMILIES,
    GENERATION_PILOT_MAX_CONTENT_ATTEMPTS,
    GENERATION_PILOT_MAX_ATTEMPTS,
    GENERATION_PILOT_MAX_TRANSPORT_ATTEMPTS_PER_CONTENT_ATTEMPT,
    GENERATION_PILOT_MINIMUM_CALLS,
    GENERATION_PILOT_STAGE,
    INTERNAL_TEST_SEMANTIC_FAMILIES,
    TRAIN_SEMANTIC_FAMILIES,
    build_generation_compatibility_contract,
    build_generation_compatibility_plan,
    generation_family_schedule,
    require_generation_compatibility_attestation,
    validate_generation_pilot_bundle,
)
from metacom_pm.pm_v2_generation_review import (
    PACKET_FIELDS as GENERATION_REVIEW_PACKET_FIELDS,
    RATING_FIELDS as GENERATION_REVIEW_RATING_FIELDS,
    SIMPLE_REVIEW_FIELDS as GENERATION_SIMPLE_REVIEW_FIELDS,
    analyze_generation_pilot_semantic_review,
    prepare_generation_pilot_semantic_review,
    require_generation_pilot_semantic_review,
)
from metacom_pm.text import conservative_token_bound, estimate_tokens, normalize_space


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_conservative_token_bound_applies_frozen_margin() -> None:
    assert estimate_tokens("x" * 8) == 2
    assert conservative_token_bound("x" * 8, safety_factor=1.5) == 3
    with pytest.raises(ValueError, match="safety factor"):
        conservative_token_bound("x", safety_factor=0.99)


def test_generate_user_bundle_defaults_to_single_physical_attempt() -> None:
    signature = inspect.signature(generate_user_bundle)
    assert signature.parameters["request_retries"].default == 1


def test_semantic_family_schedule_counterbalances_regime_positions() -> None:
    split_design = {
        "train": (24, TRAIN_SEMANTIC_FAMILIES),
        "calibration": (12, CALIBRATION_SEMANTIC_FAMILIES),
        "internal_test": (16, INTERNAL_TEST_SEMANTIC_FAMILIES),
    }
    for split, (user_count, expected_families) in split_design.items():
        schedule = generation_family_schedule(split, user_count)
        assert len(schedule) == user_count
        observed_positions = {family: set() for family in expected_families}
        observed_regimes = {family: set() for family in expected_families}
        for cohort in schedule:
            assert len(cohort) == 3
            assert len(set(cohort)) == 3
            for position, family in enumerate(cohort):
                observed_positions[family].add(position)
            for case_index, (_, regime) in enumerate(GENERATION_CASE_FIELDS):
                observed_regimes[cohort[case_index % 3]].add(regime)
        assert set(observed_positions) == set(expected_families)
        assert all(positions == {0, 1, 2} for positions in observed_positions.values())
        assert all(
            regimes == set(ResourceNeedRegime)
            for regimes in observed_regimes.values()
        )


def test_generation_pilot_uses_a_real_frozen_orthogonal_cohort() -> None:
    assert GENERATION_PILOT_CONTRACT_VERSION.startswith(
        "pm-v2-generation-compatibility-pilot-v8.12-"
    )
    assert GENERATION_PILOT_FAMILIES in (
        ("relocation_loneliness", "academic_pressure", "trust_rebuilding"),
        ("relocation_loneliness", "self_confidence", "sleep_disruption"),
        ("academic_pressure", "trust_rebuilding", "sleep_disruption"),
        ("self_confidence", "academic_pressure", "relocation_loneliness"),
    )


def _memory(
    source: MemorySource,
    index: int,
    *,
    created_session: int,
    text: str,
    stale: bool = False,
    conflict: bool = False,
    item_utility: str = "irrelevant",
) -> GeneratedMemory:
    return GeneratedMemory(
        memory_id=f"generated_{source.value}_{index}",
        source=source,
        text=text,
        created_session=created_session,
        stale=stale,
        conflicts_with_current_state=conflict,
        private_sensitivity="ordinary",
        item_utility=item_utility,
    )


def _case(
    *,
    case_id: str = "case_1",
    regime: ResourceNeedRegime = ResourceNeedRegime.CONTEXT_ONLY,
    current_user_text: str = "I feel torn about tomorrow's conversation.",
    needed_memory_sources: list[MemorySource] | None = None,
) -> GeneratedStateCase:
    default_needs = {
        ResourceNeedRegime.PROFILE_NEEDED: [MemorySource.MP],
        ResourceNeedRegime.SUMMARY_NEEDED: [MemorySource.MS],
        ResourceNeedRegime.EVENT_NEEDED: [MemorySource.ME],
        ResourceNeedRegime.MULTI_SOURCE_NEEDED: [MemorySource.MS, MemorySource.ME],
    }
    needed_sources = (
        needed_memory_sources
        if needed_memory_sources is not None
        else default_needs.get(regime, [])
    )
    return GeneratedStateCase(
        case_id=case_id,
        semantic_family="workplace_conflict",
        surface_form_id=f"surface_{case_id}",
        regime=regime,
        current_user_text=current_user_text,
        recent_dialogue=[
            DialogueTurn(role="user", content="The meeting was moved again."),
            DialogueTurn(role="assistant", content="That uncertainty sounds tiring."),
        ],
        session_summary="The user is preparing for a difficult workplace conversation.",
        session_index=6,
        profile_memories=[
            _memory(
                MemorySource.MP,
                1,
                created_session=1,
                text="The user prefers time to think before difficult conversations.",
                item_utility=(
                    "harmful"
                    if regime is ResourceNeedRegime.MEMORY_HARMFUL
                    else "helpful"
                    if MemorySource.MP in needed_sources
                    else "irrelevant"
                ),
            ),
            _memory(
                MemorySource.MP,
                2,
                created_session=2,
                text="The user values direct but calm communication.",
            ),
        ],
        summary_memories=[
            _memory(
                MemorySource.MS,
                1,
                created_session=3,
                text="Recent work uncertainty has made decision-making harder.",
                stale=MemorySource.MS not in needed_sources,
                item_utility=(
                    "helpful" if MemorySource.MS in needed_sources else "irrelevant"
                ),
            ),
            _memory(
                MemorySource.MS,
                2,
                created_session=2,
                text="The user once considered reorganizing a bookshelf.",
            ),
        ],
        event_memories=[
            _memory(
                MemorySource.ME,
                1,
                created_session=4,
                text="A manager postponed an important feedback meeting.",
                conflict=MemorySource.ME not in needed_sources,
                item_utility=(
                    "helpful" if MemorySource.ME in needed_sources else "irrelevant"
                ),
            ),
            _memory(
                MemorySource.ME,
                2,
                created_session=2,
                text="The user attended a neighborhood event last year.",
            ),
        ],
        needed_memory_sources=needed_sources,
        authorized_user_context="PRIVATE_AUTHORIZED_CONTEXT_SENTINEL",
        coverage_rationale="PRIVATE_COVERAGE_RATIONALE_SENTINEL",
    )


def _bundle() -> GeneratedUserBundle:
    regimes = [
        ResourceNeedRegime.CONTEXT_ONLY,
        ResourceNeedRegime.EVENT_NEEDED,
        ResourceNeedRegime.STRATEGY_HELPFUL,
        ResourceNeedRegime.STRATEGY_HARMFUL,
    ]
    return GeneratedUserBundle(
        user_id="user_1",
        profile_summary="A privacy-safe synthetic user.",
        stable_preferences=["calm communication"],
        boundaries=["no diagnosis"],
        cases=[
            _case(
                case_id=f"case_{index}",
                regime=regime,
                current_user_text=f"I feel torn about conversation number {index}.",
            )
            for index, regime in enumerate(regimes, 1)
        ],
        generator_seed_id="seed_1",
    )


def _full_regime_bundle(
    *,
    user_id: str = "full_regime_user",
    observed_families: tuple[str, ...] = ("family_a", "family_b", "family_c"),
) -> GeneratedUserBundle:
    cases = []
    for index, regime in enumerate(ResourceNeedRegime):
        case = _case(
            case_id=f"{user_id}_case_{index}",
            regime=regime,
            current_user_text=f"Unique current user text {user_id} number {index}.",
        )
        case.semantic_family = observed_families[index % len(observed_families)]
        cases.append(case)
    return GeneratedUserBundle(
        user_id=user_id,
        profile_summary="A privacy-safe synthetic full-regime user.",
        stable_preferences=["calm communication"],
        boundaries=["no diagnosis"],
        cases=cases,
        generator_seed_id=f"seed_{user_id}",
    )


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _write_unique_generation_seeds(path: Path, count: int = 53) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {"dialogue_text": f"user: unique held-out seed dialogue {index}."}
            )
            + "\n"
            for index in range(count)
        ),
        encoding="utf-8",
    )


def _role_slot_draft(
    families: list[str] | None = None,
) -> GeneratedBundleDraft:
    families = families or [
        "relocation_loneliness",
        "workload_burnout",
        "friendship_distance",
    ]
    regimes = list(ResourceNeedRegime)
    targets = generation_case_family_assignments(families, regimes)
    distractors = generation_distractor_family_assignments(families, regimes)
    topic = {family: GENERATION_FAMILY_TOPICS[family] for family in families}

    def item(source: MemorySource, family: str, label: str) -> dict:
        content = topic[family]
        common = {
            "relevance_explanation": f"This {label} fragment has the frozen topic.",
            "private_sensitivity": "ordinary",
        }
        if source is MemorySource.MP:
            return {"stable_user_fact": f"prefers calm reflection about {content}", **common}
        if source is MemorySource.MS:
            return {"cross_session_pattern": f"often reports strain around {content}", **common}
        return {"concrete_past_event": f"last month described one incident involving {content}", **common}

    needed_by_case = {
        "profile_needed": {MemorySource.MP},
        "summary_needed": {MemorySource.MS},
        "event_needed": {MemorySource.ME},
        "multi_source_needed": set(MemorySource),
    }
    payload: dict = {
        "profile_summary": "A user navigating "
        + ", ".join(topic[family] for family in families)
        + ".",
        "stable_preferences": [f"calm reflection about {topic[families[0]]}"],
        "boundaries": [
            "avoid assumptions about "
            + " or ".join(topic[family] for family in families[1:])
        ],
    }
    for index, (case_name, regime) in enumerate(
        (
            ("context_only", ResourceNeedRegime.CONTEXT_ONLY),
            ("profile_needed", ResourceNeedRegime.PROFILE_NEEDED),
            ("summary_needed", ResourceNeedRegime.SUMMARY_NEEDED),
            ("event_needed", ResourceNeedRegime.EVENT_NEEDED),
            ("multi_source_needed", ResourceNeedRegime.MULTI_SOURCE_NEEDED),
            ("memory_harmful", ResourceNeedRegime.MEMORY_HARMFUL),
            ("strategy_helpful", ResourceNeedRegime.STRATEGY_HELPFUL),
            ("strategy_harmful", ResourceNeedRegime.STRATEGY_HARMFUL),
            ("ambiguous", ResourceNeedRegime.AMBIGUOUS),
        ),
        1,
    ):
        target = targets[case_name]
        current_user_text = f"Concern {index} now feels tied to {topic[target]}."
        if regime is ResourceNeedRegime.STRATEGY_HELPFUL:
            current_user_text = (
                f"I want guidance about {topic[target]}; what should I do next?"
            )
        elif regime is ResourceNeedRegime.STRATEGY_HARMFUL:
            current_user_text = (
                f"Please just listen without advice while I describe {topic[target]}."
            )
        surface = {
            "current_user_text": current_user_text,
            "dialogue_before_current": [
                {"role": "user", "content": f"Earlier context number {index}."},
                {"role": "assistant", "content": f"Prior reflection number {index}."},
            ],
            "session_summary": f"The current concern centers on {topic[target]}.",
            "authorized_user_context": f"Authorized context about {topic[target]}.",
            "coverage_rationale": (
                "A structured plan and steps would help organize support."
                if regime is ResourceNeedRegime.STRATEGY_HELPFUL
                else "Directive advice would be premature; listening is safer."
                if regime is ResourceNeedRegime.STRATEGY_HARMFUL
                else f"Coverage is frozen for case {index}."
            ),
        }
        if regime is ResourceNeedRegime.MEMORY_HARMFUL:
            off_topic = distractors[case_name]["event_source"]
            surface.update(
                {
                    "profile_source": {
                        "outdated_user_fact": f"avoids talking about {topic[target]}",
                        "explicit_current_update": f"now openly discusses {topic[target]}",
                        "why_recall_is_harmful": "The old preference conflicts with the correction.",
                    },
                    "summary_source": {
                        "outdated_cross_session_pattern": f"repeatedly withdrew around {topic[target]}",
                        "explicit_current_update": f"now stays engaged around {topic[target]}",
                        "why_recall_is_harmful": "The old pattern conflicts with current context.",
                    },
                    "event_source": {
                        "unrelated_sensitive_past_event": f"a private incident involving {topic[off_topic]}",
                        "why_recall_is_intrusive": "It is unrelated and sensitive.",
                    },
                }
            )
        else:
            for source_field, source in (
                ("profile_source", MemorySource.MP),
                ("summary_source", MemorySource.MS),
                ("event_source", MemorySource.ME),
            ):
                source_payload = {
                    "distractor": item(
                        source,
                        distractors[case_name][source_field],
                        "distractor",
                    )
                }
                if source in needed_by_case.get(case_name, set()):
                    source_payload["helpful"] = item(source, target, "helpful")
                surface[source_field] = source_payload
        if regime is ResourceNeedRegime.MULTI_SOURCE_NEEDED:
            surface.update(
                {
                    "profile_unique_contribution": "A durable personal boundary.",
                    "summary_unique_contribution": "A recurring cross-session pattern.",
                    "event_unique_contribution": "A concrete prior incident.",
                }
            )
        payload[case_name] = surface
    return GeneratedBundleDraft.model_validate(payload)


def _source_grounded_bundle(
    *, user_id: str, families: list[str] | None = None
) -> GeneratedUserBundle:
    selected_families = families or [
        "relocation_loneliness",
        "workload_burnout",
        "friendship_distance",
    ]
    draft = _role_slot_draft(selected_families)
    bundle = compile_generation_draft(
        draft=draft,
        seed_dialogue="fixture held-out seed",
        user_id=user_id,
        semantic_families=selected_families,
        regimes=list(ResourceNeedRegime),
    )
    provider_draft = draft.model_dump(mode="json")
    provider_response = {
        "id": "fixture-source-grounded-provider-response",
        "choices": [
            {"message": {"content": canonical_json(provider_draft)}}
        ],
    }
    bundle.provenance.update(
        {
            "provider_draft": provider_draft,
            "provider_response": provider_response,
            "provider_response_sha256": sha256_text(
                canonical_json(provider_response)
            ),
        }
    )
    return bundle


def _source_grounded_pilot_bundle(
    contract: dict,
    *,
    request_hash: str,
) -> GeneratedUserBundle:
    bundle = _source_grounded_bundle(
        user_id=str(contract["pilot_user_id"]),
        families=[str(value) for value in contract["semantic_families"]],
    )
    bundle.provenance.update(
        {
            "generation_compatibility_contract_sha256": contract[
                "contract_sha256"
            ],
            "request_hash": request_hash,
        }
    )
    return bundle


def _surface_only_pilot_inputs(
    contract: dict,
) -> dict[str, GeneratedSurfaceOnlyCaseDraft]:
    """Project the legacy fixture onto the role-safe provider schema."""

    draft = _role_slot_draft(
        [str(value) for value in contract["semantic_families"]]
    )
    surfaces = {}
    for case_field, _ in GENERATION_CASE_FIELDS:
        compiled = getattr(draft, case_field)
        turns = compiled.dialogue_before_current
        assert len(turns) % 2 == 0
        assert all(
            turns[index].role == "user" and turns[index + 1].role == "assistant"
            for index in range(0, len(turns), 2)
        )
        exchanges = [
            {
                "user_text": turns[index].content,
                "assistant_text": turns[index + 1].content,
            }
            for index in range(0, len(turns), 2)
        ]
        target_exchanges = observable_state_design(
            user_id=str(contract["pilot_user_id"]), case_field=case_field
        )["history_turn_target"] // 2
        while len(exchanges) < target_exchanges:
            ordinal = len(exchanges) + 1
            exchanges.insert(
                0,
                {
                    "user_text": f"Earlier context detail {ordinal}.",
                    "assistant_text": f"Earlier support reply {ordinal}.",
                },
            )
        surfaces[case_field] = GeneratedSurfaceOnlyCaseDraft.model_validate(
            {
                "current_user_text": compiled.current_user_text,
                "dialogue_exchanges_before_current": exchanges[-target_exchanges:],
                "session_summary": compiled.session_summary,
                "authorized_user_context": compiled.authorized_user_context,
            }
        )
    return surfaces


def test_surface_provider_schema_makes_role_order_a_compiler_invariant() -> None:
    surface = GeneratedSurfaceOnlyCaseDraft.model_validate(
        {
            "current_user_text": "Moving to a new city still feels lonely today.",
            "dialogue_exchanges_before_current": [
                {
                    "user_text": "I have not met anyone in the new city yet.",
                    "assistant_text": "That sounds isolating; what feels hardest?",
                },
                {
                    "user_text": "Evenings feel especially quiet.",
                    "assistant_text": "The quiet evenings seem to make this sharper.",
                },
            ],
            "session_summary": "The user feels lonely after moving to a new city.",
            "authorized_user_context": "Use only the visible relocation details.",
        }
    )
    compiled = compiler_surface_from_provider(surface)
    assert [turn.role for turn in compiled.dialogue_before_current] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert compiled.dialogue_before_current[-1].role == "assistant"
    lint = lint_generation_surface_case(
        user_id="pm_v1_5_u002",
        case_field="context_only",
        regime=ResourceNeedRegime.CONTEXT_ONLY,
        family="relocation_loneliness",
        forbidden_families=("self_confidence", "sleep_disruption"),
        surface=surface,
    )
    assert lint["status"] == "PASS"
    assert lint["errors"] == []
    schema = GeneratedSurfaceOnlyCaseDraft.model_json_schema()
    assert "dialogue_exchanges_before_current" in schema["properties"]
    assert "dialogue_before_current" not in schema["properties"]


def test_surface_lint_allows_only_one_same_family_counterfactual_pair() -> None:
    surface = GeneratedSurfaceOnlyCaseDraft.model_validate(
        {
            # Exact V8.10 provider text: profile_needed and multi_source_needed
            # legitimately held the same self-confidence concern constant.
            "current_user_text": (
                "I often feel insecure about my abilities and wonder if I'm "
                "capable of achieving my goals."
            ),
            "dialogue_exchanges_before_current": [
                {
                    "user_text": "I keep doubting myself.",
                    "assistant_text": "What brings that self-doubt up today?",
                }
            ],
            "session_summary": "The user is struggling with self confidence.",
            "authorized_user_context": "Use only the visible self-doubt concern.",
        }
    )
    allowed = lint_generation_surface_case(
        user_id="pmv2_generation_compatibility_pilot",
        case_field="multi_source_needed",
        regime=ResourceNeedRegime.MULTI_SOURCE_NEEDED,
        family="self_confidence",
        forbidden_families=("relocation_loneliness", "sleep_disruption"),
        surface=surface,
        prior_current_user_texts=(surface.current_user_text,),
        prior_current_user_families=("self_confidence",),
    )
    assert allowed["status"] == "PASS"
    assert allowed["current_user_text_diversity"][
        "allowed_same_user_family_counterfactual_duplicate"
    ] is True

    cross_family = lint_generation_surface_case(
        user_id="pmv2_generation_compatibility_pilot",
        case_field="multi_source_needed",
        regime=ResourceNeedRegime.MULTI_SOURCE_NEEDED,
        family="self_confidence",
        forbidden_families=("relocation_loneliness", "sleep_disruption"),
        surface=surface,
        prior_current_user_texts=(surface.current_user_text,),
        prior_current_user_families=("relocation_loneliness",),
    )
    assert cross_family["status"] == "FAIL"
    assert cross_family["errors"][0]["check"] == (
        "current_user_text_cross_family_duplicate"
    )

    oversized = lint_generation_surface_case(
        user_id="pmv2_generation_compatibility_pilot",
        case_field="multi_source_needed",
        regime=ResourceNeedRegime.MULTI_SOURCE_NEEDED,
        family="self_confidence",
        forbidden_families=("relocation_loneliness", "sleep_disruption"),
        surface=surface,
        prior_current_user_texts=(surface.current_user_text, surface.current_user_text),
        prior_current_user_families=("self_confidence", "self_confidence"),
    )
    assert oversized["status"] == "FAIL"
    assert oversized["errors"][0]["check"] == (
        "current_user_text_duplicate_group_too_large"
    )

    third_pair = lint_generation_surface_case(
        user_id="pmv2_generation_compatibility_pilot",
        case_field="multi_source_needed",
        regime=ResourceNeedRegime.MULTI_SOURCE_NEEDED,
        family="self_confidence",
        forbidden_families=("relocation_loneliness", "sleep_disruption"),
        surface=surface,
        prior_current_user_texts=(
            "first pair",
            "first pair",
            "second pair",
            "second pair",
            surface.current_user_text,
        ),
        prior_current_user_families=(
            "self_confidence",
            "self_confidence",
            "self_confidence",
            "self_confidence",
            "self_confidence",
        ),
    )
    assert third_pair["status"] == "FAIL"
    assert third_pair["errors"][0]["check"] == (
        "maximum_counterfactual_current_text_pairs_exceeded"
    )


def test_bundle_text_diversity_allows_two_controlled_pairs_but_keeps_floor() -> None:
    rows = [
        {
            "case_field": f"case_{index}",
            "user_id": "one_user",
            "semantic_family": "family_a" if index < 2 else "family_b",
            "split": "train",
            "current_user_text": (
                "same family counterfactual"
                if index < 2
                else "second family counterfactual"
                if index in (2, 3)
                else f"distinct turn {index}"
            ),
        }
        for index in range(9)
    ]
    audit = audit_current_user_text_diversity(
        rows,
        minimum_unique_texts=MIN_UNIQUE_CURRENT_TEXTS_PER_NINE_CASE_BUNDLE,
    )
    assert audit["status"] == "PASS"
    assert audit["unique_normalized_current_user_texts"] == 7
    assert len(audit["allowed_counterfactual_duplicate_groups"]) == 2

    rows[4]["current_user_text"] = "distinct turn 5"
    rows[4]["semantic_family"] = rows[5]["semantic_family"]
    audit = audit_current_user_text_diversity(
        rows,
        minimum_unique_texts=MIN_UNIQUE_CURRENT_TEXTS_PER_NINE_CASE_BUNDLE,
    )
    assert audit["status"] == "FAIL"
    assert audit["unique_normalized_current_user_texts"] == 6


def _surface_only_pilot_bundle(
    contract: dict,
    *,
    request_hash_prefix: str = "test-surface-request",
) -> GeneratedUserBundle:
    surfaces = _surface_only_pilot_inputs(contract)
    assignments = generation_case_family_assignments(
        [str(value) for value in contract["semantic_families"]],
        list(ResourceNeedRegime),
    )
    accepted_calls: dict[str, CallResult] = {}
    accepted_messages: dict[str, list[dict[str, str]]] = {}
    for index, (case_field, regime) in enumerate(GENERATION_CASE_FIELDS):
        surface_payload = surfaces[case_field].model_dump(mode="json")
        accepted_calls[case_field] = CallResult(
            text=canonical_json(surface_payload),
            raw_response={
                "choices": [
                    {"message": {"content": canonical_json(surface_payload)}}
                ]
            },
            usage={
                "prompt_tokens": 100 + index,
                "completion_tokens": 50,
                "total_tokens": 150 + index,
            },
            latency_ms=1.0,
            request_hash=f"{request_hash_prefix}-{case_field}",
        )
        accepted_messages[case_field] = generation_case_messages(
            seed_dialogue="fixture held-out seed",
            user_id=str(contract["pilot_user_id"]),
            case_field=case_field,
            regime=regime,
            semantic_family=assignments[case_field],
            forbidden_families=[
                str(value)
                for value in contract["semantic_families"]
                if str(value) != assignments[case_field]
            ],
        )
    bundle = compile_surface_only_user_bundle(
        surfaces=surfaces,
        accepted_calls=accepted_calls,
        accepted_messages=accepted_messages,
        accepted_attempt_kinds={
            case_field: "initial" for case_field, _ in GENERATION_CASE_FIELDS
        },
        seed_dialogue="fixture held-out seed",
        user_id=str(contract["pilot_user_id"]),
        semantic_families=[
            str(value) for value in contract["semantic_families"]
        ],
        regimes=list(ResourceNeedRegime),
        generator_model=str(contract["endpoint"]["model"]),
        generator_family=contract["endpoint"].get("family"),
    )
    bundle.provenance["generation_compatibility_contract_sha256"] = contract[
        "contract_sha256"
    ]
    return bundle


def test_role_slot_compiler_guarantees_structural_bundle_without_self_reported_labels() -> None:
    regimes = list(ResourceNeedRegime)
    families = [
        "relocation_loneliness",
        "workload_burnout",
        "friendship_distance",
    ]
    draft = _role_slot_draft()
    bundle = compile_generation_draft(
        draft=draft,
        seed_dialogue="held-out seed",
        user_id="compiled_user",
        semantic_families=families,
        regimes=regimes,
    )
    assert len(bundle.cases) == 9
    assert {case.regime for case in bundle.cases} == set(regimes)
    assert {case.semantic_family for case in bundle.cases} == {
        *families,
    }
    assert all(
        case.profile_memories and case.summary_memories and case.event_memories
        for case in bundle.cases
    )
    assert all(
        len(case.profile_memories)
        == len(case.summary_memories)
        == len(case.event_memories)
        == 2
        for case in bundle.cases
    )
    for case in bundle.cases:
        topic = GENERATION_FAMILY_TOPICS[case.semantic_family].casefold()
        pools = {
            MemorySource.MP: case.profile_memories,
            MemorySource.MS: case.summary_memories,
            MemorySource.ME: case.event_memories,
        }
        for source, memories in pools.items():
            if source in set(case.needed_memory_sources):
                continue
            assert any(
                memory.item_utility == "irrelevant"
                and topic in memory.text.casefold()
                for memory in memories
            ), (
                "every non-needed source must retain a same-topic semantic decoy; "
                f"missing for {case.case_id}/{source.value}"
            )
    assert all(
        memory.created_session < case.session_index
        for case in bundle.cases
        for memory in (
            *case.profile_memories,
            *case.summary_memories,
            *case.event_memories,
        )
    )
    by_regime = {case.regime: case for case in bundle.cases}
    assert by_regime[ResourceNeedRegime.PROFILE_NEEDED].needed_memory_sources == [
        MemorySource.MP
    ]
    assert by_regime[ResourceNeedRegime.MULTI_SOURCE_NEEDED].needed_memory_sources == [
        MemorySource.MP,
        MemorySource.MS,
        MemorySource.ME,
    ]
    harmful = by_regime[ResourceNeedRegime.MEMORY_HARMFUL]
    assert harmful.needed_memory_sources == []
    assert sum(
        memory.item_utility == "harmful"
        for memory in (
            *harmful.profile_memories,
            *harmful.summary_memories,
            *harmful.event_memories,
        )
    ) == 3
    strategy_use = by_regime[ResourceNeedRegime.STRATEGY_HELPFUL]
    strategy_skip = by_regime[ResourceNeedRegime.STRATEGY_HARMFUL]
    assert strategy_use.strategy_resource_target == "use"
    assert strategy_skip.strategy_resource_target == "skip"
    assert {
        strategy_use.advice_readiness_target,
        strategy_skip.advice_readiness_target,
    } == {"listen_only", "light_suggestion"}
    assert bundle.provenance["generation_structure"] == (
        DATA_GENERATION_CONTRACT_VERSION
    )
    assert bundle.provenance["readiness_surface_protocol"] == (
        READINESS_SURFACE_PROTOCOL
    )
    for case in (strategy_use, strategy_skip):
        expected_clause = readiness_surface_clause_for_case(
            user_id=bundle.user_id, regime=case.regime
        )
        assert expected_clause
        assert case.current_user_text.endswith(expected_clause)
    assert harmful.current_user_text.count(". I used to avoid") == 1
    assert bundle.provenance["provider_memory_slots_ignored"] is True
    assert bundle.provenance["provider_coverage_rationale_ignored"] is True
    assert bundle.provenance["surface_selection"]["compiled_surface_lint"][
        "status"
    ] == "PASS"
    assert bundle.provenance["deterministic_draft_lint"]["status"] == "PASS"
    assert len(
        {
            case.session_index - memory.created_session
            for case in bundle.cases
            for memory in (
                *case.profile_memories,
                *case.summary_memories,
                *case.event_memories,
            )
        }
    ) >= 6
    messages = generation_messages(
        seed_dialogue="held-out seed",
        user_id="compiled_user",
        semantic_families=families,
        regimes=regimes,
    )
    assert "GeneratedBundleDraft" in messages[-1]["content"]
    assert "do not output this ID" in messages[-1]["content"]


def test_all_deterministic_memory_blueprints_fit_the_frozen_schema_boundary() -> None:
    report = deterministic_memory_blueprint_preflight()
    assert report["protocol"] == DETERMINISTIC_MEMORY_BLUEPRINT_PROTOCOL
    assert report["status"] == "PASS"
    assert report["checked_template_instances"] == (
        len(GENERATION_FAMILY_TOPICS) * 9
    )
    assert report["field_maximum_characters"] == (
        GENERATED_MEMORY_DRAFT_MAX_CHARS
    )
    assert report["maximum_observed_characters"] <= (
        GENERATED_MEMORY_DRAFT_MAX_CHARS
    )
    assert report["minimum_remaining_margin_characters"] >= 0

    # Regression for the first formal user's exact failure cell.  The old
    # health-routine MS decoy was 162 characters and crashed only after all
    # nine paid surface calls had succeeded.
    raw = pm_v2_data_module._semantic_decoy_source_raw(
        "health_routine_stress", MemorySource.MS
    )
    assert len(raw) <= GENERATED_MEMORY_DRAFT_MAX_CHARS


def test_visible_readiness_is_varied_and_counterbalanced_against_strategy_target() -> None:
    observations = set()
    assignments = set()
    for index in range(1, 53):
        user_id = f"pmv2_train_u{index:03d}"
        use_target = advice_readiness_target_for_case(
            user_id=user_id, regime=ResourceNeedRegime.STRATEGY_HELPFUL
        )
        skip_target = advice_readiness_target_for_case(
            user_id=user_id, regime=ResourceNeedRegime.STRATEGY_HARMFUL
        )
        assignments.add((use_target, skip_target))
        for regime in (
            ResourceNeedRegime.STRATEGY_HELPFUL,
            ResourceNeedRegime.STRATEGY_HARMFUL,
        ):
            clause = readiness_surface_clause_for_case(
                user_id=user_id, regime=regime
            )
            assert clause
            observations.add(clause)
    assert assignments == {
        ("listen_only", "light_suggestion"),
        ("light_suggestion", "listen_only"),
    }
    assert len(observations) >= 8

def test_generation_session_and_age_metadata_cannot_encode_fixed_regime() -> None:
    bundles = [
        _source_grounded_bundle(user_id=f"pmv2_train_u{index:03d}")
        for index in range(1, 10)
    ]
    split_by_user = {
        bundle.user_id: PMV2Split.TRAIN for bundle in bundles
    }
    report = validate_generation_shortcut_controls(
        bundles=bundles,
        split_by_user=split_by_user,
    )
    assert report["status"] == "PASS"
    assert report["case_ids_hide_regime"] is True
    assert report["memory_age_multiset_ignores_utility"] is True
    for counts in report["splits"]["train"].values():
        assert counts == [1] * 9

    tampered = bundles[0].model_copy(deep=True)
    case = tampered.cases[0]
    case.profile_memories[0].created_session -= 1
    with pytest.raises(ValueError, match="ages reveal item utility"):
        validate_generation_shortcut_controls(
            bundles=[tampered],
            split_by_user={tampered.user_id: PMV2Split.TRAIN},
        )


def test_paid_role_slot_surface_drift_uses_recorded_case_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _role_slot_draft().model_dump(mode="json")
    payload["ambiguous"]["current_user_text"] = payload["context_only"][
        "current_user_text"
    ]
    draft = GeneratedBundleDraft.model_validate(payload)
    paid_call = CallResult(
        text=canonical_json(payload),
        raw_response={"id": "paid-role-slot-response"},
        usage={
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "total_tokens": 300,
        },
        latency_ms=10.0,
        request_hash="paid-role-slot-request",
    )

    class FakeClient:
        def chat(self, *args, **kwargs):
            assert kwargs["response_schema"] is GeneratedBundleDraft
            return paid_call, draft

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        pm_v2_data_module, "make_client", lambda endpoint: FakeClient()
    )
    endpoint = Endpoint(
        base_url="https://api.openai.com",
        model="gpt-4o-mini",
        api_key_env="NOT_READ_BY_FAKE",
        family="openai_gpt4o",
    )
    bundle = generate_user_bundle(
        endpoint=endpoint,
        seed_dialogue="held-out seed",
        user_id="compiled_user",
        semantic_families=[
            "relocation_loneliness",
            "workload_burnout",
            "friendship_distance",
        ],
        regimes=list(ResourceNeedRegime),
        seed=7,
    )
    selection = bundle.provenance["surface_selection"]
    assert selection["provider_surface_lint"]["status"] == "FAIL"
    assert selection["compiled_surface_lint"]["status"] == "PASS"
    assert "ambiguous" in selection["fallback_cases"]
    assert bundle.provenance["provider_response"] == {
        "id": "paid-role-slot-response"
    }
    assert bundle.provenance["reported_usage"]["total_tokens"] == 300
    assert bundle.provenance["provider_memory_slots_ignored"] is True
    assert bundle.provenance["provider_coverage_rationale_ignored"] is True


def test_all_local_surface_fallbacks_fit_provider_transport_schema() -> None:
    for index, regime in enumerate(ResourceNeedRegime):
        surface = pm_v2_data_module._fallback_generation_surface(
            case_field=f"case_{index}",
            regime=regime,
            family="relocation_loneliness",
        )
        assert (
            surface.coverage_rationale
            == pm_v2_data_module.LOCAL_FALLBACK_COVERAGE_PLACEHOLDER
        )
        assert len(surface.coverage_rationale) <= 180

    # Evaluator-only rationales are compiled into GeneratedStateCase and are
    # intentionally independent from the provider transport field's size cap.
    assert len(
        pm_v2_data_module._deterministic_coverage_rationale(
            ResourceNeedRegime.CONTEXT_ONLY
        )
    ) > 180


def test_surface_lint_does_not_treat_move_forward_as_relocation() -> None:
    assert not pm_v2_data_module._family_anchor_hits(
        "I am not sure how to move forward after the disagreement.",
        "relocation_loneliness",
    )
    assert pm_v2_data_module._family_anchor_hits(
        "I moved to a new city last week.",
        "relocation_loneliness",
    )


def test_surface_lint_does_not_treat_parent_as_leaking_financial_uncertainty() -> None:
    # Regression for the V8.14 formal-generation run (pmv2_internal_test_u011/
    # multi_source_needed): financial_uncertainty's "rent" anchor is a literal
    # substring of the ordinary word "parent", so identity_transition text
    # about being "a parent" was falsely flagged as leaking financial_uncertainty.
    # A leading word-boundary must be required, while stems like "relocat"/
    # "isolat"/"judg" must still match their inflected suffixes.
    assert not pm_v2_data_module._family_anchor_hits(
        "I feel like I'm struggling to understand who I am as a parent "
        "while trying to balance everything at home.",
        "financial_uncertainty",
    )
    assert pm_v2_data_module._family_anchor_hits(
        "I am worried about paying rent this month.",
        "financial_uncertainty",
    )
    assert pm_v2_data_module._family_anchor_hits(
        "I moved to a new city last week.",
        "relocation_loneliness",
    )
    assert pm_v2_data_module._family_anchor_hits(
        "everyone is judging me",
        "social_anxiety",
    )


def test_surface_lint_does_not_treat_relationship_loss_as_grief() -> None:
    # Regression for the V8.15 formal-generation run (pmv2_internal_test_u013/
    # profile_needed): grief_adjustment's bare "loss" anchor is a real
    # standalone word (not a substring artifact), but it is a generic word
    # for any negative change, not specific to bereavement, so genuine
    # relationship_uncertainty text about a breakup ("confusion and loss
    # regarding the end of the relationship") false-positived a
    # current_leaks_other_family check. "griev" is added alongside the
    # unrelated exact word "grief" so verb forms like "grieving" still
    # anchor without needing "loss" as a fallback.
    assert not pm_v2_data_module._family_anchor_hits(
        "The user is reflecting on their feelings after a breakup with a "
        "partner, expressing confusion and loss regarding the end of the "
        "relationship.",
        "grief_adjustment",
    )
    assert pm_v2_data_module._family_anchor_hits(
        "I still feel the grief of losing my father last year.",
        "grief_adjustment",
    )
    assert pm_v2_data_module._family_anchor_hits(
        "I feel like I'm grieving the loss of normalcy since everything "
        "changed with the pandemic.",
        "grief_adjustment",
    )


def test_required_and_exclusive_anchors_cover_the_same_families() -> None:
    assert set(pm_v2_data_module.GENERATION_FAMILY_EXCLUSIVE_ANCHORS) == set(
        pm_v2_data_module.GENERATION_FAMILY_REQUIRED_ANCHORS
    )


def test_leak_check_uses_the_narrower_exclusive_anchors_not_required() -> None:
    # This is the structural fix behind the V8.13-V8.15 whack-a-mole: one
    # anchor list cannot safely serve both "does this text sufficiently
    # anchor its own family" (recall, wants breadth) and "did this text leak
    # a forbidden family" (precision, wants distinctiveness) at once.
    # workload_burnout's broad recall anchor "work" would false-positive
    # against almost any other family's text ("I need to work on my
    # confidence"), so it must not appear in the exclusive/leak set even
    # though it is a perfectly good required/recall anchor.
    assert "work" in pm_v2_data_module.GENERATION_FAMILY_REQUIRED_ANCHORS[
        "workload_burnout"
    ]
    assert "work" not in pm_v2_data_module.GENERATION_FAMILY_EXCLUSIVE_ANCHORS[
        "workload_burnout"
    ]
    assert not pm_v2_data_module._family_leak_hits(
        "I need to work on my confidence before the interview.",
        "workload_burnout",
    )
    assert pm_v2_data_module._family_leak_hits(
        "I've been doing unpaid overtime every week and I'm burnt out.",
        "workload_burnout",
    )


def test_leak_check_does_not_treat_workplace_grievance_as_grief() -> None:
    # workplace_conflict and grief_adjustment are co-rotated in a real
    # internal_test cohort (SEMANTIC_FAMILY_COHORTS_BY_SPLIT). A required
    # recall anchor broad enough to catch "grieving"/"grieved" (a bare
    # "griev" stem) also matches "grievance" (a workplace complaint, nothing
    # to do with bereavement) -- exactly the kind of collision the
    # required/exclusive split exists to prevent, verified directly here
    # rather than relying on which specific forms happen to be enumerated.
    assert not pm_v2_data_module._family_leak_hits(
        "I filed a grievance against my manager over the unfair schedule.",
        "grief_adjustment",
    )
    assert pm_v2_data_module._family_leak_hits(
        "My grandmother passed away last month.",
        "grief_adjustment",
    )


def test_families_too_generic_for_precision_have_empty_exclusive_anchors() -> None:
    # decision_paralysis and uncertain_future's entire required-anchor
    # vocabulary ("decision", "choice", "stuck", "future", "uncertain",
    # "unknown", ...) is generic enough to appear naturally in almost any
    # other family's narrative. Erring toward never flagging a leak against
    # them is safer than a false-positive block burning a paid attempt.
    assert pm_v2_data_module.GENERATION_FAMILY_EXCLUSIVE_ANCHORS[
        "decision_paralysis"
    ] == ()
    assert pm_v2_data_module.GENERATION_FAMILY_EXCLUSIVE_ANCHORS[
        "uncertain_future"
    ] == ()
    assert not pm_v2_data_module._family_leak_hits(
        "I can't decide what to do and feel stuck about the unknown future.",
        "decision_paralysis",
    )
    assert not pm_v2_data_module._family_leak_hits(
        "I can't decide what to do and feel stuck about the unknown future.",
        "uncertain_future",
    )


def test_surface_lint_accepts_natural_social_anxiety_phrasing() -> None:
    # Regression for the V8.13 formal-generation run (pmv2_train_u003/
    # multi_source_needed): real gpt-4o-mini completions wrote "meeting new
    # people" and "judging"/"judge me", which the old rigid "judged" anchor
    # rejected even though the text is accurate, on-topic social-anxiety
    # content. The anchor must match on the inflection stem, like every other
    # family in GENERATION_FAMILY_ANCHORS.
    assert pm_v2_data_module._family_anchor_hits(
        "I feel so anxious about meeting new people. It’s like everyone is "
        "judging me before I even say a word.",
        "social_anxiety",
    )
    assert pm_v2_data_module._family_anchor_hits(
        "I feel so anxious about meeting new people; I worry they'll judge me.",
        "social_anxiety",
    )


def test_generated_case_rejects_current_or_future_memory() -> None:
    case = _case().model_dump()
    case["event_memories"][0]["created_session"] = case["session_index"]
    with pytest.raises(ValidationError, match="not strictly prior"):
        GeneratedStateCase.model_validate(case)

    with pytest.raises(ValueError, match="current or a future session"):
        build_deployable_catalog_statistics(
            texts=["future memory"],
            created_sessions=[6],
            session_index=6,
        )


def test_generated_case_requires_structured_regime_memory_need() -> None:
    assert "strategy_catalog_count" not in GeneratedStateCase.model_fields
    assert "strategy_estimated_tokens" not in GeneratedStateCase.model_fields
    profile = _case(regime=ResourceNeedRegime.PROFILE_NEEDED).model_dump()
    profile["needed_memory_sources"] = []
    with pytest.raises(ValidationError, match="profile_needed requires"):
        GeneratedStateCase.model_validate(profile)

    multi = _case(regime=ResourceNeedRegime.MULTI_SOURCE_NEEDED).model_dump()
    multi["needed_memory_sources"] = [MemorySource.ME]
    with pytest.raises(ValidationError, match="two or three complementary"):
        GeneratedStateCase.model_validate(multi)


def test_private_context_is_physically_separate_from_model_state() -> None:
    case = _case()
    state = case_to_state(
        user_id="user_1",
        case=case,
        split=PMV2Split.TRAIN,
        strategy_catalog_count=12429,
        strategy_estimated_tokens=260,
        bundle_provenance={
            "generator_model": "generator-model",
            "authorized_user_context": "SHOULD_BE_DROPPED",
            "memory_items": {"MP": ["SHOULD_BE_DROPPED"]},
        },
    )
    serialized_state = canonical_json(state.model_dump(mode="json"))
    for forbidden in (
        "PRIVATE_AUTHORIZED_CONTEXT_SENTINEL",
        "PRIVATE_COVERAGE_RATIONALE_SENTINEL",
        "SHOULD_BE_DROPPED",
        "authorized_user_context",
        "coverage_rationale",
        "memory_items",
        case.profile_memories[0].text,
        '"regime"',
    ):
        assert forbidden not in serialized_state

    backend = case_to_memory_backend(state, case)
    evaluator = case_to_evaluator_context(state, case)
    assert case.profile_memories[0].text in {item.text for item in backend.items}
    assert evaluator["authorized_user_context"] == case.authorized_user_context
    assert evaluator["coverage_rationale"] == case.coverage_rationale
    assert evaluator["regime"] == case.regime.value
    assert evaluator["needed_memory_sources"] == []
    assert {row["memory_id"] for row in evaluator["memory_annotations"]} == {
        item.memory_id for item in backend.items
    }
    assert all("text" not in row for row in evaluator["memory_annotations"])


def test_evaluator_context_index_rejects_hash_map_and_future_tampering() -> None:
    case = _case()
    state = case_to_state(
        user_id="user_1",
        case=case,
        split=PMV2Split.TRAIN,
        strategy_catalog_count=12429,
        strategy_estimated_tokens=260,
    )
    row = case_to_evaluator_context(state, case)
    index = build_evaluator_context_index([row], states=[state])
    assert index.require_states([state], exact=True)[state.state_id]["regime"] == (
        ResourceNeedRegime.CONTEXT_ONLY.value
    )
    assert len(index.map_sha256) == 64

    hash_tampered = json.loads(json.dumps(row))
    hash_tampered["authorized_user_context"] = "tampered"
    with pytest.raises(RuntimeError, match="payload hash mismatch"):
        build_evaluator_context_index([hash_tampered], states=[state])

    future = json.loads(json.dumps(row))
    future["memory_annotations"][0]["created_session"] = state.session_index
    future["context_payload_sha256"] = evaluator_context_payload_sha256(future)
    with pytest.raises(RuntimeError, match="non-prior memories"):
        build_evaluator_context_index([future], states=[state])

    with pytest.raises(RuntimeError, match="not exact"):
        build_evaluator_context_index([row], states=[], require_exact=True)


def test_development_and_external_catalogs_have_golden_feature_parity() -> None:
    case = _case()
    development = case_to_state(
        user_id="user_1",
        case=case,
        split=PMV2Split.TRAIN,
        strategy_catalog_count=12429,
        strategy_estimated_tokens=260,
    )
    generated_by_source = {
        MemorySource.MP: case.profile_memories,
        MemorySource.MS: case.summary_memories,
        MemorySource.ME: case.event_memories,
    }
    memory_items = [
        MemoryItem(
            memory_id=f"mem_{index:012x}",
            source=source,
            created_session=memory.created_session,
            text=memory.text,
        )
        for index, (source, memory) in enumerate(
            (
                (source, memory)
                for source, memories in generated_by_source.items()
                for memory in memories
            ),
            1,
        )
    ]
    runtime = RuntimeState(
        state_id=development.state_id,
        card_id=development.card_id,
        user_id=development.user_id,
        split="evoemo_test",
        semantic_family=development.semantic_family,
        current_user_text=development.current_user_text,
        current_session_history=development.current_session_history,
        current_session_summary=development.current_session_summary,
        session_index=development.session_index,
        inventory={
            source: _catalog(memory_items, source, development.session_index)
            for source in MemorySource
        },
        allowed_actions=development.allowed_actions,
        provenance={"surface_form_id": development.surface_form_id},
    )
    external = runtime_to_pmv2_state(
        runtime,
        strategy_catalog_count=development.strategy_catalog_count,
        strategy_estimated_tokens=development.strategy_estimated_tokens,
    )
    fixed_policy_state = runtime_to_pmv2_state(
        runtime,
        strategy_catalog_count=development.strategy_catalog_count,
        strategy_estimated_tokens=development.strategy_estimated_tokens,
        include_step0_observation=False,
    )

    assert fixed_policy_state.step0_observation is None
    assert all(
        summary.query_similarity_mean == 0.0
        and not summary.representation_valid
        and summary.catalog_embedding == []
        for summary in fixed_policy_state.inventory.values()
    )

    for source in MemorySource:
        assert development.inventory[source].model_dump() == external.inventory[
            source
        ].model_dump()
        expected_tokens = sum(
            estimate_tokens(memory.text) for memory in generated_by_source[source]
        )
        assert development.inventory[source].estimated_tokens == expected_tokens
        assert "stale_fraction" not in type(development.inventory[source]).model_fields
        assert "conflict_fraction" not in type(development.inventory[source]).model_fields

    assert development.strategy_catalog_count == external.strategy_catalog_count == 12429
    assert development.strategy_estimated_tokens == external.strategy_estimated_tokens == 260

    builder = PMV2FeatureBuilder(
        word_features=32,
        char_features=32,
        use_precomputed_embeddings=False,
    ).fit([development, external])
    action_id = "MPMSME+R0"
    development_vector, external_vector = builder.transform(
        [(development, action_id), (external, action_id)]
    )
    assert np.allclose(development_vector, external_vector)

    leaked = external.model_dump()
    leaked["provenance"] = {"regime": "event_needed"}
    with pytest.raises(ValidationError, match="operational references only"):
        PMV2State.model_validate(leaked)


class _FakeSemanticEncoder:
    spec = FrozenSemanticEncoderSpec(
        model_id="fixture/semantic",
        revision="1" * 40,
        snapshot_tree_sha256="2" * 64,
        max_length=128,
        output_dimension=16,
        current_user_state_token_budget=48,
        session_summary_token_budget=24,
    )
    binding = SemanticEncoderBinding(
        spec_sha256=spec.digest(),
        snapshot_tree_sha256="2" * 64,
        snapshot_file_count=1,
        implementation="transformers-auto-model-cls-float32",
    )

    def encode(self, texts):
        rows = []
        for text in texts:
            vector = np.zeros(self.spec.output_dimension, dtype=float)
            for token in str(text).casefold().split():
                vector[int(sha256_text(token)[:8], 16) % len(vector)] += 1.0
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            rows.append(vector)
        return np.vstack(rows)


class _ReversibleTokenizer:
    truncation_side = "right"

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def __call__(self, text, *, add_special_tokens=False, **kwargs):
        del kwargs
        ids = []
        for token in str(text).split():
            if token not in self._token_to_id:
                token_id = 1000 + len(self._token_to_id)
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            ids.append(self._token_to_id[token])
        if add_special_tokens:
            ids = [101, *ids, 102]
        return {"input_ids": ids}

    def decode(self, ids, **kwargs):
        del kwargs
        return " ".join(
            self._id_to_token[value] for value in ids if value >= 1000
        )


class _SectionAwareFakeSemanticEncoder(_FakeSemanticEncoder):
    def __init__(self) -> None:
        self.tokenizer = _ReversibleTokenizer()
        self.encoded_batches: list[tuple[str, ...]] = []

    def encode(self, texts):
        self.encoded_batches.append(tuple(str(text) for text in texts))
        return super().encode(texts)

    def assemble_visible_dialogue_state(self, **kwargs):
        return FrozenTransformerSemanticEncoder.assemble_visible_dialogue_state(
            self, **kwargs
        )

    def tokenization_telemetry(self, texts, *, view_names=None):
        return FrozenTransformerSemanticEncoder.tokenization_telemetry(
            self, texts, view_names=view_names
        )


def test_reportable_semantic_observation_has_development_external_parity() -> None:
    encoder = _FakeSemanticEncoder()
    development = case_to_state(
        user_id="semantic_parity_user",
        case=_case(case_id="semantic_parity"),
        split=PMV2Split.TRAIN,
        strategy_catalog_count=0,
        strategy_estimated_tokens=0,
        semantic_encoder=encoder,
    )
    runtime = state_to_v1_runtime(development)
    assert all(
        runtime.inventory[source].catalog_fingerprint == [0.0] * 64
        for source in MemorySource
    )
    # The runtime projection must be reconstructible from the public audited
    # PMV2 state; excluded construction-only embeddings cannot affect it.
    public_round_trip = PMV2State.model_validate_json(
        json.dumps(development.model_dump(mode="json"))
    )
    assert state_to_v1_runtime(public_round_trip) == runtime
    external = runtime_to_pmv2_state(
        runtime,
        strategy_catalog_count=0,
        strategy_estimated_tokens=0,
        semantic_encoder=encoder,
    )

    assert np.allclose(development.text_embedding, external.text_embedding)
    assert development.provenance["semantic_observation"][
        "encoder_spec_sha256"
    ] == encoder.binding.spec_sha256
    assert all(
        development.inventory[source].query_similarity_mean
        == external.inventory[source].query_similarity_mean
        for source in MemorySource
    )
    builder = PMV2FeatureBuilder(use_precomputed_embeddings=False)
    assert np.allclose(
        builder._metadata_raw(development), builder._metadata_raw(external)
    )


def test_long_development_and_external_states_share_one_bounded_semantic_query() -> None:
    encoder = _SectionAwareFakeSemanticEncoder()
    long_history = [
        DialogueTurn(
            role="user" if index % 2 == 0 else "assistant",
            content=" ".join(
                f"history_{index}_{token}" for token in range(30)
            ),
        )
        for index in range(10)
    ]
    case = _case(case_id="long_unified_semantic_query").model_copy(
        update={
            "recent_dialogue": long_history,
            "session_summary": " ".join(
                f"summary_{index}" for index in range(80)
            ),
        }
    )
    old_unbounded_query = visible_dialogue_state_text(
        current_user_text=case.current_user_text,
        current_session_history=case.recent_dialogue,
        current_session_summary=case.session_summary,
    )
    prepared = prepare_visible_semantic_state(
        encoder,
        current_user_text=case.current_user_text,
        current_session_history=case.recent_dialogue,
        current_session_summary=case.session_summary,
    )
    assert prepared.query_text != old_unbounded_query
    assert "history_9_29" in prepared.query_text
    assert "history_0_0" not in prepared.query_text

    encoder.encoded_batches.clear()
    development = case_to_state(
        user_id="long_semantic_user",
        case=case,
        split=PMV2Split.TRAIN,
        strategy_catalog_count=0,
        strategy_estimated_tokens=0,
        semantic_encoder=encoder,
    )
    audit = development.provenance["semantic_observation"]
    assert audit["step0_semantic_query_sha256"] == audit[
        "state_embedding_query_sha256"
    ]
    assert audit["step0_semantic_query_vector_sha256"] == audit[
        "state_embedding_query_vector_sha256"
    ]
    allocation = audit["tokenization"]["section_allocation"]
    assert allocation["final_visible_state_token_count"] <= encoder.spec.max_length
    assert allocation["sections"]["recent_dialogue"]["dropped_token_count"] > 0
    assert audit["tokenization"]["views"]["visible_dialogue_state"][
        "truncated"
    ] is False
    assert not any(old_unbounded_query in batch for batch in encoder.encoded_batches)

    runtime = state_to_v1_runtime(development)
    external = runtime_to_pmv2_state(
        runtime,
        strategy_catalog_count=0,
        strategy_estimated_tokens=0,
        semantic_encoder=encoder,
    )
    external_audit = external.provenance["semantic_observation"]
    assert external_audit["state_embedding_query_sha256"] == audit[
        "state_embedding_query_sha256"
    ]
    assert external_audit["state_embedding_query_vector_sha256"] == audit[
        "state_embedding_query_vector_sha256"
    ]
    assert np.allclose(external.text_embedding, development.text_embedding)

    tampered_runtime = runtime.model_copy(
        update={
            "provenance": {
                **runtime.provenance,
                "semantic_query_sha256": "0" * 64,
            }
        }
    )
    with pytest.raises(RuntimeError, match="lineage drifts"):
        runtime_to_pmv2_state(
            tampered_runtime,
            semantic_encoder=encoder,
        )
    missing_lineage_runtime = runtime.model_copy(
        update={
            "provenance": {
                key: value
                for key, value in runtime.provenance.items()
                if not key.startswith("semantic_query")
            }
        }
    )
    with pytest.raises(RuntimeError, match="lineage drifts"):
        runtime_to_pmv2_state(
            missing_lineage_runtime,
            semantic_encoder=encoder,
        )

    tampered_state = development.model_dump(mode="json")
    tampered_state["provenance"]["semantic_observation"][
        "step0_semantic_query_sha256"
    ] = "0" * 64
    with pytest.raises(ValidationError, match="semantic queries drift"):
        PMV2State.model_validate_json(json.dumps(tampered_state))


def test_evoemo_memory_step0_and_state_embedding_share_semantic_lineage() -> None:
    encoder = _SectionAwareFakeSemanticEncoder()
    conversation = [
        {
            "role": "seeker" if index % 2 == 0 else "supporter",
            "content": " ".join(
                f"external_history_{index}_{token}" for token in range(30)
            ),
        }
        for index in range(8)
    ]
    memory_items = [
        MemoryItem(
            memory_id=f"mem_{index:012x}",
            source=source,
            created_session=1,
            text=f"Relevant {source.value} memory about workplace uncertainty.",
        )
        for index, source in enumerate(MemorySource, 1)
    ]
    runtime = make_evo_runtime_state(
        user={"id": "external_long_user"},
        topic={"idx": 1},
        conversation=conversation,
        current_user_text="I am unsure how to handle tomorrow's meeting.",
        items=memory_items,
        turn_index=4,
        condition="PM",
        semantic_encoder=encoder,
    )
    assert runtime.provenance["semantic_query_sha256"]
    assert runtime.provenance["semantic_query_vector_sha256"]
    external = runtime_to_pmv2_state(runtime, semantic_encoder=encoder)
    audit = external.provenance["semantic_observation"]
    assert runtime.provenance["semantic_query_sha256"] == audit[
        "state_embedding_query_sha256"
    ]
    assert runtime.provenance["semantic_query_vector_sha256"] == audit[
        "state_embedding_query_vector_sha256"
    ]


def test_memory_metadata_uses_retrieval_capacity_not_catalog_tail() -> None:
    base = case_to_state(
        user_id="user_tail_invariance",
        case=_case(case_id="tail_invariance"),
        split=PMV2Split.TRAIN,
        strategy_catalog_count=12429,
        strategy_estimated_tokens=260,
    )
    small_payload = base.model_dump(mode="json")
    large_payload = base.model_dump(mode="json")
    for payload, count, tokens in (
        (small_payload, 2, 200),
        (large_payload, 20, 2000),
    ):
        payload["state_id"] = f"state_tail_{count}"
        payload["card_id"] = f"card_tail_{count}"
        payload["provenance"]["backend_record_id"] = payload["card_id"]
        payload["inventory"]["MP"].update(
            {
                "available": True,
                "count": count,
                "estimated_tokens": tokens,
                "min_age_sessions": 1,
                "median_age_sessions": 3.0,
                "max_age_sessions": 5,
            }
        )
        # This test deliberately mutates the legacy inventory scale. Drop the
        # now-stale formal Step-0 snapshot so the builder recomputes it.
        payload["step0_observation"] = None
    small = PMV2State.model_validate_json(json.dumps(small_payload))
    large = PMV2State.model_validate_json(json.dumps(large_payload))
    builder = PMV2FeatureBuilder(
        word_features=32,
        char_features=32,
        use_precomputed_embeddings=False,
    )

    # Both stores expose the same two retrievable MP items at the same average
    # size. Eighteen additional tail items must not change model-visible metadata,
    # action features, or the deployable cost estimate.
    assert np.allclose(builder._metadata_raw(small), builder._metadata_raw(large))
    assert np.allclose(
        builder._action_raw(small, "MP+R0"),
        builder._action_raw(large, "MP+R0"),
    )
    assert builder.estimate_action_cost(small, "MP+R0") == pytest.approx(224.0)
    assert builder.estimate_action_cost(large, "MP+R0") == pytest.approx(224.0)


def test_required_hit_preflight_runs_before_outcomes_and_fails_closed() -> None:
    case = _case(
        case_id="required_hit",
        regime=ResourceNeedRegime.EVENT_NEEDED,
    )
    state = case_to_state(
        user_id="required_hit_user",
        case=case,
        split=PMV2Split.TRAIN,
        strategy_catalog_count=1,
        strategy_estimated_tokens=40,
    )
    backend = case_to_memory_backend(state, case)
    strategy = StrategyCard(
        strategy_id="strat_aaaaaaaaaaaa",
        strategy_label="Reflection of feelings",
        retrieval_text="difficult workplace conversation and uncertainty",
        guidance_text="Reflect the uncertainty.",
        example_response="That uncertainty sounds tiring.",
        source_dialogue_id="d1",
        source_turn_index=1,
    )
    report = validate_required_hit_preflight(
        states=[state],
        cases_by_state={state.state_id: case},
        backends_by_state={state.state_id: backend},
        strategy_cards=[strategy],
        strategy_top_k=1,
        memory_min_score=0.0,
        strategy_min_score=0.0,
    )
    assert report["status"] == "PASS"
    assert report["outcome_fields_accessed"] is False
    with pytest.raises(RuntimeError, match="required-hit"):
        validate_required_hit_preflight(
            states=[state],
            cases_by_state={state.state_id: case},
            backends_by_state={state.state_id: backend},
            strategy_cards=[strategy],
            strategy_top_k=1,
            memory_min_score=1.0,
            strategy_min_score=0.0,
        )


def test_real_evoemo_inventory_scale_does_not_force_metadata_ood_fallback() -> None:
    """Use real external aggregates only as a no-label scale regression test."""

    base = case_to_state(
        user_id="synthetic_scale_reference",
        case=_case(case_id="scale_reference"),
        split=PMV2Split.TRAIN,
        strategy_catalog_count=12429,
        strategy_estimated_tokens=260,
    )
    synthetic_states: list[PMV2State] = []
    for count in (1, 2, 3):
        payload = base.model_dump(mode="json")
        payload.update(
            {
                "state_id": f"synthetic_scale_state_{count}",
                "card_id": f"synthetic_scale_card_{count}",
                "user_id": f"synthetic_scale_user_{count}",
            }
        )
        payload["provenance"]["backend_record_id"] = payload["card_id"]
        for source in ("MP", "MS", "ME"):
            payload["inventory"][source].update(
                {
                    "available": True,
                    "count": count,
                    "min_age_sessions": 1,
                    "median_age_sessions": 3.0,
                    "max_age_sessions": 6,
                    "estimated_tokens": count * (12 + 4 * count),
                }
            )
        payload["step0_observation"] = None
        synthetic_states.append(PMV2State.model_validate_json(json.dumps(payload)))

    builder = PMV2FeatureBuilder(
        word_features=32,
        char_features=32,
        use_precomputed_embeddings=False,
    ).fit(synthetic_states)

    users = load_evoemo(PROJECT_ROOT / "data/external/evo_emo.json")
    real_counts: dict[MemorySource, list[int]] = {
        source: [] for source in MemorySource
    }
    real_token_totals: list[int] = []
    reports = []
    external_scale_states: list[PMV2State] = []
    for index, user in enumerate(users):
        items, _ = build_evo_memory(user)
        session_index = max(item.created_session for item in items) + 1
        inventory = {
            source: _catalog(items, source, session_index)
            for source in MemorySource
        }
        for source, catalog in inventory.items():
            real_counts[source].append(catalog.count)
            real_token_totals.append(catalog.estimated_tokens)
        runtime = RuntimeState(
            state_id=f"real_scale_state_{index}",
            card_id=f"real_scale_card_{index}",
            user_id=f"real_scale_user_{index}",
            split="evoemo_test",
            semantic_family=base.semantic_family,
            current_user_text=base.current_user_text,
            current_session_history=base.current_session_history,
            current_session_summary=base.current_session_summary,
            session_index=session_index,
            inventory=inventory,
            allowed_actions=base.allowed_actions,
            provenance={"surface_form_id": base.surface_form_id},
        )
        external_scale_state = runtime_to_pmv2_state(
            runtime,
            strategy_catalog_count=12429,
            strategy_estimated_tokens=260,
        )
        external_scale_states.append(external_scale_state)
        reports.append(builder.ood_report(external_scale_state))

    # Pin the currently observed external scale so the regression cannot silently
    # degrade into a small synthetic fixture. No EvoEmo label enters fitting.
    assert set(real_counts[MemorySource.MP]) == {7}
    assert (min(real_counts[MemorySource.MS]), max(real_counts[MemorySource.MS])) == (
        13,
        33,
    )
    # ME items are session-internal episode chunks (see
    # _chunk_session_episodes in evoemo.py), not one item per whole
    # session, so ME's per-user item count is now much higher than MS's
    # (which is still one item per session). This records the observed
    # scale after that change; it is not a claim that the resulting
    # per-item token size now matches the training-time compiler's ME
    # items (median ~37 tokens there vs a visibly larger median here) --
    # see the module-level note on EVO_MEMORY_EPISODE_TARGET_MIN_TOKENS/
    # MAX_TOKENS in evoemo.py.
    assert (min(real_counts[MemorySource.ME]), max(real_counts[MemorySource.ME])) == (
        37,
        109,
    )
    assert max(real_token_totals) == 9948
    assert reports
    assert not any(report["severe_metadata_ood"] for report in reports)
    assert not any(report["recommendation"] == "FALLBACK" for report in reports)
    external_vectors = builder.transform(
        [(state, "MPMSME+R0") for state in external_scale_states]
    )
    assert np.all(np.isfinite(external_vectors))
    assert float(np.max(np.abs(external_vectors))) < 10.0


def test_chunk_session_episodes_enforces_a_true_hard_cap() -> None:
    """EVO_MEMORY_EPISODE_MAX_TOKENS must be a real hard cap on every
    emitted chunk, with NO exception -- including a single turn that is
    itself over the cap, which must now be split rather than left over-cap
    or merged with a neighbor.
    """
    from metacom_pm.evoemo import (
        EVO_MEMORY_EPISODE_MAX_TOKENS,
        _chunk_session_episodes,
    )
    from metacom_pm.text import estimate_tokens

    # Two turns individually small whose sum would blow past the maximum
    # (120) if naively accumulated while "still below the target minimum"
    # -- this reproduces the real bug found in the real EvoEmo corpus
    # (items observed up to 143 tokens against a 120 cap).
    turns = [
        (0, "short " * 8),
        (1, "word " * 25),
        (2, "tail " * 3),
    ]
    chunks = _chunk_session_episodes(turns)
    for _, _, text, _span in chunks:
        assert estimate_tokens(text) <= EVO_MEMORY_EPISODE_MAX_TOKENS
    assert all(span is None for *_, span in chunks)

    # A single turn already at/over the cap is now split, never left
    # over-cap and never merged with a neighboring turn.
    oversized_turns = [(0, "small "), (1, "huge " * 200)]
    oversized_chunks = _chunk_session_episodes(oversized_turns)
    for start, end, text, span in oversized_chunks:
        assert estimate_tokens(text) <= EVO_MEMORY_EPISODE_MAX_TOKENS
        if start == 0 and end == 0:
            assert span is None
            assert text.strip() == "small"
        else:
            assert start == 1 and end == 1
            assert span is not None
            assert "small" not in text

    # Real-corpus regression: NOTHING may exceed the cap anymore, single
    # turn or not.
    users = load_evoemo(PROJECT_ROOT / "data/external/evo_emo.json")
    over_cap = 0
    for user in users:
        for session in user.get("dialog_history") or []:
            dialogue = session.get("dialogue") or []
            seeker_turns_with_index = [
                (i, normalize_space(t.get("content") or ""))
                for i, t in enumerate(dialogue)
                if t.get("role") == "seeker" and normalize_space(t.get("content") or "")
            ]
            for _start, _end, text, _span in _chunk_session_episodes(
                seeker_turns_with_index
            ):
                if estimate_tokens(text) > EVO_MEMORY_EPISODE_MAX_TOKENS:
                    over_cap += 1
    assert over_cap == 0


def test_chunk_session_episodes_splits_an_oversized_turn_losslessly() -> None:
    """A turn that itself exceeds the cap must be split into pieces that
    are, together, lossless (concatenation reproduces the original text
    exactly), non-overlapping, order-preserving, and each uniquely
    identified -- across an entire session's output, including when
    multiple turns each need splitting.
    """
    from metacom_pm.evoemo import _chunk_session_episodes
    from metacom_pm.text import estimate_tokens

    long_turn_a = " ".join(
        f"This is sentence number {i} of a very long turn that goes on and on."
        for i in range(20)
    )
    # A run-on "sentence" with no terminal punctuation at all, forcing the
    # word-boundary fallback rather than sentence-boundary splitting.
    long_turn_b = " ".join(f"word{i}" for i in range(200))
    # A single pathological "word" with no spaces, forcing the raw
    # bounded-character-span fallback.
    long_turn_c = "x" * 1000

    turns = [
        (0, "short intro"),
        (1, long_turn_a),
        (2, "short middle"),
        (3, long_turn_b),
        (4, long_turn_c),
        (5, "short outro"),
    ]
    max_tokens = 80
    chunks = _chunk_session_episodes(turns, max_tokens=max_tokens)

    for _start, _end, text, _span in chunks:
        assert estimate_tokens(text) <= max_tokens

    # Reconstruct each turn's own pieces (in emitted order, which must
    # match source order) and compare against the original text. Every
    # turn here is already normalize_space-d (single spaces between
    # tokens), so space-joining its pieces is lossless for the
    # sentence/word-boundary splits (turns 0, 1, 2, 3, 5); the single
    # pathological no-space "word" (turn 4) instead needs a raw join since
    # its pieces are exact character slices with no separator between
    # them.
    for turn_index, original_text in turns:
        pieces = [
            text for start, end, text, _span in chunks if start == turn_index == end
        ]
        assert pieces, f"turn {turn_index} produced no chunks"
        if turn_index == 4:
            assert "".join(pieces) == original_text
        else:
            assert " ".join(pieces) == original_text

    # No overlap / no loss, checked structurally: every chunk belonging to
    # a split turn, concatenated in emission order, must equal that turn's
    # original text (word-joined for sentence/word splits, raw-joined for
    # the character-span fallback) -- already checked above per turn.
    # Order preserved: chunks for a given turn appear with strictly
    # increasing span_index, and turns overall appear in non-decreasing
    # turn-index order.
    seen_turn_order = [start for start, _end, _text, _span in chunks]
    assert seen_turn_order == sorted(seen_turn_order)
    for turn_index, _ in turns:
        span_sequence = [
            span
            for start, end, _text, span in chunks
            if start == turn_index == end and span is not None
        ]
        assert span_sequence == list(range(len(span_sequence)))

    # ID uniqueness: build the same ids build_evo_memory would, across the
    # whole session's output, and confirm no collisions.
    ids = []
    for start, end, _text, span in chunks:
        id_key = f"session_x_turns_{start}_{end}"
        if span is not None:
            id_key = f"{id_key}_span{span}"
        ids.append(id_key)
    assert len(ids) == len(set(ids))


def test_evo_memory_builder_contract_hash_covers_every_algorithmic_component(
    monkeypatch,
) -> None:
    """The contract hash must change whenever ANY component that affects
    build_evo_memory's exact output changes -- not just the two threshold
    constants -- so a future change to the chunking algorithm, the
    long-turn split protocol, the id scheme, normalization, or the
    token-estimator protocol can never silently keep the old hash.
    """
    import metacom_pm.evoemo as evoemo_module

    baseline = evoemo_module.evo_memory_builder_contract_hash()

    tag_attrs = [
        "EVO_MEMORY_PROTOCOL",
        "EVO_MEMORY_CHUNKING_ALGORITHM_TAG",
        "EVO_MEMORY_LONG_TURN_SPLIT_PROTOCOL_TAG",
        "EVO_MEMORY_ID_SCHEME_TAG",
        "EVO_MEMORY_NORMALIZATION_TAG",
        "EVO_MEMORY_TOKEN_ESTIMATOR_PROTOCOL_TAG",
        "EVO_MEMORY_EPISODE_TARGET_MIN_TOKENS",
        "EVO_MEMORY_EPISODE_MAX_TOKENS",
    ]
    for attr in tag_attrs:
        original = getattr(evoemo_module, attr)
        perturbed = f"{original}-perturbed" if isinstance(original, str) else original + 1
        monkeypatch.setattr(evoemo_module, attr, perturbed)
        changed = evoemo_module.evo_memory_builder_contract_hash()
        assert changed != baseline, f"hash did not change when {attr} changed"
        monkeypatch.setattr(evoemo_module, attr, original)
        assert evoemo_module.evo_memory_builder_contract_hash() == baseline


def test_evo_memory_global_catalog_digest_is_sorted_and_binds_user_id_count_hash() -> None:
    """The global digest must be deterministic regardless of input order
    (sorted by user_id, not by however the caller happened to list users),
    and each per-user row must bind user_id/item_count/catalog_sha256
    together rather than a bare hash keyed loosely by user_id.
    """
    from metacom_pm.evoemo import (
        evo_memory_catalog_digest,
        evo_memory_global_catalog_digest,
    )

    users = load_evoemo(PROJECT_ROOT / "data/external/evo_emo.json")
    forward = evo_memory_global_catalog_digest(users)
    reversed_order = evo_memory_global_catalog_digest(list(reversed(users)))
    assert forward == reversed_order

    user_ids = [row["user_id"] for row in forward["per_user"]]
    assert user_ids == sorted(user_ids)
    assert len(set(user_ids)) == len(users)

    # Every row's bound item_count/catalog_sha256 must match what the
    # per-user digest independently computes for that same user -- not
    # just be internally self-consistent.
    by_id = {str(user["id"]): user for user in users}
    for row in forward["per_user"]:
        expected = evo_memory_catalog_digest(by_id[row["user_id"]])
        assert row["item_count"] == expected["item_count"]
        assert row["catalog_sha256"] == expected["catalog_sha256"]

    assert forward["user_count"] == len(users)
    assert forward["global_catalog_sha256"]


def test_split_manifest_rejects_same_split_duplicate_current_text() -> None:
    first = case_to_state(
        user_id="duplicate_text_user_a",
        case=_case(
            case_id="duplicate_text_a",
            current_user_text="I feel uncertain about the meeting.",
        ),
        split=PMV2Split.TRAIN,
        strategy_catalog_count=12429,
        strategy_estimated_tokens=260,
    )
    second = case_to_state(
        user_id="duplicate_text_user_b",
        case=_case(
            case_id="duplicate_text_b",
            current_user_text="  I FEEL uncertain   about the meeting. ",
        ),
        split=PMV2Split.TRAIN,
        strategy_catalog_count=12429,
        strategy_estimated_tokens=260,
    )
    split_states = {
        PMV2Split.TRAIN: [first, second],
        PMV2Split.CALIBRATION: [],
        PMV2Split.INTERNAL_TEST: [],
    }
    with pytest.raises(ValueError, match="same-user, same-family, same-split"):
        validate_split_manifests(split_states)

    second.current_user_text = "This is a genuinely different current turn."
    manifest = validate_split_manifests(split_states)
    assert manifest.total_states == 2
    assert manifest.unique_normalized_current_user_texts == 2
    assert manifest.normalized_current_user_text_unique_rate == 1.0

    second.current_user_text = first.current_user_text
    second.user_id = first.user_id
    second.semantic_family = first.semantic_family
    manifest = validate_split_manifests(split_states)
    assert manifest.unique_normalized_current_user_texts == 1
    assert manifest.normalized_current_user_text_unique_rate == 0.5


def test_cross_split_near_duplicate_audit_rejects_paraphrase_like_texts() -> None:
    states = []
    specifications = [
        (
            PMV2Split.TRAIN,
            "near_train",
            "family_train",
            "I feel overwhelmed by work deadlines and cannot sleep tonight.",
        ),
        (
            PMV2Split.CALIBRATION,
            "near_calibration",
            "family_calibration",
            "I feel overwhelmed by work deadlines and cannot sleep today.",
        ),
        (
            PMV2Split.INTERNAL_TEST,
            "near_internal",
            "family_internal",
            "My neighbor invited me to a relaxed weekend picnic.",
        ),
    ]
    for split, case_id, family, text in specifications:
        case = _case(case_id=case_id, current_user_text=text)
        case.semantic_family = family
        states.append(
            case_to_state(
                user_id=f"user_{case_id}",
                case=case,
                split=split,
                strategy_catalog_count=12429,
                strategy_estimated_tokens=260,
            )
        )
    by_split = {
        split: [state for state in states if state.split == split]
        for split in (
            PMV2Split.TRAIN,
            PMV2Split.CALIBRATION,
            PMV2Split.INTERNAL_TEST,
        )
    }
    validate_split_manifests(by_split)
    with pytest.raises(ValueError, match="near-duplicate audit failed"):
        audit_cross_split_near_duplicates(
            by_split,
            maximum_word_hash_cosine=0.70,
            maximum_char_hash_cosine=0.80,
            report_top_pairs=5,
        )

    states[1].current_user_text = (
        "After moving cities, I am unsure how to make new friends."
    )
    report = audit_cross_split_near_duplicates(
        by_split,
        maximum_word_hash_cosine=0.92,
        maximum_char_hash_cosine=0.95,
        report_top_pairs=5,
    )
    assert report["status"] == "PASS"
    assert report["cross_split_pair_count"] == 3


def test_bundle_must_cover_every_frozen_assigned_family_and_report_union(
    tmp_path: Path,
) -> None:
    script = PROJECT_ROOT / "scripts/20_generate_pm_v2_development_data.py"
    spec = importlib.util.spec_from_file_location("pm_v2_generator_contract", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    families = [
        "relocation_loneliness",
        "workload_burnout",
        "friendship_distance",
    ]
    incomplete = _source_grounded_bundle(user_id="family_incomplete")
    for case in incomplete.cases:
        if case.semantic_family == "friendship_distance":
            case.semantic_family = "workload_burnout"
    with pytest.raises(ValueError, match="exactly cover.*friendship_distance"):
        module.strict_bundle_check(
            incomplete, families
        )

    complete = _source_grounded_bundle(user_id="family_complete")
    module.strict_bundle_check(complete, families)
    report = write_development_dataset(
        bundles=[complete],
        split_by_user={complete.user_id: PMV2Split.TRAIN},
        out_dir=tmp_path / "complete",
        strategy_catalog_count=12429,
        strategy_estimated_tokens=260,
        strategy_top_k=3,
        strategy_bank_sha256="a" * 64,
        expected_semantic_families_by_split={
            PMV2Split.TRAIN: families,
            PMV2Split.CALIBRATION: [],
            PMV2Split.INTERNAL_TEST: [],
        },
    )
    coverage = report["semantic_family_coverage"]
    assert coverage["status"] == "PASS"
    assert coverage["splits"]["train"]["observed_count"] == 3
    assert report["split_manifest"][
        "normalized_current_user_text_unique_rate"
    ] == 1.0
    with pytest.raises(ValueError, match="semantic-family union"):
        write_development_dataset(
            bundles=[complete],
            split_by_user={complete.user_id: PMV2Split.TRAIN},
            out_dir=tmp_path / "incomplete_union",
            strategy_catalog_count=12429,
            strategy_estimated_tokens=260,
            strategy_top_k=3,
            strategy_bank_sha256="a" * 64,
            expected_semantic_families_by_split={
                PMV2Split.TRAIN: [
        *families,
                    "family_d",
                ],
                PMV2Split.CALIBRATION: [],
                PMV2Split.INTERNAL_TEST: [],
            },
        )

    frozen_report = {
        "n_states": 468,
        "split_counts": {
            "train": 216,
            "calibration": 108,
            "internal_test": 144,
        },
        "split_manifest": {
            "total_states": 468,
            "unique_normalized_current_user_texts": 468,
            "normalized_current_user_text_unique_rate": 1.0,
        },
        "semantic_family_coverage": {
            "splits": {
                "train": {"observed_count": 14},
                "calibration": {"observed_count": 5},
                "internal_test": {"observed_count": 5},
            }
        },
    }
    design = module.enforce_full_state_design(
        frozen_report,
        train_users=24,
        calibration_users=12,
        internal_test_users=16,
    )
    assert design["expected_total_states"] == 468
    assert design["semantic_family_union_counts"] == {
        "train": 14,
        "calibration": 5,
        "internal_test": 5,
    }
    duplicate_report = json.loads(json.dumps(frozen_report))
    duplicate_report["split_manifest"]["unique_normalized_current_user_texts"] = 363
    duplicate_report["split_manifest"][
        "normalized_current_user_text_unique_rate"
    ] = 363 / 468
    with pytest.raises(RuntimeError, match="counterfactual-diversity floor"):
        module.enforce_full_state_design(
            duplicate_report,
            train_users=24,
            calibration_users=12,
            internal_test_users=16,
        )


def test_v1_5_data_generation_binds_exact_strategy_bank_and_seed_manifest() -> None:
    script = (
        PROJECT_ROOT
        / "scripts"
        / "v1_5"
        / "20_generate_pm_v2_development_data_v1_5.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pm_v1_5_frozen_strategy_binding", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = load_config(PROJECT_ROOT / "configs" / "pm_v1_5.yaml")
    bank = PROJECT_ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl"
    selected = (
        PROJECT_ROOT
        / "data"
        / "strategy"
        / "pm_v1_5_selected_seed_sources.jsonl"
    )
    cards = [StrategyCard.model_validate(row) for row in iter_jsonl(bank)]
    report = module.require_frozen_strategy_bank_binding(
        pm_config=config,
        strategy_bank_path=bank,
        selected_seed_sources_path=selected,
        strategy_cards=cards,
    )
    assert report["status"] == "PASS"
    assert report["card_count"] == 11590
    assert report["source_dialogue_count"] == 823
    assert report["selected_seed_source_count"] == 52

    tampered = json.loads(json.dumps(config))
    tampered["strategy_bank_contract"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="Strategy Bank hash mismatch"):
        module.require_frozen_strategy_bank_binding(
            pm_config=tampered,
            strategy_bank_path=bank,
            selected_seed_sources_path=selected,
            strategy_cards=cards,
        )


def test_v1_5_formal_identity_content_binds_exact_pilot_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = (
        PROJECT_ROOT
        / "scripts"
        / "v1_5"
        / "20_generate_pm_v2_development_data_v1_5.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pm_v1_5_exact_pilot_binding", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    pilot_dir = tmp_path / "outputs" / "fresh_pilot"
    pilot_dir.mkdir(parents=True)
    attestation_path = pilot_dir / "artifact_attestation.json"
    write_json(attestation_path, {"attestation_sha256": "a" * 64})
    monkeypatch.setattr(
        module,
        "require_generation_compatibility_attestation",
        lambda *_args, **_kwargs: {"status": "PASS"},
    )
    binding, verification = module._require_exact_generation_pilot_binding(
        attestation_path,
        expected_contract={"contract_sha256": "b" * 64},
    )
    assert verification == {"status": "PASS"}
    assert binding == {
        "protocol": "pm-v1.5-exact-generation-pilot-attestation-binding-v1",
        "relative_path": "outputs/fresh_pilot/artifact_attestation.json",
        "artifact_attestation_file_sha256": sha256_file(attestation_path),
        "attestation_sha256": "a" * 64,
        "compatibility_contract_sha256": "b" * 64,
        "verification_status": "PASS",
    }
    original_file_sha256 = binding["artifact_attestation_file_sha256"]
    write_json(
        attestation_path,
        {"attestation_sha256": "a" * 64, "unexpected": True},
    )
    changed, _ = module._require_exact_generation_pilot_binding(
        attestation_path,
        expected_contract={"contract_sha256": "b" * 64},
    )
    assert changed["artifact_attestation_file_sha256"] != original_file_sha256

    outside = tmp_path.parent / "outside-pilot-attestation.json"
    write_json(outside, {"attestation_sha256": "a" * 64})
    with pytest.raises(RuntimeError, match="inside the project root"):
        module._require_exact_generation_pilot_binding(
            outside,
            expected_contract={"contract_sha256": "b" * 64},
        )


def test_saved_dry_run_mismatch_writes_field_level_diagnostic(
    tmp_path: Path,
) -> None:
    script = (
        PROJECT_ROOT
        / "scripts"
        / "v1_5"
        / "20_generate_pm_v2_development_data_v1_5.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pm_v1_5_dry_run_mismatch_diagnostic", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out_dir = tmp_path / "formal_run"
    out_dir.mkdir(parents=True)
    saved = {
        "cost_estimate_sha256": "a" * 64,
        "call_plan_sha256": "b" * 64,
        "generation_run_binding": {
            "base_generation_seed": 1234,
            "generator_config_sha256": "c" * 64,
            "code_manifest": {
                "src/metacom_pm/pm_v2_data.py": "d" * 64,
                "src/metacom_pm/api.py": "e" * 64,
            },
        },
        "expected_api_calls": 459,
    }
    write_json(out_dir / "generation_cost_estimate.json", saved)
    (out_dir / "generation_call_plan.jsonl").write_text("", encoding="utf-8")

    # An exact match must pass silently and must not write any diagnostic.
    module._require_saved_dry_run(out_dir, dict(saved))
    assert not list(out_dir.glob("generation_dry_run_mismatch_diagnostic_*.json"))

    current = json.loads(json.dumps(saved))
    current["cost_estimate_sha256"] = "f" * 64
    current["generation_run_binding"]["base_generation_seed"] = 5678
    current["generation_run_binding"]["code_manifest"][
        "src/metacom_pm/api.py"
    ] = "9" * 64
    current["expected_api_calls"] = 51

    with pytest.raises(RuntimeError, match="see .*generation_dry_run_mismatch_diagnostic"):
        module._require_saved_dry_run(out_dir, current)

    diagnostics = list(
        out_dir.glob("generation_dry_run_mismatch_diagnostic_*.json")
    )
    assert len(diagnostics) == 1
    diagnostic = read_json(diagnostics[0])
    assert diagnostic["saved_cost_estimate_sha256"] == "a" * 64
    assert diagnostic["current_cost_estimate_sha256"] == "f" * 64
    assert set(diagnostic["differing_field_paths"]) == {
        "generation_run_binding.base_generation_seed",
        "generation_run_binding.code_manifest.src/metacom_pm/api.py",
        "expected_api_calls",
        "cost_estimate_sha256",
    }


def test_generation_case_messages_duplicate_repair_lock_is_opt_in_and_exact() -> None:
    """forbidden_exact_texts is for the narrow duplicate-specific bounded
    repair case: default (empty) must not change the prompt at all -- so
    every existing generation_pilot_attestation binding (keyed to
    surface_generation_contract_hash()) stays valid -- and a non-empty
    value must name the forbidden sentence(s) verbatim without touching
    anything else about the case (family, regime, topic lock).
    """
    kwargs = dict(
        seed_dialogue="A held-out seed dialogue.",
        user_id="pmv2_calibration_u012",
        case_field="ambiguous",
        regime=ResourceNeedRegime.AMBIGUOUS,
        semantic_family="self_confidence",
        forbidden_families=["relocation_loneliness", "sleep_disruption"],
        repair=False,
    )
    baseline = generation_case_messages(**kwargs)
    default_explicit = generation_case_messages(**kwargs, forbidden_exact_texts=())
    assert baseline == default_explicit
    assert "DUPLICATE REPAIR LOCK" not in baseline[1]["content"]
    assert surface_generation_contract_hash() == surface_generation_contract_hash()

    forbidden_text = "I've been feeling really insecure about my abilities lately."
    locked = generation_case_messages(
        **kwargs, forbidden_exact_texts=[forbidden_text]
    )
    assert locked[0] == baseline[0]
    locked_user_content = locked[1]["content"]
    assert "DUPLICATE REPAIR LOCK" in locked_user_content
    assert f'"{forbidden_text}"' in locked_user_content
    assert "synthetic nonce" in locked_user_content
    # Removing only the lock section recovers exactly the baseline prompt.
    lock_start = locked_user_content.index("\nDUPLICATE REPAIR LOCK")
    lock_end = locked_user_content.index("\nSURFACE RULES")
    stripped = locked_user_content[:lock_start] + locked_user_content[lock_end:]
    assert stripped == baseline[1]["content"]


def test_carry_forward_recompiles_and_rebinds_an_older_directory_bundle(
    tmp_path: Path,
) -> None:
    script = (
        PROJECT_ROOT
        / "scripts"
        / "v1_5"
        / "20_generate_pm_v2_development_data_v1_5.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pm_v1_5_carry_forward", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    user_id = "pmv2_generation_compatibility_pilot"
    families = ["relocation_loneliness", "self_confidence", "sleep_disruption"]
    contract = {"semantic_families": families, "pilot_user_id": user_id}
    surfaces = _surface_only_pilot_inputs(contract)

    old_dir = tmp_path / "old_run"
    old_dir.mkdir()
    ledger_path = old_dir / "_generation_physical_attempt_ledger.jsonl"
    rows = []
    for case_field, _ in GENERATION_CASE_FIELDS:
        payload = surfaces[case_field].model_dump(mode="json")
        rows.append(
            {
                "event": "SUCCEEDED",
                "record_ids": {"user_id": user_id, "case_field": case_field},
                "result": {
                    "surface": payload,
                    "provider_response": {
                        "choices": [
                            {"message": {"content": canonical_json(payload)}}
                        ]
                    },
                },
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
                "request_hash": f"old-run-request-{case_field}",
            }
        )
    with ledger_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(canonical_json(row) + "\n")

    endpoint = type("Endpoint", (), {"model": "gpt-4o-mini", "family": "openai_gpt4o"})()
    bundle, outcome = module._carry_forward_bundle_from_old_directory(
        old_out_dirs=[old_dir],
        user_id=user_id,
        families=families,
        seed_dialogue="fixture held-out seed",
        seed_dialogue_source_id="fixture-seed-1",
        endpoint=endpoint,
        generation_binding={"protocol": "test-binding"},
    )
    assert isinstance(bundle, GeneratedUserBundle), outcome
    assert bundle.user_id == user_id
    assert len(bundle.cases) == 9
    carried_from = bundle.provenance["carried_forward_from"]
    assert carried_from["source_output_directories"] == [str(old_dir)]
    assert set(carried_from["case_sources"].values()) == {str(old_dir)}
    assert carried_from["source_physical_attempt_ledger_sha256_by_directory"][
        str(old_dir)
    ] == sha256_file(ledger_path)
    require_bundle_generation_binding(bundle, {"protocol": "test-binding"})

    # The ledger has not moved since carry-forward: unchanged.
    assert sha256_file(ledger_path) == carried_from[
        "source_physical_attempt_ledger_sha256_by_directory"
    ][str(old_dir)]

    # Missing even one case's successful surface means the whole user is
    # not carried forward (all-or-nothing), and the reason names the case.
    with ledger_path.open("w", encoding="utf-8") as f:
        for row in rows[:-1]:
            f.write(canonical_json(row) + "\n")
    missing_bundle, missing_outcome = module._carry_forward_bundle_from_old_directory(
        old_out_dirs=[old_dir],
        user_id=user_id,
        families=families,
        seed_dialogue="fixture held-out seed",
        seed_dialogue_source_id="fixture-seed-1",
        endpoint=endpoint,
        generation_binding={"protocol": "test-binding"},
    )
    assert missing_bundle is None
    assert "no carried-forward surface available" in missing_outcome


def test_carry_forward_falls_back_across_multiple_source_directories(
    tmp_path: Path,
) -> None:
    """A later directory that only regenerated a handful of users (e.g.
    after excluding a collision) must be preferred per-case, falling back
    to an earlier, full-cohort directory for cases it never touched --
    the real situation after the V8.16 dedup-repair attempt, where
    pmv2_train_u007 and the internal_test users were regenerated fresh in
    a NEW directory while the other 46 users' real content still lives
    only in the ORIGINAL v8.15 directory.
    """
    script = (
        PROJECT_ROOT
        / "scripts"
        / "v1_5"
        / "20_generate_pm_v2_development_data_v1_5.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pm_v1_5_carry_forward_multi_source", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    user_id = "pmv2_generation_compatibility_pilot"
    families = ["relocation_loneliness", "self_confidence", "sleep_disruption"]
    contract = {"semantic_families": families, "pilot_user_id": user_id}
    surfaces = _surface_only_pilot_inputs(contract)

    def _rows(case_fields: list[str], tag: str) -> list[dict[str, object]]:
        rows = []
        for case_field in case_fields:
            payload = surfaces[case_field].model_dump(mode="json")
            rows.append(
                {
                    "event": "SUCCEEDED",
                    "record_ids": {"user_id": user_id, "case_field": case_field},
                    "result": {
                        "surface": payload,
                        "provider_response": {
                            "choices": [
                                {"message": {"content": canonical_json(payload)}}
                            ]
                        },
                    },
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                    "request_hash": f"{tag}-request-{case_field}",
                }
            )
        return rows

    all_case_fields = [field for field, _ in GENERATION_CASE_FIELDS]
    newer_fields = all_case_fields[:3]
    older_fields = all_case_fields  # the full, original cohort directory

    newer_dir = tmp_path / "newer_run"
    newer_dir.mkdir()
    newer_ledger_path = newer_dir / "_generation_physical_attempt_ledger.jsonl"
    with newer_ledger_path.open("w", encoding="utf-8") as f:
        for row in _rows(newer_fields, "newer"):
            f.write(canonical_json(row) + "\n")

    older_dir = tmp_path / "older_run"
    older_dir.mkdir()
    older_ledger_path = older_dir / "_generation_physical_attempt_ledger.jsonl"
    with older_ledger_path.open("w", encoding="utf-8") as f:
        for row in _rows(older_fields, "older"):
            f.write(canonical_json(row) + "\n")

    endpoint = type("Endpoint", (), {"model": "gpt-4o-mini", "family": "openai_gpt4o"})()
    bundle, outcome = module._carry_forward_bundle_from_old_directory(
        old_out_dirs=[newer_dir, older_dir],
        user_id=user_id,
        families=families,
        seed_dialogue="fixture held-out seed",
        seed_dialogue_source_id="fixture-seed-1",
        endpoint=endpoint,
        generation_binding={"protocol": "test-binding"},
    )
    assert isinstance(bundle, GeneratedUserBundle), outcome
    carried_from = bundle.provenance["carried_forward_from"]
    case_sources = carried_from["case_sources"]
    for case_field in newer_fields:
        assert case_sources[case_field] == str(newer_dir)
    for case_field in all_case_fields:
        if case_field not in newer_fields:
            assert case_sources[case_field] == str(older_dir)
    assert set(carried_from["source_output_directories"]) == {
        str(newer_dir),
        str(older_dir),
    }
    assert carried_from["source_physical_attempt_ledger_sha256_by_directory"] == {
        str(newer_dir): sha256_file(newer_ledger_path),
        str(older_dir): sha256_file(older_ledger_path),
    }

    # A case missing from every source directory still fails closed and
    # names the case, exactly like the single-directory path.
    incomplete_dir = tmp_path / "incomplete_run"
    incomplete_dir.mkdir()
    incomplete_ledger_path = incomplete_dir / "_generation_physical_attempt_ledger.jsonl"
    with incomplete_ledger_path.open("w", encoding="utf-8") as f:
        for row in _rows(all_case_fields[:-1], "incomplete"):
            f.write(canonical_json(row) + "\n")
    missing_bundle, missing_outcome = module._carry_forward_bundle_from_old_directory(
        old_out_dirs=[incomplete_dir],
        user_id=user_id,
        families=families,
        seed_dialogue="fixture held-out seed",
        seed_dialogue_source_id="fixture-seed-1",
        endpoint=endpoint,
        generation_binding={"protocol": "test-binding"},
    )
    assert missing_bundle is None
    assert "no carried-forward surface available" in missing_outcome


def test_build_casewise_repair_plan_derives_forbidden_text_automatically() -> None:
    """The forbidden text and its collision partner must come ONLY from the
    automatic cross-user duplicate-repair manifest and the candidate bundle
    itself -- there is no parameter through which an operator could supply
    or override either one. This directly addresses the gap the prior
    duplicate-repair-spec design had: a hand-authored JSON file could name
    any forbidden_current_user_text/duplicate_of without being checked
    against what the automatic detector actually found.
    """
    script = (
        PROJECT_ROOT
        / "scripts"
        / "v1_5"
        / "20_generate_pm_v2_development_data_v1_5.py"
    )
    spec_module = importlib.util.spec_from_file_location(
        "pm_v1_5_casewise_repair_plan", script
    )
    assert spec_module is not None and spec_module.loader is not None
    module = importlib.util.module_from_spec(spec_module)
    spec_module.loader.exec_module(module)

    colliding_text = "I've been feeling really insecure about my abilities lately."
    all_case_fields = [field for field, _ in GENERATION_CASE_FIELDS]

    def _candidate_bundle(user_id: str, sole_case: GeneratedStateCase) -> GeneratedUserBundle:
        bundle = GeneratedUserBundle(
            user_id=user_id,
            profile_summary="A privacy-safe synthetic user.",
            stable_preferences=["calm communication"],
            boundaries=["no diagnosis"],
            generator_seed_id=f"seed_{user_id}",
            cases=[
                sole_case,
                _case(
                    case_id=f"{sole_case.case_id}_b",
                    regime=ResourceNeedRegime.CONTEXT_ONLY,
                    current_user_text=f"Unrelated context text for {user_id}.",
                ),
                _case(
                    case_id=f"{sole_case.case_id}_c",
                    regime=ResourceNeedRegime.EVENT_NEEDED,
                    current_user_text=f"Unrelated event text for {user_id}.",
                ),
                _case(
                    case_id=f"{sole_case.case_id}_d",
                    regime=ResourceNeedRegime.STRATEGY_HELPFUL,
                    current_user_text=f"Unrelated strategy text for {user_id}.",
                ),
            ],
        )
        bundle.provenance["carried_forward_from"] = {
            "case_sources": {field: "data/some_source_dir" for field in all_case_fields}
        }
        return bundle

    bundle_u010 = _candidate_bundle(
        "pmv2_calibration_u010",
        _case(
            case_id="case_u010_multi",
            regime=ResourceNeedRegime.MULTI_SOURCE_NEEDED,
            current_user_text=colliding_text,
        ),
    )
    bundle_u012 = _candidate_bundle(
        "pmv2_calibration_u012",
        _case(
            case_id="case_u012_ambiguous",
            regime=ResourceNeedRegime.AMBIGUOUS,
            current_user_text=colliding_text,
        ),
    )
    candidates = {
        "pmv2_calibration_u010": bundle_u010,
        "pmv2_calibration_u012": bundle_u012,
    }
    planned_users = ["pmv2_calibration_u010", "pmv2_calibration_u012"]
    split_by_user = {user_id: PMV2Split.CALIBRATION for user_id in planned_users}
    repair_manifest = module._compute_cross_user_duplicate_repair_manifest(
        planned_users=planned_users,
        bundles=candidates,
        split_by_user=split_by_user,
    )
    assert len(repair_manifest["repair_cases"]) == 1
    assert repair_manifest["repair_cases"][0]["user_id"] == "pmv2_calibration_u012"

    plan = module._build_casewise_repair_plan(
        candidates=candidates, repair_manifest=repair_manifest
    )
    assert set(plan) == {"pmv2_calibration_u012"}
    u012_plan = plan["pmv2_calibration_u012"]
    assert u012_plan["regenerated_case_fields"] == ["ambiguous"]
    assert set(u012_plan["carried_case_fields"]) == set(all_case_fields) - {"ambiguous"}

    duplicate_repair = u012_plan["duplicate_repair"]
    assert set(duplicate_repair) == {"ambiguous"}
    entry = duplicate_repair["ambiguous"]
    assert entry["forbidden_current_user_text"] == colliding_text
    assert entry["forbidden_current_user_text_sha256"] == sha256_text(colliding_text)
    assert entry["duplicate_of"] == {
        "user_id": "pmv2_calibration_u010",
        "case_field": "multi_source_needed",
    }

    # A clean candidate set (no collision) yields an empty plan.
    clean_manifest = module._compute_cross_user_duplicate_repair_manifest(
        planned_users=["pmv2_calibration_u010"],
        bundles={"pmv2_calibration_u010": bundle_u010},
        split_by_user={"pmv2_calibration_u010": PMV2Split.CALIBRATION},
    )
    assert module._build_casewise_repair_plan(
        candidates={"pmv2_calibration_u010": bundle_u010},
        repair_manifest=clean_manifest,
    ) == {}


def test_find_cross_user_duplicate_ignores_same_user_reuse() -> None:
    """The immediate, cheap post-generation cross-user duplicate check (run
    before the expensive final validate_split_manifests pass) must flag a
    DIFFERENT user's already-accepted text but never the same user's own
    counterfactual reuse, which validate_split_manifests itself permits.
    """
    script = (
        PROJECT_ROOT
        / "scripts"
        / "v1_5"
        / "20_generate_pm_v2_development_data_v1_5.py"
    )
    spec_module = importlib.util.spec_from_file_location(
        "pm_v1_5_find_cross_user_duplicate", script
    )
    assert spec_module is not None and spec_module.loader is not None
    module = importlib.util.module_from_spec(spec_module)
    spec_module.loader.exec_module(module)

    accepted = {"normalized already accepted text": ("pmv2_train_u001", "context_only")}
    assert module._find_cross_user_duplicate(
        normalized_text="normalized already accepted text",
        user_id="pmv2_train_u002",
        accepted_normalized_texts=accepted,
    ) == ("pmv2_train_u001", "context_only")
    # Same user reusing its own text (the allowed counterfactual pattern)
    # must not be flagged.
    assert module._find_cross_user_duplicate(
        normalized_text="normalized already accepted text",
        user_id="pmv2_train_u001",
        accepted_normalized_texts=accepted,
    ) is None
    # Genuinely new text is never flagged.
    assert module._find_cross_user_duplicate(
        normalized_text="brand new unseen text",
        user_id="pmv2_train_u002",
        accepted_normalized_texts=accepted,
    ) is None


def test_seed_casewise_carry_forward_into_ledger_recovers_only_carried_fields(
    tmp_path: Path,
) -> None:
    """Casewise repair for a user with ONE colliding case (the real V8.16
    dedup-repair situation for pmv2_calibration_u012): the other 8 cases
    must be seeded into THIS run's own ledger under THIS run's own call
    keys, using the already-paid content from the source directory, while
    the flagged case is left completely untouched for real generation.
    """
    script = (
        PROJECT_ROOT
        / "scripts"
        / "v1_5"
        / "20_generate_pm_v2_development_data_v1_5.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pm_v1_5_casewise_carry_forward_seed", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    user_id = "pmv2_calibration_u012"
    families = ["relocation_loneliness", "self_confidence", "sleep_disruption"]
    contract = {"semantic_families": families, "pilot_user_id": user_id}
    surfaces = _surface_only_pilot_inputs(contract)
    all_case_fields = [field for field, _ in GENERATION_CASE_FIELDS]
    flagged_field = all_case_fields[-1]
    carried_fields = [field for field in all_case_fields if field != flagged_field]

    old_dir = tmp_path / "source_run"
    old_dir.mkdir()
    old_ledger_path = old_dir / "_generation_physical_attempt_ledger.jsonl"
    with old_ledger_path.open("w", encoding="utf-8") as f:
        for case_field in carried_fields:
            payload = surfaces[case_field].model_dump(mode="json")
            row = {
                "event": "SUCCEEDED",
                "record_ids": {"user_id": user_id, "case_field": case_field},
                "result": {
                    "surface": payload,
                    "provider_response": {
                        "choices": [
                            {"message": {"content": canonical_json(payload)}}
                        ]
                    },
                },
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
                "request_hash": f"source-request-{case_field}",
            }
            f.write(canonical_json(row) + "\n")

    def _case_plan(case_field: str) -> dict[str, object]:
        call_key = module.physical_call_key(
            stage=module.GENERATION_STAGE,
            record_ids={
                "user_id": user_id,
                "case_field": case_field,
                "attempt_kind": "initial",
                "generation_seed": 0,
            },
            prompt_sha256=f"prompt-{case_field}",
            endpoint=type(
                "Endpoint",
                (),
                {
                    "base_url": "https://example.test",
                    "model": "gpt-4o-mini",
                    "family": "openai_gpt4o",
                    "transport": "auto",
                },
            )(),
            request_parameters={},
        )
        return {
            "case_field": case_field,
            "attempts": [
                {
                    "call_key": call_key,
                    "seed": 0,
                    "prompt_sha256": f"prompt-{case_field}",
                }
            ],
        }

    all_user_attempts = {
        user_id: {"cases": [_case_plan(field) for field in all_case_fields]},
    }
    all_call_keys = [
        case_plan["attempts"][0]["call_key"]
        for case_plan in all_user_attempts[user_id]["cases"]
    ]
    ledger = PersistentAttemptLedger(
        tmp_path / "current_run_ledger.jsonl",
        stage=module.GENERATION_STAGE,
        expected_calls={call_key: 3 for call_key in all_call_keys},
        maximum_total_attempts=27,
    )
    casewise_repair_plan = {
        user_id: {
            "carried_case_fields": carried_fields,
            "regenerated_case_fields": [flagged_field],
            "carried_case_sources": {
                field: str(old_dir) for field in carried_fields
            },
        }
    }

    provenance = module._seed_casewise_carry_forward_into_ledger(
        ledger=ledger,
        all_user_attempts=all_user_attempts,
        casewise_repair_plan=casewise_repair_plan,
    )
    assert provenance[user_id]["regenerated_case_fields"] == [flagged_field]
    assert set(provenance[user_id]["carried_case_fields"]) == set(carried_fields)
    for field in carried_fields:
        entry = provenance[user_id]["carried_case_fields"][field]
        assert entry["source_output_directory"] == str(old_dir)
        assert entry["source_physical_attempt_ledger_sha256"] == sha256_file(
            old_ledger_path
        )

    flagged_call_key = next(
        case_plan["attempts"][0]["call_key"]
        for case_plan in all_user_attempts[user_id]["cases"]
        if case_plan["case_field"] == flagged_field
    )
    for case_plan in all_user_attempts[user_id]["cases"]:
        call_key = case_plan["attempts"][0]["call_key"]
        if case_plan["case_field"] == flagged_field:
            assert not ledger.succeeded(call_key)
        else:
            assert ledger.succeeded(call_key)

    started_attempts_after_first_seed = ledger.started_attempts
    # Idempotent: re-seeding an already-seeded ledger (a repeated --dry-run
    # against a populated out_dir) must not write any new ledger rows.
    module._seed_casewise_carry_forward_into_ledger(
        ledger=ledger,
        all_user_attempts=all_user_attempts,
        casewise_repair_plan=casewise_repair_plan,
    )
    assert ledger.started_attempts == started_attempts_after_first_seed
    assert not ledger.succeeded(flagged_call_key)

    # If the source directory's content for a carried field vanishes after
    # the repair plan was computed, seeding must fail closed by name rather
    # than silently regenerating it or crashing obscurely.
    lost_field = carried_fields[0]
    with old_ledger_path.open("w", encoding="utf-8") as f:
        for case_field in carried_fields:
            if case_field == lost_field:
                continue
            payload = surfaces[case_field].model_dump(mode="json")
            row = {
                "event": "SUCCEEDED",
                "record_ids": {"user_id": user_id, "case_field": case_field},
                "result": {
                    "surface": payload,
                    "provider_response": {
                        "choices": [
                            {"message": {"content": canonical_json(payload)}}
                        ]
                    },
                },
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
                "request_hash": f"source-request-{case_field}",
            }
            f.write(canonical_json(row) + "\n")
    other_user_id = "pmv2_calibration_u099"
    other_all_user_attempts = {
        other_user_id: {
            "cases": [
                {**_case_plan(field), "case_field": field}
                for field in carried_fields
            ]
        }
    }
    for case_plan in other_all_user_attempts[other_user_id]["cases"]:
        case_plan["attempts"][0]["call_key"] = module.physical_call_key(
            stage=module.GENERATION_STAGE,
            record_ids={
                "user_id": other_user_id,
                "case_field": case_plan["case_field"],
                "attempt_kind": "initial",
                "generation_seed": 0,
            },
            prompt_sha256=f"prompt-{case_plan['case_field']}",
            endpoint=type(
                "Endpoint",
                (),
                {
                    "base_url": "https://example.test",
                    "model": "gpt-4o-mini",
                    "family": "openai_gpt4o",
                    "transport": "auto",
                },
            )(),
            request_parameters={},
        )
    other_ledger = PersistentAttemptLedger(
        tmp_path / "other_run_ledger.jsonl",
        stage=module.GENERATION_STAGE,
        expected_calls={
            case_plan["attempts"][0]["call_key"]: 3
            for case_plan in other_all_user_attempts[other_user_id]["cases"]
        },
        maximum_total_attempts=27,
    )
    with pytest.raises(RuntimeError, match=f"{other_user_id}/{lost_field}"):
        module._seed_casewise_carry_forward_into_ledger(
            ledger=other_ledger,
            all_user_attempts=other_all_user_attempts,
            casewise_repair_plan={
                other_user_id: {
                    "carried_case_fields": carried_fields,
                    "regenerated_case_fields": [flagged_field],
                    "carried_case_sources": {
                        field: str(old_dir) for field in carried_fields
                    },
                }
            },
        )


def test_real_physical_attempt_count_excludes_carried_forward_entries(
    tmp_path: Path,
) -> None:
    """V8.17's real closure had to hand-inspect record_ids to learn that
    only 1 of its "9 physical attempts" was a real paid OpenAI call and the
    other 8 were seeded carry-forward entries -- ledger.started_attempts
    conflates the two. _real_physical_attempt_count/
    _carried_forward_ledger_entry_count must split them cleanly so no
    report field silently mixes reused history with real new spend.
    """
    script = (
        PROJECT_ROOT
        / "scripts"
        / "v1_5"
        / "20_generate_pm_v2_development_data_v1_5.py"
    )
    spec_module = importlib.util.spec_from_file_location(
        "pm_v1_5_real_physical_attempt_count", script
    )
    assert spec_module is not None and spec_module.loader is not None
    module = importlib.util.module_from_spec(spec_module)
    spec_module.loader.exec_module(module)

    expected_calls = {f"call_{i}": 3 for i in range(9)}
    ledger = PersistentAttemptLedger(
        tmp_path / "ledger.jsonl",
        stage=module.GENERATION_STAGE,
        expected_calls=expected_calls,
        maximum_total_attempts=27,
    )
    # 8 carried-forward entries (matching the real V8.17 shape): seeded
    # directly as SUCCEEDED with attempt_kind="carried_forward".
    for i in range(8):
        call_key = f"call_{i}"
        reservation = ledger.reserve(
            call_key,
            record_ids={
                "user_id": "pmv2_calibration_u012",
                "case_field": f"field_{i}",
                "attempt_kind": "carried_forward",
                "generation_seed": 0,
            },
            prompt_sha256="prompt",
        )
        ledger.finish(
            reservation,
            succeeded=True,
            request_hash="carried-request",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            error=None,
            result={"surface": {}, "provider_response": {}},
        )
    assert module._real_physical_attempt_count(ledger) == 0
    assert module._carried_forward_ledger_entry_count(ledger) == 8

    # 1 genuinely real attempt: a normal "initial" attempt_kind.
    real_call_key = "call_8"
    reservation = ledger.reserve(
        real_call_key,
        record_ids={
            "user_id": "pmv2_calibration_u012",
            "case_field": "ambiguous",
            "attempt_kind": "initial",
            "generation_seed": 900000,
        },
        prompt_sha256="prompt",
    )
    ledger.finish(
        reservation,
        succeeded=True,
        request_hash="real-request",
        usage={"prompt_tokens": 1489, "completion_tokens": 237, "total_tokens": 1726},
        error=None,
        result={"surface": {}, "provider_response": {}},
    )
    assert module._real_physical_attempt_count(ledger) == 1
    assert module._carried_forward_ledger_entry_count(ledger) == 8
    # The raw, undifferentiated ledger size still counts all 9 -- kept
    # available (as generation_total_ledger_entries) but never used alone
    # for cost/budget reporting.
    assert ledger.started_attempts == 9


def test_cross_user_duplicate_repair_manifest_keeps_earliest_plan_position() -> None:
    script = (
        PROJECT_ROOT
        / "scripts"
        / "v1_5"
        / "20_generate_pm_v2_development_data_v1_5.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pm_v1_5_repair_manifest", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Regression for the real V8.16 carry-forward-fix run: two different
    # users, generated independently, produced verbatim-identical
    # current_user_text for the same semantic family within the same
    # split -- a real gpt-4o-mini coincidence, not a code or template bug.
    duplicate_text = (
        "I just feel so unmotivated lately, like I can't get myself to do "
        "anything productive."
    )
    bundle_u003 = GeneratedUserBundle(
        user_id="pmv2_train_u003",
        profile_summary="A privacy-safe synthetic user.",
        stable_preferences=["calm communication"],
        boundaries=["no diagnosis"],
        generator_seed_id="seed_u003",
        cases=[
            _case(
                case_id="case_u003_a",
                regime=ResourceNeedRegime.CONTEXT_ONLY,
                current_user_text="Unrelated first text for u003.",
            ),
            _case(
                case_id="case_u003_b",
                regime=ResourceNeedRegime.STRATEGY_HARMFUL,
                current_user_text=duplicate_text,
            ),
            _case(
                case_id="case_u003_c",
                regime=ResourceNeedRegime.EVENT_NEEDED,
                current_user_text="Unrelated third text for u003.",
            ),
            _case(
                case_id="case_u003_d",
                regime=ResourceNeedRegime.STRATEGY_HELPFUL,
                current_user_text="Unrelated fourth text for u003.",
            ),
        ],
    )
    bundle_u007 = GeneratedUserBundle(
        user_id="pmv2_train_u007",
        profile_summary="A privacy-safe synthetic user.",
        stable_preferences=["calm communication"],
        boundaries=["no diagnosis"],
        generator_seed_id="seed_u007",
        cases=[
            _case(
                case_id="case_u007_a",
                regime=ResourceNeedRegime.EVENT_NEEDED,
                current_user_text=duplicate_text,
            ),
            _case(
                case_id="case_u007_b",
                regime=ResourceNeedRegime.STRATEGY_HELPFUL,
                current_user_text="Unrelated second text for u007.",
            ),
            _case(
                case_id="case_u007_c",
                regime=ResourceNeedRegime.CONTEXT_ONLY,
                current_user_text="Unrelated third text for u007.",
            ),
            _case(
                case_id="case_u007_d",
                regime=ResourceNeedRegime.STRATEGY_HARMFUL,
                current_user_text="Unrelated fourth text for u007.",
            ),
        ],
    )
    planned_users = [
        "pmv2_train_u001",
        "pmv2_train_u002",
        "pmv2_train_u003",
        "pmv2_train_u004",
        "pmv2_train_u005",
        "pmv2_train_u006",
        "pmv2_train_u007",
    ]
    split_by_user = {user_id: PMV2Split.TRAIN for user_id in planned_users}
    manifest = module._compute_cross_user_duplicate_repair_manifest(
        planned_users=planned_users,
        bundles={"pmv2_train_u003": bundle_u003, "pmv2_train_u007": bundle_u007},
        split_by_user=split_by_user,
    )
    assert manifest["policy"] == "keep_earliest_frozen_plan_position"
    assert len(manifest["repair_cases"]) == 1
    repair = manifest["repair_cases"][0]
    # u003 appears earlier in planned_users than u007, so u007's case is
    # the one marked for regeneration, never u003's -- a deterministic
    # outcome of frozen plan order, not a human picking which user to redo.
    assert repair["user_id"] == "pmv2_train_u007"
    assert repair["case_field"] == "event_needed"
    assert repair["duplicate_of"] == {
        "user_id": "pmv2_train_u003",
        "case_field": "strategy_harmful",
    }
    assert manifest["manifest_sha256"]

    # No cross-user collision -> no repair cases, regardless of same-user
    # counterfactual reuse (which is allowed and not a leak).
    clean_manifest = module._compute_cross_user_duplicate_repair_manifest(
        planned_users=planned_users,
        bundles={"pmv2_train_u003": bundle_u003},
        split_by_user=split_by_user,
    )
    assert clean_manifest["repair_cases"] == []


def test_generation_resume_binding_rejects_legacy_and_mismatch() -> None:
    binding = {
        "generator_config_sha256": "a" * 64,
        "seed_dialogues_sha256": "b" * 64,
        "prompt_contract_sha256": "c" * 64,
        "code_manifest_sha256": "d" * 64,
    }
    bundle = _bundle()
    bind_bundle_to_generation_run(bundle, binding)
    require_bundle_generation_binding(bundle, binding)

    with pytest.raises(RuntimeError, match="different generator/seed/config/prompt/code"):
        require_bundle_generation_binding(
            bundle,
            {**binding, "seed_dialogues_sha256": "e" * 64},
        )

    legacy = _bundle()
    with pytest.raises(RuntimeError, match="predates immutable generation-run binding"):
        require_bundle_generation_binding(legacy, binding)


def test_regime_checks_reject_constant_full_resource_policy() -> None:
    assert _regime_pass("profile_needed", "MP+R0", 0.2, ["MP"])
    assert _regime_pass("summary_needed", "MS+R0", 0.2, ["MS"])
    assert _regime_pass("event_needed", "ME+R0", 0.2, ["ME"])
    for regime, needed in (
        ("profile_needed", ["MP"]),
        ("summary_needed", ["MS"]),
        ("event_needed", ["ME"]),
        ("multi_source_needed", ["MP", "ME"]),
    ):
        assert not _regime_pass(regime, "MPMSME+R0", 0.2, needed)
    assert _regime_pass(
        "multi_source_needed", "MPE+R0", 0.2, ["MP", "ME"]
    )
    assert _regime_pass("strategy_helpful", "M0+RS", 0.2, [])
    assert not _regime_pass("strategy_helpful", "MPMSME+RS", 0.2, [])


def test_explicit_train_ids_cannot_override_declared_test_split(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "dialogue_id": "test_1",
                "split": "test",
                "dialogue_text": "user: This row belongs to test.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    train_ids = tmp_path / "train_ids.txt"
    train_ids.write_text("test_1\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "18_prepare_pm_v2_seed_dialogues.py"),
            "--inputs",
            str(source),
            "--out",
            str(tmp_path / "out.jsonl"),
            "--train-ids",
            str(train_ids),
            "--minimum-seeds",
            "1",
        ],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "declared non-train source split" in result.stderr


def test_seed_output_records_hashed_lineage_and_computed_split_audit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    rows = [
        {"dialogue_id": "train_1", "split": "train", "dialogue_text": "first"},
        {"dialogue_id": "train_2", "split": "train", "dialogue_text": "second"},
        {"dialogue_id": "test_1", "split": "test", "dialogue_text": "held out"},
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    train_ids = tmp_path / "train_ids.txt"
    train_ids.write_text("train_1\ntrain_2\n", encoding="utf-8")
    excluded_ids = tmp_path / "excluded_ids.txt"
    excluded_ids.write_text("train_2\n", encoding="utf-8")
    output = tmp_path / "seeds.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "18_prepare_pm_v2_seed_dialogues.py"),
            "--inputs",
            str(source),
            "--out",
            str(output),
            "--train-ids",
            str(train_ids),
            "--excluded-ids",
            str(excluded_ids),
            "--minimum-seeds",
            "1",
        ],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=True,
    )
    audit = json.loads(
        output.with_suffix(output.suffix + ".audit.json").read_text(encoding="utf-8")
    )
    assert audit["test_or_validation_rows_in_output"] == 0
    assert audit["declared_non_train_rows_in_output"] == 0
    hashes = audit["lineage_manifest_hashes"]
    assert {
        "code_manifest_sha256",
        "source_manifest_sha256",
        "train_manifest_sha256",
        "exclusion_manifest_sha256",
    } <= set(hashes)
    assert all(len(value) == 64 for value in hashes.values())
    assert (
        audit["seed_extraction_contract_version"]
        == "pm-v2-seed-extraction-v1-strict-lineage"
    )
    assert audit["code_manifest"]["contract_version"] == audit[
        "seed_extraction_contract_version"
    ]
    assert audit["code_manifest_sha256"] == hashes["code_manifest_sha256"]
    assert audit["code_manifest_sha256"] == sha256_text(
        canonical_json(audit["code_manifest"])
    )
    manifest_files = {
        row["path"]: row["sha256"] for row in audit["code_manifest"]["files"]
    }
    assert manifest_files == {
        "scripts/18_prepare_pm_v2_seed_dialogues.py": sha256_file(
            PROJECT_ROOT / "scripts/18_prepare_pm_v2_seed_dialogues.py"
        ),
        "src/metacom_pm/io.py": sha256_file(PROJECT_ROOT / "src/metacom_pm/io.py"),
    }
    output_rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["dialogue_id"] for row in output_rows] == ["train_1"]
    assert all(output_rows[0][key] == value for key, value in hashes.items())
    assert output_rows[0]["seed_extraction_contract_version"] == audit[
        "seed_extraction_contract_version"
    ]


def test_index_split_manifest_strictly_joins_and_excludes_overlap(
    tmp_path: Path,
) -> None:
    source = tmp_path / "esconv.json"
    source.write_text(
        json.dumps(
            [
                {"dialog": [{"speaker": "seeker", "content": "train seed"}]},
                {"dialog": [{"speaker": "seeker", "content": "test row"}]},
                {"dialog": [{"speaker": "seeker", "content": "overlap row"}]},
            ]
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "split.jsonl"
    manifest_rows = [
        {
            "index": 0,
            "dialogue_id": "esconv_0000",
            "split": "train",
            "excluded_for_evoemo_overlap": False,
        },
        {
            "index": 1,
            "dialogue_id": "esconv_0001",
            "split": "test",
            "excluded_for_evoemo_overlap": False,
        },
        {
            "index": 2,
            "dialogue_id": "esconv_0002",
            "split": "train",
            "excluded_for_evoemo_overlap": True,
        },
    ]
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    output = tmp_path / "seeds.jsonl"
    base_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "18_prepare_pm_v2_seed_dialogues.py"),
        "--inputs",
        str(source),
        "--split-manifest",
        str(manifest),
        "--out",
        str(output),
        "--minimum-seeds",
        "1",
    ]
    subprocess.run(
        base_command,
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=True,
    )
    output_rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["dialogue_id"] for row in output_rows] == ["esconv_0000"]
    assert output_rows[0]["split_manifest_index"] == 0
    audit = json.loads(
        output.with_suffix(output.suffix + ".audit.json").read_text(encoding="utf-8")
    )
    assert audit["split_manifest_join"]["status"] == "PASS"
    assert audit["split_manifest_join"]["covered_source_rows"] == 3
    assert audit["evoemo_overlap_excluded_train"] == 1
    assert audit["test_or_validation_rows_in_output"] == 0
    assert "split_join_manifest_sha256" in audit["lineage_manifest_hashes"]

    train_ids = tmp_path / "train_ids.txt"
    train_ids.write_text("esconv_0001\n", encoding="utf-8")
    override = subprocess.run(
        [*base_command, "--train-ids", str(train_ids), "--overwrite"],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert override.returncode != 0
    assert "cannot override" in override.stderr

    broken_manifest = tmp_path / "broken.jsonl"
    broken_rows = [dict(row) for row in manifest_rows]
    broken_rows[-1]["index"] = 3
    broken_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in broken_rows),
        encoding="utf-8",
    )
    broken = subprocess.run(
        [
            *base_command[: base_command.index("--split-manifest") + 1],
            str(broken_manifest),
            *base_command[base_command.index("--out") :],
            "--overwrite",
        ],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert broken.returncode != 0
    assert "continuous from zero" in broken.stderr


def test_generation_dry_run_is_keyless_and_saves_exact_accepted_plan(
    tmp_path: Path,
) -> None:
    seed_path = tmp_path / "seeds.jsonl"
    _write_unique_generation_seeds(seed_path)
    out_dir = tmp_path / "generation"
    config = PROJECT_ROOT / "configs" / "experiment.yaml"
    pm_config = PROJECT_ROOT / "configs" / "pm_v2.yaml"
    env = _subprocess_env()
    env.pop("NVIDIA_API_KEY", None)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "20_generate_pm_v2_development_data.py"),
        "--config",
        str(config),
        "--pm-v2-config",
        str(pm_config),
        "--seed-dialogues",
        str(seed_path),
        "--out-dir",
        str(out_dir),
        "--max-users",
        "1",
        "--max-api-calls",
        "18",
        "--max-estimated-usd",
        "1",
        "--max-input-tokens-per-call",
        "12000",
    ]
    subprocess.run(
        [*command, "--dry-run"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    estimate = json.loads(
        (out_dir / "generation_cost_estimate.json").read_text(encoding="utf-8")
    )
    assert estimate["expected_api_calls"] == 9
    assert estimate["maximum_api_calls"] == 18
    assert estimate["historical_physical_api_attempts"] == 0
    assert estimate["maximum_physical_api_attempts_including_history"] == 18
    assert estimate["input_token_safety_factor"] == 1.5
    assert estimate["fail_on_reported_input_overrun"] is True
    assert estimate["pricing"] == {
        "input_usd_per_mtok": 0.15,
        "output_usd_per_mtok": 0.60,
    }
    assert estimate["budget_gate"]["status"] == "PASS"
    assert len(estimate["cost_estimate_sha256"]) == 64
    strategy_catalog = estimate["generation_run_binding"]["strategy_catalog"]
    assert strategy_catalog["count"] == 12429
    assert strategy_catalog["estimated_action_tokens"] == 260
    assert strategy_catalog["top_k"] == 3
    assert len(strategy_catalog["sha256"]) == 64
    call_plan_rows = list(iter_jsonl(out_dir / "generation_call_plan.jsonl"))
    assert len(call_plan_rows) == 9
    assert all(row["remaining_attempts"] == 2 for row in call_plan_rows)
    assert all(
        row["estimated_input_tokens"] > row["raw_estimated_input_tokens"]
        for case_plan in call_plan_rows
        for row in case_plan["attempts"]
    )
    pricing_override = subprocess.run(
        [*command, "--dry-run", "--input-usd-per-mtok", "0"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert pricing_override.returncode != 0
    assert "pricing override differs" in pricing_override.stderr
    all_attempts = [
        row for case_plan in call_plan_rows for row in case_plan["attempts"]
    ]
    assert len({row["call_key"] for row in all_attempts}) == 18

    # Inject the single paid failure.  The next dry-run must fail closed rather
    # than silently grant a replacement seed.
    expected_calls = {
        row["call_key"]: 1 for row in all_attempts
    }
    ledger = PersistentAttemptLedger(
        out_dir / "_generation_physical_attempt_ledger.jsonl",
        stage="pm_v2_synthetic_surface_generation",
        expected_calls=expected_calls,
        maximum_total_attempts=18,
    )
    first = call_plan_rows[0]["attempts"][0]
    reservation = ledger.reserve(
        first["call_key"],
        record_ids={
            "user_id": call_plan_rows[0]["user_id"],
            "case_field": call_plan_rows[0]["case_field"],
            "attempt_kind": first["attempt_kind"],
            "generation_seed": first["seed"],
        },
        prompt_sha256=first["prompt_sha256"],
    )
    ledger.finish(
        reservation,
        succeeded=False,
        request_hash=None,
        usage=None,
        error="injected failure",
    )
    overwrite_blocked = subprocess.run(
        [*command, "--dry-run", "--overwrite"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert overwrite_blocked.returncode != 0
    assert "refuses --overwrite" in overwrite_blocked.stderr
    assert len(list(iter_jsonl(out_dir / "_generation_physical_attempt_ledger.jsonl"))) == 2
    repair_dry_run = subprocess.run(
        [*command, "--dry-run"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    repair_estimate = read_json(out_dir / "generation_cost_estimate.json")
    repair_plan = list(iter_jsonl(out_dir / "generation_call_plan.jsonl"))
    assert repair_estimate["historical_physical_api_attempts"] == 1
    assert repair_estimate["expected_api_calls"] == 9
    assert repair_estimate["maximum_api_calls"] == 17
    assert repair_estimate["blocked_pending_users"] == []
    repaired_case = next(
        row
        for row in repair_plan
        if row["case_field"] == call_plan_rows[0]["case_field"]
    )
    assert repaired_case["remaining_attempts"] == 1
    assert repaired_case["attempts"][0]["attempt_kind"] == "repair"

    repair = call_plan_rows[0]["attempts"][1]
    repair_reservation = ledger.reserve(
        repair["call_key"],
        record_ids={
            "user_id": call_plan_rows[0]["user_id"],
            "case_field": call_plan_rows[0]["case_field"],
            "attempt_kind": "repair",
            "generation_seed": repair["seed"],
        },
        prompt_sha256=repair["prompt_sha256"],
    )
    ledger.finish(
        repair_reservation,
        succeeded=False,
        request_hash=None,
        usage=None,
        error="injected repair failure",
    )
    exhausted = subprocess.run(
        [*command, "--dry-run"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert exhausted.returncode != 0
    exhausted_estimate = read_json(out_dir / "generation_cost_estimate.json")
    assert exhausted_estimate["historical_physical_api_attempts"] == 2
    assert exhausted_estimate["maximum_api_calls"] == 16
    assert exhausted_estimate["blocked_pending_users"] == [
        {
            "user_id": call_plan_rows[0]["user_id"],
            "case_field": call_plan_rows[0]["case_field"],
            "reason": "maximum_case_attempts_exhausted",
            "historical_attempts": 2,
        }
    ]
    assert exhausted_estimate["budget_gate"]["status"] == "FAIL"


def test_generation_compatibility_pilot_dry_run_freezes_casewise_plan(
    tmp_path: Path,
) -> None:
    seed_path = tmp_path / "seeds.jsonl"
    _write_unique_generation_seeds(seed_path)
    out_dir = tmp_path / "generation_pilot"
    env = _subprocess_env()
    env.pop("NVIDIA_API_KEY", None)
    command = [
        sys.executable,
        str(
            PROJECT_ROOT
            / "scripts"
            / "20a_run_pm_v2_generation_compatibility_pilot.py"
        ),
        "--seed-dialogues",
        str(seed_path),
        "--out-dir",
        str(out_dir),
        "--max-api-calls",
        str(GENERATION_PILOT_MAX_ATTEMPTS),
        "--max-estimated-usd",
        "1",
        "--max-input-tokens-per-call",
        "12000",
    ]
    subprocess.run(
        [*command, "--dry-run"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    estimate = read_json(out_dir / "cost_estimate.json")
    plan = list(iter_jsonl(out_dir / "call_plan.jsonl"))
    assert estimate["minimum_api_calls_if_successful"] == GENERATION_PILOT_MINIMUM_CALLS
    assert estimate["maximum_physical_api_attempts"] == GENERATION_PILOT_MAX_ATTEMPTS
    assert estimate["maximum_content_attempts"] == GENERATION_PILOT_MAX_CONTENT_ATTEMPTS
    assert estimate["maximum_transport_attempts_per_content_attempt"] == (
        GENERATION_PILOT_MAX_TRANSPORT_ATTEMPTS_PER_CONTENT_ATTEMPT
    )
    assert estimate["maximum_repairs_per_case"] == 1
    assert estimate["stop_after_each_case_success"] is True
    assert estimate["pricing"] == {
        "input_usd_per_mtok": 0.15,
        "output_usd_per_mtok": 0.60,
    }
    assert estimate["input_token_safety_factor"] == 1.5
    assert estimate["maximum_input_token_upper_bound_per_call"] > 0
    assert estimate["maximum_output_tokens_per_call"] == 900
    assert estimate["budget_gate"]["status"] == "PASS"
    assert len(plan) == GENERATION_PILOT_MAX_CONTENT_ATTEMPTS
    assert len({row["physical_call_key"] for row in plan}) == len(plan)
    assert len({row["generation_seed"] for row in plan}) == len(plan)
    assert {row["attempt_kind"] for row in plan} == {"initial", "repair"}
    assert {
        row["case_field"] for row in plan
    } == {field for field, _ in GENERATION_CASE_FIELDS}
    assert all(
        sum(candidate["case_field"] == row["case_field"] for candidate in plan)
        == 2
        for row in plan
    )
    assert all(
        row["maximum_physical_attempts"]
        == GENERATION_PILOT_MAX_TRANSPORT_ATTEMPTS_PER_CONTENT_ATTEMPT
        for row in plan
    )

    ledger_path = out_dir / "physical_attempt_ledger.jsonl"
    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=GENERATION_PILOT_STAGE,
        expected_calls={
            str(row["physical_call_key"]): int(row["maximum_physical_attempts"])
            for row in plan
        },
        maximum_total_attempts=GENERATION_PILOT_MAX_ATTEMPTS,
    )
    reservation = ledger.reserve(
        str(plan[0]["physical_call_key"]),
        record_ids={
            "user_id": plan[0]["user_id"],
            "generation_seed": plan[0]["generation_seed"],
            "case_field": plan[0]["case_field"],
            "attempt_kind": plan[0]["attempt_kind"],
        },
        prompt_sha256=str(plan[0]["prompt_sha256"]),
    )
    ledger.finish(
        reservation,
        succeeded=False,
        request_hash=None,
        usage=None,
        error="injected pilot failure",
    )
    overwrite_blocked = subprocess.run(
        [*command, "--dry-run", "--overwrite"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert overwrite_blocked.returncode != 0
    assert "refuses --overwrite" in overwrite_blocked.stderr
    assert len(list(iter_jsonl(ledger_path))) == 2

    saved_estimate_bytes = (out_dir / "cost_estimate.json").read_bytes()
    saved_plan_bytes = (out_dir / "call_plan.jsonl").read_bytes()
    exact_spent_dry_run = subprocess.run(
        [*command, "--dry-run"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert exact_spent_dry_run.returncode == 0
    assert (out_dir / "cost_estimate.json").read_bytes() == saved_estimate_bytes
    assert (out_dir / "call_plan.jsonl").read_bytes() == saved_plan_bytes
    stale_spent_dry_run = subprocess.run(
        [*command, "--dry-run", "--max-estimated-usd", "2"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert stale_spent_dry_run.returncode != 0
    assert "spent pilot ledger freezes" in stale_spent_dry_run.stderr
    assert (out_dir / "cost_estimate.json").read_bytes() == saved_estimate_bytes
    assert (out_dir / "call_plan.jsonl").read_bytes() == saved_plan_bytes

    dry_run_again = subprocess.run(
        [*command, "--dry-run", "--input-usd-per-mtok", "0"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert dry_run_again.returncode != 0
    assert "input pricing override differs" in dry_run_again.stderr


def test_v1_5_generation_pilot_budget_is_config_frozen_and_reproducible(
    tmp_path: Path,
) -> None:
    env = _subprocess_env()
    env.pop("OPENAI_API_KEY", None)
    # A clean checkout intentionally has no ignored development seed corpus.
    # Keep this budget-contract test hermetic by supplying the minimum unique
    # synthetic cohort required for 52 users plus one held-out seed.
    seed_path = tmp_path / "generation_seeds.jsonl"
    _write_unique_generation_seeds(seed_path)
    script = (
        PROJECT_ROOT
        / "scripts"
        / "v1_5"
        / "20a_run_generation_compatibility_pilot_v1_5.py"
    )
    estimates = []
    for suffix in ("first", "second"):
        out_dir = tmp_path / suffix
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--dry-run",
                "--seed-dialogues",
                str(seed_path),
                "--out-dir",
                str(out_dir),
            ],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        estimates.append(read_json(out_dir / "cost_estimate.json"))
    assert estimates[0]["budget_limits"] == {
        "max_api_calls": GENERATION_PILOT_MAX_ATTEMPTS,
        "max_estimated_usd": 0.054,
        "max_input_tokens_per_call": 4000,
    }
    assert (
        estimates[0]["cost_estimate_sha256"]
        == estimates[1]["cost_estimate_sha256"]
    )

    rejected = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--seed-dialogues",
            str(seed_path),
            "--out-dir",
            str(tmp_path / "override"),
            "--max-estimated-usd",
            "2.0",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "override differs from frozen generation-pilot budget" in rejected.stderr


def test_generation_pilot_recovers_post_success_crash_without_second_http(
    tmp_path: Path, monkeypatch
) -> None:
    script = (
        PROJECT_ROOT
        / "scripts"
        / "20a_run_pm_v2_generation_compatibility_pilot.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pm_v2_generation_pilot_recovery", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    seed_path = tmp_path / "seeds.jsonl"
    _write_unique_generation_seeds(seed_path)
    out_dir = tmp_path / "pilot"
    common = [
        str(script),
        "--seed-dialogues",
        str(seed_path),
        "--out-dir",
        str(out_dir),
        "--max-api-calls",
        str(GENERATION_PILOT_MAX_ATTEMPTS),
        "--max-estimated-usd",
        "1",
        "--max-input-tokens-per-call",
        "12000",
    ]
    monkeypatch.setattr(sys, "argv", [*common, "--dry-run"])
    module.main()

    experiment_config_path = PROJECT_ROOT / "configs" / "experiment.yaml"
    pm_v2_config_path = PROJECT_ROOT / "configs" / "pm_v2.yaml"
    experiment_config = load_config(experiment_config_path)
    pm_v2_config = load_config(pm_v2_config_path)
    generation_cfg = pm_v2_config["data_generation"]
    endpoint = endpoint_from_config(
        experiment_config, str(generation_cfg["generator_endpoint"])
    )
    full_user_count = sum(
        int(generation_cfg[key])
        for key in ("train_users", "calibration_users", "internal_test_users")
    )
    contract = build_generation_compatibility_contract(
        project_root=PROJECT_ROOT,
        experiment_config_path=experiment_config_path,
        pm_v2_config_path=pm_v2_config_path,
        seed_dialogues_path=seed_path,
        endpoint=endpoint,
        base_generation_seed=int(generation_cfg["base_seed"]),
        full_user_count=full_user_count,
        input_token_safety_factor=1.5,
        fail_on_reported_input_overrun=True,
        input_usd_per_mtok=0.15,
        output_usd_per_mtok=0.60,
    )
    surfaces = _surface_only_pilot_inputs(contract)

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def chat(self, messages, **kwargs):
            prompt = messages[1]["content"]
            case_field = next(
                field
                for field, _ in GENERATION_CASE_FIELDS
                if f"CASE SLOT (never mention this label): {field}" in prompt
            )
            self.calls.append(case_field)
            surface = surfaces[case_field]
            payload = surface.model_dump(mode="json")
            return (
                CallResult(
                    text=canonical_json(payload),
                    raw_response={
                        "choices": [
                            {"message": {"content": canonical_json(payload)}}
                        ]
                    },
                    usage={
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                    latency_ms=1.0,
                    request_hash=f"fake-{case_field}",
                ),
                surface,
            )

        def close(self) -> None:
            return None

    fake = FakeClient()
    monkeypatch.setattr(module, "make_client", lambda endpoint: fake)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    accepted_hash = read_json(out_dir / "cost_estimate.json")[
        "cost_estimate_sha256"
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *common,
            "--run",
            "--accept-cost-estimate-sha256",
            accepted_hash,
        ],
    )
    module.main()
    assert fake.calls == [field for field, _ in GENERATION_CASE_FIELDS]
    assert read_json(out_dir / "summary.json")["status"] == "PASS"
    assert (out_dir / "artifact_attestation.json").is_file()
    ledger_rows = list(
        iter_jsonl(out_dir / "physical_attempt_ledger.jsonl")
    )
    assert len(ledger_rows) == 2 * GENERATION_PILOT_MINIMUM_CALLS
    assert [row["event"] for row in ledger_rows] == [
        event
        for _ in GENERATION_CASE_FIELDS
        for event in ("STARTED", "SUCCEEDED")
    ]

    def forbidden_client(endpoint):
        raise AssertionError("completed pilot recovery must not create a client")

    monkeypatch.setattr(module, "make_client", forbidden_client)
    module.main()


def test_generation_pilot_uses_one_bounded_repair_and_never_repeats_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = (
        PROJECT_ROOT
        / "scripts"
        / "20a_run_pm_v2_generation_compatibility_pilot.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pm_v2_generation_pilot_retry", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    seed_path = tmp_path / "seeds.jsonl"
    _write_unique_generation_seeds(seed_path)
    out_dir = tmp_path / "pilot"
    common = [
        str(script),
        "--seed-dialogues",
        str(seed_path),
        "--out-dir",
        str(out_dir),
        "--max-api-calls",
        str(GENERATION_PILOT_MAX_ATTEMPTS),
        "--max-estimated-usd",
        "1",
        "--max-input-tokens-per-call",
        "12000",
    ]
    monkeypatch.setattr(sys, "argv", [*common, "--dry-run"])
    module.main()
    accepted_hash = read_json(out_dir / "cost_estimate.json")[
        "cost_estimate_sha256"
    ]
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    experiment = load_config(PROJECT_ROOT / "configs" / "experiment.yaml")
    pm_config = load_config(PROJECT_ROOT / "configs" / "pm_v2.yaml")
    generation = pm_config["data_generation"]
    endpoint = endpoint_from_config(
        experiment, generation["generator_endpoint"]
    )
    contract = build_generation_compatibility_contract(
        project_root=PROJECT_ROOT,
        experiment_config_path=PROJECT_ROOT / "configs" / "experiment.yaml",
        pm_v2_config_path=PROJECT_ROOT / "configs" / "pm_v2.yaml",
        seed_dialogues_path=seed_path,
        endpoint=endpoint,
        base_generation_seed=int(generation["base_seed"]),
        full_user_count=sum(
            int(generation[key])
            for key in ("train_users", "calibration_users", "internal_test_users")
        ),
        input_token_safety_factor=1.5,
        fail_on_reported_input_overrun=True,
        input_usd_per_mtok=0.15,
        output_usd_per_mtok=0.60,
    )
    surfaces = _surface_only_pilot_inputs(contract)
    invalid_context = surfaces["context_only"].model_copy(
        update={
            "current_user_text": "I feel unsettled today.",
            "session_summary": "The user feels unsettled today.",
        }
    )

    class RepairingFakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []

        def chat(self, messages, **kwargs):
            prompt = messages[1]["content"]
            case_field = next(
                field
                for field, _ in GENERATION_CASE_FIELDS
                if f"CASE SLOT (never mention this label): {field}" in prompt
            )
            repair = "one pre-authorized repair attempt" in prompt
            self.calls.append((case_field, repair))
            surface = (
                invalid_context
                if case_field == "context_only" and not repair
                else surfaces[case_field]
            )
            payload = surface.model_dump(mode="json")
            response = {
                "choices": [
                    {"message": {"content": canonical_json(payload)}}
                ]
            }
            return (
                CallResult(
                    text=canonical_json(payload),
                    raw_response=response,
                    usage={
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                    latency_ms=1.0,
                    request_hash=f"fake-{case_field}-{'repair' if repair else 'initial'}",
                ),
                surface,
            )

        def close(self) -> None:
            return None

    fake = RepairingFakeClient()
    monkeypatch.setattr(module, "make_client", lambda endpoint: fake)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *common,
            "--run",
            "--accept-cost-estimate-sha256",
            accepted_hash,
        ],
    )
    module.main()
    assert fake.calls[:2] == [
        ("context_only", False),
        ("context_only", True),
    ]
    assert len(fake.calls) == GENERATION_PILOT_MINIMUM_CALLS + 1
    failed = read_json(out_dir / "failed_context_only_initial.json")
    assert failed["lint"]["status"] == "FAIL"
    assert failed["lint"]["errors"][0]["check"] == "current_family_anchor"
    ledger_rows = list(iter_jsonl(out_dir / "physical_attempt_ledger.jsonl"))
    assert [row["event"] for row in ledger_rows[:4]] == [
        "STARTED", "FAILED", "STARTED", "SUCCEEDED"
    ]
    assert ledger_rows[1]["usage"]["total_tokens"] == 150
    summary = read_json(out_dir / "summary.json")
    assert summary["status"] == "PASS"
    assert summary["physical_attempts"] == GENERATION_PILOT_MINIMUM_CALLS + 1
    assert summary["repair_cases"] == ["context_only"]
    assert summary["repair_case_count"] == 1

    def forbidden_client(endpoint):
        raise AssertionError("accepted casewise traces must never be repeated")

    monkeypatch.setattr(module, "make_client", forbidden_client)
    module.main()


def test_generation_pilot_recovers_transient_500_without_content_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = (
        PROJECT_ROOT
        / "scripts"
        / "20a_run_pm_v2_generation_compatibility_pilot.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pm_v2_generation_pilot_transport_retry", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    seed_path = tmp_path / "seeds.jsonl"
    _write_unique_generation_seeds(seed_path)
    out_dir = tmp_path / "pilot"
    common = [
        str(script),
        "--seed-dialogues",
        str(seed_path),
        "--out-dir",
        str(out_dir),
        "--max-api-calls",
        str(GENERATION_PILOT_MAX_ATTEMPTS),
        "--max-estimated-usd",
        "1",
        "--max-input-tokens-per-call",
        "12000",
    ]
    monkeypatch.setattr(sys, "argv", [*common, "--dry-run"])
    module.main()
    accepted_hash = read_json(out_dir / "cost_estimate.json")[
        "cost_estimate_sha256"
    ]
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")

    experiment = load_config(PROJECT_ROOT / "configs" / "experiment.yaml")
    pm_config = load_config(PROJECT_ROOT / "configs" / "pm_v2.yaml")
    generation = pm_config["data_generation"]
    endpoint = endpoint_from_config(experiment, generation["generator_endpoint"])
    contract = build_generation_compatibility_contract(
        project_root=PROJECT_ROOT,
        experiment_config_path=PROJECT_ROOT / "configs" / "experiment.yaml",
        pm_v2_config_path=PROJECT_ROOT / "configs" / "pm_v2.yaml",
        seed_dialogues_path=seed_path,
        endpoint=endpoint,
        base_generation_seed=int(generation["base_seed"]),
        full_user_count=sum(
            int(generation[key])
            for key in ("train_users", "calibration_users", "internal_test_users")
        ),
        input_token_safety_factor=1.5,
        fail_on_reported_input_overrun=True,
        input_usd_per_mtok=0.15,
        output_usd_per_mtok=0.60,
    )
    surfaces = _surface_only_pilot_inputs(contract)

    class TransientThenSuccessfulClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []
            self.injected = False

        def chat(self, messages, **kwargs):
            prompt = messages[1]["content"]
            case_field = next(
                field
                for field, _ in GENERATION_CASE_FIELDS
                if f"CASE SLOT (never mention this label): {field}" in prompt
            )
            repair = "one pre-authorized repair attempt" in prompt
            self.calls.append((case_field, repair))
            if case_field == "context_only" and not repair and not self.injected:
                self.injected = True
                raise RetryableProviderError(
                    "injected HTTP 500",
                    last_retry_class="http_5xx",
                    last_status_code=500,
                    attempts_tried=1,
                    response_diagnostics={"status_code": 500},
                )
            surface = surfaces[case_field]
            payload = surface.model_dump(mode="json")
            return (
                CallResult(
                    text=canonical_json(payload),
                    raw_response={
                        "choices": [
                            {"message": {"content": canonical_json(payload)}}
                        ]
                    },
                    usage={
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                    latency_ms=1.0,
                    request_hash=f"fake-{case_field}-{len(self.calls)}",
                ),
                surface,
            )

        def close(self) -> None:
            return None

    fake = TransientThenSuccessfulClient()
    monkeypatch.setattr(module, "make_client", lambda endpoint: fake)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *common,
            "--run",
            "--accept-cost-estimate-sha256",
            accepted_hash,
        ],
    )
    module.main()

    assert fake.calls[:2] == [
        ("context_only", False),
        ("context_only", False),
    ]
    assert all(not repair for _, repair in fake.calls)
    assert len(fake.calls) == GENERATION_PILOT_MINIMUM_CALLS + 1
    summary = read_json(out_dir / "summary.json")
    assert summary["status"] == "PASS"
    assert summary["repair_cases"] == []
    assert summary["physical_attempts"] == GENERATION_PILOT_MINIMUM_CALLS + 1
    assert summary["transport_retry_summary"]["failed_attempts_by_class"] == {
        "http_5xx": 1
    }
    assert summary["transport_retry_summary"][
        "logical_calls_recovered_after_retry"
    ] == 1
    ledger_rows = list(iter_jsonl(out_dir / "physical_attempt_ledger.jsonl"))
    assert [row["event"] for row in ledger_rows[:4]] == [
        "STARTED",
        "FAILED",
        "STARTED",
        "SUCCEEDED",
    ]
    assert ledger_rows[1]["metadata"]["retry_class"] == "http_5xx"


def test_generation_contract_ignores_downstream_judges_but_binds_generator(
    tmp_path: Path,
) -> None:
    seed_path = tmp_path / "seeds.jsonl"
    _write_unique_generation_seeds(seed_path)
    source_experiment = PROJECT_ROOT / "configs" / "experiment.yaml"
    source_pm = PROJECT_ROOT / "configs" / "pm_v1_5.yaml"
    experiment_path = tmp_path / "experiment.yaml"
    pm_path = tmp_path / "pm.yaml"
    experiment_text = source_experiment.read_text(encoding="utf-8")
    pm_text = source_pm.read_text(encoding="utf-8")
    experiment_path.write_text(experiment_text, encoding="utf-8")
    pm_path.write_text(pm_text, encoding="utf-8")

    def build() -> dict:
        experiment = load_config(experiment_path)
        pm_config = load_config(pm_path)
        generation = pm_config["data_generation"]
        endpoint = endpoint_from_config(
            experiment, str(generation["generator_endpoint"])
        )
        return build_generation_compatibility_contract(
            project_root=PROJECT_ROOT,
            experiment_config_path=experiment_path,
            pm_v2_config_path=pm_path,
            seed_dialogues_path=seed_path,
            endpoint=endpoint,
            base_generation_seed=int(generation["base_seed"]),
            full_user_count=sum(
                int(generation[key])
                for key in ("train_users", "calibration_users", "internal_test_users")
            ),
            input_token_safety_factor=1.5,
            fail_on_reported_input_overrun=True,
            input_usd_per_mtok=0.15,
            output_usd_per_mtok=0.60,
        )

    baseline = build()
    experiment_path.write_text(
        experiment_text.replace('model: "gpt-4o"', 'model: "gpt-4o-review-only"'),
        encoding="utf-8",
    )
    pm_path.write_text(
        pm_text.replace(
            "pm-v1.5-automated-semantic-review-v4-native-gemini-"
            "deterministic27-plus-paid9",
            "pm-v1.5-automated-semantic-review-v4-native-gemini-"
            "deterministic27-plus-paid9-doc-only-change",
        ),
        encoding="utf-8",
    )
    assert build() == baseline

    experiment_path.write_text(
        experiment_text.replace(
            'model: "gpt-4o-mini"', 'model: "gpt-4o-mini-generator-change"', 1
        ),
        encoding="utf-8",
    )
    pm_path.write_text(pm_text, encoding="utf-8")
    assert build()["contract_sha256"] != baseline["contract_sha256"]

    experiment_path.write_text(experiment_text, encoding="utf-8")
    pm_path.write_text(
        pm_text.replace("max_estimated_usd: 0.054", "max_estimated_usd: 0.053"),
        encoding="utf-8",
    )
    assert build()["contract_sha256"] != baseline["contract_sha256"]


def test_generation_compatibility_attestation_is_exact_and_tamper_evident(
    tmp_path: Path,
) -> None:
    seed_path = tmp_path / "seeds.jsonl"
    _write_unique_generation_seeds(seed_path)
    experiment_config_path = PROJECT_ROOT / "configs" / "experiment.yaml"
    pm_v2_config_path = PROJECT_ROOT / "configs" / "pm_v2.yaml"
    experiment_config = load_config(experiment_config_path)
    pm_v2_config = load_config(pm_v2_config_path)
    generation_cfg = pm_v2_config["data_generation"]
    endpoint = endpoint_from_config(
        experiment_config, str(generation_cfg["generator_endpoint"])
    )
    full_user_count = sum(
        int(generation_cfg[key])
        for key in ("train_users", "calibration_users", "internal_test_users")
    )
    contract = build_generation_compatibility_contract(
        project_root=PROJECT_ROOT,
        experiment_config_path=experiment_config_path,
        pm_v2_config_path=pm_v2_config_path,
        seed_dialogues_path=seed_path,
        endpoint=endpoint,
        base_generation_seed=int(generation_cfg["base_seed"]),
        full_user_count=full_user_count,
        input_token_safety_factor=1.5,
        fail_on_reported_input_overrun=True,
        input_usd_per_mtok=0.15,
        output_usd_per_mtok=0.60,
    )
    _, call_plan, _ = build_generation_compatibility_plan(
        contract, endpoint=endpoint
    )
    assert contract["held_out_seed_index"] >= full_user_count
    assert contract["generator_pricing_usd_per_mtok"] == {
        "input": 0.15,
        "output": 0.60,
    }
    assert contract["generation_config_projection"]["protocol"] == (
        "pm-v2-generation-config-projection-v1"
    )
    assert "experiment_config_sha256" not in contract
    assert "pm_v2_config_sha256" not in contract

    pilot_dir = tmp_path / "pilot_artifacts"
    pilot_dir.mkdir()
    attestation_path = pilot_dir / "artifact_attestation.json"
    with pytest.raises(RuntimeError, match="requires a PASS"):
        require_generation_compatibility_attestation(
            attestation_path, expected_contract=contract
        )

    bundle = _surface_only_pilot_bundle(contract)
    bundle_report = validate_generation_pilot_bundle(bundle, contract)
    assert bundle_report["status"] == "PASS"
    bundle_path = pilot_dir / "pilot_bundle.json"
    write_json(bundle_path, bundle.model_dump(mode="json"))

    ledger_path = pilot_dir / "physical_attempt_ledger.jsonl"
    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=GENERATION_PILOT_STAGE,
        expected_calls={
            str(row["physical_call_key"]): int(row["maximum_physical_attempts"])
            for row in call_plan
        },
        maximum_total_attempts=GENERATION_PILOT_MAX_ATTEMPTS,
    )
    initial_rows = [
        row for row in call_plan if row["attempt_kind"] == "initial"
    ]
    provider_drafts = bundle.provenance["provider_surface_drafts"]
    provider_responses = bundle.provenance["provider_surface_responses"]
    for row in initial_rows:
        case_field = str(row["case_field"])
        reservation = ledger.reserve(
            str(row["physical_call_key"]),
            record_ids={
                "user_id": contract["pilot_user_id"],
                "case_field": case_field,
                "attempt_kind": "initial",
                "generation_seed": row["generation_seed"],
            },
            prompt_sha256=str(row["prompt_sha256"]),
        )
        ledger.finish(
            reservation,
            succeeded=True,
            request_hash=f"test-request-{case_field}",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            error=None,
            result={
                "surface": provider_drafts[case_field],
                "provider_response": provider_responses[case_field],
                "attempt_kind": "initial",
            },
        )
    summary_path = pilot_dir / "summary.json"
    summary = {
        "status": "PASS",
        "compatibility_contract_sha256": contract["contract_sha256"],
        "accepted_surface_calls": initial_rows,
        "accepted_case_count": len(initial_rows),
        "repair_cases": [],
        "repair_case_count": 0,
        "bundle_validation": bundle_report,
        "physical_attempts": len(initial_rows),
        "actual_usage": {
            "prompt_tokens": 100 * len(initial_rows),
            "completion_tokens": 50 * len(initial_rows),
            "total_tokens": 150 * len(initial_rows),
        },
        "transport_retry_summary": retry_ledger_summary(
            ledger, list(ledger.expected_calls)
        ),
    }
    write_json(summary_path, summary)
    run_manifest_path = pilot_dir / "run_manifest.json"
    cost_estimate_path = pilot_dir / "cost_estimate.json"
    call_plan_path = pilot_dir / "call_plan.jsonl"
    write_json(run_manifest_path, {"stage": GENERATION_PILOT_STAGE})
    cost_payload = {
        "stage": GENERATION_PILOT_STAGE,
        "call_plan_sha256": sha256_text(canonical_json(call_plan)),
        "maximum_content_attempts": len(call_plan),
        "maximum_physical_api_attempts": GENERATION_PILOT_MAX_ATTEMPTS,
    }
    accepted_cost_hash = sha256_text(canonical_json(cost_payload))
    write_json(
        cost_estimate_path,
        {
            **cost_payload,
            "cost_estimate_sha256": accepted_cost_hash,
            "budget_gate": {"status": "PASS"},
        },
    )
    write_jsonl(call_plan_path, call_plan)
    create_artifact_attestation(
        attestation_path,
        stage=GENERATION_PILOT_STAGE,
        inputs={
            "experiment_config": experiment_config_path,
            "pm_v2_config": pm_v2_config_path,
            "seed_dialogues": seed_path,
            "run_manifest": run_manifest_path,
            "cost_estimate": cost_estimate_path,
            "call_plan": call_plan_path,
        },
        outputs={
            "pilot_bundle": (bundle_path, False),
            "physical_attempt_ledger": (ledger_path, True),
            "summary": (summary_path, False),
        },
        parameters={
            "compatibility_contract": contract,
            "compatibility_contract_sha256": contract["contract_sha256"],
            "accepted_cost_estimate_sha256": accepted_cost_hash,
            "provider_trace_mode": "surface_only_casewise",
            "retry_contract_protocol": RETRY_CONTRACT_PROTOCOL,
            "maximum_transport_attempts_per_content_attempt": (
                GENERATION_PILOT_MAX_TRANSPORT_ATTEMPTS_PER_CONTENT_ATTEMPT
            ),
        },
        expected={
            "minimum_physical_attempts": GENERATION_PILOT_MINIMUM_CALLS,
            "maximum_physical_attempts": GENERATION_PILOT_MAX_ATTEMPTS,
            "actual_physical_attempts": len(initial_rows),
            "regimes": 9,
            "maximum_repairs_per_case": 1,
        },
    )
    verification = require_generation_compatibility_attestation(
        attestation_path, expected_contract=contract
    )
    assert verification["status"] == "PASS"

    review_dir = tmp_path / "generation_semantic_review"
    prepared = prepare_generation_pilot_semantic_review(
        pilot_attestation_path=attestation_path,
        out_dir=review_dir,
    )
    packet_path = Path(prepared["packet"])
    with packet_path.open("r", encoding="utf-8", newline="") as handle:
        packet_rows = list(csv.DictReader(handle))
    completed_paths = []
    for annotator_id in ("reviewer_a", "reviewer_b"):
        completed_path = review_dir / f"completed_{annotator_id}.csv"
        completed_rows = []
        for source_row in packet_rows:
            row = (
                {"item_id": source_row["item_id"]}
                if annotator_id == "reviewer_a"
                else dict(source_row)
            )
            row["annotator_id"] = annotator_id
            row["notes"] = ""
            for field in GENERATION_REVIEW_RATING_FIELDS:
                row[field] = "1"
            completed_rows.append(row)
        with completed_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = (
                GENERATION_SIMPLE_REVIEW_FIELDS
                if annotator_id == "reviewer_a"
                else GENERATION_REVIEW_PACKET_FIELDS
            )
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames
            )
            writer.writeheader()
            writer.writerows(completed_rows)
        completed_paths.append(completed_path)
    review_report_path = review_dir / "generation_pilot_semantic_review_report.json"
    review_attestation_path = review_dir / "semantic_attestation.json"
    review_report = analyze_generation_pilot_semantic_review(
        completed_paths=completed_paths,
        pilot_attestation_path=attestation_path,
        packet_path=packet_path,
        manual_path=Path(prepared["manual"]),
        plan_path=Path(prepared["plan"]),
        report_path=review_report_path,
        attestation_path=review_attestation_path,
    )
    assert review_report["status"] == "PASS"
    review_verification = require_generation_pilot_semantic_review(
        review_attestation_path,
        expected_pilot_attestation_path=attestation_path,
        expected_contract=contract,
    )
    assert review_verification["annotation_count"] == 18

    with completed_paths[1].open("r", encoding="utf-8", newline="") as handle:
        rejected_rows = list(csv.DictReader(handle))
    rejected_rows[0][GENERATION_REVIEW_RATING_FIELDS[0]] = "0"
    with completed_paths[1].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=GENERATION_REVIEW_PACKET_FIELDS
        )
        writer.writeheader()
        writer.writerows(rejected_rows)
    rejected_report = analyze_generation_pilot_semantic_review(
        completed_paths=completed_paths,
        pilot_attestation_path=attestation_path,
        packet_path=packet_path,
        manual_path=Path(prepared["manual"]),
        plan_path=Path(prepared["plan"]),
        report_path=review_report_path,
        attestation_path=review_attestation_path,
        overwrite=True,
    )
    assert rejected_report["status"] == "FAIL"
    with pytest.raises(RuntimeError, match="not PASS"):
        require_generation_pilot_semantic_review(
            review_attestation_path,
            expected_pilot_attestation_path=attestation_path,
            expected_contract=contract,
        )

    wrong_contract = json.loads(json.dumps(contract))
    wrong_contract["generator_pricing_usd_per_mtok"]["input"] = 0.16
    with pytest.raises(RuntimeError, match="lineage differs"):
        require_generation_compatibility_attestation(
            attestation_path, expected_contract=wrong_contract
        )

    write_json(summary_path, {**summary, "status": "FAIL"})
    with pytest.raises(RuntimeError, match="outputs hash mismatch"):
        require_generation_compatibility_attestation(
            attestation_path, expected_contract=contract
        )

    invalid_bundle = _bundle().model_copy(
        update={
            "user_id": contract["pilot_user_id"],
            "provenance": {
                "generation_compatibility_contract_sha256": contract[
                    "contract_sha256"
                ]
            },
        }
    )
    invalid_report = validate_generation_pilot_bundle(invalid_bundle, contract)
    assert invalid_report["status"] == "FAIL"
    assert invalid_report["checks"]["exact_nine_regimes"] is False


def test_full_generation_run_requires_pilot_attestation_before_api_key(
    tmp_path: Path,
) -> None:
    seed_path = tmp_path / "seeds.jsonl"
    _write_unique_generation_seeds(seed_path)
    out_dir = tmp_path / "full_generation"
    missing_attestation = tmp_path / "missing_pilot" / "artifact_attestation.json"
    env = _subprocess_env()
    env.pop("NVIDIA_API_KEY", None)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "20_generate_pm_v2_development_data.py"),
        "--seed-dialogues",
        str(seed_path),
        "--out-dir",
        str(out_dir),
        "--generation-pilot-attestation",
        str(missing_attestation),
        "--max-api-calls",
        "936",
        "--max-estimated-usd",
        "1",
        "--max-input-tokens-per-call",
        "12000",
    ]
    subprocess.run(
        [*command, "--dry-run"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    accepted = read_json(out_dir / "generation_cost_estimate.json")[
        "cost_estimate_sha256"
    ]
    blocked = subprocess.run(
        [
            *command,
            "--run",
            "--accept-cost-estimate-sha256",
            accepted,
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert blocked.returncode != 0
    assert "requires a PASS generation compatibility pilot attestation" in (
        blocked.stderr
    )
    assert "NVIDIA_API_KEY" not in blocked.stderr


def test_human_rating_manual_has_shared_one_to_five_anchors() -> None:
    script = PROJECT_ROOT / "scripts" / "27_prepare_pm_v2_human_audit.py"
    spec = importlib.util.spec_from_file_location("pm_v2_human_packet", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.RESPONSE_SCALE_ANCHORS) == {1, 2, 3, 4, 5}
    assert set(module.RISK_SCALE_ANCHORS) == {0, 1, 2, 3}
    assert set(module.RESPONSE_DIMENSION_DEFINITIONS) == set(module.RESPONSE_FIELDS)
    assert set(module.RESPONSE_DIMENSION_BOUNDARIES) == set(module.RESPONSE_FIELDS)
    assert set(module.RISK_DIMENSION_DEFINITIONS) == set(module.RISK_FIELDS)
    manual = module.rating_manual_markdown(2)
    assert module.MANUAL_VERSION in manual
    assert "Score every dimension independently" in manual
    assert "Do not enter, infer, average, or report an `overall`" in manual
    assert "Generic empathy can score high" in manual
    assert "An appropriate decision not to mention memory can score 5" in manual
    assert "score grounding independently from temporal_consistency" in manual
    assert "at least 2 independent annotators" in manual
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--seed" not in help_result.stdout


def test_human_sample_plan_is_yaml_seeded_and_hash_bound(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs/pm_v2.yaml").read_text(encoding="utf-8")
    )
    config["human_label_audit"]["items"] = 1
    config["human_label_audit"]["sample_seed"] = 9876
    config_path = tmp_path / "pm_v2.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    case = _case(case_id="human_sample_plan")
    state = case_to_state(
        user_id="human_sample_user",
        case=case,
        split=PMV2Split.TRAIN,
        strategy_catalog_count=12429,
        strategy_estimated_tokens=260,
    )
    states_path = tmp_path / "states.jsonl"
    states_path.write_text(
        json.dumps(state.model_dump(mode="json")) + "\n", encoding="utf-8"
    )
    evaluator_path = tmp_path / "evaluator_contexts.jsonl"
    evaluator_path.write_text(
        json.dumps(case_to_evaluator_context(state, case)) + "\n",
        encoding="utf-8",
    )
    outcomes_path = tmp_path / "outcomes.jsonl"
    outcome = {
        "card_id": state.card_id,
        "state_id": state.state_id,
        "user_id": state.user_id,
        "action_id": "M0+R0",
        "response": "That uncertainty sounds difficult; we can take this one step at a time.",
        "selected_memory_ids": [],
        "selected_strategy_ids": [],
        "memory_view": [],
        "strategy_view": [],
        "cost": {
            "pm_input_tokens_est": 100,
            "retrieval_calls": 0,
            "reranker_calls": 0,
            "memory_tokens": 0,
            "strategy_tokens": 0,
            "base_prompt_tokens": 100,
            "total_input_tokens": 100,
            "output_tokens": 15,
            "latency_ms": 1.0,
        },
        "model_name": "fixture-generator",
        "prompt_hash": "fixture-prompt",
        "request_hash": "fixture-request",
        "provenance": {},
    }
    outcomes_path.write_text(json.dumps(outcome) + "\n", encoding="utf-8")

    quality_cfg = config["quality_composite"]
    composite = CompositeSpec(
        version=quality_cfg["version"], weights=quality_cfg["weights"]
    )
    response = ResponseDimensions(
        emotional_support=4,
        personalization=3,
        memory_appropriateness=5,
        factual_grounding=5,
        temporal_consistency=5,
        non_intrusiveness=5,
    )
    risk = RiskDimensions(
        selected_context_misuse=0,
        unnecessary_exposure=0,
        stale_or_conflicting_use=0,
        unsupported_personal_claim=0,
        memory_omission=0,
        strategy_overuse=0,
        strategy_omission=0,
    )
    dimension_mad = {
        **{f"response.{name}": 0.0 for name in ResponseDimensions.model_fields},
        **{f"risk.{name}": 0.0 for name in RiskDimensions.model_fields},
    }
    label = ActionLabel(
        state_id=state.state_id,
        card_id=state.card_id,
        user_id=state.user_id,
        semantic_family=state.semantic_family,
        action_id="M0+R0",
        response=response,
        risk=risk,
        observed_input_tokens=100,
        retrieval_calls=0,
        judge_families=["fixture-a", "fixture-b"],
        judge_count=2,
        max_dimension_mad=0.0,
        dimension_mad=dimension_mad,
        label_reliable=True,
        composite_spec_version=composite.version,
        composite_weights_sha256=sha256_text(canonical_json(composite.weights)),
    )
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(
        json.dumps(label.model_dump(mode="json")) + "\n", encoding="utf-8"
    )
    manifest_path = tmp_path / "judge_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "prompt_contract_hash": prompt_contract_hash(),
                "pm_v2_config_sha256": sha256_file(config_path),
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "human"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/27_prepare_pm_v2_human_audit.py"),
        "--pm-v2-config",
        str(config_path),
        "--states",
        str(states_path),
        "--evaluator-contexts",
        str(evaluator_path),
        "--outcomes",
        str(outcomes_path),
        "--labels",
        str(labels_path),
        "--judge-manifest",
        str(manifest_path),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=True,
    )
    key = json.loads(
        (out_dir / "human_rating_key.json").read_text(encoding="utf-8")
    )
    plan = json.loads(
        (out_dir / "human_sample_plan.json").read_text(encoding="utf-8")
    )
    assert plan["sample_seed"] == 9876
    assert plan["seed_source"] == "pm_v2.yaml:human_label_audit.sample_seed"
    assert key["sample_plan"]["sha256"] == sha256_text(canonical_json(plan))
    assert key["sample_plan"]["file_sha256"] == sha256_file(
        out_dir / "human_sample_plan.json"
    )
    assert key["llm_rubric_contract"]["prompt_contract_sha256"] == (
        prompt_contract_hash()
    )
    assert key["llm_rubric_contract"]["requests_overall_field"] is False
    packet_header = (out_dir / "human_rating_packet.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    assert "overall" not in packet_header
