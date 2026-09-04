from __future__ import annotations

import copy

import pytest

from metacom_pm.api import Endpoint
from metacom_pm.contracts import MemorySource
from metacom_pm.pm_v1_6_contracts import (
    RetrievalAttempt,
    SourceProbeObservation,
    Step0Cost,
    Step0Observation,
    StrategyFamilyObservation,
    StrategyReadiness,
    STRATEGY_FAMILIES,
    assert_pm_visible_step0_payload,
    realized_action_from_evidence,
)
from metacom_pm.pm_v1_6_judge_isolation import (
    JudgeIsolationPolicy,
    require_development_judge_isolation,
)
from metacom_pm.pm_v1_6_router import (
    StrongTransparentRouter,
    TransparentRouterConfig,
)


def _step0() -> Step0Observation:
    return Step0Observation(
        state_id="state_test",
        sources={
            MemorySource.MP: SourceProbeObservation(
                source=MemorySource.MP,
                available=True,
                bounded_count=1,
                min_age_sessions=2,
                median_age_sessions=2.0,
                max_age_sessions=2,
                estimated_retrievable_tokens=40,
                query_to_source_similarity=0.8,
                representation_valid=True,
            ),
            MemorySource.MS: SourceProbeObservation(
                source=MemorySource.MS,
                available=False,
                bounded_count=0,
                min_age_sessions=None,
                median_age_sessions=None,
                max_age_sessions=None,
                estimated_retrievable_tokens=0,
                query_to_source_similarity=0.0,
                representation_valid=False,
            ),
            MemorySource.ME: SourceProbeObservation(
                source=MemorySource.ME,
                available=True,
                bounded_count=1,
                min_age_sessions=1,
                median_age_sessions=1.0,
                max_age_sessions=1,
                estimated_retrievable_tokens=80,
                query_to_source_similarity=0.2,
                representation_valid=True,
            ),
        },
        strategy_families=[
            StrategyFamilyObservation(
                family=family,
                query_to_family_similarity=(0.6 if family == "reflection" else 0.0),
                representation_valid=(family == "reflection"),
            )
            for family in STRATEGY_FAMILIES
        ],
        readiness=StrategyReadiness(
            advice_requested=False,
            advice_rejected=True,
            listening_requested=True,
            clarification_needed=False,
            action_readiness=0.1,
        ),
        step0_cost=Step0Cost(
            query_encoding_latency_ms=1.0,
            memory_source_comparisons=3,
            strategy_family_comparisons=len(STRATEGY_FAMILIES),
            catalog_build_amortized_ms=0.2,
        ),
    )


def test_step0_payload_excludes_audit_and_item_fields() -> None:
    observation = _step0()
    payload = observation.model_dump(mode="json")
    assert_pm_visible_step0_payload(payload)
    assert "catalog_embedding" not in str(payload)
    assert "representation_sha256" not in str(payload)

    poisoned = copy.deepcopy(payload)
    poisoned["sources"]["MP"]["memory_id"] = "mem_0123456789ab"
    with pytest.raises(ValueError, match="forbidden PM-visible"):
        assert_pm_visible_step0_payload(poisoned)


def test_unavailable_source_cannot_smuggle_similarity() -> None:
    with pytest.raises(ValueError, match="similarity must be zero"):
        SourceProbeObservation(
            source=MemorySource.MS,
            available=False,
            bounded_count=0,
            estimated_retrievable_tokens=0,
            query_to_source_similarity=0.2,
            representation_valid=False,
        )


def test_zero_hit_attempt_is_legal_but_call_is_retained() -> None:
    attempt = RetrievalAttempt(
        source="ME",
        call_count=1,
        hit_count=0,
        retrieved_tokens=0,
        latency_ms=2.5,
    )
    assert attempt.call_count == 1
    assert attempt.hit_count == 0
    with pytest.raises(ValueError, match="zero-hit"):
        RetrievalAttempt(
            source="ME",
            call_count=1,
            hit_count=0,
            retrieved_tokens=8,
            latency_ms=2.5,
        )


def test_realized_action_comes_from_actual_evidence() -> None:
    assert (
        realized_action_from_evidence(
            [MemorySource.MP, MemorySource.ME], strategy_card_count=0
        )
        == "MPE+R0"
    )
    assert realized_action_from_evidence([], strategy_card_count=2) == "M0+RS"


def test_development_judge_denylist_checks_resolved_metadata() -> None:
    policy = JudgeIsolationPolicy()
    endpoints = {
        "dev_a": Endpoint(
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "gemini-2.5-flash",
            "GEMINI_API_KEY",
            family="google_gemini",
        ),
        "dev_b": Endpoint(
            "https://integrate.api.nvidia.com",
            "deepseek-ai/deepseek-v4-flash",
            "NVIDIA_API_KEY",
            family="deepseek",
        ),
    }
    report = require_development_judge_isolation(
        endpoints, ["dev_a", "dev_b"], policy=policy
    )
    assert report["status"] == "PASS"

    renamed_final = {
        **endpoints,
        "innocent_alias": Endpoint(
            "https://api.openai.com",
            "gpt-4o",
            "OPENAI_API_KEY",
            family="something_else",
        ),
    }
    with pytest.raises(RuntimeError, match="final judge model/provider marker"):
        require_development_judge_isolation(
            renamed_final, ["dev_a", "innocent_alias"], policy=policy
        )


def test_transparent_router_uses_step0_and_respects_advice_rejection() -> None:
    decision = StrongTransparentRouter(
        TransparentRouterConfig(
            source_selection_threshold=0.1,
            maximum_sources=1,
            strategy_selection_threshold=0.25,
        )
    ).choose(_step0())
    assert decision.action_id == "MP+R0"
    assert all(value > -float("inf") for value in decision.source_scores.values())
