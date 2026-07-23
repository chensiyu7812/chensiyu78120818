"""Unit tests for scripts/v1_5/41_diagnose_merged_ood_audit_v1_5.py's pure
logic (dimension categorization, per-domain aggregation).

Deliberately does not exercise main() end-to-end: that needs the real
semantic encoder plus the full synthetic/ESConv-auxiliary/ESConv-test/EvoEmo
corpora, which is slow and already verified manually against real data (see
the commit introducing this script). These tests cover the reusable helpers
with small, fast, synthetic fixtures instead.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from metacom_pm.contracts import DialogueTurn, MemorySource, canonical_action_id, StrategyMode
from metacom_pm.pm_v2_contracts import ObservableSourceSummary, PMV2Split, PMV2State
from metacom_pm.pm_v2_features import PMV2FeatureBuilder

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "v1_5" / "41_diagnose_merged_ood_audit_v1_5.py"
    spec = importlib.util.spec_from_file_location(
        "v1_5_merged_ood_audit_test", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tiny_state(state_id: str, *, mp_available: bool) -> PMV2State:
    inventory = {
        MemorySource.MP: ObservableSourceSummary(
            available=mp_available,
            count=2 if mp_available else 0,
            min_age_sessions=1 if mp_available else None,
            median_age_sessions=1.5 if mp_available else None,
            max_age_sessions=2 if mp_available else None,
            estimated_tokens=80 if mp_available else 0,
        ),
        MemorySource.MS: ObservableSourceSummary(available=False, count=0),
        MemorySource.ME: ObservableSourceSummary(
            available=True,
            count=2,
            min_age_sessions=1,
            median_age_sessions=1.5,
            max_age_sessions=2,
            estimated_tokens=100,
        ),
    }
    available_sources = [s for s in MemorySource if inventory[s].available]
    allowed_actions = []
    for bits in range(1 << len(available_sources)):
        subset = frozenset(
            s for i, s in enumerate(available_sources) if bits & (1 << i)
        )
        for strategy in (StrategyMode.R0, StrategyMode.RS):
            allowed_actions.append(canonical_action_id(subset, strategy))
    return PMV2State(
        state_id=state_id,
        card_id=f"card_{state_id}",
        user_id=f"user_{state_id}",
        split=PMV2Split.TRAIN,
        semantic_family="family",
        surface_form_id=f"surface_{state_id}",
        current_user_text="I am having a difficult day.",
        current_session_history=[DialogueTurn(role="user", content="I feel tense.")],
        current_session_summary="",
        session_index=4,
        inventory=inventory,
        allowed_actions=allowed_actions,
    )


def test_dimension_category_classifies_known_names():
    module = _load_module()
    assert module._dimension_category("session_index") == "age_staleness_drift"
    assert module._dimension_category("MP.max_age_ratio") == "age_staleness_drift"
    assert module._dimension_category("ME.age_span_ratio") == "age_staleness_drift"
    assert module._dimension_category("MS.count_ratio") == "catalog_tail_drift"
    assert (
        module._dimension_category("ME.expected_tokens_frac") == "catalog_tail_drift"
    )
    assert module._dimension_category("strategy.family.question") == "strategy_drift"
    assert (
        module._dimension_category("strategy.representation_valid")
        == "strategy_drift"
    )
    assert module._dimension_category("MP.representation_valid") == "other_metadata"


def test_domain_report_handles_empty_states():
    module = _load_module()
    train = [_tiny_state("t0", mp_available=False), _tiny_state("t1", mp_available=False)]
    builder = PMV2FeatureBuilder(use_precomputed_embeddings=False).fit(train)
    report = module._domain_report(builder, [], domain_label="empty")
    assert report == {
        "domain": "empty",
        "n_states": 0,
        "n_severe": 0,
        "severe_rate": None,
    }


def test_domain_report_flags_a_shifted_state_and_reconstructs_the_aggregate():
    module = _load_module()
    train = [_tiny_state(f"train_{i}", mp_available=False) for i in range(5)]
    calibration = [_tiny_state(f"cal_{i}", mp_available=False) for i in range(5)]
    builder = PMV2FeatureBuilder(use_precomputed_embeddings=False).fit(train)
    builder.calibrate_ood(
        calibration,
        semantic_false_positive_quantile=0.99,
        metadata_false_positive_quantile=0.99,
        maximum_joint_in_distribution_fallback_rate=0.05,
        minimum_semantic_challenge_detection_rate=0.80,
        minimum_metadata_challenge_detection_rate=0.95,
    )
    in_distribution = _tiny_state("indist", mp_available=False)
    shifted = _tiny_state("shifted", mp_available=True)  # MP never available in train
    report = module._domain_report(
        builder, [in_distribution, shifted], domain_label="mixed"
    )
    assert report["n_states"] == 2
    assert report["domain"] == "mixed"
    # The shifted state's MP dimensions were never seen available in train,
    # so at least the metadata channel should register some severity.
    assert report["n_severe_metadata_ood"] >= 1
    total_dims = len(builder.metadata_dimension_names())
    reconstructed_mean = (
        sum(report["metadata_ood_contribution_sum_by_category"].values()) / total_dims
    )
    # Recompute the true average metadata_ood_score across both states the
    # same way _domain_report does, to confirm the category sums are a
    # faithful decomposition (not just plausible-looking numbers).
    true_mean = sum(
        builder.ood_report(state)["metadata_ood_score"]
        for state in (in_distribution, shifted)
    ) / 2
    assert abs(reconstructed_mean - true_mean) < 1e-9
