from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from metacom_pm.api import (
    CallResult,
    Endpoint,
    chat_request_payload,
    request_payload_has_schema,
)
from metacom_pm.artifacts import require_content_addressed_attestation
from metacom_pm.attempt_ledger import (
    PersistentAttemptLedger,
    forbid_overwrite_of_spent_attempts,
    physical_call_key,
    reported_prompt_token_error,
)
from metacom_pm.config import load_config
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.fixed_seeker_contract import FixedSeekerGenerationContract
from metacom_pm.evoemo import fixed_seeker_cost_planning_contract
from metacom_pm.pm_v2_contracts import (
    ActionLabel,
    CompositeSpec,
    PMV2State,
    ResponseDimensions,
    RiskDimensions,
)
from metacom_pm.pm_v2_evoemo import (
    EVALUATION_UNIT_CONTRACT_PROTOCOL,
    bind_external_generation_request_log,
    build_generation_evaluation_units,
    compare_cost_matched_token_rows,
    compare_observed_cost_matched_turns,
    require_generation_evaluation_unit_contract,
    reconcile_succeeded_generation_turns,
    run_pmv2_fixed_evoemo,
    summarize_external_generation_raw_matrix,
    summarize_action_preflight,
)
from metacom_pm.pm_v2_external_eval import (
    build_external_pointwise_case,
    load_fixed_turns,
    validate_external_score_table,
)
from metacom_pm.pm_v2_external_schema_smoke import (
    POINTWISE_SCHEMA_SMOKE_PROTOCOL,
    build_pointwise_schema_smoke_plan,
    precreate_pointwise_schema_smoke_clients,
)
from metacom_pm.pm_v2_judging import (
    ResponseJudgeOutput,
    RiskJudgeOutput,
    composite_spec_from_config,
    composite_weights_hash,
    dimension_applicability_by_action,
    dimension_applicability_contract_sha256,
    dimension_health,
    dimensions_inapplicable_to_every_action,
    judge_family_directional_preference_report,
    judge_one,
    labeling_settings_from_config,
    prompt_contract_hash,
    validate_raw_judge_family_health,
    validate_raw_judge_family_subgroup_health,
    validate_judge_table,
)
from metacom_pm.pm_v2_forced_swap import (
    forced_swap_compatibility_gate,
    forced_swap_efficacy_gate,
    forced_swap_family_direction_agreement,
    forced_swap_schema_futility_reason,
    resolve_dual_order_preference,
    select_forced_swap_units,
)
from metacom_pm.pm_v2_features import PMV2FeatureBuilder
from metacom_pm.pm_v2_model import PMV2Model, SelectionConfig
from metacom_pm.text import conservative_token_bound, estimate_tokens


def test_prompt_contract_hash_is_not_the_legacy_schema_only_hash():
    legacy = sha256_text(
        canonical_json(
            {
                "response_schema": ResponseJudgeOutput.model_json_schema(),
                "risk_schema": RiskJudgeOutput.model_json_schema(),
                "version": "pmv2-judge-v1-no-overall",
            }
        )
    )
    assert prompt_contract_hash() != legacy


def test_quality_and_label_gates_are_loaded_from_pmv2_yaml():
    config = load_config("configs/pm_v2.yaml")
    spec = composite_spec_from_config(config)
    gates = labeling_settings_from_config(config)
    assert spec.version == config["quality_composite"]["version"]
    assert spec.weights == config["quality_composite"]["weights"]
    assert gates["minimum_families"] == 2
    assert gates["duplicate_exact_match_rate"] == 0.98
    assert gates["composite_support_exact_match_rate"] == 0.98
    assert gates["maximum_absolute_composite_support_correlation"] == 0.995
    assert gates["minimum_low_mad_coverage_per_dimension"] == 0.80


def test_judge_gate_rejects_composite_that_collapses_to_support():
    weights = {name: 0.0 for name in ResponseDimensions.model_fields}
    weights["emotional_support"] = 1.0
    spec = CompositeSpec(version="degenerate-support-composite", weights=weights)
    dimension_names = [
        *[f"response.{name}" for name in ResponseDimensions.model_fields],
        *[f"risk.{name}" for name in RiskDimensions.model_fields],
    ]
    labels = []
    for index in range(10):
        labels.append(
            ActionLabel(
                state_id=f"degenerate-s{index}",
                card_id=f"degenerate-c{index}",
                user_id=f"degenerate-u{index}",
                semantic_family=f"degenerate-f{index}",
                action_id="M0+R0",
                response=ResponseDimensions(
                    **{
                        name: 1.0 + float((index * (offset + 1) + offset) % 5)
                        for offset, name in enumerate(ResponseDimensions.model_fields)
                    }
                ),
                risk=RiskDimensions(
                    **{
                        name: float((index * (offset + 1) + offset) % 4)
                        for offset, name in enumerate(RiskDimensions.model_fields)
                    }
                ),
                observed_input_tokens=100,
                retrieval_calls=0,
                judge_families=["a", "b"],
                judge_count=2,
                max_dimension_mad=0.0,
                dimension_mad={name: 0.0 for name in dimension_names},
                label_reliable=True,
                composite_spec_version=spec.version,
                composite_weights_sha256=composite_weights_hash(spec),
            )
        )
    with pytest.raises(RuntimeError, match="judge quality gate failed"):
        validate_judge_table(
            labels,
            duplicate_exact_match_rate=1.1,
            maximum_absolute_dimension_correlation=1.1,
            reject_constant_response_dimensions=False,
            reject_constant_risk_dimensions=False,
            composite_spec=spec,
        )


def test_raw_family_gate_cannot_be_masked_by_an_independent_family():
    response_pairs = ((1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (4, 0))
    risk_pairs = ((1, 0), (1, 1), (1, 2), (1, 3), (3, 0), (3, 1), (3, 2))
    rows = []
    for index in range(40):
        support = 1.0 + float(index % 5)
        varied_response = {
            name: 1.0 + float((index * multiplier + offset) % 5)
            for name, (multiplier, offset) in zip(
                ResponseDimensions.model_fields, response_pairs
            )
        }
        varied_risk = {
            name: float((index * multiplier + offset) % 4)
            for name, (multiplier, offset) in zip(
                RiskDimensions.model_fields, risk_pairs
            )
        }
        rows.extend(
            [
                {
                    "judge_family": "collapsed",
                    "response": {
                        **{
                            name: support
                            for name in ResponseDimensions.model_fields
                        },
                        "rationale": "copied dimensions",
                    },
                    "risk": {**varied_risk, "rationale": "varied risks"},
                },
                {
                    "judge_family": "independent",
                    "response": {
                        **varied_response,
                        "rationale": "independent dimensions",
                    },
                    "risk": {**varied_risk, "rationale": "varied risks"},
                },
            ]
        )
    report = validate_raw_judge_family_health(
        rows,
        expected_families=["collapsed", "independent"],
        duplicate_exact_match_rate=0.98,
        maximum_absolute_dimension_correlation=1.1,
        composite_support_exact_match_rate=1.1,
        maximum_absolute_composite_support_correlation=1.1,
        reject_constant_response_dimensions=False,
        reject_constant_risk_dimensions=False,
        composite_spec=CompositeSpec(),
        raise_on_failure=False,
    )
    assert report["status"] == "FAIL"
    assert report["failed_families"] == ["collapsed"]
    assert report["families"]["independent"]["status"] == "PASS"


def test_raw_family_subgroup_gate_catches_action_local_collapse():
    rows = []
    for action_id in ("M0+R0", "ME+RS"):
        for index in range(20):
            for family in ("family_a", "family_b"):
                values = {
                    name: 1.0
                    + float(
                        (
                            index * (offset + 1)
                            + (0 if family == "family_a" else offset)
                        )
                        % 5
                    )
                    for offset, name in enumerate(ResponseDimensions.model_fields)
                }
                if action_id == "M0+R0" and family == "family_a":
                    copied = 1.0 + float(index % 5)
                    values = {
                        name: copied for name in ResponseDimensions.model_fields
                    }
                rows.append(
                    {
                        "judge_family": family,
                        "action_id": action_id,
                        "response": {**values, "rationale": "response"},
                        "risk": {
                            **{
                                name: float((index * (offset + 1) + offset) % 4)
                                for offset, name in enumerate(
                                    RiskDimensions.model_fields
                                )
                            },
                            "rationale": "risk",
                        },
                    }
                )
    report = validate_raw_judge_family_subgroup_health(
        rows,
        subgroup_key="action_id",
        expected_subgroups=("M0+R0", "ME+RS"),
        expected_families=("family_a", "family_b"),
        duplicate_exact_match_rate=0.98,
        maximum_absolute_dimension_correlation=1.1,
        composite_support_exact_match_rate=1.1,
        maximum_absolute_composite_support_correlation=1.1,
        reject_constant_response_dimensions=False,
        reject_constant_risk_dimensions=False,
        composite_spec=CompositeSpec(),
        raise_on_failure=False,
    )
    assert report["status"] == "FAIL"
    assert "M0+R0" in report["failed_subgroups"]
    assert report["subgroups"]["M0+R0"]["families"]["family_a"]["status"] == "FAIL"


def test_raw_family_subgroup_gate_allows_structurally_inapplicable_zero_risks():
    rows = []
    for index in range(20):
        for family_offset, family in enumerate(("family_a", "family_b")):
            response = {
                name: 1.0
                + float(
                    (
                        index * (offset + 1)
                        + family_offset * (offset + 2)
                        + index // (offset + 2)
                    )
                    % 5
                )
                for offset, name in enumerate(ResponseDimensions.model_fields)
            }
            rows.append(
                {
                    "judge_family": family,
                    "action_id": "M0+R0",
                    "response": {**response, "rationale": "response"},
                    "risk": {
                        **{name: 0.0 for name in RiskDimensions.model_fields},
                        "rationale": "not applicable",
                    },
                }
            )
    report = validate_raw_judge_family_subgroup_health(
        rows,
        subgroup_key="action_id",
        expected_subgroups=("M0+R0",),
        expected_families=("family_a", "family_b"),
        duplicate_exact_match_rate=1.1,
        maximum_absolute_dimension_correlation=1.1,
        composite_support_exact_match_rate=1.1,
        maximum_absolute_composite_support_correlation=1.1,
        reject_constant_response_dimensions=False,
        reject_constant_risk_dimensions=True,
        composite_spec=CompositeSpec(),
        raise_on_failure=False,
    )
    assert report["status"] == "PASS"
    assert report["check_risk_dimension_health"] is False
    family_report = report["subgroups"]["M0+R0"]["families"]["family_a"]
    assert family_report["risk_dimension_prevalence"]
    assert family_report["duplicate_dimension_pairs"] == []
    assert family_report["constant_dimensions"] == []


def test_dimension_applicability_by_action_matches_applicable_risk_fields():
    by_action = dimension_applicability_by_action(["M0+R0", "M0+RS"])
    assert by_action["M0+R0"] == frozenset(
        {"risk.selected_context_misuse", "risk.unnecessary_exposure",
         "risk.stale_or_conflicting_use", "risk.strategy_overuse"}
    )
    assert by_action["M0+RS"] == frozenset(
        {"risk.selected_context_misuse", "risk.unnecessary_exposure",
         "risk.stale_or_conflicting_use"}
    )


def test_dimensions_inapplicable_to_every_action_is_the_intersection():
    # M0+R0/M0+RS never select memory; MP+R0 does -- so the memory-misuse
    # dimensions are inapplicable to the first two but applicable to the
    # third, and must NOT appear in the "inapplicable to every action" set.
    only_memoryless = dimensions_inapplicable_to_every_action(["M0+R0", "M0+RS"])
    assert "risk.selected_context_misuse" in only_memoryless
    mixed = dimensions_inapplicable_to_every_action(["M0+R0", "MP+R0"])
    assert "risk.selected_context_misuse" not in mixed
    assert dimensions_inapplicable_to_every_action([]) == frozenset()


def test_dimension_applicability_contract_sha256_is_order_independent_and_sensitive():
    a = dimension_applicability_contract_sha256(["M0+R0", "M0+RS"])
    b = dimension_applicability_contract_sha256(["M0+RS", "M0+R0"])
    c = dimension_applicability_contract_sha256(["M0+R0", "MP+R0"])
    assert a == b
    assert a != c


def test_raw_family_gate_ignores_declared_inapplicable_constant_risk_dimension():
    rows = []
    for index in range(20):
        for family in ("family_a", "family_b"):
            risk = {
                name: float((index + hash(name)) % 4)
                for name in RiskDimensions.model_fields
            }
            # selected_context_misuse is declared inapplicable and held at a
            # real structural zero; every other risk dimension still varies.
            risk["selected_context_misuse"] = 0.0
            rows.append(
                {
                    "judge_family": family,
                    "response": {
                        **{
                            name: 1.0 + float((index + i) % 5)
                            for i, name in enumerate(ResponseDimensions.model_fields)
                        },
                        "rationale": "response",
                    },
                    "risk": {**risk, "rationale": "risk"},
                }
            )
    kwargs = dict(
        expected_families=["family_a", "family_b"],
        duplicate_exact_match_rate=1.1,
        maximum_absolute_dimension_correlation=1.1,
        composite_support_exact_match_rate=1.1,
        maximum_absolute_composite_support_correlation=1.1,
        reject_constant_response_dimensions=False,
        reject_constant_risk_dimensions=True,
        composite_spec=CompositeSpec(),
        raise_on_failure=False,
    )
    without_contract = validate_raw_judge_family_health(rows, **kwargs)
    assert without_contract["status"] == "FAIL"
    assert "risk.selected_context_misuse" in (
        without_contract["families"]["family_a"]["constant_dimensions"]
    )
    with_contract = validate_raw_judge_family_health(
        rows,
        inapplicable_risk_dimensions=frozenset({"risk.selected_context_misuse"}),
        **kwargs,
    )
    assert with_contract["status"] == "PASS"
    assert with_contract["families"]["family_a"]["constant_dimensions"] == []

    # A genuinely constant *applicable* dimension must still fail.
    for row in rows:
        row["risk"]["memory_omission"] = 0.0
    still_fails = validate_raw_judge_family_health(
        rows,
        inapplicable_risk_dimensions=frozenset({"risk.selected_context_misuse"}),
        **kwargs,
    )
    assert still_fails["status"] == "FAIL"
    assert "risk.memory_omission" in (
        still_fails["families"]["family_a"]["constant_dimensions"]
    )


def test_judge_table_ignores_declared_inapplicable_dimension_for_low_mad_coverage():
    spec = CompositeSpec()
    dimensions = [
        *[f"response.{name}" for name in ResponseDimensions.model_fields],
        *[f"risk.{name}" for name in RiskDimensions.model_fields],
    ]
    labels = []
    for index in range(20):
        # risk.selected_context_misuse disagrees on every single label (MAD
        # always above threshold) -- a real, structural artifact of a
        # dimension that is never applicable to M0+R0, not a judge defect.
        dimension_mad = {name: 0.0 for name in dimensions}
        dimension_mad["risk.selected_context_misuse"] = 1.0
        labels.append(
            ActionLabel(
                state_id=f"s{index}",
                card_id=f"c{index}",
                user_id=f"u{index}",
                semantic_family=f"f{index}",
                action_id="M0+R0",
                response=ResponseDimensions(
                    **{
                        name: 1.0 + float((index * (offset + 1)) % 5)
                        for offset, name in enumerate(ResponseDimensions.model_fields)
                    }
                ),
                risk=RiskDimensions(
                    **{name: 0.0 for name in RiskDimensions.model_fields}
                ),
                observed_input_tokens=100,
                retrieval_calls=0,
                judge_families=["a", "b"],
                judge_count=2,
                max_dimension_mad=1.0,
                dimension_mad=dimension_mad,
                label_reliable=True,
                composite_weights_sha256=composite_weights_hash(spec),
            )
        )
    kwargs = dict(
        minimum_reliable_rate=0.0,
        reliable_mad_threshold=0.75,
        minimum_low_mad_coverage_per_dimension=0.90,
        minimum_low_mad_coverage_per_action_dimension=0.90,
        duplicate_exact_match_rate=1.1,
        maximum_absolute_dimension_correlation=1.1,
        reject_constant_response_dimensions=False,
        reject_constant_risk_dimensions=False,
        composite_spec=spec,
        raise_on_failure=False,
    )
    without_contract = validate_judge_table(labels, **kwargs)
    assert without_contract["status"] == "FAIL"
    assert "risk.selected_context_misuse" in without_contract["low_coverage_dimensions"]
    assert (
        "M0+R0/risk.selected_context_misuse"
        in without_contract["low_coverage_action_dimensions"]
    )
    with_contract = validate_judge_table(
        labels,
        inapplicable_risk_dimensions=frozenset({"risk.selected_context_misuse"}),
        inapplicable_risk_dimensions_by_action=dimension_applicability_by_action(
            ["M0+R0"]
        ),
        **kwargs,
    )
    assert with_contract["status"] == "PASS"
    assert with_contract["low_coverage_dimensions"] == []
    assert with_contract["low_coverage_action_dimensions"] == []
    # The raw coverage dicts must show an explicit N/A (None), not a computed
    # (here, artificially low) number that merely happens to be excluded from
    # the failure lists above -- the report itself must not misrepresent an
    # inapplicable cell as if it were a real (bad or good) measurement.
    assert (
        with_contract["dimension_low_mad_coverage"]["risk.selected_context_misuse"]
        is None
    )
    assert (
        with_contract["action_dimension_low_mad_coverage"]["M0+R0"][
            "risk.selected_context_misuse"
        ]
        is None
    )
    # An applicable dimension on the same table must still report a real
    # computed coverage value, not be swept into N/A by accident.
    assert isinstance(
        with_contract["dimension_low_mad_coverage"]["response.emotional_support"],
        float,
    )


def test_dimension_health_pairwise_applicability_restricts_before_informative_filter():
    # action A makes "overuse" inapplicable; its own (positively-correlated)
    # noise would otherwise dilute the real, cleanly anti-correlated signal
    # that action B alone provides between "overuse" and "omission".
    field_names = ("overuse", "omission")
    action_ids = ["A"] * 10 + ["B"] * 10
    a_overuse = [1.0, 0.0] * 5
    a_omission = [1.0, 0.0] * 5
    b_overuse = [1.0, 0.0] * 5
    b_omission = [0.0, 1.0] * 5
    matrix = np.asarray(
        list(zip(a_overuse + b_overuse, a_omission + b_omission)), dtype=float
    )
    inapplicable_by_action = {"A": frozenset({"risk.overuse"}), "B": frozenset()}

    (
        _dup_without,
        corr_without,
        _const_without,
        _prev_without,
        mutual_without,
    ) = dimension_health(
        matrix,
        field_names,
        prefix="risk",
        duplicate_exact_match_rate=1.1,
        maximum_absolute_dimension_correlation=0.95,
        minimum_nonzero_observations=5,
        split_correlation_by_sign=True,
    )
    assert corr_without == []
    assert mutual_without == []

    (
        _dup_with,
        corr_with,
        _const_with,
        _prev_with,
        mutual_with,
    ) = dimension_health(
        matrix,
        field_names,
        prefix="risk",
        duplicate_exact_match_rate=1.1,
        maximum_absolute_dimension_correlation=0.95,
        minimum_nonzero_observations=5,
        action_ids=action_ids,
        inapplicable_risk_dimensions_by_action=inapplicable_by_action,
        split_correlation_by_sign=True,
    )
    assert corr_with == []
    assert len(mutual_with) == 1
    pair = mutual_with[0]
    assert pair["pairwise_applicable_rows"] == 10
    assert pair["informative_correlation"] == pytest.approx(-1.0)


def test_dimension_health_splits_correlation_by_sign():
    field_names = ("a", "b")
    positive_matrix = np.asarray([[1.0, 1.0], [0.0, 0.0]] * 6, dtype=float)
    negative_matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]] * 6, dtype=float)

    (_dup, positive_high, _const, _prev, positive_mutual) = dimension_health(
        positive_matrix,
        field_names,
        prefix="risk",
        duplicate_exact_match_rate=1.1,
        maximum_absolute_dimension_correlation=0.95,
        split_correlation_by_sign=True,
    )
    assert len(positive_high) == 1
    assert positive_mutual == []

    (_dup2, negative_high, _const2, _prev2, negative_mutual) = dimension_health(
        negative_matrix,
        field_names,
        prefix="risk",
        duplicate_exact_match_rate=1.1,
        maximum_absolute_dimension_correlation=0.95,
        split_correlation_by_sign=True,
    )
    assert negative_high == []
    assert len(negative_mutual) == 1

    # split_correlation_by_sign=False (the default) preserves the exact
    # unsplit prior behavior: both directions gate as high_correlation_pairs.
    (_dup3, negative_unsplit, _const3, _prev3, negative_mutual_unsplit) = (
        dimension_health(
            negative_matrix,
            field_names,
            prefix="risk",
            duplicate_exact_match_rate=1.1,
            maximum_absolute_dimension_correlation=0.95,
        )
    )
    assert len(negative_unsplit) == 1
    assert negative_mutual_unsplit == []


def test_judge_family_directional_preference_report_concordance_and_contingency():
    dims = list(ResponseDimensions.model_fields)
    risk_dims = list(RiskDimensions.model_fields)

    def row(state_id, action_id, family, support_value):
        return {
            "state_id": state_id,
            "action_id": action_id,
            "judge_family": family,
            "response": {
                **{name: support_value for name in dims},
                "rationale": "r",
            },
            "risk": {**{name: 0.0 for name in risk_dims}, "rationale": "r"},
        }

    rows = []
    # s1: both families score M0+RS higher -> concordant.
    for family in ("fam_a", "fam_b"):
        rows.append(row("s1", "M0+R0", family, 3.0))
        rows.append(row("s1", "M0+RS", family, 4.0))
    # s2/s3: fam_a prefers M0+R0, fam_b prefers M0+RS -> discordant.
    for state_id in ("s2", "s3"):
        rows.append(row(state_id, "M0+R0", "fam_a", 5.0))
        rows.append(row(state_id, "M0+RS", "fam_a", 1.0))
        rows.append(row(state_id, "M0+R0", "fam_b", 1.0))
        rows.append(row(state_id, "M0+RS", "fam_b", 5.0))

    report = judge_family_directional_preference_report(
        rows,
        action_a="M0+R0",
        action_b="M0+RS",
        expected_families=["fam_a", "fam_b"],
        dialogue_by_state={"s1": "d1", "s2": "d2", "s3": "d3"},
        risk_weight=0.25,
        bootstrap_replicates=200,
        bootstrap_seed=3,
    )
    assert report["diagnostic_only"] is True
    assert report["never_gates"] is True
    assert report["n_comparable_states"] == 3
    assert report["concordant_states"] == 1
    assert report["discordant_states"] == 2
    assert report["tie_involved_states"] == 0
    assert report["concordance_rate"] == pytest.approx(1.0 / 3.0)
    assert report["directional_contingency_table"]["M0+R0"]["M0+RS"] == 2
    assert report["directional_contingency_table"]["M0+RS"]["M0+RS"] == 1
    bootstrap = report["dialogue_cluster_bootstrap"]
    assert bootstrap["n_groups"] == 3
    assert bootstrap["concordance_rate_ci_lower"] is not None
    assert bootstrap["concordance_rate_ci_upper"] is not None
    assert 0.0 <= bootstrap["concordance_rate_ci_lower"] <= bootstrap[
        "concordance_rate_ci_upper"
    ] <= 1.0


def test_judge_family_directional_preference_report_ties_excluded_from_decisive_rates():
    dims = list(ResponseDimensions.model_fields)
    risk_dims = list(RiskDimensions.model_fields)

    def row(state_id, action_id, family, support_value):
        return {
            "state_id": state_id,
            "action_id": action_id,
            "judge_family": family,
            "response": {
                **{name: support_value for name in dims},
                "rationale": "r",
            },
            "risk": {**{name: 0.0 for name in risk_dims}, "rationale": "r"},
        }

    rows = []
    # fam_a ties on every state (identical scores for both actions).
    for state_id in ("s1", "s2", "s3"):
        rows.append(row(state_id, "M0+R0", "fam_a", 3.0))
        rows.append(row(state_id, "M0+RS", "fam_a", 3.0))
        rows.append(row(state_id, "M0+R0", "fam_b", 5.0))
        rows.append(row(state_id, "M0+RS", "fam_b", 1.0))

    report = judge_family_directional_preference_report(
        rows,
        action_a="M0+R0",
        action_b="M0+RS",
        expected_families=["fam_a", "fam_b"],
        dialogue_by_state={"s1": "d1", "s2": "d2", "s3": "d3"},
        risk_weight=0.25,
        bootstrap_replicates=200,
        bootstrap_seed=1,
    )
    assert report["tie_involved_states"] == 3
    assert report["concordant_states"] == 0
    assert report["discordant_states"] == 0
    assert report["concordance_rate"] is None
    assert report["discordance_rate"] is None
    assert report["tie_rate"] == 1.0


def test_success_ledger_reconciles_generation_turn_after_append_crash(tmp_path):
    call_key = "a" * 64
    ledger = PersistentAttemptLedger(
        tmp_path / "attempts.jsonl",
        stage="evoemo_pm_v2_supporter",
        expected_calls={call_key: 1},
        maximum_total_attempts=1,
    )
    unit = ("user-1", 2, "pm_v2", 7, "sim", "fixed", 3)
    reservation = ledger.reserve(
        call_key,
        record_ids={"user_id": "user-1", "turn_index": 3},
        prompt_sha256="b" * 64,
    )
    turn_record = {
        "user_id": "user-1",
        "topic_index": 2,
        "condition": "pm_v2",
        "seed": 7,
        "simulator_id": "sim",
        "interaction_mode": "fixed",
        "turn_index": 3,
        "supporter_message": "I hear you.",
        "cost": {"total_input_tokens": 100, "output_tokens": 5},
        "physical_call_key": call_key,
        "physical_attempt_index": reservation.attempt_index,
        "physical_attempt_key": reservation.attempt_key,
    }
    ledger.finish(
        reservation,
        succeeded=True,
        request_hash="c" * 64,
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "total_tokens": 105,
        },
        error=None,
        result={"turn_record": turn_record, "request_log": {"ok": True}},
    )
    turn_path = tmp_path / "turns.jsonl"
    turn_index = {}
    recovered = reconcile_succeeded_generation_turns(
        ledger=ledger,
        expected_turns={
            unit: {"call_key": call_key, "input_tokens_est": 100}
        },
        turn_index=turn_index,
        turn_path=turn_path,
    )
    assert recovered == 1
    assert turn_index[unit] == turn_record
    assert list(iter_jsonl(turn_path)) == [turn_record]


def test_external_raw_matrix_requires_explicit_complete_treatment_bindings():
    supporter = SupporterGenerationContract.from_config(
        load_config("configs/pm_v2.yaml")
    )
    fixed = {"version": "fixed-fixture"}
    fixed_sha = sha256_text(canonical_json(fixed))
    rows = [
        bind_external_generation_request_log(
            {
                "physical_call_key": key,
                "normalized_finish_reason": "complete",
                "error": None,
            },
            supporter_generation_treatment=supporter.payload(),
            supporter_generation_treatment_sha256=supporter.digest(),
            fixed_seeker_generation_treatment=fixed,
            fixed_seeker_generation_treatment_sha256=fixed_sha,
        )
        for key in ("a" * 64, "b" * 64)
    ]
    report = summarize_external_generation_raw_matrix(
        rows,
        planned_call_keys=["a" * 64, "b" * 64],
        supporter_generation_treatment=supporter.payload(),
        supporter_generation_treatment_sha256=supporter.digest(),
        fixed_seeker_generation_treatment=fixed,
        fixed_seeker_generation_treatment_sha256=fixed_sha,
    )
    assert report["status"] == "PASS"
    assert report["non_complete_finish_reason_count"] == 0

    truncated = [{**rows[0], "normalized_finish_reason": "length"}, rows[1]]
    failed = summarize_external_generation_raw_matrix(
        truncated,
        planned_call_keys=["a" * 64, "b" * 64],
        supporter_generation_treatment=supporter.payload(),
        supporter_generation_treatment_sha256=supporter.digest(),
        fixed_seeker_generation_treatment=fixed,
        fixed_seeker_generation_treatment_sha256=fixed_sha,
    )
    assert failed["status"] == "FAIL"
    assert failed["non_complete_finish_reason_count"] == 1


def test_reported_usage_gate_fails_closed_for_missing_or_nonpositive_prompt_tokens():
    assert reported_prompt_token_error(
        None,
        maximum_prompt_tokens=100,
        stage="paid evaluation",
        require_positive=True,
    ) is not None
    assert reported_prompt_token_error(
        {"prompt_tokens": 0},
        maximum_prompt_tokens=100,
        stage="paid evaluation",
        require_positive=True,
    ) is not None
    assert reported_prompt_token_error(
        {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 13,
        },
        maximum_prompt_tokens=100,
        stage="paid evaluation",
        require_positive=True,
    ) is not None


def test_external_gate_rejects_copied_risk_dimensions():
    rows = []
    for index in range(5):
        copied = float(index % 4)
        rows.append(
            {
                "emotional_support": 1.0 + index,
                "personalization": 1.0 + ((index + 1) % 5),
                "memory_appropriateness": 1.0 + ((index + 2) % 5),
                "factual_grounding": 1.0 + ((index + 3) % 5),
                "temporal_consistency": 1.0 + ((index + 4) % 5),
                "non_intrusiveness": 1.0 + ((index * 2) % 5),
                "selected_context_misuse": copied,
                "unnecessary_exposure": copied,
                "stale_or_conflicting_use": float((index + 1) % 4),
                "unsupported_personal_claim": float((index + 2) % 4),
                "memory_omission": float((index + 3) % 4),
                "strategy_overuse": float((index * 2) % 4),
                "strategy_omission": float((index * 3) % 4),
                "label_reliable": True,
            }
        )
    with pytest.raises(RuntimeError, match="external judge gate failed"):
        validate_external_score_table(
            rows,
            duplicate_exact_match_rate=0.98,
            minimum_reliable_rate=0.80,
            reject_constant_response_dimensions=True,
            reject_constant_risk_dimensions=True,
        )


def test_external_turn_loader_requires_frozen_expected_units(tmp_path):
    path = tmp_path / "turns.jsonl"
    rows = []
    for condition in ("pm_v2", "baseline"):
        rows.append(
            {
                "user_id": "u1",
                "topic_index": 1,
                "seed": 101,
                "simulator_id": "seeker_main",
                "turn_index": 3,
                "condition": condition,
                "interaction_mode": "fixed",
                "context_sha256": "same",
                "seeker_message": "same",
            }
        )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="matrix is incomplete"):
        load_fixed_turns(
            [path],
            conditions=["pm_v2", "baseline"],
            turn_indices=[3],
            expected_units=[
                ("u1", 1, 101, "seeker_main", 3),
                ("u2", 1, 101, "seeker_main", 3),
            ],
        )


def test_joint_reliability_is_diagnostic_when_each_dimension_has_coverage():
    spec = CompositeSpec()
    dimensions = [
        *[f"response.{name}" for name in ResponseDimensions.model_fields],
        *[f"risk.{name}" for name in RiskDimensions.model_fields],
    ]
    labels = []
    for index, failed_dimension in enumerate(dimensions):
        dimension_mad = {name: 0.0 for name in dimensions}
        dimension_mad[failed_dimension] = 1.0
        labels.append(
            ActionLabel(
                state_id=f"s{index}",
                card_id=f"c{index}",
                user_id=f"u{index}",
                semantic_family=f"f{index}",
                action_id="M0+R0",
                response=ResponseDimensions(
                    **{
                        name: 1.0 + float((index * (offset + 1)) % 5)
                        for offset, name in enumerate(ResponseDimensions.model_fields)
                    }
                ),
                risk=RiskDimensions(
                    **{
                        name: float((index * (offset + 1)) % 4)
                        for offset, name in enumerate(RiskDimensions.model_fields)
                    }
                ),
                observed_input_tokens=100,
                retrieval_calls=0,
                judge_families=["a", "b"],
                judge_count=2,
                max_dimension_mad=1.0,
                dimension_mad=dimension_mad,
                label_reliable=False,
                composite_weights_sha256=composite_weights_hash(spec),
            )
        )
    report = validate_judge_table(
        labels,
        minimum_reliable_rate=1.0,
        reliable_mad_threshold=0.75,
        minimum_low_mad_coverage_per_dimension=0.90,
        minimum_low_mad_coverage_per_action_dimension=0.90,
        duplicate_exact_match_rate=1.1,
        maximum_absolute_dimension_correlation=1.1,
        reject_constant_response_dimensions=False,
        reject_constant_risk_dimensions=False,
        composite_spec=spec,
    )
    assert report["status"] == "PASS"
    assert report["reliable_label_rate"] == 0.0
    assert report["joint_reliable_rate_is_diagnostic_only"] is True


def test_cost_match_requires_exact_units_and_gates_estimated_and_observed(tmp_path):
    def rows(condition, tokens):
        return [
            {
                "user_id": "u1",
                "topic_index": 1,
                "seed": 101,
                "simulator_id": "seeker_main",
                "turn_index": index + 1,
                "condition": condition,
                "input_tokens_est": value,
                "input_tokens": value,
            }
            for index, value in enumerate(tokens)
        ]

    treatment = rows("pm_v2", [100, 100])
    baseline = rows("pm_v2_cost_matched_fixed", [105, 105])
    report = compare_cost_matched_token_rows(
        treatment,
        baseline,
        token_field="input_tokens_est",
        maximum_relative_deviation=0.10,
        stage="test",
    )
    assert report["status"] == "PASS"
    failed = compare_cost_matched_token_rows(
        treatment,
        rows("pm_v2_cost_matched_fixed", [130, 130]),
        token_field="input_tokens_est",
        maximum_relative_deviation=0.10,
        stage="test",
    )
    assert failed["status"] == "FAIL"
    with pytest.raises(RuntimeError, match="matrices differ"):
        compare_cost_matched_token_rows(
            treatment,
            baseline[:1],
            token_field="input_tokens_est",
            maximum_relative_deviation=0.10,
            stage="test",
        )

    treatment_path = tmp_path / "treatment.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    treatment_path.write_text(
        "".join(json.dumps(row) + "\n" for row in treatment), encoding="utf-8"
    )
    baseline_path.write_text(
        "".join(json.dumps(row) + "\n" for row in baseline), encoding="utf-8"
    )
    observed = compare_observed_cost_matched_turns(
        treatment_path,
        baseline_path,
        maximum_relative_deviation=0.10,
    )
    assert observed["status"] == "PASS"
    assert observed["n_units"] == 2


def test_evoemo_turn_resume_never_repeats_successful_or_failed_http_attempts(
    tmp_path, monkeypatch, tiny_state, tiny_memories, tiny_strategy
):
    import metacom_pm.pm_v2_evoemo as module

    evoemo_path = tmp_path / "evoemo.json"
    strategy_path = tmp_path / "strategies.jsonl"
    checkpoint_path = tmp_path / "checkpoint.joblib"
    fixed_tracks_path = tmp_path / "fixed_tracks.jsonl"
    fixed_attestation_path = tmp_path / "fixed_attestation.json"
    fixed_contract = FixedSeekerGenerationContract.from_mapping(
        load_config("configs/pm_v2.yaml")["fixed_seeker_generation_treatment"]
    )
    fixed_bound = fixed_contract.bind_endpoint(
        fixed_contract.seeker_endpoint,
        Endpoint(
            "https://seeker.invalid/v1",
            "fixture-seeker",
            "UNSET",
            family="fixture-seeker",
        ),
    )
    fixed_cost_planning = fixed_seeker_cost_planning_contract(
        load_config("configs/pm_v2.yaml")["fixed_seeker_cost_planning"]
    )
    fixed_cost_planning_sha256 = sha256_text(
        canonical_json(fixed_cost_planning)
    )
    planned_budget_gate = {
        "status": "PASS",
        "checks": {
            "api_calls": True,
            "estimated_cost_usd": True,
            "max_input_tokens_per_call": True,
        },
        "limits": {
            "max_api_calls": 3,
            "max_estimated_usd": 10.0,
            "max_input_tokens_per_call": 20_000,
        },
    }
    observed_budget_gate = {
        "status": "PASS",
        "checks": {"fixture": True},
        "limits": planned_budget_gate["limits"],
    }
    for path in (evoemo_path, checkpoint_path, fixed_tracks_path):
        path.write_text("fixture\n", encoding="utf-8")
    write_jsonl(strategy_path, [tiny_strategy.model_dump(mode="json")])
    write_json(
        fixed_attestation_path,
        {
            "parameters": {
                "simulator_id": "sim",
                "max_turns": 3,
                "seeds": [7],
                "fixed_seeker_generation_contract": fixed_bound.payload(),
                "fixed_seeker_generation_contract_sha256": fixed_bound.digest(),
                "fixed_seeker_cost_planning": fixed_cost_planning,
                "fixed_seeker_cost_planning_sha256": (
                    fixed_cost_planning_sha256
                ),
                "planned_budget_gate": planned_budget_gate,
                "observed_budget_gate": observed_budget_gate,
            },
            "expected": {
                "tracks": 1,
                "turns_per_track": 3,
                "completion_truncated_count": 0,
                "planned_budget_gate": planned_budget_gate,
                "observed_budget_gate": observed_budget_gate,
            },
        },
    )
    write_json(
        tmp_path / "summary.json",
        {
            "status": "COMPLETE",
            "completion_truncated_count": 0,
            "fixed_seeker_generation_contract": fixed_bound.payload(),
            "fixed_seeker_generation_contract_sha256": fixed_bound.digest(),
            "fixed_seeker_cost_planning": fixed_cost_planning,
            "fixed_seeker_cost_planning_sha256": fixed_cost_planning_sha256,
            "planned_budget_gate": planned_budget_gate,
            "observed_budget_gate": observed_budget_gate,
        },
    )
    write_jsonl(
        tmp_path / "raw_seeker_calls.jsonl",
        [
            {
                "normalized_finish_reason": "complete",
                "fixed_seeker_generation_contract_sha256": fixed_bound.digest(),
                "fixed_seeker_cost_planning_sha256": fixed_cost_planning_sha256,
            }
        ],
    )

    user = {"id": "u1", "subsequent_topics": [{"idx": 1}]}
    track_key = module._track_key("u1", 1, 7, "sim")
    track = {
        "track_id": "track-1",
        "seeker_turns": ["first", "second", "third"],
        "turn_provenance": [
            {
                "normalized_finish_reason": "complete",
                "fixed_seeker_generation_contract_sha256": fixed_bound.digest(),
                "fixed_seeker_cost_planning_sha256": fixed_cost_planning_sha256,
            }
            for _ in range(3)
        ],
        "fixed_seeker_generation_contract": fixed_bound.payload(),
        "fixed_seeker_generation_contract_sha256": fixed_bound.digest(),
        "fixed_seeker_cost_planning": fixed_cost_planning,
        "fixed_seeker_cost_planning_sha256": fixed_cost_planning_sha256,
    }
    runtime = tiny_state.model_copy(
        update={"provenance": {"exogenous_state_id": "exo-fixture"}}
    )

    class Decision:
        chosen_action = "M0+R0"
        semantic_ood_score = 0.0
        metadata_ood_score = 0.0
        ood_fallback_used = False

        def model_dump(self, mode="json"):
            return {
                "chosen_action": self.chosen_action,
                "semantic_ood_score": self.semantic_ood_score,
                "metadata_ood_score": self.metadata_ood_score,
                "ood_fallback_used": self.ood_fallback_used,
            }

    class FakeModel:
        format_version = "pm-v2.1"
        selection_config = SimpleNamespace(digest=lambda: "selection-hash")

        def choose(self, state):
            return Decision()

    class FailMiddleClient:
        calls = 0
        overrun = False
        truncated = False

        def __init__(self, endpoint):
            self.endpoint = endpoint

        def close(self):
            pass

        def chat(self, *args, **kwargs):
            type(self).calls += 1
            if type(self).overrun:
                return (
                    CallResult(
                        text="overrun response",
                        raw_response={"fixture": True},
                        usage={
                            "prompt_tokens": 999_999,
                            "completion_tokens": 2,
                            "total_tokens": 1_000_001,
                        },
                        latency_ms=1.0,
                        request_hash="request-overrun",
                        provider_finish_reason="stop",
                        normalized_finish_reason="complete",
                    ),
                    None,
                )
            if type(self).calls == 2:
                raise RuntimeError("injected turn failure")
            return (
                CallResult(
                    text=f"supporter-{type(self).calls}",
                    raw_response={"fixture": True},
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                    latency_ms=1.0,
                    request_hash=f"request-{type(self).calls}",
                    provider_finish_reason=(
                        "length" if type(self).truncated else "stop"
                    ),
                    normalized_finish_reason=(
                        "length" if type(self).truncated else "complete"
                    ),
                ),
                None,
            )

    monkeypatch.setattr(
        module,
        "require_content_addressed_attestation",
        lambda *args, **kwargs: {"attestation_sha256": "a" * 64},
    )
    monkeypatch.setattr(module, "load_evoemo", lambda path: [user])
    monkeypatch.setattr(module, "_load_fixed_tracks", lambda path: {track_key: track})
    monkeypatch.setattr(
        module,
        "_fixed_context_before_turn",
        lambda fixed_track, turn: [{"role": "seeker", "content": f"ctx-{turn}"}],
    )
    monkeypatch.setattr(module, "build_evo_memory", lambda value: (tiny_memories, {}))
    monkeypatch.setattr(
        module,
        "make_evo_runtime_state",
        lambda *args, **kwargs: runtime,
    )
    monkeypatch.setattr(module, "runtime_to_pmv2_state", lambda *args, **kwargs: object())
    monkeypatch.setattr(module.PMV2Model, "load", lambda path: FakeModel())
    monkeypatch.setattr(module, "OpenAICompatibleClient", FailMiddleClient)

    endpoint = Endpoint(
        "https://invalid.example", "fixture-model", "UNSET", family="fixture"
    )
    gates = {
        "maximum_severe_ood_fallback_rate": 1.0,
        "maximum_no_feasible_fallback_rate": 1.0,
        "minimum_m0_rate": 0.0,
        "minimum_r0_rate": 0.0,
        "minimum_m0_r0_rate": 0.0,
        "minimum_nonfallback_rate": 0.0,
        "maximum_action_share": 1.0,
        "minimum_action_entropy_bits": 0.0,
    }
    out_dir = tmp_path / "out"
    kwargs = {
        "project_root": Path(__file__).resolve().parents[1],
        "generator_endpoint": endpoint,
        "supporter_generation_contract": SupporterGenerationContract.from_config(
            load_config("configs/pm_v2.yaml")
        ),
        "fixed_seeker_generation_contract": fixed_bound.payload(),
        "fixed_seeker_generation_contract_sha256": fixed_bound.digest(),
        "simulator_id": "sim",
        "fixed_tracks_attestation_path": fixed_attestation_path,
        "condition": "fixture_fixed",
        "max_turns": 3,
        "seeds": [7],
        "evaluation_unit_contract": {
            "protocol": EVALUATION_UNIT_CONTRACT_PROTOCOL,
            "evaluation_turn_indices": [1, 3],
            "expected_unit_count": 2,
            "expected_units_sha256": sha256_text(
                canonical_json(
                    [("u1", 1, 7, "sim", 1), ("u1", 1, 7, "sim", 3)]
                )
            ),
        },
        "max_scenarios": 1,
        "max_api_calls": 2,
        "max_estimated_usd": 10.0,
        "max_input_tokens_per_call": 12000,
        "input_usd_per_mtok": 0.15,
        "output_usd_per_mtok": 0.60,
        "input_token_safety_factor": 1.5,
        "fail_on_reported_input_overrun": True,
        "action_preflight_gates": gates,
        "maximum_cost_matched_relative_deviation": 0.10,
    }
    dry_run = run_pmv2_fixed_evoemo(
        evoemo_path,
        strategy_path,
        checkpoint_path,
        fixed_tracks_path,
        out_dir,
        run=False,
        **kwargs,
    )
    accepted = dry_run["cost_estimate"]["cost_estimate_sha256"]
    dry_plan = list(iter_jsonl(out_dir / "call_plan.jsonl"))
    assert dry_run["cost_estimate"]["input_token_safety_factor"] == 1.5
    assert all(
        row["input_tokens_est"] > row["raw_input_tokens_est"]
        for row in dry_plan
    )
    assert [row["turn_index"] for row in dry_plan] == [1, 3]
    assert {row["max_output_tokens"] for row in dry_plan} == {300}
    assert all(
        row["supporter_generation_treatment_sha256"]
        == kwargs["supporter_generation_contract"].digest()
        for row in dry_plan
    )
    assert dry_run["preflight"]["evaluation_turn_indices"] == [1, 3]
    assert dry_run["preflight"]["all_turn_diagnostic"]["n_states"] == 3
    with pytest.raises(RuntimeError, match="generation incomplete"):
        run_pmv2_fixed_evoemo(
            evoemo_path,
            strategy_path,
            checkpoint_path,
            fixed_tracks_path,
            out_dir,
            run=True,
            accept_cost_estimate_sha256=accepted,
            **kwargs,
        )
    assert FailMiddleClient.calls == 2
    assert [row["turn_index"] for row in iter_jsonl(out_dir / "turns.jsonl")] == [1]
    ledger_before = list(iter_jsonl(out_dir / "physical_attempt_ledger.jsonl"))
    assert [row["event"] for row in ledger_before] == [
        "STARTED",
        "SUCCEEDED",
        "STARTED",
        "FAILED",
    ]
    failed_run_raw = list(iter_jsonl(out_dir / "raw_api_calls.jsonl"))
    assert len(failed_run_raw) == 2
    assert all(
        row["supporter_generation_treatment"]
        == kwargs["supporter_generation_contract"].payload()
        and row["supporter_generation_treatment_sha256"]
        == kwargs["supporter_generation_contract"].digest()
        and row["fixed_seeker_generation_treatment"]
        == kwargs["fixed_seeker_generation_contract"]
        and row["fixed_seeker_generation_treatment_sha256"]
        == kwargs["fixed_seeker_generation_contract_sha256"]
        for row in failed_run_raw
    )
    assert failed_run_raw[1]["normalized_finish_reason"] is None
    assert failed_run_raw[1]["error"] == "RuntimeError: injected turn failure"
    saved_plan_before = (out_dir / "call_plan.jsonl").read_bytes()
    saved_estimate_before = (out_dir / "cost_estimate.json").read_bytes()
    with pytest.raises(RuntimeError, match="forbids changing its call plan or cost"):
        run_pmv2_fixed_evoemo(
            evoemo_path,
            strategy_path,
            checkpoint_path,
            fixed_tracks_path,
            out_dir,
            run=False,
            **{**kwargs, "max_estimated_usd": 11.0},
        )
    assert (out_dir / "call_plan.jsonl").read_bytes() == saved_plan_before
    assert (out_dir / "cost_estimate.json").read_bytes() == saved_estimate_before

    # A resumed run sees both successful turn outputs and the spent failed call
    # key.  It performs zero additional HTTP attempts.
    with pytest.raises(RuntimeError, match="generation incomplete"):
        run_pmv2_fixed_evoemo(
            evoemo_path,
            strategy_path,
            checkpoint_path,
            fixed_tracks_path,
            out_dir,
            run=True,
            accept_cost_estimate_sha256=accepted,
            **kwargs,
        )
    assert FailMiddleClient.calls == 2
    resumed = read_json(out_dir / "generation_summary.json")
    assert resumed["new_physical_http_attempts"] == 0
    assert resumed["historical_physical_http_attempts"] == 2
    assert resumed["total_physical_http_attempts"] == 2
    assert resumed["raw_generation_contract_gate"]["status"] == "FAIL"
    assert list(iter_jsonl(out_dir / "physical_attempt_ledger.jsonl")) == ledger_before

    FailMiddleClient.calls = 0
    FailMiddleClient.overrun = True
    overrun_dir = tmp_path / "overrun"
    overrun_dry_run = run_pmv2_fixed_evoemo(
        evoemo_path,
        strategy_path,
        checkpoint_path,
        fixed_tracks_path,
        overrun_dir,
        run=False,
        **kwargs,
    )
    with pytest.raises(RuntimeError, match="generation incomplete"):
        run_pmv2_fixed_evoemo(
            evoemo_path,
            strategy_path,
            checkpoint_path,
            fixed_tracks_path,
            overrun_dir,
            run=True,
            accept_cost_estimate_sha256=overrun_dry_run["cost_estimate"][
                "cost_estimate_sha256"
            ],
            **kwargs,
        )
    assert FailMiddleClient.calls == 1
    overrun_ledger = list(
        iter_jsonl(overrun_dir / "physical_attempt_ledger.jsonl")
    )
    assert [row["event"] for row in overrun_ledger] == ["STARTED", "FAILED"]
    assert overrun_ledger[-1]["usage"]["prompt_tokens"] == 999_999
    overrun_summary = read_json(overrun_dir / "generation_summary.json")
    assert overrun_summary["aborted_on_input_token_overrun"] is True

    FailMiddleClient.calls = 0
    FailMiddleClient.overrun = False
    FailMiddleClient.truncated = True
    truncated_dir = tmp_path / "truncated"
    truncated_dry_run = run_pmv2_fixed_evoemo(
        evoemo_path,
        strategy_path,
        checkpoint_path,
        fixed_tracks_path,
        truncated_dir,
        run=False,
        **kwargs,
    )
    with pytest.raises(RuntimeError, match="generation incomplete"):
        run_pmv2_fixed_evoemo(
            evoemo_path,
            strategy_path,
            checkpoint_path,
            fixed_tracks_path,
            truncated_dir,
            run=True,
            accept_cost_estimate_sha256=truncated_dry_run["cost_estimate"][
                "cost_estimate_sha256"
            ],
            **kwargs,
        )
    assert FailMiddleClient.calls == 1
    assert not (truncated_dir / "turns.jsonl").exists()
    truncated_ledger = list(
        iter_jsonl(truncated_dir / "physical_attempt_ledger.jsonl")
    )
    assert [row["event"] for row in truncated_ledger] == ["STARTED", "FAILED"]
    assert "output-token limit" in truncated_ledger[-1]["error"]
    truncated_raw = list(iter_jsonl(truncated_dir / "raw_api_calls.jsonl"))
    assert truncated_raw[0]["provider_finish_reason"] == "length"
    assert truncated_raw[0]["normalized_finish_reason"] == "length"
    assert truncated_raw[0]["completion_truncated"] is True
    assert truncated_raw[0]["supporter_generation_treatment"] == kwargs[
        "supporter_generation_contract"
    ].payload()
    assert truncated_raw[0]["fixed_seeker_generation_treatment"] == kwargs[
        "fixed_seeker_generation_contract"
    ]
    truncated_summary = read_json(truncated_dir / "generation_summary.json")
    assert truncated_summary["aborted_on_completion_gate_error"] is True
    assert truncated_summary["completed_turns"] == 0


def test_generation_evaluation_unit_contract_is_sparse_exact_and_content_addressed():
    scenarios = [
        ({"id": "u2"}, {"idx": 2}),
        ({"id": "u1"}, {"idx": 1}),
    ]
    units = build_generation_evaluation_units(
        scenarios,
        seeds=[101],
        simulator_id="seeker_main",
        evaluation_turn_indices=[3, 8],
    )
    assert units == [
        ("u1", 1, 101, "seeker_main", 3),
        ("u1", 1, 101, "seeker_main", 8),
        ("u2", 2, 101, "seeker_main", 3),
        ("u2", 2, 101, "seeker_main", 8),
    ]
    contract = {
        "protocol": EVALUATION_UNIT_CONTRACT_PROTOCOL,
        "evaluation_turn_indices": [3, 8],
        "expected_unit_count": 4,
        "expected_units_sha256": sha256_text(canonical_json(units)),
    }
    normalized, observed = require_generation_evaluation_unit_contract(
        contract,
        scenarios=scenarios,
        seeds=[101],
        simulator_id="seeker_main",
        max_turns=10,
    )
    assert normalized == contract
    assert observed == units
    with pytest.raises(RuntimeError, match="differs from the frozen exact unit set"):
        require_generation_evaluation_unit_contract(
            {**contract, "expected_unit_count": 20},
            scenarios=scenarios,
            seeds=[101],
            simulator_id="seeker_main",
            max_turns=10,
        )


def test_external_pointwise_schema_smoke_is_exact_anonymous_two_by_two_matrix():
    unit = ("u1", 1, 101, "seeker_main", 3)
    turn_row = {
        "user_id": "u1",
        "state_id": "state_0123456789abcdef",
        "card_id": "card_0123456789abcdef",
        "seeker_message": "I feel unsettled today.",
        "context_before_turn": [
            {"role": "seeker", "content": "It has been a hard week."}
        ],
        "supporter_message": "That sounds exhausting; what feels hardest right now?",
        "selected_memory": [],
        "selected_strategy": [],
    }
    endpoints = [
        Endpoint(
            "https://example.invalid/v1",
            "judge-a",
            "IGNORED",
            family="family_a",
        ),
        Endpoint(
            "https://api.anthropic.com/v1",
            "judge-b",
            "IGNORED",
            family="family_b",
        ),
    ]
    plan, execution = build_pointwise_schema_smoke_plan(
        turn_row=turn_row,
        authorized_user_context="{}",
        unit=unit,
        condition="pm_v2",
        endpoints=endpoints,
        pricing_usd_per_mtok={
            "family_a": {"input": 1.0, "output": 2.0},
            "family_b": {"input": 3.0, "output": 4.0},
        },
        input_token_safety_factor=1.25,
        judge_seed=3701,
        estimated_response_output_tokens=600,
        estimated_risk_output_tokens=700,
        study_freeze_sha256="f" * 64,
    )
    assert len(plan) == len(execution) == 4
    assert {
        (row["judge_family"], row["judge_type"]) for row in plan
    } == {
        ("family_a", "response"),
        ("family_a", "risk"),
        ("family_b", "response"),
        ("family_b", "risk"),
    }
    assert all(row["request_payload_includes_schema"] for row in plan)
    assert all(row["condition_identity_hidden_from_prompt"] for row in plan)
    assert len({row["physical_call_key"] for row in plan}) == 4


def test_full_external_requires_schema_smoke_before_core_client_path():
    source = Path("scripts/25_eval_pm_v2_external.py").read_text(encoding="utf-8")
    smoke_gate = source.index("pointwise_schema_smoke_verification = (")
    full_core = source.index("result = run_external_response_evaluation(")
    assert smoke_gate < full_core
    freeze_source = Path("scripts/26_freeze_pm_v2_study.py").read_text(
        encoding="utf-8"
    )
    assert POINTWISE_SCHEMA_SMOKE_PROTOCOL in freeze_source


def test_external_eval_requires_exact_reference_raw_generation_gate():
    script = Path("scripts/25_eval_pm_v2_external.py")
    spec = importlib.util.spec_from_file_location(
        "pmv22_external_eval_raw_gate_script", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gate = {
        "status": "PASS",
        "expected_rows": 4,
        "observed_rows": 4,
        **{
            key: True
            for key in module.REFERENCE_RAW_GATE_BOOLEAN_CHECKS
        },
    }
    summary = {"raw_generation_contract_gate": gate}
    attestation = {
        "parameters": {"raw_generation_contract_gate": gate},
        "expected": {"raw_generation_contract_gate": gate},
    }
    assert module.require_reference_raw_generation_contract_gate(
        summary=summary, attestation=attestation, row_count=4
    ) == gate

    mixed = {**gate, "policy_lock_exact": False}
    with pytest.raises(RuntimeError, match="exact PASS raw"):
        module.require_reference_raw_generation_contract_gate(
            summary={"raw_generation_contract_gate": mixed},
            attestation=attestation,
            row_count=4,
        )


def test_freeze_finish_reason_gate_accepts_zero_buckets_but_rejects_nonzero():
    script = Path("scripts/26_freeze_pm_v2_study.py")
    spec = importlib.util.spec_from_file_location(
        "pmv22_freeze_finish_reason_gate_script", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    initialized_counts = {
        "complete": 2,
        "length": 0,
        "tool_call": 0,
        "content_filter": 0,
        "unknown": 0,
    }
    assert module.require_complete_only_finish_reason_counts(
        initialized_counts, expected_rows=2
    ) == initialized_counts

    with pytest.raises(RuntimeError, match="incomplete calls"):
        module.require_complete_only_finish_reason_counts(
            {**initialized_counts, "length": 1}, expected_rows=2
        )


def test_external_turn_gate_rejects_mixed_or_noncomplete_generation(tmp_path):
    script = Path("scripts/25_eval_pm_v2_external.py")
    spec = importlib.util.spec_from_file_location("pmv22_external_eval_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract = SupporterGenerationContract.from_config(load_config("configs/pm_v2.yaml"))
    fixed_contract = FixedSeekerGenerationContract.from_mapping(
        load_config("configs/pm_v2.yaml")["fixed_seeker_generation_treatment"]
    )
    fixed_bound = fixed_contract.bind_endpoint(
        fixed_contract.seeker_endpoint,
        Endpoint(
            "https://seeker.invalid/v1",
            "seeker",
            "UNSET",
            family="seeker-family",
        ),
    )
    turn_path = tmp_path / "turns.jsonl"
    row = {
        "user_id": "u1",
        "topic_index": 1,
        "condition": "baseline",
        "seed": 7,
        "simulator_id": "sim",
        "turn_index": 3,
        "interaction_mode": "fixed",
        "supporter_message": "I hear how difficult this is.",
        "normalized_finish_reason": "complete",
        "supporter_generation_treatment": contract.payload(),
        "supporter_generation_treatment_sha256": contract.digest(),
        "fixed_seeker_generation_treatment": fixed_bound.payload(),
        "fixed_seeker_generation_treatment_sha256": fixed_bound.digest(),
    }
    write_jsonl(turn_path, [row])
    assert len(
        module.require_treatment_bound_turn_file(
            turn_path,
            expected_conditions=["baseline"],
            expected_units=[("u1", 1, 7, "sim", 3)],
            treatment=contract.payload(),
            treatment_sha256=contract.digest(),
            fixed_seeker_treatment=fixed_bound.payload(),
            fixed_seeker_treatment_sha256=fixed_bound.digest(),
        )
    ) == 1

    write_jsonl(turn_path, [{**row, "normalized_finish_reason": "length"}])
    with pytest.raises(RuntimeError, match="mixed or non-complete"):
        module.require_treatment_bound_turn_file(
            turn_path,
            expected_conditions=["baseline"],
            expected_units=[("u1", 1, 7, "sim", 3)],
            treatment=contract.payload(),
            treatment_sha256=contract.digest(),
            fixed_seeker_treatment=fixed_bound.payload(),
            fixed_seeker_treatment_sha256=fixed_bound.digest(),
        )

    write_jsonl(
        turn_path,
        [{**row, "supporter_generation_treatment_sha256": "0" * 64}],
    )
    with pytest.raises(RuntimeError, match="mixed or non-complete"):
        module.require_treatment_bound_turn_file(
            turn_path,
            expected_conditions=["baseline"],
            expected_units=[("u1", 1, 7, "sim", 3)],
            treatment=contract.payload(),
            treatment_sha256=contract.digest(),
            fixed_seeker_treatment=fixed_bound.payload(),
            fixed_seeker_treatment_sha256=fixed_bound.digest(),
        )


def test_freeze_reference_baseline_contract_rejects_v1_and_truncation(
    tmp_path, monkeypatch
):
    script = Path("scripts/26_freeze_pm_v2_study.py")
    spec = importlib.util.spec_from_file_location("pmv22_freeze_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "require_content_addressed_attestation",
        lambda *args, **kwargs: {"attestation_sha256": "a" * 64},
    )
    contract = SupporterGenerationContract.from_config(load_config("configs/pm_v2.yaml"))
    fixed_contract = FixedSeekerGenerationContract.from_mapping(
        load_config("configs/pm_v2.yaml")["fixed_seeker_generation_treatment"]
    )
    fixed_bound = fixed_contract.bind_endpoint(
        fixed_contract.seeker_endpoint,
        Endpoint(
            "https://seeker.invalid/v1",
            "seeker",
            "UNSET",
            family="seeker-family",
        ),
    )
    unit = ("u1", 1, 7, "sim", 3)
    condition = "no_memory_r0"
    evaluation_contract = {
        "protocol": EVALUATION_UNIT_CONTRACT_PROTOCOL,
        "evaluation_turn_indices": [3],
        "expected_unit_count": 1,
        "expected_units_sha256": sha256_text(canonical_json([unit])),
    }
    evidence_processing_contracts = {
        condition: {
            "protocol": "pm-v2.2-reference-baseline-evidence-processing-v1",
            "memory_processing": "none",
            "strategy_processing": "none",
        }
    }
    evidence_contract = evidence_processing_contracts[condition]
    evidence_processing_contracts_sha256 = sha256_text(
        canonical_json(evidence_processing_contracts)
    )
    evidence_filter_config_sha256 = "e" * 64
    evidence_filter_model_binding = {"checkpoint_sha256": "f" * 64}
    strategy_bank_approval = {"status": "APPROVED"}
    policy_checkpoint = tmp_path / "pm_v2.joblib"
    policy_training_report = tmp_path / "policy_training_report.json"
    policy_checkpoint.write_bytes(b"frozen-policy\n")
    policy_training_report.write_text("{}\n", encoding="utf-8")
    policy_lock = {
        "policy_checkpoint_sha256": sha256_file(policy_checkpoint),
        "policy_training_report_sha256": sha256_file(policy_training_report),
        "policy_lock_timing": module.POLICY_LOCK_TIMING,
        "post_generation_policy_tuning_prohibited": (
            module.POST_GENERATION_POLICY_TUNING_PROHIBITED
        ),
    }
    common = {
        "conditions": [condition],
        "supporter_generation_treatment": contract.payload(),
        "supporter_generation_treatment_sha256": contract.digest(),
        "fixed_seeker_generation_treatment": fixed_bound.payload(),
        "fixed_seeker_generation_treatment_sha256": fixed_bound.digest(),
        "evidence_processing_contracts": evidence_processing_contracts,
        "evidence_processing_contracts_sha256": (
            evidence_processing_contracts_sha256
        ),
        "evidence_filter_config_sha256": evidence_filter_config_sha256,
        "evidence_filter_model": evidence_filter_model_binding,
        "strategy_bank_approval": strategy_bank_approval,
        **policy_lock,
        "simulator_id": "sim",
        "max_turns": 10,
        "seeds": [7],
        "evaluation_unit_contract": evaluation_contract,
        "paid_generation_scope": "frozen_evaluation_turns_only",
    }
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    turns = bundle / "turns.jsonl"
    turn_row = {
        "user_id": "u1",
        "topic_index": 1,
        "condition": condition,
        "seed": 7,
        "simulator_id": "sim",
        "turn_index": 3,
        "interaction_mode": "fixed",
        "supporter_message": "That sounds painful.",
        "normalized_finish_reason": "complete",
        "supporter_generation_treatment": contract.payload(),
        "supporter_generation_treatment_sha256": contract.digest(),
        "fixed_seeker_generation_treatment": fixed_bound.payload(),
        "fixed_seeker_generation_treatment_sha256": fixed_bound.digest(),
        "evidence_processing_contract": evidence_contract,
        "evidence_processing_contract_sha256": sha256_text(
            canonical_json(evidence_contract)
        ),
        **policy_lock,
    }
    write_jsonl(turns, [turn_row])
    plan = [
        {
            **{key: turn_row[key] for key in (
                "user_id",
                "topic_index",
                "condition",
                "seed",
                "simulator_id",
                "turn_index",
            )},
            "max_output_tokens": contract.max_output_tokens,
            "call_key": "c" * 64,
            "supporter_generation_treatment": contract.payload(),
            "supporter_generation_treatment_sha256": contract.digest(),
            "fixed_seeker_generation_treatment": fixed_bound.payload(),
            "fixed_seeker_generation_treatment_sha256": fixed_bound.digest(),
            "evidence_processing_contract": evidence_contract,
            "evidence_processing_contract_sha256": sha256_text(
                canonical_json(evidence_contract)
            ),
            **policy_lock,
        }
    ]
    write_jsonl(bundle / "call_plan.jsonl", plan)
    cost_payload = {
        "supporter_generation_treatment": contract.payload(),
        "supporter_generation_treatment_sha256": contract.digest(),
        "fixed_seeker_generation_treatment": fixed_bound.payload(),
        "fixed_seeker_generation_treatment_sha256": fixed_bound.digest(),
        "evidence_processing_contracts": evidence_processing_contracts,
        "evidence_processing_contracts_sha256": (
            evidence_processing_contracts_sha256
        ),
        "evidence_filter_config_sha256": evidence_filter_config_sha256,
        "evidence_filter_model": evidence_filter_model_binding,
        **policy_lock,
        "call_plan_sha256": sha256_text(canonical_json(plan)),
    }
    write_json(
        bundle / "cost_estimate.json",
        {
            **cost_payload,
            "cost_estimate_sha256": sha256_text(canonical_json(cost_payload)),
            "budget_gate": {"status": "PASS"},
        },
    )
    write_jsonl(
        bundle / "raw_api_calls.jsonl",
        [
            {
                "normalized_finish_reason": "complete",
                "supporter_generation_treatment": contract.payload(),
                "supporter_generation_treatment_sha256": contract.digest(),
                "fixed_seeker_generation_treatment": fixed_bound.payload(),
                "fixed_seeker_generation_treatment_sha256": fixed_bound.digest(),
                "condition": condition,
                "evidence_processing_contract": evidence_contract,
                "evidence_processing_contract_sha256": sha256_text(
                    canonical_json(evidence_contract)
                ),
                **policy_lock,
                "physical_call_key": "c" * 64,
            }
        ],
    )
    write_jsonl(
        bundle / "physical_attempt_ledger.jsonl",
        [{"event": "STARTED"}, {"event": "SUCCEEDED"}],
    )
    manifest_payload = {
        "stage": module.PMV22_REFERENCE_BASELINE_STAGE,
        **common,
        "generator_model": "generator-model",
        "generator_family": "generator-family",
        "generator_base_url": "https://generator.invalid/v1",
    }
    manifest = {
        **manifest_payload,
        "manifest_sha256": sha256_text(canonical_json(manifest_payload)),
    }
    write_json(bundle / "run_manifest.json", manifest)
    raw_generation_contract_gate = {
        "status": "PASS",
        "expected_rows": 1,
        "observed_rows": 1,
        "row_count_exact": True,
        "call_keys_exact": True,
        "finish_reasons_complete": True,
        "errors_absent": True,
        "supporter_generation_treatment_exact": True,
        "fixed_seeker_generation_treatment_exact": True,
        "evidence_processing_contract_exact": True,
        "policy_lock_exact": True,
    }
    write_json(
        bundle / "generation_summary.json",
        {
            **common,
            "status": "COMPLETE",
            "completed_turns": 1,
            "non_complete_finish_reason_count": 0,
            "raw_generation_contract_gate": raw_generation_contract_gate,
        },
    )
    attestation = {
        "stage": module.PMV22_REFERENCE_BASELINE_STAGE,
        "parameters": {
            **common,
            "raw_generation_contract_gate": raw_generation_contract_gate,
        },
        "expected": {
            "raw_generation_contract_gate": raw_generation_contract_gate,
        },
        "outputs": {"turns": {"rows": 1}},
    }
    attestation_path = bundle / "artifact_attestation.json"
    write_json(attestation_path, attestation)
    kwargs = {
        "attestation_path": attestation_path,
        "turns_path": turns,
        "manifest_path": bundle / "run_manifest.json",
        "summary_path": bundle / "generation_summary.json",
        "evoemo_path": tmp_path / "evoemo.json",
        "strategy_bank_path": tmp_path / "strategy.jsonl",
        "policy_checkpoint_path": policy_checkpoint,
        "policy_training_report_path": policy_training_report,
        "pm_v2_config_path": tmp_path / "pm_v2.yaml",
        "evidence_filter_checkpoint_path": tmp_path / "evidence_filter.joblib",
        "evidence_filter_report_path": tmp_path / "evidence_filter_report.json",
        "evidence_filter_attestation_path": tmp_path
        / "evidence_filter_attestation.json",
        "strategy_bank_approval_path": tmp_path / "strategy_bank_approval.json",
        "fixed_tracks_path": tmp_path / "tracks.jsonl",
        "fixed_tracks_attestation_path": tmp_path / "tracks_attestation.json",
        "conditions": [condition],
        "expected_units": [unit],
        "treatment": contract.payload(),
        "treatment_sha256": contract.digest(),
        "fixed_seeker_treatment": fixed_bound.payload(),
        "fixed_seeker_treatment_sha256": fixed_bound.digest(),
        "evidence_processing_contracts": evidence_processing_contracts,
        "evidence_processing_contracts_sha256": (
            evidence_processing_contracts_sha256
        ),
        "evidence_filter_config_sha256": evidence_filter_config_sha256,
        "evidence_filter_model_binding": evidence_filter_model_binding,
        "strategy_bank_approval": strategy_bank_approval,
        "generator_endpoint": {
            "model": "generator-model",
            "family": "generator-family",
            "base_url": "https://generator.invalid/v1",
        },
        "simulator_id": "sim",
        "max_turns": 10,
        "seeds": [7],
        "evaluation_unit_contract": evaluation_contract,
    }
    verified = module.require_pmv22_reference_baseline_bundle(**kwargs)
    assert verified["stage"] == module.PMV22_REFERENCE_BASELINE_STAGE
    assert verified["turns_rows"] == 1

    write_jsonl(turns, [{**turn_row, "normalized_finish_reason": "length"}])
    with pytest.raises(RuntimeError, match="mixed, truncated"):
        module.require_pmv22_reference_baseline_bundle(**kwargs)
    write_jsonl(turns, [turn_row])
    write_json(attestation_path, {**attestation, "stage": "evoemo_generation"})
    with pytest.raises(RuntimeError, match="legacy/V1"):
        module.require_pmv22_reference_baseline_bundle(**kwargs)


def test_freeze_requires_training_report_checkpoint_hash_binding(tmp_path):
    script = Path("scripts/26_freeze_pm_v2_study.py")
    spec = importlib.util.spec_from_file_location("pmv22_freeze_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = tmp_path / "pm_v2.joblib"
    checkpoint.write_bytes(b"selected-policy\n")
    expected = sha256_file(checkpoint)
    assert (
        module.require_training_report_checkpoint_binding(
            {"checkpoint_sha256": expected}, checkpoint
        )
        == expected
    )
    with pytest.raises(RuntimeError, match="checkpoint hash"):
        module.require_training_report_checkpoint_binding(
            {"checkpoint_sha256": "0" * 64}, checkpoint
        )


def test_schema_smoke_second_family_client_failure_spends_zero_attempts(
    tmp_path, monkeypatch
):
    import metacom_pm.pm_v2_external_schema_smoke as module

    endpoints = [
        Endpoint("https://a.invalid/v1", "a", "IGNORED", family="family_a"),
        Endpoint("https://b.invalid/v1", "b", "IGNORED", family="family_b"),
    ]
    first_client = SimpleNamespace(close_calls=0)
    first_client.close = lambda: setattr(
        first_client, "close_calls", first_client.close_calls + 1
    )
    creations = 0

    def fake_make_client(endpoint):
        nonlocal creations
        creations += 1
        if creations == 2:
            raise RuntimeError("family-b credential failure")
        return first_client

    monkeypatch.setattr(module, "make_client", fake_make_client)
    ledger = PersistentAttemptLedger(
        tmp_path / "ledger.jsonl",
        stage="pm_v2_external_pointwise_schema_smoke",
        expected_calls={"planned-call": 1},
        maximum_total_attempts=4,
    )
    with pytest.raises(RuntimeError, match="family-b credential failure"):
        precreate_pointwise_schema_smoke_clients(endpoints)
    assert ledger.started_attempts == 0
    assert not (tmp_path / "ledger.jsonl").exists()
    assert first_client.close_calls == 1


def test_forced_swap_is_deterministic_and_any_order_disagreement_is_tie():
    units = [(f"u{index}", 1, 101, "seeker_main", 3) for index in range(10)]
    first = select_forced_swap_units(units, sample_units=4, seed=9173)
    second = select_forced_swap_units(list(reversed(units)), sample_units=4, seed=9173)
    assert first == second
    assert (
        resolve_dual_order_preference(
            "pm_v2",
            "pm_v2_cost_matched_fixed",
            treatment="pm_v2",
            baseline="pm_v2_cost_matched_fixed",
        )
        == "tie"
    )


def test_external_action_preflight_excludes_fallback_actions_from_learned_metrics():
    rows = [
        {"chosen_action": "M0+R0", "fallback_type": "severe_ood"}
        for _ in range(8)
    ] + [
        {"chosen_action": "ME+RS", "fallback_type": None}
        for _ in range(2)
    ]
    report = summarize_action_preflight(
        rows,
        condition="pm_v2",
        gates={
            "maximum_severe_ood_fallback_rate": 1.0,
            "maximum_no_feasible_fallback_rate": 1.0,
            "minimum_m0_rate": 0.05,
            "minimum_r0_rate": 0.10,
            "minimum_m0_r0_rate": 0.05,
            "minimum_nonfallback_rate": 0.0,
            "maximum_action_share": 0.50,
            "minimum_action_entropy_bits": 1.0,
        },
    )
    assert report["n_learned_policy_states"] == 2
    assert report["n_fallback_states"] == 8
    assert report["m0_rate"] == 0.0
    assert report["r0_rate"] == 0.0
    assert report["m0_r0_rate"] == 0.0
    assert report["maximum_action_share"] == 1.0
    assert report["action_entropy_bits"] == 0.0
    assert report["distinct_actions"] == 1
    assert report["learned_action_distribution"] == {"ME+RS": 2}
    assert report["overall_action_distribution"] == {"M0+R0": 8, "ME+RS": 2}
    assert report["status"] == "FAIL"


def test_forced_swap_all_tie_constant_panel_is_never_key_claim_reportable():
    gate = forced_swap_compatibility_gate(
        analysis={
            "dual_order_resolved_preference": {
                "pm_v2": 0,
                "pm_v2_cost_matched_fixed": 0,
                "tie": 40,
            },
            "all_resolved_preferences_tie": True,
            "family_support_delta_direction_agreement": True,
            "cross_family_support_delta_correlation": None,
            "cross_family_constant_delta": {"family_a": True, "family_b": True},
            "order_disagreement_rate": 0.0,
            "support_delta_cluster_bootstrap_ci": {"lower": 0.0, "upper": 0.0},
        },
        endpoint_families=("family_a", "family_b"),
        compatibility_thresholds={
            "maximum_order_disagreement_rate": 0.25,
            "minimum_schema_success_rate": 1.0,
            "minimum_cross_family_support_delta_correlation": 0.50,
            "require_cross_family_direction_agreement": True,
            "minimum_support_delta_ci_upper_for_continuation": 0.0,
        },
        schema_success_rate=1.0,
        schema_successes=160,
        schema_attempts=160,
        schema_expected_calls=160,
        observed_cost_match_pass=True,
        futility_reason=None,
    )
    assert gate["status"] == "NONREPORTABLE"
    assert gate["checks"]["resolved_preference_sensitivity"] is False
    assert gate["checks"]["per_family_support_delta_variation"] is False
    assert gate["checks"]["cross_family_support_delta_correlation"] is False
    assert gate["cross_family_correlation_constant_waiver"] is False


def test_forced_swap_direction_requires_nonzero_signal_from_every_family():
    assert not forced_swap_family_direction_agreement(
        {"family_a": 0.0, "family_b": 0.25}
    )
    assert not forced_swap_family_direction_agreement(
        {"family_a": -0.25, "family_b": 0.25}
    )
    assert forced_swap_family_direction_agreement(
        {"family_a": -0.10, "family_b": -0.25}
    )


def test_nonfutile_negative_forced_swap_never_verifies_efficacy():
    gate = forced_swap_efficacy_gate(
        analysis={
            "support_delta_cluster_bootstrap_ci": {
                "lower": -0.20,
                "upper": -0.05,
            },
            "family_support_delta_treatment_minus_baseline": {
                "family_a": -0.10,
                "family_b": -0.12,
            },
            "dual_order_resolved_preference": {
                "pm_v2": 2,
                "pm_v2_cost_matched_fixed": 11,
                "tie": 27,
            },
        },
        thresholds={
            "treatment": "pm_v2",
            "baseline": "pm_v2_cost_matched_fixed",
            "minimum_support_delta_ci_lower_for_advantage": 0.0,
            "require_positive_support_delta_every_family": True,
            "minimum_resolved_preference_margin": 1,
        },
        compatibility_pass=True,
    )
    assert gate["status"] == "NOT_VERIFIED"
    assert not gate["checks"]["support_ci_lower_advantage"]
    assert not gate["checks"]["positive_support_delta_every_family"]
    assert not gate["checks"]["resolved_preference_margin"]


def test_forced_swap_failed_call_persists_terminal_before_futility_stop(tmp_path):
    endpoint = Endpoint(
        base_url="https://example.invalid/v1",
        model="judge",
        family="family_a",
        api_key_env="IGNORED",
    )
    call_key = physical_call_key(
        stage="pm_v2_forced_swap_key_claim",
        record_ids={
            "unit_id": "u1",
            "order_variant": 0,
            "judge_family": "family_a",
        },
        prompt_sha256="f" * 64,
        endpoint=endpoint,
        request_parameters={"retries": 1},
    )
    ledger = PersistentAttemptLedger(
        tmp_path / "judge_call_ledger.jsonl",
        stage="pm_v2_forced_swap_key_claim",
        expected_calls={call_key: 1},
        maximum_total_attempts=4,
    )
    reservation = ledger.reserve(
        call_key,
        record_ids={
            "unit_id": "u1",
            "order_variant": 0,
            "judge_family": "family_a",
        },
        prompt_sha256="f" * 64,
    )
    ledger.finish(
        reservation,
        succeeded=False,
        request_hash=None,
        usage=None,
        error="RuntimeError: endpoint failed",
    )
    reason = forced_swap_schema_futility_reason(
        schema_failures=1,
        expected_calls=4,
        minimum_schema_success_rate=1.0,
    )
    assert [row["event"] for row in ledger.event_rows] == ["STARTED", "FAILED"]
    assert reason is not None
    assert ledger.exhausted(call_key)


def test_judge_one_caps_each_physical_http_call_at_one(monkeypatch):
    calls = []

    class FakeClient:
        def chat(self, _messages, **kwargs):
            calls.append(kwargs)
            schema = kwargs["response_schema"]
            if schema is ResponseJudgeOutput:
                parsed = schema(
                    emotional_support=3,
                    personalization=3,
                    memory_appropriateness=3,
                    factual_grounding=3,
                    temporal_consistency=3,
                    non_intrusiveness=3,
                    rationale="ok",
                )
            else:
                parsed = schema(
                    selected_context_misuse=0,
                    unnecessary_exposure=0,
                    stale_or_conflicting_use=0,
                    unsupported_personal_claim=0,
                    memory_omission=0,
                    strategy_overuse=0,
                    strategy_omission=0,
                    rationale="ok",
                )
            return SimpleNamespace(request_hash=f"h{len(calls)}"), parsed

        def close(self):
            return None

    monkeypatch.setattr("metacom_pm.pm_v2_judging.make_client", lambda _: FakeClient())
    state = PMV2State.model_construct(
        current_user_text="I feel worried.",
        current_session_history=[],
        current_session_summary="",
    )
    judge_one(
        endpoint=Endpoint(
            base_url="https://example.invalid/v1",
            model="judge",
            family="family",
            api_key_env="IGNORED",
        ),
        state=state,
        authorized_user_context="none",
        selected_context="",
        candidate_response="That sounds difficult.",
    )
    assert len(calls) == 2
    assert all(call["retries"] == 1 for call in calls)


def test_existing_fixed_track_bundle_relocates_by_content_and_tamper_fails(tmp_path):
    root = Path(__file__).resolve().parents[1]
    bundle = root / "outputs" / "evoemo_fixed_tracks"
    result = require_content_addressed_attestation(
        bundle / "artifact_attestation.json",
        required_stage="evoemo_fixed_seeker_tracks",
        relocated_inputs={
            "evoemo": root / "data" / "external" / "evo_emo.json",
            "run_manifest": bundle / "run_manifest.json",
        },
        relocated_outputs={
            "tracks": bundle / "fixed_seeker_tracks.jsonl",
            "summary": bundle / "summary.json",
        },
    )
    assert result["ok"] is True
    copied_tracks = tmp_path / "fixed_seeker_tracks.jsonl"
    copied_tracks.write_bytes((bundle / "fixed_seeker_tracks.jsonl").read_bytes())
    copied_tracks.write_text(
        copied_tracks.read_text(encoding="utf-8") + "{}\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="hash mismatch"):
        require_content_addressed_attestation(
            bundle / "artifact_attestation.json",
            required_stage="evoemo_fixed_seeker_tracks",
            relocated_inputs={
                "evoemo": root / "data" / "external" / "evo_emo.json",
                "run_manifest": bundle / "run_manifest.json",
            },
            relocated_outputs={
                "tracks": copied_tracks,
                "summary": bundle / "summary.json",
            },
        )


def test_make_fixed_preserves_all_conformal_state_after_serialization(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "pmv2_fixed_script", root / "scripts" / "29_prepare_pm_v2_fixed_baselines.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    response_radii = {
        name: 0.1 + index / 100
        for index, name in enumerate(ResponseDimensions.model_fields)
    }
    risk_radii = {
        name: 0.2 + index / 100
        for index, name in enumerate(RiskDimensions.model_fields)
    }
    base = PMV2Model(
        feature_builder=PMV2FeatureBuilder(),
        response_heads={},
        risk_heads={},
        selection_config=SelectionConfig(),
        response_conformal_radii=response_radii,
        risk_conformal_radii=risk_radii,
        quality_conformal_radius=0.33,
        conformal_calibration_report={"status": "COMPLETE", "marker": [1, 2]},
        training_report={"status": "COMPLETE", "marker": {"a": 1}},
    )
    fixed = module.make_fixed(
        base,
        action_id="ME+R0",
        name="event_memory_r0",
        source_sha="a" * 64,
    )
    checkpoint = tmp_path / "fixed.joblib"
    fixed.save(checkpoint)
    loaded = PMV2Model.load(checkpoint)
    assert loaded.response_conformal_radii == response_radii
    assert loaded.risk_conformal_radii == risk_radii
    assert loaded.quality_conformal_radius == 0.33
    assert loaded.conformal_calibration_report == base.conformal_calibration_report
    assert loaded.training_report == base.training_report
    assert loaded.conformal_calibration_report is not base.conformal_calibration_report
    assert loaded.training_report is not base.training_report


def test_external_turn_loader_accepts_only_exact_frozen_pilot_exclusion(tmp_path):
    path = tmp_path / "turns.jsonl"
    full_units = [
        ("u1", 1, 101, "seeker_main", 3),
        ("u2", 1, 101, "seeker_main", 3),
    ]
    rows = [
        {
            "user_id": unit[0],
            "topic_index": unit[1],
            "seed": unit[2],
            "simulator_id": unit[3],
            "turn_index": unit[4],
            "condition": condition,
            "interaction_mode": "fixed",
            "context_sha256": f"context-{unit[0]}",
            "seeker_message": f"message-{unit[0]}",
        }
        for unit in full_units
        for condition in ("pm_v2", "baseline")
    ]
    write_jsonl(path, rows)
    matrix = load_fixed_turns(
        [path],
        conditions=["pm_v2", "baseline"],
        turn_indices=[3],
        expected_units=[full_units[1]],
        full_expected_units=full_units,
        excluded_units=[full_units[0]],
    )
    assert len(matrix) == 2
    assert all(key[:5] == full_units[1] for key in matrix)

    write_jsonl(
        path,
        [
            *rows,
            {
                "user_id": "u3",
                "topic_index": 1,
                "seed": 101,
                "simulator_id": "seeker_main",
                "turn_index": 3,
                "condition": "pm_v2",
                "interaction_mode": "fixed",
                "context_sha256": "context-u3",
                "seeker_message": "message-u3",
            },
        ],
    )
    with pytest.raises(RuntimeError, match="unexpected external unit"):
        load_fixed_turns(
            [path],
            conditions=["pm_v2", "baseline"],
            turn_indices=[3],
            expected_units=[full_units[1]],
            full_expected_units=full_units,
            excluded_units=[full_units[0]],
        )


def test_external_gate_rejects_deterministic_composite_support_collapse():
    rows = []
    for index in range(10):
        score = float(index % 5 + 1)
        rows.append(
            {
                **{name: score for name in ResponseDimensions.model_fields},
                **{
                    name: float((index + offset) % 4)
                    for offset, name in enumerate(RiskDimensions.model_fields)
                },
                "condition": "pm_v2",
                "label_reliable": True,
                "response_dimension_mad": {
                    name: 0.0 for name in ResponseDimensions.model_fields
                },
                "risk_dimension_mad": {
                    name: 0.0 for name in RiskDimensions.model_fields
                },
            }
        )
    with pytest.raises(RuntimeError, match="external judge gate failed"):
        validate_external_score_table(
            rows,
            duplicate_exact_match_rate=1.1,
            minimum_reliable_rate=0.0,
            reject_constant_response_dimensions=False,
            reject_constant_risk_dimensions=False,
            maximum_absolute_dimension_correlation=1.1,
            composite_support_exact_match_rate=0.98,
            maximum_absolute_composite_support_correlation=1.1,
            composite_spec=CompositeSpec(),
        )


def test_external_raw_family_gate_cannot_be_masked_by_another_family():
    rows = []
    for family in ("degenerate", "healthy"):
        for index in range(10):
            if family == "degenerate":
                response = {
                    name: float(index % 5 + 1)
                    for name in ResponseDimensions.model_fields
                }
            else:
                response = {
                    name: float((index * (offset + 1)) % 5 + 1)
                    for offset, name in enumerate(ResponseDimensions.model_fields)
                }
            response["rationale"] = "structured response assessment"
            rows.append(
                {
                    "judge_family": family,
                    "response_scores": response,
                    "risk_scores": {
                        **{
                            name: float((index * (offset + 1)) % 4)
                            for offset, name in enumerate(RiskDimensions.model_fields)
                        },
                        "rationale": "structured risk assessment",
                    },
                }
            )
    with pytest.raises(RuntimeError, match="raw judge-family quality gate failed"):
        validate_raw_judge_family_health(
            [
                {
                    "judge_family": row["judge_family"],
                    "response": row["response_scores"],
                    "risk": row["risk_scores"],
                }
                for row in rows
            ],
            expected_families=("degenerate", "healthy"),
            composite_spec=CompositeSpec(),
            duplicate_exact_match_rate=0.98,
            maximum_absolute_dimension_correlation=1.1,
            composite_support_exact_match_rate=0.98,
            maximum_absolute_composite_support_correlation=1.1,
            reject_constant_response_dimensions=False,
            reject_constant_risk_dimensions=False,
        )


def test_attempt_ledger_never_reissues_unknown_started_attempt(tmp_path):
    endpoint = Endpoint(
        base_url="https://example.invalid/v1",
        model="judge",
        family="family",
        api_key_env="IGNORED",
    )
    call_key = physical_call_key(
        stage="unit-test",
        record_ids={"unit_id": "u1"},
        prompt_sha256="p" * 64,
        endpoint=endpoint,
        request_parameters={"seed": 1, "retries": 1},
    )
    path = tmp_path / "attempts.jsonl"
    ledger = PersistentAttemptLedger(
        path,
        stage="unit-test",
        expected_calls={call_key: 1},
        maximum_total_attempts=1,
    )
    ledger.reserve(
        call_key,
        record_ids={"unit_id": "u1"},
        prompt_sha256="p" * 64,
    )
    persisted = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in persisted] == ["STARTED"]

    resumed = PersistentAttemptLedger(
        path,
        stage="unit-test",
        expected_calls={call_key: 1},
        maximum_total_attempts=1,
    )
    assert resumed.started_attempts == 1
    assert resumed.exhausted(call_key)
    assert not resumed.succeeded(call_key)
    with pytest.raises(RuntimeError, match="exhausted"):
        resumed.reserve(
            call_key,
            record_ids={"unit_id": "u1"},
            prompt_sha256="p" * 64,
        )


def test_dry_run_overwrite_cannot_reset_a_spent_attempt_ledger(tmp_path):
    path = tmp_path / "judge_call_ledger.jsonl"
    path.write_text('{"event":"STARTED"}\n', encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="physical-attempt ledger.*non-empty"):
        forbid_overwrite_of_spent_attempts(
            path,
            overwrite=True,
            stage="external dry-run",
        )
    assert path.read_bytes() == before


def test_external_and_forced_paid_paths_guard_overwrite_before_work():
    root = Path(__file__).resolve().parents[1]
    sources = [
        (root / "src" / "metacom_pm" / name).read_text(encoding="utf-8")
        for name in ("pm_v2_external_eval.py", "pm_v2_forced_swap.py")
    ]
    for source in sources:
        assert "if run and overwrite:" in source
        assert "forbid_overwrite_of_spent_attempts(" in source


def test_full_structured_payload_is_safety_factored_for_cost_planning():
    endpoint = Endpoint(
        base_url="https://example.invalid/v1",
        model="judge",
        family="family",
        api_key_env="IGNORED",
    )
    payload = chat_request_payload(
        endpoint,
        [{"role": "user", "content": "score this"}],
        temperature=0.0,
        max_tokens=600,
        seed=7,
        response_schema=ResponseJudgeOutput,
    )
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert request_payload_has_schema(payload) is True
    serialized = canonical_json(payload)
    assert conservative_token_bound(serialized, safety_factor=1.5) >= int(
        estimate_tokens(serialized) * 1.5
    )


def test_anthropic_structured_payload_freezes_exact_schema_tool_contract():
    endpoint = Endpoint(
        base_url="https://api.anthropic.com",
        model="claude-test",
        family="anthropic",
        api_key_env="IGNORED",
    )
    payload = chat_request_payload(
        endpoint,
        [
            {"role": "system", "content": "judge strictly"},
            {"role": "user", "content": "score this"},
        ],
        temperature=0.0,
        max_tokens=600,
        seed=7,
        response_schema=ResponseJudgeOutput,
    )
    assert "response_format" not in payload
    assert payload["tools"] == [
        {
            "name": "submit_structured_response",
            "description": "Return the evaluator result using the required strict schema.",
            "input_schema": ResponseJudgeOutput.model_json_schema(),
        }
    ]
    assert payload["tool_choice"] == {
        "type": "tool",
        "name": "submit_structured_response",
    }
    assert payload["system"] == "judge strictly"
    assert request_payload_has_schema(payload) is True
    assert conservative_token_bound(
        canonical_json(payload), safety_factor=1.5
    ) > estimate_tokens(canonical_json(payload))


def test_freeze_source_requires_semantic_sanity_and_binds_sweep_contract():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "26_freeze_pm_v2_study.py").read_text(
        encoding="utf-8"
    )
    assert "require_semantic_sanity_pass(" in source
    assert '"semantic_sanity": semantic_sanity' in source
    assert "semantic_sanity_input_paths" in source


def test_external_sources_forbid_v1_baselines_and_require_v22_track_approval():
    root = Path(__file__).resolve().parents[1]
    external_source = (root / "scripts" / "25_eval_pm_v2_external.py").read_text(
        encoding="utf-8"
    )
    freeze_source = (root / "scripts" / "26_freeze_pm_v2_study.py").read_text(
        encoding="utf-8"
    )
    generation_source = (root / "scripts" / "24_run_pm_v2_evoemo.py").read_text(
        encoding="utf-8"
    )
    combined = external_source + freeze_source + generation_source
    assert "evoemo_selective" not in combined
    assert 'required_stage="evoemo_generation"' not in combined
    assert "evoemo_pmv22_reference_baselines" in combined
    assert "evoemo_fixed_tracks_v22" in combined
    assert "require_strategy_bank_human_approval(" in freeze_source
    assert "fixed_seeker_generation_treatment_sha256" in combined


def test_external_judge_pricing_and_safety_are_freeze_only_not_cli_overrides():
    root = Path(__file__).resolve().parents[1]
    forced_source = (root / "scripts" / "30_eval_pm_v2_forced_swap.py").read_text(
        encoding="utf-8"
    )
    external_source = (root / "scripts" / "25_eval_pm_v2_external.py").read_text(
        encoding="utf-8"
    )
    freeze_source = (root / "scripts" / "26_freeze_pm_v2_study.py").read_text(
        encoding="utf-8"
    )
    assert "--input-usd-per-mtok" not in forced_source + external_source
    assert "--output-usd-per-mtok" not in forced_source + external_source
    assert '"judge_pricing_usd_per_mtok"' in freeze_source
    assert '"generator_pricing_usd_per_mtok"' in freeze_source
    assert "generator_pricing != {\"input\": 0.15, \"output\": 0.60}" in freeze_source
    assert "require_development_pilot_gate(" in freeze_source
    assert '"development_pilot_gate": development_pilot_gate' in freeze_source
    assert '"input_token_safety_factor"' in freeze_source
    assert '"fail_on_reported_input_overrun"' in freeze_source


def test_external_generation_rejects_zero_price_cli_override():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "24_run_pm_v2_evoemo.py"
    spec = importlib.util.spec_from_file_location("pm_v2_external_generation", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract = {
        "generator_pricing_usd_per_mtok": {"input": 0.15, "output": 0.60}
    }
    with pytest.raises(RuntimeError, match="must equal the frozen value"):
        module.resolve_frozen_generator_pricing(
            contract,
            input_override=0.0,
            output_override=None,
        )
    assert module.resolve_frozen_generator_pricing(
        contract,
        input_override=None,
        output_override=None,
    ) == {"input": 0.15, "output": 0.60}
