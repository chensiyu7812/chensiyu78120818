from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import MemorySource, StrategyMode, canonical_action_id
from .io import canonical_json, sha256_text
from .pm_v2_contracts import ActionPrediction, PMV2State, PolicyDecision
from .pm_v2_features import PMV2FeatureBuilder
from .pm_v2_model import (
    SelectionConfig,
    TRANSPARENT_RULE_SELECTION_REASON,
    evaluate_policy,
    estimated_action_cost_profile,
)


RULE_ROUTER_PROTOCOL = "pm-v1.5-transparent-step0-rule-router-v2"
RULE_SCORE_DIAGNOSTIC_PROTOCOL = "pm-v1.5-transparent-rule-score-diagnostic-v1"
RULE_GRID_DIAGNOSTIC_PROTOCOL = "pm-v1.5-transparent-rule-grid-diagnostic-v1"
DEVELOPMENT_EXTERNAL_SCORE_COMPARISON_PROTOCOL = (
    "pm-v1.5-development-external-step0-score-comparison-v1"
)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray([float(value) for value in values], dtype=float)
    if not len(array):
        return {"count": 0, "minimum": None, "q10": None, "median": None, "q90": None, "maximum": None}
    return {
        "count": len(array),
        "minimum": float(np.min(array)),
        "q10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
    }


def _top_margin(values: Mapping[str, float]) -> float:
    ordered = sorted((float(value) for value in values.values()), reverse=True)
    return ordered[0] - ordered[1] if len(ordered) >= 2 else 0.0


def transparent_rule_score_diagnostics(
    states: Sequence[PMV2State],
) -> dict[str, Any]:
    """Report Step-0 score scales without consulting outcome labels."""

    source_values = {source.value: [] for source in MemorySource}
    family_values: list[float] = []
    readiness_values: list[float] = []
    family_margins: list[float] = []
    readiness_margins: list[float] = []
    for state in states:
        if state.step0_observation is None:
            raise RuntimeError("rule score diagnostics require formal Step-0")
        for source in MemorySource:
            row = state.step0_observation.memory_sources[source]
            if row.available and row.representation_valid:
                source_values[source.value].append(row.query_to_source_similarity)
        strategy = state.step0_observation.strategy
        if strategy.available and strategy.representation_valid:
            family_values.extend(strategy.family_similarities.values())
            readiness_values.extend(strategy.advice_readiness_similarities.values())
            family_margins.append(_top_margin(strategy.family_similarities))
            readiness_margins.append(
                _top_margin(strategy.advice_readiness_similarities)
            )
    return {
        "protocol": RULE_SCORE_DIAGNOSTIC_PROTOCOL,
        "outcome_labels_used": False,
        "state_count": len(states),
        "source_similarity": {
            source: _distribution(values) for source, values in source_values.items()
        },
        "strategy_family_similarity": _distribution(family_values),
        "advice_readiness_similarity": _distribution(readiness_values),
        "strategy_family_top1_top2_margin": _distribution(family_margins),
        "advice_readiness_top1_top2_margin": _distribution(readiness_margins),
    }


def transparent_rule_grid_diagnostics(
    states: Sequence[PMV2State],
    candidates: Sequence[TransparentRuleConfig],
    selection_config: SelectionConfig,
    *,
    minimum_unique_policy_mappings: int = 2,
    minimum_maximum_pairwise_disagreement_rate: float = 0.01,
) -> dict[str, Any]:
    """Outcome-free audit of whether the finite rule grid changes policies."""

    if not states or not candidates:
        raise ValueError("rule-grid diagnostics require states and candidates")
    ordered_states = sorted(states, key=lambda state: state.state_id)
    candidate_rows: list[dict[str, Any]] = []
    mappings: list[list[str]] = []
    for config in candidates:
        router = TransparentRuleRouter.create(config, selection_config)
        actions = [router.choose_action(state) for state in ordered_states]
        mappings.append(actions)
        candidate_rows.append(
            {
                "config_sha256": config.digest(),
                "action_by_state_sha256": sha256_text(
                    canonical_json(
                        list(
                            zip(
                                [state.state_id for state in ordered_states],
                                actions,
                                strict=True,
                            )
                        )
                    )
                ),
                "action_distribution": dict(sorted(Counter(actions).items())),
            }
        )
    disagreement: list[float] = []
    for left in range(len(mappings)):
        for right in range(left + 1, len(mappings)):
            disagreement.append(
                float(
                    np.mean(
                        np.asarray(mappings[left], dtype=object)
                        != np.asarray(mappings[right], dtype=object)
                    )
                )
            )
    unique_mappings = len({tuple(actions) for actions in mappings})
    maximum_disagreement = max(disagreement, default=0.0)
    checks = {
        "unique_policy_mappings": unique_mappings
        >= int(minimum_unique_policy_mappings),
        "maximum_pairwise_action_disagreement_rate": maximum_disagreement
        >= float(minimum_maximum_pairwise_disagreement_rate),
    }
    return {
        "protocol": RULE_GRID_DIAGNOSTIC_PROTOCOL,
        "status": "PASS" if all(checks.values()) else "DEGENERATE_GRID",
        "outcome_labels_used": False,
        "state_split": "train_and_calibration_only",
        "state_count": len(ordered_states),
        "state_ids_sha256": sha256_text(
            canonical_json([state.state_id for state in ordered_states])
        ),
        "candidate_count": len(candidate_rows),
        "unique_policy_mapping_count": unique_mappings,
        "pairwise_action_disagreement_rate": _distribution(disagreement),
        "maximum_pairwise_action_disagreement_rate": maximum_disagreement,
        "checks": checks,
        "limits": {
            "minimum_unique_policy_mappings": int(minimum_unique_policy_mappings),
            "minimum_maximum_pairwise_disagreement_rate": float(
                minimum_maximum_pairwise_disagreement_rate
            ),
        },
        "candidates": candidate_rows,
    }


def compare_development_external_score_diagnostics(
    development_by_split: Mapping[str, Any],
    external: Mapping[str, Any],
) -> dict[str, Any]:
    """Content-ready, outcome-free scale comparison with no retuning authority."""

    reference = development_by_split.get("calibration")
    if not isinstance(reference, Mapping):
        raise RuntimeError("score comparison requires calibration diagnostics")
    if (
        reference.get("protocol") != RULE_SCORE_DIAGNOSTIC_PROTOCOL
        or external.get("protocol") != RULE_SCORE_DIAGNOSTIC_PROTOCOL
    ):
        raise RuntimeError("score comparison received stale diagnostics")

    paths = [
        ("source_similarity.MP", ("source_similarity", "MP")),
        ("source_similarity.MS", ("source_similarity", "MS")),
        ("source_similarity.ME", ("source_similarity", "ME")),
        ("strategy_family_similarity", ("strategy_family_similarity",)),
        ("advice_readiness_similarity", ("advice_readiness_similarity",)),
        (
            "strategy_family_top1_top2_margin",
            ("strategy_family_top1_top2_margin",),
        ),
        (
            "advice_readiness_top1_top2_margin",
            ("advice_readiness_top1_top2_margin",),
        ),
    ]

    def at(payload: Mapping[str, Any], path: Sequence[str]) -> Mapping[str, Any]:
        value: Any = payload
        for key in path:
            value = value.get(key) if isinstance(value, Mapping) else None
        if not isinstance(value, Mapping) or "count" not in value:
            raise RuntimeError(f"score comparison lacks distribution: {'.'.join(path)}")
        return value

    metrics: dict[str, Any] = {}
    for name, path in paths:
        left = at(reference, path)
        right = at(external, path)
        if int(left.get("count", 0)) <= 0 or int(right.get("count", 0)) <= 0:
            metrics[name] = {
                "status": "UNAVAILABLE_NO_OBSERVATIONS",
                "development_calibration_count": int(left.get("count", 0)),
                "external_count": int(right.get("count", 0)),
                "quantile_shifts": None,
                "quantile_l1_shift": None,
                "development_calibration": None,
                "external": None,
            }
            continue
        quantile_shifts = {
            key: float(right[key]) - float(left[key])
            for key in ("q10", "median", "q90")
        }
        metrics[name] = {
            "status": "AVAILABLE",
            "development_calibration_count": int(left["count"]),
            "external_count": int(right["count"]),
            "quantile_shifts": quantile_shifts,
            "quantile_l1_shift": float(
                np.mean([abs(value) for value in quantile_shifts.values()])
            ),
            "development_calibration": {
                key: left[key] for key in ("q10", "median", "q90")
            },
            "external": {
                key: right[key] for key in ("q10", "median", "q90")
            },
        }
    return {
        "protocol": DEVELOPMENT_EXTERNAL_SCORE_COMPARISON_PROTOCOL,
        "status": "REPORT_ONLY",
        "outcome_labels_used": False,
        "external_threshold_selection_or_retuning_authorized": False,
        "development_reference_split": "calibration",
        "available_distribution_count": sum(
            row["status"] == "AVAILABLE" for row in metrics.values()
        ),
        "unavailable_distributions": sorted(
            name
            for name, row in metrics.items()
            if row["status"] != "AVAILABLE"
        ),
        "metrics": metrics,
    }


@dataclass
class FixedActionBaselineRouter:
    """Step-0-free baseline used only by the no-Step-0 residual ablation."""

    action_id: str = "M0+R0"

    def choose_action(self, state: PMV2State) -> str:
        if self.action_id not in state.allowed_actions:
            raise RuntimeError(
                f"fixed residual baseline {self.action_id} is illegal for {state.state_id}"
            )
        return self.action_id


class TransparentRuleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    protocol: str = RULE_ROUTER_PROTOCOL
    source_similarity_weight: float = Field(ge=0.0)
    source_age_penalty: float = Field(ge=0.0)
    source_cost_penalty: float = Field(ge=0.0)
    source_minimum_score: float
    maximum_memory_sources: int = Field(ge=0, le=3)
    strategy_family_similarity_weight: float = Field(ge=0.0)
    strategy_readiness_alignment_weight: float = Field(ge=0.0)
    question_bonus: float = Field(ge=0.0)
    strategy_cost_penalty: float = Field(ge=0.0)
    strategy_minimum_score: float

    @model_validator(mode="after")
    def valid_protocol(self):
        if self.protocol != RULE_ROUTER_PROTOCOL:
            raise ValueError("unsupported transparent rule-router protocol")
        return self

    def digest(self) -> str:
        return sha256_text(canonical_json(self.model_dump(mode="json")))


@dataclass
class TransparentRuleRouter:
    """Low-capacity, auditable router using exactly the formal Step-0 scalars."""

    config: TransparentRuleConfig
    selection_config: SelectionConfig
    feature_builder: PMV2FeatureBuilder
    format_version: str = RULE_ROUTER_PROTOCOL

    @classmethod
    def create(
        cls,
        config: TransparentRuleConfig,
        selection_config: SelectionConfig,
    ) -> "TransparentRuleRouter":
        return cls(
            config=config,
            selection_config=selection_config,
            feature_builder=PMV2FeatureBuilder(
                word_features=8,
                char_features=8,
                use_precomputed_embeddings=False,
            ),
        )

    def _memory_scores(self, state: PMV2State) -> dict[MemorySource, float]:
        if state.step0_observation is None:
            raise RuntimeError("transparent rule router requires formal Step-0")
        observations = state.step0_observation.memory_sources
        token_scale = max(
            max(row.expected_retrieval_tokens for row in observations.values()),
            1,
        )
        scores: dict[MemorySource, float] = {}
        for source in MemorySource:
            row = observations[source]
            if not row.available or not row.representation_valid:
                scores[source] = float("-inf")
                continue
            age_ratio = min(
                float(row.median_age_sessions or 0.0) / max(state.session_index, 1),
                1.0,
            )
            cost_ratio = row.expected_retrieval_tokens / token_scale
            scores[source] = (
                self.config.source_similarity_weight
                * row.query_to_source_similarity
                - self.config.source_age_penalty * age_ratio
                - self.config.source_cost_penalty * cost_ratio
            )
        return scores

    def choose_action(self, state: PMV2State) -> str:
        if state.step0_observation is None:
            raise RuntimeError("transparent rule router requires formal Step-0")
        source_scores = self._memory_scores(state)
        ranked_sources = sorted(
            (
                (score, source)
                for source, score in source_scores.items()
                if score >= self.config.source_minimum_score
            ),
            key=lambda item: (item[0], item[1].value),
            reverse=True,
        )
        sources = {
            source
            for _, source in ranked_sources[: self.config.maximum_memory_sources]
        }

        strategy = state.step0_observation.strategy
        family = strategy.family_similarities
        readiness = strategy.advice_readiness_similarities
        stage_fits = {
            "listen_only": (
                max(
                    family["reflection"],
                    family["restatement"],
                    family["affirmation_reassurance"],
                ),
                readiness["listen_only"],
            ),
            "explore_first": (
                max(family["question"], family["reflection"], family["restatement"]),
                readiness["explore_first"],
            ),
            "light_suggestion": (
                max(family["suggestion"], family["information"], family["question"]),
                readiness["light_suggestion"],
            ),
            "structured_plan": (
                max(family["suggestion"], family["information"]),
                readiness["structured_plan"],
            ),
            "ambiguous": (
                max(family.values(), default=0.0),
                readiness["ambiguous"],
            ),
        }
        maximum_stage_fit = max(
            self.config.strategy_family_similarity_weight * family_score
            + self.config.strategy_readiness_alignment_weight * readiness_score
            for family_score, readiness_score in stage_fits.values()
        )
        strategy_cost_ratio = min(
            strategy.expected_retrieval_tokens / 768.0,
            1.0,
        )
        strategy_score = (
            maximum_stage_fit
            + self.config.question_bonus * float(strategy.question_present)
            - self.config.strategy_cost_penalty * strategy_cost_ratio
        )
        strategy_mode = (
            StrategyMode.RS
            if strategy.available
            and strategy_score >= self.config.strategy_minimum_score
            else StrategyMode.R0
        )
        action_id = canonical_action_id(sources, strategy_mode)
        if action_id not in state.allowed_actions:
            raise RuntimeError(
                f"transparent rule produced illegal action {action_id} for {state.state_id}"
            )
        return action_id

    def choose(self, state: PMV2State) -> PolicyDecision:
        chosen = self.choose_action(state)
        profile = estimated_action_cost_profile(self.feature_builder, state)
        predictions = {
            action_id: ActionPrediction(
                action_id=action_id,
                response={},
                risk={},
                quality_mean=0.0,
                quality_lcb=0.0,
                risk_ucb=0.0,
                estimated_cost=float(row["estimated_resource_cost"]),
                normalized_cost=float(row["normalized_estimated_resource_cost"]),
                utility=-self.selection_config.cost_weight
                * float(row["normalized_estimated_resource_cost"]),
                feasible=True,
                resource_gate_passed=True,
                strategy_gate_passed=True,
            )
            for action_id, row in profile.items()
        }
        return PolicyDecision(
            state_id=state.state_id,
            chosen_action=chosen,
            predictions=predictions,
            semantic_ood_score=0.0,
            metadata_ood_score=0.0,
            ood_fallback_used=False,
            decision_reason=TRANSPARENT_RULE_SELECTION_REASON,
            config_hash=self.config.digest(),
        )


def transparent_rule_candidates(
    grid: Mapping[str, Sequence[float | int]],
) -> list[TransparentRuleConfig]:
    required = (
        "source_similarity_weights",
        "source_age_penalties",
        "source_cost_penalties",
        "source_minimum_scores",
        "maximum_memory_sources",
        "strategy_family_similarity_weights",
        "strategy_readiness_alignment_weights",
        "question_bonuses",
        "strategy_cost_penalties",
        "strategy_minimum_scores",
    )
    if set(grid) != set(required):
        raise ValueError("transparent rule grid keys do not match the frozen contract")
    values = [list(grid[key]) for key in required]
    if any(not rows for rows in values):
        raise ValueError("transparent rule grid dimensions cannot be empty")
    return [
        TransparentRuleConfig(
            source_similarity_weight=float(row[0]),
            source_age_penalty=float(row[1]),
            source_cost_penalty=float(row[2]),
            source_minimum_score=float(row[3]),
            maximum_memory_sources=int(row[4]),
            strategy_family_similarity_weight=float(row[5]),
            strategy_readiness_alignment_weight=float(row[6]),
            question_bonus=float(row[7]),
            strategy_cost_penalty=float(row[8]),
            strategy_minimum_score=float(row[9]),
        )
        for row in product(*values)
    ]


def tune_transparent_rule_router(
    *,
    states: Sequence[PMV2State],
    labels,
    selection_config: SelectionConfig,
    candidates: Sequence[TransparentRuleConfig],
    minimum_quality: float,
    maximum_risk: float,
    selection_data_role: Literal["train", "train_fold"],
) -> tuple[TransparentRuleRouter, dict[str, Any]]:
    """Select rule numbers on train data with one frozen utility ruler."""

    rows: list[dict[str, Any]] = []
    for config in candidates:
        router = TransparentRuleRouter.create(config, selection_config)
        metrics = evaluate_policy(router, states, labels)
        eligible = (
            metrics["mean_quality"] >= minimum_quality
            and metrics["mean_risk"] <= maximum_risk
        )
        rows.append(
            {
                "config": config.model_dump(mode="json"),
                "config_sha256": config.digest(),
                "eligible": eligible,
                "mean_quality": metrics["mean_quality"],
                "mean_risk": metrics["mean_risk"],
                "mean_realized_utility": metrics["mean_realized_utility"],
                "mean_observed_input_tokens": metrics["mean_observed_input_tokens"],
                "action_distribution": metrics["action_distribution"],
            }
        )
    eligible_rows = [row for row in rows if row["eligible"]]
    if not eligible_rows:
        raise RuntimeError("no transparent rule candidate satisfies train-data guards")
    selected = max(
        eligible_rows,
        key=lambda row: (
            row["mean_realized_utility"],
            row["mean_quality"],
            -row["mean_risk"],
            -row["mean_observed_input_tokens"],
            row["config_sha256"],
        ),
    )
    selected_config = TransparentRuleConfig.model_validate(selected["config"])
    distribution_signatures = {
        canonical_json(row["action_distribution"]) for row in rows
    }
    grid_diagnostics = transparent_rule_grid_diagnostics(
        states, candidates, selection_config
    )
    selected_mapping = next(
        row
        for row in grid_diagnostics["candidates"]
        if row["config_sha256"] == selected_config.digest()
    )
    report = {
        "protocol": RULE_ROUTER_PROTOCOL,
        "selection_split": selection_data_role,
        "selection_data_role": (
            "train_only" if selection_data_role == "train" else "train_fold_only"
        ),
        "candidate_count": len(rows),
        "minimum_quality": float(minimum_quality),
        "maximum_risk": float(maximum_risk),
        "selected_config": selected_config.model_dump(mode="json"),
        "selected_config_sha256": selected_config.digest(),
        "selected_action_distribution": selected["action_distribution"],
        "selected_action_by_state_sha256": selected_mapping[
            "action_by_state_sha256"
        ],
        "distinct_candidate_action_distributions": len(distribution_signatures),
        "unique_policy_mapping_count": grid_diagnostics[
            "unique_policy_mapping_count"
        ],
        "pairwise_action_disagreement_rate": grid_diagnostics[
            "pairwise_action_disagreement_rate"
        ],
        "grid_diagnostics": grid_diagnostics,
        "score_diagnostics": transparent_rule_score_diagnostics(states),
        "candidates": rows,
    }
    return TransparentRuleRouter.create(selected_config, selection_config), report
