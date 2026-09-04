from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.ensemble import HistGradientBoostingRegressor

from .contracts import MemorySource, StrategyMode, canonical_action_id, parse_action_id
from .io import canonical_json, sha256_text
from .pm_v2_contracts import (
    ActionLabel,
    ActionPrediction,
    CompositeSpec,
    PMV2State,
    PolicyDecision,
    PredictionInterval,
    ResponseDimensions,
)
from .pm_v2_features import PMV2FeatureBuilder


RESPONSE_FIELDS = tuple(ResponseDimensions.model_fields)
RISK_FIELDS = (
    "selected_context_misuse",
    "unnecessary_exposure",
    "stale_or_conflicting_use",
    "unsupported_personal_claim",
    "memory_omission",
    "strategy_overuse",
    "strategy_omission",
)


def composite_weights_digest(spec: CompositeSpec) -> str:
    return sha256_text(canonical_json(spec.weights))


SEVERE_OOD_FALLBACK_REASON = (
    "fallback_type=severe_ood; preregistered conservative fallback"
)
NO_FEASIBLE_FALLBACK_REASON = (
    "fallback_type=no_feasible; no action passed conservative gates"
)
LEARNED_SELECTION_REASON = (
    "fallback_type=none; max conservative utility after explicit risk, "
    "resource-benefit, strategy-benefit and cost terms"
)


class SelectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: str = "pmv2-selection-v2"
    composite_spec: CompositeSpec = Field(default_factory=CompositeSpec)
    uncertainty_z: float = Field(default=1.0, ge=0.0, le=3.0)
    max_risk_ucb: float = Field(default=0.45, ge=0.0, le=1.0)
    risk_ucb_thresholds: dict[str, float] = Field(default_factory=dict)
    risk_weight: float = Field(default=0.25, ge=0.0)
    cost_weight: float = Field(default=0.10, ge=0.0)
    resource_min_gain: float = Field(default=-0.02, ge=-1.0, le=1.0)
    strategy_min_gain: float = Field(default=-0.01, ge=-1.0, le=1.0)
    m0_omission_trigger: float = Field(default=0.35, ge=0.0, le=1.0)
    r0_strategy_omission_trigger: float = Field(default=0.35, ge=0.0, le=1.0)
    fallback_action: str = "M0+R0"
    fail_closed_on_ood: bool = True

    @model_validator(mode="after")
    def validate_fallback(self):
        parse_action_id(self.fallback_action)
        if self.risk_weight == 0 and self.cost_weight == 0:
            raise ValueError("at least one of risk_weight or cost_weight must be positive")
        unknown_risks = set(self.risk_ucb_thresholds) - set(RISK_FIELDS)
        if unknown_risks:
            raise ValueError(f"unknown risk UCB thresholds: {sorted(unknown_risks)}")
        invalid_risks = {
            key: value
            for key, value in self.risk_ucb_thresholds.items()
            if not 0.0 <= float(value) <= 1.0
        }
        if invalid_risks:
            raise ValueError(f"risk UCB thresholds must be in [0, 1]: {invalid_risks}")
        return self

    def risk_threshold(self, field_name: str) -> float:
        if field_name not in RISK_FIELDS:
            raise KeyError(f"unknown PM-v2 risk field: {field_name}")
        # max_risk_ucb remains a tunable global cap; optional per-dimension values
        # can only make a safety dimension stricter.
        return min(
            self.max_risk_ucb,
            float(self.risk_ucb_thresholds.get(field_name, self.max_risk_ucb)),
        )

    def digest(self) -> str:
        return sha256_text(canonical_json(self.model_dump(mode="json")))


@dataclass
class BootstrapRegressor:
    """Group-bootstrap ensemble for mean and epistemic uncertainty."""

    n_models: int = 7
    seed: int = 17
    max_iter: int = 180
    max_leaf_nodes: int = 15
    learning_rate: float = 0.05
    l2_regularization: float = 1.0
    models: list[HistGradientBoostingRegressor] = field(default_factory=list)
    fit_diagnostics: dict[str, Any] = field(default_factory=dict)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        groups: Sequence[str],
        sample_weight: np.ndarray | None = None,
    ) -> "BootstrapRegressor":
        if x.shape[0] != len(y) or len(y) != len(groups):
            raise ValueError("x, y and groups must have equal row counts")
        unique_groups = np.asarray(sorted(set(groups)), dtype=object)
        if unique_groups.size < 3:
            raise ValueError("group bootstrap requires at least three unique groups")
        group_array = np.asarray(groups, dtype=object)
        rng = np.random.default_rng(self.seed)
        self.models = []
        member_unique_group_counts: list[int] = []
        for index in range(self.n_models):
            sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            multiplicity: dict[str, int] = {}
            for group in sampled_groups:
                multiplicity[str(group)] = multiplicity.get(str(group), 0) + 1
            member_unique_group_counts.append(len(multiplicity))
            row_weights = np.asarray(
                [float(multiplicity.get(str(group), 0)) for group in group_array],
                dtype=float,
            )
            keep = row_weights > 0
            if sample_weight is not None:
                row_weights = row_weights * sample_weight
            model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                max_leaf_nodes=self.max_leaf_nodes,
                l2_regularization=self.l2_regularization,
                random_state=self.seed + index,
            )
            model.fit(x[keep], y[keep], sample_weight=row_weights[keep])
            self.models.append(model)
        self.fit_diagnostics = {
            "n_groups": int(unique_groups.size),
            "member_unique_group_counts": member_unique_group_counts,
            "member_group_coverage_rates": [
                float(value / unique_groups.size) for value in member_unique_group_counts
            ],
            "minimum_member_group_coverage_rate": float(
                min(member_unique_group_counts) / unique_groups.size
            ),
            "mean_member_group_coverage_rate": float(
                np.mean(member_unique_group_counts) / unique_groups.size
            ),
        }
        return self

    def predict_members(self, x: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("bootstrap regressor is not fitted")
        return np.vstack([model.predict(x) for model in self.models])

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = self.predict_members(x)
        return matrix.mean(axis=0), matrix.std(axis=0, ddof=0)


def applicable_risk_fields(action_id: str) -> tuple[str, ...]:
    """Risk dimensions that can make the given resource action infeasible."""

    sources, strategy = parse_action_id(action_id)
    applicable = ["unsupported_personal_claim", "memory_omission"]
    if sources:
        applicable[0:0] = [
            "selected_context_misuse",
            "unnecessary_exposure",
            "stale_or_conflicting_use",
        ]
    # RS is a request to the retrieval/execution layer, not a guarantee that a
    # useful card survives post-retrieval filtering or is reflected in the
    # response.  Therefore an RS system can incur both overuse and omission;
    # R0 can incur omission only.
    if strategy is StrategyMode.RS:
        applicable.append("strategy_overuse")
    applicable.append("strategy_omission")
    return tuple(applicable)


def estimated_action_cost_profile(feature_builder, state: PMV2State) -> dict[str, dict[str, float]]:
    """Return the selector's raw and within-state normalized resource cost."""

    estimated = {
        action: float(feature_builder.estimate_action_cost(state, action))
        for action in state.allowed_actions
    }
    if any(not np.isfinite(value) or value < 0.0 for value in estimated.values()):
        raise RuntimeError(f"invalid estimated action cost for state {state.state_id}")
    scale = max(max(estimated.values()), 1.0)
    return {
        action: {
            "estimated_resource_cost": value,
            "normalized_estimated_resource_cost": value / scale,
        }
        for action, value in estimated.items()
    }


@dataclass
class _PredictionBundle:
    response: dict[str, dict[str, PredictionInterval]]
    risk: dict[str, dict[str, PredictionInterval]]
    quality_members: dict[str, np.ndarray]


@dataclass
class PMV2Model:
    feature_builder: PMV2FeatureBuilder
    response_heads: dict[str, BootstrapRegressor]
    risk_heads: dict[str, BootstrapRegressor]
    selection_config: SelectionConfig
    response_conformal_radii: dict[str, float] = field(default_factory=dict)
    risk_conformal_radii: dict[str, float] = field(default_factory=dict)
    quality_conformal_radius: float = 0.0
    conformal_calibration_report: dict[str, Any] = field(default_factory=dict)
    format_version: str = "pm-v2.1"
    training_report: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def train(
        cls,
        states: Sequence[PMV2State],
        labels: Sequence[ActionLabel],
        *,
        selection_config: SelectionConfig | None = None,
        n_models: int = 7,
        seed: int = 17,
        dimension_mad_scale: float = 0.75,
        bootstrap_group_key: str = "user_id",
        use_precomputed_embeddings: bool = True,
        word_features: int = 256,
        char_features: int = 256,
    ) -> "PMV2Model":
        effective_selection = selection_config or SelectionConfig()
        state_map = {state.state_id: state for state in states}
        if len(state_map) != len(states):
            raise ValueError("duplicate PM-v2 state_id")
        if not state_map:
            raise ValueError("cannot train PM-v2 without states")
        label_keys = [(label.state_id, label.action_id) for label in labels]
        if len(label_keys) != len(set(label_keys)):
            raise ValueError("duplicate PM-v2 action labels")
        missing_states = sorted({label.state_id for label in labels} - set(state_map))
        if missing_states:
            raise ValueError(f"labels reference unknown states: {missing_states[:5]}")
        expected_weights_hash = composite_weights_digest(
            effective_selection.composite_spec
        )
        bad_weight_hashes = sorted(
            {
                label.composite_weights_sha256
                for label in labels
                if label.composite_weights_sha256 != expected_weights_hash
            }
        )
        if bad_weight_hashes:
            raise ValueError(
                "PM-v2 labels do not match selection composite weights hash; "
                f"expected={expected_weights_hash}, observed={bad_weight_hashes[:5]}"
            )
        if not np.isfinite(dimension_mad_scale) or dimension_mad_scale <= 0.0:
            raise ValueError("dimension_mad_scale must be finite and positive")
        # The aggregated median matrix is the estimand. ``label_reliable`` is an
        # all-dimension diagnostic and must not delete an otherwise complete action
        # row: doing so makes complete-state retention exponentially unlikely as the
        # number of heads/actions grows. Each head instead receives a continuous
        # weight from its own MAD below.
        usable = list(labels)
        if not usable:
            raise ValueError("no PM-v2 action labels")
        labeled_actions: dict[str, set[str]] = {}
        for label in usable:
            labeled_actions.setdefault(label.state_id, set()).add(label.action_id)
        incomplete: list[dict[str, Any]] = []
        for state_id, state in sorted(state_map.items()):
            expected = set(state.allowed_actions)
            observed = labeled_actions.get(state_id, set())
            if observed != expected:
                incomplete.append(
                    {
                        "state_id": state_id,
                        "missing": sorted(expected - observed),
                        "extra": sorted(observed - expected),
                    }
                )
        if incomplete:
            raise ValueError(
                "every input training state must retain a complete set of "
                "aggregated labels for all legal actions; "
                f"incomplete={incomplete[:5]}"
            )
        unique_states = [state_map[state_id] for state_id in sorted(state_map)]
        builder = PMV2FeatureBuilder(
            word_features=word_features,
            char_features=char_features,
            use_precomputed_embeddings=use_precomputed_embeddings,
        ).fit(unique_states)
        allowed_group_keys = {"user_id", "state_id", "card_id", "semantic_family"}
        if bootstrap_group_key not in allowed_group_keys:
            raise ValueError(
                "bootstrap_group_key must be one of "
                f"{sorted(allowed_group_keys)}; got {bootstrap_group_key!r}"
            )
        rows = [(state_map[label.state_id], label.action_id) for label in usable]
        x = builder.transform(rows)
        groups = [
            str(getattr(state_map[label.state_id], bootstrap_group_key)) for label in usable
        ]
        state_action_counts: dict[str, int] = {}
        for label in usable:
            state_action_counts[label.state_id] = (
                state_action_counts.get(label.state_id, 0) + 1
            )
        # Equalize states, while the bootstrap resamples the configured higher-level
        # group (user_id by default) to preserve within-user state dependence.
        state_weights = np.asarray(
            [1.0 / state_action_counts[label.state_id] for label in usable], dtype=float
        )

        def mad_weight(label: ActionLabel, *, prefix: str, field_name: str) -> float:
            mad = float(label.dimension_mad[f"{prefix}.{field_name}"])
            # A Cauchy inverse-variance proxy: MAD=scale receives half weight,
            # agreement receives full weight, and no finite row is silently dropped.
            return float(1.0 / (1.0 + (mad / dimension_mad_scale) ** 2))

        def weight_report(values: np.ndarray) -> dict[str, float]:
            total = float(np.sum(values))
            return {
                "minimum": float(np.min(values)),
                "mean": float(np.mean(values)),
                "maximum": float(np.max(values)),
                "effective_sample_size": float(
                    total**2 / max(float(np.sum(values**2)), 1e-12)
                ),
            }

        response_heads: dict[str, BootstrapRegressor] = {}
        response_weight_reports: dict[str, dict[str, float]] = {}
        for field_name in RESPONSE_FIELDS:
            y = np.asarray(
                [
                    (float(getattr(label.response, field_name)) - 1.0) / 4.0
                    for label in usable
                ],
                dtype=float,
            )
            head_weights = state_weights * np.asarray(
                [
                    mad_weight(label, prefix="response", field_name=field_name)
                    for label in usable
                ],
                dtype=float,
            )
            response_weight_reports[field_name] = weight_report(head_weights)
            response_heads[field_name] = BootstrapRegressor(
                n_models=n_models, seed=seed
            ).fit(x, y, groups, head_weights)

        risk_heads: dict[str, BootstrapRegressor] = {}
        risk_weight_reports: dict[str, dict[str, float]] = {}
        for field_name in RISK_FIELDS:
            y = np.asarray(
                [float(getattr(label.risk, field_name)) / 3.0 for label in usable],
                dtype=float,
            )
            head_weights = state_weights * np.asarray(
                [
                    mad_weight(label, prefix="risk", field_name=field_name)
                    for label in usable
                ],
                dtype=float,
            )
            risk_weight_reports[field_name] = weight_report(head_weights)
            risk_heads[field_name] = BootstrapRegressor(
                n_models=n_models, seed=seed + 101
            ).fit(x, y, groups, head_weights)

        report = {
            "n_states": len(unique_states),
            "n_action_labels": len(usable),
            "n_unreliable_labels_included": int(
                sum(not label.label_reliable for label in usable)
            ),
            "label_reliable_rate_diagnostic": float(
                np.mean([label.label_reliable for label in usable])
            ),
            "response_fields": list(RESPONSE_FIELDS),
            "risk_fields": list(RISK_FIELDS),
            "feature_config_hash": builder.config_hash(),
            "estimated_resource_cost_contract": builder.cost_contract(),
            "word_hash_features": word_features,
            "char_hash_features": char_features,
            "use_precomputed_embeddings": use_precomputed_embeddings,
            "m0_r0_coverage": 1.0,
            "complete_legal_action_coverage": 1.0,
            "row_reliability_used_for_filtering": False,
            "dimension_mad_weighting": {
                "formula": "1 / (1 + (dimension_mad / scale)^2)",
                "scale": float(dimension_mad_scale),
                "response_heads": response_weight_reports,
                "risk_heads": risk_weight_reports,
            },
            "composite_weights_sha256": expected_weights_hash,
            "bootstrap_group_key": bootstrap_group_key,
            "bootstrap_unique_groups": len(set(groups)),
            "joint_response_bootstrap_members": True,
            "bootstrap_fit_diagnostics": {
                "response_members": response_heads[
                    RESPONSE_FIELDS[0]
                ].fit_diagnostics,
                "risk_members": risk_heads[RISK_FIELDS[0]].fit_diagnostics,
            },
        }
        return cls(
            feature_builder=builder,
            response_heads=response_heads,
            risk_heads=risk_heads,
            selection_config=effective_selection,
            training_report=report,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "PMV2Model":
        value = joblib.load(path)
        if not isinstance(value, PMV2Model):
            raise TypeError("checkpoint is not a PMV2Model")
        if value.format_version != "pm-v2.1":
            raise RuntimeError(f"unsupported PM-v2 format: {value.format_version}")
        if set(value.response_conformal_radii) != set(RESPONSE_FIELDS):
            raise RuntimeError("PM-v2.1 checkpoint lacks calibrated response radii")
        if set(value.risk_conformal_radii) != set(RISK_FIELDS):
            raise RuntimeError("PM-v2.1 checkpoint lacks calibrated risk radii")
        all_radii = [
            *value.response_conformal_radii.values(),
            *value.risk_conformal_radii.values(),
            value.quality_conformal_radius,
        ]
        if any(not np.isfinite(radius) or radius < 0.0 for radius in all_radii):
            raise RuntimeError("PM-v2.1 checkpoint has invalid conformal radii")
        if value.conformal_calibration_report.get("status") != "COMPLETE":
            raise RuntimeError("PM-v2.1 checkpoint lacks completed conformal calibration")
        return value

    @staticmethod
    def _interval(
        mean: float,
        std: float,
        *,
        z: float,
        conformal_radius: float = 0.0,
        low: float,
        high: float,
    ) -> PredictionInterval:
        if not np.isfinite(conformal_radius) or conformal_radius < 0.0:
            raise ValueError("conformal radius must be finite and non-negative")
        return PredictionInterval(
            mean=float(np.clip(mean, low, high)),
            std=float(max(0.0, std)),
            lower=float(np.clip(mean - z * std - conformal_radius, low, high)),
            upper=float(np.clip(mean + z * std + conformal_radius, low, high)),
        )

    def _conformal_radius(self, kind: str, field_name: str) -> float:
        radii = (
            self.response_conformal_radii
            if kind == "response"
            else self.risk_conformal_radii
        )
        expected = set(RESPONSE_FIELDS if kind == "response" else RISK_FIELDS)
        if radii and set(radii) != expected:
            raise RuntimeError(
                f"PM-v2 checkpoint has incomplete {kind} conformal radii: "
                f"expected={sorted(expected)}, observed={sorted(radii)}"
            )
        return float(radii.get(field_name, 0.0))

    def _prediction_bundle(self, state: PMV2State) -> _PredictionBundle:
        actions = list(state.allowed_actions)
        x = self.feature_builder.transform([(state, action) for action in actions])
        z = self.selection_config.uncertainty_z
        response: dict[str, dict[str, PredictionInterval]] = {action: {} for action in actions}
        risk: dict[str, dict[str, PredictionInterval]] = {action: {} for action in actions}
        if set(self.response_heads) != set(RESPONSE_FIELDS):
            raise RuntimeError("PM-v2 checkpoint lacks the exact response head set")
        if set(self.risk_heads) != set(RISK_FIELDS):
            raise RuntimeError("PM-v2 checkpoint lacks the exact risk head set")
        response_members: dict[str, np.ndarray] = {}
        response_member_count: int | None = None
        for field_name in RESPONSE_FIELDS:
            matrix = self.response_heads[field_name].predict_members(x)
            if matrix.shape[1] != len(actions):
                raise RuntimeError(f"response head {field_name} returned wrong action count")
            if response_member_count is None:
                response_member_count = matrix.shape[0]
            elif matrix.shape[0] != response_member_count:
                raise RuntimeError("response heads have unaligned bootstrap member counts")
            # Clip each member before aggregation so the joint composite remains in
            # the preregistered [0, 1] quality scale.
            normalized_members = np.clip(matrix, 0.0, 1.0)
            response_members[field_name] = normalized_members
            for index, action in enumerate(actions):
                values = 1.0 + 4.0 * normalized_members[:, index]
                response[action][field_name] = self._interval(
                    float(np.mean(values)),
                    float(np.std(values, ddof=0)),
                    z=z,
                    conformal_radius=self._conformal_radius(
                        "response", field_name
                    ),
                    low=1.0,
                    high=5.0,
                )
        for field_name in RISK_FIELDS:
            matrix = self.risk_heads[field_name].predict_members(x)
            if matrix.shape[1] != len(actions):
                raise RuntimeError(f"risk head {field_name} returned wrong action count")
            normalized_members = np.clip(matrix, 0.0, 1.0)
            for index, action in enumerate(actions):
                values = 3.0 * normalized_members[:, index]
                risk[action][field_name] = self._interval(
                    float(np.mean(values)),
                    float(np.std(values, ddof=0)),
                    z=z,
                    conformal_radius=self._conformal_radius("risk", field_name),
                    low=0.0,
                    high=3.0,
                )
        spec = self.selection_config.composite_spec
        quality_members = {
            action: sum(
                float(spec.weights[field_name]) * response_members[field_name][:, index]
                for field_name in RESPONSE_FIELDS
            )
            for index, action in enumerate(actions)
        }
        return _PredictionBundle(
            response=response,
            risk=risk,
            quality_members=quality_members,
        )

    def _raw_predictions(
        self, state: PMV2State
    ) -> tuple[dict[str, dict[str, PredictionInterval]], dict[str, dict[str, PredictionInterval]]]:
        """Compatibility view of the dimension-level prediction bundle."""

        bundle = self._prediction_bundle(state)
        return bundle.response, bundle.risk

    def predict_actions(self, state: PMV2State) -> dict[str, ActionPrediction]:
        bundle = self._prediction_bundle(state)
        response, risk = bundle.response, bundle.risk
        actions = list(state.allowed_actions)
        cost_profile = estimated_action_cost_profile(self.feature_builder, state)
        z = self.selection_config.uncertainty_z
        quality_mean = {
            action: float(np.mean(bundle.quality_members[action])) for action in actions
        }
        quality_lcb = {
            action: float(
                np.clip(
                    np.mean(bundle.quality_members[action])
                    - z * np.std(bundle.quality_members[action], ddof=0)
                    - self.quality_conformal_radius,
                    0.0,
                    1.0,
                )
            )
            for action in actions
        }

        def paired_quality_delta_lcb(action: str, baseline: str) -> float:
            delta = bundle.quality_members[action] - bundle.quality_members[baseline]
            # The marginal conformal quality bound applies to each endpoint;
            # subtracting twice the radius gives a conservative difference bound.
            return float(
                np.mean(delta)
                - z * np.std(delta, ddof=0)
                - 2.0 * self.quality_conformal_radius
            )

        risk_ucb = {}
        risk_violations: dict[str, list[str]] = {}
        for action in actions:
            applicable = applicable_risk_fields(action)
            normalized = {
                name: risk[action][name].upper / 3.0 for name in applicable
            }
            risk_ucb[action] = max(normalized.values(), default=0.0)
            risk_violations[action] = [
                name
                for name, value in normalized.items()
                if value > self.selection_config.risk_threshold(name)
            ]
        predictions: dict[str, ActionPrediction] = {}
        for action in actions:
            sources, strategy = parse_action_id(action)
            reasons: list[str] = []
            feasible = not risk_violations[action]
            if not feasible:
                reasons.extend(
                    f"risk_ucb_exceeds_threshold:{name}"
                    for name in risk_violations[action]
                )
            resource_gate = True
            if sources:
                # Isolate memory's incremental value from the strategy decision.
                # In particular, M+RS must beat M0+RS rather than M0+R0.
                memory_baseline = canonical_action_id(frozenset(), strategy)
                if memory_baseline not in actions:
                    raise RuntimeError(
                        f"state {state.state_id} lacks same-strategy memory baseline "
                        f"{memory_baseline}"
                    )
                baseline_omission_ucb = (
                    risk[memory_baseline]["memory_omission"].upper / 3.0
                )
                resource_gain = paired_quality_delta_lcb(action, memory_baseline)
                resource_gate = (
                    resource_gain >= self.selection_config.resource_min_gain
                    or baseline_omission_ucb >= self.selection_config.m0_omission_trigger
                )
                if not resource_gate:
                    reasons.append("resource_benefit_not_supported")
            strategy_gate = True
            if strategy is StrategyMode.RS:
                r0_action = canonical_action_id(sources, StrategyMode.R0)
                if r0_action in actions:
                    r0_strategy_omission = risk[r0_action]["strategy_omission"].upper / 3.0
                    strategy_gain = paired_quality_delta_lcb(action, r0_action)
                    strategy_gate = (
                        strategy_gain >= self.selection_config.strategy_min_gain
                        or r0_strategy_omission
                        >= self.selection_config.r0_strategy_omission_trigger
                    )
                    if not strategy_gate:
                        reasons.append("strategy_benefit_not_supported")
            estimated_cost = cost_profile[action]["estimated_resource_cost"]
            normalized_cost = cost_profile[action]["normalized_estimated_resource_cost"]
            utility = (
                quality_lcb[action]
                - self.selection_config.risk_weight * risk_ucb[action]
                - self.selection_config.cost_weight * normalized_cost
            )
            predictions[action] = ActionPrediction(
                action_id=action,
                response=response[action],
                risk=risk[action],
                quality_mean=quality_mean[action],
                quality_lcb=quality_lcb[action],
                risk_ucb=risk_ucb[action],
                estimated_cost=estimated_cost,
                normalized_cost=normalized_cost,
                utility=utility,
                feasible=feasible,
                resource_gate_passed=resource_gate,
                strategy_gate_passed=strategy_gate,
                exclusion_reasons=reasons,
            )
        return predictions

    def choose(self, state: PMV2State) -> PolicyDecision:
        ood = self.feature_builder.ood_report(state)
        predictions = self.predict_actions(state)
        fallback = self.selection_config.fallback_action
        if fallback not in predictions:
            fallback = min(predictions, key=lambda action: predictions[action].estimated_cost)
        severe_ood = bool(ood["severe_semantic_ood"] or ood["severe_metadata_ood"])
        if severe_ood and self.selection_config.fail_closed_on_ood:
            chosen = fallback
            reason = SEVERE_OOD_FALLBACK_REASON
            fallback_used = True
        else:
            candidates = [
                prediction
                for prediction in predictions.values()
                if prediction.feasible
                and prediction.resource_gate_passed
                and prediction.strategy_gate_passed
            ]
            if not candidates:
                chosen = fallback
                reason = NO_FEASIBLE_FALLBACK_REASON
                # This is deliberately not an OOD fallback. The machine-readable
                # decision reason lets evaluation count it separately.
                fallback_used = False
            else:
                selected = max(
                    candidates,
                    key=lambda item: (
                        item.utility,
                        item.quality_lcb,
                        -item.risk_ucb,
                        -item.estimated_cost,
                        item.action_id,
                    ),
                )
                chosen = selected.action_id
                reason = LEARNED_SELECTION_REASON
                fallback_used = False
        return PolicyDecision(
            state_id=state.state_id,
            chosen_action=chosen,
            predictions=predictions,
            semantic_ood_score=float(ood["semantic_ood_score"]),
            metadata_ood_score=float(ood["metadata_ood_score"]),
            ood_fallback_used=fallback_used,
            decision_reason=reason,
            config_hash=self.selection_config.digest(),
        )


def decision_fallback_kind(decision: PolicyDecision) -> str | None:
    """Return the explicit fallback class encoded by :meth:`PMV2Model.choose`.

    ``ood_fallback_used`` remains the backwards-compatible severe-OOD indicator.
    No-feasible fallbacks are separately encoded in ``decision_reason`` so they
    cannot be counted as learned abstention/strategy-off decisions.
    """

    if decision.decision_reason == SEVERE_OOD_FALLBACK_REASON:
        if not decision.ood_fallback_used:
            raise RuntimeError("severe-OOD fallback is missing its OOD fallback flag")
        return "severe_ood"
    if decision.decision_reason == NO_FEASIBLE_FALLBACK_REASON:
        if decision.ood_fallback_used:
            raise RuntimeError("no-feasible fallback cannot carry the OOD fallback flag")
        return "no_feasible"
    if decision.decision_reason == LEARNED_SELECTION_REASON:
        if decision.ood_fallback_used:
            raise RuntimeError("learned decision cannot carry the OOD fallback flag")
        return None
    raise RuntimeError(
        f"unrecognized PM-v2 decision reason for state {decision.state_id}: "
        f"{decision.decision_reason!r}"
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1) + 1.0
        index = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def cost_calibration_diagnostics(
    estimated_resource_costs: Sequence[float],
    observed_input_tokens: Sequence[float],
) -> dict[str, Any]:
    """Diagnose, but never substitute, observed tokens for selector cost."""

    estimated = np.asarray(estimated_resource_costs, dtype=float)
    observed = np.asarray(observed_input_tokens, dtype=float)
    if estimated.shape != observed.shape or estimated.ndim != 1 or not len(estimated):
        raise ValueError("estimated and observed cost vectors must be non-empty and aligned")
    if not np.all(np.isfinite(estimated)) or not np.all(np.isfinite(observed)):
        raise ValueError("cost diagnostics require finite values")
    if np.any(estimated < 0.0) or np.any(observed < 0.0):
        raise ValueError("cost diagnostics require non-negative values")
    pearson = _correlation(estimated, observed)
    spearman = _correlation(_average_ranks(estimated), _average_ranks(observed))
    estimated_variance = float(np.var(estimated))
    if estimated_variance <= 1e-12:
        slope = None
        intercept = float(np.mean(observed))
        calibrated = np.full_like(observed, intercept)
    else:
        slope = float(np.cov(estimated, observed, ddof=0)[0, 1] / estimated_variance)
        intercept = float(np.mean(observed) - slope * np.mean(estimated))
        calibrated = intercept + slope * estimated
    residual = observed - calibrated
    total_variance = float(np.sum((observed - np.mean(observed)) ** 2))
    r_squared = (
        None
        if total_variance <= 1e-12
        else float(1.0 - np.sum(residual**2) / total_variance)
    )
    return {
        "n": int(len(estimated)),
        "mean_estimated_resource_cost": float(np.mean(estimated)),
        "mean_observed_input_tokens": float(np.mean(observed)),
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
        "observed_on_estimated_ols": {
            "intercept": intercept,
            "slope": slope,
            "r_squared": r_squared,
            "mean_absolute_error": float(np.mean(np.abs(residual))),
            "root_mean_squared_error": float(np.sqrt(np.mean(residual**2))),
        },
        "raw_mean_observed_minus_estimated": float(np.mean(observed - estimated)),
    }


def evaluate_cost_diagnostics(
    model: PMV2Model,
    states: Sequence[PMV2State],
    labels: Sequence[ActionLabel],
) -> dict[str, Any]:
    """Compare estimated resource cost with observed tokens on all legal actions."""

    state_map = {state.state_id: state for state in states}
    if len(state_map) != len(states) or not state_map:
        raise ValueError("cost diagnostics require unique, non-empty states")
    label_keys = [(label.state_id, label.action_id) for label in labels]
    if len(label_keys) != len(set(label_keys)):
        raise ValueError("duplicate PM-v2 cost diagnostic labels")
    unknown = sorted({label.state_id for label in labels} - set(state_map))
    if unknown:
        raise ValueError(f"cost diagnostic labels reference unknown states: {unknown[:10]}")
    label_map = {(label.state_id, label.action_id): label for label in labels}
    estimated: list[float] = []
    observed: list[float] = []
    by_action: dict[str, dict[str, list[float]]] = {}
    incomplete: list[tuple[str, str]] = []
    for state_id, state in sorted(state_map.items()):
        profile = estimated_action_cost_profile(model.feature_builder, state)
        for action_id in state.allowed_actions:
            label = label_map.get((state_id, action_id))
            if label is None:
                incomplete.append((state_id, action_id))
                continue
            estimate = profile[action_id]["estimated_resource_cost"]
            actual = float(label.observed_input_tokens)
            estimated.append(estimate)
            observed.append(actual)
            bucket = by_action.setdefault(action_id, {"estimated": [], "observed": []})
            bucket["estimated"].append(estimate)
            bucket["observed"].append(actual)
    if incomplete or len(estimated) != sum(len(state.allowed_actions) for state in states):
        raise ValueError(
            "cost diagnostics require the exact legal action matrix; "
            f"missing={incomplete[:10]}"
        )
    return {
        "all_legal_actions": cost_calibration_diagnostics(estimated, observed),
        "by_action": {
            action_id: cost_calibration_diagnostics(values["estimated"], values["observed"])
            for action_id, values in sorted(by_action.items())
        },
    }


def evaluate_prediction_coverage(
    model: PMV2Model,
    states: Sequence[PMV2State],
    labels: Sequence[ActionLabel],
    *,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Measure head-wise coverage using users as the independent blocks.

    Every legal action remains in the diagnostic table, so a collapsed selector
    cannot hide uncertainty failures.  Confidence bounds, however, are computed
    over *user blocks*: a user is a hit for a head only when every applicable
    state/action label for that user is covered.  Treating the 16 correlated
    actions of a state (or nine states from one synthetic user) as independent
    Bernoulli trials would produce spuriously narrow Wilson bounds.

    The result is head-wise, not simultaneous across all 13 response/risk heads.
    """

    if not 0.5 < confidence_level < 1.0:
        raise ValueError("coverage confidence_level must be in (0.5, 1.0)")
    critical_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    state_map = {state.state_id: state for state in states}
    if len(state_map) != len(states):
        raise ValueError("duplicate PM-v2 coverage state_id")
    if not state_map:
        raise ValueError("cannot evaluate PM-v2 coverage without states")
    label_keys = [(label.state_id, label.action_id) for label in labels]
    if len(label_keys) != len(set(label_keys)):
        raise ValueError("duplicate PM-v2 coverage action labels")
    unknown_states = sorted({label.state_id for label in labels} - set(state_map))
    if unknown_states:
        raise ValueError(
            f"PM-v2 coverage labels reference unknown states: {unknown_states[:10]}"
        )
    label_map = {(label.state_id, label.action_id): label for label in labels}
    incomplete: list[dict[str, Any]] = []
    for state_id, state in sorted(state_map.items()):
        observed = {
            action_id
            for candidate_state_id, action_id in label_map
            if candidate_state_id == state_id
        }
        expected = set(state.allowed_actions)
        if observed != expected:
            incomplete.append(
                {
                    "state_id": state_id,
                    "missing": sorted(expected - observed),
                    "extra": sorted(observed - expected),
                }
            )
    if incomplete:
        raise ValueError(
            "PM-v2 coverage requires exact labels for every legal action; "
            f"incomplete={incomplete[:10]}"
        )

    response_hits: dict[str, list[bool]] = {name: [] for name in RESPONSE_FIELDS}
    response_widths: dict[str, list[float]] = {name: [] for name in RESPONSE_FIELDS}
    risk_hits: dict[str, list[bool]] = {name: [] for name in RISK_FIELDS}
    risk_widths: dict[str, list[float]] = {name: [] for name in RISK_FIELDS}
    quality_hits: list[bool] = []
    quality_widths: list[float] = []
    users = sorted({state.user_id for state in states})
    response_hits_by_user: dict[str, dict[str, list[bool]]] = {
        name: {user_id: [] for user_id in users} for name in RESPONSE_FIELDS
    }
    risk_hits_by_user: dict[str, dict[str, list[bool]]] = {
        name: {user_id: [] for user_id in users} for name in RISK_FIELDS
    }
    quality_hits_by_user: dict[str, list[bool]] = {user_id: [] for user_id in users}
    z = model.selection_config.uncertainty_z
    for state_id, state in sorted(state_map.items()):
        bundle = model._prediction_bundle(state)
        for action_id in state.allowed_actions:
            label = label_map[(state_id, action_id)]
            for name in RESPONSE_FIELDS:
                interval = bundle.response[action_id][name]
                observed = float(getattr(label.response, name))
                hit = interval.lower <= observed <= interval.upper
                response_hits[name].append(hit)
                response_hits_by_user[name][state.user_id].append(hit)
                response_widths[name].append(interval.upper - interval.lower)
            for name in applicable_risk_fields(action_id):
                interval = bundle.risk[action_id][name]
                observed = float(getattr(label.risk, name))
                hit = interval.lower <= observed <= interval.upper
                risk_hits[name].append(hit)
                risk_hits_by_user[name][state.user_id].append(hit)
                risk_widths[name].append(interval.upper - interval.lower)
            members = bundle.quality_members[action_id]
            mean = float(np.mean(members))
            std = float(np.std(members, ddof=0))
            lower = float(
                np.clip(mean - z * std - model.quality_conformal_radius, 0.0, 1.0)
            )
            upper = float(
                np.clip(mean + z * std + model.quality_conformal_radius, 0.0, 1.0)
            )
            observed_quality = model.selection_config.composite_spec.score(label.response)
            hit = lower <= observed_quality <= upper
            quality_hits.append(hit)
            quality_hits_by_user[state.user_id].append(hit)
            quality_widths.append(upper - lower)

    def summarize(hits: list[bool], widths: list[float]) -> dict[str, Any]:
        if not hits:
            return {
                "n": 0,
                "coverage": 0.0,
                "coverage_lower_bound": 0.0,
                "mean_width": 0.0,
            }
        n = len(hits)
        coverage = float(np.mean(hits))
        denominator = 1.0 + critical_value**2 / n
        centre = (coverage + critical_value**2 / (2.0 * n)) / denominator
        radius = (
            critical_value
            * np.sqrt(
                coverage * (1.0 - coverage) / n
                + critical_value**2 / (4.0 * n**2)
            )
            / denominator
        )
        return {
            "n": n,
            "coverage": coverage,
            "coverage_lower_bound": float(max(0.0, centre - radius)),
            "mean_width": float(np.mean(widths)),
        }

    def summarize_user_blocks(
        hits_by_user: dict[str, list[bool]], widths: list[float]
    ) -> dict[str, Any]:
        nonempty = {
            user_id: values for user_id, values in hits_by_user.items() if values
        }
        block_hits = [all(nonempty[user_id]) for user_id in sorted(nonempty)]
        summary = summarize(block_hits, widths)
        within_user = [
            float(np.mean(nonempty[user_id])) for user_id in sorted(nonempty)
        ]
        summary.update(
            {
                "unit": "user_block_all_applicable_state_actions",
                "n_underlying_state_actions": int(
                    sum(len(values) for values in nonempty.values())
                ),
                "mean_within_user_coverage": (
                    float(np.mean(within_user)) if within_user else 0.0
                ),
                "minimum_within_user_coverage": (
                    float(min(within_user)) if within_user else 0.0
                ),
            }
        )
        return summary

    response_summary = {
        name: summarize_user_blocks(
            response_hits_by_user[name], response_widths[name]
        )
        for name in RESPONSE_FIELDS
    }
    risk_summary = {
        name: summarize_user_blocks(risk_hits_by_user[name], risk_widths[name])
        for name in RISK_FIELDS
    }
    response_state_action_diagnostics = {
        name: summarize(response_hits[name], response_widths[name])
        for name in RESPONSE_FIELDS
    }
    risk_state_action_diagnostics = {
        name: summarize(risk_hits[name], risk_widths[name]) for name in RISK_FIELDS
    }
    expected_state_actions = sum(len(state.allowed_actions) for state in states)
    return {
        "n_states": len(states),
        "n_users": len(users),
        "n_state_actions": len(quality_hits),
        "expected_state_actions": expected_state_actions,
        "coverage_confidence_level": confidence_level,
        "coverage_unit": "user_block_all_legal_actions",
        "coverage_scope": "headwise_not_joint_13_head",
        "quality": summarize_user_blocks(quality_hits_by_user, quality_widths),
        "response_dimensions": response_summary,
        "risk_dimensions_applicable_only": risk_summary,
        "state_action_marginal_diagnostics": {
            "quality": summarize(quality_hits, quality_widths),
            "response_dimensions": response_state_action_diagnostics,
            "risk_dimensions_applicable_only": risk_state_action_diagnostics,
        },
        "minimum_response_dimension_coverage": float(
            min(row["coverage"] for row in response_summary.values())
        ),
        "minimum_risk_dimension_coverage": float(
            min(row["coverage"] for row in risk_summary.values())
        ),
        "minimum_response_dimension_coverage_lower_bound": float(
            min(row["coverage_lower_bound"] for row in response_summary.values())
        ),
        "minimum_risk_dimension_coverage_lower_bound": float(
            min(row["coverage_lower_bound"] for row in risk_summary.values())
        ),
    }


def _finite_sample_conformal_radius(
    residuals: Sequence[float], target_coverage: float
) -> dict[str, Any]:
    """Return the standard finite-sample ``higher`` split-conformal quantile."""

    values = np.asarray(residuals, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("split conformal requires non-empty one-dimensional residuals")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("split-conformal residuals must be finite and non-negative")
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("split-conformal target coverage must be in (0, 1)")
    rank = int(np.ceil((values.size + 1) * target_coverage))
    if rank > values.size:
        raise ValueError(
            "calibration sample is too small for requested finite-sample coverage: "
            f"n={values.size}, target={target_coverage}, maximum={values.size / (values.size + 1):.6f}"
        )
    radius = float(np.sort(values)[rank - 1])
    return {
        "n": int(values.size),
        "target_coverage": float(target_coverage),
        "finite_sample_rank": rank,
        "quantile_method": "ceil((n+1)*coverage), order_statistic_higher",
        "radius": radius,
        "mean_residual": float(np.mean(values)),
        "maximum_residual": float(np.max(values)),
        "empirical_fraction_at_or_below_radius": float(np.mean(values <= radius)),
    }


def calibrate_uncertainty_multiplier(
    model: PMV2Model,
    states: Sequence[PMV2State],
    labels: Sequence[ActionLabel],
    *,
    z_candidates: Iterable[float],
    target_coverage: float,
    minimum_quality_coverage_lower_bound: float,
    minimum_response_coverage_lower_bound: float,
    minimum_risk_coverage_lower_bound: float,
    coverage_confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Fit additive split-conformal residual radii on the calibration split.

    The bootstrap multiplier is preregistered rather than selected on the same
    residuals: exactly one candidate is accepted.  Nonconformity is first
    collapsed to one maximum per calibration user (over that user's states and
    legal actions), then the finite-sample quantile is taken across users.
    Response/risk heads receive native-scale, head-wise radii and the response
    composite receives its own radius.  The construction does not claim joint
    simultaneous coverage across all 13 heads.
    """

    candidates = sorted({float(value) for value in z_candidates})
    if len(candidates) != 1:
        raise ValueError(
            "split-conformal calibration requires exactly one preregistered "
            "bootstrap z; tuning z and residual radii on the same split is forbidden"
        )
    z = candidates[0]
    if not 0.0 <= z <= 3.0:
        raise ValueError("uncertainty z must be in [0, 3]")
    thresholds = {
        "quality": float(minimum_quality_coverage_lower_bound),
        "response": float(minimum_response_coverage_lower_bound),
        "risk": float(minimum_risk_coverage_lower_bound),
    }
    if any(not 0.0 <= value <= 1.0 for value in thresholds.values()):
        raise ValueError("uncertainty coverage thresholds must be in [0, 1]")

    state_map = {state.state_id: state for state in states}
    if len(state_map) != len(states) or not state_map:
        raise ValueError("split conformal requires unique non-empty calibration states")
    label_keys = [(label.state_id, label.action_id) for label in labels]
    if len(label_keys) != len(set(label_keys)):
        raise ValueError("split conformal received duplicate state-action labels")
    label_map = {(label.state_id, label.action_id): label for label in labels}
    expected_keys = {
        (state.state_id, action_id)
        for state in states
        for action_id in state.allowed_actions
    }
    if set(label_map) != expected_keys:
        raise ValueError(
            "split conformal requires the exact complete legal action matrix; "
            f"missing={sorted(expected_keys - set(label_map))[:10]}, "
            f"extra={sorted(set(label_map) - expected_keys)[:10]}"
        )
    for (state_id, _), label in label_map.items():
        if label.user_id != state_map[state_id].user_id:
            raise ValueError(f"split-conformal label user mismatch for {state_id}")

    original_config = model.selection_config
    original_response_radii = dict(model.response_conformal_radii)
    original_risk_radii = dict(model.risk_conformal_radii)
    original_quality_radius = float(model.quality_conformal_radius)
    original_report = dict(model.conformal_calibration_report)
    config_data = original_config.model_dump(mode="json")
    config_data["uncertainty_z"] = z
    model.selection_config = SelectionConfig.model_validate(config_data)
    # Calibration residuals must be computed against the uncalibrated bootstrap
    # intervals, never against radii left by an earlier run.
    model.response_conformal_radii = {}
    model.risk_conformal_radii = {}
    model.quality_conformal_radius = 0.0

    user_ids = sorted({state.user_id for state in states})
    response_residuals: dict[str, dict[str, list[float]]] = {
        name: {user_id: [] for user_id in user_ids} for name in RESPONSE_FIELDS
    }
    risk_residuals: dict[str, dict[str, list[float]]] = {
        name: {user_id: [] for user_id in user_ids} for name in RISK_FIELDS
    }
    quality_residuals: dict[str, list[float]] = {
        user_id: [] for user_id in user_ids
    }
    try:
        for state in sorted(states, key=lambda row: row.state_id):
            bundle = model._prediction_bundle(state)
            for action_id in state.allowed_actions:
                label = label_map[(state.state_id, action_id)]
                for name in RESPONSE_FIELDS:
                    interval = bundle.response[action_id][name]
                    observed = float(getattr(label.response, name))
                    response_residuals[name][state.user_id].append(
                        max(interval.lower - observed, observed - interval.upper, 0.0)
                    )
                for name in applicable_risk_fields(action_id):
                    interval = bundle.risk[action_id][name]
                    observed = float(getattr(label.risk, name))
                    risk_residuals[name][state.user_id].append(
                        max(interval.lower - observed, observed - interval.upper, 0.0)
                    )
                members = bundle.quality_members[action_id]
                mean = float(np.mean(members))
                std = float(np.std(members, ddof=0))
                observed_quality = model.selection_config.composite_spec.score(
                    label.response
                )
                quality_residuals[state.user_id].append(
                    max(abs(observed_quality - mean) - z * std, 0.0)
                )

        def user_block_maxima(
            values_by_user: dict[str, list[float]], *, head: str
        ) -> list[float]:
            missing = sorted(
                user_id for user_id, values in values_by_user.items() if not values
            )
            if missing:
                raise ValueError(
                    f"split conformal head {head} lacks applicable labels for users "
                    f"{missing}"
                )
            return [
                max(values_by_user[user_id]) for user_id in sorted(values_by_user)
            ]

        response_reports = {
            name: _finite_sample_conformal_radius(
                user_block_maxima(values, head=name), target_coverage
            )
            for name, values in response_residuals.items()
        }
        risk_reports = {
            name: _finite_sample_conformal_radius(
                user_block_maxima(values, head=name), target_coverage
            )
            for name, values in risk_residuals.items()
        }
        quality_report = _finite_sample_conformal_radius(
            user_block_maxima(quality_residuals, head="quality_composite"),
            target_coverage,
        )
        model.response_conformal_radii = {
            name: float(report["radius"])
            for name, report in response_reports.items()
        }
        model.risk_conformal_radii = {
            name: float(report["radius"]) for name, report in risk_reports.items()
        }
        model.quality_conformal_radius = float(quality_report["radius"])
        coverage = evaluate_prediction_coverage(
            model,
            states,
            labels,
            confidence_level=coverage_confidence_level,
        )
        # Calibration data fit the radii; Wilson bounds on this same split are
        # descriptive only.  The preregistered lower-bound thresholds are
        # enforced later on independent internal users.
        checks = {
            "finite_sample_user_block_target_attainable": all(
                report["finite_sample_rank"] <= report["n"]
                for report in [
                    quality_report,
                    *response_reports.values(),
                    *risk_reports.values(),
                ]
            ),
            "calibration_empirical_user_block_coverage_at_least_target": (
                coverage["quality"]["coverage"] >= target_coverage
                and min(
                    row["coverage"] for row in coverage["response_dimensions"].values()
                )
                >= target_coverage
                and min(
                    row["coverage"]
                    for row in coverage["risk_dimensions_applicable_only"].values()
                )
                >= target_coverage
            ),
        }
        report = {
            "status": "COMPLETE" if all(checks.values()) else "FAIL",
            "method": "additive_user_block_split_conformal_residual_radius",
            "calibration_unit": "user_block_max_over_states_and_legal_actions",
            "coverage_scope": "headwise_not_joint_13_head",
            "n_calibration_users": len(user_ids),
            "bootstrap_z": z,
            "target_coverage": float(target_coverage),
            "coverage_confidence_level": float(coverage_confidence_level),
            "independent_internal_coverage_lower_bound_thresholds": thresholds,
            "response_heads": response_reports,
            "risk_heads_applicable_only": risk_reports,
            "quality_composite": quality_report,
            "coverage": coverage,
            "checks": checks,
            # Compatibility shape for callers that previously inspected selected.
            "selected": {
                "uncertainty_z": z,
                "response_conformal_radii": dict(model.response_conformal_radii),
                "risk_conformal_radii": dict(model.risk_conformal_radii),
                "quality_conformal_radius": model.quality_conformal_radius,
                "passed": all(checks.values()),
            },
            "candidates": [{"uncertainty_z": z, "passed": all(checks.values())}],
        }
        if not all(checks.values()):
            raise RuntimeError(
                "user-block split-conformal calibration failed structural/empirical "
                f"sanity checks: {report}"
            )
        model.conformal_calibration_report = report
        return report
    except Exception:
        model.selection_config = original_config
        model.response_conformal_radii = original_response_radii
        model.risk_conformal_radii = original_risk_radii
        model.quality_conformal_radius = original_quality_radius
        model.conformal_calibration_report = original_report
        raise


def evaluate_policy(
    model: PMV2Model,
    states: Sequence[PMV2State],
    labels: Sequence[ActionLabel],
) -> dict[str, Any]:
    state_map = {state.state_id: state for state in states}
    if len(state_map) != len(states):
        raise ValueError("duplicate PM-v2 evaluation state_id")
    if not state_map:
        raise ValueError("cannot evaluate PM-v2 without states")
    label_keys = [(label.state_id, label.action_id) for label in labels]
    if len(label_keys) != len(set(label_keys)):
        raise ValueError("duplicate PM-v2 evaluation action labels")
    label_map = {(label.state_id, label.action_id): label for label in labels}
    rows: list[dict[str, Any]] = []
    missing_selected_labels: list[tuple[str, str]] = []
    for state_id, state in sorted(state_map.items()):
        decision = model.choose(state)
        if decision.state_id != state_id:
            raise RuntimeError(
                f"policy returned state_id={decision.state_id!r} for {state_id!r}"
            )
        label = label_map.get((state_id, decision.chosen_action))
        if label is None:
            missing_selected_labels.append((state_id, decision.chosen_action))
            continue
        if label.user_id != state.user_id:
            raise ValueError(
                f"selected label user_id mismatch for state {state_id}: "
                f"state={state.user_id!r}, label={label.user_id!r}"
            )
        quality = model.selection_config.composite_spec.score(label.response)
        risk = max(
            float(getattr(label.risk, name)) / 3.0
            for name in applicable_risk_fields(decision.chosen_action)
        )
        sources, strategy = parse_action_id(decision.chosen_action)
        fallback_kind = decision_fallback_kind(decision)
        learned = fallback_kind is None
        chosen_prediction = decision.predictions[decision.chosen_action]
        normalized_estimated_cost = float(chosen_prediction.normalized_cost)
        realized_utility = (
            quality
            - model.selection_config.risk_weight * risk
            - model.selection_config.cost_weight * normalized_estimated_cost
        )
        rows.append(
            {
                "state_id": state_id,
                "user_id": state.user_id,
                "action_id": decision.chosen_action,
                "quality": quality,
                "risk": risk,
                "realized_utility": realized_utility,
                "estimated_resource_cost": float(chosen_prediction.estimated_cost),
                "normalized_estimated_resource_cost": normalized_estimated_cost,
                "observed_input_tokens": float(label.observed_input_tokens),
                # Backwards-compatible resource-report alias. It is not used by
                # selection or tuning utility.
                "cost": float(label.observed_input_tokens),
                "response_dimensions": {
                    name: float(getattr(label.response, name)) for name in RESPONSE_FIELDS
                },
                "m0": not bool(sources),
                "r0": strategy is StrategyMode.R0,
                "learned": learned,
                "learned_m0": learned and not bool(sources),
                "learned_r0": learned and strategy is StrategyMode.R0,
                "nonfallback_m0_r0": (
                    learned and not bool(sources) and strategy is StrategyMode.R0
                ),
                "fallback_type": fallback_kind,
                "fallback": fallback_kind is not None,
                "severe_ood_fallback": fallback_kind == "severe_ood",
                "no_feasible_fallback": fallback_kind == "no_feasible",
                "ood_fallback": fallback_kind == "severe_ood",
            }
        )
    if missing_selected_labels:
        raise ValueError(
            "PM-v2 evaluation is missing labels for selected actions; "
            f"missing={missing_selected_labels[:10]}"
        )
    if len(rows) != len(state_map):
        raise RuntimeError(
            f"PM-v2 evaluation row count mismatch: expected={len(state_map)}, "
            f"observed={len(rows)}"
        )
    actions = [row["action_id"] for row in rows]
    counts = {action: actions.count(action) for action in sorted(set(actions))}
    probabilities = np.asarray(list(counts.values()), dtype=float) / len(rows)
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    learned_actions = [row["action_id"] for row in rows if row["learned"]]
    learned_counts = {
        action: learned_actions.count(action) for action in sorted(set(learned_actions))
    }
    if learned_actions:
        learned_probabilities = np.asarray(list(learned_counts.values()), dtype=float) / len(
            learned_actions
        )
        learned_entropy = float(
            -np.sum(learned_probabilities * np.log2(learned_probabilities))
        )
        learned_maximum_share = float(max(learned_counts.values()) / len(learned_actions))
        learned_m0_share = float(
            np.mean([row["m0"] for row in rows if row["learned"]])
        )
        learned_r0_share = float(
            np.mean([row["r0"] for row in rows if row["learned"]])
        )
    else:
        # Fail-closed values for distribution gates when nothing was learned.
        learned_entropy = 0.0
        learned_maximum_share = 1.0
        learned_m0_share = 0.0
        learned_r0_share = 0.0
    mean_response_dimensions = {
        name: float(np.mean([row["response_dimensions"][name] for row in rows]))
        for name in RESPONSE_FIELDS
    }
    selected_cost_diagnostics = cost_calibration_diagnostics(
        [row["estimated_resource_cost"] for row in rows],
        [row["observed_input_tokens"] for row in rows],
    )
    return {
        "n": len(rows),
        "expected_n": len(state_map),
        "mean_quality": float(np.mean([row["quality"] for row in rows])),
        "mean_response_dimensions": mean_response_dimensions,
        "mean_risk": float(np.mean([row["risk"] for row in rows])),
        "mean_realized_utility": float(
            np.mean([row["realized_utility"] for row in rows])
        ),
        "mean_estimated_resource_cost": float(
            np.mean([row["estimated_resource_cost"] for row in rows])
        ),
        "mean_normalized_estimated_resource_cost": float(
            np.mean([row["normalized_estimated_resource_cost"] for row in rows])
        ),
        "mean_observed_input_tokens": float(
            np.mean([row["observed_input_tokens"] for row in rows])
        ),
        "estimated_vs_observed_selected_cost": selected_cost_diagnostics,
        "mean_cost": float(np.mean([row["observed_input_tokens"] for row in rows])),
        "m0_rate": float(np.mean([row["m0"] for row in rows])),
        "r0_rate": float(np.mean([row["r0"] for row in rows])),
        "learned_m0_rate": float(np.mean([row["learned_m0"] for row in rows])),
        "learned_r0_rate": float(np.mean([row["learned_r0"] for row in rows])),
        "nonfallback_m0_r0_rate": float(
            np.mean([row["nonfallback_m0_r0"] for row in rows])
        ),
        "learned_m0_share": learned_m0_share,
        "learned_r0_share": learned_r0_share,
        "learned_decision_rate": float(np.mean([row["learned"] for row in rows])),
        "fallback_rate": float(np.mean([row["fallback"] for row in rows])),
        "severe_ood_fallback_rate": float(
            np.mean([row["severe_ood_fallback"] for row in rows])
        ),
        "no_feasible_fallback_rate": float(
            np.mean([row["no_feasible_fallback"] for row in rows])
        ),
        "ood_fallback_rate": float(np.mean([row["ood_fallback"] for row in rows])),
        "action_entropy_bits": entropy,
        "learned_action_entropy_bits": learned_entropy,
        "learned_maximum_action_share": learned_maximum_share,
        "action_distribution": counts,
        "learned_action_distribution": learned_counts,
        "rows": rows,
    }


def tune_selection_config(
    model: PMV2Model,
    states: Sequence[PMV2State],
    labels: Sequence[ActionLabel],
    *,
    cost_weights: Iterable[float] = (0.05, 0.10, 0.20, 0.30),
    risk_weights: Iterable[float] = (0.15, 0.25, 0.40),
    resource_gains: Iterable[float] = (-0.03, -0.01, 0.0, 0.02),
    strategy_gains: Iterable[float] = (-0.02, 0.0, 0.02),
    max_risks: Iterable[float] = (0.30, 0.40, 0.50),
    minimum_quality: float | None = None,
    objective_risk_weight: float = 0.25,
    objective_cost_weight: float = 0.10,
    objective_version: str = "pmv2-calibration-utility-v1",
) -> dict[str, Any]:
    """Tune only on a frozen calibration split.

    Candidate weights change which action is selected, but all candidates are
    scored with one frozen, candidate-independent quality-risk-cost ruler.
    Otherwise a candidate could improve its reported objective merely by
    shrinking its own penalties.  The objective uses the same within-state
    normalized estimated resource cost as online selection; observed input
    tokens remain a diagnostic only.
    """

    if objective_risk_weight < 0.0 or objective_cost_weight < 0.0:
        raise ValueError("calibration objective weights must be non-negative")
    if not objective_version:
        raise ValueError("calibration objective_version must be non-empty")
    original = model.selection_config
    candidates: list[dict[str, Any]] = []
    for cost_weight in cost_weights:
        for risk_weight in risk_weights:
            for resource_gain in resource_gains:
                for strategy_gain in strategy_gains:
                    for max_risk in max_risks:
                        config = original.model_copy(
                            update={
                                "cost_weight": float(cost_weight),
                                "risk_weight": float(risk_weight),
                                "resource_min_gain": float(resource_gain),
                                "strategy_min_gain": float(strategy_gain),
                                "max_risk_ucb": float(max_risk),
                            }
                        )
                        model.selection_config = config
                        metrics = evaluate_policy(model, states, labels)
                        objective = (
                            metrics["mean_quality"]
                            - float(objective_risk_weight) * metrics["mean_risk"]
                            - float(objective_cost_weight)
                            * metrics["mean_normalized_estimated_resource_cost"]
                        )
                        quality_ok = (
                            minimum_quality is None
                            or metrics["mean_quality"] >= minimum_quality
                        )
                        candidates.append(
                            {
                                "config": config.model_dump(mode="json"),
                                "config_hash": config.digest(),
                                "objective": float(objective),
                                "quality_constraint_passed": quality_ok,
                                "metrics": {key: value for key, value in metrics.items() if key != "rows"},
                            }
                        )
    valid = [row for row in candidates if row["quality_constraint_passed"]]
    if not valid:
        model.selection_config = original
        raise RuntimeError("no PM-v2 selection config satisfies calibration constraints")
    best = max(
        valid,
        key=lambda row: (
            row["objective"],
            row["metrics"]["mean_quality"],
            -row["metrics"]["mean_risk"],
            -row["metrics"]["mean_normalized_estimated_resource_cost"],
        ),
    )
    model.selection_config = SelectionConfig.model_validate(best["config"])
    return {
        "status": "COMPLETE",
        "objective_spec": {
            "version": objective_version,
            "risk_weight": float(objective_risk_weight),
            "cost_weight": float(objective_cost_weight),
            "cost_basis": "within_state_normalized_estimated_resource_cost",
            "observed_input_tokens_used": False,
        },
        "selected": best,
        "candidate_count": len(candidates),
        "pareto_candidates": sorted(valid, key=lambda row: row["objective"], reverse=True)[:25],
    }


@dataclass
class LearnedPMV2Policy:
    """Compatibility adapter for the existing EvoEmo generation runner."""

    model: PMV2Model
    strategy_estimated_tokens: int
    name: str = "pm_v2"
    strategy_catalog_count: int = 0
    last_decision_report: dict[str, Any] | None = None

    def __post_init__(self):
        if self.strategy_estimated_tokens <= 0:
            raise ValueError("strategy_estimated_tokens must be explicitly positive")

    def choose(self, state):
        from .pm_v2_data import runtime_to_pmv2_state

        pm_state = runtime_to_pmv2_state(
            state,
            strategy_catalog_count=self.strategy_catalog_count,
            strategy_estimated_tokens=self.strategy_estimated_tokens,
        )
        decision = self.model.choose(pm_state)
        self.last_decision_report = decision.model_dump(mode="json")
        return decision.chosen_action
