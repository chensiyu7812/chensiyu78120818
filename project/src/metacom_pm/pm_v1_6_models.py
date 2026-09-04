from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import RobustScaler

from .contracts import MemorySource, StrategyMode, parse_action_id
from .io import canonical_json, sha256_text
from .pm_v1_6_contracts import AlgorithmCandidate, STRATEGY_FAMILIES, Step0Observation
from .pm_v1_6_router import StrongTransparentRouter
from .pm_v2_contracts import ActionLabel, CompositeSpec, PMV2State
from .pm_v2_features import state_text
from .pm_v2_model import RESPONSE_FIELDS, RISK_FIELDS, applicable_risk_fields

MODEL_FORMAT = "pm-v1.6-outcome-model-v1"
CALL_PENALTY = 24.0
STRATEGY_EXPECTED_TOKENS = 240.0


class OutcomeMode(str, Enum):
    ABSOLUTE = AlgorithmCandidate.ABSOLUTE_HGB.value
    STATE_CENTERED = AlgorithmCandidate.STATE_CENTERED_HGB.value
    RULE_RELATIVE = AlgorithmCandidate.RULE_RELATIVE_HGB.value


class OverrideConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    protocol: str = "pm-v1.6-conservative-residual-override-v1"
    uncertainty_z: float = Field(default=1.0, ge=0.0, le=3.0)
    quality_noninferiority_margin: float = Field(default=0.02, ge=0.0, le=1.0)
    emotional_support_noninferiority_margin: float = Field(default=0.02, ge=0.0, le=1.0)
    absolute_risk_ceiling: float = Field(default=0.45, ge=0.0, le=1.0)
    relative_risk_margin: float = Field(default=0.0, ge=-1.0, le=1.0)
    risk_weight: float = Field(default=0.25, ge=0.0)
    cost_weight: float = Field(default=0.10, ge=0.0)
    minimum_utility_lcb: float = 0.0

    @model_validator(mode="after")
    def meaningful(self) -> "OverrideConfig":
        if self.risk_weight == 0.0 and self.cost_weight == 0.0:
            raise ValueError("override must penalize risk or cost")
        return self

    def digest(self) -> str:
        return sha256_text(canonical_json(self.model_dump(mode="json")))


@dataclass
class GroupBootstrapHGB:
    n_models: int = 7
    seed: int = 1701
    max_iter: int = 180
    max_leaf_nodes: int = 15
    learning_rate: float = 0.05
    l2_regularization: float = 1.0
    models: list[HistGradientBoostingRegressor] = field(default_factory=list)

    def fit(self, x: np.ndarray, y: np.ndarray, groups: Sequence[str], sample_weight: np.ndarray | None = None) -> "GroupBootstrapHGB":
        if x.shape[0] != len(y) or len(y) != len(groups):
            raise ValueError("x, y, and groups must be aligned")
        unique = np.asarray(sorted(set(str(group) for group in groups)), dtype=object)
        if len(unique) < 3:
            raise ValueError("group bootstrap requires at least three user groups")
        group_array = np.asarray([str(group) for group in groups], dtype=object)
        rng = np.random.default_rng(self.seed)
        self.models = []
        for index in range(self.n_models):
            sampled = rng.choice(unique, size=len(unique), replace=True)
            multiplicity: dict[str, int] = {}
            for group in sampled:
                multiplicity[str(group)] = multiplicity.get(str(group), 0) + 1
            weights = np.asarray([float(multiplicity.get(str(group), 0)) for group in group_array], dtype=float)
            if sample_weight is not None:
                weights *= np.asarray(sample_weight, dtype=float)
            keep = weights > 0
            estimator = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                max_leaf_nodes=self.max_leaf_nodes,
                l2_regularization=self.l2_regularization,
                random_state=self.seed + index,
            )
            estimator.fit(x[keep], y[keep], sample_weight=weights[keep])
            self.models.append(estimator)
        return self

    def predict_members(self, x: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("bootstrap HGB is not fitted")
        return np.vstack([model.predict(x) for model in self.models])


@dataclass
class PMV16FeatureBuilder:
    word_features: int = 256
    char_features: int = 256
    include_step0: bool = True
    metadata_scaler: RobustScaler = field(default_factory=RobustScaler)
    fitted: bool = False

    def __post_init__(self) -> None:
        self.word = HashingVectorizer(n_features=self.word_features, alternate_sign=False, norm="l2", lowercase=True, ngram_range=(1, 2), analyzer="word")
        self.char = HashingVectorizer(n_features=self.char_features, alternate_sign=False, norm="l2", lowercase=True, ngram_range=(3, 5), analyzer="char_wb")

    @staticmethod
    def _bounded(value: float, upper: float) -> float:
        clipped = min(max(float(value), 0.0), float(upper))
        return float(np.log1p(clipped) / np.log1p(float(upper)))

    def _text_vector(self, state: PMV2State) -> np.ndarray:
        text = state_text(state)
        word = self.word.transform([text]).toarray()[0].astype(np.float64)
        char = self.char.transform([text]).toarray()[0].astype(np.float64)
        return np.concatenate([word, char])

    def _metadata_raw(self, state: PMV2State, step0: Step0Observation) -> np.ndarray:
        values: list[float] = [self._bounded(state.session_index, 64)]
        for source in MemorySource:
            row = step0.sources[source]
            values.extend([
                float(row.available), row.bounded_count / 3.0,
                float(row.min_age_sessions or 0) / max(state.session_index, 1),
                float(row.median_age_sessions or 0) / max(state.session_index, 1),
                float(row.max_age_sessions or 0) / max(state.session_index, 1),
                self._bounded(row.estimated_retrievable_tokens, 768),
                row.query_to_source_similarity if self.include_step0 and row.representation_valid else 0.0,
                float(row.representation_valid and self.include_step0),
            ])
        family_map = {row.family: row for row in step0.strategy_families}
        for family in STRATEGY_FAMILIES:
            row = family_map[family]
            values.extend([
                row.query_to_family_similarity if self.include_step0 and row.representation_valid else 0.0,
                float(row.representation_valid and self.include_step0),
            ])
        readiness = step0.readiness
        values.extend([
            float(readiness.advice_requested), float(readiness.advice_rejected),
            float(readiness.listening_requested), float(readiness.clarification_needed),
            readiness.action_readiness,
        ] if self.include_step0 else [0.0] * 5)
        return np.asarray(values, dtype=np.float64)

    def _action_raw(self, action_id: str, step0: Step0Observation) -> np.ndarray:
        sources, strategy = parse_action_id(action_id)
        bits = np.asarray([
            float(MemorySource.MP in sources), float(MemorySource.MS in sources),
            float(MemorySource.ME in sources), float(strategy is StrategyMode.RS),
        ], dtype=np.float64)
        pairwise = np.asarray([bits[i] * bits[j] for i in range(4) for j in range(i + 1, 4)], dtype=np.float64)
        triples = np.asarray([
            bits[0] * bits[1] * bits[2], bits[0] * bits[1] * bits[3],
            bits[0] * bits[2] * bits[3], bits[1] * bits[2] * bits[3],
        ], dtype=np.float64)
        source_tokens = sum(step0.sources[source].estimated_retrievable_tokens for source in sources)
        calls = len(sources) + int(strategy is StrategyMode.RS)
        estimated_cost = source_tokens + (STRATEGY_EXPECTED_TOKENS if strategy is StrategyMode.RS else 0.0) + CALL_PENALTY * calls
        return np.concatenate([
            bits,
            np.asarray([float(len(sources)), float(len(sources) ** 2), self._bounded(estimated_cost, 2048), float(calls)]),
            pairwise, triples, np.asarray([float(np.prod(bits))]),
        ])

    def fit(self, states: Sequence[PMV2State], step0_by_state: Mapping[str, Step0Observation]) -> "PMV16FeatureBuilder":
        if not states:
            raise ValueError("cannot fit feature builder without states")
        metadata = np.vstack([self._metadata_raw(state, step0_by_state[state.state_id]) for state in states])
        self.metadata_scaler.fit(metadata)
        self.fitted = True
        return self

    def state_vector(self, state: PMV2State, step0: Step0Observation) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("feature builder is not fitted")
        text = self._text_vector(state)
        metadata = self.metadata_scaler.transform([self._metadata_raw(state, step0)])[0]
        return np.concatenate([text, metadata]).astype(np.float32)

    def state_action_vector(self, state: PMV2State, step0: Step0Observation, action_id: str) -> np.ndarray:
        state_vec = self.state_vector(state, step0).astype(np.float64)
        action = self._action_raw(action_id, step0)
        interactions = np.concatenate([state_vec * bit for bit in action[:4]])
        return np.concatenate([state_vec, action, interactions]).astype(np.float32)

    def relative_vector(self, state: PMV2State, step0: Step0Observation, candidate_action: str, baseline_action: str) -> np.ndarray:
        state_vec = self.state_vector(state, step0).astype(np.float64)
        candidate = self._action_raw(candidate_action, step0)
        baseline = self._action_raw(baseline_action, step0)
        return np.concatenate([state_vec, candidate, baseline, candidate - baseline, candidate * baseline]).astype(np.float32)

    def estimate_action_cost(self, action_id: str, step0: Step0Observation) -> float:
        sources, strategy = parse_action_id(action_id)
        tokens = sum(step0.sources[source].estimated_retrievable_tokens for source in sources)
        if strategy is StrategyMode.RS:
            tokens += STRATEGY_EXPECTED_TOKENS
        calls = len(sources) + int(strategy is StrategyMode.RS)
        return float(tokens + CALL_PENALTY * calls)

    def config_hash(self) -> str:
        return sha256_text(canonical_json({
            "format": MODEL_FORMAT, "word_features": self.word_features,
            "char_features": self.char_features, "include_step0": self.include_step0,
            "strategy_family_order": list(STRATEGY_FAMILIES),
            "action_factors": ["MP", "MS", "ME", "RS"],
            "action_interactions": "all_orders_2_power_4",
            "forbidden": ["raw text", "item ids", "item-level scores", "catalog embedding statistics"],
        }))


def _normalized_head_value(label: ActionLabel, head: str) -> float:
    if head in RESPONSE_FIELDS:
        return (float(getattr(label.response, head)) - 1.0) / 4.0
    return float(getattr(label.risk, head)) / 3.0


def _label_weight(label: ActionLabel, head: str, scale: float) -> float:
    prefix = "response" if head in RESPONSE_FIELDS else "risk"
    mad = float(label.dimension_mad[f"{prefix}.{head}"])
    agreement = 1.0 / (1.0 + (mad / scale) ** 2)
    alias = float(label.provenance.get("shared_quality_label_weight", 1.0))
    if not 0.0 < alias <= 1.0:
        raise ValueError("shared quality label weight must be in (0,1]")
    return agreement * alias


@dataclass
class PMV16OutcomeModel:
    mode: OutcomeMode
    feature_builder: PMV16FeatureBuilder
    composite_spec: CompositeSpec
    primary_heads: dict[str, GroupBootstrapHGB] = field(default_factory=dict)
    state_heads: dict[str, GroupBootstrapHGB] = field(default_factory=dict)
    effect_heads: dict[str, GroupBootstrapHGB] = field(default_factory=dict)
    rule_action_by_state: dict[str, str] = field(default_factory=dict)
    format_version: str = MODEL_FORMAT
    training_report: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def train(cls, *, mode: OutcomeMode, states: Sequence[PMV2State], labels: Sequence[ActionLabel], step0_by_state: Mapping[str, Step0Observation], composite_spec: CompositeSpec | None = None, rule_action_by_state: Mapping[str, str] | None = None, n_models: int = 7, seed: int = 1701, dimension_mad_scale: float = 0.75, include_step0: bool = True) -> "PMV16OutcomeModel":
        state_map = {state.state_id: state for state in states}
        if len(state_map) != len(states) or not state_map:
            raise ValueError("states must be unique and non-empty")
        if set(step0_by_state) != set(state_map):
            raise ValueError("Step-0 map must exactly match training states")
        by_state: dict[str, list[ActionLabel]] = {state_id: [] for state_id in state_map}
        for label in labels:
            if label.state_id not in by_state:
                raise ValueError("label references unknown state")
            by_state[label.state_id].append(label)
        for state_id, state in state_map.items():
            if {label.action_id for label in by_state[state_id]} != set(state.allowed_actions):
                raise ValueError(f"incomplete action matrix for state {state_id}")
        rules = {str(key): str(value) for key, value in (rule_action_by_state or {}).items()}
        if mode is OutcomeMode.RULE_RELATIVE:
            if set(rules) != set(state_map):
                raise ValueError("rule-relative training requires one frozen rule action per state")
            for state_id, action_id in rules.items():
                if action_id not in state_map[state_id].allowed_actions:
                    raise ValueError(f"illegal rule action for {state_id}: {action_id}")
        builder = PMV16FeatureBuilder(include_step0=include_step0).fit(list(states), step0_by_state)
        spec = composite_spec or CompositeSpec()
        model = cls(mode=mode, feature_builder=builder, composite_spec=spec, rule_action_by_state=rules)
        all_heads = [*RESPONSE_FIELDS, *RISK_FIELDS]
        unique_states = [state_map[key] for key in sorted(state_map)]
        state_groups = [state.user_id for state in unique_states]
        for head_index, head in enumerate(all_heads):
            rows = [label for state_id in sorted(by_state) for label in by_state[state_id]]
            state_counts = {state_id: len(by_state[state_id]) for state_id in by_state}
            row_weights = np.asarray([_label_weight(label, head, dimension_mad_scale) / state_counts[label.state_id] for label in rows], dtype=float)
            y = np.asarray([_normalized_head_value(label, head) for label in rows])
            groups = [state_map[label.state_id].user_id for label in rows]
            head_seed = seed + head_index * 17
            if mode is OutcomeMode.ABSOLUTE:
                x = np.vstack([builder.state_action_vector(state_map[label.state_id], step0_by_state[label.state_id], label.action_id) for label in rows])
                model.primary_heads[head] = GroupBootstrapHGB(n_models=n_models, seed=head_seed).fit(x, y, groups, row_weights)
                continue
            if mode is OutcomeMode.STATE_CENTERED:
                means: dict[str, float] = {}
                for state_id, state_labels in by_state.items():
                    values = np.asarray([_normalized_head_value(label, head) for label in state_labels])
                    weights = np.asarray([_label_weight(label, head, dimension_mad_scale) for label in state_labels])
                    means[state_id] = float(np.average(values, weights=weights))
                state_x = np.vstack([builder.state_vector(state, step0_by_state[state.state_id]) for state in unique_states])
                state_y = np.asarray([means[state.state_id] for state in unique_states])
                model.state_heads[head] = GroupBootstrapHGB(n_models=n_models, seed=head_seed).fit(state_x, state_y, state_groups)
                effect_x = np.vstack([builder.state_action_vector(state_map[label.state_id], step0_by_state[label.state_id], label.action_id) for label in rows])
                effect_y = np.asarray([_normalized_head_value(label, head) - means[label.state_id] for label in rows])
                model.effect_heads[head] = GroupBootstrapHGB(n_models=n_models, seed=head_seed).fit(effect_x, effect_y, groups, row_weights)
                continue
            baseline_labels: dict[str, ActionLabel] = {}
            for state_id, state_labels in by_state.items():
                baseline_action = rules[state_id]
                baseline_labels[state_id] = next(label for label in state_labels if label.action_id == baseline_action)
            base_x = np.vstack([builder.state_action_vector(state, step0_by_state[state.state_id], rules[state.state_id]) for state in unique_states])
            base_y = np.asarray([_normalized_head_value(baseline_labels[state.state_id], head) for state in unique_states])
            model.state_heads[head] = GroupBootstrapHGB(n_models=n_models, seed=head_seed).fit(base_x, base_y, state_groups)
            delta_x = np.vstack([builder.relative_vector(state_map[label.state_id], step0_by_state[label.state_id], label.action_id, rules[label.state_id]) for label in rows])
            delta_y = np.asarray([_normalized_head_value(label, head) - _normalized_head_value(baseline_labels[label.state_id], head) for label in rows])
            model.effect_heads[head] = GroupBootstrapHGB(n_models=n_models, seed=head_seed).fit(delta_x, delta_y, groups, row_weights)
        model.training_report = {
            "format_version": MODEL_FORMAT, "mode": mode.value,
            "n_states": len(states), "n_labels": len(labels),
            "n_users": len({state.user_id for state in states}),
            "feature_config_sha256": builder.config_hash(),
            "include_step0": include_step0,
            "alias_weighting": "1/prompt_equivalence_class_size_when_provided",
            "absolute_risk_heads_retained": True,
            "rule_actions_train_only_required": mode is OutcomeMode.RULE_RELATIVE,
        }
        return model

    def _head_members(self, *, state: PMV2State, step0: Step0Observation, action_id: str, head: str, rule_action: str | None) -> np.ndarray:
        if self.mode is OutcomeMode.ABSOLUTE:
            x = self.feature_builder.state_action_vector(state, step0, action_id)[None, :]
            return self.primary_heads[head].predict_members(x)[:, 0]
        if self.mode is OutcomeMode.STATE_CENTERED:
            state_x = self.feature_builder.state_vector(state, step0)[None, :]
            action_x = self.feature_builder.state_action_vector(state, step0, action_id)[None, :]
            return self.state_heads[head].predict_members(state_x)[:, 0] + self.effect_heads[head].predict_members(action_x)[:, 0]
        if rule_action is None:
            raise ValueError("rule-relative prediction requires the current rule action")
        base_x = self.feature_builder.state_action_vector(state, step0, rule_action)[None, :]
        baseline = self.state_heads[head].predict_members(base_x)[:, 0]
        if action_id == rule_action:
            return baseline
        delta_x = self.feature_builder.relative_vector(state, step0, action_id, rule_action)[None, :]
        return baseline + self.effect_heads[head].predict_members(delta_x)[:, 0]

    def predict_members(self, *, state: PMV2State, step0: Step0Observation, action_id: str, rule_action: str | None = None) -> dict[str, np.ndarray]:
        if action_id not in state.allowed_actions:
            raise ValueError(f"illegal action {action_id} for state {state.state_id}")
        return {head: np.clip(self._head_members(state=state, step0=step0, action_id=action_id, head=head, rule_action=rule_action), 0.0, 1.0) for head in [*RESPONSE_FIELDS, *RISK_FIELDS]}

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "PMV16OutcomeModel":
        value = joblib.load(path)
        if not isinstance(value, PMV16OutcomeModel):
            raise TypeError("checkpoint is not a PMV16OutcomeModel")
        if value.format_version != MODEL_FORMAT:
            raise RuntimeError("unsupported PM-v1.6 checkpoint format")
        return value


@dataclass(frozen=True)
class PMV16Decision:
    chosen_action: str
    rule_action: str
    override_used: bool
    eligible_actions: tuple[str, ...]
    rejected_actions: Mapping[str, tuple[str, ...]]
    predicted_delta_utility_lcb: Mapping[str, float]
    model_mode: str
    override_config_sha256: str


@dataclass
class ConservativeResidualPolicy:
    outcome_model: PMV16OutcomeModel
    rule_router: StrongTransparentRouter
    override_config: OverrideConfig

    def choose(self, state: PMV2State, step0: Step0Observation) -> PMV16Decision:
        rule = self.rule_router.choose(step0).action_id
        if rule not in state.allowed_actions:
            raise RuntimeError(f"transparent rule produced illegal action: {rule}")
        rule_members = self.outcome_model.predict_members(state=state, step0=step0, action_id=rule, rule_action=rule)
        weights = self.outcome_model.composite_spec.weights
        rule_quality = sum(float(weights[head]) * rule_members[head] for head in RESPONSE_FIELDS)
        rule_support = rule_members["emotional_support"]
        rule_risk = np.maximum.reduce([rule_members[head] for head in applicable_risk_fields(rule)])
        rule_cost = self.outcome_model.feature_builder.estimate_action_cost(rule, step0)
        max_cost = max(self.outcome_model.feature_builder.estimate_action_cost(action, step0) for action in state.allowed_actions)
        z = self.override_config.uncertainty_z
        eligible = [rule]
        rejected: dict[str, tuple[str, ...]] = {}
        delta_utility_lcb: dict[str, float] = {rule: 0.0}
        best_action, best_delta = rule, 0.0
        for action in state.allowed_actions:
            if action == rule:
                continue
            members = self.outcome_model.predict_members(state=state, step0=step0, action_id=action, rule_action=rule)
            quality = sum(float(weights[head]) * members[head] for head in RESPONSE_FIELDS)
            support = members["emotional_support"]
            risk = np.maximum.reduce([members[head] for head in applicable_risk_fields(action)])
            quality_delta, support_delta, risk_delta = quality - rule_quality, support - rule_support, risk - rule_risk
            action_cost = self.outcome_model.feature_builder.estimate_action_cost(action, step0)
            cost_delta = (action_cost - rule_cost) / max(max_cost, 1.0)
            utility_delta = quality_delta - self.override_config.risk_weight * risk_delta - self.override_config.cost_weight * cost_delta
            q_lcb = float(np.mean(quality_delta) - z * np.std(quality_delta))
            s_lcb = float(np.mean(support_delta) - z * np.std(support_delta))
            absolute_risk_ucb = float(np.mean(risk) + z * np.std(risk))
            risk_delta_ucb = float(np.mean(risk_delta) + z * np.std(risk_delta))
            utility_lcb = float(np.mean(utility_delta) - z * np.std(utility_delta))
            delta_utility_lcb[action] = utility_lcb
            reasons = []
            if absolute_risk_ucb > self.override_config.absolute_risk_ceiling:
                reasons.append("absolute_risk_ceiling")
            if q_lcb < -self.override_config.quality_noninferiority_margin:
                reasons.append("quality_noninferiority")
            if s_lcb < -self.override_config.emotional_support_noninferiority_margin:
                reasons.append("emotional_support_noninferiority")
            if risk_delta_ucb > self.override_config.relative_risk_margin:
                reasons.append("relative_risk_margin")
            if utility_lcb <= self.override_config.minimum_utility_lcb:
                reasons.append("utility_advantage")
            if reasons:
                rejected[action] = tuple(reasons)
                continue
            eligible.append(action)
            if utility_lcb > best_delta or (utility_lcb == best_delta and action_cost < self.outcome_model.feature_builder.estimate_action_cost(best_action, step0)):
                best_action, best_delta = action, utility_lcb
        return PMV16Decision(
            chosen_action=best_action,
            rule_action=rule,
            override_used=best_action != rule,
            eligible_actions=tuple(sorted(eligible)),
            rejected_actions=rejected,
            predicted_delta_utility_lcb=delta_utility_lcb,
            model_mode=self.outcome_model.mode.value,
            override_config_sha256=self.override_config.digest(),
        )
