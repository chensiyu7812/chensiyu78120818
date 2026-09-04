from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import RobustScaler

from .contracts import MemorySource, StrategyMode, parse_action_id
from .io import canonical_json, sha256_text
from .pm_v2_contracts import (
    ADVICE_READINESS_IDS,
    PMV2State,
    STRATEGY_FAMILY_IDS,
    Step0Observation,
)
from .pm_v1_5_step0 import build_step0_observation
from .pm_v1_5_semantic import visible_dialogue_state_text
from .retrieval import DEFAULT_MEMORY_TOP_K


SOURCE_ORDER = (MemorySource.MP, MemorySource.MS, MemorySource.ME)

# These are the actual per-source retrieval capacities used by MemoryRetriever.
# Inventory metadata is expressed in terms of what one action can retrieve, rather
# than the size of the backing store.  This prevents a large deployed catalog from
# becoming a domain/identity shortcut when development catalogs contain only a few
# synthetic items.
MEMORY_TOP_K = dict(DEFAULT_MEMORY_TOP_K)
RETRIEVAL_CALL_PENALTY = 24.0
SESSION_INDEX_FEATURE_CLIP = 64
MEMORY_ITEM_FEATURE_TOKEN_CLIP = 256
STRATEGY_FEATURE_TOKEN_CLIP = 512
ACTION_FEATURE_TOKEN_CLIP = (
    sum(MEMORY_TOP_K.values()) * MEMORY_ITEM_FEATURE_TOKEN_CLIP
    + STRATEGY_FEATURE_TOKEN_CLIP
)


def state_text(state: PMV2State) -> str:
    return visible_dialogue_state_text(
        current_user_text=state.current_user_text,
        current_session_history=state.current_session_history,
        current_session_summary=state.current_session_summary,
    )


@dataclass
class PMV2FeatureBuilder:
    """OOV-robust, strictly pre-retrieval state-action features.

    The builder uses the current/recent text plus the formal Step-0 scalars. It never
    reads actual memory snippets, retrieved IDs, item-level top scores, or a
    current-state conflict oracle. Query relevance is limited to similarity against a
    cached source-level catalog representation supplied by the runtime state. Raw
    catalog vectors and their norm/mean/std are deliberately outside this class.
    """

    word_features: int = 256
    char_features: int = 256
    use_precomputed_embeddings: bool = True
    require_precomputed_embeddings: bool = False
    semantic_projection_dimensions: int = 48
    step0_signal_mode: Literal["full", "none"] = "full"
    metadata_scaler: RobustScaler = field(default_factory=RobustScaler)
    embedding_dim: int = 0
    raw_embedding_dim: int = 0
    semantic_projector: PCA | None = None
    semantic_encoder_spec_sha256: str | None = None
    train_text_centroid: np.ndarray | None = None
    train_text_radius_p95: float = 0.0
    metadata_reference_low: np.ndarray | None = None
    metadata_reference_high: np.ndarray | None = None
    semantic_ood_threshold: float | None = None
    metadata_ood_threshold: float | None = None
    ood_calibration_report: dict[str, Any] = field(default_factory=dict)
    fitted: bool = False

    def __post_init__(self):
        if self.step0_signal_mode not in {"full", "none"}:
            raise ValueError("step0_signal_mode must be 'full' or 'none'")
        self._word = HashingVectorizer(
            n_features=self.word_features,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(1, 2),
            analyzer="word",
        )
        self._char = HashingVectorizer(
            n_features=self.char_features,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(3, 5),
            analyzer="char_wb",
        )

    @property
    def state_text_dim(self) -> int:
        return self.word_features + self.char_features + self.embedding_dim

    @staticmethod
    def _bounded_log_fraction(value: float, upper: float) -> float:
        """Map a deployable non-negative quantity to a stable [0, 1] feature."""

        if upper <= 0:
            raise ValueError("feature upper bound must be positive")
        clipped = min(max(0.0, float(value)), float(upper))
        return float(np.log1p(clipped) / np.log1p(float(upper)))

    @staticmethod
    def _age_ratio(age: float | int | None, session_index: int) -> float:
        if age is None:
            return 0.0
        return float(min(max(float(age) / max(float(session_index), 1.0), 0.0), 1.0))

    def _expected_source_retrieval_tokens(
        self, state: PMV2State, source: MemorySource
    ) -> float:
        """Estimate tokens visible after one bounded source retrieval.

        The estimate intentionally depends on catalog-average item size and the
        retriever's top-k capacity.  Catalog tail cardinality/tokens cannot increase
        this value after the retrieval capacity is full.
        """

        if self.step0_signal_mode == "none":
            return float(MEMORY_TOP_K[source] * MEMORY_ITEM_FEATURE_TOKEN_CLIP)
        if state.step0_observation is not None:
            return float(
                state.step0_observation.memory_sources[source].expected_retrieval_tokens
            )
        cat = state.inventory[source]
        if not cat.available or cat.count <= 0:
            return 0.0
        retrieved_count = min(cat.count, MEMORY_TOP_K[source])
        mean_item_tokens = cat.estimated_tokens / cat.count
        return float(min(cat.estimated_tokens, mean_item_tokens * retrieved_count))

    @staticmethod
    def _step0(state: PMV2State) -> Step0Observation:
        if state.step0_observation is not None:
            return state.step0_observation
        return build_step0_observation(
            query_text=state_text(state),
            inventory=state.inventory,
            strategy_catalog_count=state.strategy_catalog_count,
            strategy_estimated_tokens=state.strategy_estimated_tokens,
        )

    def _semantic_vector(self, state: PMV2State) -> np.ndarray:
        if not self.use_precomputed_embeddings or not self.raw_embedding_dim:
            return np.empty(0, dtype=np.float64)
        if len(state.text_embedding) != self.raw_embedding_dim:
            raise ValueError(
                f"state {state.card_id} has embedding dim {len(state.text_embedding)}; "
                f"expected {self.raw_embedding_dim}"
            )
        if self.semantic_encoder_spec_sha256 is not None:
            semantic = state.provenance.get("semantic_observation") or {}
            if semantic.get("encoder_spec_sha256") != self.semantic_encoder_spec_sha256:
                raise RuntimeError(
                    f"state {state.card_id} semantic encoder binding does not match "
                    "the fitted PM feature contract"
                )
        embedding = np.asarray(state.text_embedding, dtype=np.float64)
        norm = np.linalg.norm(embedding)
        embedding = embedding / norm if norm > 0 else embedding
        if self.semantic_projector is not None:
            embedding = self.semantic_projector.transform([embedding])[0]
            projected_norm = np.linalg.norm(embedding)
            embedding = embedding / projected_norm if projected_norm > 0 else embedding
        return embedding

    def _state_text_vector(self, state: PMV2State) -> np.ndarray:
        text = state_text(state)
        word = self._word.transform([text]).toarray()[0].astype(np.float64)
        char = self._char.transform([text]).toarray()[0].astype(np.float64)
        blocks = [word, char]
        semantic = self._semantic_vector(state)
        if semantic.size:
            blocks.append(semantic)
        return np.concatenate(blocks)

    def _metadata_raw(self, state: PMV2State) -> np.ndarray:
        step0 = self._step0(state)
        values: list[float] = [
            self._bounded_log_fraction(
                state.session_index, SESSION_INDEX_FEATURE_CLIP
            )
        ]
        if self.step0_signal_mode == "none":
            # Preserve dimensionality so algorithm comparisons change only the
            # observation, not downstream head capacity.
            return np.asarray(
                [
                    *values,
                    *(
                        [0.0]
                        * (
                            len(SOURCE_ORDER) * 9
                            + 2
                            + len(STRATEGY_FAMILY_IDS)
                            + len(ADVICE_READINESS_IDS)
                            + 1
                        )
                    ),
                ],
                dtype=np.float64,
            )
        for source in SOURCE_ORDER:
            cat = step0.memory_sources[source]
            top_k = MEMORY_TOP_K[source]
            expected_tokens = float(cat.expected_retrieval_tokens)
            values.extend(
                [
                    float(cat.available),
                    float(min(cat.count, top_k) / top_k),
                    self._age_ratio(cat.min_age_sessions, state.session_index),
                    self._age_ratio(cat.median_age_sessions, state.session_index),
                    self._age_ratio(cat.max_age_sessions, state.session_index),
                    self._bounded_log_fraction(
                        expected_tokens,
                        top_k * MEMORY_ITEM_FEATURE_TOKEN_CLIP,
                    ),
                    # Source-centroid similarity only. Max/P90 item-level scores are
                    # intentionally excluded because they approximate retrieval.
                    float(cat.representation_valid),
                    float(cat.query_to_source_similarity),
                    self._age_ratio(
                        max(
                            0.0,
                            float(cat.max_age_sessions or 0)
                            - float(cat.min_age_sessions or 0),
                        ),
                        state.session_index,
                    ),
                ]
            )
        strategy = step0.strategy
        values.extend(
            [
                # The total bank cardinality is constant deployment metadata, not
                # a state signal.  Keeping it in the model created a catastrophic
                # development/external OOD shortcut when the synthetic generator
                # invented a small catalog count.
                self._bounded_log_fraction(
                    strategy.expected_retrieval_tokens,
                    STRATEGY_FEATURE_TOKEN_CLIP,
                ),
                float(strategy.representation_valid),
                *[
                    float(strategy.family_similarities[family])
                    for family in STRATEGY_FAMILY_IDS
                ],
                *[
                    float(strategy.advice_readiness_similarities[readiness])
                    for readiness in ADVICE_READINESS_IDS
                ],
                float(strategy.question_present),
            ]
        )
        return np.asarray(values, dtype=np.float64)

    def _action_raw(self, state: PMV2State, action_id: str) -> np.ndarray:
        sources, strategy = parse_action_id(action_id)
        source_bits = [float(source in sources) for source in SOURCE_ORDER]
        source_count = float(len(sources))
        memory_tokens = sum(
            self._expected_source_retrieval_tokens(state, source)
            for source in sources
        )
        retrieval_calls = len(sources) + int(strategy is StrategyMode.RS)
        estimated_tokens = memory_tokens + (
            (
                STRATEGY_FEATURE_TOKEN_CLIP
                if self.step0_signal_mode == "none"
                else state.strategy_estimated_tokens
            )
            if strategy is StrategyMode.RS
            else 0
        )
        return np.asarray(
            [
                *source_bits,
                float(strategy is StrategyMode.RS),
                source_count,
                source_count**2,
                self._bounded_log_fraction(
                    estimated_tokens, ACTION_FEATURE_TOKEN_CLIP
                ),
                float(retrieval_calls),
            ],
            dtype=np.float64,
        )

    def estimate_action_cost(self, state: PMV2State, action_id: str) -> float:
        sources, strategy = parse_action_id(action_id)
        estimated_tokens = sum(
            self._expected_source_retrieval_tokens(state, source)
            for source in sources
        )
        if strategy is StrategyMode.RS:
            estimated_tokens += (
                STRATEGY_FEATURE_TOKEN_CLIP
                if self.step0_signal_mode == "none"
                else state.strategy_estimated_tokens
            )
        retrieval_calls = len(sources) + int(strategy is StrategyMode.RS)
        return max(
            0.0,
            estimated_tokens + RETRIEVAL_CALL_PENALTY * retrieval_calls,
        )

    @staticmethod
    def cost_contract() -> dict[str, Any]:
        """Frozen selector-cost constants shared with the actual retriever."""

        return {
            "version": "pmv2-estimated-resource-cost-v1",
            "memory_top_k": {
                source.value: int(MEMORY_TOP_K[source]) for source in SOURCE_ORDER
            },
            "memory_top_k_source": "metacom_pm.retrieval.DEFAULT_MEMORY_TOP_K",
            "strategy_tokens_source": "PMV2State.strategy_estimated_tokens",
            "retrieval_call_penalty": RETRIEVAL_CALL_PENALTY,
            "normalization": "divide_by_max_legal_action_cost_within_state",
        }

    def fit(self, states: Sequence[PMV2State]) -> "PMV2FeatureBuilder":
        if not states:
            raise ValueError("cannot fit PM-v2 features without states")
        populated = [bool(state.text_embedding) for state in states]
        if any(populated) and not all(populated):
            raise ValueError("semantic embeddings must be present on all states or none")
        embedding_dims = {len(state.text_embedding) for state in states if state.text_embedding}
        if len(embedding_dims) > 1:
            raise ValueError(f"inconsistent text embedding dimensions: {sorted(embedding_dims)}")
        self.raw_embedding_dim = (
            next(iter(embedding_dims), 0) if self.use_precomputed_embeddings else 0
        )
        if self.require_precomputed_embeddings and self.raw_embedding_dim == 0:
            raise RuntimeError("reportable PM-v1.5 requires semantic embeddings on every state")
        bindings = {
            str((state.provenance.get("semantic_observation") or {}).get("encoder_spec_sha256"))
            for state in states
            if state.text_embedding
        }
        if self.raw_embedding_dim:
            if len(bindings) != 1 or "None" in bindings or "" in bindings:
                raise RuntimeError("semantic state embeddings lack one exact encoder binding")
            self.semantic_encoder_spec_sha256 = next(iter(bindings))
            raw = np.vstack(
                [np.asarray(state.text_embedding, dtype=np.float64) for state in states]
            )
            raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
            requested = int(self.semantic_projection_dimensions)
            maximum = min(len(states) - 1, self.raw_embedding_dim)
            if self.require_precomputed_embeddings and maximum < requested:
                raise RuntimeError(
                    "reportable semantic projection has fewer train states than its "
                    f"frozen dimension: requested={requested}, maximum={maximum}"
                )
            components = min(requested, maximum)
            if 0 < components < self.raw_embedding_dim:
                self.semantic_projector = PCA(
                    n_components=components,
                    svd_solver="full",
                    whiten=False,
                ).fit(raw)
                self.embedding_dim = components
            else:
                self.semantic_projector = None
                self.embedding_dim = self.raw_embedding_dim
        else:
            self.semantic_encoder_spec_sha256 = None
            self.semantic_projector = None
            self.embedding_dim = 0
        text = np.vstack([self._state_text_vector(state) for state in states])
        metadata = np.vstack([self._metadata_raw(state) for state in states])
        self.metadata_scaler.fit(metadata)
        self.metadata_reference_low = np.quantile(metadata, 0.01, axis=0)
        self.metadata_reference_high = np.quantile(metadata, 0.99, axis=0)
        normalized_text = text / np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-12)
        centroid = normalized_text.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        self.train_text_centroid = centroid
        distances = 1.0 - normalized_text @ centroid
        self.train_text_radius_p95 = float(np.quantile(distances, 0.95))
        self.fitted = True
        return self

    def transform(self, rows: Sequence[tuple[PMV2State, str]]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("PMV2FeatureBuilder must be fitted before transform")
        if not rows:
            raise ValueError("cannot transform an empty row set")
        result: list[np.ndarray] = []
        for state, action_id in rows:
            if action_id not in state.allowed_actions:
                raise ValueError(f"action {action_id} is not legal for state {state.card_id}")
            text = self._state_text_vector(state)
            metadata = self.metadata_scaler.transform([self._metadata_raw(state)])[0]
            action = self._action_raw(state, action_id)
            source_bits = action[:3]
            strategy_bit = action[3]
            interactions = np.concatenate(
                [
                    text * source_bits[0],
                    text * source_bits[1],
                    text * source_bits[2],
                    text * strategy_bit,
                ]
            )
            result.append(np.concatenate([text, metadata, action, interactions]))
        return np.vstack(result).astype(np.float32)

    def _ood_scores(self, state: PMV2State) -> tuple[float, float]:
        if not self.fitted or self.train_text_centroid is None:
            raise RuntimeError("PMV2FeatureBuilder must be fitted before OOD reporting")
        text = self._state_text_vector(state)
        text /= max(float(np.linalg.norm(text)), 1e-12)
        semantic_distance = float(1.0 - text @ self.train_text_centroid)
        metadata = self._metadata_raw(state)
        assert self.metadata_reference_low is not None
        assert self.metadata_reference_high is not None
        lower_span = np.maximum(np.abs(self.metadata_reference_low), 1.0)
        upper_span = np.maximum(np.abs(self.metadata_reference_high), 1.0)
        below = np.maximum(0.0, self.metadata_reference_low - metadata) / lower_span
        above = np.maximum(0.0, metadata - self.metadata_reference_high) / upper_span
        metadata_ood = float(np.mean(np.maximum(below, above)))
        return semantic_distance, metadata_ood

    @staticmethod
    def _semantic_challenge(state: PMV2State) -> PMV2State:
        history = [
            state.current_session_history[0].model_copy(
                update={
                    "role": "user",
                    "content": "Recompile the cryptographic bootloader register map.",
                }
            )
        ]
        return state.model_copy(
            deep=True,
            update={
                "current_user_text": (
                    "Kernel interrupt vector checksum failed during orbital telemetry "
                    f"reconciliation {state.state_id}."
                ),
                "current_session_history": history,
                "current_session_summary": (
                    "Unrelated machine maintenance and satellite packet diagnostics."
                ),
                "text_embedding": [
                    -float(value) for value in state.text_embedding
                ],
            },
        )

    @staticmethod
    def _metadata_challenge(state: PMV2State) -> PMV2State:
        step0 = PMV2FeatureBuilder._step0(state)
        inventory = {}
        memory_observations = {}
        for source in SOURCE_ORDER:
            catalog = state.inventory[source]
            if not catalog.available:
                inventory[source] = catalog.model_copy(deep=True)
                memory_observations[source] = step0.memory_sources[source].model_copy(
                    deep=True
                )
                continue
            challenged_similarity = (
                -1.0
                if float(step0.memory_sources[source].query_to_source_similarity)
                >= 0.0
                else 1.0
            )
            inventory[source] = catalog.model_copy(
                deep=True,
                update={
                    "query_similarity_mean": challenged_similarity,
                    "representation_valid": True,
                },
            )
            memory_observations[source] = step0.memory_sources[source].model_copy(
                deep=True,
                update={
                    "query_to_source_similarity": challenged_similarity,
                    "representation_valid": True,
                },
            )
        strategy = step0.strategy.model_copy(
            deep=True,
            update={
                "family_similarities": {
                    family: (
                        -1.0
                        if float(step0.strategy.family_similarities[family]) >= 0.0
                        else 1.0
                    )
                    for family in STRATEGY_FAMILY_IDS
                },
                "representation_valid": True,
                "advice_readiness_similarities": {
                    readiness: -float(value)
                    for readiness, value in step0.strategy.advice_readiness_similarities.items()
                },
                "question_present": not step0.strategy.question_present,
            },
        )
        challenged_step0 = step0.model_copy(
            deep=True,
            update={"memory_sources": memory_observations, "strategy": strategy},
        )
        return state.model_copy(
            deep=True,
            update={"inventory": inventory, "step0_observation": challenged_step0},
        )

    def calibrate_ood(
        self,
        states: Sequence[PMV2State],
        *,
        semantic_false_positive_quantile: float,
        metadata_false_positive_quantile: float,
        maximum_joint_in_distribution_fallback_rate: float,
        minimum_semantic_challenge_detection_rate: float,
        minimum_metadata_challenge_detection_rate: float,
    ) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("fit the feature builder before OOD calibration")
        if len(states) < 3:
            raise ValueError("OOD calibration requires at least three states")
        for name, value in (
            ("semantic_false_positive_quantile", semantic_false_positive_quantile),
            ("metadata_false_positive_quantile", metadata_false_positive_quantile),
            (
                "maximum_joint_in_distribution_fallback_rate",
                maximum_joint_in_distribution_fallback_rate,
            ),
            (
                "minimum_semantic_challenge_detection_rate",
                minimum_semantic_challenge_detection_rate,
            ),
            (
                "minimum_metadata_challenge_detection_rate",
                minimum_metadata_challenge_detection_rate,
            ),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        in_distribution = [self._ood_scores(state) for state in states]
        semantic_scores = np.asarray([row[0] for row in in_distribution])
        metadata_scores = np.asarray([row[1] for row in in_distribution])
        self.semantic_ood_threshold = float(
            np.quantile(
                semantic_scores,
                float(semantic_false_positive_quantile),
                method="higher",
            )
            + 1e-12
        )
        self.metadata_ood_threshold = float(
            np.quantile(
                metadata_scores,
                float(metadata_false_positive_quantile),
                method="higher",
            )
            + 1e-12
        )
        semantic_challenge_scores = np.asarray(
            [self._ood_scores(self._semantic_challenge(state))[0] for state in states]
        )
        metadata_challenge_scores = np.asarray(
            [self._ood_scores(self._metadata_challenge(state))[1] for state in states]
        )
        joint_id_rate = float(
            np.mean(
                (semantic_scores > self.semantic_ood_threshold)
                | (metadata_scores > self.metadata_ood_threshold)
            )
        )
        semantic_detection = float(
            np.mean(semantic_challenge_scores > self.semantic_ood_threshold)
        )
        metadata_detection = float(
            np.mean(metadata_challenge_scores > self.metadata_ood_threshold)
        )
        checks = {
            "joint_in_distribution_fallback_rate": joint_id_rate
            <= float(maximum_joint_in_distribution_fallback_rate),
            "semantic_challenge_detection_rate": semantic_detection
            >= float(minimum_semantic_challenge_detection_rate),
            "metadata_challenge_detection_rate": metadata_detection
            >= float(minimum_metadata_challenge_detection_rate),
        }
        report = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "protocol": "pm-v2-ood-calibration-v1",
            "calibration_state_count": len(states),
            "semantic_ood_threshold": self.semantic_ood_threshold,
            "metadata_ood_threshold": self.metadata_ood_threshold,
            "joint_in_distribution_fallback_rate": joint_id_rate,
            "semantic_challenge_detection_rate": semantic_detection,
            "metadata_challenge_detection_rate": metadata_detection,
            "checks": checks,
            "threshold_settings": {
                "semantic_false_positive_quantile": float(
                    semantic_false_positive_quantile
                ),
                "metadata_false_positive_quantile": float(
                    metadata_false_positive_quantile
                ),
                "maximum_joint_in_distribution_fallback_rate": float(
                    maximum_joint_in_distribution_fallback_rate
                ),
                "minimum_semantic_challenge_detection_rate": float(
                    minimum_semantic_challenge_detection_rate
                ),
                "minimum_metadata_challenge_detection_rate": float(
                    minimum_metadata_challenge_detection_rate
                ),
            },
            "challenge_scope": (
                "deterministic_semantic_domain_shift_and_metadata_range_attack"
            ),
            "claim_boundary": (
                "challenge detection is a plumbing gate, not real-world OOD coverage"
            ),
        }
        self.ood_calibration_report = report
        if report["status"] != "PASS":
            raise RuntimeError("PM-v2 OOD calibration gate failed: " + canonical_json(report))
        return report

    def ood_report(self, state: PMV2State) -> dict[str, float | bool | str]:
        semantic_distance, metadata_ood = self._ood_scores(state)
        semantic_threshold = (
            float(self.semantic_ood_threshold)
            if self.semantic_ood_threshold is not None
            else max(0.45, self.train_text_radius_p95 * 2.0)
        )
        metadata_threshold = (
            float(self.metadata_ood_threshold)
            if self.metadata_ood_threshold is not None
            else 0.25
        )
        severe_semantic = semantic_distance > semantic_threshold
        severe_metadata = metadata_ood > metadata_threshold
        return {
            "semantic_ood_score": semantic_distance,
            "metadata_ood_score": metadata_ood,
            "train_semantic_radius_p95": self.train_text_radius_p95,
            "semantic_ood_threshold": semantic_threshold,
            "metadata_ood_threshold": metadata_threshold,
            "threshold_source": (
                "calibration_split"
                if self.ood_calibration_report.get("status") == "PASS"
                else "legacy_uncalibrated_heuristic"
            ),
            "severe_semantic_ood": severe_semantic,
            "severe_metadata_ood": severe_metadata,
            "recommendation": "FALLBACK" if severe_semantic or severe_metadata else "OK",
        }

    def config_hash(self) -> str:
        payload: dict[str, Any] = {
            "word_features": self.word_features,
            "char_features": self.char_features,
            "use_precomputed_embeddings": self.use_precomputed_embeddings,
            "require_precomputed_embeddings": self.require_precomputed_embeddings,
            "semantic_projection_dimensions": self.semantic_projection_dimensions,
            "step0_signal_mode": self.step0_signal_mode,
            "embedding_dim": self.embedding_dim,
            "raw_embedding_dim": self.raw_embedding_dim,
            "semantic_encoder_spec_sha256": self.semantic_encoder_spec_sha256,
            "source_similarity": "catalog_centroid_only",
            "inventory_scale_contract": "retrieval_capacity_v1",
            "memory_top_k": {
                source.value: MEMORY_TOP_K[source] for source in SOURCE_ORDER
            },
            "estimated_resource_cost_contract": self.cost_contract(),
            "semantic_ood_threshold": self.semantic_ood_threshold,
            "metadata_ood_threshold": self.metadata_ood_threshold,
            "ood_calibration_protocol": self.ood_calibration_report.get("protocol"),
            "session_index_feature_clip": SESSION_INDEX_FEATURE_CLIP,
            "memory_item_feature_token_clip": MEMORY_ITEM_FEATURE_TOKEN_CLIP,
            "strategy_feature_token_clip": STRATEGY_FEATURE_TOKEN_CLIP,
            "action_feature_token_clip": ACTION_FEATURE_TOKEN_CLIP,
            "catalog_count_feature": "min_count_top_k_fraction",
            "catalog_token_feature": "bounded_log_expected_top_k_tokens",
            "catalog_age_feature": "session_relative_min_median_max_span",
            "generated_stale_feature": False,
            "current_state_conflict_feature": False,
            "strategy_catalog_count_feature": False,
            "strategy_cost_feature": "configured_action_tokens",
        }
        return sha256_text(canonical_json(payload))
