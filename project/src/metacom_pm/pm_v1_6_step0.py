from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from .contracts import MemoryItem, MemorySource, RuntimeState, StrategyCard
from .io import canonical_json, sha256_text
from .pm_v1_6_contracts import (
    STEP0_PROTOCOL,
    STRATEGY_FAMILIES,
    SourceProbeObservation,
    Step0AuditBinding,
    Step0Cost,
    Step0Observation,
    StrategyFamilyObservation,
    StrategyReadiness,
)
from .retrieval import DEFAULT_MEMORY_TOP_K, context_query
from .text import normalize_space

STEP0_ALGORITHM_ID = "deterministic-hashing-centroid-source-probe-v1"
STEP0_ENCODER_ID = "sklearn-hashing-word-char-norm-l2-v1"
STEP0_REFRESH_POLICY = "rebuild_only_when_authorized_catalog_content_changes"
DEFAULT_DIMENSION = 512

_FAMILY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("exploration_question", ("question", "exploration", "ask")),
    ("restatement_paraphrase", ("restatement", "paraphras", "restate")),
    ("reflection", ("reflection", "feeling")),
    ("self_disclosure", ("self-disclosure", "self disclosure", "disclosure")),
    (
        "affirmation_reassurance",
        ("affirmation", "reassurance", "encouragement", "validation"),
    ),
    ("suggestion_action", ("suggestion", "advice", "action", "planning")),
    ("information", ("information", "resource", "psychoeducation")),
    ("non_directive_listening", ("listen", "non-directive", "non directive")),
)


@dataclass(frozen=True)
class FrozenStep0Config:
    dimension: int = DEFAULT_DIMENSION
    word_features: int = 256
    char_features: int = 256
    maximum_source_items: Mapping[MemorySource, int] | None = None

    def __post_init__(self) -> None:
        if self.dimension != self.word_features + self.char_features:
            raise ValueError("Step-0 dimension must equal word + char dimensions")
        if self.word_features <= 0 or self.char_features <= 0:
            raise ValueError("Step-0 feature dimensions must be positive")

    def source_cap(self, source: MemorySource) -> int:
        values = self.maximum_source_items or DEFAULT_MEMORY_TOP_K
        return int(values[source])

    def payload(self) -> dict[str, Any]:
        return {
            "protocol": STEP0_PROTOCOL,
            "algorithm_id": STEP0_ALGORITHM_ID,
            "encoder_id": STEP0_ENCODER_ID,
            "dimension": self.dimension,
            "word_features": self.word_features,
            "char_features": self.char_features,
            "maximum_source_items": {
                source.value: self.source_cap(source) for source in MemorySource
            },
            "strategy_family_order": list(STRATEGY_FAMILIES),
            "refresh_policy": STEP0_REFRESH_POLICY,
        }

    def digest(self) -> str:
        return sha256_text(canonical_json(self.payload()))


class _FrozenEncoder:
    def __init__(self, config: FrozenStep0Config):
        self.config = config
        self.word = HashingVectorizer(
            n_features=config.word_features,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(1, 2),
            analyzer="word",
        )
        self.char = HashingVectorizer(
            n_features=config.char_features,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(3, 5),
            analyzer="char_wb",
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        clean = [normalize_space(text) for text in texts if normalize_space(text)]
        if not clean:
            return np.zeros((0, self.config.dimension), dtype=np.float64)
        word = self.word.transform(clean).toarray().astype(np.float64)
        char = self.char.transform(clean).toarray().astype(np.float64)
        matrix = np.concatenate([word, char], axis=1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)


def canonical_strategy_family(label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip()
    for family, patterns in _FAMILY_PATTERNS:
        if any(pattern in normalized for pattern in patterns):
            return family
    return "other"


def _centroid(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((matrix.shape[1] if matrix.ndim == 2 else DEFAULT_DIMENSION,), dtype=np.float64)
    value = matrix.mean(axis=0)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 0 else value


def _similarity(query_vector: np.ndarray, representation: np.ndarray) -> float:
    if not query_vector.size or not representation.size:
        return 0.0
    return float(np.clip(query_vector @ representation, -1.0, 1.0))


def _readiness(text: str) -> StrategyReadiness:
    value = normalize_space(text).lower()
    advice_requested = bool(
        re.search(
            r"\b(what should i do|what can i do|any advice|next step|how should i|"
            r"help me decide|suggest|recommend)\b",
            value,
        )
    )
    advice_rejected = bool(
        re.search(
            r"\b(no advice|do not give me advice|don't give me advice|not ready for advice|"
            r"without advice|i don'?t want suggestions)\b",
            value,
        )
    )
    listening_requested = bool(
        re.search(r"\b(just listen|hear me out|need you to listen|just need to vent)\b", value)
    )
    clarification_needed = bool(
        re.search(r"\b(i don'?t know|not sure|confused|hard to explain|can'?t put it)\b", value)
    )
    readiness = 0.5
    readiness += 0.35 if advice_requested else 0.0
    readiness -= 0.45 if advice_rejected else 0.0
    readiness -= 0.25 if listening_requested else 0.0
    readiness += 0.10 if "ready" in value or "plan" in value else 0.0
    return StrategyReadiness(
        advice_requested=advice_requested,
        advice_rejected=advice_rejected,
        listening_requested=listening_requested,
        clarification_needed=clarification_needed,
        action_readiness=float(np.clip(readiness, 0.0, 1.0)),
    )


def build_step0_observation(
    *,
    runtime_state: RuntimeState,
    memory_items: Sequence[MemoryItem],
    strategy_cards: Sequence[StrategyCard],
    config: FrozenStep0Config | None = None,
    catalog_build_amortized_ms: float = 0.0,
) -> tuple[Step0Observation, Step0AuditBinding]:
    """Build one PM-visible source probe plus a separate audit-only binding."""

    frozen = config or FrozenStep0Config()
    encoder = _FrozenEncoder(frozen)
    query = context_query(
        runtime_state.current_user_text,
        [row.model_dump(mode="json") for row in runtime_state.current_session_history],
        runtime_state.current_session_summary,
    )
    start = time.perf_counter()
    query_matrix = encoder.encode([query])
    query_vector = query_matrix[0] if len(query_matrix) else np.zeros(frozen.dimension)

    source_rows: dict[MemorySource, SourceProbeObservation] = {}
    source_hashes: dict[MemorySource, str] = {}
    for source in MemorySource:
        source_items = [item for item in memory_items if item.source is source]
        matrix = encoder.encode([item.text for item in source_items])
        representation = _centroid(matrix)
        representation_valid = bool(source_items and np.linalg.norm(representation) > 0)
        cat = runtime_state.inventory[source]
        cap = frozen.source_cap(source)
        bounded_count = min(int(cat.count), cap)
        mean_tokens = (
            int(round(cat.estimated_tokens / cat.count))
            if cat.count > 0
            else 0
        )
        retrievable_tokens = min(
            int(cat.estimated_tokens),
            bounded_count * max(mean_tokens, 0),
        )
        source_rows[source] = SourceProbeObservation(
            source=source,
            available=bool(cat.available),
            bounded_count=bounded_count,
            min_age_sessions=cat.min_age_sessions,
            median_age_sessions=(
                float(cat.min_age_sessions + cat.max_age_sessions) / 2.0
                if cat.min_age_sessions is not None
                and cat.max_age_sessions is not None
                else None
            ),
            max_age_sessions=cat.max_age_sessions,
            estimated_retrievable_tokens=retrievable_tokens,
            query_to_source_similarity=(
                _similarity(query_vector, representation)
                if representation_valid
                else 0.0
            ),
            representation_valid=representation_valid,
        )
        source_hashes[source] = sha256_text(
            canonical_json(
                {
                    "source": source.value,
                    "algorithm": STEP0_ALGORITHM_ID,
                    "authorized_text_sha256": [
                        sha256_text(normalize_space(item.text)) for item in source_items
                    ],
                    "representation": representation.round(12).tolist(),
                }
            )
        )

    family_cards: dict[str, list[StrategyCard]] = {
        family: [] for family in STRATEGY_FAMILIES
    }
    for card in strategy_cards:
        family_cards[canonical_strategy_family(card.strategy_label)].append(card)
    family_rows: list[StrategyFamilyObservation] = []
    family_hashes: dict[str, str] = {}
    for family in STRATEGY_FAMILIES:
        cards = family_cards[family]
        texts = [
            " ".join((card.retrieval_text, card.guidance_text))
            for card in cards
        ]
        representation = _centroid(encoder.encode(texts))
        valid = bool(cards and np.linalg.norm(representation) > 0)
        family_rows.append(
            StrategyFamilyObservation(
                family=family,
                query_to_family_similarity=(
                    _similarity(query_vector, representation) if valid else 0.0
                ),
                representation_valid=valid,
            )
        )
        family_hashes[family] = sha256_text(
            canonical_json(
                {
                    "family": family,
                    "algorithm": STEP0_ALGORITHM_ID,
                    "card_count": len(cards),
                    "authorized_card_content_sha256": [
                        sha256_text(normalize_space(text)) for text in texts
                    ],
                    "representation": representation.round(12).tolist(),
                }
            )
        )

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    observation = Step0Observation(
        state_id=runtime_state.state_id,
        sources=source_rows,
        strategy_families=family_rows,
        readiness=_readiness(runtime_state.current_user_text),
        step0_cost=Step0Cost(
            query_encoding_latency_ms=elapsed_ms,
            memory_source_comparisons=len(MemorySource),
            strategy_family_comparisons=len(STRATEGY_FAMILIES),
            catalog_build_amortized_ms=float(catalog_build_amortized_ms),
        ),
    )
    audit = Step0AuditBinding(
        state_id=runtime_state.state_id,
        algorithm_id=STEP0_ALGORITHM_ID,
        encoder_id=STEP0_ENCODER_ID,
        representation_dimension=frozen.dimension,
        source_representation_sha256=source_hashes,
        strategy_family_representation_sha256=family_hashes,
        catalog_build_sha256=sha256_text(
            canonical_json(
                {
                    "step0_config_sha256": frozen.digest(),
                    "source_hashes": {
                        source.value: digest for source, digest in source_hashes.items()
                    },
                    "family_hashes": family_hashes,
                }
            )
        ),
        refresh_policy=STEP0_REFRESH_POLICY,
    )
    return observation, audit
