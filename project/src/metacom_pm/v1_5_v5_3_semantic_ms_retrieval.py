"""V5.3 candidate-discovery fix: BGE-M3 semantic scoring for MS retrieval.

Why this exists
----------------
Within V1.5/V5.2's typed candidate selector, the ranking/scoring step
(``retrieval.MemoryRetriever.retrieve()``) ranks every memory source -- MP,
MS, ME -- with ``text.lexical_score``: raw term-frequency cosine, no IDF, no
stopword weighting. (The selector as a whole is more than this scoring
function -- candidate discovery, owner/version/eligibility checks, and typed
candidate construction also matter; this module only replaces the scoring
step, and only for MS.) Two EXPLORATORY, single-annotator diagnostics
(``scripts/v1_5/42_diagnose_ms_retrieval_scoring_v1_5.py`` on the frozen
ES-MemEval session-retrieval task, and
``scripts/v1_5/43_diagnose_ms_rank1_on_real_states_v1_5.py`` on the real 138
EvoEmo response-generation states) suggest this scorer is A significant
contributor to EvoEmo's low MS candidate topical-relevance rate -- not
confirmed as the sole or dominant bottleneck, and not yet measured through a
same-stack qualifying trial. On a 50-state single-pass manual read (not
persisted as a structured, auditable per-item label table -- see the
manifest's "manual_annotation_disclosure"), lexical's Rank-1 pick looked
topically unrelated to the current turn roughly 32% of the time and precisely
on-topic roughly 36% of the time, vs. roughly 2% unrelated / 70% precisely
on-topic for BGE-M3 on the identical sample. These numbers are diagnostic
direction, not audited results. Sanitized summary + artifact manifest (git
does not track the raw outputs/ directory, per artifact policy):
``docs/PM_V1_5_V5_3_MS_RETRIEVAL_SCORING_DIAGNOSTIC_ZH.md`` and
``docs/PM_V1_5_V5_3_MS_RETRIEVAL_DIAGNOSTIC_ARTIFACT_MANIFEST_20260805_ZH.json``.
The candidate pools used by scripts 43/44 do not explicitly filter by
`created_session < current_session_index` (a real construction gap versus the
formal protocol); a per-state check of all 138 final Rank-1 picks found zero
future/current-session selections by either scorer, so this is a candidate-
pool-contract imprecision and potential-risk issue, not a demonstrated result
leak -- the formal qualifying trial must still fix the construction gap
before this can be cited as leak-free by design. Authorized for adoption
into V5.3 by the user on 2026-08-05; the actual same-stack qualifying trial
is deferred until git version governance is complete.

Scope: MS only
--------------
The evidence above is specific to MS (session-summary/episodic narrative
text). It says nothing about MP (short factual/preference statements, a very
different content type) or ME, which were not separately validated. Do not
extend this scorer to MP or ME without repeating the same measurement for
that source -- see V15-OBS-19 in the failure ledger, where an embedding
retriever *lost* to lexical for a different content type (Strategy Bank
cards). This module intentionally exposes only MS-scoped entry points so a
caller cannot accidentally apply it somewhere unvalidated.

Status: implemented, unit-tested, NOT yet wired into any generation pipeline.
V5.3 is still PRE_IMPLEMENTATION; the frozen V5.2 ``retrieval.py`` is
untouched, per the P1 rule "不得原地修改V5.2实现或结果". Wiring this into the
V5.3 executor's Top-k discovery step is P1 integration work, not part of this
module.

No network access: local snapshot only, ``local_files_only=True``, matching
the project's existing BGE-M3 usage in
``scripts/v1_5/28c_materialize_v5_2_external_e3_retrieval_v1_5.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .contracts import MemoryItem, MemorySource

SEMANTIC_MS_RETRIEVAL_PROTOCOL = "pm-v1.5-v5.3-semantic-ms-retrieval-v1"

DEFAULT_BGE_M3_SNAPSHOT = Path(
    "/home/tokkio/.cache/huggingface/hub/models--BAAI--bge-m3/"
    "snapshots/5617a9f61b028005a4858fdac845db406aefb181"
)


class TextEncoder(Protocol):
    """Minimal interface this module needs.

    Lets tests inject a fake encoder instead of loading the real ~2GB BGE-M3
    weights; ``BgeM3Encoder`` below is the real implementation.
    """

    def encode(self, texts: Sequence[str]):  # -> numpy.ndarray, shape (n, d)
        ...


class BgeM3Encoder:
    """Lazy-loaded local BGE-M3 encoder. Requires torch/transformers; only
    imported inside __init__/encode so importing this module never requires
    them (matches pm_v1_5_semantic.py's lazy-import convention)."""

    def __init__(self, snapshot_path: Path = DEFAULT_BGE_M3_SNAPSHOT) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if not snapshot_path.is_dir():
            raise RuntimeError(f"local BGE-M3 snapshot missing: {snapshot_path}")
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(snapshot_path, local_files_only=True)
        self.model.eval()

    def encode(self, texts: Sequence[str]):
        import numpy as np

        torch = self._torch
        if not texts:
            return np.zeros((0, 1024), dtype="float32")
        pieces = []
        with torch.inference_mode():
            for start in range(0, len(texts), 8):
                batch = self.tokenizer(
                    list(texts[start : start + 8]),
                    padding=True,
                    truncation=True,
                    max_length=8192,
                    return_tensors="pt",
                )
                vector = self.model(**batch).last_hidden_state[:, 0]
                vector = torch.nn.functional.normalize(vector, p=2, dim=1)
                pieces.append(vector.numpy())
        return np.concatenate(pieces, axis=0)


class CachedTextEncoder:
    """Exact-text memoization wrapper for repeated same-catalog scoring.

    P2R evaluates several counterfactual current turns against the same
    13--33-session user catalog. Re-encoding every unchanged candidate for
    every turn adds latency and memory churn but no new information. This
    wrapper caches only the encoder's deterministic text vector; it does not
    cache rankings, labels, actions, or outcomes and therefore cannot change
    the scoring method.
    """

    def __init__(self, base: TextEncoder) -> None:
        self.base = base
        self._vectors: dict[str, object] = {}

    def encode(self, texts: Sequence[str]):
        import numpy as np

        ordered = list(texts)
        missing = list(dict.fromkeys(text for text in ordered if text not in self._vectors))
        if missing:
            vectors = self.base.encode(missing)
            if len(vectors) != len(missing):
                raise ValueError("encoder returned a different number of vectors than texts")
            for text, vector in zip(missing, vectors):
                self._vectors[text] = vector
        if not ordered:
            return np.zeros((0, 0), dtype="float32")
        return np.stack([self._vectors[text] for text in ordered], axis=0)


@dataclass(frozen=True)
class RankedMsCandidate:
    item: MemoryItem
    score: float
    rank: int


def rank_ms_candidates(
    query: str,
    items: Sequence[MemoryItem],
    *,
    encoder: TextEncoder,
    top_k: int | None = None,
) -> list[RankedMsCandidate]:
    """Rank MS-source candidates by BGE-M3 cosine similarity to ``query``.

    Deterministic tie-break matches the project's existing convention
    (score desc, then ``created_session`` desc, then ``memory_id`` desc), the
    same pattern used by ``retrieval.MemoryRetriever`` and
    ``hybrid_retrieval._ranked_positions``.

    Raises if any item is not ``MemorySource.MS`` -- this scorer is only
    validated for MS; see the module docstring.
    """

    ms_items = [item for item in items if item.source is MemorySource.MS]
    if len(ms_items) != len(items):
        raise ValueError(
            "rank_ms_candidates only accepts MemorySource.MS items; this scorer "
            "has not been validated for MP or ME (see module docstring)"
        )
    if not ms_items:
        return []

    vectors = encoder.encode([query, *[item.text for item in ms_items]])
    query_vector, item_vectors = vectors[0], vectors[1:]
    scored = [
        (float(query_vector @ item_vector), item.created_session, item.memory_id, item)
        for item_vector, item in zip(item_vectors, ms_items)
    ]
    scored.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    ranked = [
        RankedMsCandidate(item=item, score=score, rank=rank)
        for rank, (score, _created_session, _memory_id, item) in enumerate(scored, start=1)
    ]
    return ranked[:top_k] if top_k is not None else ranked
