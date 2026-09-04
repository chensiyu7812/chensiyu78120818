from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
LEAK_PREFIXES = (
    "unrelated prior event:",
    "outdated note:",
    "conflicting note:",
    "audit:",
    "gold:",
    "expected source:",
)

# Frozen before the single PM V1.5 scale-stable memory-feature repair.  These
# are ordinary English function words plus compiler wrapper words that can be
# repeated across every memory item without making that item relevant.  The
# list is deliberately small and corpus-independent; it was not selected from
# ESConv/EvoEmo outcomes.
CONTENT_MATCH_IGNORED_WORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does doing for
    from had has have having he her hers herself him himself his how i if in
    into is it its itself just may me might more most my myself no nor not of
    on once only or other our ours ourselves out over own same she should so
    some such than that the their theirs them themselves then there these they
    this those through to too under until up very was we were what when where
    which while who whom why will with would you your yours yourself yourselves
    again already also any anything around back because before between both
    down during each even ever every few further get got here like many much
    now off one please really said say see since still take tell thing think
    today together tonight two us want way well went whenever whether within
    without yes yet
    stable preference support style profile prior event longitudinal pattern
    session summary user seeker supporter assistant current recent relevant
    memory discussing
    """.split()
)


def normalize_space(text: str) -> str:
    return " ".join(str(text or "").split())


def normalize_for_hash(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def token_set(text: str) -> set[str]:
    return set(tokens(text))


def estimate_tokens(text: str) -> int:
    # Provider-neutral base approximation; API budget gates add a frozen margin.
    return max(1, math.ceil(len(str(text or "")) / 4))


def conservative_token_bound(text: str, *, safety_factor: float) -> int:
    """Return the frozen provider-neutral planning bound for input tokens."""

    if not math.isfinite(safety_factor) or safety_factor < 1.0:
        raise ValueError("input-token safety factor must be finite and at least 1")
    return max(1, math.ceil(estimate_tokens(text) * safety_factor))


def clean_memory_text(text: str) -> str:
    value = normalize_space(text)
    lowered = value.lower()
    for prefix in LEAK_PREFIXES:
        if lowered.startswith(prefix):
            value = value[len(prefix):].strip()
            lowered = value.lower()
    # Neutralize source-construction labels while preserving content.
    for prefix in ("prior event:", "longitudinal pattern:", "support preference:"):
        if lowered.startswith(prefix):
            value = value[len(prefix):].strip()
            lowered = value.lower()
    if not value:
        raise ValueError("memory text became empty after cleaning")
    return value


def lexical_score(query: str, document: str) -> float:
    q = Counter(tokens(query))
    d = Counter(tokens(document))
    if not q or not d:
        return 0.0
    dot = sum(q[t] * d.get(t, 0) for t in q)
    qn = math.sqrt(sum(v * v for v in q.values()))
    dn = math.sqrt(sum(v * v for v in d.values()))
    return dot / (qn * dn) if qn and dn else 0.0


def content_words(text: str) -> set[str]:
    """Return a transparent, corpus-independent set of English content words."""

    return {
        token
        for token in tokens(text)
        if len(token) >= 3
        and not token.isdigit()
        and token not in CONTENT_MATCH_IGNORED_WORDS
    }


def content_word_match_level(query: str, document: str) -> float:
    """Return the frozen coarse overlap level ``0``, ``0.5``, or ``1``.

    The old feature used term-frequency cosine, whose value changed strongly
    with query/catalog length across the short internal histories and EvoEmo.
    This replacement asks only whether the query and candidate share zero,
    one, or at least two distinct content words.  It is bounded and does not
    depend on either text's length.
    """

    overlap = len(content_words(query) & content_words(document))
    if overlap == 0:
        return 0.0
    if overlap == 1:
        return 0.5
    return 1.0


def shingle_set(text: str, n: int = 5) -> set[tuple[str, ...]]:
    ts = tokens(text)
    if len(ts) < n:
        return {tuple(ts)} if ts else set()
    return {tuple(ts[i : i + n]) for i in range(len(ts) - n + 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def dialogue_text(dialogue: Iterable[dict]) -> str:
    return "\n".join(
        f"{row.get('speaker') or row.get('role')}: {normalize_space(row.get('content'))}"
        for row in dialogue
    )
