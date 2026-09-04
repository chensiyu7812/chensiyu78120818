"""Frozen minimal Strategy Guidance RAG runtime for PM V1.5.

The runtime is deliberately small and fail-closed:

1. derive only explicit observable opportunity/boundary flags;
2. apply hard eligibility before ranking;
3. rank eligible cards with term-frequency cosine;
4. inject at most one card, or abstain below the frozen score floor.

It does not estimate whether Strategy Guidance will improve the response.  That
component-effect question belongs to the PM clean-pair labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence

from .contracts import StrategyCard
from .text import lexical_score, normalize_space


PROTOCOL = "pm-v1.5-qualified-minimal-strategy-rag-runtime-v1"
TOP_K = 1
MINIMUM_SCORE = 0.05
MOVE_IDS = frozenset(
    {
        "AM01_invite_open_expression",
        "AM02_ask_one_focused_clarification",
        "AM04_tentative_paraphrase_check",
        "AM05_grounded_validation",
        "AM10_offer_one_optional_micro_step",
        "AM14_supportive_transition",
    }
)

_ACTIVE_HIGH_STAKES_RE = re.compile(
    r"\b(?:suicid(?:e|al)|self[- ]?harm|"
    r"(?:want|plan|going|might|may|could|feel like|thinking about)"
    r"(?:\s+\w+){0,3}\s+(?:kill|hurt|harm)(?:ing)?"
    r"(?:\s+\w+){0,2}\s+(?:myself|someone|him|her|them)|"
    r"(?:immediate|in|real)\s+danger|"
    r"(?:partner|spouse|boyfriend|girlfriend|parent)"
    r"(?:\s+\w+){0,4}\s+(?:hit|beat|threaten|abuse)(?:s|d|ing)?)\b",
    flags=re.IGNORECASE,
)
_EXPLICIT_STOP_RE = re.compile(
    r"(?:\b(?:please\s+)?(?:stop|end|close|quit)"
    r"(?:\s+(?:this|the))?\s+(?:chat|conversation|talking|survey)\b|"
    r"^\s*(?:bye|goodbye|good-bye)\W*$)",
    flags=re.IGNORECASE,
)
_TRANSITION_RE = re.compile(
    r"(?:\b(?:wrap (?:this|it) up|change the subject|talk about something else|"
    r"done (?:talking|for now)|enough for (?:today|now)|pause (?:here|this)|"
    r"continue (?:later|another time)|good ?night)\b|"
    r"^\s*(?:thanks|thank you)(?:\s+(?:again|so much))?[.! ]*$)",
    flags=re.IGNORECASE,
)
_LISTEN_ONLY_RE = re.compile(
    r"\b(?:just|only)\s+(?:want|need)(?:\s+you)?\s+to\s+listen\b|"
    r"\b(?:do not|don't|dont)\s+(?:want|need)\s+(?:any\s+)?advice\b|"
    r"\bno\s+advice\b",
    flags=re.IGNORECASE,
)
_ADVICE_WELCOME_RE = re.compile(
    r"(?:\bwhat\s+(?:do|should|can|could)\s+i\s+(?:do|try)\b|"
    r"\bhow\s+(?:do|can|should|could)\s+i\b|"
    r"\bany\s+advice\b|"
    r"\b(?:can|could|would)\s+you\s+(?:suggest|recommend)\b|"
    r"\b(?:can|could|would)\s+you\s+give\s+me\s+(?:some|any|one)?\s*advice\b|"
    r"\b(?:can|could|would)\s+you\s+help\s+me\s+"
    r"(?:decide|figure (?:this|it|things) out|plan|start)\b|"
    r"\bwhat\s+would\s+you\s+(?:do|suggest|recommend)\b)",
    flags=re.IGNORECASE,
)
_OPEN_EXPRESSION_RE = re.compile(
    r"\b(?:i\s+(?:want|need|would like)\s+to\s+talk|can\s+i\s+talk|"
    r"need\s+someone\s+to\s+listen|"
    r"(?:do not|don't|dont)\s+know\s+(?:where|how)\s+to\s+(?:start|begin)|"
    r"not\s+sure\s+(?:where|how)\s+to\s+(?:start|begin)|"
    r"something\s+(?:is\s+)?bothering\s+me)\b",
    flags=re.IGNORECASE,
)
_CLARIFICATION_RE = re.compile(
    r"\b(?:hard|difficult)\s+to\s+explain\b|"
    r"\bnot\s+sure\s+how\s+to\s+(?:explain|describe|say)\b|"
    r"\b(?:it|this|things?)\s+(?:is|are|'s)\s+complicated\b|"
    r"\bdepends?\s+on\b|"
    r"\b(?:have not|haven't|havent)\s+(?:said|explained|mentioned)\b",
    flags=re.IGNORECASE,
)
_PARAPHRASE_TENSION_RE = re.compile(
    r"\bpart\s+of\s+me\b.*\b(?:but|and)\b.*\bpart\s+of\s+me\b|"
    r"\b(?:i\s+(?:want|feel|think|know|am|I'm)[^.!?]{2,80})"
    r"\b(?:but|though|yet)\b[^.!?]{2,100}",
    flags=re.IGNORECASE,
)
_FEELING_RE = re.compile(
    r"\b(?:i\s+(?:feel|felt|am|'m)\s+"
    r"(?:sad|angry|upset|hurt|afraid|scared|anxious|worried|nervous|"
    r"overwhelmed|exhausted|tired|lonely|frustrated|confused|embarrassed|"
    r"ashamed|guilty|disappointed|stressed|depressed|discouraged|lost)|"
    r"it\s+(?:hurts|hurt)|"
    r"(?:this|it)\s+(?:is|has been|'s)\s+"
    r"(?:hard|difficult|painful|overwhelming|exhausting|frustrating))\b",
    flags=re.IGNORECASE,
)
_ANSWER_ALREADY_VISIBLE_RE = re.compile(
    r"\b(?:i\s+(?:already|just)\s+(?:answered|said|told)|"
    r"you\s+(?:already|just)\s+asked|"
    r"that\s+question\s+(?:was|has been)\s+answered)\b",
    flags=re.IGNORECASE,
)


def _speaker(row: dict[str, Any] | object) -> str:
    if isinstance(row, dict):
        value = row.get("speaker") or row.get("role")
    else:
        value = getattr(row, "speaker", None) or getattr(row, "role", None)
    normalized = str(getattr(value, "value", value) or "").casefold()
    return "seeker" if normalized in {"seeker", "user"} else "supporter"


def _content(row: dict[str, Any] | object) -> str:
    if isinstance(row, dict):
        return normalize_space(row.get("content"))
    return normalize_space(getattr(row, "content", ""))


def _latest_seeker_text(recent_dialogue: Sequence[dict[str, Any] | object]) -> str:
    return next(
        (_content(turn) for turn in reversed(recent_dialogue) if _speaker(turn) == "seeker"),
        "",
    )


def _recent_seeker_text(
    recent_dialogue: Sequence[dict[str, Any] | object],
    count: int = 3,
) -> str:
    rows = [_content(turn) for turn in recent_dialogue if _speaker(turn) == "seeker"]
    return " ".join(rows[-count:])


def observable_flags(
    recent_dialogue: Sequence[dict[str, Any] | object],
) -> dict[str, bool]:
    latest = _latest_seeker_text(recent_dialogue)
    recent_user = _recent_seeker_text(recent_dialogue)
    active_high_stakes = bool(_ACTIVE_HIGH_STAKES_RE.search(recent_user))
    explicit_stop = bool(_EXPLICIT_STOP_RE.search(latest))
    listen_only = bool(_LISTEN_ONLY_RE.search(latest))
    advice_welcome = bool(_ADVICE_WELCOME_RE.search(latest)) and not listen_only
    return {
        "active_high_stakes": active_high_stakes,
        "explicit_stop": explicit_stop,
        "listen_only": listen_only,
        "question_repetition_block": bool(_ANSWER_ALREADY_VISIBLE_RE.search(latest)),
        "open_expression_opportunity": bool(_OPEN_EXPRESSION_RE.search(latest)),
        "focused_clarification_opportunity": bool(_CLARIFICATION_RE.search(latest)),
        "paraphrase_check_opportunity": bool(_PARAPHRASE_TENSION_RE.search(latest)),
        "grounded_validation_opportunity": bool(_FEELING_RE.search(latest)),
        "explicit_advice_welcome": advice_welcome,
        "one_low_risk_step_available": advice_welcome and not active_high_stakes,
        "transition_opportunity": bool(_TRANSITION_RE.search(latest)),
    }


def eligible_moves(flags: dict[str, bool]) -> list[str]:
    if flags["active_high_stakes"]:
        return []
    # Routine stop/bye is an RS negative control: the base generator can close
    # naturally, so opening a resource adds cost without a plausible material
    # benefit.  Non-trivial transition guidance belongs to a later qualified
    # Bank, not this frozen six-card development runtime.
    if flags["explicit_stop"]:
        return []
    eligible: list[str] = []
    repeated = flags["question_repetition_block"]
    if flags["open_expression_opportunity"] and not repeated:
        eligible.append("AM01_invite_open_expression")
    if (
        flags["focused_clarification_opportunity"]
        and not repeated
        and not flags["listen_only"]
    ):
        eligible.append("AM02_ask_one_focused_clarification")
    if flags["paraphrase_check_opportunity"]:
        eligible.append("AM04_tentative_paraphrase_check")
    if flags["grounded_validation_opportunity"]:
        eligible.append("AM05_grounded_validation")
    if (
        flags["explicit_advice_welcome"]
        and flags["one_low_risk_step_available"]
        and not flags["listen_only"]
    ):
        eligible.append("AM10_offer_one_optional_micro_step")
    # Routine thanks/goodnight are also handled by R0.  Keep AM14 in the
    # historical six-card catalog for provenance, but do not make it an
    # opportunity in the corrected research runtime.
    return eligible


def query_text(recent_dialogue: Sequence[dict[str, Any] | object]) -> str:
    visible = list(recent_dialogue)[-6:]
    return "\n".join(
        f"{'User' if _speaker(row) == 'seeker' else 'Supporter'}: {_content(row)}"
        for row in visible
    )


@dataclass(frozen=True)
class StrategyRAGDecision:
    status: str
    observable_flags: dict[str, bool]
    eligible_move_ids: tuple[str, ...]
    selected_cards: tuple[StrategyCard, ...]
    selected_score: float | None
    query_text: str


class QualifiedStrategyRAG:
    """The only Strategy Guidance retriever qualified for new V1.5 G3 work."""

    def __init__(
        self,
        cards: Sequence[StrategyCard],
        *,
        minimum_score: float = MINIMUM_SCORE,
        top_k: int = TOP_K,
    ):
        if int(top_k) != TOP_K:
            raise ValueError("qualified V1.5 Strategy RAG requires top_k=1")
        if float(minimum_score) != MINIMUM_SCORE:
            raise ValueError("qualified V1.5 Strategy RAG requires score floor 0.05")
        self.cards = tuple(cards)
        labels = [card.strategy_label for card in self.cards]
        if len(self.cards) != 6 or set(labels) != MOVE_IDS or len(set(labels)) != 6:
            raise ValueError("qualified V1.5 Strategy RAG requires the frozen six cards")
        self.card_by_move = {card.strategy_label: card for card in self.cards}
        self.minimum_score = MINIMUM_SCORE
        self.top_k = TOP_K

    def rank_eligible(
        self,
        query: str,
        eligible_move_ids: Sequence[str],
    ) -> list[tuple[float, StrategyCard]]:
        unknown = set(eligible_move_ids) - MOVE_IDS
        if unknown:
            raise ValueError(f"unknown eligible Strategy move(s): {sorted(unknown)}")
        return sorted(
            (
                (
                    lexical_score(query, self.card_by_move[move].retrieval_text),
                    self.card_by_move[move],
                )
                for move in eligible_move_ids
            ),
            key=lambda pair: (pair[0], pair[1].strategy_id),
            reverse=True,
        )

    def retrieve(
        self,
        recent_dialogue: Sequence[dict[str, Any] | object],
    ) -> StrategyRAGDecision:
        flags = observable_flags(recent_dialogue)
        eligible = eligible_moves(flags)
        query = query_text(recent_dialogue)
        if not eligible:
            status = (
                "off_active_high_stakes"
                if flags["active_high_stakes"]
                else "off_no_explicit_opportunity"
            )
            return StrategyRAGDecision(
                status=status,
                observable_flags=flags,
                eligible_move_ids=(),
                selected_cards=(),
                selected_score=None,
                query_text=query,
            )
        ranked = self.rank_eligible(query, eligible)
        score, card = ranked[0]
        if score < self.minimum_score:
            return StrategyRAGDecision(
                status="off_below_score_floor",
                observable_flags=flags,
                eligible_move_ids=tuple(eligible),
                selected_cards=(),
                selected_score=float(score),
                query_text=query,
            )
        return StrategyRAGDecision(
            status="retrieved_top1",
            observable_flags=flags,
            eligible_move_ids=tuple(eligible),
            selected_cards=(card,),
            selected_score=float(score),
            query_text=query,
        )


def _strategy_prompt_guidance(
    strategy: StrategyCard | dict[str, Any],
) -> str:
    if isinstance(strategy, dict):
        value = strategy.get("prompt_guidance") or strategy.get(
            "guidance_text"
        )
    else:
        value = strategy.guidance_text
    normalized = normalize_space(str(value or ""))
    if not normalized:
        raise ValueError("strategy guidance is empty")
    return normalized


def compile_strategy_guidance(
    strategies: Sequence[StrategyCard | dict[str, Any]],
) -> str:
    """Compile the one shared V1.5 RS injection surface.

    Both the qualified V4 mappings and the external runtime's ``StrategyCard``
    adapter enter here. Raw source examples are never injected.
    """

    if len(strategies) > TOP_K:
        raise ValueError("qualified V1.5 prompt compiler accepts at most one card")
    lines = [
        f"- {_strategy_prompt_guidance(strategy)}"
        for strategy in strategies
    ]
    return (
        "Potential emotional-support technique. Treat it as the primary "
        "move, not the whole reply. Use it only when it fits; otherwise "
        "ignore it. Ground every factual, emotional, and temporal claim "
        "in the visible dialogue. Ask at most one question OR give at "
        "most one suggestion, never both or a list. You may add one short "
        "natural continuation when needed. Compose new wording and never "
        "mention this guidance:\n"
        + "\n".join(lines)
    )
