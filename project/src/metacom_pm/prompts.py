from __future__ import annotations

import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .contracts import MemoryItem, RuntimeState, StrategyCard


BASE_SUPPORTER_SYSTEM = """You are an emotional-support conversation assistant.
Respond with empathy, respect, and appropriate practical help. Do not diagnose,
claim clinical authority, or invent personal facts. Use supplied memory only
when it clearly helps the current exchange. Do not mention retrieval, memory
stores, strategy cards, scores, or system conditions. If supplied history is
irrelevant or uncertain, ignore it. Keep the reply concise and natural."""


OFFICIAL_ESMEM_SYSTEM = """You are an emotional-support conversation assistant.
This is the official-style long-term-memory recall condition. Use the supplied
past dialogues to demonstrate accurate memory of the user's previous events and
information, even when a recalled detail is only loosely connected to the
current exchange. Do not invent memory. Keep the reply concise and emotionally
supportive."""


SELECTIVE_ESMEM_SYSTEM = """You are an emotional-support conversation assistant.
Use supplied past information only when it clearly helps the current user.
Do not force a memory reference merely to demonstrate recall. Ignore irrelevant,
outdated, or awkwardly intrusive information, and never invent personal facts.
Keep the reply concise, natural, and emotionally supportive."""


SUPPORTER_SYSTEM_PROMPTS: Mapping[str, str] = MappingProxyType(
    {
        "base_supporter_v1": BASE_SUPPORTER_SYSTEM,
        "official_esmem_v1": OFFICIAL_ESMEM_SYSTEM,
        "selective_esmem_v1": SELECTIVE_ESMEM_SYSTEM,
    }
)


def resolve_supporter_system_prompt(prompt_id: str) -> str:
    """Resolve an allowlisted supporter prompt by its frozen semantic ID."""

    try:
        return SUPPORTER_SYSTEM_PROMPTS[prompt_id]
    except KeyError as exc:
        allowed = ", ".join(sorted(SUPPORTER_SYSTEM_PROMPTS))
        raise ValueError(
            f"unknown supporter system prompt {prompt_id!r}; allowed: {allowed}"
        ) from exc


def common_context(state: RuntimeState) -> str:
    # Some source adapters historically included the current user turn as the
    # last history item and then supplied it again as current_user_text.  Drop
    # that duplicate centrally so neither the generator nor a judge receives a
    # spuriously repeated cue.
    history_turns = list(state.current_session_history)
    if (
        history_turns
        and history_turns[-1].role == "user"
        and " ".join(history_turns[-1].content.split()).casefold()
        == " ".join(state.current_user_text.split()).casefold()
    ):
        history_turns = history_turns[:-1]
    history = "\n".join(f"{turn.role}: {turn.content}" for turn in history_turns)
    return (
        f"Current-session summary:\n{state.current_session_summary or '(none)'}\n\n"
        f"Recent dialogue:\n{history or '(none)'}\n\n"
        f"Current user message:\n{state.current_user_text}"
    )


def generation_messages(
    state: RuntimeState,
    memories: Sequence[MemoryItem],
    strategies: Sequence[StrategyCard],
    *,
    system_prompt: str = BASE_SUPPORTER_SYSTEM,
) -> list[dict[str, str]]:
    sections = [common_context(state)]
    if memories:
        memory_lines = []
        for item in memories:
            when = item.timestamp or f"session {item.created_session}"
            memory_lines.append(f"- [{item.source.value}; {when}] {item.text}")
        sections.append(
            "Potentially useful past information. Use selectively:\n"
            + "\n".join(memory_lines)
        )
    if strategies:
        strategy_lines = []
        for card in strategies:
            strategy_lines.append(
                f"- {card.guidance_text}\n  Example style (adapt, do not copy): "
                f"{card.example_response}"
            )
        sections.append(
            "Potential emotional-support guidance. Use only when fitting:\n"
            + "\n".join(strategy_lines)
        )
    sections.append("Write only the counselor's next response.")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


RESPONSE_JUDGE_SYSTEM = """You are a blind evaluator of two emotional-support
responses to the same visible dialogue. You do not know whether memory,
retrieval, or strategy guidance was used.

POSITION INDEPENDENCE (critical): The labels A and B are assigned randomly and
carry no information about quality. You must reach the same conclusion regardless
of which response appears as A or B. Before returning your verdict, ask yourself:
"If I swap A and B, would I still prefer the same underlying text?" If not,
choose 'tie'. Never prefer a response based on its position (first or second).

AUTHORIZED MEMORY FACTS: Some responses may mention a personal detail that is
not visible in the current-session dialogue (e.g. a past event, relationship, or
work situation). You cannot confirm whether this detail is accurate. Apply these
two rules strictly:
  (1) Do NOT award extra credit for referencing unseen personal history. A
      response that mentions an unverifiable past detail is NOT better than an
      equally warm and contextually fitting response that stays within the
      visible dialogue.
  (2) Do NOT penalize a response solely for referencing unseen personal history,
      UNLESS the reference is abrupt, intrusive, overconfident, or poorly fitted
      to the visible exchange.
In short: unseen personal details are neutral evidence. They neither help nor
hurt unless the integration quality itself differs.

Judge emotional-support quality, contextual fit, guidance fit, non-intrusiveness,
coherence, and empathy. Use 'tie' when quality is genuinely indistinguishable.
When in doubt about position effects, choose 'tie'.

REQUIRED JSON SCHEMA (return exactly this structure, all fields required):
{"preference":"A|B|tie","empathy":"A|B|tie","contextual_fit":"A|B|tie","guidance_fit":"A|B|tie","non_intrusiveness":"A|B|tie","coherence":"A|B|tie","reason":"one sentence"}
Return only JSON."""


def response_pair_messages(
    state: RuntimeState,
    response_a: str,
    response_b: str,
) -> list[dict[str, str]]:
    payload = {
        "shared_visible_context": common_context(state),
        "response_A": response_a,
        "response_B": response_b,
    }
    return [
        {"role": "system", "content": RESPONSE_JUDGE_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


M1_SYSTEM = """You are an evaluator-only auditor of long-term memory opportunity.
You may inspect all past memory items and timestamps. Determine whether each
item is relevant or potentially helpful now, whether newer information makes it
stale or conflicting, and whether mentioning it would be intrusive. Do not
infer anything from opaque IDs. You are not choosing a policy action and you
must not reward using more memory.

REQUIRED JSON SCHEMA (return exactly this structure, one entry per memory item):
{"items":[{"memory_id":"<id>","current_relevance":<0|1|2>,"potential_helpfulness":<0|1|2>,"stale":<true|false>,"conflicts_with_newer_information":<true|false>,"intrusive_if_mentioned":<true|false>}],"reason":"one sentence"}
Return only JSON."""


def memory_opportunity_messages(
    state: RuntimeState,
    all_items: Sequence[MemoryItem],
) -> list[dict[str, str]]:
    items = [
        {
            "memory_id": item.memory_id,
            "source": item.source.value,
            "created_session": item.created_session,
            "timestamp": item.timestamp,
            "text": item.text,
        }
        for item in all_items
    ]
    return [
        {"role": "system", "content": M1_SYSTEM},
        {
            "role": "user",
            "content": common_context(state)
            + "\n\nAll past memory items:\n"
            + json.dumps(items, ensure_ascii=False, indent=2),
        },
    ]


M2_SYSTEM = """You are an evaluator-only auditor of realized long-term-memory
use. You receive the complete authorized memory timeline, the subset selected
by the system, and the generated response. Evaluate only memory use and memory
omission; do not evaluate strategy guidance or general writing quality.

STALENESS: A memory item is stale when the timeline contains a newer entry that
updates, contradicts, or supersedes it. If the response uses such a stale fact
as though it were still current, set stale_or_conflicting_use=1.

GROUNDEDNESS: A memory item is grounded only when the response explicitly draws
on content from that item — using the fact, referencing the event, or tailoring
advice to the preference described. If the response is generic and makes no
discernible use of the selected memory, set utilization=0. Do not count the
item as grounded merely because it could have informed the response.

Use the full timeline to identify newer updates, contradictions, stale facts,
and important memories that the selected subset failed to cover. Return one
source assessment for every and only selected source. If multiple selected
memory snippets share the same source, aggregate them into exactly one
assessment for that source. Never return one assessment per memory item, and
never assess an unselected source merely because it appears in the full
timeline. Treat overgeneralizing a memory as misuse: turning a single past event into
"always/never" claims, deterministic predictions about future behaviour, or
causal certainty should raise unsupported_personal_claim. Do NOT raise
unsupported_personal_claim merely because the response uses cautious hedged
language ("it sounds like", "I remember you mentioned") — hedged references to
memory are appropriate professional practice. Only flag the field when the
response asserts an ungrounded fact as definite truth or fabricates a detail not
present in any selected or timeline item. Do not infer anything from opaque IDs
and do not reward selecting more sources. You do not receive any previous judge
score.

REQUIRED JSON SCHEMA (return exactly this structure, no extra keys):
{
  "source_assessments": [
    {
      "source": "<MP|MS|ME>",
      "utilization": <0|1|2>,
      "unused_retrieval": <0|1|2>,
      "unnecessary_exposure": <0|1|2>,
      "stale_or_conflicting_use": <0|1|2>,
      "unsupported_personal_claim": <0|1|2>
    }
  ],
  "overall_source_set_appropriateness": <0|1|2>,
  "reason": "<one sentence>"
}
One entry per selected source. No "reason" key inside source_assessments.
Return only JSON."""


def memory_use_messages(
    state: RuntimeState,
    selected_items: Sequence[MemoryItem],
    response: str,
    *,
    all_items: Sequence[MemoryItem] | None = None,
) -> list[dict[str, str]]:
    expected_sources = sorted({item.source.value for item in selected_items})
    selected = [
        {
            "memory_id": item.memory_id,
            "source": item.source.value,
            "created_session": item.created_session,
            "timestamp": item.timestamp,
            "text": item.text,
        }
        for item in selected_items
    ]
    timeline_items = list(all_items) if all_items is not None else list(selected_items)
    timeline = [
        {
            "memory_id": item.memory_id,
            "memory_source": item.source.value,
            "created_session": item.created_session,
            "timestamp": item.timestamp,
            "text": item.text,
            "selected": item.memory_id in {x.memory_id for x in selected_items},
        }
        for item in sorted(
            timeline_items,
            key=lambda x: (x.created_session, x.timestamp or "", x.memory_id),
        )
    ]
    return [
        {"role": "system", "content": M2_SYSTEM},
        {
            "role": "user",
            "content": common_context(state)
            + "\n\nExpected selected sources for source_assessments:\n"
            + json.dumps(expected_sources, ensure_ascii=False)
            + "\nReturn exactly "
            + str(len(expected_sources))
            + " source_assessments, one for each expected selected source and no others."
            + "\n\nComplete authorized memory timeline (selected is explicitly marked):\n"
            + json.dumps(timeline, ensure_ascii=False, indent=2)
            + "\n\nSelected memory snippets:\n"
            + json.dumps(selected, ensure_ascii=False, indent=2)
            + "\n\nGenerated response:\n"
            + response,
        },
    ]


M2B_SYSTEM = """You are an evaluator-only auditor of selected memory-source
coverage. This audit is only for non-M0 actions where the system selected at
least one long-term memory source. You receive the complete authorized memory
timeline, the selected memory subset, and the generated response.

Your task is narrow: decide whether the selected source set omitted another
available memory source (MP, MS, or ME) that was materially needed for this
current exchange. Do NOT evaluate general writing quality, strategy guidance,
or whether the response is pleasant. Do NOT reward selecting more sources.

Scoring:
- selected_set_sufficiency: 2 = selected sources were sufficient; 1 = partly
  sufficient but missed optional useful context; 0 = insufficient because an
  omitted source was materially needed.
- selected_set_omission_severity: 0 = no material omitted source; 1 = omitted
  source may have helped but was optional; 2 = omitted source was important
  enough that the memory decision was incomplete or misleading.
- missed_useful_sources: list only omitted sources from MP/MS/ME. Do not list
  a selected source. Use an empty list when severity is 0.

A source is materially needed only when it would improve accuracy,
appropriate personalization, stale/conflict handling, or understanding of the
current user message. A source is not needed merely because it exists.

REQUIRED JSON SCHEMA (return exactly this structure, no extra keys):
{
  "selected_set_sufficiency": <0|1|2>,
  "selected_set_omission_severity": <0|1|2>,
  "missed_useful_sources": ["<MP|MS|ME>", "..."],
  "reason": "<one sentence>"
}
Return only JSON."""


def memory_selected_set_omission_messages(
    state: RuntimeState,
    selected_items: Sequence[MemoryItem],
    response: str,
    *,
    all_items: Sequence[MemoryItem],
) -> list[dict[str, str]]:
    selected_ids = {item.memory_id for item in selected_items}
    selected_sources = sorted({item.source.value for item in selected_items})
    available_sources = sorted({item.source.value for item in all_items})
    timeline = [
        {
            "memory_id": item.memory_id,
            "memory_source": item.source.value,
            "created_session": item.created_session,
            "timestamp": item.timestamp,
            "text": item.text,
            "selected": item.memory_id in selected_ids,
        }
        for item in sorted(
            all_items,
            key=lambda x: (x.created_session, x.timestamp or "", x.memory_id),
        )
    ]
    return [
        {"role": "system", "content": M2B_SYSTEM},
        {
            "role": "user",
            "content": common_context(state)
            + "\n\nAvailable memory sources:\n"
            + json.dumps(available_sources, ensure_ascii=False)
            + "\n\nSelected memory sources:\n"
            + json.dumps(selected_sources, ensure_ascii=False)
            + "\nmissed_useful_sources must be a subset of available sources "
            + "that are NOT selected."
            + "\n\nComplete authorized memory timeline (selected is explicitly marked):\n"
            + json.dumps(timeline, ensure_ascii=False, indent=2)
            + "\n\nGenerated response:\n"
            + response,
        },
    ]


M0_SYSTEM = """You are an evaluator-only auditor of the decision to omit
long-term memory. Inspect the visible dialogue, all available past memory items,
and the generated no-memory response. A relevant memory is not automatically
necessary: omission is wrong only when using memory would materially improve
accuracy, understanding, or appropriate personalization. Also detect invented
personal facts.

REQUIRED JSON SCHEMA (return exactly this structure, all fields required):
{"omission_appropriateness":<0|1|2>,"missed_memory_opportunity_severity":<0|1|2>,"unsupported_personal_claim":<0|1|2>,"reason":"one sentence"}
Return only JSON."""


def memory_omission_messages(
    state: RuntimeState,
    all_items: Sequence[MemoryItem],
    response: str,
) -> list[dict[str, str]]:
    items = [
        {
            "memory_id": item.memory_id,
            "source": item.source.value,
            "created_session": item.created_session,
            "timestamp": item.timestamp,
            "text": item.text,
        }
        for item in all_items
    ]
    return [
        {"role": "system", "content": M0_SYSTEM},
        {
            "role": "user",
            "content": common_context(state)
            + "\n\nAll available past memory items:\n"
            + json.dumps(items, ensure_ascii=False, indent=2)
            + "\n\nNo-memory response:\n"
            + response,
        },
    ]


STRATEGY_SYSTEM = """You are an evaluator-only auditor of selected
emotional-support strategy guidance. Do not evaluate long-term memory and do not
assign an overall response-quality score. Judge whether the guidance was
relevant, reflected in the response, over-structured, or encouraged advice too
early.

Scoring rules:
- strategy_relevance: 0 = the selected guidance does not fit the visible user
  need; 1 = partially/generally relevant; 2 = clearly relevant.
- strategy_utilization: 0 = the response does not reflect the selected
  guidance; 1 = partially reflects it; 2 = clearly reflects it.
- If your reason says the response aligned with, followed, reflected, or used
  the guidance, strategy_relevance and strategy_utilization must not be 0.
- If the guidance is "reflect feelings before advice" and the response names or
  mirrors the user's feeling before moving on, utilization should be 1 or 2.

REQUIRED JSON SCHEMA (return exactly this structure, all fields required):
{"strategy_relevance":<0|1|2>,"strategy_utilization":<0|1|2>,"over_structuring":<0|1|2>,"premature_advice":<0|1|2>,"reason":"one sentence"}
Return only JSON."""


def strategy_use_messages(
    state: RuntimeState,
    strategies: Sequence[StrategyCard],
    response: str,
) -> list[dict[str, str]]:
    cards = [
        {
            "guidance": card.guidance_text,
            "example": card.example_response,
        }
        for card in strategies
    ]
    return [
        {"role": "system", "content": STRATEGY_SYSTEM},
        {
            "role": "user",
            "content": common_context(state)
            + "\n\nSelected strategy guidance:\n"
            + json.dumps(cards, ensure_ascii=False, indent=2)
            + "\n\nGenerated response:\n"
            + response,
        },
    ]


STRATEGY_OMISSION_SYSTEM = """You are an evaluator-only auditor of the
decision not to retrieve emotional-support strategy guidance. Inspect the
visible dialogue and the generated R0 response. Omission is inappropriate only
when external strategy guidance would materially improve sequencing, empathy,
or advice timing; do not penalize a natural response merely for not using a
named strategy.

REQUIRED JSON SCHEMA (return exactly this structure, all fields required):
{"strategy_omission_appropriateness":<0|1|2>,"missed_strategy_opportunity_severity":<0|1|2>,"premature_or_overstructured_without_strategy":<0|1|2>,"reason":"one sentence"}
Return only JSON."""


def strategy_omission_messages(
    state: RuntimeState,
    response: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": STRATEGY_OMISSION_SYSTEM},
        {
            "role": "user",
            "content": common_context(state)
            + "\n\nResponse generated without Strategy RAG:\n"
            + response,
        },
    ]


OFFICIAL_DIALOGUE_JUDGE_SYSTEM = """You are an evaluator-only expert assessing
a long-term emotional-support dialogue. You receive authorized ground truth
(profile, prior sessions, and the current private scenario) plus the transcript.
The system condition, retrieval action, and selected snippets are hidden.

Rate 1–5 integers. Memory measures accurate and useful integration of prior
experiences. Personalization measures tailoring to verified user information.
Emotional support measures empathy and appropriate help. Factual grounding
penalizes invented personal details. Temporal consistency penalizes reliance on
older facts contradicted by newer updates. Do not reward mentioning more facts
for its own sake. Return only JSON."""


def official_dialogue_score_messages(
    dialogue: list[dict[str, str]],
    evaluator_context: Mapping[str, Any],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": OFFICIAL_DIALOGUE_JUDGE_SYSTEM},
        {"role": "user", "content": json.dumps({
            "authorized_ground_truth": evaluator_context,
            "dialogue": dialogue,
        }, ensure_ascii=False, indent=2)},
    ]


FIXED_CONTEXT_SCORE_SYSTEM = """You are an evaluator-only expert assessing
multiple independent, fixed-context emotional-support responses. Every case
contains the complete policy-independent dialogue context immediately before a
seeker turn, that seeker turn, and one candidate supporter response. Evaluated
responses were not fed into later cases, so do not infer a closed-loop
trajectory or reward cross-case narrative continuity.

Use the authorized profile/history/topic only to verify factual and temporal
grounding. Rate aggregate 1–5 integers for memory, personalization, emotional
support, factual grounding, and temporal consistency. Do not reward mentioning
more personal facts. Return only JSON."""


def fixed_context_score_messages(
    turns: Sequence[Mapping[str, Any]],
    evaluator_context: Mapping[str, Any],
) -> list[dict[str, str]]:
    cases = [
        {
            "turn_index": int(turn["turn_index"]),
            "context_before_turn": turn.get("context_before_turn") or [],
            "current_seeker_message": turn["seeker_message"],
            "supporter_response": turn["supporter_message"],
        }
        for turn in turns
    ]
    return [
        {"role": "system", "content": FIXED_CONTEXT_SCORE_SYSTEM},
        {"role": "user", "content": json.dumps({
            "authorized_ground_truth": evaluator_context,
            "independent_fixed_context_cases": cases,
        }, ensure_ascii=False, indent=2)},
    ]


FIXED_CONTEXT_PAIR_SYSTEM = """You are an evaluator-only blind comparison
judge for two systems on the same bundle of independent fixed-context emotional-
support cases. In every case, both systems saw exactly the same complete context
and current seeker message. Their replies were not fed into later cases. Compare
emotional support, useful personalization, memory appropriateness, non-
intrusiveness, factual grounding, and temporal consistency. Do not judge the
bundle as a closed-loop conversation and do not reward mentioning more facts.
Return only JSON."""


def fixed_context_pair_messages(
    turns_a: Sequence[Mapping[str, Any]],
    turns_b: Sequence[Mapping[str, Any]],
    evaluator_context: Mapping[str, Any],
) -> list[dict[str, str]]:
    if len(turns_a) != len(turns_b):
        raise ValueError("fixed-context systems have different turn counts")
    cases = []
    for left, right in zip(turns_a, turns_b):
        left_context = left.get("context_before_turn") or []
        right_context = right.get("context_before_turn") or []
        if (
            int(left["turn_index"]) != int(right["turn_index"])
            or left_context != right_context
            or left["seeker_message"] != right["seeker_message"]
        ):
            raise ValueError("fixed-context comparison inputs differ across systems")
        cases.append({
            "turn_index": int(left["turn_index"]),
            "context_before_turn": left_context,
            "current_seeker_message": left["seeker_message"],
            "response_A": left["supporter_message"],
            "response_B": right["supporter_message"],
        })
    return [
        {"role": "system", "content": FIXED_CONTEXT_PAIR_SYSTEM},
        {"role": "user", "content": json.dumps({
            "authorized_ground_truth": evaluator_context,
            "independent_fixed_context_cases": cases,
        }, ensure_ascii=False, indent=2)},
    ]


OBSERVATION_RELEVANCE_SYSTEM = """Score how helpful a past observation is for
responding to the current seeker message. Use exactly 0.0 for irrelevant, 0.5
for partially helpful but unnecessary, and 1.0 for clearly relevant and
meaningfully helpful. Return only JSON."""


def observation_relevance_messages(
    seeker_message: str,
    observation: str,
    current_session_history: Sequence[Mapping[str, str]] = (),
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": OBSERVATION_RELEVANCE_SYSTEM},
        {"role": "user", "content": json.dumps({
            "seeker_message": seeker_message,
            "past_observation": observation,
            "current_session_before_response": list(current_session_history),
        }, ensure_ascii=False)},
    ]


OBSERVATION_USAGE_SYSTEM = """Determine whether the supporter response uses the
supplied past observation. You also receive the complete current-session
dialogue before this response. Mark already_disclosed_in_current_session true
when the same information was revealed earlier in this session. Mark
attributable_to_long_term_memory true only when the response uses observation
content that was not already available from the current session. Do not count a
generic response as use. Return only JSON."""


def observation_usage_messages(
    seeker_message: str,
    supporter_message: str,
    observation: str,
    current_session_history: Sequence[Mapping[str, str]] = (),
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": OBSERVATION_USAGE_SYSTEM},
        {"role": "user", "content": json.dumps({
            "seeker_message": seeker_message,
            "supporter_message": supporter_message,
            "past_observation": observation,
            "current_session_before_response": list(current_session_history),
        }, ensure_ascii=False)},
    ]


DIALOGUE_PAIR_SYSTEM = """You are an evaluator-only blind system-comparison
judge. You receive the same authorized profile, prior-session timeline, and
private scenario for two dialogues, but you do not know either policy,
retrieval condition, or action. Compare emotional support, personalization,
memory appropriateness, non-intrusiveness, coherence, factual grounding, and
temporal consistency. Do not reward mentioning more facts. Penalize invented,
stale, forced, or intrusive references. Return only JSON."""


def dialogue_pair_messages(
    dialogue_a: list[dict[str, str]],
    dialogue_b: list[dict[str, str]],
    evaluator_context: Mapping[str, Any],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DIALOGUE_PAIR_SYSTEM},
        {"role": "user", "content": json.dumps({
            "authorized_ground_truth": evaluator_context,
            "dialogue_A": dialogue_a,
            "dialogue_B": dialogue_b,
        }, ensure_ascii=False, indent=2)},
    ]
