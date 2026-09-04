from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import Field, field_validator, model_validator

from .artifacts import create_artifact_attestation, require_artifact_attestation
from .contracts import DialogueTurn, MemorySource, StrategyCard, StrictModel
from .io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
)


V8_PROTOCOL = "pm-v2-generation-semantic-review-v8-marginal-value-12d"
V8_REVIEW_STAGE = "pm_v2_generation_semantic_review_v8"
V8_AUDIT_STAGE = "pm_v2_generation_semantic_leakage_audit_v8"
MINIMUM_ANNOTATORS = 2
SMOKE_CASES_PER_REGIME = 1
VALIDATION_CASES_PER_REGIME = 3
# Strategy targets remain hypotheses until the inherited V1 card bank and the
# production retriever have been audited and frozen.
V1_STRATEGY_RAG_AUDIT_COMPLETE = False
PROVISIONAL_STATUS = "PROVISIONAL_NOT_READY_FOR_ANNOTATION_OR_TRAINING"

V8_REGIMES = (
    "context_only",
    "profile_useful",
    "summary_useful",
    "event_useful",
    "multi_source_useful",
    "memory_harmful",
    "strategy_helpful",
    "advice_harmful",
    "ambiguous",
)

RATING_FIELDS = (
    "semantic_family_match",
    "regime_match",
    "memory_sources_marginal_value_match",
    "memory_item_utility_match",
    "source_type_match",
    "dialogue_temporal_order_match",
    "context_grounding_match",
    "memory_age_design_match",
    "strategy_resource_need_match",
    "strategy_item_utility_match",
    "advice_readiness_match",
    "surface_naturalness_match",
)

REVIEW_QUESTIONS_ZH = {
    "semantic_family_match": "当前话语、历史和 summary 是否属于同一广义语义家庭，并允许合理子话题？",
    "regime_match": "结合完整资源候选，这个候选资源情形是否成立？",
    "memory_sources_marginal_value_match": "列出的 memory 来源是否提供非重复且实质性的边际价值，未列出的来源是否无明显价值或有风险？",
    "memory_item_utility_match": "每条 MP/MS/ME 的 helpful、irrelevant、harmful 标签是否与文本和当前状态一致？",
    "source_type_match": "MP 是否为稳定偏好/属性，MS 是否为跨 session 模式，ME 是否为具体过去事件？",
    "dialogue_temporal_order_match": "历史对话是否严格早于当前轮、角色交替正常，且没有未来信息或复制当前话语？",
    "context_grounding_match": "summary、authorized context 和 current correction 是否都可追溯到可见对话或有 provenance 的外部更新？",
    "memory_age_design_match": "memory age 与行顺序是否有效、可复现，且没有直接泄漏 utility 或 regime？",
    "strategy_resource_need_match": "根据基础生成器能力和实际候选卡，开启、关闭或暂不确定 RS 的边际价值判断是否合理？",
    "strategy_item_utility_match": "每张策略卡的 helpful、irrelevant、harmful 标签是否适合当前对话？",
    "advice_readiness_match": "当前适合倾听、探索、轻量建议还是结构化计划的标签是否正确？",
    "surface_naturalness_match": "用户话语、历史和 summary 是否自然，且没有暴露 regime、utility 或数据生成意图？",
}

REVIEW_QUESTIONS_EN = {
    "semantic_family_match": "Do the current turn, history, and summary share one broad semantic family while allowing reasonable subtopics?",
    "regime_match": "Does the proposed resource situation hold when all candidate evidence is considered?",
    "memory_sources_marginal_value_match": "Do listed memory sources add material, nonredundant value, while omitted sources add no clear value or create risk?",
    "memory_item_utility_match": "Is each MP/MS/ME helpful, irrelevant, or harmful label supported by its text and the current state?",
    "source_type_match": "Is MP a stable preference/profile fact, MS a cross-session pattern, and ME a concrete past event?",
    "dialogue_temporal_order_match": "Is all history prior to the current turn, naturally alternating, and free of future or copied-current information?",
    "context_grounding_match": "Can every summary, authorized-context, and correction claim be traced to visible dialogue or a provenance-bearing external update?",
    "memory_age_design_match": "Are ages and row order valid and reproducible without directly leaking utility or regime?",
    "strategy_resource_need_match": "Given base-generator capability and the displayed cards, is the marginal-value decision to use, skip, or remain uncertain about RS reasonable?",
    "strategy_item_utility_match": "Is each strategy card's helpful, irrelevant, or harmful label appropriate for this dialogue?",
    "advice_readiness_match": "Is listen-only, exploration, light suggestion, structured planning, or ambiguity the right level of directiveness?",
    "surface_naturalness_match": "Are the user turn, history, and summary natural and free of regime, utility, or data-generation leakage?",
}

PROHIBITED_SURFACE_PHRASES = (
    "recurring and personal",
    "one detail alone does not explain",
    "i need multiple memory sources",
    "helpful memory",
    "harmful memory",
    "strategy is needed",
    "context only is enough",
    "authorized current context about",
    "candidate semantic family",
    "regime",
)

# These IDs point into the inherited V1 bank; no new cards are created here.
# Card selection and utility labels are experimental hypotheses pending the V1
# bank/retriever audit and must not be treated as frozen evidence.
STRATEGY_CARD_IDS = {
    "gentle_question": "strat_47334c3b32ccbec779d6",
    "open_restatement": "strat_642845a2daee6358bfcc",
    "stress_reflection": "strat_284b8c8dab5181fbd3fe",
    "exam_reflection": "strat_b2153382385d3a2c2a35",
    "one_problem_suggestion": "strat_4d1c7e2473c3b2decf5e",
    "social_connection_suggestion": "strat_21d8cc822a6296e6a98e",
    "low_pressure_connection": "strat_27d35e6ee25862ab2ace",
    "intrusive_long_plan": "strat_5c3cb487bb7e0cda1268",
    "video_call_self_disclosure": "strat_b15e7d1997231f6b4961",
    "support_affirmation": "strat_aeb4657acbd1ed58969c",
    "job_information": "strat_c84c2e608e09e596154a",
}


class V8MemoryEvidence(StrictModel):
    memory_id: str
    source: MemorySource
    text: str = Field(min_length=1)
    created_session: int = Field(ge=0)
    age_sessions: int = Field(ge=1)
    utility: Literal["helpful", "irrelevant", "harmful"]
    stale: bool
    conflicts_with_current_state: bool
    private_sensitivity: Literal["ordinary", "sensitive"]
    provenance_type: Literal[
        "profile_record", "cross_session_summary", "past_event_record"
    ]
    provenance_source: str = Field(min_length=1)
    provenance_session: int = Field(ge=0)

    @field_validator("source", mode="before")
    @classmethod
    def parse_source(cls, value):
        return value if isinstance(value, MemorySource) else MemorySource(str(value))

    @model_validator(mode="after")
    def coherent(self):
        expected = {
            MemorySource.MP: "profile_record",
            MemorySource.MS: "cross_session_summary",
            MemorySource.ME: "past_event_record",
        }[self.source]
        if self.provenance_type != expected:
            raise ValueError(f"{self.source.value} requires {expected} provenance")
        if self.created_session != self.provenance_session:
            raise ValueError("memory creation and provenance sessions differ")
        if self.utility == "helpful" and (
            self.stale or self.conflicts_with_current_state
        ):
            raise ValueError("helpful memory cannot be stale or conflicting")
        return self


class V8ContextProvenance(StrictModel):
    target_field: Literal[
        "session_summary", "authorized_user_context", "current_correction"
    ]
    claim: str = Field(min_length=1)
    provenance_type: Literal[
        "current_user", "dialogue_turn", "external_state_update"
    ]
    provenance_source: str = Field(min_length=1)
    provenance_session: int = Field(ge=0)
    evidence_quote: str = Field(min_length=1)


class V8StrategyTarget(StrictModel):
    use_strategy_rag: Literal["on", "off", "ambiguous"]
    advice_readiness: Literal[
        "listen_only",
        "explore_first",
        "light_suggestion",
        "structured_plan",
        "ambiguous",
    ]
    base_generator_assessment: str = Field(min_length=1)
    marginal_value_rationale: str = Field(min_length=1)


class V8StrategyEvidence(StrictModel):
    card_id: str
    strategy_type: str = Field(min_length=1)
    utility: Literal["helpful", "irrelevant", "harmful"]
    card_text: str = Field(min_length=1)
    retrieval_text: str = Field(min_length=1)
    guidance_text: str = Field(min_length=1)
    marginal_value_rationale: str = Field(min_length=1)
    source_dialogue_id: str = Field(min_length=1)
    source_turn_index: int = Field(ge=0)


class V8ReviewCase(StrictModel):
    item_id: str
    case_id: str
    variant_index: int = Field(ge=1, le=VALIDATION_CASES_PER_REGIME)
    semantic_family: str = Field(min_length=1)
    semantic_subtopics: list[str] = Field(min_length=1)
    regime: Literal[
        "context_only",
        "profile_useful",
        "summary_useful",
        "event_useful",
        "multi_source_useful",
        "memory_harmful",
        "strategy_helpful",
        "advice_harmful",
        "ambiguous",
    ]
    current_user_text: str = Field(min_length=1)
    dialogue_before_current: list[DialogueTurn] = Field(min_length=2)
    session_summary: str = Field(min_length=1)
    authorized_user_context: str = Field(min_length=1)
    current_correction: str | None = None
    context_provenance: list[V8ContextProvenance] = Field(min_length=2)
    coverage_rationale: str = Field(min_length=1)
    session_index: int = Field(ge=1)
    materially_useful_memory_sources: list[MemorySource]
    profile_memories: list[V8MemoryEvidence] = Field(min_length=2)
    summary_memories: list[V8MemoryEvidence] = Field(min_length=2)
    event_memories: list[V8MemoryEvidence] = Field(min_length=2)
    strategy_target: V8StrategyTarget
    strategy_evidence: list[V8StrategyEvidence] = Field(min_length=2)
    generation_seed: int
    age_sampling_protocol: str = Field(min_length=1)
    row_shuffle_protocol: str = Field(min_length=1)

    @property
    def needed_memory_sources(self) -> list[MemorySource]:
        """Legacy bridge; V8 semantics are material marginal value, not necessity."""

        return list(self.materially_useful_memory_sources)

    @field_validator("materially_useful_memory_sources")
    @classmethod
    def unique_sources(cls, value: list[MemorySource]) -> list[MemorySource]:
        if len(value) != len(set(value)):
            raise ValueError("materially useful memory sources must be unique")
        return value

    @field_validator("materially_useful_memory_sources", mode="before")
    @classmethod
    def parse_material_sources(cls, value):
        if not isinstance(value, list):
            raise TypeError("materially useful memory sources must be a list")
        return [
            source
            if isinstance(source, MemorySource)
            else MemorySource(str(source))
            for source in value
        ]

    @model_validator(mode="after")
    def coherent(self):
        if self.dialogue_before_current[-1].role != "assistant":
            raise ValueError("dialogue history must end with assistant")
        if any(
            self.dialogue_before_current[index - 1].role
            == self.dialogue_before_current[index].role
            for index in range(1, len(self.dialogue_before_current))
        ):
            raise ValueError("dialogue roles must alternate")
        if any(
            turn.content.strip().casefold() == self.current_user_text.strip().casefold()
            for turn in self.dialogue_before_current
        ):
            raise ValueError("history cannot copy current user text")
        pools = {
            MemorySource.MP: self.profile_memories,
            MemorySource.MS: self.summary_memories,
            MemorySource.ME: self.event_memories,
        }
        for source, memories in pools.items():
            if any(item.source is not source for item in memories):
                raise ValueError(f"wrong source in {source.value} pool")
            if any(
                item.created_session >= self.session_index
                or item.age_sessions != self.session_index - item.created_session
                for item in memories
            ):
                raise ValueError(f"invalid age in {source.value} pool")
        helpful_sources = {
            source
            for source, memories in pools.items()
            if any(item.utility == "helpful" for item in memories)
        }
        if helpful_sources != set(self.materially_useful_memory_sources):
            raise ValueError(
                "helpful memory sources must equal materially useful sources"
            )
        if self.regime == "memory_harmful" and not any(
            item.utility == "harmful"
            for memories in pools.values()
            for item in memories
        ):
            raise ValueError("memory_harmful requires harmful evidence")
        if self.regime == "advice_harmful":
            if self.strategy_target.advice_readiness != "listen_only":
                raise ValueError("advice_harmful must be listen_only")
            if not any(
                item.utility == "helpful"
                and item.strategy_type
                in {"Reflection of feelings", "Restatement or Paraphrasing"}
                for item in self.strategy_evidence
            ):
                raise ValueError("advice_harmful needs a helpful listening card")
            if not any(item.utility == "harmful" for item in self.strategy_evidence):
                raise ValueError("advice_harmful needs a harmful directive card")
        return self


def _seeded_rng(seed: int, *parts: object) -> random.Random:
    digest = sha256_text("|".join([str(seed), *map(str, parts)]))
    return random.Random(int(digest[:16], 16))


def load_strategy_catalog(
    path: str | Path,
    *,
    strategy_card_ids: Mapping[str, str] | None = None,
) -> dict[str, StrategyCard]:
    role_map = dict(strategy_card_ids or STRATEGY_CARD_IDS)
    if set(role_map) != set(STRATEGY_CARD_IDS):
        raise RuntimeError("V8 strategy-card role map is incomplete")
    wanted = set(role_map.values())
    cards: dict[str, StrategyCard] = {}
    for row in iter_jsonl(path):
        card_id = str(row.get("strategy_id") or "")
        if card_id in wanted:
            cards[card_id] = StrategyCard.model_validate(row)
    missing = sorted(wanted - set(cards))
    if missing:
        raise RuntimeError(f"strategy bank lacks frozen V8 cards: {missing}")
    return cards


def _surface_variants() -> dict[str, list[dict[str, Any]]]:
    """Natural, source-grounded smoke/validation surfaces; none are V7 text."""

    return {
        "context_only": [
            {
                "family": "relocation_loneliness",
                "subtopics": ["quiet_evenings", "missing_familiar_people"],
                "dialogue": [
                    ("user", "I unpacked another box after work, but the apartment still does not feel like mine."),
                    ("assistant", "Settling in can take longer than the unpacking. What has felt hardest so far?"),
                ],
                "current": "Evenings have been the hardest since the move. Once it gets quiet, I really miss hearing familiar voices.",
                "summary": "The user recently moved and finds the quiet evenings lonely.",
                "authorized": "Use only the visible facts that the move and quiet evenings are connected to the user's loneliness.",
            },
            {
                "family": "relocation_loneliness",
                "subtopics": ["weekend_isolation", "unfamiliar_neighborhood"],
                "dialogue": [
                    ("user", "I walked around the neighborhood this morning, but everything still felt unfamiliar."),
                    ("assistant", "That unfamiliarity can make a new place feel especially isolating."),
                ],
                "current": "The empty weekend is getting to me more than I expected. I keep thinking about how easy it used to be to call someone nearby.",
                "summary": "The user is adjusting to an unfamiliar neighborhood and feels isolated during an empty weekend.",
                "authorized": "Use the visible adjustment and weekend isolation; do not invent local relationships or activities.",
            },
            {
                "family": "relocation_loneliness",
                "subtopics": ["new_home", "evening_silence"],
                "dialogue": [
                    ("user", "I finally put up a few photos from home today."),
                    ("assistant", "It sounds like you are trying to make the new place feel more familiar."),
                ],
                "current": "The photos help a little, but the silence after dinner still makes me feel far away from everyone.",
                "summary": "The user is making a new home feel familiar but still feels distant from others in the evening.",
                "authorized": "Use the visible photos, evening silence, and feeling of distance without adding assumptions.",
            },
        ],
        "profile_useful": [
            {
                "family": "academic_pressure",
                "subtopics": ["exam_anxiety", "support_style"],
                "dialogue": [
                    ("user", "I looked at the practice exam and my mind went blank."),
                    ("assistant", "That sounds frightening, especially with the exam getting closer."),
                ],
                "current": "My exam is next Friday and my brain is racing. I am not ready to turn this into a study schedule yet.",
                "summary": "The user feels anxious about an exam next Friday and is not ready to plan studying yet.",
                "authorized": "Use the visible exam timing and the user's request to avoid moving straight into a schedule.",
            },
            {
                "family": "academic_pressure",
                "subtopics": ["presentation_anxiety", "support_style"],
                "dialogue": [
                    ("user", "I stumbled through the practice presentation in front of my roommate."),
                    ("assistant", "A rough practice can make the real presentation feel much closer and scarier."),
                ],
                "current": "The presentation is in four days. Could we slow down and sort out what is scaring me before we talk about rehearsal tips?",
                "summary": "The user is anxious about a presentation in four days and wants reflection before practical tips.",
                "authorized": "Use the visible deadline and stated preference to reflect before offering rehearsal advice.",
            },
            {
                "family": "academic_pressure",
                "subtopics": ["assignment_overwhelm", "support_style"],
                "dialogue": [
                    ("user", "I opened the assignment brief three times tonight and closed it each time."),
                    ("assistant", "It sounds as if even looking at it is bringing up a lot of pressure."),
                ],
                "current": "I know I will have to make a plan later, but right now I need help naming why this assignment feels so impossible.",
                "summary": "The user feels overwhelmed by an assignment and wants to understand the feeling before planning.",
                "authorized": "Use the visible overwhelm and the user's sequencing preference: reflection now, planning later.",
            },
        ],
        "summary_useful": [
            {
                "family": "workplace_stress",
                "subtopics": ["coworker_conflict", "repeated_misunderstanding"],
                "dialogue": [
                    ("user", "A coworker corrected me in front of the team again today."),
                    ("assistant", "Being corrected publicly can feel exposing. What stayed with you afterward?"),
                ],
                "current": "I keep replaying the conversation and wondering whether I am overreacting or whether something between us really is not working.",
                "summary": "The user is distressed after a coworker corrected them publicly and is unsure how to interpret the relationship.",
                "authorized": "Use the visible public correction and uncertainty; a cross-session pattern may be used only if it appears in MS evidence.",
            },
            {
                "family": "workplace_stress",
                "subtopics": ["manager_feedback", "repeated_tension"],
                "dialogue": [
                    ("user", "My manager rewrote my update in the meeting without asking me."),
                    ("assistant", "That may have left you feeling dismissed in front of everyone."),
                ],
                "current": "I cannot tell if this was one awkward meeting or if I have been feeling brushed aside for a while.",
                "summary": "The user felt dismissed when a manager rewrote their update and is unsure whether it reflects a larger pattern.",
                "authorized": "Use the visible meeting and uncertainty; only MS evidence may establish a cross-session pattern.",
            },
            {
                "family": "workplace_stress",
                "subtopics": ["team_communication", "repeated_misunderstanding"],
                "dialogue": [
                    ("user", "My teammate answered a question that had been directed to me."),
                    ("assistant", "That interruption sounds frustrating and possibly undermining."),
                ],
                "current": "Part of me wants to dismiss it, but another part feels tired in a way that seems bigger than today.",
                "summary": "The user is frustrated after a teammate spoke over them and wonders whether the fatigue extends beyond today.",
                "authorized": "Use today's interruption and the user's uncertainty; rely on MS, not inference, for any repeated pattern.",
            },
        ],
        "event_useful": [
            {
                "family": "relocation_loneliness",
                "subtopics": ["social_connection", "low_pressure_step"],
                "dialogue": [
                    ("user", "I passed the community center again and almost went inside."),
                    ("assistant", "Almost going in suggests part of you wants connection even though it feels difficult."),
                ],
                "current": "I do want to meet someone here. Could we think of one low-pressure thing I might try this week?",
                "summary": "The user wants local connection and asks for one low-pressure step this week.",
                "authorized": "Use the visible community-center hesitation and request for one low-pressure social step.",
            },
            {
                "family": "relocation_loneliness",
                "subtopics": ["social_connection", "small_action"],
                "dialogue": [
                    ("user", "There is a notice for a neighborhood book exchange in the lobby."),
                    ("assistant", "It caught your attention, though approaching new people may still feel like a lot."),
                ],
                "current": "I would like to stop avoiding everything. What is one small way I could test the waters without committing to a whole event?",
                "summary": "The user noticed a neighborhood activity and asks for one small, low-commitment social step.",
                "authorized": "Use the visible notice and request for a small step without assuming the user will attend an event.",
            },
            {
                "family": "relocation_loneliness",
                "subtopics": ["social_connection", "gentle_suggestion"],
                "dialogue": [
                    ("user", "I exchanged a few words with someone at the corner shop today."),
                    ("assistant", "That brief exchange seems to have mattered to you."),
                ],
                "current": "It made the neighborhood feel less closed off. Can you help me think of a similarly small next step?",
                "summary": "A brief local interaction helped, and the user asks for another similarly small step.",
                "authorized": "Use the visible shop interaction and request for a similarly small next step.",
            },
        ],
        "multi_source_useful": [
            {
                "family": "academic_pressure",
                "subtopics": ["task_freeze", "repeated_overload", "past_coping"],
                "dialogue": [
                    ("user", "I opened my notes after dinner and just stared at the first page."),
                    ("assistant", "Freezing like that can be upsetting when you were trying to get started."),
                ],
                "current": "I ended up closing them again. I keep landing in this same stuck place even when I try to begin earlier.",
                "summary": "The user froze while trying to study and feels repeatedly stuck despite trying to start earlier.",
                "authorized": "Use the visible study attempt and repeated feeling; MP/MS/ME add only what their displayed records support.",
            },
            {
                "family": "workplace_stress",
                "subtopics": ["task_freeze", "competing_deadlines", "past_coping"],
                "dialogue": [
                    ("user", "I switched between three work documents and finished none of them."),
                    ("assistant", "Being pulled in several directions seems to have left you frozen."),
                ],
                "current": "I am embarrassed that this keeps happening when deadlines pile up. I do not know how to talk about it without feeling judged.",
                "summary": "The user froze between competing work tasks and fears judgment because the problem has happened repeatedly.",
                "authorized": "Use the visible task freeze and fear of judgment; additional pattern and coping facts require displayed memory evidence.",
            },
            {
                "family": "academic_pressure",
                "subtopics": ["competing_assignments", "repeated_overload", "past_coping"],
                "dialogue": [
                    ("user", "I moved from the lab report to the reading and then to my slides."),
                    ("assistant", "It sounds as though the number of demands made it hard to stay with any one of them."),
                ],
                "current": "Now I feel guilty and stuck. This is not the first time several deadlines have made me shut down.",
                "summary": "Several academic deadlines led the user to switch tasks, shut down, and feel guilty.",
                "authorized": "Use the visible competing deadlines and shutdown; any support preference or past successful response needs memory evidence.",
            },
        ],
        "memory_harmful": [
            {
                "family": "workplace_stress",
                "subtopics": ["coworker_conflict", "current_preference_correction"],
                "dialogue": [
                    ("user", "My coworker dismissed my concern in the meeting this afternoon."),
                    ("assistant", "That sounds painful, especially in front of the rest of the team."),
                ],
                "current": "I used to avoid talking about conflict at work, but this time I do not want to run from it. I want to say clearly what happened.",
                "summary": "After being dismissed in a meeting, the user says they no longer want to avoid workplace conflict and want to describe what happened.",
                "authorized": "Use the visible correction that the user now wants to discuss the conflict rather than avoid it.",
                "correction": "The user explicitly replaces an old avoidance preference with a wish to discuss this conflict clearly.",
            },
            {
                "family": "workplace_stress",
                "subtopics": ["manager_conflict", "current_preference_correction"],
                "dialogue": [
                    ("user", "My manager spoke over me when I tried to explain the delay."),
                    ("assistant", "Being cut off may have made an already tense conversation harder."),
                ],
                "current": "In the past I would have changed the subject, but I do not want that now. I need room to talk through the exchange directly.",
                "summary": "The user was cut off by a manager and explicitly wants to discuss the exchange instead of changing the subject.",
                "authorized": "Use the current preference to discuss the exchange directly; do not apply an older avoidance preference.",
                "correction": "The user explicitly rejects changing the subject and asks to discuss the exchange directly.",
            },
            {
                "family": "workplace_stress",
                "subtopics": ["team_conflict", "current_preference_correction"],
                "dialogue": [
                    ("user", "A teammate blamed me for a missed handoff in front of everyone."),
                    ("assistant", "That public blame sounds upsetting and unfair."),
                ],
                "current": "I normally keep these things to myself, but I do not want to minimize this one. I want to put the details into words.",
                "summary": "The user was blamed publicly and explicitly wants to describe the event rather than keep it private or minimize it.",
                "authorized": "Use the visible update that the user wants to put this event into words now.",
                "correction": "The user explicitly replaces keeping conflict private with a wish to describe this event.",
            },
        ],
        "strategy_helpful": [
            {
                "family": "relocation_loneliness",
                "subtopics": ["social_connection", "gentle_action"],
                "dialogue": [
                    ("user", "I have been thinking about saying hello to someone in my building."),
                    ("assistant", "That sounds like a meaningful possibility, even if it also feels awkward."),
                ],
                "current": "Could you help me choose one low-pressure way to start, without making it a big social challenge?",
                "summary": "The user asks for one low-pressure way to begin connecting with someone in the building.",
                "authorized": "Use the visible request for a small social step and keep any suggestion low pressure.",
            },
            {
                "family": "relocation_loneliness",
                "subtopics": ["social_connection", "gentle_action"],
                "dialogue": [
                    ("user", "The person at the front desk seems friendly, but I always hurry past."),
                    ("assistant", "A brief interaction may feel possible, though starting it can still be uncomfortable."),
                ],
                "current": "Can we come up with one simple opening line I could use next time, with no pressure to keep talking?",
                "summary": "The user asks for one simple, no-pressure opening line for a brief local interaction.",
                "authorized": "Use the stated wish for one brief opening line without escalating it into a larger plan.",
            },
            {
                "family": "relocation_loneliness",
                "subtopics": ["community_connection", "gentle_action"],
                "dialogue": [
                    ("user", "There is a small plant swap in the courtyard on Saturday."),
                    ("assistant", "It sounds as if you are curious about it but do not want to overwhelm yourself."),
                ],
                "current": "What would be a gentle way to show up for five minutes and still give myself permission to leave?",
                "summary": "The user asks for a gentle, time-limited way to try a community activity.",
                "authorized": "Use the visible five-minute boundary and permission to leave when framing a suggestion.",
            },
        ],
        "advice_harmful": [
            {
                "family": "academic_pressure",
                "subtopics": ["fear_of_failure", "need_to_be_heard"],
                "dialogue": [
                    ("user", "I got my practice score back and it was much lower than I expected."),
                    ("assistant", "That result seems to have shaken you."),
                ],
                "current": "Please do not turn this into a study plan yet. I need to say how scared I am that I will fail.",
                "summary": "The user is scared of failing after a low practice score and explicitly asks not to receive a study plan yet.",
                "authorized": "Use the visible fear and listen-only boundary; do not offer a study plan in this turn.",
            },
            {
                "family": "academic_pressure",
                "subtopics": ["presentation_fear", "need_to_be_heard"],
                "dialogue": [
                    ("user", "My voice shook during the rehearsal and I forgot the middle section."),
                    ("assistant", "That rehearsal sounds as though it brought the fear right to the surface."),
                ],
                "current": "I am not asking for presentation tricks right now. Could you just stay with me while I talk about how humiliating it felt?",
                "summary": "The user felt humiliated during a rehearsal and asks to be heard without presentation advice.",
                "authorized": "Use the visible humiliation and request for listening; avoid presentation tips in this turn.",
            },
            {
                "family": "academic_pressure",
                "subtopics": ["assignment_fear", "need_to_be_heard"],
                "dialogue": [
                    ("user", "I saw the feedback and immediately felt sick to my stomach."),
                    ("assistant", "The feedback seems to have landed very hard."),
                ],
                "current": "I know we can discuss what to do later. For now, I need someone to hear how ashamed and frightened I feel.",
                "summary": "The user feels ashamed and frightened by feedback and asks for listening before later problem solving.",
                "authorized": "Use the visible emotional impact and sequencing boundary: listen now, problem-solve later.",
            },
        ],
        "ambiguous": [
            {
                "family": "workplace_stress",
                "subtopics": ["coworker_conflict", "deadline_pressure"],
                "dialogue": [
                    ("user", "A coworker's comment bothered me, and I also have a deadline tomorrow."),
                    ("assistant", "You are carrying both an uncomfortable interaction and immediate time pressure."),
                ],
                "current": "I am not sure which part is getting to me most. I think I need to untangle it before deciding what would help.",
                "summary": "The user is uncertain whether coworker tension or a deadline is the main source of workplace stress.",
                "authorized": "Use both visible workplace subtopics and keep the next response exploratory.",
            },
            {
                "family": "workplace_stress",
                "subtopics": ["manager_feedback", "workload_pressure"],
                "dialogue": [
                    ("user", "My manager's message felt sharp, but I was already behind when I read it."),
                    ("assistant", "The tone and the workload may be amplifying each other."),
                ],
                "current": "I cannot tell whether I need to talk about the message or just catch my breath from the workload first.",
                "summary": "The user is unsure whether a manager's message or workload pressure needs attention first.",
                "authorized": "Use both visible workplace concerns and explore before offering a direction.",
            },
            {
                "family": "workplace_stress",
                "subtopics": ["team_tension", "deadline_pressure"],
                "dialogue": [
                    ("user", "The team chat became tense just as I was trying to finish a late task."),
                    ("assistant", "That combination could make it hard to know what deserves attention first."),
                ],
                "current": "Part of me wants advice and part of me just wants to understand why the whole evening felt so heavy.",
                "summary": "The user is torn between advice and exploration after team tension coincided with deadline pressure.",
                "authorized": "Use the visible uncertainty and avoid assuming that advice or listening alone is already preferred.",
            },
        ],
    }


def _memory_specs(
    regime: str,
    *,
    variant_index: int,
) -> dict[MemorySource, list[dict[str, Any]]]:
    suffix = ("morning", "afternoon", "weekend")[variant_index - 1]
    irrelevant = {
        MemorySource.MP: {
            "text": f"The user prefers instrumental music while cooking on a {suffix}.",
            "utility": "irrelevant",
        },
        MemorySource.MS: {
            "text": f"Across earlier sessions, the user often compared grocery lists on a {suffix}.",
            "utility": "irrelevant",
        },
        MemorySource.ME: {
            "text": f"In a past session, the user described visiting a museum on a {suffix}.",
            "utility": "irrelevant",
            "private_sensitivity": "sensitive",
        },
    }
    secondary = {
        MemorySource.MP: {
            "text": "The user generally likes concise weather updates.",
            "utility": "irrelevant",
        },
        MemorySource.MS: {
            "text": "Several old sessions mentioned reorganizing a bookshelf.",
            "utility": "irrelevant",
        },
        MemorySource.ME: {
            "text": "The user once took a different route home after shopping.",
            "utility": "irrelevant",
        },
    }
    result = {
        source: [dict(irrelevant[source]), dict(secondary[source])]
        for source in MemorySource
    }
    if regime == "profile_useful":
        result[MemorySource.MP][0] = {
            "text": "Under academic pressure, the user prefers having feelings named before receiving problem-solving advice.",
            "utility": "helpful",
        }
    elif regime == "summary_useful":
        result[MemorySource.MS][0] = {
            "text": "Across three prior work sessions, public corrections repeatedly led the user to question their standing with the same colleague.",
            "utility": "helpful",
        }
    elif regime == "event_useful":
        result[MemorySource.ME][0] = {
            "text": "Last month, the user entered a library conversation group, asked the organizer one question, and left after ten minutes feeling relieved they had tried.",
            "utility": "helpful",
        }
    elif regime == "multi_source_useful":
        result[MemorySource.MP][0] = {
            "text": "When overwhelmed, the user prefers acknowledgment before a concrete suggestion.",
            "utility": "helpful",
        }
        result[MemorySource.MS][0] = {
            "text": "Across recent sessions, the user's task freeze repeatedly intensified when several deadlines converged.",
            "utility": "helpful",
        }
        result[MemorySource.ME][0] = {
            "text": "Last term, when similarly frozen, the user emailed a professor to clarify the first priority and then completed a short outline.",
            "utility": "helpful",
        }
    elif regime == "memory_harmful":
        result[MemorySource.MP][0] = {
            "text": "The user prefers never to discuss workplace conflict and wants the subject changed immediately.",
            "utility": "harmful",
            "stale": True,
            "conflicts_with_current_state": True,
        }
        result[MemorySource.MS][0] = {
            "text": "Older sessions showed the user withdrawing whenever a disagreement with a colleague was mentioned.",
            "utility": "harmful",
            "stale": True,
            "conflicts_with_current_state": True,
        }
        result[MemorySource.ME][0] = {
            "text": "Years ago, the user disclosed a sensitive family dispute unrelated to the current workplace exchange.",
            "utility": "harmful",
            "private_sensitivity": "sensitive",
        }
    return result


def _materially_useful_sources(regime: str) -> list[MemorySource]:
    return {
        "profile_useful": [MemorySource.MP],
        "summary_useful": [MemorySource.MS],
        "event_useful": [MemorySource.ME],
        "multi_source_useful": [MemorySource.MP, MemorySource.MS, MemorySource.ME],
    }.get(regime, [])


def _strategy_design(regime: str) -> tuple[dict[str, str], list[dict[str, str]]]:
    targets: dict[str, dict[str, str]] = {
        "context_only": {
            "use_strategy_rag": "off",
            "advice_readiness": "explore_first",
            "base_generator_assessment": "The visible turn already supports natural acknowledgment and a gentle follow-up without external strategy text.",
            "marginal_value_rationale": "The displayed listening card is generic and the suggestion card is premature, so RS adds token and retrieval cost without clear marginal gain.",
        },
        "profile_useful": {
            "use_strategy_rag": "ambiguous",
            "advice_readiness": "listen_only",
            "base_generator_assessment": "The base generator can reflect exam anxiety once the MP support-style preference is visible.",
            "marginal_value_rationale": "An affirmation card could help, but its value beyond a profile-conditioned base response is uncertain; a job-information card is unrelated.",
        },
        "summary_useful": {
            "use_strategy_rag": "ambiguous",
            "advice_readiness": "explore_first",
            "base_generator_assessment": "The base generator can acknowledge the incident, while MS supplies the otherwise missing cross-session pattern.",
            "marginal_value_rationale": "A reflection card fits, but it may duplicate normal base behavior; the directive card would move too quickly into problem solving.",
        },
        "event_useful": {
            "use_strategy_rag": "on",
            "advice_readiness": "light_suggestion",
            "base_generator_assessment": "The base generator can empathize, but the user explicitly asks for one small next step.",
            "marginal_value_rationale": "A low-pressure connection card complements the concrete ME precedent and can guide a bounded suggestion.",
        },
        "multi_source_useful": {
            "use_strategy_rag": "ambiguous",
            "advice_readiness": "explore_first",
            "base_generator_assessment": "The three memory sources already add support style, repeated pattern, and a concrete past coping event.",
            "marginal_value_rationale": "A restatement card may help organize the response but may also duplicate the base generator; a suggestion card would be early.",
        },
        "memory_harmful": {
            "use_strategy_rag": "ambiguous",
            "advice_readiness": "explore_first",
            "base_generator_assessment": "The current correction is visible, so the base generator can acknowledge it while memory is withheld.",
            "marginal_value_rationale": "A gentle question could support the user's wish to describe the event, but a detailed directive is poorly timed; marginal RS value remains uncertain.",
        },
        "strategy_helpful": {
            "use_strategy_rag": "on",
            "advice_readiness": "light_suggestion",
            "base_generator_assessment": "The base generator could answer generally, but the user asks for a bounded, low-pressure social step.",
            "marginal_value_rationale": "The selected suggestion card supplies a fitting connection strategy, while self-disclosure would add little and could shift focus.",
        },
        "advice_harmful": {
            "use_strategy_rag": "on",
            "advice_readiness": "listen_only",
            "base_generator_assessment": "The base generator can listen, but a well-matched restatement card can strengthen acknowledgment without becoming directive.",
            "marginal_value_rationale": "RS remains useful for listening support even though the suggestion card is harmful in this turn; advice readiness and RS are deliberately independent.",
        },
        "ambiguous": {
            "use_strategy_rag": "ambiguous",
            "advice_readiness": "explore_first",
            "base_generator_assessment": "The base generator can ask which workplace concern feels most urgent.",
            "marginal_value_rationale": "A gentle question card may help exploration but may duplicate base behavior; directive advice is premature until the concern is clarified.",
        },
    }
    designs: dict[str, list[dict[str, str]]] = {
        "context_only": [
            {
                "role": "open_restatement",
                "utility": "irrelevant",
                "rationale": "It is broadly acceptable but adds no material capability beyond a natural base acknowledgment.",
            },
            {
                "role": "one_problem_suggestion",
                "utility": "harmful",
                "rationale": "Problem prioritization would move away from the user's immediate loneliness before enough exploration.",
            },
        ],
        "profile_useful": [
            {
                "role": "support_affirmation",
                "utility": "helpful",
                "rationale": "Its supportive stance fits the requested emotional pacing, though its marginal value over a profile-conditioned base response is uncertain.",
            },
            {
                "role": "job_information",
                "utility": "irrelevant",
                "rationale": "The source example concerns employment information rather than the academic fear in this turn.",
            },
        ],
        "summary_useful": [
            {
                "role": "stress_reflection",
                "utility": "helpful",
                "rationale": "A stress reflection fits the uncertainty and can invite elaboration without assuming intent.",
            },
            {
                "role": "intrusive_long_plan",
                "utility": "harmful",
                "rationale": "The long directive template is unrelated and far too prescriptive for an uncertain coworker pattern.",
            },
        ],
        "event_useful": [
            {
                "role": "low_pressure_connection",
                "utility": "helpful",
                "rationale": "It supports a small connection step and can be adapted to the user's explicit low-pressure request.",
            },
            {
                "role": "video_call_self_disclosure",
                "utility": "irrelevant",
                "rationale": "The self-disclosure shifts attention to the supporter's habit and does not use the user's local context.",
            },
        ],
        "multi_source_useful": [
            {
                "role": "open_restatement",
                "utility": "helpful",
                "rationale": "A short restatement could organize acknowledgment before any concrete next step.",
            },
            {
                "role": "one_problem_suggestion",
                "utility": "irrelevant",
                "rationale": "The generic prioritization prompt may add little after the specific past coping event is available.",
            },
        ],
        "memory_harmful": [
            {
                "role": "gentle_question",
                "utility": "helpful",
                "rationale": "A gentle invitation can follow the user's current wish to put the event into words.",
            },
            {
                "role": "intrusive_long_plan",
                "utility": "harmful",
                "rationale": "A detailed directive would override the user's wish to first describe what happened.",
            },
        ],
        "strategy_helpful": [
            {
                "role": "low_pressure_connection",
                "utility": "helpful",
                "rationale": "The card offers a bounded way to connect and matches the request for a gentle social step.",
            },
            {
                "role": "video_call_self_disclosure",
                "utility": "irrelevant",
                "rationale": "The supporter's own video-call habit is not needed and risks making the answer about the supporter.",
            },
        ],
        "advice_harmful": [
            {
                "role": "open_restatement",
                "utility": "helpful",
                "rationale": "A brief restatement supports listening and does not violate the explicit no-advice boundary.",
            },
            {
                "role": "one_problem_suggestion",
                "utility": "harmful",
                "rationale": "Asking the user to prioritize a problem initiates problem solving after they explicitly requested listening.",
            },
        ],
        "ambiguous": [
            {
                "role": "gentle_question",
                "utility": "helpful",
                "rationale": "A gentle question can clarify whether interpersonal tension or time pressure is most salient.",
            },
            {
                "role": "one_problem_suggestion",
                "utility": "harmful",
                "rationale": "A problem-first instruction would impose a direction before the user has clarified what they need.",
            },
        ],
    }
    return targets[regime], designs[regime]


def _context_provenance(
    surface: dict[str, Any], *, session_index: int
) -> list[V8ContextProvenance]:
    current = str(surface["current"])
    rows = [
        V8ContextProvenance(
            target_field="session_summary",
            claim=str(surface["summary"]),
            provenance_type="current_user",
            provenance_source="current_user_text",
            provenance_session=session_index,
            evidence_quote=current,
        ),
        V8ContextProvenance(
            target_field="authorized_user_context",
            claim=str(surface["authorized"]),
            provenance_type="current_user",
            provenance_source="current_user_text",
            provenance_session=session_index,
            evidence_quote=current,
        ),
    ]
    correction = surface.get("correction")
    if correction:
        rows.append(
            V8ContextProvenance(
                target_field="current_correction",
                claim=str(correction),
                provenance_type="current_user",
                provenance_source="current_user_text",
                provenance_session=session_index,
                evidence_quote=current,
            )
        )
    return rows


def _compile_memory_pool(
    *,
    seed: int,
    regime: str,
    variant_index: int,
    source: MemorySource,
    session_index: int,
) -> list[V8MemoryEvidence]:
    specs = _memory_specs(regime, variant_index=variant_index)[source]
    # Utility is never an RNG input. Age and row-order use separate streams.
    age_rng = _seeded_rng(seed, "age", regime, variant_index, source.value)
    ages = age_rng.sample(list(range(1, 13)), k=len(specs))
    items: list[V8MemoryEvidence] = []
    provenance_type = {
        MemorySource.MP: "profile_record",
        MemorySource.MS: "cross_session_summary",
        MemorySource.ME: "past_event_record",
    }[source]
    for logical_index, (spec, age) in enumerate(zip(specs, ages, strict=True), 1):
        created_session = session_index - age
        memory_id = "mem_" + sha256_text(
            f"v8|{seed}|{regime}|{variant_index}|{source.value}|{logical_index}|{spec['text']}"
        )[:24]
        items.append(
            V8MemoryEvidence(
                memory_id=memory_id,
                source=source,
                text=str(spec["text"]),
                created_session=created_session,
                age_sessions=age,
                utility=str(spec["utility"]),
                stale=bool(spec.get("stale", False)),
                conflicts_with_current_state=bool(
                    spec.get("conflicts_with_current_state", False)
                ),
                private_sensitivity=str(
                    spec.get("private_sensitivity", "ordinary")
                ),
                provenance_type=provenance_type,
                provenance_source=f"synthetic_{source.value.lower()}_record_{logical_index}",
                provenance_session=created_session,
            )
        )
    order_rng = _seeded_rng(seed, "row_order", regime, variant_index, source.value)
    order_rng.shuffle(items)
    return items


def _compile_strategy_evidence(
    *,
    regime: str,
    catalog: dict[str, StrategyCard],
    strategy_card_ids: Mapping[str, str],
) -> tuple[V8StrategyTarget, list[V8StrategyEvidence]]:
    target, designs = _strategy_design(regime)
    evidence: list[V8StrategyEvidence] = []
    for design in designs:
        card = catalog[strategy_card_ids[design["role"]]]
        evidence.append(
            V8StrategyEvidence(
                card_id=card.strategy_id,
                strategy_type=card.strategy_label,
                utility=design["utility"],
                card_text=card.example_response,
                retrieval_text=card.retrieval_text,
                guidance_text=card.guidance_text,
                marginal_value_rationale=design["rationale"],
                source_dialogue_id=card.source_dialogue_id,
                source_turn_index=card.source_turn_index,
            )
        )
    return V8StrategyTarget.model_validate(target), evidence


def _coverage_rationale(regime: str) -> str:
    return {
        "context_only": "Visible context is sufficient for acknowledgment and exploration; external memory adds no material value and the shown strategy cards add no clear marginal value.",
        "profile_useful": "MP changes how support should be paced; MS and ME are unrelated, while RS value remains uncertain after profile conditioning.",
        "summary_useful": "MS contributes a cross-session pattern that is not present in the current incident; other memory sources add no material value.",
        "event_useful": "ME supplies a concrete prior low-pressure action that can inform a bounded suggestion; MP and MS do not add relevant information.",
        "multi_source_useful": "MP supplies support style, MS supplies a repeated overload pattern, and ME supplies a concrete prior coping action; their contributions are distinct but are not claimed to be indispensable.",
        "memory_harmful": "Current visible language corrects old avoidance records, and the unrelated sensitive event would be intrusive; memory should be withheld.",
        "strategy_helpful": "No memory adds material value, while a concrete low-pressure connection card has clear marginal value for the user's request.",
        "advice_harmful": "Directive advice violates the current boundary, but a listening-oriented strategy card can still add value; RS and directiveness are separate decisions.",
        "ambiguous": "Two workplace subtopics are visible, but neither memory nor external strategy retrieval has a uniquely established marginal advantage before further exploration.",
    }[regime]


def generate_v8_review_cases(
    *,
    strategy_bank_path: str | Path,
    cases_per_regime: int,
    seed: int,
    strategy_card_ids: Mapping[str, str] | None = None,
) -> list[V8ReviewCase]:
    if cases_per_regime not in {
        SMOKE_CASES_PER_REGIME,
        VALIDATION_CASES_PER_REGIME,
    }:
        raise ValueError("cases_per_regime must be 1 (smoke) or 3 (validation)")
    role_map = dict(strategy_card_ids or STRATEGY_CARD_IDS)
    catalog = load_strategy_catalog(
        strategy_bank_path, strategy_card_ids=role_map
    )
    surfaces = _surface_variants()
    cases: list[V8ReviewCase] = []
    item_index = 0
    for regime in V8_REGIMES:
        for variant_index in range(1, cases_per_regime + 1):
            item_index += 1
            surface = surfaces[regime][variant_index - 1]
            session_index = 18 + variant_index * 3
            strategy_target, strategy_evidence = _compile_strategy_evidence(
                regime=regime,
                catalog=catalog,
                strategy_card_ids=role_map,
            )
            case_digest = sha256_text(
                f"v8|{seed}|{regime}|{variant_index}|{surface['current']}"
            )
            cases.append(
                V8ReviewCase(
                    item_id=f"v8_item_{item_index:02d}",
                    case_id=f"case_{case_digest[:24]}",
                    variant_index=variant_index,
                    semantic_family=surface["family"],
                    semantic_subtopics=list(surface["subtopics"]),
                    regime=regime,
                    current_user_text=surface["current"],
                    dialogue_before_current=[
                        DialogueTurn(role=role, content=content)
                        for role, content in surface["dialogue"]
                    ],
                    session_summary=surface["summary"],
                    authorized_user_context=surface["authorized"],
                    current_correction=surface.get("correction"),
                    context_provenance=_context_provenance(
                        surface, session_index=session_index
                    ),
                    coverage_rationale=_coverage_rationale(regime),
                    session_index=session_index,
                    materially_useful_memory_sources=_materially_useful_sources(
                        regime
                    ),
                    profile_memories=_compile_memory_pool(
                        seed=seed,
                        regime=regime,
                        variant_index=variant_index,
                        source=MemorySource.MP,
                        session_index=session_index,
                    ),
                    summary_memories=_compile_memory_pool(
                        seed=seed,
                        regime=regime,
                        variant_index=variant_index,
                        source=MemorySource.MS,
                        session_index=session_index,
                    ),
                    event_memories=_compile_memory_pool(
                        seed=seed,
                        regime=regime,
                        variant_index=variant_index,
                        source=MemorySource.ME,
                        session_index=session_index,
                    ),
                    strategy_target=strategy_target,
                    strategy_evidence=strategy_evidence,
                    generation_seed=seed,
                    age_sampling_protocol=(
                        "sha256-separated RNG stream keyed by seed/case/source; "
                        "utility is not an RNG input"
                    ),
                    row_shuffle_protocol=(
                        "independent sha256-separated RNG stream after item construction"
                    ),
                )
            )
    return cases


PROTECTED_FIELDS = (
    "item_id",
    "case_id",
    "variant_index",
    "candidate_semantic_family",
    "candidate_semantic_subtopics_json",
    "candidate_regime",
    "candidate_materially_useful_memory_sources_json",
    "legacy_needed_memory_sources_json",
    "session_index",
    "current_user_text",
    "dialogue_before_current_json",
    "session_summary",
    "authorized_user_context",
    "current_correction",
    "coverage_rationale",
    "context_provenance_json",
    "profile_memories_json",
    "summary_memories_json",
    "event_memories_json",
    "strategy_target_json",
    "strategy_evidence_json",
    "generation_seed",
    "age_sampling_protocol",
    "row_shuffle_protocol",
)
PACKET_FIELDS = (*PROTECTED_FIELDS, *RATING_FIELDS, "annotator_id", "notes")
SIMPLE_REVIEW_FIELDS = ("item_id", *RATING_FIELDS, "annotator_id", "notes")


def cases_to_packet_rows(cases: Sequence[V8ReviewCase]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case in cases:
        materially_useful = [
            source.value for source in case.materially_useful_memory_sources
        ]
        rows.append(
            {
                "item_id": case.item_id,
                "case_id": case.case_id,
                "variant_index": str(case.variant_index),
                "candidate_semantic_family": case.semantic_family,
                "candidate_semantic_subtopics_json": canonical_json(
                    case.semantic_subtopics
                ),
                "candidate_regime": case.regime,
                "candidate_materially_useful_memory_sources_json": canonical_json(
                    materially_useful
                ),
                # Compatibility only: the meaning is exactly the V8 material-value set.
                "legacy_needed_memory_sources_json": canonical_json(materially_useful),
                "session_index": str(case.session_index),
                "current_user_text": case.current_user_text,
                "dialogue_before_current_json": canonical_json(
                    [turn.model_dump(mode="json") for turn in case.dialogue_before_current]
                ),
                "session_summary": case.session_summary,
                "authorized_user_context": case.authorized_user_context,
                "current_correction": case.current_correction or "",
                "coverage_rationale": case.coverage_rationale,
                "context_provenance_json": canonical_json(
                    [row.model_dump(mode="json") for row in case.context_provenance]
                ),
                "profile_memories_json": canonical_json(
                    [row.model_dump(mode="json") for row in case.profile_memories]
                ),
                "summary_memories_json": canonical_json(
                    [row.model_dump(mode="json") for row in case.summary_memories]
                ),
                "event_memories_json": canonical_json(
                    [row.model_dump(mode="json") for row in case.event_memories]
                ),
                "strategy_target_json": canonical_json(
                    case.strategy_target.model_dump(mode="json")
                ),
                "strategy_evidence_json": canonical_json(
                    [row.model_dump(mode="json") for row in case.strategy_evidence]
                ),
                "generation_seed": str(case.generation_seed),
                "age_sampling_protocol": case.age_sampling_protocol,
                "row_shuffle_protocol": case.row_shuffle_protocol,
                **{field: "" for field in RATING_FIELDS},
                "annotator_id": "",
                "notes": "",
            }
        )
    return rows


def _write_csv(
    path: Path, rows: Sequence[dict[str, str]], *, fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_reviewer_csv(
    path: Path, cases: Sequence[V8ReviewCase], *, annotator_id: str
) -> None:
    rows = [
        {
            "item_id": case.item_id,
            **{field: "" for field in RATING_FIELDS},
            "annotator_id": annotator_id,
            "notes": "",
        }
        for case in cases
    ]
    _write_csv(path, rows, fieldnames=SIMPLE_REVIEW_FIELDS)


def _memory_rows(case: V8ReviewCase):
    for source, memories in (
        (MemorySource.MP, case.profile_memories),
        (MemorySource.MS, case.summary_memories),
        (MemorySource.ME, case.event_memories),
    ):
        for position, item in enumerate(memories, 1):
            yield source, position, item


def _resolve_provenance(case: V8ReviewCase, row: V8ContextProvenance) -> str | None:
    if row.provenance_source == "current_user_text":
        if row.provenance_type != "current_user":
            return None
        return case.current_user_text
    match = re.fullmatch(r"dialogue_before_current\[(\d+)\]", row.provenance_source)
    if match:
        index = int(match.group(1))
        if row.provenance_type != "dialogue_turn" or index >= len(
            case.dialogue_before_current
        ):
            return None
        return case.dialogue_before_current[index].content
    # No V8 smoke/validation case currently authorizes hidden external updates.
    return None


def _pearson_binary_age(rows: Sequence[tuple[int, int]]) -> float:
    if not rows:
        return 0.0
    xs = [float(age) for age, _ in rows]
    ys = [float(label) for _, label in rows]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum(
        (left - mean_x) * (right - mean_y)
        for left, right in zip(xs, ys, strict=True)
    )
    denom_x = math.sqrt(sum((value - mean_x) ** 2 for value in xs))
    denom_y = math.sqrt(sum((value - mean_y) ** 2 for value in ys))
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return numerator / (denom_x * denom_y)


def _normalized_surface(value: str) -> str:
    return " ".join(value.casefold().split())


def _v7_surface_texts(v7_packet_path: str | Path | None) -> set[str]:
    if v7_packet_path is None:
        return set()
    path = Path(v7_packet_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    values: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for field in (
                "current_user_text",
                "session_summary",
                "authorized_user_context",
            ):
                value = str(row.get(field) or "")
                if value:
                    values.add(_normalized_surface(value))
            for turn in json.loads(str(row.get("dialogue_before_current_json") or "[]")):
                values.add(_normalized_surface(str(turn.get("content") or "")))
    return values


def audit_generation_pilot_semantics(
    cases: Sequence[V8ReviewCase],
    *,
    strategy_bank_path: str | Path,
    v7_packet_path: str | Path | None = None,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("semantic audit requires cases")
    catalog = load_strategy_catalog(strategy_bank_path)
    regime_counts = Counter(case.regime for case in cases)
    cases_per_regime = sorted(set(regime_counts.values()))
    expected_balanced = (
        len(regime_counts) == len(V8_REGIMES)
        and set(regime_counts) == set(V8_REGIMES)
        and len(cases_per_regime) == 1
        and cases_per_regime[0] in {1, 3}
    )

    utility_source_counts: dict[str, Counter[str]] = {
        source.value: Counter() for source in MemorySource
    }
    utility_age_values: dict[str, dict[str, list[int]]] = {
        source.value: defaultdict(list) for source in MemorySource
    }
    utility_position_counts: dict[str, dict[str, Counter[int]]] = {
        source.value: defaultdict(Counter) for source in MemorySource
    }
    pools: list[dict[str, Any]] = []
    binary_age_rows: list[tuple[int, int]] = []
    for case in cases:
        by_source: dict[MemorySource, list[V8MemoryEvidence]] = {
            MemorySource.MP: case.profile_memories,
            MemorySource.MS: case.summary_memories,
            MemorySource.ME: case.event_memories,
        }
        for source, memories in by_source.items():
            for position, item in enumerate(memories, 1):
                utility_source_counts[source.value][item.utility] += 1
                utility_age_values[source.value][item.utility].append(
                    item.age_sessions
                )
                utility_position_counts[source.value][item.utility][position] += 1
                binary_age_rows.append(
                    (item.age_sessions, int(item.utility != "irrelevant"))
                )
            targets = [item for item in memories if item.utility != "irrelevant"]
            if len(targets) == 1:
                youngest = min(memories, key=lambda item: item.age_sessions)
                pools.append(
                    {
                        "case_id": case.case_id,
                        "source": source.value,
                        "youngest_correct": youngest.memory_id == targets[0].memory_id,
                        "first_correct": memories[0].memory_id == targets[0].memory_id,
                    }
                )

    youngest_accuracy = (
        sum(int(row["youngest_correct"]) for row in pools) / len(pools)
        if pools
        else 0.0
    )
    first_accuracy = (
        sum(int(row["first_correct"]) for row in pools) / len(pools)
        if pools
        else 0.0
    )
    age_correlation = _pearson_binary_age(binary_age_rows)

    counterexamples: dict[str, dict[str, bool]] = {}
    for source in MemorySource:
        source_rows = [
            (position, item)
            for case in cases
            for row_source, position, item in _memory_rows(case)
            if row_source is source
        ]
        helpful_ages = [
            item.age_sessions for _, item in source_rows if item.utility == "helpful"
        ]
        harmful_ages = [
            item.age_sessions for _, item in source_rows if item.utility == "harmful"
        ]
        irrelevant_ages = [
            item.age_sessions
            for _, item in source_rows
            if item.utility == "irrelevant"
        ]
        counterexamples[source.value] = {
            "older_helpful_than_some_irrelevant": bool(
                helpful_ages
                and irrelevant_ages
                and max(helpful_ages) > min(irrelevant_ages)
            ),
            "newer_irrelevant_than_some_target": bool(
                irrelevant_ages
                and (helpful_ages or harmful_ages)
                and min(irrelevant_ages) < max([*helpful_ages, *harmful_ages])
            ),
            "older_harmful_than_some_irrelevant": bool(
                harmful_ages
                and irrelevant_ages
                and max(harmful_ages) > min(irrelevant_ages)
            ),
            "newer_helpful_than_some_irrelevant": bool(
                helpful_ages
                and irrelevant_ages
                and min(helpful_ages) < max(irrelevant_ages)
            ),
            "first_row_can_be_irrelevant": any(
                position == 1 and item.utility == "irrelevant"
                for position, item in source_rows
            ),
            "second_row_can_be_target": any(
                position == 2 and item.utility != "irrelevant"
                for position, item in source_rows
            ),
        }

    position_to_utilities: dict[str, dict[str, list[str]]] = {}
    for source in MemorySource:
        position_to_utilities[source.value] = {}
        for position in (1, 2):
            position_to_utilities[source.value][str(position)] = sorted(
                {
                    item.utility
                    for case in cases
                    for row_source, row_position, item in _memory_rows(case)
                    if row_source is source and row_position == position
                }
            )
    no_position_mapping = all(
        len(utilities) >= 2
        for rows in position_to_utilities.values()
        for utilities in rows.values()
    )
    all_counterexamples = all(
        all(checks.values()) for checks in counterexamples.values()
    )

    leakage_hits: list[dict[str, str]] = []
    for case in cases:
        surfaces = {
            "current_user_text": case.current_user_text,
            "session_summary": case.session_summary,
            **{
                f"dialogue_before_current[{index}]": turn.content
                for index, turn in enumerate(case.dialogue_before_current)
            },
        }
        for field, text in surfaces.items():
            normalized = text.casefold()
            for phrase in PROHIBITED_SURFACE_PHRASES:
                if phrase in normalized:
                    leakage_hits.append(
                        {
                            "case_id": case.case_id,
                            "field": field,
                            "phrase": phrase,
                        }
                    )

    grounding_errors: list[dict[str, str]] = []
    for case in cases:
        required_targets = {"session_summary", "authorized_user_context"}
        if case.current_correction:
            required_targets.add("current_correction")
        observed_targets = {row.target_field for row in case.context_provenance}
        if observed_targets != required_targets:
            grounding_errors.append(
                {
                    "case_id": case.case_id,
                    "error": "provenance targets do not exactly cover grounded fields",
                }
            )
        for row in case.context_provenance:
            source_text = _resolve_provenance(case, row)
            if source_text is None or row.evidence_quote not in source_text:
                grounding_errors.append(
                    {
                        "case_id": case.case_id,
                        "error": f"unresolved provenance for {row.target_field}",
                    }
                )
            if row.provenance_session > case.session_index:
                grounding_errors.append(
                    {
                        "case_id": case.case_id,
                        "error": f"future provenance for {row.target_field}",
                    }
                )

    strategy_errors: list[dict[str, str]] = []
    for case in cases:
        if len(case.strategy_evidence) < 2:
            strategy_errors.append(
                {"case_id": case.case_id, "error": "fewer than two strategy cards"}
            )
        for evidence in case.strategy_evidence:
            card = catalog.get(evidence.card_id)
            if card is None:
                strategy_errors.append(
                    {"case_id": case.case_id, "error": "unknown strategy card"}
                )
                continue
            if (
                evidence.strategy_type != card.strategy_label
                or evidence.card_text != card.example_response
                or evidence.retrieval_text != card.retrieval_text
                or evidence.guidance_text != card.guidance_text
            ):
                strategy_errors.append(
                    {"case_id": case.case_id, "error": "strategy card drift"}
                )
        utilities = {row.utility for row in case.strategy_evidence}
        if "helpful" not in utilities and not utilities <= {"irrelevant", "harmful"}:
            strategy_errors.append(
                {"case_id": case.case_id, "error": "invalid strategy contrast"}
            )

    v7_texts = _v7_surface_texts(v7_packet_path)
    v7_reuse: list[dict[str, str]] = []
    if v7_texts:
        for case in cases:
            for field, value in {
                "current_user_text": case.current_user_text,
                "session_summary": case.session_summary,
                "authorized_user_context": case.authorized_user_context,
                **{
                    f"dialogue_before_current[{index}]": turn.content
                    for index, turn in enumerate(case.dialogue_before_current)
                },
            }.items():
                if _normalized_surface(value) in v7_texts:
                    v7_reuse.append(
                        {"case_id": case.case_id, "field": field, "text": value}
                    )

    advice_strategy_pairs = {
        (
            case.strategy_target.use_strategy_rag,
            case.strategy_target.advice_readiness,
        )
        for case in cases
    }
    advice_strategy_independent = (
        ("on", "listen_only") in advice_strategy_pairs
        and ("on", "light_suggestion") in advice_strategy_pairs
        and ("off", "explore_first") in advice_strategy_pairs
        and any(left == "ambiguous" for left, _ in advice_strategy_pairs)
    )

    checks = {
        "balanced_9_regime_design": expected_balanced,
        "age_and_utility_not_perfectly_correlated": abs(age_correlation) < 0.80,
        "youngest_item_heuristic_not_perfect": bool(pools)
        and youngest_accuracy < 1.0,
        "first_row_heuristic_not_perfect": bool(pools) and first_accuracy < 1.0,
        "no_deterministic_row_position_mapping": no_position_mapping,
        "source_level_counterexamples_complete": all_counterexamples,
        "no_prohibited_surface_phrases": not leakage_hits,
        "context_grounding_complete": not grounding_errors,
        "strategy_card_records_match_current_unfrozen_v1_bank": not strategy_errors,
        "advice_and_strategy_decisions_are_independent": advice_strategy_independent,
        "no_exact_v7_surface_reuse": not v7_reuse,
    }
    technical_checks_status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "status": (
            PROVISIONAL_STATUS
            if technical_checks_status == "PASS"
            else "FAIL"
        ),
        "technical_checks_status": technical_checks_status,
        "strategy_ground_truth_status": "PENDING_V1_STRATEGY_RAG_AUDIT",
        "ready_for_annotation": False,
        "ready_for_training": False,
        "protocol": V8_PROTOCOL,
        "checks": checks,
        "case_count": len(cases),
        "cases_per_regime": cases_per_regime[0] if len(cases_per_regime) == 1 else None,
        "regime_counts": dict(sorted(regime_counts.items())),
        "utility_by_source": {
            source: dict(sorted(counts.items()))
            for source, counts in utility_source_counts.items()
        },
        "utility_age_distribution": {
            source: {
                utility: {
                    "n": len(ages),
                    "minimum": min(ages),
                    "maximum": max(ages),
                    "mean": sum(ages) / len(ages),
                    "values": ages,
                }
                for utility, ages in sorted(rows.items())
            }
            for source, rows in utility_age_values.items()
        },
        "utility_row_position_distribution": {
            source: {
                utility: {str(position): count for position, count in sorted(counts.items())}
                for utility, counts in sorted(rows.items())
            }
            for source, rows in utility_position_counts.items()
        },
        "youngest_item_heuristic": {
            "eligible_pools": len(pools),
            "accuracy": youngest_accuracy,
        },
        "first_row_heuristic": {
            "eligible_pools": len(pools),
            "accuracy": first_accuracy,
        },
        "age_non_irrelevant_pearson_correlation": age_correlation,
        "counterexamples_by_source": counterexamples,
        "position_to_observed_utilities": position_to_utilities,
        "surface_leakage_hits": leakage_hits,
        "grounding_errors": grounding_errors,
        "strategy_errors": strategy_errors,
        "v7_exact_surface_reuse": v7_reuse,
        "advice_strategy_pairs": [list(pair) for pair in sorted(advice_strategy_pairs)],
        "fail_closed_rule": (
            "Any perfect shortcut, missing source counterexample, provenance failure, "
            "strategy drift, prohibited phrase, or exact V7 reuse fails the audit."
        ),
    }
    return report


def semantic_audit_markdown(report: dict[str, Any]) -> str:
    checks = report["checks"]
    lines = [
        "# PM V2 generation semantic leakage audit — V8",
        "",
        f"- Overall status: **{report['status']}**",
        f"- Automatic technical checks: **{report['technical_checks_status']}**",
        "- Strategy ground truth: **PENDING V1 STRATEGY RAG AUDIT**",
        "- Annotation/training: **NOT READY**",
        f"- Cases: {report['case_count']}",
        f"- Cases per regime: {report['cases_per_regime']}",
        f"- Youngest-item heuristic accuracy: {report['youngest_item_heuristic']['accuracy']:.3f}",
        f"- First-row heuristic accuracy: {report['first_row_heuristic']['accuracy']:.3f}",
        f"- Age/non-irrelevant Pearson correlation: {report['age_non_irrelevant_pearson_correlation']:.3f}",
        "",
        "## Gate checks",
        "",
        "| Check | Result |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{name}` | {'PASS' if value else 'FAIL'} |"
        for name, value in checks.items()
    )
    lines.extend(
        [
            "",
            "## Utility × source",
            "",
            "| Source | Helpful | Irrelevant | Harmful |",
            "|---|---:|---:|---:|",
        ]
    )
    for source, counts in report["utility_by_source"].items():
        lines.append(
            f"| {source} | {counts.get('helpful', 0)} | "
            f"{counts.get('irrelevant', 0)} | {counts.get('harmful', 0)} |"
        )
    lines.extend(["", "## Source-level counterexamples", ""])
    for source, values in report["counterexamples_by_source"].items():
        lines.append(f"### {source}")
        lines.append("")
        for name, value in values.items():
            lines.append(f"- `{name}`: {'PASS' if value else 'FAIL'}")
        lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "This provisional audit rejects obvious deterministic memory/grounding shortcuts. It does not validate Strategy RAG on/off or card utility. The inherited V1 card bank and real retriever must be audited and frozen before strategy fields can become annotation targets.",
            "",
        ]
    )
    return "\n".join(lines)


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _case_markdown(case: V8ReviewCase, *, language: Literal["ZH", "EN"]) -> str:
    zh = language == "ZH"
    lines = [
        f"## {case.item_id}: {case.regime}",
        "",
        f"- {'广义语义家庭' if zh else 'Broad semantic family'}: `{case.semantic_family}`",
        f"- {'语义子话题' if zh else 'Semantic subtopics'}: `{canonical_json(case.semantic_subtopics)}`",
        f"- {'候选资源情形' if zh else 'Candidate resource situation'}: `{case.regime}`",
        f"- {'具有实质边际价值的 memory 来源' if zh else 'Memory sources with material marginal value'}: `{canonical_json([source.value for source in case.materially_useful_memory_sources])}`",
        f"- {'当前 session' if zh else 'Current session'}: `{case.session_index}`",
        "",
        f"### {'可见对话' if zh else 'Visible dialogue'}",
        "",
    ]
    for turn in case.dialogue_before_current:
        lines.append(f"- **{turn.role}**: {turn.content}")
    lines.extend(
        [
            f"- **current user**: {case.current_user_text}",
            "",
            f"### {'上下文与依据' if zh else 'Context and rationale'}",
            "",
            f"- **Session summary**: {case.session_summary}",
            f"- **Authorized context**: {case.authorized_user_context}",
            f"- **Current correction**: {case.current_correction or ('无' if zh else 'None')}",
            f"- **Rationale**: {case.coverage_rationale}",
            "",
            f"### {'上下文 provenance' if zh else 'Context provenance'}",
            "",
            "| Target | Type | Source | Session | Evidence quote | Claim |",
            "|---|---|---|---:|---|---|",
        ]
    )
    for row in case.context_provenance:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.target_field,
                    row.provenance_type,
                    row.provenance_source,
                    str(row.provenance_session),
                    _md_cell(row.evidence_quote),
                    _md_cell(row.claim),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"### {'Memory evidence（表内顺序即审查行顺序）' if zh else 'Memory evidence (table order is review row order)'}",
            "",
            "| Source | Row | Utility | Age | Created session | Stale | Conflict | Sensitive | Provenance | Text |",
            "|---|---:|---|---:|---:|---|---|---|---|---|",
        ]
    )
    for source, position, item in _memory_rows(case):
        lines.append(
            f"| {source.value} | {position} | {item.utility} | {item.age_sessions} | "
            f"{item.created_session} | {item.stale} | {item.conflicts_with_current_state} | "
            f"{item.private_sensitivity} | {item.provenance_type}:{item.provenance_source}:s{item.provenance_session} | "
            f"{_md_cell(item.text)} |"
        )
    target = case.strategy_target
    lines.extend(
        [
            "",
            f"### {'Strategy RAG 目标（与 advice 独立）' if zh else 'Strategy RAG target (independent of advice)'}",
            "",
            f"- `use_strategy_rag`: `{target.use_strategy_rag}`",
            f"- `advice_readiness`: `{target.advice_readiness}`",
            f"- `base_generator_assessment`: {target.base_generator_assessment}",
            f"- `marginal_value_rationale`: {target.marginal_value_rationale}",
            "",
            f"### {'真实 Strategy evidence' if zh else 'Actual strategy evidence'}",
            "",
            "| Card ID | Type | Utility | Card text | Marginal-value rationale | Source |",
            "|---|---|---|---|---|---|",
        ]
    )
    for item in case.strategy_evidence:
        lines.append(
            f"| `{item.card_id}` | {item.strategy_type} | {item.utility} | "
            f"{_md_cell(item.card_text)} | "
            f"{_md_cell(item.marginal_value_rationale)} | "
            f"{item.source_dialogue_id}:{item.source_turn_index} |"
        )
    questions = REVIEW_QUESTIONS_ZH if zh else REVIEW_QUESTIONS_EN
    lines.extend(
        [
            "",
            f"### {'本例 12 项人工判断' if zh else 'Twelve human judgments for this case'}",
            "",
            "| CSV field | 0/1 | Question |",
            "|---|---:|---|",
        ]
    )
    for field in RATING_FIELDS:
        lines.append(f"| `{field}` |  | {questions[field]} |")
    lines.extend(["", f"{'备注' if zh else 'Notes'}:", ""])
    return "\n".join(lines)


def readable_review(
    cases: Sequence[V8ReviewCase], *, language: Literal["ZH", "EN"]
) -> str:
    zh = language == "ZH"
    title = (
        "# PM V2 generation pilot V8 中文完整审查材料"
        if zh
        else "# PM V2 generation pilot V8 complete review material"
    )
    intro = (
        "【PROVISIONAL / 暂不可标注 / 不可训练】必须先完成 V1 Strategy RAG 卡库与真实 retriever audit。下列 Strategy on/off 和卡片 utility 只是待审计假设，不是 ground truth。其余材料仅用于检查 memory 边际价值、上下文 provenance、直接建议强度和自然度。"
        if zh
        else "PROVISIONAL / NOT READY FOR ANNOTATION / NOT FOR TRAINING. Audit and freeze the inherited V1 Strategy RAG card bank and real retriever first. Strategy on/off and card-utility fields below are hypotheses, not ground truth."
    )
    parts = [title, "", intro, ""]
    parts.extend(_case_markdown(case, language=language) for case in cases)
    return "\n".join(parts).rstrip() + "\n"


def annotation_guidelines(*, language: Literal["ZH", "EN"]) -> str:
    zh = language == "ZH"
    questions = REVIEW_QUESTIONS_ZH if zh else REVIEW_QUESTIONS_EN
    lines = [
        (
            "# PM V2 V8 双人独立语义审查指南"
            if zh
            else "# PM V2 V8 independent semantic-review guidelines"
        ),
        "",
        f"Protocol: `{V8_PROTOCOL}`",
        "",
        (
            "**PROVISIONAL：当前材料不得开始正式标注或训练。Strategy 字段必须等待 V1 卡库与真实 retriever audit。**"
            if zh
            else "**PROVISIONAL: do not begin formal annotation or training. Strategy fields must wait for the V1 card-bank and real-retriever audit.**"
        ),
        "",
        (
            "每名审查者只填写自己的 CSV。明确符合填 `1`；不符合或不能确定填 `0`。填 `0` 时 `notes` 必须指出 case、字段和原因。不要把候选标签当答案；必须阅读全部对话、memory、strategy card 与 provenance。"
            if zh
            else "Each reviewer completes only their own CSV. Enter `1` only when clearly supported; enter `0` when unsupported or uncertain. Every `0` requires notes naming the case, field, and reason. Candidate labels are not answers: read all dialogue, memory, strategy-card, and provenance evidence."
        ),
        "",
        (
            "`materially_useful_memory_sources` 表示相对 Context Only 具有实质、非重复边际价值，不表示缺少其中任何来源就绝对无法作答。legacy `needed_memory_sources` 只是同一集合的兼容别名。"
            if zh
            else "`materially_useful_memory_sources` means material, nonredundant value over Context Only; it does not claim that a response is impossible without every listed source. Legacy `needed_memory_sources` is only a compatibility alias for the same set."
        ),
        "",
        (
            "Strategy RAG 是 ESC 支持策略资源，不等于行动计划。`use_strategy_rag` 与 `advice_readiness` 必须分别判断；listen-only 情形仍可能受益于 reflection/restatement 卡。"
            if zh
            else "Strategy RAG is an ESC support-strategy resource, not a synonym for an action plan. Judge `use_strategy_rag` separately from `advice_readiness`; listen-only turns may still benefit from reflection or restatement cards."
        ),
        "",
        "## " + ("12 个评分字段" if zh else "Twelve rating fields"),
        "",
    ]
    for field in RATING_FIELDS:
        lines.append(f"- `{field}`: {questions[field]}")
    lines.extend(
        [
            "",
            "## " + ("Gate" if zh else "Gate"),
            "",
            (
                "任一缺失值、非法值、任一 0、0 无 notes、审查者不独立、packet/hash 漂移或自动 audit FAIL 都会 fail-closed。9 例 smoke 只用于快速检查；正式 generation 必须等待 27 例 validation 的独立双人全通过证明。"
                if zh
                else "Any missing/invalid value, any 0, a 0 without notes, non-independent reviewers, packet/hash drift, or failed automatic audit fails closed. The nine-case smoke set is diagnostic only; formal generation must wait for an all-pass two-reviewer attestation on the 27-case validation set."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _v7_hash_manifest(v7_dir: Path) -> dict[str, str]:
    if not v7_dir.is_dir():
        raise FileNotFoundError(v7_dir)
    return {
        path.name: sha256_file(path)
        for path in sorted(v7_dir.iterdir())
        if path.is_file()
    }


def prepare_generation_semantic_review_v8(
    *,
    strategy_bank_path: str | Path,
    out_dir: str | Path,
    v7_dir: str | Path,
    cases_per_regime: int,
    seed: int,
) -> dict[str, Any]:
    out_dir = Path(out_dir).resolve()
    strategy_bank_path = Path(strategy_bank_path).resolve()
    v7_dir = Path(v7_dir).resolve()
    if out_dir.exists():
        raise RuntimeError(
            f"refusing to overwrite V8 review directory; use a new path: {out_dir}"
        )
    v7_hashes_before = _v7_hash_manifest(v7_dir)
    cases = generate_v8_review_cases(
        strategy_bank_path=strategy_bank_path,
        cases_per_regime=cases_per_regime,
        seed=seed,
    )
    audit = audit_generation_pilot_semantics(
        cases,
        strategy_bank_path=strategy_bank_path,
        v7_packet_path=v7_dir / "generation_pilot_semantic_review_packet.csv",
    )
    if audit["technical_checks_status"] != "PASS":
        raise RuntimeError(
            "V8 semantic audit failed before materialization: "
            + ", ".join(name for name, value in audit["checks"].items() if not value)
        )
    out_dir.mkdir(parents=True)
    paths = {
        "packet": out_dir / "generation_pilot_semantic_review_packet.csv",
        "readable_zh": out_dir / "generation_pilot_semantic_review_readable_ZH.md",
        "readable_en": out_dir / "generation_pilot_semantic_review_readable_EN.md",
        "reviewer_a": out_dir / "reviewer_a.csv",
        "reviewer_b": out_dir / "reviewer_b.csv",
        "guidelines_zh": out_dir / "annotation_guidelines_ZH.md",
        "guidelines_en": out_dir / "annotation_guidelines_EN.md",
        "audit_json": out_dir / "semantic_review_audit.json",
        "audit_md": out_dir / "semantic_review_audit.md",
        "changelog": out_dir / "V7_TO_V8_CHANGELOG.md",
        "readme": out_dir / "README.md",
        "cases": out_dir / "generation_pilot_semantic_review_cases.json",
        "plan": out_dir / "generation_pilot_semantic_review_plan.json",
    }
    packet_rows = cases_to_packet_rows(cases)
    _write_csv(paths["packet"], packet_rows, fieldnames=PACKET_FIELDS)
    _write_reviewer_csv(paths["reviewer_a"], cases, annotator_id="reviewer_a")
    _write_reviewer_csv(paths["reviewer_b"], cases, annotator_id="reviewer_b")
    paths["readable_zh"].write_text(
        readable_review(cases, language="ZH"), encoding="utf-8"
    )
    paths["readable_en"].write_text(
        readable_review(cases, language="EN"), encoding="utf-8"
    )
    paths["guidelines_zh"].write_text(
        annotation_guidelines(language="ZH"), encoding="utf-8"
    )
    paths["guidelines_en"].write_text(
        annotation_guidelines(language="EN"), encoding="utf-8"
    )
    write_json(paths["audit_json"], audit)
    paths["audit_md"].write_text(semantic_audit_markdown(audit), encoding="utf-8")
    write_json(
        paths["cases"], [case.model_dump(mode="json") for case in cases]
    )
    paths["changelog"].write_text(
        """# V7 → V8 changelog

- V7 remains immutable diagnostic input; no V7 file is edited, deleted, or overwritten.
- Canonical memory target is now `materially_useful_memory_sources`; legacy `needed_memory_sources` is a compatibility alias with the same material-value meaning.
- Strategy RAG is modeled as an ESC resource rather than a synonym for planning, but its current card evidence and on/off labels remain provisional until the inherited V1 bank/retriever audit.
- Advice readiness is a separate target. `advice_harmful` explicitly includes helpful listening RS evidence and harmful directive evidence.
- Canonical human review expands from 8 to 12 dimensions, including grounding, strategy-item utility, advice readiness, and surface naturalness.
- Memory ages and row order use independent reproducible RNG streams that never take utility as input; automatic shortcut audits are fail-closed.
- `multi_source_needed` becomes `multi_source_useful`; the claim is material, nonredundant marginal value rather than absolute indispensability.
- Broad `semantic_family` and narrower `semantic_subtopics` are both displayed.
- Summary, authorized context, and current corrections include claim-level provenance bound to visible text.
- V8 surfaces are new and are checked against V7 for exact reuse.
""",
        encoding="utf-8",
    )
    scope = "smoke" if cases_per_regime == 1 else "validation"
    paths["readme"].write_text(
        f"""# PM V2 generation semantic review V8

- Protocol: `{V8_PROTOCOL}`
- Scope: `{scope}`
- Cases: `{len(cases)}` ({cases_per_regime} per regime)
- Seed: `{seed}`
- Automatic non-strategy safety checks: `{audit['technical_checks_status']}`
- Overall status: `{PROVISIONAL_STATUS}`

PROVISIONAL / NOT READY FOR ANNOTATION / NOT FOR TRAINING. Do not fill reviewer CSVs yet. First audit and freeze the inherited V1 Strategy RAG card bank and production retriever. All current strategy on/off and card-utility labels are hypotheses only.

当前 v8 是 provisional draft，不得开始正式标注或训练，需先完成 V1 Strategy RAG audit。
""",
        encoding="utf-8",
    )
    protected_items = [
        {
            "item_id": row["item_id"],
            "protected_packet_row_sha256": sha256_text(
                canonical_json({field: row[field] for field in PROTECTED_FIELDS})
            ),
        }
        for row in packet_rows
    ]
    plan = {
        "protocol": V8_PROTOCOL,
        "scope": scope,
        "cases_per_regime": cases_per_regime,
        "case_count": len(cases),
        "seed": seed,
        "minimum_annotators": MINIMUM_ANNOTATORS,
        "rating_fields": list(RATING_FIELDS),
        "strategy_bank_path": str(strategy_bank_path),
        "strategy_bank_sha256": sha256_file(strategy_bank_path),
        "packet_sha256": sha256_file(paths["packet"]),
        "readable_zh_sha256": sha256_file(paths["readable_zh"]),
        "readable_en_sha256": sha256_file(paths["readable_en"]),
        "guidelines_zh_sha256": sha256_file(paths["guidelines_zh"]),
        "guidelines_en_sha256": sha256_file(paths["guidelines_en"]),
        "audit_json_sha256": sha256_file(paths["audit_json"]),
        "cases_sha256": sha256_file(paths["cases"]),
        "v7_immutable_hashes": v7_hashes_before,
        "items": protected_items,
    }
    plan["plan_sha256"] = sha256_text(canonical_json(plan))
    write_json(paths["plan"], plan)
    if _v7_hash_manifest(v7_dir) != v7_hashes_before:
        raise RuntimeError("V7 changed while preparing V8")
    create_artifact_attestation(
        out_dir / "preparation_attestation.json",
        stage=V8_AUDIT_STAGE,
        inputs={
            "strategy_bank": strategy_bank_path,
            **{
                f"v7_{name}": v7_dir / name
                for name in sorted(v7_hashes_before)
            },
        },
        outputs={name: (path, False) for name, path in paths.items()},
        parameters={
            "protocol": V8_PROTOCOL,
            "scope": scope,
            "cases_per_regime": cases_per_regime,
            "seed": seed,
            "plan_sha256": plan["plan_sha256"],
        },
        expected={
            "technical_checks_status": "PASS",
            "overall_status": PROVISIONAL_STATUS,
            "case_count": len(cases),
        },
    )
    return {
        "status": PROVISIONAL_STATUS,
        "scope": scope,
        "case_count": len(cases),
        "out_dir": str(out_dir),
        "audit_status": audit["status"],
        "technical_checks_status": audit["technical_checks_status"],
        "plan_sha256": plan["plan_sha256"],
        "v7_unchanged": _v7_hash_manifest(v7_dir) == v7_hashes_before,
    }


def load_v8_cases(path: str | Path) -> list[V8ReviewCase]:
    value = read_json(path)
    if not isinstance(value, list):
        raise RuntimeError("V8 cases artifact must contain a list")
    return [V8ReviewCase.model_validate(row) for row in value]


def _read_completed_reviews(
    *,
    completed_paths: Sequence[str | Path],
    item_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], list[Path]]:
    expected_items = set(item_ids)
    annotations: list[dict[str, Any]] = []
    completed: list[Path] = []
    global_annotators: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for raw_path in completed_paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise RuntimeError(f"completed review is missing: {path}")
        completed.append(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != SIMPLE_REVIEW_FIELDS:
                raise RuntimeError(f"V8 completed review header differs: {path}")
            rows = list(reader)
        if len(rows) != len(item_ids):
            raise RuntimeError(f"V8 completed review row count differs: {path}")
        file_items: set[str] = set()
        file_annotators: set[str] = set()
        for row in rows:
            item_id = str(row.get("item_id") or "")
            annotator_id = str(row.get("annotator_id") or "").strip()
            if item_id not in expected_items or item_id in file_items:
                raise RuntimeError(f"invalid/duplicate V8 review item: {item_id}")
            if not annotator_id:
                raise RuntimeError(f"missing annotator_id in {path}")
            pair = (item_id, annotator_id)
            if pair in seen_pairs:
                raise RuntimeError(f"duplicate annotator/item pair: {pair}")
            seen_pairs.add(pair)
            file_items.add(item_id)
            file_annotators.add(annotator_id)
            ratings: dict[str, int] = {}
            for field in RATING_FIELDS:
                raw = str(row.get(field) or "").strip()
                if raw not in {"0", "1"}:
                    raise RuntimeError(
                        f"{path}:{item_id}:{field} must be exactly 0 or 1"
                    )
                ratings[field] = int(raw)
            notes = str(row.get("notes") or "").strip()
            if any(value == 0 for value in ratings.values()) and not notes:
                raise RuntimeError(
                    f"{path}:{item_id} has a 0 rating but empty notes"
                )
            annotations.append(
                {
                    "item_id": item_id,
                    "annotator_id": annotator_id,
                    "ratings": ratings,
                    "notes": notes,
                    "path": str(path),
                }
            )
        if file_items != expected_items:
            raise RuntimeError(f"V8 completed review item set differs: {path}")
        if len(file_annotators) != 1:
            raise RuntimeError(f"each V8 review file must use one annotator: {path}")
        global_annotators.update(file_annotators)
    if len(global_annotators) < MINIMUM_ANNOTATORS:
        raise RuntimeError(
            f"V8 review requires {MINIMUM_ANNOTATORS} independent annotators"
        )
    return annotations, completed


def analyze_generation_semantic_review_v8(
    *,
    review_dir: str | Path,
    completed_paths: Sequence[str | Path],
    report_path: str | Path,
    attestation_path: str | Path,
) -> dict[str, Any]:
    if not V1_STRATEGY_RAG_AUDIT_COMPLETE:
        raise RuntimeError(
            "V8 is provisional: audit and freeze the inherited V1 Strategy RAG "
            "card bank and real retriever before starting annotation analysis"
        )
    review_dir = Path(review_dir).resolve()
    report_path = Path(report_path).resolve()
    attestation_path = Path(attestation_path).resolve()
    if report_path.exists() or attestation_path.exists():
        raise RuntimeError("refusing to overwrite an existing V8 review result")
    plan_path = review_dir / "generation_pilot_semantic_review_plan.json"
    packet_path = review_dir / "generation_pilot_semantic_review_packet.csv"
    cases_path = review_dir / "generation_pilot_semantic_review_cases.json"
    audit_path = review_dir / "semantic_review_audit.json"
    for path in (plan_path, packet_path, cases_path, audit_path):
        if not path.is_file():
            raise RuntimeError(f"V8 review input missing: {path}")
    plan = read_json(plan_path)
    stored_plan_hash = str(plan.get("plan_sha256") or "")
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if stored_plan_hash != sha256_text(canonical_json(payload)):
        raise RuntimeError("V8 review plan self-hash mismatch")
    if plan.get("protocol") != V8_PROTOCOL:
        raise RuntimeError("V8 review protocol mismatch")
    exact_lineage = (
        plan.get("packet_sha256") == sha256_file(packet_path)
        and plan.get("cases_sha256") == sha256_file(cases_path)
        and plan.get("audit_json_sha256") == sha256_file(audit_path)
        and plan.get("readable_zh_sha256")
        == sha256_file(review_dir / "generation_pilot_semantic_review_readable_ZH.md")
        and plan.get("readable_en_sha256")
        == sha256_file(review_dir / "generation_pilot_semantic_review_readable_EN.md")
        and plan.get("guidelines_zh_sha256")
        == sha256_file(review_dir / "annotation_guidelines_ZH.md")
        and plan.get("guidelines_en_sha256")
        == sha256_file(review_dir / "annotation_guidelines_EN.md")
    )
    if not exact_lineage:
        raise RuntimeError("V8 review lineage is stale")
    audit = read_json(audit_path)
    if audit.get("status") != "PASS" or not all(
        (audit.get("checks") or {}).values()
    ):
        raise RuntimeError("V8 automatic semantic audit is not PASS")
    cases = load_v8_cases(cases_path)
    packet_rows = cases_to_packet_rows(cases)
    with packet_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PACKET_FIELDS:
            raise RuntimeError("V8 packet header differs")
        stored_rows = list(reader)
    if stored_rows != packet_rows:
        raise RuntimeError("V8 packet differs from serialized cases")
    for row, planned in zip(stored_rows, plan["items"], strict=True):
        row_hash = sha256_text(
            canonical_json({field: row[field] for field in PROTECTED_FIELDS})
        )
        if (
            planned.get("item_id") != row["item_id"]
            or planned.get("protected_packet_row_sha256") != row_hash
        ):
            raise RuntimeError("V8 protected packet row differs from plan")

    annotations, completed = _read_completed_reviews(
        completed_paths=completed_paths,
        item_ids=[case.item_id for case in cases],
    )
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotations:
        by_item[row["item_id"]].append(row)
    underannotated = sorted(
        item_id
        for item_id in (case.item_id for case in cases)
        if len(by_item[item_id]) < MINIMUM_ANNOTATORS
    )
    negative = [
        {
            "item_id": row["item_id"],
            "annotator_id": row["annotator_id"],
            "field": field,
            "notes": row["notes"],
        }
        for row in annotations
        for field, value in row["ratings"].items()
        if value == 0
    ]
    pair_agreements: list[int] = []
    for item_rows in by_item.values():
        for left, right in combinations(item_rows, 2):
            pair_agreements.extend(
                int(left["ratings"][field] == right["ratings"][field])
                for field in RATING_FIELDS
            )
    exact_agreement = (
        sum(pair_agreements) / len(pair_agreements)
        if pair_agreements
        else 0.0
    )
    checks = {
        "exact_lineage": exact_lineage,
        "automatic_audit_pass": True,
        "expected_item_count": len(cases)
        == int(plan["cases_per_regime"]) * len(V8_REGIMES),
        "minimum_two_independent_annotators_per_item": not underannotated,
        "all_twelve_ratings_affirmative": not negative,
        "pairwise_exact_agreement": exact_agreement == 1.0,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": V8_PROTOCOL,
        "scope": plan["scope"],
        "cases_per_regime": plan["cases_per_regime"],
        "checks": checks,
        "item_count": len(cases),
        "annotation_count": len(annotations),
        "annotators": sorted({row["annotator_id"] for row in annotations}),
        "underannotated_items": underannotated,
        "negative_ratings": negative,
        "pairwise_exact_agreement": exact_agreement,
        "decision_rule": "all cases x 12 fields x all annotators must equal 1",
        "training_authorized": (
            all(checks.values())
            and plan["scope"] == "validation"
            and plan["cases_per_regime"] == VALIDATION_CASES_PER_REGIME
        ),
    }
    write_json(report_path, report)
    inputs: dict[str, Path] = {
        "plan": plan_path,
        "packet": packet_path,
        "cases": cases_path,
        "automatic_audit": audit_path,
        "readable_zh": review_dir
        / "generation_pilot_semantic_review_readable_ZH.md",
        "readable_en": review_dir
        / "generation_pilot_semantic_review_readable_EN.md",
    }
    for index, path in enumerate(completed, 1):
        inputs[f"completed_review_{index:02d}"] = path
    create_artifact_attestation(
        attestation_path,
        stage=V8_REVIEW_STAGE,
        inputs=inputs,
        outputs={"report": (report_path, False)},
        parameters={
            "protocol": V8_PROTOCOL,
            "scope": plan["scope"],
            "cases_per_regime": plan["cases_per_regime"],
            "plan_sha256": stored_plan_hash,
            "minimum_annotators": MINIMUM_ANNOTATORS,
            "required_all_affirmative": True,
        },
        expected={
            "status": "PASS",
            "training_authorized_only_for_validation": True,
        },
    )
    return report


def require_generation_semantic_review_v8(
    attestation_path: str | Path, *, require_validation: bool = True
) -> dict[str, Any]:
    if not V1_STRATEGY_RAG_AUDIT_COMPLETE:
        raise RuntimeError(
            "formal generation is blocked until the V1 Strategy RAG card bank "
            "and production retriever are audited and frozen"
        )
    attestation_path = Path(attestation_path).resolve()
    if not attestation_path.is_file():
        raise RuntimeError(
            "formal PM-v2 generation requires a PASS V8 validation review: "
            f"{attestation_path}"
        )
    attestation = read_json(attestation_path)
    report_record = (attestation.get("outputs") or {}).get("report") or {}
    report_path = Path(str(report_record.get("path") or "")).resolve()
    verification = require_artifact_attestation(
        attestation_path,
        required_stage=V8_REVIEW_STAGE,
        required_output_paths={"report": report_path},
    )
    parameters = attestation.get("parameters") or {}
    if parameters.get("protocol") != V8_PROTOCOL:
        raise RuntimeError("formal generation review protocol is not V8")
    report = read_json(report_path)
    if report.get("status") != "PASS" or not all(
        (report.get("checks") or {}).values()
    ):
        raise RuntimeError("V8 generation semantic review is not PASS")
    if require_validation and not report.get("training_authorized"):
        raise RuntimeError(
            "V8 smoke review cannot authorize formal generation; complete the "
            "27-case two-reviewer validation gate"
        )
    return {
        **verification,
        "status": "PASS",
        "protocol": V8_PROTOCOL,
        "scope": report["scope"],
        "cases_per_regime": report["cases_per_regime"],
        "review_report_sha256": sha256_file(report_path),
        "annotators": report["annotators"],
        "annotation_count": report["annotation_count"],
        "training_authorized": report["training_authorized"],
    }
