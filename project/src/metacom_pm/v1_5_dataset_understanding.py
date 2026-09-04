"""Outcome-blind structural profiling for the local ESConv and EvoEmo data.

The profile is deliberately descriptive.  ESConv strategy annotations are
observed supporter moves, not optimal PM actions; seeker feedback and survey
scores are post-treatment observations.  EvoEmo questions, reference answers,
related-session labels, observations, and future-topic text are evaluator-only
and never become training labels here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from .contracts import MemorySource
from .evoemo import build_evo_memory, evo_memory_builder_contract_hash
from .io import canonical_json, sha256_text
from .text import normalize_for_hash, normalize_space


DATASET_UNDERSTANDING_PROTOCOL = "pm-v1.5-esconv-evoemo-understanding-v1"

_ADVICE_REQUEST_RE = re.compile(
    r"\b(what (?:should|can|could) i do|any advice|help me|suggest|recommend|"
    r"how (?:do|can|should) i)\b",
    re.IGNORECASE,
)
_LISTEN_BOUNDARY_RE = re.compile(
    r"\b(just listen|only listen|don'?t (?:give|offer) (?:me )?advice|"
    r"not looking for advice|need to vent|let me vent|"
    r"need (?:someone|you) to listen|just (?:want|need) to talk)\b",
    re.IGNORECASE,
)
_DIRECTIVE_RE = re.compile(
    r"\b(you should|you could|try to|consider|recommend|suggest|"
    r"it may help|you need to)\b",
    re.IGNORECASE,
)
_FIRST_PERSON_RE = re.compile(
    r"\b(i|i'm|i've|my|me|we|our)\b",
    re.IGNORECASE,
)
_GREETING_OR_CLOSING_RE = re.compile(
    r"^(hi|hello|hey|thanks?|thank you|bye|goodbye|you'?re welcome|"
    r"you are welcome)[.! ]*$",
    re.IGNORECASE,
)


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _quantiles(values: Iterable[float]) -> dict[str, float | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None}

    def at(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "min": _round(ordered[0]),
        "p25": _round(at(0.25)),
        "median": _round(at(0.5)),
        "p75": _round(at(0.75)),
        "max": _round(ordered[-1]),
    }


def _entropy(counter: Mapping[str, int]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counter.values()
        if count > 0
    )


def _mutual_information(rows: Sequence[tuple[str, str]]) -> float:
    if not rows:
        return 0.0
    x_counts = Counter(x for x, _ in rows)
    y_counts = Counter(y for _, y in rows)
    joint = Counter(rows)
    total = len(rows)
    result = 0.0
    for (x_value, y_value), count in joint.items():
        p_xy = count / total
        p_x = x_counts[x_value] / total
        p_y = y_counts[y_value] / total
        result += p_xy * math.log2(p_xy / (p_x * p_y))
    return result


def _normalized_mi(rows: Sequence[tuple[str, str]]) -> float:
    if not rows:
        return 0.0
    y_entropy = _entropy(Counter(y for _, y in rows))
    if y_entropy == 0.0:
        return 0.0
    return _mutual_information(rows) / y_entropy


def _numeric(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _ordered_counter(counter: Mapping[str, int]) -> dict[str, int]:
    return {
        key: int(counter[key])
        for key in sorted(counter, key=lambda item: (-counter[item], item))
    }


def _stage(index: int, count: int) -> str:
    if count <= 1:
        return "only"
    fraction = index / (count - 1)
    if fraction < 1 / 3:
        return "early"
    if fraction < 2 / 3:
        return "middle"
    return "late"


def _strategy_decisions(
    esconv: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for dialogue_index, (dialogue_row, split_row) in enumerate(
        zip(esconv, split_rows, strict=True)
    ):
        dialogue_id = f"esconv_{dialogue_index:04d}"
        if (
            int(split_row.get("index", -1)) != dialogue_index
            or str(split_row.get("dialogue_id")) != dialogue_id
        ):
            raise RuntimeError("ESConv split manifest is reordered")
        if split_row.get("split") != "train" or bool(
            split_row.get("excluded_for_evoemo_overlap")
        ):
            continue
        turns = list(dialogue_row.get("dialog") or [])
        supporter_indices = [
            index
            for index, turn in enumerate(turns)
            if turn.get("speaker") == "supporter"
            and normalize_space(turn.get("content") or "")
        ]
        previous_strategy = "START"
        latest_seeker_text = ""
        supporter_ordinal = 0
        for turn_index, turn in enumerate(turns):
            speaker = str(turn.get("speaker") or "")
            content = normalize_space(turn.get("content") or "")
            if speaker == "seeker" and content:
                latest_seeker_text = content
                continue
            if speaker != "supporter" or not content:
                continue
            strategy = normalize_space(
                (turn.get("annotation") or {}).get("strategy") or "Others"
            )
            feedback = None
            for next_turn in turns[turn_index + 1 :]:
                if next_turn.get("speaker") == "supporter":
                    break
                if next_turn.get("speaker") == "seeker":
                    feedback = _numeric(
                        (next_turn.get("annotation") or {}).get("feedback")
                    )
                    break
            decisions.append(
                {
                    "dialogue_id": dialogue_id,
                    "problem_type": normalize_space(
                        dialogue_row.get("problem_type") or "unknown"
                    ),
                    "emotion_type": normalize_space(
                        dialogue_row.get("emotion_type") or "unknown"
                    ),
                    "experience_type": normalize_space(
                        dialogue_row.get("experience_type") or "unknown"
                    ),
                    "strategy": strategy,
                    "previous_strategy": previous_strategy,
                    "stage": _stage(supporter_ordinal, len(supporter_indices)),
                    "current_user_text": latest_seeker_text,
                    "has_prior_seeker_turn": bool(latest_seeker_text),
                    "advice_request_cue": bool(
                        _ADVICE_REQUEST_RE.search(latest_seeker_text)
                    ),
                    "listen_boundary_cue": bool(
                        _LISTEN_BOUNDARY_RE.search(latest_seeker_text)
                    ),
                    "question_mark_cue": "?" in latest_seeker_text,
                    "current_user_word_count": len(latest_seeker_text.split()),
                    "supporter_has_question_mark": "?" in content,
                    "supporter_has_directive_cue": bool(
                        _DIRECTIVE_RE.search(content)
                    ),
                    "supporter_has_first_person_cue": bool(
                        _FIRST_PERSON_RE.search(content)
                    ),
                    "supporter_greeting_or_closing_only": bool(
                        _GREETING_OR_CLOSING_RE.fullmatch(content)
                    ),
                    "supporter_word_count": len(content.split()),
                    "normalized_response": normalize_for_hash(content),
                    "feedback": feedback,
                }
            )
            previous_strategy = strategy
            supporter_ordinal += 1
    return decisions


def profile_esconv(
    esconv: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Profile ESConv without using validation/test strategy or outcome rows."""

    if len(esconv) != len(split_rows):
        raise RuntimeError("ESConv and split manifest lengths differ")
    split_counts = Counter(str(row.get("split")) for row in split_rows)
    overlap_counts = Counter(
        str(row.get("split"))
        for row in split_rows
        if bool(row.get("excluded_for_evoemo_overlap"))
    )
    all_decisions = _strategy_decisions(esconv, split_rows)
    decisions = [
        row for row in all_decisions if bool(row["has_prior_seeker_turn"])
    ]
    eligible_dialogues = [
        row
        for row, split_row in zip(esconv, split_rows, strict=True)
        if split_row.get("split") == "train"
        and not bool(split_row.get("excluded_for_evoemo_overlap"))
    ]
    strategy_counts = Counter(str(row["strategy"]) for row in decisions)
    transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    stage_counts: dict[str, Counter[str]] = defaultdict(Counter)
    feedback_by_strategy: dict[str, list[float]] = defaultdict(list)
    duplicate_counts = Counter(
        str(row["normalized_response"])
        for row in all_decisions
        if row["normalized_response"]
    )
    for row in decisions:
        transition_counts[str(row["previous_strategy"])][str(row["strategy"])] += 1
        stage_counts[str(row["stage"])][str(row["strategy"])] += 1
        if row["feedback"] is not None:
            feedback_by_strategy[str(row["strategy"])].append(float(row["feedback"]))

    def strategy_distribution(predicate) -> dict[str, Any]:
        selected = [row for row in decisions if predicate(row)]
        counts = Counter(str(row["strategy"]) for row in selected)
        return {
            "n": len(selected),
            "strategy_counts": _ordered_counter(counts),
            "strategy_shares": {
                key: _round(count / len(selected))
                for key, count in _ordered_counter(counts).items()
            }
            if selected
            else {},
        }

    n_question = sum(
        bool(row["supporter_has_question_mark"]) for row in all_decisions
    )
    n_directive = sum(
        bool(row["supporter_has_directive_cue"]) for row in all_decisions
    )
    n_first_person = sum(
        bool(row["supporter_has_first_person_cue"]) for row in all_decisions
    )
    label_question = sum(
        row["strategy"] == "Question" for row in all_decisions
    )
    label_suggestion = sum(
        row["strategy"] == "Providing Suggestions" for row in all_decisions
    )
    label_self_disclosure = sum(
        row["strategy"] == "Self-disclosure" for row in all_decisions
    )
    post_treatment_surveys = []
    for index, row in enumerate(esconv):
        split_row = split_rows[index]
        if split_row.get("split") != "train" or bool(
            split_row.get("excluded_for_evoemo_overlap")
        ):
            continue
        survey = row.get("survey_score") or {}
        seeker = survey.get("seeker") or {}
        post_treatment_surveys.append(
            {
                "initial": _numeric(seeker.get("initial_emotion_intensity")),
                "final": _numeric(seeker.get("final_emotion_intensity")),
                "empathy": _numeric(seeker.get("empathy")),
                "relevance": _numeric(seeker.get("relevance")),
                "supporter_relevance": _numeric(
                    (survey.get("supporter") or {}).get("relevance")
                ),
            }
        )

    survey_fields = {}
    for field in (
        "initial",
        "final",
        "empathy",
        "relevance",
        "supporter_relevance",
    ):
        values = [
            float(row[field])
            for row in post_treatment_surveys
            if row[field] is not None
        ]
        survey_fields[field] = {
            "n": len(values),
            "missing": len(post_treatment_surveys) - len(values),
            "mean": _round(_mean(values)),
            "distribution": _ordered_counter(
                Counter(str(value) for value in values)
            ),
        }

    feature_pairs = {
        "previous_strategy": [
            (str(row["previous_strategy"]), str(row["strategy"]))
            for row in decisions
        ],
        "dialogue_stage": [
            (str(row["stage"]), str(row["strategy"])) for row in decisions
        ],
        "problem_type": [
            (str(row["problem_type"]), str(row["strategy"])) for row in decisions
        ],
        "emotion_type": [
            (str(row["emotion_type"]), str(row["strategy"])) for row in decisions
        ],
        "experience_type": [
            (str(row["experience_type"]), str(row["strategy"])) for row in decisions
        ],
        "advice_request_cue": [
            (str(bool(row["advice_request_cue"])), str(row["strategy"]))
            for row in decisions
        ],
        "question_mark_cue": [
            (str(bool(row["question_mark_cue"])), str(row["strategy"]))
            for row in decisions
        ],
    }
    return {
        "analysis_scope": {
            "all_dialogue_count_for_structure_only": len(esconv),
            "split_counts": _ordered_counter(split_counts),
            "evoemo_overlap_counts_by_split": _ordered_counter(overlap_counts),
            "strategy_and_outcome_analysis_split": "train",
            "evoemo_overlap_excluded": True,
            "validation_and_test_strategy_annotations_opened": False,
            "validation_and_test_survey_outcomes_opened": False,
        },
        "train_observed_strategy_decisions": len(decisions),
        "train_supporter_turns_before_any_seeker": len(all_decisions)
        - len(decisions),
        "train_dialogues_with_decisions": len(
            {str(row["dialogue_id"]) for row in decisions}
        ),
        "train_dialogue_metadata": {
            "problem_type_counts": _ordered_counter(
                Counter(
                    normalize_space(row.get("problem_type") or "unknown")
                    for row in eligible_dialogues
                )
            ),
            "emotion_type_counts": _ordered_counter(
                Counter(
                    normalize_space(row.get("emotion_type") or "unknown")
                    for row in eligible_dialogues
                )
            ),
            "experience_type_counts": _ordered_counter(
                Counter(
                    normalize_space(row.get("experience_type") or "unknown")
                    for row in eligible_dialogues
                )
            ),
            "current_user_word_count_at_decision": _quantiles(
                int(row["current_user_word_count"]) for row in decisions
            ),
        },
        "strategy_counts": _ordered_counter(strategy_counts),
        "strategy_entropy_bits": _round(_entropy(strategy_counts)),
        "normalized_mutual_information_with_observed_strategy": {
            key: _round(_normalized_mi(rows))
            for key, rows in feature_pairs.items()
        },
        "strategy_by_stage": {
            stage: _ordered_counter(counter)
            for stage, counter in sorted(stage_counts.items())
        },
        "strategy_transitions": {
            previous: _ordered_counter(counter)
            for previous, counter in sorted(transition_counts.items())
        },
        "transparent_user_cues": {
            "advice_request": strategy_distribution(
                lambda row: bool(row["advice_request_cue"])
            ),
            "listen_boundary": strategy_distribution(
                lambda row: bool(row["listen_boundary_cue"])
            ),
            "question_mark": strategy_distribution(
                lambda row: bool(row["question_mark_cue"])
            ),
        },
        "turn_level_feedback_post_treatment_report_only": {
            "available_n": sum(
                len(values) for values in feedback_by_strategy.values()
            ),
            "missing_n": len(decisions)
            - sum(len(values) for values in feedback_by_strategy.values()),
            "by_observed_strategy": {
                strategy: {
                    "n": len(values),
                    "mean": _round(_mean(values)),
                    "distribution": _ordered_counter(
                        Counter(str(value) for value in values)
                    ),
                }
                for strategy, values in sorted(feedback_by_strategy.items())
            },
            "causal_or_optimal_action_label": False,
        },
        "conversation_survey_post_treatment_report_only": {
            "dialogue_n": len(post_treatment_surveys),
            "fields": survey_fields,
            "causal_or_turn_level_label": False,
        },
        "surface_quality_diagnostics": {
            "normalized_duplicate_groups": sum(
                count > 1 for count in duplicate_counts.values()
            ),
            "rows_in_normalized_duplicate_groups": sum(
                count for count in duplicate_counts.values() if count > 1
            ),
            "maximum_normalized_duplicate_group": max(
                duplicate_counts.values(), default=0
            ),
            "greeting_or_closing_only_rows": sum(
                bool(row["supporter_greeting_or_closing_only"])
                for row in all_decisions
            ),
            "question_label_rows": label_question,
            "rows_with_question_mark": n_question,
            "question_label_without_question_mark": sum(
                row["strategy"] == "Question"
                and not row["supporter_has_question_mark"]
                for row in all_decisions
            ),
            "nonquestion_label_with_question_mark": sum(
                row["strategy"] != "Question"
                and row["supporter_has_question_mark"]
                for row in all_decisions
            ),
            "suggestion_label_rows": label_suggestion,
            "rows_with_directive_cue": n_directive,
            "suggestion_label_without_directive_cue": sum(
                row["strategy"] == "Providing Suggestions"
                and not row["supporter_has_directive_cue"]
                for row in all_decisions
            ),
            "nonsuggestion_label_with_directive_cue": sum(
                row["strategy"] != "Providing Suggestions"
                and row["supporter_has_directive_cue"]
                for row in all_decisions
            ),
            "self_disclosure_label_rows": label_self_disclosure,
            "rows_with_first_person_cue": n_first_person,
            "short_supporter_rows_le_3_words": sum(
                int(row["supporter_word_count"]) <= 3 for row in all_decisions
            ),
            "interpretation": (
                "Surface cues are diagnostics, not proof that an annotation is wrong."
            ),
        },
        "label_semantics": {
            "strategy": "observed_supporter_move_not_optimal_pm_action",
            "turn_feedback": "post_response_subjective_feedback_not_causal_gold",
            "survey": "post_treatment_conversation_level_report_only",
        },
    }


def _valid_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def profile_evoemo(users: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Profile deployable EvoEmo inputs while keeping evaluator answers sealed."""

    session_counts: list[int] = []
    event_counts: list[int] = []
    memory_counts: dict[str, list[int]] = {
        source.value: [] for source in MemorySource
    }
    memory_lengths: dict[str, list[int]] = {
        source.value: [] for source in MemorySource
    }
    memory_ages_at_future_query: dict[str, list[int]] = {
        source.value: [] for source in MemorySource
    }
    session_turn_counts: list[int] = []
    session_gap_days: list[int] = []
    session_summary_lengths: list[int] = []
    topic_counts: Counter[str] = Counter()
    emotion_counts: Counter[str] = Counter()
    capability_counts: Counter[str] = Counter()
    summary_capability_counts: Counter[str] = Counter()
    subsequent_topic_counts: list[int] = []
    related_session_counts: list[int] = []
    esc_source_indices: set[int] = set()
    invalid_esc_source_indices: list[int] = []
    invalid_event_conv_refs = 0
    invalid_related_session_refs = 0
    invalid_influenced_by_refs = 0
    duplicate_session_ids = 0
    empty_dialogue_turns = 0
    same_role_adjacent_turn_pairs = 0
    supporter_short_rows = 0
    supporter_rows = 0
    supporter_normalized = Counter()
    input_order_date_reversals = 0
    input_order_date_pair_inversions = 0
    users_with_date_order_inversions = 0

    for user in users:
        sessions = list(user.get("dialog_history") or [])
        events = list(user.get("event_experience") or [])
        session_counts.append(len(sessions))
        event_counts.append(len(events))
        session_ids = [str(row.get("id") or "") for row in sessions]
        duplicate_session_ids += len(session_ids) - len(set(session_ids))
        session_id_set = set(session_ids)
        event_ids = {str(row.get("id") or "") for row in events}
        previous_date = None
        parsed_dates = [
            _valid_date(session.get("timestamp")) for session in sessions
        ]
        user_pair_inversions = sum(
            first is not None and second is not None and first > second
            for index, first in enumerate(parsed_dates)
            for second in parsed_dates[index + 1 :]
        )
        input_order_date_pair_inversions += user_pair_inversions
        users_with_date_order_inversions += user_pair_inversions > 0
        for session in sessions:
            session_id = str(session.get("id") or "")
            match = re.fullmatch(r"esc(\d+)", session_id)
            if match:
                source_index = int(match.group(1))
                esc_source_indices.add(source_index)
                if not 0 <= source_index < 1300:
                    invalid_esc_source_indices.append(source_index)
            parsed_date = _valid_date(session.get("timestamp"))
            if previous_date is not None and parsed_date is not None:
                gap = (parsed_date - previous_date).days
                if gap < 0:
                    input_order_date_reversals += 1
                else:
                    session_gap_days.append(gap)
            if parsed_date is not None:
                previous_date = parsed_date
            summary = normalize_space(session.get("summary") or "")
            if summary:
                session_summary_lengths.append(len(summary.split()))
            topic_counts[normalize_space(session.get("topic") or "unknown")] += 1
            emotion_counts[
                normalize_space(session.get("emotion") or "unknown")
            ] += 1
            dialogue = list(session.get("dialogue") or [])
            session_turn_counts.append(len(dialogue))
            previous_role = None
            for turn in dialogue:
                role = str(turn.get("role") or "")
                content = normalize_space(turn.get("content") or "")
                if not content:
                    empty_dialogue_turns += 1
                if previous_role == role:
                    same_role_adjacent_turn_pairs += 1
                previous_role = role
                if role == "supporter" and content:
                    supporter_rows += 1
                    supporter_short_rows += len(content.split()) <= 3
                    supporter_normalized[normalize_for_hash(content)] += 1

        for event in events:
            conv_id = str(event.get("conv_id") or "")
            if conv_id and conv_id not in session_id_set:
                invalid_event_conv_refs += 1
            for dependency in event.get("influenced_by") or []:
                if str(dependency) not in event_ids:
                    invalid_influenced_by_refs += 1

        items, _ = build_evo_memory(dict(user))
        counts = Counter(item.source.value for item in items)
        for source in MemorySource:
            memory_counts[source.value].append(int(counts[source.value]))
        for item in items:
            memory_lengths[item.source.value].append(len(item.text.split()))
            memory_ages_at_future_query[item.source.value].append(
                len(sessions) + 1 - int(item.created_session)
            )

        questions = list(user.get("questions") or [])
        for group in questions:
            for question in group.get("questions") or []:
                capability_counts[
                    normalize_space(question.get("capability") or "unknown")
                ] += 1
        for row in user.get("summaries") or []:
            summary_capability_counts[
                normalize_space(row.get("capability") or "unknown")
            ] += 1

        topics = list(user.get("subsequent_topics") or [])
        subsequent_topic_counts.append(len(topics))
        for topic in topics:
            related = [str(value) for value in topic.get("related_sessions") or []]
            related_session_counts.append(len(related))
            invalid_related_session_refs += sum(
                value not in session_id_set for value in related
            )

    return {
        "user_count": len(users),
        "session_count": sum(session_counts),
        "session_count_per_user": _quantiles(session_counts),
        "event_count": sum(event_counts),
        "event_count_per_user": _quantiles(event_counts),
        "dialogue_turn_count": sum(session_turn_counts),
        "session_turn_count": _quantiles(session_turn_counts),
        "session_gap_days_nonnegative": _quantiles(session_gap_days),
        "session_summary_word_count": _quantiles(session_summary_lengths),
        "topic_counts": _ordered_counter(topic_counts),
        "emotion_counts": _ordered_counter(emotion_counts),
        "deterministic_esconv_overlap": {
            "session_count": len(esc_source_indices),
            "unique_esconv_source_indices": len(esc_source_indices),
            "invalid_source_indices": sorted(invalid_esc_source_indices),
            "not_independent_of_esconv": bool(esc_source_indices),
        },
        "deployable_memory_catalog": {
            "builder_contract_sha256": evo_memory_builder_contract_hash(),
            "counts_per_user": {
                source: _quantiles(values)
                for source, values in sorted(memory_counts.items())
            },
            "item_word_count": {
                source: _quantiles(values)
                for source, values in sorted(memory_lengths.items())
            },
            "item_age_sessions_at_future_query": {
                source: _quantiles(values)
                for source, values in sorted(memory_ages_at_future_query.items())
            },
            "total_items": {
                source: sum(values)
                for source, values in sorted(memory_counts.items())
            },
            "evaluator_only_fields_excluded": [
                "event_experience",
                "dialog_history.observation",
                "questions",
                "summaries",
                "subsequent_topics",
            ],
        },
        "evaluator_schema_only": {
            "question_capability_counts": _ordered_counter(capability_counts),
            "summary_capability_counts": _ordered_counter(
                summary_capability_counts
            ),
            "subsequent_topic_count": sum(subsequent_topic_counts),
            "subsequent_topics_per_user": _quantiles(subsequent_topic_counts),
            "related_sessions_per_topic": _quantiles(related_session_counts),
            "question_text_opened_for_modeling": False,
            "answer_text_opened_for_modeling": False,
            "evidence_labels_used_for_modeling": False,
            "related_session_labels_used_for_modeling": False,
            "future_topic_text_used_for_modeling": False,
        },
        "integrity_diagnostics": {
            "duplicate_session_ids": duplicate_session_ids,
            "invalid_event_conv_refs": invalid_event_conv_refs,
            "invalid_influenced_by_refs": invalid_influenced_by_refs,
            "invalid_related_session_refs": invalid_related_session_refs,
            "input_order_date_reversals": input_order_date_reversals,
            "input_order_date_pair_inversions": input_order_date_pair_inversions,
            "users_with_date_order_inversions": users_with_date_order_inversions,
            "empty_dialogue_turns": empty_dialogue_turns,
            "same_role_adjacent_turn_pairs": same_role_adjacent_turn_pairs,
        },
        "supporter_surface_quality_diagnostics": {
            "supporter_rows": supporter_rows,
            "short_rows_le_3_words": supporter_short_rows,
            "normalized_duplicate_groups": sum(
                count > 1 for count in supporter_normalized.values()
            ),
            "rows_in_normalized_duplicate_groups": sum(
                count
                for count in supporter_normalized.values()
                if count > 1
            ),
            "maximum_normalized_duplicate_group": max(
                supporter_normalized.values(), default=0
            ),
            "clean_response_quality_training_corpus": False,
        },
        "data_role": {
            "deployable_input_shape_source": True,
            "longitudinal_memory_routing_evaluation_source": True,
            "clean_support_response_training_source": False,
            "questions_and_future_topics_are_training_labels": False,
        },
    }


def build_dataset_understanding_profile(
    *,
    esconv: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
    evoemo_users: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    evoemo_chronology_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    core = {
        "protocol": DATASET_UNDERSTANDING_PROTOCOL,
        "source_hashes": dict(sorted(source_hashes.items())),
        "esconv": profile_esconv(esconv, split_rows),
        "evoemo": profile_evoemo(evoemo_users),
        "cross_domain_conclusions": {
            "one_shared_pm": True,
            "shared_decision_axes": [
                "current_support_mode_and_goal",
                "dialogue_phase_and_user_boundary",
                "strategy_opportunity_and_readiness",
                "memory_source_availability_and_opportunity",
                "multi_source_interference",
                "uncertainty_ood_and_abstention",
                "deterministic_cost",
            ],
            "esconv_unique_role": (
                "single-session memory-unavailable boundary plus noisy observed "
                "support-move and post-treatment feedback signals"
            ),
            "evoemo_unique_role": (
                "longitudinal memory-catalog shape and external memory-routing "
                "evaluation; not clean response-quality supervision"
            ),
            "safe_weak_supervision": [
                "ESConv observed strategy as a low-weight move prior",
                "ESConv seeker feedback as post-response confidence evidence only",
                "ESConv problem/emotion/experience metadata for coverage stratification",
                "EvoEmo catalog count/age/session-index ranges for outcome-free shape augmentation",
            ],
            "forbidden_or_invalid_supervision": [
                "ESConv observed strategy as optimal RS gold",
                "ESConv survey score as turn-level causal utility",
                "EvoEmo question answers or evidence as PM training labels",
                "EvoEmo subsequent-topic text or related_sessions for training/tuning",
                "EvoEmo supporter turns as a clean response-quality corpus",
            ],
        },
        "api_calls_made": 0,
        "internal_test_outcomes_opened": False,
        "external_answer_or_future_outcome_text_used_for_modeling": False,
    }
    core["evoemo"]["source_chronology_normalization"] = (
        dict(evoemo_chronology_report)
        if evoemo_chronology_report is not None
        else {"status": "NOT_PROVIDED"}
    )
    return {**core, "profile_sha256": sha256_text(canonical_json(core))}
