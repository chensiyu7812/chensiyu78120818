"""Deterministic, outcome-blind Strategy Bank V2 candidate construction.

The raw V1 bank remains provenance only.  This module creates a small,
transparent technique library and uses eligible ESConv train rows only as
source coverage evidence.  Raw supporter responses are never copied into the
V2 prompt surface.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import re
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import Field, field_validator, model_validator

from .io import canonical_json, iter_jsonl, sha256_file, sha256_text, stable_hex
from .pm_v2_contracts import StrictModel
from .strategy_bank import load_esconv
from .text import normalize_for_hash, normalize_space


STRATEGY_BANK_V2_PROTOCOL = "pm-v1.5-strategy-bank-v2-technique-candidate-v1"
STRATEGY_BANK_V2_QUALITY_STATUS = (
    "RULE_FILTERED_PENDING_LLM_WEAK_AND_HUMAN_AUDIT"
)
FORMAL_TECHNIQUE_FAMILIES = (
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Affirmation and Reassurance",
    "Providing Suggestions",
)

_FAMILY_SPEC: dict[str, dict[str, Any]] = {
    "Question": {
        "support_move": "Ask one focused, low-burden question that helps the user clarify the most relevant feeling, fact, or goal.",
        "when_to_use": "Use during exploration when one answer would materially improve understanding.",
        "when_not_to_use": "Do not stack questions, repeat already answered facts, or question the user when they explicitly ask to be heard without probing.",
        "modes": ["explore"],
        "phases": ["exploration"],
        "goals": ["make_sense", "decide"],
        "burden": "light",
        "risk_flags": ["question_burden", "repeated_question"],
    },
    "Restatement or Paraphrasing": {
        "support_move": "Briefly paraphrase the user's central concern and leave room for correction.",
        "when_to_use": "Use when accurate understanding and being heard matter more than adding new advice.",
        "when_not_to_use": "Do not introduce motives, diagnoses, facts, or stronger conclusions than the user stated.",
        "modes": ["listen", "explore"],
        "phases": ["exploration", "comforting"],
        "goals": ["be_heard", "make_sense"],
        "burden": "none",
        "risk_flags": ["unsupported_inference"],
    },
    "Reflection of feelings": {
        "support_move": "Reflect the emotion or tension that is directly supported by the user's words, using tentative language when uncertain.",
        "when_to_use": "Use when naming and validating the present emotional experience may reduce load or deepen understanding.",
        "when_not_to_use": "Do not intensify, diagnose, or invent an emotion that is not supported by the visible dialogue.",
        "modes": ["listen", "explore", "comfort_reassure"],
        "phases": ["exploration", "comforting"],
        "goals": ["be_heard", "make_sense", "stabilize"],
        "burden": "none",
        "risk_flags": ["emotion_overreach", "unsupported_inference"],
    },
    "Affirmation and Reassurance": {
        "support_move": "Acknowledge the user's effort or understandable reaction without generic praise or unsupported promises.",
        "when_to_use": "Use when the user needs validation, steadiness, or permission to move at a manageable pace.",
        "when_not_to_use": "Do not minimize the problem, guarantee outcomes, or praise traits not demonstrated in the visible dialogue.",
        "modes": ["comfort_reassure", "listen"],
        "phases": ["comforting"],
        "goals": ["be_heard", "stabilize"],
        "burden": "none",
        "risk_flags": ["generic_praise", "false_reassurance"],
    },
    "Providing Suggestions": {
        "support_move": "Offer at most one optional, low-risk next step that follows the user's stated goal and can be declined.",
        "when_to_use": "Use only when advice is welcome and a concrete small step fits the current goal and phase.",
        "when_not_to_use": "Do not advise after a listen-only boundary, overload the user with multiple tasks, or provide medical, legal, financial, or other domain-specific claims.",
        "modes": ["light_guidance", "structured_planning"],
        "phases": ["action"],
        "goals": ["decide", "act"],
        "burden": "light",
        "risk_flags": [
            "unrequested_advice",
            "directive_overload",
            "domain_specific_claim",
        ],
    },
}

_META_RE = re.compile(
    r"\b(?:survey|questionnaire|platform|mechanical turk|mturk|rate (?:this|the)|"
    r"end (?:this|the) (?:chat|session)|task is complete)\b",
    flags=re.IGNORECASE,
)
_GREETING_RE = re.compile(
    r"^(?:hi|hello|hey|good (?:morning|afternoon|evening)|how are you|"
    r"how'?s (?:it going|things))[\s!,.?]*$",
    flags=re.IGNORECASE,
)
_CLOSING_RE = re.compile(
    r"^(?:thank you|thanks|good luck|take care|bye|goodbye|have a good "
    r"(?:day|night))[\s!,.?]*$",
    flags=re.IGNORECASE,
)
_DOMAIN_CLAIM_RE = re.compile(
    r"\b(?:dose|dosage|milligram|medication|prescription|diagnos(?:e|is)|"
    r"lawyer|legal advice|lawsuit|interest rate|investment|stock market|"
    r"tax advice)\b",
    flags=re.IGNORECASE,
)


class StrategyBankV2Card(StrictModel):
    protocol: Literal[
        "pm-v1.5-strategy-bank-v2-technique-candidate-v1"
    ] = STRATEGY_BANK_V2_PROTOCOL
    card_id: str = Field(min_length=1)
    strategy_family: str = Field(min_length=1)
    support_move: str = Field(min_length=1)
    when_to_use: str = Field(min_length=1)
    when_not_to_use: str = Field(min_length=1)
    compatible_support_modes: list[str]
    compatible_dialogue_phases: list[str]
    goal_types: list[str]
    native_problem_types: dict[str, int]
    native_emotion_types: dict[str, int]
    experience_types: dict[str, int]
    directive_burden: Literal["none", "light", "structured"]
    content_scope: Literal["technique_only", "domain_information"]
    risk_flags: list[str]
    retrieval_text: str = Field(min_length=1)
    prompt_guidance: str = Field(min_length=1)
    source_provenance: dict[str, Any]
    source_quality_observations: dict[str, Any]
    native_metadata_source: Literal["official_esconv"] = "official_esconv"
    derived_metadata_source: Literal[
        "deterministic_family_mapping_v1"
    ] = "deterministic_family_mapping_v1"
    quality_status: Literal[
        "RULE_FILTERED_PENDING_LLM_WEAK_AND_HUMAN_AUDIT"
    ] = STRATEGY_BANK_V2_QUALITY_STATUS
    eligible_for_formal_rs: Literal[False] = False

    @field_validator(
        "compatible_support_modes",
        "compatible_dialogue_phases",
        "goal_types",
        "risk_flags",
    )
    @classmethod
    def unique_nonempty_lists(cls, value):
        if not value or len(value) != len(set(value)):
            raise ValueError("Strategy Bank V2 list fields must be unique/non-empty")
        return value

    @model_validator(mode="after")
    def frozen_candidate_scope(self):
        if self.strategy_family not in FORMAL_TECHNIQUE_FAMILIES:
            raise ValueError("unsupported formal technique family")
        if self.content_scope != "technique_only":
            raise ValueError("first-paper Strategy Bank V2 must be technique-only")
        if "example_response" in self.source_provenance:
            raise ValueError("raw example response cannot enter V2 provenance surface")
        if self.source_provenance.get("selected_packet_source_overlap_count") != 0:
            raise ValueError("Strategy Bank V2 card overlaps the need packet")
        return self


class StrategyBankV2HumanAnnotation(StrictModel):
    card_id: str = Field(min_length=1)
    support_move_clear: bool
    when_to_use_valid: bool
    when_not_to_use_valid: bool
    mode_phase_goal_fit_valid: bool
    burden_and_risk_flags_valid: bool
    safe_general_technique: bool
    approve_for_train_only_pilot: bool
    confidence: int = Field(ge=1, le=5)
    required_corrections: str = ""
    notes: str = ""

    @model_validator(mode="after")
    def coherent_review(self):
        checks = (
            self.support_move_clear,
            self.when_to_use_valid,
            self.when_not_to_use_valid,
            self.mode_phase_goal_fit_valid,
            self.burden_and_risk_flags_valid,
            self.safe_general_technique,
        )
        if self.approve_for_train_only_pilot and not all(checks):
            raise ValueError("a Bank V2 card cannot be approved with a failed check")
        if not all(checks) and not self.required_corrections.strip():
            raise ValueError("a failed Bank V2 card check requires corrections")
        if self.confidence <= 2 and not self.notes.strip():
            raise ValueError("low-confidence Bank V2 review requires notes")
        return self


def validate_strategy_bank_v2_human_annotations(
    *,
    candidate_dir: str | Path,
    annotations_path: str | Path,
) -> dict[str, Any]:
    """Require exact five-card human coverage without promoting formal use."""

    candidate_dir = Path(candidate_dir)
    annotations_path = Path(annotations_path)
    build_report = json.loads(
        (candidate_dir / "build_report.json").read_text(encoding="utf-8")
    )
    report_core = dict(build_report)
    expected_report_sha = str(report_core.pop("report_sha256", ""))
    if expected_report_sha != sha256_text(canonical_json(report_core)):
        raise RuntimeError("Strategy Bank V2 build report hash is invalid")
    review_rows = [
        dict(row)
        for row in iter_jsonl(candidate_dir / "human_review_packet.jsonl")
    ]
    if (
        sha256_text(canonical_json(review_rows))
        != build_report.get("human_review_packet_sha256")
    ):
        raise RuntimeError("Strategy Bank V2 human review packet hash drifted")
    annotations = [
        StrategyBankV2HumanAnnotation.model_validate(row)
        for row in iter_jsonl(annotations_path)
    ]
    expected_ids = [str(row["card_id"]) for row in review_rows]
    observed_ids = [row.card_id for row in annotations]
    if len(observed_ids) != len(set(observed_ids)):
        raise RuntimeError("Strategy Bank V2 human reviews contain duplicate IDs")
    if set(observed_ids) != set(expected_ids):
        raise RuntimeError("Strategy Bank V2 human review coverage is not exact")
    by_id = {
        row.card_id: row.model_dump(mode="json")
        for row in annotations
    }
    ordered = [by_id[card_id] for card_id in expected_ids]
    approved_count = sum(
        bool(row["approve_for_train_only_pilot"]) for row in ordered
    )
    status = (
        "HUMAN_REVIEW_PASS_PENDING_LLM_WEAK_AUDIT"
        if approved_count == len(ordered)
        else "HUMAN_REVIEW_REQUIRES_CARD_REVISION"
    )
    core = {
        "protocol": "pm-v1.5-strategy-bank-v2-human-review-binding-v1",
        "status": status,
        "build_report_sha256": expected_report_sha,
        "human_review_packet_sha256": build_report[
            "human_review_packet_sha256"
        ],
        "human_annotation_file_sha256": sha256_file(annotations_path),
        "human_annotation_content_sha256": sha256_text(canonical_json(ordered)),
        "review_count": len(ordered),
        "approved_count": approved_count,
        "formal_rs_promoted": False,
        "llm_weak_audit_complete": False,
        "internal_test_outcomes_opened": False,
        "external_outcomes_opened": False,
        "api_calls_made": 0,
    }
    return {**core, "binding_sha256": sha256_text(canonical_json(core))}


def _numeric(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _raw_filter_reasons(row: Mapping[str, Any]) -> list[str]:
    response = normalize_space(row.get("example_response") or "")
    reasons: list[str] = []
    if not response:
        reasons.append("empty_response")
    if _GREETING_RE.fullmatch(response):
        reasons.append("greeting_only")
    if _CLOSING_RE.fullmatch(response):
        reasons.append("closing_only")
    if _META_RE.search(response):
        reasons.append("platform_or_survey_meta")
    if _DOMAIN_CLAIM_RE.search(response):
        reasons.append("domain_specific_claim")
    if str(row.get("strategy_label")) not in FORMAL_TECHNIQUE_FAMILIES:
        reasons.append("family_not_in_first_paper_technique_pool")
    return reasons


def build_strategy_bank_v2_candidate(
    *,
    raw_strategy_bank_path: str | Path,
    esconv_path: str | Path,
    split_manifest_path: str | Path,
    support_need_lineage_path: str | Path,
) -> dict[str, Any]:
    """Build five transparent family cards plus complete source lineage."""

    raw_rows = [dict(row) for row in iter_jsonl(raw_strategy_bank_path)]
    if not raw_rows:
        raise RuntimeError("raw Strategy Bank is empty")
    esconv = load_esconv(esconv_path)
    split_rows = [dict(row) for row in iter_jsonl(split_manifest_path)]
    if len(esconv) != len(split_rows):
        raise RuntimeError("ESConv/split rows differ")
    split_by_id = {
        str(row["dialogue_id"]): row
        for row in split_rows
    }
    esconv_by_id = {
        f"esconv_{index:04d}": row
        for index, row in enumerate(esconv)
    }
    packet_source_ids = {
        str(row["dialogue_id"])
        for row in iter_jsonl(support_need_lineage_path)
    }
    if not packet_source_ids:
        raise RuntimeError("support-need packet lineage is empty")

    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusion_counts: Counter[str] = Counter()
    normalized_groups: Counter[str] = Counter(
        normalize_for_hash(row.get("example_response") or "")
        for row in raw_rows
    )
    rows_in_duplicate_groups = sum(
        count for value, count in normalized_groups.items() if value and count > 1
    )
    duplicate_group_count = sum(
        1 for value, count in normalized_groups.items() if value and count > 1
    )
    seen_normalized_response: set[tuple[str, str]] = set()
    lineage_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        source_id = str(row.get("source_dialogue_id") or "")
        split = split_by_id.get(source_id)
        if (
            split is None
            or split.get("split") != "train"
            or bool(split.get("excluded_for_evoemo_overlap"))
        ):
            raise RuntimeError("raw Strategy Bank has an ineligible source")
        if source_id in packet_source_ids:
            exclusion_counts["support_need_packet_source"] += 1
            continue
        source = esconv_by_id[source_id]
        source_dialogue = list(source.get("dialog") or [])
        source_turn_index = int(row["source_turn_index"])
        if (
            source_turn_index < 0
            or source_turn_index >= len(source_dialogue)
            or source_dialogue[source_turn_index].get("speaker") != "supporter"
        ):
            raise RuntimeError("raw Strategy Bank source turn lineage is invalid")
        if not any(
            turn.get("speaker") == "seeker"
            and normalize_space(turn.get("content") or "")
            for turn in source_dialogue[:source_turn_index]
        ):
            exclusion_counts["no_prior_seeker_turn"] += 1
            continue
        reasons = _raw_filter_reasons(row)
        if reasons:
            exclusion_counts.update(reasons)
            continue
        family = str(row["strategy_label"])
        normalized_response = normalize_for_hash(row["example_response"])
        duplicate_key = (family, normalized_response)
        if duplicate_key in seen_normalized_response:
            exclusion_counts["redundant_normalized_response"] += 1
            continue
        seen_normalized_response.add(duplicate_key)
        evidence = {
            "strategy_id": str(row["strategy_id"]),
            "source_dialogue_id": source_id,
            "source_turn_index": int(row["source_turn_index"]),
            "problem_type": normalize_space(source.get("problem_type") or "unknown"),
            "emotion_type": normalize_space(source.get("emotion_type") or "unknown"),
            "experience_type": normalize_space(
                source.get("experience_type") or "unknown"
            ),
        }
        family_rows[family].append(evidence)

    cards: list[dict[str, Any]] = []
    for family in FORMAL_TECHNIQUE_FAMILIES:
        evidence_rows = family_rows.get(family) or []
        source_ids = sorted(
            {str(row["source_dialogue_id"]) for row in evidence_rows}
        )
        if len(source_ids) < 30:
            raise RuntimeError(f"insufficient source support for {family}")
        spec = _FAMILY_SPEC[family]
        source_surveys = [
            esconv_by_id[source_id].get("survey_score") or {}
            for source_id in source_ids
        ]
        seeker_empathy = [
            value
            for survey in source_surveys
            if (
                value := _numeric(
                    (survey.get("seeker") or {}).get("empathy")
                )
            )
            is not None
        ]
        seeker_relevance = [
            value
            for survey in source_surveys
            if (
                value := _numeric(
                    (survey.get("seeker") or {}).get("relevance")
                )
            )
            is not None
        ]
        supporter_relevance = [
            value
            for survey in source_surveys
            if (
                value := _numeric(
                    (survey.get("supporter") or {}).get("relevance")
                )
            )
            is not None
        ]
        emotion_change = []
        for survey in source_surveys:
            seeker = survey.get("seeker") or {}
            initial = _numeric(seeker.get("initial_emotion_intensity"))
            final = _numeric(seeker.get("final_emotion_intensity"))
            if initial is not None and final is not None:
                emotion_change.append(initial - final)
        source_ids_sha = sha256_text(canonical_json(source_ids))
        raw_strategy_ids = sorted(
            str(row["strategy_id"]) for row in evidence_rows
        )
        raw_ids_sha = sha256_text(canonical_json(raw_strategy_ids))
        card_id = "strategy_v2_" + stable_hex(
            STRATEGY_BANK_V2_PROTOCOL,
            family,
            spec,
            source_ids_sha,
            raw_ids_sha,
            n=24,
        )
        problem_counts = Counter(
            str(row["problem_type"]) for row in evidence_rows
        )
        emotion_counts = Counter(
            str(row["emotion_type"]) for row in evidence_rows
        )
        experience_counts = Counter(
            str(row["experience_type"]) for row in evidence_rows
        )
        retrieval_text = " ".join(
            [
                spec["support_move"],
                spec["when_to_use"],
                "Compatible modes:",
                ", ".join(spec["modes"]),
                "Goals:",
                ", ".join(spec["goals"]),
            ]
        )
        prompt_guidance = " ".join(
            [
                spec["support_move"],
                spec["when_not_to_use"],
                "Use only information visible in the current prompt.",
            ]
        )
        card = StrategyBankV2Card(
            card_id=card_id,
            strategy_family=family,
            support_move=spec["support_move"],
            when_to_use=spec["when_to_use"],
            when_not_to_use=spec["when_not_to_use"],
            compatible_support_modes=spec["modes"],
            compatible_dialogue_phases=spec["phases"],
            goal_types=spec["goals"],
            native_problem_types=dict(sorted(problem_counts.items())),
            native_emotion_types=dict(sorted(emotion_counts.items())),
            experience_types=dict(sorted(experience_counts.items())),
            directive_burden=spec["burden"],
            content_scope="technique_only",
            risk_flags=spec["risk_flags"],
            retrieval_text=retrieval_text,
            prompt_guidance=prompt_guidance,
            source_provenance={
                "source_dialogue_count": len(source_ids),
                "source_dialogue_ids_sha256": source_ids_sha,
                "raw_strategy_card_count": len(raw_strategy_ids),
                "raw_strategy_ids_sha256": raw_ids_sha,
                "selected_packet_source_overlap_count": len(
                    set(source_ids) & packet_source_ids
                ),
                "raw_examples_exposed_to_generator": False,
            },
            source_quality_observations={
                "role": "post_treatment_conversation_level_report_only",
                "seeker_empathy_mean": _mean(seeker_empathy),
                "seeker_empathy_n": len(seeker_empathy),
                "seeker_relevance_mean": _mean(seeker_relevance),
                "seeker_relevance_n": len(seeker_relevance),
                "supporter_relevance_mean": _mean(supporter_relevance),
                "supporter_relevance_n": len(supporter_relevance),
                "emotion_intensity_reduction_mean": _mean(emotion_change),
                "emotion_intensity_reduction_n": len(emotion_change),
                "used_for_online_pm": False,
                "used_as_turn_level_utility": False,
            },
        )
        cards.append(card.model_dump(mode="json"))
        for row in evidence_rows:
            lineage_rows.append(
                {
                    "card_id": card_id,
                    **row,
                    "raw_example_response_exposed_to_generator": False,
                }
            )

    card_ids = [str(row["card_id"]) for row in cards]
    if len(card_ids) != len(set(card_ids)):
        raise RuntimeError("Strategy Bank V2 card IDs are not unique")
    if any(
        row["source_dialogue_id"] in packet_source_ids
        for row in lineage_rows
    ):
        raise RuntimeError("Strategy Bank V2 lineage overlaps the need packet")
    human_review_rows = [
        {
            "card_id": row["card_id"],
            "strategy_family": row["strategy_family"],
            "support_move": row["support_move"],
            "when_to_use": row["when_to_use"],
            "when_not_to_use": row["when_not_to_use"],
            "compatible_support_modes": row["compatible_support_modes"],
            "compatible_dialogue_phases": row["compatible_dialogue_phases"],
            "goal_types": row["goal_types"],
            "directive_burden": row["directive_burden"],
            "risk_flags": row["risk_flags"],
            "source_coverage": {
                "source_dialogue_count": row["source_provenance"][
                    "source_dialogue_count"
                ],
                "raw_strategy_card_count": row["source_provenance"][
                    "raw_strategy_card_count"
                ],
            },
        }
        for row in cards
    ]
    human_annotation_template = [
        {
            "card_id": row["card_id"],
            "support_move_clear": None,
            "when_to_use_valid": None,
            "when_not_to_use_valid": None,
            "mode_phase_goal_fit_valid": None,
            "burden_and_risk_flags_valid": None,
            "safe_general_technique": None,
            "approve_for_train_only_pilot": None,
            "confidence": None,
            "required_corrections": "",
            "notes": "",
        }
        for row in cards
    ]
    report_core = {
        "protocol": "pm-v1.5-strategy-bank-v2-build-report-v1",
        "status": "CANDIDATE_BUILT_AWAITING_LLM_WEAK_AND_HUMAN_AUDIT",
        "raw_card_count": len(raw_rows),
        "raw_source_dialogue_count": len(
            {str(row["source_dialogue_id"]) for row in raw_rows}
        ),
        "packet_source_dialogue_count": len(packet_source_ids),
        "packet_source_excluded_raw_card_count": int(
            exclusion_counts["support_need_packet_source"]
        ),
        "no_prior_seeker_turn_excluded_raw_card_count": int(
            exclusion_counts["no_prior_seeker_turn"]
        ),
        "normalized_duplicate_group_count": duplicate_group_count,
        "rows_in_normalized_duplicate_groups": rows_in_duplicate_groups,
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "candidate_card_count": len(cards),
        "candidate_families": [row["strategy_family"] for row in cards],
        "candidate_lineage_row_count": len(lineage_rows),
        "candidate_source_dialogue_count": len(
            {str(row["source_dialogue_id"]) for row in lineage_rows}
        ),
        "candidate_packet_source_overlap_count": 0,
        "candidate_raw_examples_exposed_to_generator": False,
        "candidate_quality_status": STRATEGY_BANK_V2_QUALITY_STATUS,
        "candidate_eligible_for_formal_rs": False,
        "source_lineage": {
            "raw_strategy_bank_sha256": sha256_file(raw_strategy_bank_path),
            "esconv_sha256": sha256_file(esconv_path),
            "split_manifest_sha256": sha256_file(split_manifest_path),
            "support_need_lineage_sha256": sha256_file(
                support_need_lineage_path
            ),
        },
        "cards_sha256": sha256_text(canonical_json(cards)),
        "lineage_sha256": sha256_text(canonical_json(lineage_rows)),
        "human_review_packet_sha256": sha256_text(
            canonical_json(human_review_rows)
        ),
        "human_annotation_template_sha256": sha256_text(
            canonical_json(human_annotation_template)
        ),
        "internal_test_outcomes_opened": False,
        "external_outcomes_opened": False,
        "api_calls_made": 0,
    }
    report = {
        **report_core,
        "report_sha256": sha256_text(canonical_json(report_core)),
    }
    return {
        "cards": cards,
        "lineage_rows": lineage_rows,
        "human_review_rows": human_review_rows,
        "human_annotation_template": human_annotation_template,
        "report": report,
    }
