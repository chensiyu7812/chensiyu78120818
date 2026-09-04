"""Outcome-blind selection of fresh ESConv train dialogues for need annotation."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import Field, field_validator, model_validator

from .pm_v2_contracts import StrictModel
from .io import canonical_json, iter_jsonl, sha256_file, sha256_text, stable_hex
from .strategy_bank import load_esconv
from .text import normalize_space
from .v1_5_support_need import explicit_support_boundaries


SUPPORT_NEED_PACKET_PROTOCOL = "pm-v1.5-support-need-human-packet-v1"
POSITION_STRATA = ("early", "middle", "late")
HUMAN_SUPPORT_MODES = (
    "listen",
    "explore",
    "comfort_reassure",
    "light_guidance",
    "structured_planning",
)
HUMAN_GOALS = ("be_heard", "make_sense", "stabilize", "decide", "act")
HUMAN_PHASES = ("exploration", "comforting", "action")
HUMAN_URGENCIES = ("routine", "elevated", "acute")
HUMAN_BOUNDARY_FIELDS = (
    "advice_rejected",
    "advice_requested",
    "one_small_step_requested",
    "listen_first_requested",
    "question_or_task_burden_limit",
)
FORBIDDEN_BLIND_KEYS = frozenset(
    {
        "dialogue_id",
        "dialogue_index",
        "turn_index",
        "problem_type",
        "emotion_type",
        "experience_type",
        "strategy",
        "gold_strategy",
        "gold_response",
        "survey_score",
        "split",
        "expected_mode",
    }
)


class SupportNeedHumanAnnotation(StrictModel):
    """One outcome-blind human anchor; never an automatic action gold."""

    blind_item_id: str = Field(min_length=1)
    support_mode: Literal[
        "listen",
        "explore",
        "comfort_reassure",
        "light_guidance",
        "structured_planning",
    ] | None
    goals: list[
        Literal["be_heard", "make_sense", "stabilize", "decide", "act"]
    ]
    dialogue_phase: Literal["exploration", "comforting", "action"] | None
    nonclinical_urgency: Literal["routine", "elevated", "acute"] | None
    advice_rejected: bool | None
    advice_requested: bool | None
    one_small_step_requested: bool | None
    listen_first_requested: bool | None
    question_or_task_burden_limit: bool | None
    abstain: bool
    confidence: int = Field(ge=1, le=5)
    notes: str = ""

    @field_validator("goals")
    @classmethod
    def unique_goals(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("human support-need goals must be unique")
        return value

    @model_validator(mode="after")
    def coherent_human_anchor(self):
        if self.abstain:
            if self.support_mode is not None:
                raise ValueError("abstaining human anchor cannot carry support_mode")
        elif (
            self.support_mode is None
            or self.dialogue_phase is None
            or self.nonclinical_urgency is None
            or not self.goals
        ):
            raise ValueError(
                "non-abstaining human anchor requires mode, goal, phase, urgency"
            )
        if self.confidence <= 2 and not str(self.notes).strip():
            raise ValueError("low-confidence human anchor requires notes")
        return self


class SupportNeedBoundaryEvidence(StrictModel):
    boundary_type: Literal[
        "advice_rejected",
        "advice_requested",
        "one_small_step_requested",
        "listen_first_requested",
        "question_or_task_burden_limit",
    ]
    exact_user_quote: str = Field(min_length=1)


class SupportNeedHumanAnnotationV2(StrictModel):
    """Revised anchor schema with evidence-backed boundaries and burden."""

    blind_item_id: str = Field(min_length=1)
    support_mode: Literal[
        "listen",
        "explore",
        "comfort_reassure",
        "light_guidance",
        "structured_planning",
    ] | None
    goals: list[
        Literal["be_heard", "make_sense", "stabilize", "decide", "act"]
    ]
    dialogue_phase: Literal["exploration", "comforting", "action"] | None
    nonclinical_urgency: Literal["routine", "elevated", "acute"] | None
    recommended_response_burden: Literal[
        "minimal_presence",
        "one_focus",
        "multi_step_ok",
    ] | None
    active_explicit_boundary_evidence: list[SupportNeedBoundaryEvidence]
    abstain: bool
    confidence: int = Field(ge=1, le=5)
    notes: str = ""

    @field_validator("goals")
    @classmethod
    def unique_v2_goals(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("human support-need V2 goals must be unique")
        return value

    @model_validator(mode="after")
    def coherent_v2_anchor(self):
        evidence_keys = [
            (row.boundary_type, normalize_space(row.exact_user_quote))
            for row in self.active_explicit_boundary_evidence
        ]
        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError("explicit boundary evidence must be unique")
        if self.abstain:
            if (
                self.support_mode is not None
                or self.goals
                or self.dialogue_phase is not None
                or self.nonclinical_urgency is not None
                or self.recommended_response_burden is not None
                or self.active_explicit_boundary_evidence
            ):
                raise ValueError("abstaining V2 anchor must not carry labels")
        elif (
            self.support_mode is None
            or not self.goals
            or self.dialogue_phase is None
            or self.nonclinical_urgency is None
            or self.recommended_response_burden is None
        ):
            raise ValueError(
                "non-abstaining V2 anchor requires mode/goals/phase/urgency/burden"
            )
        if self.confidence <= 2 and not normalize_space(self.notes):
            raise ValueError("low-confidence V2 anchor requires notes")
        return self


def validate_support_need_human_annotations(
    *,
    packet_dir: str | Path,
    annotations_path: str | Path,
) -> dict[str, Any]:
    """Validate exact anchor coverage and bind annotations to the packet."""

    packet_dir = Path(packet_dir)
    annotations_path = Path(annotations_path)
    contract_path = packet_dir / "qualification_contract.json"
    anchor_path = packet_dir / "human_blind_packet.jsonl"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_core = dict(contract)
    expected_contract_sha = str(contract_core.pop("contract_sha256", ""))
    if expected_contract_sha != sha256_text(canonical_json(contract_core)):
        raise RuntimeError("support-need packet contract hash is invalid")
    anchors = [dict(row) for row in iter_jsonl(anchor_path)]
    if (
        sha256_text(canonical_json(anchors))
        != contract.get("human_anchor_packet_sha256")
    ):
        raise RuntimeError("support-need human anchor packet hash drifted")
    parsed = [
        SupportNeedHumanAnnotation.model_validate(row)
        for row in iter_jsonl(annotations_path)
    ]
    observed_ids = [row.blind_item_id for row in parsed]
    expected_ids = [str(row["blind_item_id"]) for row in anchors]
    if len(observed_ids) != len(set(observed_ids)):
        raise RuntimeError("support-need human annotations contain duplicate IDs")
    if set(observed_ids) != set(expected_ids):
        raise RuntimeError("support-need human annotation ID coverage is not exact")
    ordered = {
        row.blind_item_id: row.model_dump(mode="json")
        for row in parsed
    }
    canonical_rows = [ordered[blind_id] for blind_id in expected_ids]
    status = "COMPLETE_HUMAN_ANCHORS_NOT_AUTOMATIC_GOLD"
    report_core = {
        "protocol": "pm-v1.5-support-need-human-anchor-binding-v1",
        "status": status,
        "packet_contract_sha256": expected_contract_sha,
        "human_anchor_packet_sha256": contract["human_anchor_packet_sha256"],
        "human_annotation_file_sha256": sha256_file(annotations_path),
        "human_annotation_content_sha256": sha256_text(
            canonical_json(canonical_rows)
        ),
        "annotation_count": len(canonical_rows),
        "abstain_count": sum(bool(row["abstain"]) for row in canonical_rows),
        "support_mode_counts": dict(
            sorted(
                Counter(
                    str(row["support_mode"])
                    for row in canonical_rows
                    if row["support_mode"] is not None
                ).items()
            )
        ),
        "mean_confidence": sum(
            float(row["confidence"]) for row in canonical_rows
        )
        / len(canonical_rows),
        "annotation_role": "human_reliability_and_boundary_anchor",
        "automatic_gold_label": False,
        "pm_action_gold": False,
        "internal_test_outcomes_opened": False,
        "external_outcomes_opened": False,
        "api_calls_made": 0,
    }
    return {
        **report_core,
        "binding_sha256": sha256_text(canonical_json(report_core)),
    }


def normalize_support_need_human_anchors(
    *,
    packet_dir: str | Path,
    annotations_path: str | Path,
) -> dict[str, Any]:
    """Bind completed anchors while separating two different semantics.

    The first annotation UI used ``question_or_task_burden_limit`` both for an
    explicit user boundary and for a human recommendation that the next
    response should remain low burden.  Those are not interchangeable:

    * explicit boundaries are deterministic observations from the current user
      text and are inputs to the PM;
    * recommended interaction burden is a contextual, fallible human partial
      label and is never claimed to be an explicit user statement.

    The raw annotation remains byte-bound and is embedded losslessly in every
    normalized row.  This transformation therefore needs no re-annotation and
    cannot silently rewrite a human judgement into a stronger fact.
    """

    packet_dir = Path(packet_dir)
    annotations_path = Path(annotations_path)
    validation = validate_support_need_human_annotations(
        packet_dir=packet_dir,
        annotations_path=annotations_path,
    )
    blind_rows = [
        dict(row)
        for row in iter_jsonl(packet_dir / "human_blind_packet.jsonl")
    ]
    annotation_by_id = {
        parsed.blind_item_id: parsed
        for parsed in (
            SupportNeedHumanAnnotation.model_validate(row)
            for row in iter_jsonl(annotations_path)
        )
    }
    normalized_rows: list[dict[str, Any]] = []
    for blind_row in blind_rows:
        blind_item_id = str(blind_row["blind_item_id"])
        visible_state = dict(blind_row["visible_state"])
        annotation = annotation_by_id[blind_item_id]
        raw_annotation = annotation.model_dump(mode="json")
        contextual_judgments = {
            field: raw_annotation[field] for field in HUMAN_BOUNDARY_FIELDS
        }
        deterministic_boundaries = explicit_support_boundaries(
            str(visible_state["current_user_text"])
        ).model_dump(mode="json")
        row_core = {
            "protocol": "pm-v1.5-support-need-human-anchor-normalized-v2",
            "blind_item_id": blind_item_id,
            "visible_state_sha256": sha256_text(canonical_json(visible_state)),
            "raw_annotation": raw_annotation,
            "contextual_human_boundary_judgments": contextual_judgments,
            "deterministic_current_turn_explicit_boundaries": (
                deterministic_boundaries
            ),
            "human_recommended_low_interaction_burden": raw_annotation[
                "question_or_task_burden_limit"
            ],
            "semantic_roles": {
                "deterministic_current_turn_explicit_boundaries": (
                    "observable_pm_input_not_label"
                ),
                "contextual_human_boundary_judgments": (
                    "legacy_contextual_partial_judgment_not_explicit_fact"
                ),
                "human_recommended_low_interaction_burden": (
                    "human_partial_label_not_user_statement"
                ),
                "support_mode_goals_phase_urgency": (
                    "human_partial_labels_not_pm_action_gold"
                ),
            },
            "automatic_gold_label": False,
            "pm_action_gold": False,
            "internal_test_outcomes_opened": False,
            "external_outcomes_opened": False,
        }
        normalized_rows.append(
            {
                **row_core,
                "normalized_row_sha256": sha256_text(canonical_json(row_core)),
            }
        )

    recommendation_counts = Counter(
        (
            "unknown"
            if row["human_recommended_low_interaction_burden"] is None
            else str(
                bool(row["human_recommended_low_interaction_burden"])
            ).lower()
        )
        for row in normalized_rows
    )
    explicit_positive_counts = {
        field: sum(
            bool(row["deterministic_current_turn_explicit_boundaries"][field])
            for row in normalized_rows
        )
        for field in HUMAN_BOUNDARY_FIELDS
    }
    contextual_counts = {
        field: dict(
            sorted(
                Counter(
                    (
                        "unknown"
                        if row["contextual_human_boundary_judgments"][field]
                        is None
                        else str(
                            bool(
                                row[
                                    "contextual_human_boundary_judgments"
                                ][field]
                            )
                        ).lower()
                    )
                    for row in normalized_rows
                ).items()
            )
        )
        for field in HUMAN_BOUNDARY_FIELDS
    }
    report_core = {
        "protocol": "pm-v1.5-support-need-human-anchor-normalization-v2",
        "status": "COMPLETE_SEMANTICALLY_SEPARATED_HUMAN_ANCHORS",
        "input_binding_sha256": validation["binding_sha256"],
        "packet_contract_sha256": validation["packet_contract_sha256"],
        "human_annotation_file_sha256": validation[
            "human_annotation_file_sha256"
        ],
        "human_annotation_content_sha256": validation[
            "human_annotation_content_sha256"
        ],
        "normalized_rows_sha256": sha256_text(canonical_json(normalized_rows)),
        "annotation_count": len(normalized_rows),
        "abstain_count": validation["abstain_count"],
        "explicit_boundary_positive_counts": explicit_positive_counts,
        "contextual_human_boundary_judgment_counts": contextual_counts,
        "human_recommended_low_interaction_burden_counts": dict(
            sorted(recommendation_counts.items())
        ),
        "legacy_field_interpretation": {
            "question_or_task_burden_limit": (
                "human_recommended_low_interaction_burden"
            ),
            "reason": (
                "completed notes use the field mainly as a contextual response-"
                "burden recommendation, not solely as an explicit user quote"
            ),
            "raw_annotation_preserved": True,
            "reannotation_required": False,
        },
        "automatic_gold_label": False,
        "pm_action_gold": False,
        "api_calls_made": 0,
        "internal_test_outcomes_opened": False,
        "external_outcomes_opened": False,
    }
    report = {
        **report_core,
        "normalization_binding_sha256": sha256_text(
            canonical_json(report_core)
        ),
    }
    return {"normalized_rows": normalized_rows, "report": report}


def build_support_need_expansion_packet(
    *,
    source_packet_dir: str | Path,
    packet_size: int = 24,
    fit_anchor_count: int = 16,
    seed: int = 7727,
) -> dict[str, Any]:
    """Select fresh anchors without consulting completed human labels."""

    source_packet_dir = Path(source_packet_dir)
    all_rows = [
        dict(row)
        for row in iter_jsonl(source_packet_dir / "support_need_packet.jsonl")
    ]
    initial_rows = [
        dict(row)
        for row in iter_jsonl(source_packet_dir / "human_blind_packet.jsonl")
    ]
    lineage_rows = [
        dict(row)
        for row in iter_jsonl(source_packet_dir / "private_lineage.jsonl")
    ]
    lineage_by_id = {
        str(row["blind_item_id"]): row for row in lineage_rows
    }
    if (
        len(lineage_by_id) != len(lineage_rows)
        or set(lineage_by_id)
        != {str(row["blind_item_id"]) for row in all_rows}
    ):
        raise RuntimeError("support-need source packet lineage is malformed")
    if (
        packet_size < 12
        or fit_anchor_count < 1
        or fit_anchor_count >= packet_size
    ):
        raise ValueError("support-need expansion sizes are invalid")
    initial_ids = {str(row["blind_item_id"]) for row in initial_rows}
    eligible = [
        row
        for row in all_rows
        if str(row["blind_item_id"]) not in initial_ids
    ]
    if len(eligible) < packet_size:
        raise RuntimeError("not enough fresh support-need expansion anchors")
    eligible.sort(
        key=lambda row: stable_hex(
            "pm-v1.5-support-need-human-expansion-packet-v2",
            seed,
            "selection",
            row["blind_item_id"],
            n=32,
        )
    )
    selected = eligible[:packet_size]
    role_order = sorted(
        (str(row["blind_item_id"]) for row in selected),
        key=lambda blind_item_id: stable_hex(
            "pm-v1.5-support-need-human-expansion-packet-v2",
            seed,
            "fit-confirmation-role",
            blind_item_id,
            n=32,
        ),
    )
    fit_ids = set(role_order[:fit_anchor_count])
    private_rows = [
        {
            **lineage_by_id[str(row["blind_item_id"])],
            "anchor_role": (
                "expansion_fit"
                if str(row["blind_item_id"]) in fit_ids
                else "untouched_confirmation"
            ),
            "selection_used_completed_human_labels": False,
        }
        for row in selected
    ]
    template_rows = [
        {
            "blind_item_id": row["blind_item_id"],
            "support_mode": None,
            "goals": [],
            "dialogue_phase": None,
            "nonclinical_urgency": None,
            "recommended_response_burden": None,
            "active_explicit_boundary_evidence": [],
            "abstain": None,
            "confidence": None,
            "notes": "",
        }
        for row in selected
    ]
    source_contract_path = source_packet_dir / "qualification_contract.json"
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    report_core = {
        "protocol": "pm-v1.5-support-need-human-expansion-packet-v2",
        "status": "PREPARED_FRESH_TRAIN_ONLY_AWAITING_HUMAN_ANNOTATION",
        "seed": seed,
        "source_packet_contract_sha256": source_contract["contract_sha256"],
        "source_packet_file_sha256": sha256_file(
            source_packet_dir / "support_need_packet.jsonl"
        ),
        "source_initial_human_packet_sha256": sha256_file(
            source_packet_dir / "human_blind_packet.jsonl"
        ),
        "selected_packet_sha256": sha256_text(canonical_json(selected)),
        "template_sha256": sha256_text(canonical_json(template_rows)),
        "private_lineage_sha256": sha256_text(canonical_json(private_rows)),
        "packet_size": packet_size,
        "fit_anchor_count": fit_anchor_count,
        "untouched_confirmation_count": packet_size - fit_anchor_count,
        "selection_used_completed_human_labels": False,
        "selection_used_target_supporter_responses": False,
        "selection_used_target_strategy_annotations": False,
        "selection_used_surveys": False,
        "blind_packet_exposes_anchor_role": False,
        "annotation_semantics": {
            "explicit_boundary": "requires_exact_quote_from_visible_user_text",
            "recommended_response_burden": (
                "contextual_human_partial_label_not_explicit_user_fact"
            ),
            "support_mode": "human_partial_label_not_pm_action_gold",
        },
        "automatic_gold_label": False,
        "pm_action_gold": False,
        "api_calls_made": 0,
        "internal_test_outcomes_opened": False,
        "external_outcomes_opened": False,
    }
    return {
        "packet_rows": selected,
        "template_rows": template_rows,
        "private_rows": private_rows,
        "contract": {
            **report_core,
            "contract_sha256": sha256_text(canonical_json(report_core)),
        },
    }


def build_support_need_expansion_fit_packet(
    *,
    expansion_packet_dir: str | Path,
    bakeoff_binding_path: str | Path,
) -> dict[str, Any]:
    """Expose only preregistered fit anchors while sealing confirmation rows."""

    expansion_packet_dir = Path(expansion_packet_dir)
    bakeoff_binding_path = Path(bakeoff_binding_path)
    source_contract_path = expansion_packet_dir / "qualification_contract.json"
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    source_contract_core = dict(source_contract)
    source_contract_sha256 = str(
        source_contract_core.pop("contract_sha256", "")
    )
    if source_contract_sha256 != sha256_text(
        canonical_json(source_contract_core)
    ):
        raise RuntimeError("support-need expansion source contract is invalid")
    source_packet_rows = [
        dict(row)
        for row in iter_jsonl(
            expansion_packet_dir / "human_blind_packet.jsonl"
        )
    ]
    source_template_rows = [
        dict(row)
        for row in iter_jsonl(
            expansion_packet_dir / "human_annotation_template.jsonl"
        )
    ]
    source_private_rows = [
        dict(row)
        for row in iter_jsonl(expansion_packet_dir / "private_lineage.jsonl")
    ]
    if (
        sha256_text(canonical_json(source_packet_rows))
        != source_contract.get("selected_packet_sha256")
        or sha256_text(canonical_json(source_template_rows))
        != source_contract.get("template_sha256")
        or sha256_text(canonical_json(source_private_rows))
        != source_contract.get("private_lineage_sha256")
    ):
        raise RuntimeError("support-need expansion source files drifted")
    packet_by_id = {
        str(row["blind_item_id"]): row for row in source_packet_rows
    }
    template_by_id = {
        str(row["blind_item_id"]): row for row in source_template_rows
    }
    private_by_id = {
        str(row["blind_item_id"]): row for row in source_private_rows
    }
    if (
        len(packet_by_id) != len(source_packet_rows)
        or len(template_by_id) != len(source_template_rows)
        or len(private_by_id) != len(source_private_rows)
        or set(packet_by_id) != set(template_by_id)
        or set(packet_by_id) != set(private_by_id)
    ):
        raise RuntimeError("support-need expansion source IDs are malformed")
    fit_ids = {
        blind_item_id
        for blind_item_id, row in private_by_id.items()
        if row.get("anchor_role") == "expansion_fit"
    }
    confirmation_ids = {
        blind_item_id
        for blind_item_id, row in private_by_id.items()
        if row.get("anchor_role") == "untouched_confirmation"
    }
    if (
        fit_ids & confirmation_ids
        or fit_ids | confirmation_ids != set(packet_by_id)
        or len(fit_ids) != int(source_contract.get("fit_anchor_count", -1))
        or len(confirmation_ids)
        != int(source_contract.get("untouched_confirmation_count", -1))
    ):
        raise RuntimeError("support-need expansion roles are malformed")
    bakeoff_binding = json.loads(
        bakeoff_binding_path.read_text(encoding="utf-8")
    )
    binding_project_root = bakeoff_binding_path.resolve().parents[2]

    def resolve_bound_report(path_value: object) -> Path:
        path = Path(str(path_value))
        if path.is_absolute():
            return path
        candidates = (
            binding_project_root / path,
            Path.cwd() / path,
            bakeoff_binding_path.parent / path,
        )
        return next(
            (candidate for candidate in candidates if candidate.is_file()),
            candidates[0],
        )

    candidate_binding = bakeoff_binding.get("candidate_report", {})
    reproduction_binding = bakeoff_binding.get("reproduction_report", {})
    candidate_report_path = resolve_bound_report(
        candidate_binding.get("path", "")
    )
    reproduction_report_path = resolve_bound_report(
        reproduction_binding.get("path", "")
    )
    if (
        not candidate_report_path.is_file()
        or not reproduction_report_path.is_file()
        or sha256_file(candidate_report_path)
        != candidate_binding.get("file_sha256")
        or sha256_file(reproduction_report_path)
        != reproduction_binding.get("file_sha256")
        or candidate_report_path.read_bytes()
        != reproduction_report_path.read_bytes()
        or reproduction_binding.get("byte_identical_to_candidate") is not True
        or int(reproduction_binding.get("field_diff_count", -1)) != 0
    ):
        raise RuntimeError(
            "support-need factorized bakeoff reports are missing or drifted"
        )
    candidate_report = json.loads(
        candidate_report_path.read_text(encoding="utf-8")
    )
    candidate_report_core = dict(candidate_report)
    candidate_report_sha256 = str(
        candidate_report_core.pop("report_sha256", "")
    )
    if (
        candidate_report_sha256
        != sha256_text(canonical_json(candidate_report_core))
        or candidate_report_sha256
        != candidate_binding.get("report_sha256")
    ):
        raise RuntimeError(
            "support-need factorized bakeoff report hash is invalid"
        )
    if (
        bakeoff_binding.get("status")
        != (
            "COMPLETE_REPRODUCIBLE_QWEN_NOT_PROMOTED_"
            "AXIS_SPECIFIC_SIGNAL_MORE_HUMAN_ANCHORS_REQUIRED"
        )
        or bakeoff_binding.get("next_action", {}).get("decision")
        != "COLLECT_ONLY_PREPARED_EXPANSION_FIT_ANCHORS"
        or bakeoff_binding.get("formal_fit_authorized") is not False
        or bakeoff_binding.get("representation_promotion_authorized")
        is not False
        or int(
            bakeoff_binding.get("next_action", {}).get(
                "fit_anchor_count", -1
            )
        )
        != len(fit_ids)
        or int(
            bakeoff_binding.get("next_action", {}).get(
                "untouched_confirmation_count", -1
            )
        )
        != len(confirmation_ids)
        or candidate_report.get("formal_fit_authorized") is not False
        or candidate_report.get("representation_promotion_authorized")
        is not False
        or candidate_report.get("expansion_human_annotations_opened")
        is not False
        or candidate_report.get("internal_test_outcomes_opened") is not False
        or candidate_report.get("external_outcomes_opened") is not False
    ):
        raise RuntimeError(
            "support-need factorized bakeoff does not authorize fit anchors"
        )
    selected_packet_rows = [
        row
        for row in source_packet_rows
        if str(row["blind_item_id"]) in fit_ids
    ]
    selected_template_rows = [
        row
        for row in source_template_rows
        if str(row["blind_item_id"]) in fit_ids
    ]
    selected_private_rows = [
        row
        for row in source_private_rows
        if str(row["blind_item_id"]) in fit_ids
    ]
    if any(
        "anchor_role" in row
        or set(row) & FORBIDDEN_BLIND_KEYS
        for row in selected_packet_rows
    ):
        raise RuntimeError("fit-only blind packet exposes private lineage")
    report_core = {
        "protocol": "pm-v1.5-support-need-expansion-fit-packet-v1",
        "status": "PREPARED_FIT_ANCHORS_CONFIRMATION_ROWS_NOT_EXPOSED",
        "source_expansion_contract_sha256": source_contract_sha256,
        "source_expansion_contract_file_sha256": sha256_file(
            source_contract_path
        ),
        "source_bakeoff_binding_file_sha256": sha256_file(
            bakeoff_binding_path
        ),
        "selected_packet_sha256": sha256_text(
            canonical_json(selected_packet_rows)
        ),
        "template_sha256": sha256_text(
            canonical_json(selected_template_rows)
        ),
        "private_lineage_sha256": sha256_text(
            canonical_json(selected_private_rows)
        ),
        "confirmation_id_set_sha256": sha256_text(
            canonical_json(sorted(confirmation_ids))
        ),
        "packet_size": len(selected_packet_rows),
        "fit_anchor_count": len(selected_packet_rows),
        "untouched_confirmation_count": len(confirmation_ids),
        "confirmation_rows_exposed": False,
        "blind_packet_exposes_anchor_role": False,
        "selection_used_completed_human_labels": False,
        "annotation_semantics": source_contract["annotation_semantics"],
        "automatic_gold_label": False,
        "pm_action_gold": False,
        "api_calls_made": 0,
        "internal_test_outcomes_opened": False,
        "external_outcomes_opened": False,
    }
    return {
        "packet_rows": selected_packet_rows,
        "template_rows": selected_template_rows,
        "private_rows": selected_private_rows,
        "contract": {
            **report_core,
            "contract_sha256": sha256_text(canonical_json(report_core)),
        },
    }


def validate_support_need_expansion_annotations(
    *,
    packet_dir: str | Path,
    annotations_path: str | Path,
) -> dict[str, Any]:
    """Validate V2 annotations, including exact user-only boundary quotes."""

    packet_dir = Path(packet_dir)
    annotations_path = Path(annotations_path)
    contract_path = packet_dir / "qualification_contract.json"
    packet_path = packet_dir / "human_blind_packet.jsonl"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_core = dict(contract)
    contract_sha = str(contract_core.pop("contract_sha256", ""))
    if contract_sha != sha256_text(canonical_json(contract_core)):
        raise RuntimeError("support-need expansion contract hash is invalid")
    packet_rows = [dict(row) for row in iter_jsonl(packet_path)]
    if (
        sha256_text(canonical_json(packet_rows))
        != contract["selected_packet_sha256"]
    ):
        raise RuntimeError("support-need expansion packet drifted")
    parsed = [
        SupportNeedHumanAnnotationV2.model_validate(row)
        for row in iter_jsonl(annotations_path)
    ]
    expected_ids = [str(row["blind_item_id"]) for row in packet_rows]
    observed_ids = [row.blind_item_id for row in parsed]
    if len(observed_ids) != len(set(observed_ids)):
        raise RuntimeError("support-need expansion annotations duplicate IDs")
    if set(observed_ids) != set(expected_ids):
        raise RuntimeError(
            "support-need expansion annotation coverage is not exact"
        )
    packet_by_id = {
        str(row["blind_item_id"]): row for row in packet_rows
    }
    annotation_by_id = {row.blind_item_id: row for row in parsed}
    canonical_rows = []
    for blind_item_id in expected_ids:
        annotation = annotation_by_id[blind_item_id]
        visible = dict(packet_by_id[blind_item_id]["visible_state"])
        visible_user_texts = [
            normalize_space(visible.get("current_user_text") or "")
        ] + [
            normalize_space(turn.get("content") or "")
            for turn in visible.get("recent_dialogue") or []
            if turn.get("role") == "user"
        ]
        for evidence in annotation.active_explicit_boundary_evidence:
            quote = normalize_space(evidence.exact_user_quote)
            if not any(quote in text for text in visible_user_texts):
                raise RuntimeError(
                    "support-need explicit boundary quote is not an exact "
                    f"visible user substring: {blind_item_id}"
                )
        canonical_rows.append(annotation.model_dump(mode="json"))
    report_core = {
        "protocol": "pm-v1.5-support-need-human-expansion-binding-v2",
        "status": "COMPLETE_HUMAN_ANCHORS_NOT_AUTOMATIC_GOLD",
        "packet_contract_sha256": contract_sha,
        "annotation_file_sha256": sha256_file(annotations_path),
        "annotation_content_sha256": sha256_text(
            canonical_json(canonical_rows)
        ),
        "annotation_count": len(canonical_rows),
        "abstain_count": sum(bool(row["abstain"]) for row in canonical_rows),
        "automatic_gold_label": False,
        "pm_action_gold": False,
        "api_calls_made": 0,
        "internal_test_outcomes_opened": False,
        "external_outcomes_opened": False,
    }
    return {
        **report_core,
        "binding_sha256": sha256_text(canonical_json(report_core)),
    }


def _visible_candidates(dialogue: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for turn_index, turn in enumerate(dialogue):
        if turn.get("speaker") != "supporter":
            continue
        previous = list(dialogue[:turn_index])
        current_index = next(
            (
                index
                for index in range(len(previous) - 1, -1, -1)
                if previous[index].get("speaker") == "seeker"
            ),
            None,
        )
        if current_index is None:
            continue
        current = normalize_space(previous[current_index].get("content") or "")
        if not current:
            continue
        history = [
            {
                "role": (
                    "user" if prior.get("speaker") == "seeker" else "assistant"
                ),
                "content": normalize_space(prior.get("content") or ""),
            }
            for prior in previous[:current_index][-8:]
            if prior.get("speaker") in {"seeker", "supporter"}
            and normalize_space(prior.get("content") or "")
        ]
        candidates.append(
            {
                "turn_index": int(turn_index),
                "current_user_text": current,
                "recent_dialogue": history,
            }
        )
    return candidates


def _pick_position(
    candidates: Sequence[Mapping[str, Any]], position: str
) -> dict[str, Any]:
    if not candidates or position not in POSITION_STRATA:
        raise ValueError("visible candidate position is invalid")
    fractions = {"early": 0.2, "middle": 0.5, "late": 0.8}
    index = int(round(fractions[position] * (len(candidates) - 1)))
    return dict(candidates[index])


def build_support_need_annotation_packet(
    *,
    esconv_path: str | Path,
    split_manifest_path: str | Path,
    existing_seed_sources_path: str | Path,
    strategy_bank_path: str | Path,
    packet_size: int = 75,
    human_anchor_size: int = 24,
    seed: int = 4311,
) -> dict[str, Any]:
    """Select one visible response state per fresh ESConv train dialogue.

    Problem/emotion/experience metadata is used only for outcome-blind sampling
    coverage and is omitted from the blind packet.  Target supporter responses,
    target strategy annotations, surveys, and all validation/test rows are omitted.
    """

    if packet_size < 15 or human_anchor_size < 1 or human_anchor_size > packet_size:
        raise ValueError("support-need packet sizes are invalid")
    esconv = load_esconv(esconv_path)
    split_rows = [dict(row) for row in iter_jsonl(split_manifest_path)]
    if len(esconv) != len(split_rows):
        raise RuntimeError("ESConv and split manifest lengths differ")
    excluded_seed_ids = {
        str(row["dialogue_id"]) for row in iter_jsonl(existing_seed_sources_path)
    }
    strategy_bank_rows = [dict(row) for row in iter_jsonl(strategy_bank_path)]
    if not strategy_bank_rows or any(
        not str(row.get("source_dialogue_id") or "").strip()
        for row in strategy_bank_rows
    ):
        raise RuntimeError("Strategy Bank source lineage is missing")
    strategy_bank_source_ids = {
        str(row["source_dialogue_id"]) for row in strategy_bank_rows
    }
    candidates: list[dict[str, Any]] = []
    for index, (dialogue_row, split_row) in enumerate(
        zip(esconv, split_rows, strict=True)
    ):
        dialogue_id = f"esconv_{index:04d}"
        if (
            int(split_row.get("index", -1)) != index
            or str(split_row.get("dialogue_id")) != dialogue_id
        ):
            raise RuntimeError("ESConv split manifest is reordered")
        if (
            split_row.get("split") != "train"
            or bool(split_row.get("excluded_for_evoemo_overlap"))
            or dialogue_id in excluded_seed_ids
        ):
            continue
        visible = _visible_candidates(dialogue_row.get("dialog") or [])
        if not visible:
            continue
        candidates.append(
            {
                "dialogue_id": dialogue_id,
                "dialogue_index": index,
                "problem_type": normalize_space(
                    dialogue_row.get("problem_type") or "unknown"
                ),
                "emotion_type": normalize_space(
                    dialogue_row.get("emotion_type") or "unknown"
                ),
                "experience_type": normalize_space(
                    dialogue_row.get("experience_type") or "unknown"
                ),
                "situation": normalize_space(dialogue_row.get("situation") or ""),
                "visible_candidates": visible,
            }
        )
    if len(candidates) < packet_size:
        raise RuntimeError("not enough fresh ESConv train dialogues for need packet")

    # Round-robin over problem types prevents the most common topic from filling
    # the packet.  Within each stratum, selection is a seeded content-blind hash.
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_problem[row["problem_type"]].append(row)
    for problem, rows in by_problem.items():
        rows.sort(
            key=lambda row: stable_hex(
                SUPPORT_NEED_PACKET_PROTOCOL,
                seed,
                problem,
                row["dialogue_id"],
                n=32,
            )
        )
    problem_order = sorted(
        by_problem,
        key=lambda problem: stable_hex(
            SUPPORT_NEED_PACKET_PROTOCOL, seed, "problem", problem, n=32
        ),
    )
    selected: list[dict[str, Any]] = []
    while len(selected) < packet_size:
        progress = False
        for problem in problem_order:
            rows = by_problem[problem]
            if rows and len(selected) < packet_size:
                selected.append(rows.pop(0))
                progress = True
        if not progress:
            raise RuntimeError("fresh ESConv dialogue selection exhausted early")

    blind_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    for ordinal, source in enumerate(selected):
        position = POSITION_STRATA[ordinal % len(POSITION_STRATA)]
        state = _pick_position(source["visible_candidates"], position)
        blind_item_id = "need_" + stable_hex(
            SUPPORT_NEED_PACKET_PROTOCOL,
            seed,
            source["dialogue_id"],
            state["turn_index"],
            n=20,
        )
        visible_state = {
            "current_user_text": state["current_user_text"],
            "recent_dialogue": state["recent_dialogue"],
            "session_summary": source["situation"],
        }
        blind_rows.append(
            {
                "blind_item_id": blind_item_id,
                "visible_state": visible_state,
            }
        )
        lineage_rows.append(
            {
                "blind_item_id": blind_item_id,
                "dialogue_id": source["dialogue_id"],
                "dialogue_index": source["dialogue_index"],
                "turn_index": state["turn_index"],
                "position_stratum": position,
                "problem_type": source["problem_type"],
                "emotion_type": source["emotion_type"],
                "experience_type": source["experience_type"],
                "must_be_excluded_from_strategy_bank_v2": True,
                "target_supporter_response_read": False,
                "target_strategy_annotation_read": False,
                "survey_outcome_read": False,
            }
        )

    anchor_buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, lineage in enumerate(lineage_rows):
        anchor_buckets[
            (lineage["emotion_type"], lineage["position_stratum"])
        ].append(index)
    for key, indices in anchor_buckets.items():
        indices.sort(
            key=lambda index: stable_hex(
                SUPPORT_NEED_PACKET_PROTOCOL,
                seed,
                "human-anchor",
                *key,
                blind_rows[index]["blind_item_id"],
                n=32,
            )
        )
    anchor_bucket_order = sorted(
        anchor_buckets,
        key=lambda key: stable_hex(
            SUPPORT_NEED_PACKET_PROTOCOL,
            seed,
            "human-anchor-bucket",
            *key,
            n=32,
        ),
    )
    anchor_indices: list[int] = []
    while len(anchor_indices) < human_anchor_size:
        progress = False
        for key in anchor_bucket_order:
            indices = anchor_buckets[key]
            if indices and len(anchor_indices) < human_anchor_size:
                anchor_indices.append(indices.pop(0))
                progress = True
        if not progress:
            raise RuntimeError("human-anchor stratification exhausted early")
    anchor_rows = [blind_rows[index] for index in sorted(anchor_indices)]
    template_rows = [
        {
            "blind_item_id": row["blind_item_id"],
            "support_mode": None,
            "goals": [],
            "dialogue_phase": None,
            "nonclinical_urgency": None,
            "advice_rejected": None,
            "advice_requested": None,
            "one_small_step_requested": None,
            "listen_first_requested": None,
            "question_or_task_burden_limit": None,
            "abstain": None,
            "confidence": None,
            "notes": "",
        }
        for row in anchor_rows
    ]
    problem_counts = Counter(row["problem_type"] for row in lineage_rows)
    emotion_counts = Counter(row["emotion_type"] for row in lineage_rows)
    position_counts = Counter(row["position_stratum"] for row in lineage_rows)
    anchor_lineage = [lineage_rows[index] for index in anchor_indices]
    anchor_position_counts = Counter(
        row["position_stratum"] for row in anchor_lineage
    )
    anchor_emotion_counts = Counter(row["emotion_type"] for row in anchor_lineage)
    selected_dialogue_ids = {
        str(row["dialogue_id"]) for row in lineage_rows
    }
    bank_overlap_ids = selected_dialogue_ids & strategy_bank_source_ids
    bank_overlap_card_count = sum(
        str(row["source_dialogue_id"]) in selected_dialogue_ids
        for row in strategy_bank_rows
    )
    core = {
        "protocol": SUPPORT_NEED_PACKET_PROTOCOL,
        "status": "PREPARED_TRAIN_ONLY_OUTCOME_BLIND_AWAITING_HUMAN_ANNOTATION",
        "seed": int(seed),
        "packet_size": len(blind_rows),
        "human_anchor_size": len(anchor_rows),
        "unique_dialogues": len({row["dialogue_id"] for row in lineage_rows}),
        "selection": {
            "split": "train",
            "evoemo_overlap_excluded": True,
            "existing_seed_dialogues_excluded": True,
            "one_state_per_dialogue": True,
            "problem_type_round_robin": True,
            "position_strata": list(POSITION_STRATA),
            "position_counts": dict(sorted(position_counts.items())),
            "problem_type_counts": dict(sorted(problem_counts.items())),
            "emotion_type_counts": dict(sorted(emotion_counts.items())),
            "human_anchor_selection": "round-robin-emotion-by-position",
            "human_anchor_position_counts": dict(
                sorted(anchor_position_counts.items())
            ),
            "human_anchor_emotion_counts": dict(
                sorted(anchor_emotion_counts.items())
            ),
            "metadata_used_only_for_sampling": True,
        },
        "bank_v2_requirement": {
            "selected_dialogues_must_be_removed_from_cards": True,
            "current_raw_bank_source_dialogue_overlap_count": len(
                bank_overlap_ids
            ),
            "current_raw_bank_card_overlap_count": int(bank_overlap_card_count),
            "required_bank_v2_source_dialogue_overlap_count": 0,
            "required_bank_v2_card_overlap_count": 0,
            "selected_dialogue_ids_sha256": sha256_text(
                canonical_json(
                    sorted(row["dialogue_id"] for row in lineage_rows)
                )
            ),
            "current_raw_bank_overlap_dialogue_ids_sha256": sha256_text(
                canonical_json(sorted(bank_overlap_ids))
            ),
        },
        "blind_packet_sha256": sha256_text(canonical_json(blind_rows)),
        "human_anchor_packet_sha256": sha256_text(canonical_json(anchor_rows)),
        "annotation_template_sha256": sha256_text(canonical_json(template_rows)),
        "lineage_sha256": sha256_text(canonical_json(lineage_rows)),
        "source_lineage": {
            "esconv_sha256": sha256_file(esconv_path),
            "split_manifest_sha256": sha256_file(split_manifest_path),
            "existing_seed_sources_sha256": sha256_file(
                existing_seed_sources_path
            ),
            "current_raw_strategy_bank_sha256": sha256_file(
                strategy_bank_path
            ),
        },
        "forbidden_blind_keys": sorted(FORBIDDEN_BLIND_KEYS),
        "target_supporter_responses_used": False,
        "target_strategy_annotations_used": False,
        "survey_outcomes_used": False,
        "internal_test_outcomes_opened": False,
        "external_outcomes_opened": False,
        "api_calls_made": 0,
    }
    contract = {
        **core,
        "contract_sha256": sha256_text(canonical_json(core)),
    }
    return {
        "contract": contract,
        "blind_rows": blind_rows,
        "human_anchor_rows": anchor_rows,
        "annotation_template_rows": template_rows,
        "lineage_rows": lineage_rows,
    }
