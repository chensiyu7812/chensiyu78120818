"""Typed, pre-outcome contract for V5.3 stagewise accountability.

The ledger is deliberately not a judge and does not infer blame from reply
quality.  It makes every experimental realization addressable at
``state_id x policy_condition x seed_label`` and preserves the inputs and
observations needed to assign responsibility after the frozen reviews.

No outcome-generation code is imported here.  The module can therefore be
schema-frozen and tested before any paid call or semantic outcome is read.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal, Mapping, Sequence

from pydantic import Field, field_validator, model_validator

from .contracts import ALL_ACTION_IDS, StrictModel, parse_action_id
from .io import append_jsonl, canonical_json, iter_jsonl, stable_hex


ACCOUNTABILITY_PROTOCOL = "pm-v1.5-v5.3-stagewise-accountability-ledger-v1"
COMPONENTS = ("MP", "MS", "ME", "RS")
Component = Literal["MP", "MS", "ME", "RS"]


class LifecycleStage(str, Enum):
    PRE_GENERATION = "PRE_GENERATION"
    POST_GENERATION = "POST_GENERATION"
    POST_SEMANTIC_REVIEW = "POST_SEMANTIC_REVIEW"


class TernaryAssessment(str, Enum):
    YES = "YES"
    NO = "NO"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class QualityOutcome(str, Enum):
    POSITIVE = "POSITIVE"
    NONPOSITIVE = "NONPOSITIVE"
    TIE = "TIE"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class RiskOutcome(str, Enum):
    MATERIAL_RISK = "MATERIAL_RISK"
    NO_MATERIAL_RISK = "NO_MATERIAL_RISK"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class FailureOwner(str, Enum):
    NONE = "NONE"
    RETRIEVER = "RETRIEVER"
    HARD_ELIGIBILITY = "HARD_ELIGIBILITY"
    PM_STEP1 = "PM_STEP1"
    EXECUTOR = "EXECUTOR"
    GENERATOR = "GENERATOR"
    RESOURCE_NONPOSITIVE = "RESOURCE_NONPOSITIVE"
    OUTCOME_NOISE = "OUTCOME_NOISE"
    FULL_SYSTEM = "FULL_SYSTEM"
    UNKNOWN = "UNKNOWN"


def accountability_row_id(
    *, state_id: str, policy_condition: str, seed_label: str
) -> str:
    return "acct_" + stable_hex(
        ACCOUNTABILITY_PROTOCOL, state_id, policy_condition, seed_label, n=24
    )


def _component_keys(value: Mapping[str, object], field_name: str) -> Mapping[str, object]:
    keys = set(value)
    expected = set(COMPONENTS)
    if keys != expected:
        raise ValueError(
            f"{field_name} must contain exactly {sorted(expected)}; "
            f"missing={sorted(expected - keys)}, extra={sorted(keys - expected)}"
        )
    return value


class StagewiseAccountabilityRow(StrictModel):
    protocol: Literal[
        "pm-v1.5-v5.3-stagewise-accountability-ledger-v1"
    ] = ACCOUNTABILITY_PROTOCOL
    row_id: str = Field(min_length=1)
    state_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    semantic_family: str = Field(min_length=1)
    counterfactual_group_id: str = Field(min_length=1)
    policy_condition: str = Field(min_length=1)
    seed_label: str = Field(min_length=1)
    lifecycle_stage: LifecycleStage

    candidate_ids_topk: dict[Component, list[str]]
    exact_rank1_id: dict[Component, str | None]
    candidate_owner_id: dict[Component, str | None]
    candidate_version: dict[Component, str | None]
    retrieval_fit_label: dict[Component, TernaryAssessment]
    eligibility_owner_time: dict[Component, TernaryAssessment]
    eligibility_goal_function: dict[Component, TernaryAssessment]
    eligibility_boundary_burden: dict[Component, TernaryAssessment]
    eligibility_specific_increment: dict[Component, TernaryAssessment]

    pm_probability: dict[Component, float | None]
    pm_threshold: dict[Component, float | None]
    pm_decision: dict[Component, bool | None]
    pm_correct: dict[Component, TernaryAssessment]
    requested_action: str
    realized_action: str | None

    generator_received_evidence_ids: list[str]
    generator_reported_used_evidence_ids: list[str]
    normalized_used_evidence_ids: list[str]
    used_evidence_ids: list[str]
    required_evidence_use: dict[Component, TernaryAssessment]
    functional_contribution: dict[Component, TernaryAssessment]
    grounding_fidelity: dict[Component, TernaryAssessment]
    atomic_move_compliance: TernaryAssessment
    execution_valid: TernaryAssessment
    scaffold_exposure: TernaryAssessment
    fallback_reason: str | None

    quality_outcome: QualityOutcome
    risk_outcome: RiskOutcome
    prompt_tokens: int | None = Field(ge=0)
    total_tokens: int | None = Field(ge=0)
    primary_failure_owner: FailureOwner
    secondary_failure_owner: FailureOwner

    @field_validator(
        "candidate_ids_topk",
        "exact_rank1_id",
        "candidate_owner_id",
        "candidate_version",
        "retrieval_fit_label",
        "eligibility_owner_time",
        "eligibility_goal_function",
        "eligibility_boundary_burden",
        "eligibility_specific_increment",
        "pm_probability",
        "pm_threshold",
        "pm_decision",
        "pm_correct",
        "required_evidence_use",
        "functional_contribution",
        "grounding_fidelity",
    )
    @classmethod
    def all_component_keys(cls, value, info):
        return _component_keys(value, info.field_name)

    @field_validator("requested_action")
    @classmethod
    def requested_action_is_legal(cls, value: str) -> str:
        parse_action_id(value)
        return value

    @field_validator("realized_action")
    @classmethod
    def realized_action_is_legal(cls, value: str | None) -> str | None:
        if value is not None:
            parse_action_id(value)
        return value

    @field_validator("pm_probability", "pm_threshold")
    @classmethod
    def probabilities_are_unit_interval(cls, value, info):
        for component, item in value.items():
            if item is not None and not 0.0 <= item <= 1.0:
                raise ValueError(
                    f"{info.field_name}[{component}] must be within [0, 1]"
                )
        return value

    @field_validator(
        "candidate_ids_topk",
        "generator_received_evidence_ids",
        "generator_reported_used_evidence_ids",
        "normalized_used_evidence_ids",
        "used_evidence_ids",
    )
    @classmethod
    def ids_are_unique(cls, value, info):
        groups = value.values() if isinstance(value, dict) else (value,)
        for ids in groups:
            if len(ids) != len(set(ids)):
                raise ValueError(f"{info.field_name} contains duplicate IDs")
        return value

    @model_validator(mode="after")
    def coherent(self):
        expected_row_id = accountability_row_id(
            state_id=self.state_id,
            policy_condition=self.policy_condition,
            seed_label=self.seed_label,
        )
        if self.row_id != expected_row_id:
            raise ValueError("row_id does not match the frozen logical row identity")

        for component in COMPONENTS:
            rank1 = self.exact_rank1_id[component]
            topk = self.candidate_ids_topk[component]
            if rank1 is not None and (not topk or topk[0] != rank1):
                raise ValueError(
                    f"exact_rank1_id[{component}] must equal candidate_ids_topk[{component}][0]"
                )
            if rank1 is None and (
                self.candidate_owner_id[component] is not None
                or self.candidate_version[component] is not None
            ):
                raise ValueError(
                    f"absent Rank-1 {component} cannot have owner/version metadata"
                )

        if self.normalized_used_evidence_ids != self.used_evidence_ids:
            raise ValueError(
                "used_evidence_ids must equal normalized_used_evidence_ids; "
                "the raw model report belongs only in generator_reported_used_evidence_ids"
            )

        if self.total_tokens is not None and self.prompt_tokens is not None:
            if self.total_tokens < self.prompt_tokens:
                raise ValueError("total_tokens cannot be lower than prompt_tokens")

        if self.lifecycle_stage is LifecycleStage.PRE_GENERATION:
            if self.realized_action is not None:
                raise ValueError("PRE_GENERATION row cannot have realized_action")
            if any(
                (
                    self.generator_received_evidence_ids,
                    self.generator_reported_used_evidence_ids,
                    self.normalized_used_evidence_ids,
                    self.used_evidence_ids,
                )
            ):
                raise ValueError("PRE_GENERATION row cannot contain execution traces")
            if self.prompt_tokens is not None or self.total_tokens is not None:
                raise ValueError("PRE_GENERATION row cannot contain token outcomes")
            if self.quality_outcome is not QualityOutcome.UNKNOWN:
                raise ValueError("PRE_GENERATION quality_outcome must be UNKNOWN")
            if self.risk_outcome is not RiskOutcome.UNKNOWN:
                raise ValueError("PRE_GENERATION risk_outcome must be UNKNOWN")
            if self.primary_failure_owner is not FailureOwner.UNKNOWN:
                raise ValueError("PRE_GENERATION primary_failure_owner must be UNKNOWN")
            if self.secondary_failure_owner is not FailureOwner.UNKNOWN:
                raise ValueError("PRE_GENERATION secondary_failure_owner must be UNKNOWN")
        else:
            if self.realized_action is None:
                raise ValueError("post-generation row requires realized_action")
            if self.prompt_tokens is None or self.total_tokens is None:
                raise ValueError("post-generation row requires token accounting")

        if self.requested_action == "M0+R0":
            if self.generator_received_evidence_ids:
                raise ValueError("M0+R0 cannot send evidence to the generator")
            if self.normalized_used_evidence_ids or self.used_evidence_ids:
                raise ValueError("M0+R0 normalized evidence use must be empty")

        if self.lifecycle_stage is LifecycleStage.POST_SEMANTIC_REVIEW:
            if self.quality_outcome is QualityOutcome.UNKNOWN:
                raise ValueError("reviewed row requires a quality outcome")
            if self.risk_outcome is RiskOutcome.UNKNOWN:
                raise ValueError("reviewed row requires a risk outcome")
            if self.primary_failure_owner is FailureOwner.UNKNOWN:
                raise ValueError("reviewed row requires an explicit primary owner or NONE")
            if self.secondary_failure_owner is FailureOwner.UNKNOWN:
                raise ValueError("reviewed row requires an explicit secondary owner or NONE")
        return self


def validate_accountability_rows(
    rows: Sequence[StagewiseAccountabilityRow],
    *,
    expected_keys: set[tuple[str, str, str]] | None = None,
) -> None:
    """Fail on duplicate, missing, or unexpected experimental realizations."""

    observed: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row.state_id, row.policy_condition, row.seed_label)
        if key in observed:
            raise ValueError(f"duplicate accountability row key: {key}")
        observed.add(key)
    if expected_keys is not None and observed != expected_keys:
        raise ValueError(
            "accountability row identity mismatch: "
            f"missing={sorted(expected_keys - observed)}, "
            f"extra={sorted(observed - expected_keys)}"
        )


def load_accountability_ledger(path: str | Path) -> list[StagewiseAccountabilityRow]:
    # StrictModel deliberately rejects enum strings through the Python-object
    # validation path.  JSONL necessarily serializes enums as strings, so the
    # inverse operation must use Pydantic's JSON validation path; otherwise a
    # row written by append_accountability_row cannot be read back.
    rows = [
        StagewiseAccountabilityRow.model_validate_json(canonical_json(row))
        for row in iter_jsonl(path)
    ]
    validate_accountability_rows(rows)
    return rows


def append_accountability_row(
    path: str | Path,
    row: StagewiseAccountabilityRow,
    *,
    expected_protocol: str = ACCOUNTABILITY_PROTOCOL,
) -> None:
    """Append one validated final snapshot without overwriting prior rows."""

    if row.protocol != expected_protocol:
        raise ValueError("accountability protocol mismatch")
    existing = load_accountability_ledger(path) if Path(path).exists() else []
    key = (row.state_id, row.policy_condition, row.seed_label)
    if any((r.state_id, r.policy_condition, r.seed_label) == key for r in existing):
        raise ValueError(f"accountability row already exists for {key}")
    append_jsonl(path, row.model_dump(mode="json"))


def empty_component_map(value):
    """Small construction helper used by frozen blueprint writers/tests."""

    return {component: value() if callable(value) else value for component in COMPONENTS}


def legal_action_ids() -> tuple[str, ...]:
    """Expose the shared 16-action vocabulary without defining a second one."""

    return ALL_ACTION_IDS
