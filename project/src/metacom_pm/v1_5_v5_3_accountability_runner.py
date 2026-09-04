"""Runtime wiring for the frozen V5.3 accountability row schema.

The schema module defines what one snapshot contains.  This module supplies
the missing lifecycle plumbing: separate append-only PRE, POST, and REVIEW
ledgers; deterministic stage transitions; and cross-stage immutability
checks.  It is outcome-agnostic and performs no API calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .v1_5_v5_3_accountability import (
    FailureOwner,
    LifecycleStage,
    QualityOutcome,
    RiskOutcome,
    StagewiseAccountabilityRow,
    TernaryAssessment,
    accountability_row_id,
    append_accountability_row,
    load_accountability_ledger,
    validate_accountability_rows,
)
from .v1_5_v5_3_typed_response_program import TypedResponseExecutionResult


LogicalKey = tuple[str, str, str]


def logical_key(row: StagewiseAccountabilityRow) -> LogicalKey:
    return (row.state_id, row.policy_condition, row.seed_label)


_PRE_IMMUTABLE_FIELDS = (
    "row_id",
    "state_id",
    "user_id",
    "semantic_family",
    "counterfactual_group_id",
    "policy_condition",
    "seed_label",
    "candidate_ids_topk",
    "exact_rank1_id",
    "candidate_owner_id",
    "candidate_version",
    "pm_probability",
    "pm_threshold",
    "pm_decision",
    "requested_action",
)

_POST_IMMUTABLE_FIELDS = _PRE_IMMUTABLE_FIELDS + (
    "realized_action",
    "generator_received_evidence_ids",
    "generator_reported_used_evidence_ids",
    "normalized_used_evidence_ids",
    "used_evidence_ids",
    "prompt_tokens",
    "total_tokens",
    "fallback_reason",
)


def _require_same_fields(
    earlier: StagewiseAccountabilityRow,
    later: StagewiseAccountabilityRow,
    fields: Iterable[str],
) -> None:
    if logical_key(earlier) != logical_key(later):
        raise ValueError("accountability transition changed logical row identity")
    changed = [field for field in fields if getattr(earlier, field) != getattr(later, field)]
    if changed:
        raise ValueError(f"accountability transition changed immutable fields: {changed}")


@dataclass(frozen=True)
class AccountabilityLedgerPaths:
    root: Path

    @property
    def pre(self) -> Path:
        return self.root / "pre_generation_accountability.jsonl"

    @property
    def post(self) -> Path:
        return self.root / "post_generation_accountability.jsonl"

    @property
    def review(self) -> Path:
        return self.root / "post_semantic_review_accountability.jsonl"


class StagewiseAccountabilityLedger:
    """Append-only lifecycle store with cross-stage integrity checks."""

    def __init__(self, root: str | Path):
        self.paths = AccountabilityLedgerPaths(Path(root))
        self.paths.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _index(path: Path) -> dict[LogicalKey, StagewiseAccountabilityRow]:
        if not path.exists():
            return {}
        return {logical_key(row): row for row in load_accountability_ledger(path)}

    def append_pre(self, row: StagewiseAccountabilityRow) -> None:
        if row.lifecycle_stage is not LifecycleStage.PRE_GENERATION:
            raise ValueError("append_pre requires PRE_GENERATION")
        append_accountability_row(self.paths.pre, row)

    def append_post(self, row: StagewiseAccountabilityRow) -> None:
        if row.lifecycle_stage is not LifecycleStage.POST_GENERATION:
            raise ValueError("append_post requires POST_GENERATION")
        previous = self._index(self.paths.pre).get(logical_key(row))
        if previous is None:
            raise ValueError("POST_GENERATION row has no PRE_GENERATION parent")
        _require_same_fields(previous, row, _PRE_IMMUTABLE_FIELDS)
        append_accountability_row(self.paths.post, row)

    def append_review(self, row: StagewiseAccountabilityRow) -> None:
        if row.lifecycle_stage is not LifecycleStage.POST_SEMANTIC_REVIEW:
            raise ValueError("append_review requires POST_SEMANTIC_REVIEW")
        previous = self._index(self.paths.post).get(logical_key(row))
        if previous is None:
            raise ValueError("POST_SEMANTIC_REVIEW row has no POST_GENERATION parent")
        _require_same_fields(previous, row, _POST_IMMUTABLE_FIELDS)
        append_accountability_row(self.paths.review, row)

    def validate(
        self,
        *,
        expected_keys: set[LogicalKey] | None = None,
        require_post: bool = False,
        require_review: bool = False,
    ) -> dict[str, int]:
        pre = list(self._index(self.paths.pre).values())
        post = list(self._index(self.paths.post).values())
        review = list(self._index(self.paths.review).values())
        validate_accountability_rows(pre, expected_keys=expected_keys)
        pre_keys = {logical_key(row) for row in pre}
        post_keys = {logical_key(row) for row in post}
        review_keys = {logical_key(row) for row in review}
        if not post_keys <= pre_keys:
            raise ValueError("POST_GENERATION contains keys absent from PRE_GENERATION")
        if not review_keys <= post_keys:
            raise ValueError("POST_SEMANTIC_REVIEW contains keys absent from POST_GENERATION")
        if require_post and post_keys != pre_keys:
            raise ValueError("POST_GENERATION ledger is incomplete")
        if require_review and review_keys != pre_keys:
            raise ValueError("POST_SEMANTIC_REVIEW ledger is incomplete")
        return {"pre": len(pre), "post": len(post), "review": len(review)}


def pre_generation_row(
    *,
    state_id: str,
    user_id: str,
    semantic_family: str,
    counterfactual_group_id: str,
    policy_condition: str,
    seed_label: str,
    candidate_ids_topk: dict,
    exact_rank1_id: dict,
    candidate_owner_id: dict,
    candidate_version: dict,
    pm_probability: dict,
    pm_threshold: dict,
    pm_decision: dict,
    requested_action: str,
) -> StagewiseAccountabilityRow:
    """Construct the PRE snapshot used by all formal policy conditions."""

    unknown = {component: TernaryAssessment.UNKNOWN for component in ("MP", "MS", "ME", "RS")}
    return StagewiseAccountabilityRow(
        row_id=accountability_row_id(
            state_id=state_id,
            policy_condition=policy_condition,
            seed_label=seed_label,
        ),
        state_id=state_id,
        user_id=user_id,
        semantic_family=semantic_family,
        counterfactual_group_id=counterfactual_group_id,
        policy_condition=policy_condition,
        seed_label=seed_label,
        lifecycle_stage=LifecycleStage.PRE_GENERATION,
        candidate_ids_topk=candidate_ids_topk,
        exact_rank1_id=exact_rank1_id,
        candidate_owner_id=candidate_owner_id,
        candidate_version=candidate_version,
        retrieval_fit_label=dict(unknown),
        eligibility_owner_time=dict(unknown),
        eligibility_goal_function=dict(unknown),
        eligibility_boundary_burden=dict(unknown),
        eligibility_specific_increment=dict(unknown),
        pm_probability=pm_probability,
        pm_threshold=pm_threshold,
        pm_decision=pm_decision,
        pm_correct=dict(unknown),
        requested_action=requested_action,
        realized_action=None,
        generator_received_evidence_ids=[],
        generator_reported_used_evidence_ids=[],
        normalized_used_evidence_ids=[],
        used_evidence_ids=[],
        required_evidence_use=dict(unknown),
        functional_contribution=dict(unknown),
        grounding_fidelity=dict(unknown),
        atomic_move_compliance=TernaryAssessment.UNKNOWN,
        execution_valid=TernaryAssessment.UNKNOWN,
        scaffold_exposure=TernaryAssessment.UNKNOWN,
        fallback_reason=None,
        quality_outcome=QualityOutcome.UNKNOWN,
        risk_outcome=RiskOutcome.UNKNOWN,
        prompt_tokens=None,
        total_tokens=None,
        primary_failure_owner=FailureOwner.UNKNOWN,
        secondary_failure_owner=FailureOwner.UNKNOWN,
    )


def post_row_from_execution(
    pre: StagewiseAccountabilityRow,
    execution: TypedResponseExecutionResult,
    *,
    generator_received_evidence_ids: list[str],
    prompt_tokens: int,
    total_tokens: int,
) -> StagewiseAccountabilityRow:
    """Create a POST snapshot without treating generator telemetry as gold."""

    if pre.lifecycle_stage is not LifecycleStage.PRE_GENERATION:
        raise ValueError("post_row_from_execution requires a PRE_GENERATION row")
    if execution.requested_action_id != pre.requested_action:
        raise ValueError("typed execution requested action differs from accountability PRE row")
    response = execution.response
    if response is None or execution.realized_action_id is None:
        raise ValueError(
            "execution produced no final response; apply the deterministic fallback before "
            "writing POST_GENERATION"
        )
    reported = list(execution.reported_used_evidence_ids)
    normalized = list(execution.used_evidence_ids)
    fallback_reason = (
        execution.status if execution.status in {"fell_back_to_m0", "no_structured_output"} else None
    )
    execution_valid = (
        TernaryAssessment.YES
        if execution.status in {"clean", "fixed_by_rewrite"}
        else TernaryAssessment.NO
    )
    data = pre.model_dump(mode="python")
    data.update(
        {
            "lifecycle_stage": LifecycleStage.POST_GENERATION,
            "realized_action": execution.realized_action_id,
            "generator_received_evidence_ids": generator_received_evidence_ids,
            "generator_reported_used_evidence_ids": reported,
            "normalized_used_evidence_ids": normalized,
            "used_evidence_ids": normalized,
            "execution_valid": execution_valid,
            "fallback_reason": fallback_reason,
            "quality_outcome": QualityOutcome.UNKNOWN,
            "risk_outcome": RiskOutcome.UNKNOWN,
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
            "primary_failure_owner": FailureOwner.UNKNOWN,
            "secondary_failure_owner": FailureOwner.UNKNOWN,
        }
    )
    return StagewiseAccountabilityRow.model_validate(data)
