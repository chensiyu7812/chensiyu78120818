from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import MemoryItem, StrategyCard, parse_action_id
from .io import canonical_json, iter_jsonl, read_json, sha256_file, sha256_text
from .pm_v1_6_contracts import (
    ACTION_LINEAGE_PROTOCOL,
    RetrievalAttempt,
    realized_action_from_evidence,
)
from .pm_v1_6_preflight import PREFLIGHT_PROTOCOL, require_preflight_pass
from .pm_v2_contracts import ActionLabel

FORMAL_SWEEP_PROTOCOL = "pm-v1.6-formal-action-sweep-artifact-v1"
BOUND_LABEL_PROTOCOL = "pm-v1.6-lineage-bound-action-label-v1"


class FormalActionOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    protocol: str = FORMAL_SWEEP_PROTOCOL
    card_id: str
    state_id: str
    user_id: str
    split: str
    regime: str
    requested_action_id: str
    retrieval_attempts: list[RetrievalAttempt]
    realized_action_id: str
    prompt_equivalence_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_equivalence_class_size: int = Field(ge=1)
    prompt_equivalence_actions: list[str]
    shared_quality_label_weight: float = Field(gt=0.0, le=1.0)
    response: str = Field(min_length=1)
    selected_memory_ids: list[str]
    selected_strategy_ids: list[str]
    memory_view: list[MemoryItem]
    strategy_view: list[StrategyCard]
    candidate_memory_view: list[MemoryItem]
    candidate_strategy_view: list[StrategyCard]
    generator_cost: dict[str, Any]
    model_name: str
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_hash: str
    source_sweep_provenance: dict[str, Any]
    lineage_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("requested_action_id", "realized_action_id")
    @classmethod
    def valid_action(cls, value: str) -> str:
        parse_action_id(value)
        return value

    @model_validator(mode="after")
    def coherent(self) -> "FormalActionOutcome":
        inferred = realized_action_from_evidence(
            [item.source for item in self.memory_view],
            strategy_card_count=len(self.strategy_view),
        )
        if inferred != self.realized_action_id:
            raise ValueError("formal realized action differs from actual evidence")
        if self.selected_memory_ids != [item.memory_id for item in self.memory_view]:
            raise ValueError("formal selected memory IDs mismatch")
        if self.selected_strategy_ids != [item.strategy_id for item in self.strategy_view]:
            raise ValueError("formal selected strategy IDs mismatch")
        if self.prompt_equivalence_class_size != len(
            self.prompt_equivalence_actions
        ):
            raise ValueError("prompt equivalence class size/actions mismatch")
        if self.requested_action_id not in self.prompt_equivalence_actions:
            raise ValueError("requested action missing from prompt equivalence class")
        expected_weight = 1.0 / self.prompt_equivalence_class_size
        if abs(self.shared_quality_label_weight - expected_weight) > 1e-12:
            raise ValueError("shared quality label weight is not inverse alias size")
        payload = self.model_dump(mode="json", exclude={"lineage_sha256"})
        if self.lineage_sha256 != sha256_text(canonical_json(payload)):
            raise ValueError("formal action lineage hash mismatch")
        return self


def _preflight_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        key = (str(row["state_id"]), str(row["requested_action_id"]))
        if key in result:
            raise RuntimeError(f"duplicate preflight action key: {key}")
        result[key] = row
    if not result:
        raise RuntimeError("preflight action plan is empty")
    return result


def finalize_action_sweep(
    *,
    source_outcomes_path: str | Path,
    preflight_rows_path: str | Path,
    preflight_summary_path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_outcomes_path = Path(source_outcomes_path)
    preflight_rows_path = Path(preflight_rows_path)
    preflight_summary_path = Path(preflight_summary_path)
    preflight_summary = read_json(preflight_summary_path)
    require_preflight_pass(preflight_summary)
    if preflight_summary.get("protocol") != PREFLIGHT_PROTOCOL:
        raise RuntimeError("preflight protocol mismatch")
    if preflight_summary.get("action_preflight_sha256") not in {
        None,
        sha256_file(preflight_rows_path),
    }:
        raise RuntimeError("preflight summary/action plan hash mismatch")

    plan = _preflight_index(list(iter_jsonl(preflight_rows_path)))
    source_rows = list(iter_jsonl(source_outcomes_path))
    source_index: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in source_rows:
        row = dict(raw)
        requested = str(row.get("action_id") or row.get("requested_action_id") or "")
        key = (str(row["state_id"]), requested)
        if key in source_index:
            raise RuntimeError(f"duplicate source sweep outcome: {key}")
        source_index[key] = row
    if set(source_index) != set(plan):
        missing = sorted(set(plan) - set(source_index))
        extra = sorted(set(source_index) - set(plan))
        raise RuntimeError(
            "source sweep/preflight action matrices differ: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    formal_rows: list[dict[str, Any]] = []
    transitions: Counter[tuple[str, str]] = Counter()
    for key in sorted(plan):
        preflight = plan[key]
        source = source_index[key]
        memory_view = [MemoryItem.model_validate(row) for row in source.get("memory_view") or []]
        strategy_view = [
            StrategyCard.model_validate(row) for row in source.get("strategy_view") or []
        ]
        candidate_memory = [
            MemoryItem.model_validate(row)
            for row in (
                source.get("candidate_memory_view")
                if source.get("candidate_memory_view") is not None
                else source.get("memory_view") or []
            )
        ]
        candidate_strategy = [
            StrategyCard.model_validate(row)
            for row in (
                source.get("candidate_strategy_view")
                if source.get("candidate_strategy_view") is not None
                else source.get("strategy_view") or []
            )
        ]
        realized = realized_action_from_evidence(
            [item.source for item in memory_view],
            strategy_card_count=len(strategy_view),
        )
        if realized != preflight["realized_action_id"]:
            raise RuntimeError(
                f"sweep/preflight realized action drift for {key}: "
                f"sweep={realized}, preflight={preflight['realized_action_id']}"
            )
        prompt_hash = str(source.get("prompt_hash") or "")
        if prompt_hash != str(preflight["prompt_sha256"]):
            raise RuntimeError(f"sweep/preflight prompt hash drift for {key}")
        if [item.memory_id for item in candidate_memory] != list(
            preflight["candidate_memory_ids"]
        ):
            raise RuntimeError(f"candidate memory drift for {key}")
        if [item.strategy_id for item in candidate_strategy] != list(
            preflight["candidate_strategy_ids"]
        ):
            raise RuntimeError(f"candidate strategy drift for {key}")

        payload = {
            "protocol": FORMAL_SWEEP_PROTOCOL,
            "card_id": str(source["card_id"]),
            "state_id": str(source["state_id"]),
            "user_id": str(source["user_id"]),
            "split": str(preflight["split"]),
            "regime": str(preflight["regime"]),
            "requested_action_id": str(preflight["requested_action_id"]),
            "retrieval_attempts": list(preflight["retrieval_attempts"]),
            "realized_action_id": realized,
            "prompt_equivalence_id": str(preflight["prompt_equivalence_id"]),
            "prompt_equivalence_class_size": int(
                preflight["prompt_equivalence_class_size"]
            ),
            "prompt_equivalence_actions": list(
                preflight["prompt_equivalence_actions"]
            ),
            "shared_quality_label_weight": float(
                preflight["shared_quality_label_weight"]
            ),
            "response": str(source["response"]),
            "selected_memory_ids": [item.memory_id for item in memory_view],
            "selected_strategy_ids": [item.strategy_id for item in strategy_view],
            "memory_view": [item.model_dump(mode="json") for item in memory_view],
            "strategy_view": [item.model_dump(mode="json") for item in strategy_view],
            "candidate_memory_view": [
                item.model_dump(mode="json") for item in candidate_memory
            ],
            "candidate_strategy_view": [
                item.model_dump(mode="json") for item in candidate_strategy
            ],
            "generator_cost": dict(source["cost"]),
            "model_name": str(source["model_name"]),
            "prompt_sha256": prompt_hash,
            "request_hash": str(source["request_hash"]),
            "source_sweep_provenance": dict(source.get("provenance") or {}),
        }
        payload["lineage_sha256"] = sha256_text(canonical_json(payload))
        validated = FormalActionOutcome.model_validate(payload)
        formal_rows.append(validated.model_dump(mode="json"))
        transitions[(payload["requested_action_id"], realized)] += 1

    alias_rows = [
        row for row in formal_rows if row["prompt_equivalence_class_size"] > 1
    ]
    mismatch_rows = [
        row
        for row in formal_rows
        if row["requested_action_id"] != row["realized_action_id"]
    ]
    summary = {
        "status": "COMPLETE",
        "protocol": FORMAL_SWEEP_PROTOCOL,
        "source_outcomes_sha256": sha256_file(source_outcomes_path),
        "preflight_rows_sha256": sha256_file(preflight_rows_path),
        "preflight_summary_sha256": sha256_file(preflight_summary_path),
        "formal_rows": len(formal_rows),
        "requested_realized_mismatch_rows": len(mismatch_rows),
        "requested_realized_mismatch_rate": (
            len(mismatch_rows) / len(formal_rows) if formal_rows else 0.0
        ),
        "prompt_alias_rows": len(alias_rows),
        "prompt_alias_rate": len(alias_rows) / len(formal_rows) if formal_rows else 0.0,
        "requested_to_realized_transition_matrix": [
            {
                "requested_action_id": requested,
                "realized_action_id": realized,
                "count": count,
            }
            for (requested, realized), count in sorted(transitions.items())
        ],
        "formal_matrix_sha256": sha256_text(canonical_json(formal_rows)),
        "source_effective_action_field_is_not_authoritative": True,
    }
    return summary, formal_rows


def bind_action_labels_to_lineage(
    *,
    labels_path: str | Path,
    formal_outcomes_path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels_path = Path(labels_path)
    formal_outcomes_path = Path(formal_outcomes_path)
    formal_rows = [
        FormalActionOutcome.model_validate(row)
        for row in iter_jsonl(formal_outcomes_path)
    ]
    lineage = {
        (row.state_id, row.requested_action_id): row for row in formal_rows
    }
    if len(lineage) != len(formal_rows):
        raise RuntimeError("duplicate formal outcome lineage key")
    labels = [ActionLabel.model_validate(row) for row in iter_jsonl(labels_path)]
    label_keys = {(row.state_id, row.action_id) for row in labels}
    if label_keys != set(lineage):
        missing = sorted(set(lineage) - label_keys)
        extra = sorted(label_keys - set(lineage))
        raise RuntimeError(
            "ActionLabel/formal outcome matrices differ: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    bound: list[dict[str, Any]] = []
    for label in labels:
        row = lineage[(label.state_id, label.action_id)]
        provenance = dict(label.provenance)
        forbidden_existing = {
            "pm_v1_6_lineage_sha256",
            "requested_action_id",
            "realized_action_id",
            "prompt_equivalence_id",
            "prompt_equivalence_class_size",
            "shared_quality_label_weight",
        } & set(provenance)
        if forbidden_existing:
            raise RuntimeError(
                "untrusted label already contains PM-v1.6 lineage fields: "
                f"{sorted(forbidden_existing)}"
            )
        provenance.update(
            {
                "lineage_protocol": BOUND_LABEL_PROTOCOL,
                "pm_v1_6_lineage_sha256": row.lineage_sha256,
                "requested_action_id": row.requested_action_id,
                "realized_action_id": row.realized_action_id,
                "prompt_equivalence_id": row.prompt_equivalence_id,
                "prompt_equivalence_class_size": row.prompt_equivalence_class_size,
                "shared_quality_label_weight": row.shared_quality_label_weight,
            }
        )
        bound_label = label.model_copy(update={"provenance": provenance})
        bound.append(bound_label.model_dump(mode="json"))
    summary = {
        "status": "COMPLETE",
        "protocol": BOUND_LABEL_PROTOCOL,
        "labels_sha256": sha256_file(labels_path),
        "formal_outcomes_sha256": sha256_file(formal_outcomes_path),
        "bound_labels": len(bound),
        "alias_weighted_labels": sum(
            float(row["provenance"]["shared_quality_label_weight"]) < 1.0
            for row in bound
        ),
        "bound_matrix_sha256": sha256_text(canonical_json(bound)),
    }
    return summary, bound
