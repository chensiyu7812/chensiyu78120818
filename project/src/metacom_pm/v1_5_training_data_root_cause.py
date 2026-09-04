"""Train-only root-cause checks for PM-v1.5 supervision data.

These helpers inspect realized retrieval composition, not response quality or
held-out outcomes.  They are intentionally separate from model fitting: a
dataset must first show that its requested component treatment is identifiable.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .contracts import MemorySource, parse_action_id
from .io import canonical_json, read_json, sha256_text


ROOT_CAUSE_AUDIT_PROTOCOL = "pm-v1.5-training-data-root-cause-audit-v1"
LEARNABLE_TRANSFER_TRAINING_PROTOCOL = (
    "pm-v1.5-learnable-transfer-training-data-v2"
)


def require_learnable_transfer_training_contract(
    path: str | Path,
) -> dict[str, Any]:
    """Load the redesigned training-data contract and verify its self binding."""

    contract = read_json(path)
    recorded = str(contract.get("contract_sha256") or "")
    payload = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }
    actual = sha256_text(canonical_json(payload))
    if recorded != actual:
        raise RuntimeError(
            "learnable-transfer training contract hash mismatch: "
            f"recorded={recorded}, actual={actual}"
        )
    if contract.get("protocol") != LEARNABLE_TRANSFER_TRAINING_PROTOCOL:
        raise RuntimeError("unexpected learnable-transfer training protocol")
    if contract.get("status") != "DESIGN_FROZEN_BEFORE_NEW_GENERATION":
        raise RuntimeError("learnable-transfer training contract is not frozen")
    return contract


def summarize_memory_treatment_composition(
    *,
    action_rows: Iterable[Mapping[str, Any]],
    evaluator_contexts_by_state: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize selected helpful/non-helpful items at state-action grain."""

    totals: collections.Counter[str] = collections.Counter()
    by_regime: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for row in action_rows:
        sources, _ = parse_action_id(str(row["requested_action_id"]))
        if not sources:
            continue
        state_id = str(row["state_id"])
        context = evaluator_contexts_by_state.get(state_id)
        if context is None:
            raise ValueError(f"missing evaluator context for state {state_id}")
        annotations = {
            str(item["memory_id"]): item
            for item in context.get("memory_annotations") or []
        }
        selected_ids = [str(value) for value in row.get("selected_memory_ids") or []]
        try:
            selected = [annotations[memory_id] for memory_id in selected_ids]
        except KeyError as exc:
            raise ValueError(
                f"selected memory {exc.args[0]} lacks an evaluator annotation"
            ) from exc
        utilities = collections.Counter(
            str(item["item_utility"]) for item in selected
        )
        counters = (totals, by_regime[str(context["regime"])])
        for counter in counters:
            counter["rows"] += 1
            counter["selected_items"] += len(selected)
            counter["helpful_items"] += utilities["helpful"]
            counter["irrelevant_items"] += utilities["irrelevant"]
            counter["harmful_items"] += utilities["harmful"]
            has_helpful = utilities["helpful"] > 0
            has_nonhelpful = utilities["irrelevant"] + utilities["harmful"] > 0
            counter["rows_with_helpful"] += int(has_helpful)
            counter["rows_with_nonhelpful"] += int(has_nonhelpful)
            counter["rows_with_mixed_helpful_and_nonhelpful"] += int(
                has_helpful and has_nonhelpful
            )

    def finalize(counter: collections.Counter[str]) -> dict[str, Any]:
        rows = int(counter["rows"])
        result = {key: int(value) for key, value in sorted(counter.items())}
        result["mean_selected_items"] = (
            float(counter["selected_items"]) / rows if rows else None
        )
        result["mixed_helpful_nonhelpful_row_rate"] = (
            float(counter["rows_with_mixed_helpful_and_nonhelpful"]) / rows
            if rows
            else None
        )
        return result

    return {
        "protocol": ROOT_CAUSE_AUDIT_PROTOCOL,
        "all_non_m0_actions": finalize(totals),
        "by_regime": {
            regime: finalize(counter)
            for regime, counter in sorted(by_regime.items())
        },
    }


def matching_target_treatment_rows(
    *,
    action_rows: Iterable[Mapping[str, Any]],
    evaluator_contexts_by_state: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return rows requesting exactly the memory set targeted by the regime."""

    target_sources = {
        "profile_needed": frozenset({MemorySource.MP}),
        "summary_needed": frozenset({MemorySource.MS}),
        "event_needed": frozenset({MemorySource.ME}),
        "multi_source_needed": frozenset(MemorySource),
        "memory_harmful": frozenset(MemorySource),
    }
    result = []
    for row in action_rows:
        context = evaluator_contexts_by_state.get(str(row["state_id"]))
        if context is None:
            raise ValueError(f"missing evaluator context for state {row['state_id']}")
        target = target_sources.get(str(context["regime"]))
        if target is None:
            continue
        sources, _ = parse_action_id(str(row["requested_action_id"]))
        if sources == target:
            result.append(row)
    return result
