"""Outcome-blind, same-stack response baseline planning for PM V1.5 V5.3.

The planner deliberately stops before response generation.  It receives one
frozen candidate/Step-2 surface per state and produces six *logical* policy
bindings.  Bindings with the same state, action and seed are collapsed to one
physical call while every policy alias is retained.

The matched-random control permutes learned actions only within exchangeable
cells that have the same stratum, eligibility mask and per-component token-cost
vector.  Consequently it exactly preserves component ON counts and estimated
incremental resource cost without reading a response or evaluation outcome.
Small cells may make this arm alias the learned arm; that lack of contrast is
reported rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Literal, Mapping, Sequence

from .contracts import ALL_ACTION_IDS
from .v1_5b_policy_runtime import (
    COMPONENTS,
    Component,
    action_component_bits,
    compile_component_bits,
)


PolicyName = Literal[
    "always_off",
    "fixed_high_eligible",
    "transparent_rule",
    "cost_matched_fixed",
    "cost_and_on_rate_matched_random",
    "learned_pm_full",
]

POLICIES: tuple[PolicyName, ...] = (
    "always_off",
    "fixed_high_eligible",
    "transparent_rule",
    "cost_matched_fixed",
    "cost_and_on_rate_matched_random",
    "learned_pm_full",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_hex(*parts: str, length: int = 24) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return sha256(payload).hexdigest()[:length]


def _strict_bits(value: Mapping[str, bool], *, field: str) -> dict[Component, bool]:
    if set(value) != set(COMPONENTS):
        raise ValueError(f"{field} must contain exactly {COMPONENTS}")
    if any(type(value[component]) is not bool for component in COMPONENTS):
        raise TypeError(f"{field} values must be bool")
    return {component: value[component] for component in COMPONENTS}


@dataclass(frozen=True)
class FrozenResponseState:
    """Policy-independent material frozen before baseline action selection."""

    state_id: str
    stratum: str
    candidate_snapshot_sha256: str
    step2_shared_surface_sha256: str
    eligible_bits: Mapping[str, bool]
    candidate_ids: Mapping[str, str | None]
    incremental_tokens: Mapping[str, int]
    transparent_rule_bits: Mapping[str, bool]
    learned_pm_bits: Mapping[str, bool]

    def __post_init__(self) -> None:
        for field_name in (
            "state_id",
            "stratum",
            "candidate_snapshot_sha256",
            "step2_shared_surface_sha256",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        eligible = _strict_bits(self.eligible_bits, field="eligible_bits")
        transparent = _strict_bits(
            self.transparent_rule_bits, field="transparent_rule_bits"
        )
        learned = _strict_bits(self.learned_pm_bits, field="learned_pm_bits")
        if set(self.candidate_ids) != set(COMPONENTS):
            raise ValueError("candidate_ids must contain exactly four components")
        if set(self.incremental_tokens) != set(COMPONENTS):
            raise ValueError("incremental_tokens must contain exactly four components")
        for component in COMPONENTS:
            candidate = self.candidate_ids[component]
            if eligible[component] != bool(candidate and str(candidate).strip()):
                raise ValueError(
                    f"candidate identity/eligibility mismatch for {component}"
                )
            tokens = self.incremental_tokens[component]
            if type(tokens) is not int or tokens < 0:
                raise ValueError("incremental token counts must be nonnegative ints")
            if not eligible[component] and tokens != 0:
                raise ValueError("ineligible components must have zero incremental tokens")
            if transparent[component] and not eligible[component]:
                raise ValueError("transparent rule cannot enable an ineligible component")
            if learned[component] and not eligible[component]:
                raise ValueError("learned PM cannot enable an ineligible component")


@dataclass(frozen=True)
class BaselineFreeze:
    protocol: str
    seed_labels: tuple[str, ...]
    cost_matched_fixed_action_by_stratum: Mapping[str, str]
    matched_random_seed: str

    def __post_init__(self) -> None:
        if not self.protocol.strip() or not self.matched_random_seed.strip():
            raise ValueError("protocol and matched_random_seed must be non-empty")
        if not self.seed_labels or any(not seed.strip() for seed in self.seed_labels):
            raise ValueError("at least one non-empty seed label is required")
        if len(set(self.seed_labels)) != len(self.seed_labels):
            raise ValueError("seed labels must be unique")
        invalid = set(self.cost_matched_fixed_action_by_stratum.values()) - set(
            ALL_ACTION_IDS
        )
        if invalid:
            raise ValueError(f"invalid cost-matched action(s): {sorted(invalid)}")


def _project(bits: Mapping[str, bool], eligible: Mapping[str, bool]) -> dict[Component, bool]:
    return {
        component: bool(bits[component] and eligible[component])
        for component in COMPONENTS
    }


def _exchangeability_cell(state: FrozenResponseState) -> tuple[Any, ...]:
    return (
        state.stratum,
        tuple(bool(state.eligible_bits[component]) for component in COMPONENTS),
        tuple(int(state.incremental_tokens[component]) for component in COMPONENTS),
    )


def matched_random_actions(
    states: Sequence[FrozenResponseState], *, seed: str
) -> tuple[dict[str, dict[Component, bool]], dict[str, str], list[dict[str, Any]]]:
    """Protocol-hash permute learned actions in exactly exchangeable cells."""

    cells: dict[tuple[Any, ...], list[FrozenResponseState]] = {}
    for state in states:
        cells.setdefault(_exchangeability_cell(state), []).append(state)

    assigned: dict[str, dict[Component, bool]] = {}
    donors: dict[str, str] = {}
    audits: list[dict[str, Any]] = []
    for cell, members in sorted(cells.items(), key=lambda item: repr(item[0])):
        ordered = sorted(
            members,
            key=lambda state: (
                _stable_hex(seed, state.stratum, state.state_id, length=64),
                state.state_id,
            ),
        )
        shift = 0 if len(ordered) == 1 else 1 + int(
            _stable_hex(seed, repr(cell), "shift", length=8), 16
        ) % (len(ordered) - 1)
        before_counts = {component: 0 for component in COMPONENTS}
        after_counts = {component: 0 for component in COMPONENTS}
        before_cost = 0
        after_cost = 0
        changed = 0
        for index, recipient in enumerate(ordered):
            donor = ordered[(index + shift) % len(ordered)]
            donor_bits = _strict_bits(donor.learned_pm_bits, field="learned_pm_bits")
            random_bits = _project(donor_bits, recipient.eligible_bits)
            assigned[recipient.state_id] = random_bits
            donors[recipient.state_id] = donor.state_id
            changed += int(random_bits != dict(recipient.learned_pm_bits))
            for component in COMPONENTS:
                before_counts[component] += int(recipient.learned_pm_bits[component])
                after_counts[component] += int(random_bits[component])
                before_cost += int(recipient.learned_pm_bits[component]) * int(
                    recipient.incremental_tokens[component]
                )
                after_cost += int(random_bits[component]) * int(
                    recipient.incremental_tokens[component]
                )
        if before_counts != after_counts or before_cost != after_cost:
            raise AssertionError("matched-random invariants were not preserved")
        audits.append(
            {
                "cell_sha256": sha256(_canonical_json(cell).encode("utf-8")).hexdigest(),
                "stratum": cell[0],
                "states": len(ordered),
                "shift": shift,
                "changed_assignments": changed,
                "component_on_counts": before_counts,
                "estimated_incremental_tokens": before_cost,
                "exact_on_rate_match": True,
                "exact_estimated_cost_match": True,
            }
        )
    return assigned, donors, audits


def _policy_bits(
    state: FrozenResponseState,
    policy: PolicyName,
    *,
    freeze: BaselineFreeze,
    random_bits: Mapping[str, Mapping[str, bool]],
) -> tuple[dict[Component, bool], str | None]:
    eligible = _strict_bits(state.eligible_bits, field="eligible_bits")
    if policy == "always_off":
        return {component: False for component in COMPONENTS}, None
    if policy == "fixed_high_eligible":
        return eligible, None
    if policy == "transparent_rule":
        return _project(state.transparent_rule_bits, eligible), None
    if policy == "learned_pm_full":
        return _project(state.learned_pm_bits, eligible), None
    if policy == "cost_and_on_rate_matched_random":
        return _project(random_bits[state.state_id], eligible), None
    if policy == "cost_matched_fixed":
        try:
            fixed = freeze.cost_matched_fixed_action_by_stratum[state.stratum]
        except KeyError as exc:
            raise ValueError(
                f"missing cost-matched action for stratum {state.stratum!r}"
            ) from exc
        return _project(action_component_bits(fixed), eligible), fixed
    raise ValueError(f"unsupported policy: {policy}")


def build_response_baseline_plan(
    states: Sequence[FrozenResponseState], *, freeze: BaselineFreeze
) -> dict[str, Any]:
    """Build logical policy bindings plus deduplicated physical call rows."""

    if not states:
        raise ValueError("at least one state is required")
    state_ids = [state.state_id for state in states]
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("state_id must be unique")
    missing_strata = {state.stratum for state in states} - set(
        freeze.cost_matched_fixed_action_by_stratum
    )
    if missing_strata:
        raise ValueError(f"missing cost-match strata: {sorted(missing_strata)}")

    random_by_state, donor_by_state, cell_audit = matched_random_actions(
        states, seed=freeze.matched_random_seed
    )
    logical: list[dict[str, Any]] = []
    physical_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for state in sorted(states, key=lambda item: item.state_id):
        candidate_ids = {
            component: state.candidate_ids[component] for component in COMPONENTS
        }
        for policy in POLICIES:
            bits, fixed_before_projection = _policy_bits(
                state, policy, freeze=freeze, random_bits=random_by_state
            )
            action_id = compile_component_bits(bits)
            selected_candidate_ids = {
                component: candidate_ids[component]
                for component in COMPONENTS
                if bits[component]
            }
            action_material_sha256 = sha256(
                _canonical_json(
                    {
                        "state_id": state.state_id,
                        "candidate_snapshot_sha256": state.candidate_snapshot_sha256,
                        "step2_shared_surface_sha256": state.step2_shared_surface_sha256,
                        "action_id": action_id,
                        "selected_candidate_ids": selected_candidate_ids,
                    }
                ).encode("utf-8")
            ).hexdigest()
            estimated_tokens = sum(
                int(state.incremental_tokens[component])
                for component in COMPONENTS
                if bits[component]
            )
            for seed_label in freeze.seed_labels:
                physical_key = (state.state_id, action_id, seed_label)
                physical_call_id = "v53resp_" + _stable_hex(
                    freeze.protocol, *physical_key
                )
                row = {
                    "protocol": freeze.protocol,
                    "state_id": state.state_id,
                    "stratum": state.stratum,
                    "seed_label": seed_label,
                    "policy": policy,
                    "action_id": action_id,
                    "requested_bits": bits,
                    "candidate_snapshot_sha256": state.candidate_snapshot_sha256,
                    "step2_shared_surface_sha256": state.step2_shared_surface_sha256,
                    "selected_candidate_ids": selected_candidate_ids,
                    "action_material_sha256": action_material_sha256,
                    "estimated_incremental_tokens": estimated_tokens,
                    "physical_call_id": physical_call_id,
                    "cost_matched_action_before_projection": fixed_before_projection,
                    "matched_random_donor_state_id": (
                        donor_by_state[state.state_id]
                        if policy == "cost_and_on_rate_matched_random"
                        else None
                    ),
                    "response_or_evaluation_outcome_read": False,
                }
                logical.append(row)
                if physical_key not in physical_by_key:
                    physical_by_key[physical_key] = {
                        key: value
                        for key, value in row.items()
                        if key
                        not in {
                            "policy",
                            "cost_matched_action_before_projection",
                            "matched_random_donor_state_id",
                        }
                    }
                    physical_by_key[physical_key]["policy_aliases"] = []
                physical_by_key[physical_key]["policy_aliases"].append(policy)

    physical = sorted(
        physical_by_key.values(), key=lambda row: str(row["physical_call_id"])
    )
    for row in physical:
        row["policy_aliases"] = sorted(set(row["policy_aliases"]))

    logical_expected = len(states) * len(POLICIES) * len(freeze.seed_labels)
    if len(logical) != logical_expected:
        raise AssertionError("logical policy bindings are incomplete")
    for state in states:
        shared = {
            (
                row["candidate_snapshot_sha256"],
                row["step2_shared_surface_sha256"],
            )
            for row in logical
            if row["state_id"] == state.state_id
        }
        if shared != {
            (state.candidate_snapshot_sha256, state.step2_shared_surface_sha256)
        }:
            raise AssertionError("policy arm changed candidate or Step-2 shared surface")

    alias_groups = [row for row in physical if len(row["policy_aliases"]) > 1]
    random_changed = sum(item["changed_assignments"] for item in cell_audit)
    report = {
        "protocol": freeze.protocol,
        "status": "COMPLETE_ZERO_API_OUTCOME_BLIND_BASELINE_SCAFFOLD",
        "policies": list(POLICIES),
        "states": len(states),
        "seed_labels": list(freeze.seed_labels),
        "logical_policy_bindings": len(logical),
        "physical_calls_after_alias_deduplication": len(physical),
        "alias_groups": len(alias_groups),
        "matched_random": {
            "exchangeability_cells": len(cell_audit),
            "changed_state_assignments": random_changed,
            "has_nonalias_contrast": random_changed > 0,
            "all_cells_exact_on_rate_match": all(
                item["exact_on_rate_match"] for item in cell_audit
            ),
            "all_cells_exact_estimated_cost_match": all(
                item["exact_estimated_cost_match"] for item in cell_audit
            ),
            "cell_audit": cell_audit,
        },
        "fairness_attestations": {
            "six_logical_arms_retained_even_when_aliased": True,
            "same_candidate_snapshot_within_state": True,
            "same_step2_shared_surface_within_state": True,
            "only_policy_action_may_differ": True,
            "response_or_evaluation_outcome_read": False,
            "api_calls": 0,
        },
    }
    return {"logical_bindings": logical, "physical_calls": physical, "report": report}
