"""Frozen factorized policy runtime for the PM V1.5b system experiment.

The module deliberately does not retrieve resources, score responses, or read
evaluation outcomes.  It receives one outcome-blind observation per component
and compiles the requested component bits into the existing 16-action space.
This keeps policy differences separate from candidate discovery and Step-2
generator execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from .contracts import MemorySource, StrategyMode, canonical_action_id, parse_action_id


Component = Literal["MP", "MS", "ME", "RS"]
PolicyName = Literal[
    "always_off",
    "fixed_high_resource",
    "transparent_rule",
    "learned_pm_full",
    "learned_pm_conservative",
    "cost_matched_fixed",
]

COMPONENTS: tuple[Component, ...] = ("MP", "MS", "ME", "RS")
CONSERVATIVE_LEARNED_COMPONENTS = frozenset({"ME", "RS"})


@dataclass(frozen=True)
class ComponentOpportunity:
    """Outcome-blind state available before resource injection/generation."""

    component: Component
    candidate_present: bool
    hard_gate_pass: bool
    transparent_rule_on: bool
    learned_in_support: bool
    learned_on: bool
    learned_probability: float | None = None
    decision_reason: str = ""

    @property
    def eligible(self) -> bool:
        return bool(self.candidate_present and self.hard_gate_pass)


@dataclass(frozen=True)
class PolicyRoute:
    policy: PolicyName
    requested_bits: Mapping[Component, bool]
    action_id: str
    reasons: Mapping[Component, str]


@dataclass(frozen=True)
class ComponentFeasibility:
    """Outcome-blind constraints applied after four independent head outputs."""

    component: Component
    candidate_present: bool
    hard_gate_pass: bool
    opportunity_score: float
    incremental_tokens: int
    conflicts_with: frozenset[Component] = frozenset()

    def __post_init__(self) -> None:
        if self.component not in COMPONENTS:
            raise ValueError(f"unknown component: {self.component}")
        if not 0.0 <= self.opportunity_score <= 1.0:
            raise ValueError("opportunity_score must be in [0, 1]")
        if self.incremental_tokens < 0:
            raise ValueError("incremental_tokens must be nonnegative")
        unknown = set(self.conflicts_with) - set(COMPONENTS)
        if unknown or self.component in self.conflicts_with:
            raise ValueError("invalid joint conflict declaration")


@dataclass(frozen=True)
class JointFeasibilityResult:
    requested_action_id: str
    feasible_action_id: str
    requested_bits: Mapping[Component, bool]
    feasible_bits: Mapping[Component, bool]
    dropped_components: tuple[Component, ...]
    reasons: Mapping[Component, str]
    incremental_tokens: int


@dataclass(frozen=True)
class PolicyExecutionTrace:
    """Keep requested, jointly feasible, and actually injected actions distinct."""

    requested_action_id: str
    feasible_action_id: str
    realized_action_id: str

    def __post_init__(self) -> None:
        requested = action_component_bits(self.requested_action_id)
        feasible = action_component_bits(self.feasible_action_id)
        realized = action_component_bits(self.realized_action_id)
        for component in COMPONENTS:
            if feasible[component] and not requested[component]:
                raise ValueError("feasible action cannot add an unrequested component")
            if realized[component] and not feasible[component]:
                raise ValueError("realized action cannot add an infeasible component")


def compile_component_bits(bits: Mapping[str, bool]) -> str:
    """Mechanically map four component bits to one canonical legal action."""

    unknown = set(bits) - set(COMPONENTS)
    missing = set(COMPONENTS) - set(bits)
    if unknown or missing:
        raise ValueError(
            f"component bit schema mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    sources = {
        MemorySource(component)
        for component in ("MP", "MS", "ME")
        if bool(bits[component])
    }
    strategy = StrategyMode.RS if bool(bits["RS"]) else StrategyMode.R0
    return canonical_action_id(sources, strategy)


def action_component_bits(action_id: str) -> dict[Component, bool]:
    """Inverse mapping used by fixed baselines and mechanical audits."""

    sources, strategy = parse_action_id(action_id)
    return {
        "MP": MemorySource.MP in sources,
        "MS": MemorySource.MS in sources,
        "ME": MemorySource.ME in sources,
        "RS": strategy is StrategyMode.RS,
    }


def route_policy(
    *,
    policy: PolicyName,
    opportunities: Mapping[str, ComponentOpportunity],
    cost_matched_action_id: str | None = None,
) -> PolicyRoute:
    """Apply one frozen policy while preserving deterministic hard gates.

    ``fixed_high_resource`` means every *eligible* resource, not every catalog
    entry.  No policy may bypass owner/time/conflict/boundary/budget gates.
    ``learned_pm_full`` preserves all four frozen heads.  Only the explicitly
    separate conservative arm masks MP/MS.
    """

    if set(opportunities) != set(COMPONENTS):
        raise ValueError("exactly one opportunity observation per component is required")
    for component in COMPONENTS:
        if opportunities[component].component != component:
            raise ValueError(f"opportunity key/component mismatch for {component}")

    fixed_bits = (
        action_component_bits(cost_matched_action_id)
        if policy == "cost_matched_fixed" and cost_matched_action_id
        else None
    )
    if policy == "cost_matched_fixed" and fixed_bits is None:
        raise ValueError("cost_matched_fixed requires its internally frozen action id")
    bits: dict[Component, bool] = {}
    reasons: dict[Component, str] = {}
    for component in COMPONENTS:
        observation = opportunities[component]
        if not observation.eligible:
            bits[component] = False
            reasons[component] = "HARD_OFF_OR_NO_CANDIDATE"
            continue

        if policy == "always_off":
            on = False
            reason = "POLICY_ALWAYS_OFF"
        elif policy == "fixed_high_resource":
            on = True
            reason = "ELIGIBLE_FIXED_HIGH_ON"
        elif policy == "transparent_rule":
            on = bool(observation.transparent_rule_on)
            reason = "TRANSPARENT_RULE_ON" if on else "TRANSPARENT_RULE_OFF"
        elif policy == "learned_pm_full":
            on = bool(observation.learned_in_support and observation.learned_on)
            reason = (
                observation.decision_reason
                or ("LEARNED_HEAD_ON" if on else "LEARNED_HEAD_OFF_OR_OOD")
            )
        elif policy == "learned_pm_conservative":
            allowed = component in CONSERVATIVE_LEARNED_COMPONENTS
            on = bool(
                allowed and observation.learned_in_support and observation.learned_on
            )
            reason = (
                observation.decision_reason
                if allowed and observation.decision_reason
                else "CONSERVATIVE_MP_MS_MASKED"
                if not allowed
                else "LEARNED_HEAD_OFF_OR_OOD"
            )
        elif policy == "cost_matched_fixed":
            assert fixed_bits is not None
            on = bool(fixed_bits[component])
            reason = "COST_MATCHED_FIXED_ON" if on else "COST_MATCHED_FIXED_OFF"
        else:  # pragma: no cover - Literal plus defensive runtime guard.
            raise ValueError(f"unknown policy: {policy}")
        bits[component] = on
        reasons[component] = reason

    action_id = compile_component_bits(bits)
    return PolicyRoute(
        policy=policy,
        requested_bits=bits,
        action_id=action_id,
        reasons=reasons,
    )


def realize_guarded_action(
    *, requested_action_id: str, component_passed: Mapping[str, bool]
) -> str:
    """Project a requested action to components that survived Step-2 guards."""

    requested = action_component_bits(requested_action_id)
    unknown = set(component_passed) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown guarded component(s): {sorted(unknown)}")
    realized = {
        component: bool(requested[component] and component_passed.get(component, False))
        for component in COMPONENTS
    }
    return compile_component_bits(realized)


def project_joint_feasibility(
    *,
    requested_action_id: str,
    feasibility: Mapping[str, ComponentFeasibility],
    maximum_incremental_tokens: int,
) -> JointFeasibilityResult:
    """Project independent head requests into one jointly executable action.

    Components are first removed for missing candidates or failed hard gates.
    Explicit pairwise conflicts and the shared token budget are then resolved
    deterministically: lower opportunity loses; at equal opportunity the more
    expensive component loses; remaining ties use the frozen component order.
    This layer never reads response quality or outcomes.
    """

    if maximum_incremental_tokens < 0:
        raise ValueError("maximum_incremental_tokens must be nonnegative")
    if set(feasibility) != set(COMPONENTS):
        raise ValueError("joint feasibility requires all four components")
    for component in COMPONENTS:
        if feasibility[component].component != component:
            raise ValueError(f"feasibility key/component mismatch for {component}")

    requested = action_component_bits(requested_action_id)
    active = {component for component in COMPONENTS if requested[component]}
    reasons: dict[Component, str] = {
        component: "NOT_REQUESTED" for component in COMPONENTS
    }
    for component in tuple(active):
        row = feasibility[component]
        if not row.candidate_present:
            active.remove(component)
            reasons[component] = "DROPPED_CANDIDATE_ABSENT"
        elif not row.hard_gate_pass:
            active.remove(component)
            reasons[component] = "DROPPED_HARD_GATE"
        else:
            reasons[component] = "FEASIBLE"

    order = {component: index for index, component in enumerate(COMPONENTS)}

    def retention_key(component: Component) -> tuple[float, int, int]:
        row = feasibility[component]
        return (
            float(row.opportunity_score),
            -int(row.incremental_tokens),
            -order[component],
        )

    while True:
        conflict: tuple[Component, Component] | None = None
        for left in COMPONENTS:
            if left not in active:
                continue
            for right in COMPONENTS:
                if order[right] <= order[left] or right not in active:
                    continue
                if (
                    right in feasibility[left].conflicts_with
                    or left in feasibility[right].conflicts_with
                ):
                    conflict = (left, right)
                    break
            if conflict is not None:
                break
        if conflict is None:
            break
        loser = min(conflict, key=retention_key)
        active.remove(loser)
        reasons[loser] = "DROPPED_JOINT_CONFLICT"

    total_tokens = sum(feasibility[c].incremental_tokens for c in active)
    while total_tokens > maximum_incremental_tokens and active:
        loser = min(active, key=retention_key)
        active.remove(loser)
        reasons[loser] = "DROPPED_JOINT_TOKEN_BUDGET"
        total_tokens = sum(feasibility[c].incremental_tokens for c in active)

    feasible_bits: dict[Component, bool] = {
        component: component in active for component in COMPONENTS
    }
    dropped = tuple(
        component
        for component in COMPONENTS
        if requested[component] and not feasible_bits[component]
    )
    return JointFeasibilityResult(
        requested_action_id=requested_action_id,
        feasible_action_id=compile_component_bits(feasible_bits),
        requested_bits=requested,
        feasible_bits=feasible_bits,
        dropped_components=dropped,
        reasons=reasons,
        incremental_tokens=total_tokens,
    )
