from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt
from statistics import fmean
from typing import Any, Mapping, Sequence

from .io import canonical_json, sha256_text
from .pm_v2_contracts import PMV2Split, PMV2State


FACTORIZED_SIGNAL_AUDIT_PROTOCOL = (
    "pm-v1.5-train-only-factorized-weak-signal-audit-v1"
)
SOURCE_COMPONENTS = ("MP", "MS", "ME", "RS")


def action_components(action_id: str) -> frozenset[str]:
    """Return the source/strategy switches represented by an action."""

    try:
        memory, strategy = action_id.split("+", 1)
    except ValueError as exc:
        raise ValueError(f"invalid action id: {action_id!r}") from exc
    if strategy not in {"R0", "RS"}:
        raise ValueError(f"invalid strategy switch: {action_id!r}")
    memory_components = {
        "M0": frozenset(),
        "MP": frozenset({"MP"}),
        "MS": frozenset({"MS"}),
        "ME": frozenset({"ME"}),
        "MPMS": frozenset({"MP", "MS"}),
        "MPE": frozenset({"MP", "ME"}),
        "MSE": frozenset({"MS", "ME"}),
        "MPMSME": frozenset({"MP", "MS", "ME"}),
    }
    if memory not in memory_components:
        raise ValueError(f"invalid memory source set: {action_id!r}")
    result = set(memory_components[memory])
    if strategy == "RS":
        result.add("RS")
    return frozenset(result)


def response_quality(
    response: Mapping[str, Any], weights: Mapping[str, float]
) -> float:
    if not weights or abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
        raise ValueError("response quality weights must be non-empty and sum to one")
    values = []
    for field, weight in weights.items():
        score = float(response[field])
        if not 1.0 <= score <= 5.0:
            raise ValueError(f"response score is outside [1,5]: {field}")
        values.append(float(weight) * (score - 1.0) / 4.0)
    return float(sum(values))


def _standard_deviation(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = fmean(values)
    return float(sqrt(fmean([(value - mean) ** 2 for value in values])))


def _structural_target(
    context: Mapping[str, Any] | None, component: str
) -> str:
    if context is None:
        return "unavailable"
    if component in {"MP", "MS", "ME"}:
        if component in set(context.get("needed_memory_sources") or []):
            return "positive"
        if context.get("regime") == "memory_harmful":
            annotations = list(context.get("memory_annotations") or [])
            harmful = any(
                row.get("source") == component
                and (
                    row.get("item_utility") == "harmful"
                    or row.get("stale") is True
                    or row.get("conflicts_with_current_state") is True
                )
                for row in annotations
            )
            if harmful:
                return "negative"
        return "ambiguous"
    target = str(context.get("strategy_resource_target") or "ambiguous")
    return {
        "helpful": "positive",
        "use": "positive",
        "harmful": "negative",
        "skip": "negative",
    }.get(target, "ambiguous")


def _consensus(effects: Mapping[str, float], margin: float) -> str:
    values = list(effects.values())
    if all(value > margin for value in values):
        return "positive"
    if all(value < -margin for value in values):
        return "negative"
    if min(values) < -margin and max(values) > margin:
        return "opposite"
    return "uncertain"


def build_factorized_signal_report(
    *,
    states: Sequence[PMV2State],
    weak_labels: Sequence[Mapping[str, Any]],
    raw_judge_rows: Sequence[Mapping[str, Any]],
    response_weights: Mapping[str, float],
    expected_judge_families: Sequence[str],
    evaluator_contexts: Mapping[str, Mapping[str, Any]] | None = None,
    effect_margin: float = 0.01,
) -> dict[str, Any]:
    """Audit within-state component effects without making 16-way pseudo-gold.

    Every component effect is first computed separately for each judge family
    and each matched factorial background, then averaged within
    ``(state, component, family)``.  Factorial backgrounds and judge families
    therefore remain repeated measurements rather than independent examples.
    """

    if effect_margin <= 0.0:
        raise ValueError("effect margin must be positive")
    if any(state.split is not PMV2Split.TRAIN for state in states):
        raise RuntimeError("factorized weak-signal audit is train-only")
    state_by_id = {state.state_id: state for state in states}
    if len(state_by_id) != len(states) or not states:
        raise ValueError("states must be unique and non-empty")
    families = tuple(sorted(str(value) for value in expected_judge_families))
    if len(families) < 2:
        raise ValueError("factorized audit requires at least two judge families")

    label_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in weak_labels:
        key = (str(row["state_id"]), str(row["action_id"]))
        if key in label_by_key:
            raise RuntimeError(f"duplicate weak label: {key}")
        label_by_key[key] = row
    expected_label_keys = {
        (state.state_id, action_id)
        for state in states
        for action_id in state.allowed_actions
    }
    if set(label_by_key) != expected_label_keys:
        raise RuntimeError("weak labels do not exactly cover train state-actions")

    raw_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in raw_judge_rows:
        if row.get("status") != "SUCCESS" or row.get("schema_success") is not True:
            continue
        state_id = str(row["state_id"])
        if state_id not in state_by_id:
            continue
        key = (state_id, str(row["action_id"]), str(row["judge_family"]))
        if key in raw_by_key:
            raise RuntimeError(f"duplicate raw judge row: {key}")
        raw_by_key[key] = row

    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    background_pair_count = 0
    alias_pair_count = 0
    incomplete_family_pair_count = 0
    for state in sorted(states, key=lambda row: row.state_id):
        for baseline in state.allowed_actions:
            baseline_components = action_components(baseline)
            for treatment in state.allowed_actions:
                treatment_components = action_components(treatment)
                added = treatment_components - baseline_components
                if len(added) != 1 or baseline_components - treatment_components:
                    continue
                component = next(iter(added))
                baseline_label = label_by_key[(state.state_id, baseline)]
                treatment_label = label_by_key[(state.state_id, treatment)]
                baseline_equivalence = (
                    baseline_label.get("provenance") or {}
                ).get("prompt_equivalence_id")
                treatment_equivalence = (
                    treatment_label.get("provenance") or {}
                ).get("prompt_equivalence_id")
                if (
                    baseline_equivalence
                    and baseline_equivalence == treatment_equivalence
                ):
                    alias_pair_count += 1
                    continue
                pair_complete = True
                pair_values: dict[str, float] = {}
                for family in families:
                    baseline_raw = raw_by_key.get(
                        (state.state_id, baseline, family)
                    )
                    treatment_raw = raw_by_key.get(
                        (state.state_id, treatment, family)
                    )
                    if baseline_raw is None or treatment_raw is None:
                        pair_complete = False
                        break
                    pair_values[family] = response_quality(
                        treatment_raw["response"], response_weights
                    ) - response_quality(
                        baseline_raw["response"], response_weights
                    )
                if not pair_complete:
                    incomplete_family_pair_count += 1
                    continue
                background_pair_count += 1
                for family, value in pair_values.items():
                    values[(state.state_id, component, family)].append(value)

    effect_rows: list[dict[str, Any]] = []
    incomplete_state_components = 0
    for state in sorted(states, key=lambda row: row.state_id):
        for component in SOURCE_COMPONENTS:
            family_values = {
                family: values.get((state.state_id, component, family), [])
                for family in families
            }
            if not all(family_values.values()):
                incomplete_state_components += 1
                continue
            effects = {
                family: float(fmean(rows))
                for family, rows in family_values.items()
            }
            context_dispersion = {
                family: _standard_deviation(rows)
                for family, rows in family_values.items()
            }
            target = _structural_target(
                (evaluator_contexts or {}).get(state.state_id), component
            )
            if target == "helpful":
                target = "positive"
            elif target == "harmful":
                target = "negative"
            effect_rows.append(
                {
                    "state_id": state.state_id,
                    "user_id": state.user_id,
                    "component": component,
                    "family_effects": effects,
                    "family_effect_range": float(
                        max(effects.values()) - min(effects.values())
                    ),
                    "background_effect_standard_deviation": context_dispersion,
                    "factorial_background_count": len(
                        next(iter(family_values.values()))
                    ),
                    "consensus": _consensus(effects, effect_margin),
                    "structural_target": target,
                }
            )

    by_component: dict[str, Any] = {}
    for component in SOURCE_COMPONENTS:
        rows = [row for row in effect_rows if row["component"] == component]
        counts = Counter(row["consensus"] for row in rows)
        targeted = [
            row
            for row in rows
            if row["structural_target"] in {"positive", "negative"}
        ]
        aligned = sum(
            row["consensus"] == row["structural_target"] for row in targeted
        )
        by_component[component] = {
            "state_component_rows": len(rows),
            "consensus_counts": dict(sorted(counts.items())),
            "consensus_direction_rate": float(
                (counts["positive"] + counts["negative"]) / len(rows)
            )
            if rows
            else 0.0,
            "opposite_family_direction_rate": float(
                counts["opposite"] / len(rows)
            )
            if rows
            else 0.0,
            "structurally_targeted_rows": len(targeted),
            "structural_target_alignment_rate": float(aligned / len(targeted))
            if targeted
            else None,
            "mean_family_effect": {
                family: float(fmean(row["family_effects"][family] for row in rows))
                if rows
                else None
                for family in families
            },
        }

    report = {
        "protocol": FACTORIZED_SIGNAL_AUDIT_PROTOCOL,
        "data_role": "train_only_diagnostic_not_training_labels",
        "automatic_gold_claimed": False,
        "state_count": len(states),
        "weak_label_count": len(weak_labels),
        "judge_families": list(families),
        "effect_margin": float(effect_margin),
        "component_effect_definition": (
            "within-state mean quality difference over every matched factorial "
            "background, retained separately by judge family"
        ),
        "background_pair_count": background_pair_count,
        "prompt_equivalent_pairs_excluded": alias_pair_count,
        "incomplete_family_pairs_excluded": incomplete_family_pair_count,
        "incomplete_state_components": incomplete_state_components,
        "state_component_rows": len(effect_rows),
        "by_component": by_component,
        "effect_rows": effect_rows,
        "interpretation_boundary": (
            "Consensus measures agreement, not accuracy. Structural targets are "
            "synthetic design annotations, not response-quality gold labels."
        ),
    }
    report["report_sha256"] = sha256_text(canonical_json(report))
    return report


def select_human_anchor_rows(
    effect_rows: Sequence[Mapping[str, Any]],
    *,
    per_component_per_stratum: int = 2,
    seed: int = 4311,
) -> list[dict[str, Any]]:
    """Select a deterministic, state-disjoint train-only calibration packet."""

    if per_component_per_stratum < 1:
        raise ValueError("per-component stratum count must be positive")
    strata = ("positive", "negative", "opposite", "uncertain")
    used_states: set[str] = set()
    selected: list[dict[str, Any]] = []
    for component in SOURCE_COMPONENTS:
        for stratum in strata:
            candidates = [
                dict(row)
                for row in effect_rows
                if row.get("component") == component
                and row.get("consensus") == stratum
            ]
            candidates.sort(
                key=lambda row: sha256_text(
                    f"{seed}|{component}|{stratum}|{row['state_id']}"
                )
            )
            chosen = [
                row
                for row in candidates
                if str(row["state_id"]) not in used_states
            ][:per_component_per_stratum]
            if len(chosen) != per_component_per_stratum:
                raise RuntimeError(
                    f"insufficient state-disjoint anchors for {component}/{stratum}"
                )
            for row in chosen:
                used_states.add(str(row["state_id"]))
                selected.append(row)
    return selected
