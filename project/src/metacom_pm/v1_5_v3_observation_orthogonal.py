"""Private blueprint for the V3 Observation factor-orthogonal dataset.

This module deliberately does not reuse the legacy eligibility blueprint's
``reason -> requirement mutation`` path.  Every observable factor is written
directly into the private construction plan.  The plan is not human gold and
is never exposed to the model or reviewer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations, product
from math import sqrt
from typing import Any, Iterable

from .io import stable_hex


PROTOCOL = "pm-v1.5-v3-observation-factor-orthogonal-blueprint-v1"
COMPONENTS = ("MP", "MS", "ME", "RS")
FACTORS = (
    "owner_time_entity_valid",
    "goal_function_fit",
    "boundary_burden_compatible",
    "specific_increment",
)
RS_DETERMINISTIC_FACTORS = ("owner_time_entity_valid",)
LEARNED_FACTORS = {
    component: tuple(
        factor
        for factor in FACTORS
        if not (component == "RS" and factor in RS_DETERMINISTIC_FACTORS)
    )
    for component in COMPONENTS
}
TOPICS = (
    "workload_deadline",
    "relationship_change",
    "family_care",
    "sleep_schedule",
    "social_disconnection",
    "study_pressure",
    "creative_block",
    "relocation_adjustment",
)
HISTORY_SCALES = ("SMALL", "MEDIUM", "LARGE", "EVOEMO_LIKE_LARGE")
SOURCE_CATALOG_TARGETS = {
    "MP": {"SMALL": 2, "MEDIUM": 4, "LARGE": 7, "EVOEMO_LIKE_LARGE": 10},
    "MS": {"SMALL": 4, "MEDIUM": 10, "LARGE": 22, "EVOEMO_LIKE_LARGE": 34},
    "ME": {"SMALL": 5, "MEDIUM": 16, "LARGE": 40, "EVOEMO_LIKE_LARGE": 68},
    "RS": {"SMALL": 80, "MEDIUM": 80, "LARGE": 80, "EVOEMO_LIKE_LARGE": 80},
}
PREFIX_FAMILIES = tuple(f"prefix_{index}" for index in range(8))
LENGTH_DIRECTIONS = ("NEGATIVE_LONGER", "POSITIVE_LONGER")


COMPONENT_SURFACES: dict[str, dict[str, str]] = {
    "MP": {
        "candidate_subtype": "MP_PROFILE_OR_PREFERENCE",
        "candidate_function": "apply_stable_preference_or_practical_constraint",
        "positive_owner": "same user's stable profile or preference; still current",
        "negative_owner": "different person's profile or an explicitly superseded preference",
        "positive_goal": "profile detail changes the requested response form or feasibility",
        "negative_goal": "true profile detail is unrelated to the current response goal",
        "positive_boundary": "profile use respects the current turn's burden and explicit boundary",
        "negative_boundary": "profile use conflicts with the current turn's explicit format or burden boundary",
        "positive_increment": "specific profile detail is absent from the visible current dialogue",
        "negative_increment": "same detail is already stated in the visible current dialogue",
    },
    "MS": {
        "candidate_subtype": "MS_SESSION",
        "candidate_function": "carry_a_specific_prior_session_goal_or_distinction",
        "positive_owner": "strictly prior record belongs to the current user and remains time-valid",
        "negative_owner": "record belongs to a friend or is explicitly resolved and obsolete",
        "positive_goal": "specific prior goal or distinction directly serves the current request",
        "negative_goal": "same-topic prior material does not perform the requested function",
        "positive_boundary": "tentative continuity is allowed under the current turn boundary",
        "negative_boundary": "current turn explicitly forbids revisiting or inferring from the old session",
        "positive_increment": "record contains a concrete distinction, constraint, or unfinished thread not visible now",
        "negative_increment": "record only says the topic was discussed or repeats the current message",
    },
    "ME": {
        "candidate_subtype": "ME_REUSABLE_OUTCOME",
        "candidate_function": "reuse_a_past_action_or_choice_with_result_or_mechanism",
        "positive_owner": "past event and outcome belong to the current user and are still applicable as evidence",
        "negative_owner": "outcome belongs to another person or was explicitly superseded",
        "positive_goal": "current request is action-oriented and the past result can serve it tentatively",
        "negative_goal": "current request asks only for listening, naming, or factual recall rather than action reuse",
        "positive_boundary": "one tentative reuse is allowed by the current turn boundary",
        "negative_boundary": "current turn explicitly forbids offering or reusing an action",
        "positive_increment": "candidate contains a concrete past action or choice plus result or mechanism not visible now",
        "negative_increment": "candidate is background, unresolved status, or a result already stated now",
    },
    "RS": {
        "candidate_subtype": "RS_ATOMIC_MOVE",
        "candidate_function": "apply_one_topic_agnostic_support_move",
        "positive_owner": "not applicable: frozen shared bank card has no private owner or temporal claim",
        "negative_owner": "not constructed: RS owner/time is deterministic true",
        "positive_goal": "atomic move performs the current requested support function",
        "negative_goal": "card is safe in general but performs a different support function",
        "positive_boundary": "one atomic move fits the current turn's burden and explicit boundary",
        "negative_boundary": "card would violate listen-only, no-advice, no-repeat, or one-point constraints",
        "positive_increment": "the move has not already been fully specified or executed in the visible dialogue",
        "negative_increment": "the same move was just executed or is already fully specified by the user",
    },
}

RS_FUNCTIONS = (
    ("focused_question", "Question"),
    ("tentative_paraphrase_check", "Restatement or Paraphrasing"),
    ("evidence_grounded_reflection", "Reflection of feelings"),
    ("one_reversible_suggestion", "Providing Suggestions"),
)


def _phi(left: Iterable[bool], right: Iterable[bool]) -> float:
    pairs = list(zip(left, right, strict=True))
    n11 = sum(a and b for a, b in pairs)
    n10 = sum(a and not b for a, b in pairs)
    n01 = sum(not a and b for a, b in pairs)
    n00 = sum(not a and not b for a, b in pairs)
    denominator = sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return 0.0 if denominator == 0 else (n11 * n00 - n10 * n01) / denominator


def _factorial_vectors(component: str) -> list[tuple[bool, bool, bool, bool]]:
    if component != "RS":
        return [tuple(bits) for bits in product((False, True), repeat=4)]  # type: ignore[list-item]
    rows: list[tuple[bool, bool, bool, bool]] = []
    for repetition in (False, True):
        del repetition  # repetition is represented by row position, not a semantic factor
        for goal, boundary, increment in product((False, True), repeat=3):
            rows.append((True, goal, boundary, increment))
    return rows


def _nuisance_assignment(component: str, index: int) -> dict[str, str]:
    """Assign nuisances so every learned bit is balanced within each category.

    MP/MS/ME use the four semantic bits as a full factorial.  RS uses the three
    learned bits plus the repetition bit as a four-dimensional design variable.
    Complement pairs define topic/prefix, two parity checks define scale, and
    total parity defines opposite length directions.
    """

    if component == "RS":
        repetition = index // 8
        local = index % 8
        goal = (local >> 2) & 1
        boundary = (local >> 1) & 1
        increment = local & 1
        design_bits = (goal, boundary, increment, repetition)
        complement_index = index ^ 0b1111
    else:
        design_bits = tuple((index >> shift) & 1 for shift in (3, 2, 1, 0))
        complement_index = index ^ 0b1111
    topic_index = min(index, complement_index)
    scale_index = ((design_bits[0] ^ design_bits[1]) << 1) | (design_bits[2] ^ design_bits[3])
    prefix_index = (topic_index * 5) % len(PREFIX_FAMILIES)
    length_direction = LENGTH_DIRECTIONS[sum(design_bits) % 2]
    return {
        "topic_family": TOPICS[topic_index],
        "history_scale": HISTORY_SCALES[scale_index],
        "prefix_family": PREFIX_FAMILIES[prefix_index],
        "length_direction": length_direction,
    }


def _factor_plan(component: str, values: dict[str, bool]) -> dict[str, dict[str, Any]]:
    surface = COMPONENT_SURFACES[component]
    return {
        factor: {
            "private_target": value,
            "construction_instruction": surface[
                {
                    "owner_time_entity_valid": "positive_owner" if value else "negative_owner",
                    "goal_function_fit": "positive_goal" if value else "negative_goal",
                    "boundary_burden_compatible": "positive_boundary" if value else "negative_boundary",
                    "specific_increment": "positive_increment" if value else "negative_increment",
                }[factor]
            ],
            "is_learning_target": factor in LEARNED_FACTORS[component],
        }
        for factor, value in values.items()
    }


def _semantic_surface_families(
    *, component: str, track: str, ordinal: int, values: dict[str, bool]
) -> dict[str, str]:
    if component == "RS":
        design = [
            int(values["goal_function_fit"]),
            int(values["boundary_burden_compatible"]),
            int(values["specific_increment"]),
            ordinal // 8,
        ]
        positions = {
            "goal_function_fit": 0,
            "boundary_burden_compatible": 1,
            "specific_increment": 2,
        }
    else:
        design = [int(values[factor]) for factor in FACTORS]
        positions = {factor: index for index, factor in enumerate(FACTORS)}
    result: dict[str, str] = {}
    for factor in LEARNED_FACTORS[component]:
        if track == "ELIGIBILITY_CONFIRMATION":
            variant = 4 + (ordinal % 4)
        else:
            other = [bit for index, bit in enumerate(design) if index != positions[factor]]
            variant = (other[0] << 1) | other[1]
        result[factor] = f"semantic_surface_{variant}"
    return result


def _row(
    *,
    track: str,
    component: str,
    ordinal: int,
    values: dict[str, bool],
    nuisance: dict[str, str],
    fault_pattern: tuple[str, ...],
) -> dict[str, Any]:
    row_id = f"obsbp_{stable_hex(PROTOCOL, track, component, ordinal, n=24)}"
    surface_variant = f"surface_{track.lower()}_{component.lower()}_{ordinal:02d}"
    counterfactual_group_id = (
        f"obscf_{stable_hex(PROTOCOL, component, nuisance['topic_family'], n=20)}"
        if track == "FACTOR_FIT"
        else f"obsconfirm_{stable_hex(PROTOCOL, component, ordinal, n=20)}"
    )
    signature_payload = (
        component,
        tuple(values.items()),
        tuple(nuisance.items()),
        surface_variant,
    )
    eligibility = all(values.values())
    design_bits = tuple(int(values[factor]) for factor in FACTORS)
    if component == "MP":
        subtype = "MP_PREFERENCE" if sum(design_bits) % 2 else "MP_PROFILE"
        function = (
            "change_response_form_or_burden"
            if subtype == "MP_PREFERENCE"
            else "apply_practical_constraint"
        )
        strategy_family = None
    elif component == "RS":
        family_index = ((design_bits[1] ^ design_bits[2]) << 1) | (
            design_bits[3] ^ (ordinal // 8)
        )
        function, strategy_family = RS_FUNCTIONS[family_index]
        subtype = "RS_ATOMIC_MOVE"
    else:
        subtype = COMPONENT_SURFACES[component]["candidate_subtype"]
        function = COMPONENT_SURFACES[component]["candidate_function"]
        strategy_family = None
    return {
        "protocol": PROTOCOL,
        "blueprint_row_id": row_id,
        "user_id": f"user_{row_id}",
        "group_id": f"group_{row_id}",
        "counterfactual_group_id": counterfactual_group_id,
        "track": track,
        "target_component": component,
        "surface_variant": surface_variant,
        "decision_surface_signature": stable_hex(PROTOCOL, "surface", signature_payload, n=32),
        **nuisance,
        "source_catalog_size_target": SOURCE_CATALOG_TARGETS[component][
            nuisance["history_scale"]
        ],
        "candidate_subtype_target": subtype,
        "required_candidate_function": function,
        "required_strategy_family": strategy_family,
        "private_factor_plan": _factor_plan(component, values),
        "private_semantic_surface_family": _semantic_surface_families(
            component=component,
            track=track,
            ordinal=ordinal,
            values=values,
        ),
        "private_factorial_repetition": ordinal // 8 if component == "RS" else 0,
        "private_fault_pattern": list(fault_pattern),
        "private_composed_eligibility": eligibility,
        "construction_intent_is_model_input": False,
        "construction_intent_is_reviewer_input": False,
        "construction_intent_is_gold": False,
        "actual_rank1_candidate": None,
        "human_factor_labels": None,
        "human_composed_eligibility": None,
        "response_generated": False,
        "step1_worth_opening_gold": None,
        "external_text_or_outcome_read": False,
    }


def build_factor_fit_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for component in COMPONENTS:
        for ordinal, vector in enumerate(_factorial_vectors(component)):
            values = dict(zip(FACTORS, vector, strict=True))
            faults = tuple(factor for factor in LEARNED_FACTORS[component] if not values[factor])
            rows.append(
                _row(
                    track="FACTOR_FIT",
                    component=component,
                    ordinal=ordinal,
                    values=values,
                    nuisance=_nuisance_assignment(component, ordinal),
                    fault_pattern=faults,
                )
            )
    return rows


def _confirmation_faults(component: str) -> tuple[tuple[str, ...], ...]:
    learned = LEARNED_FACTORS[component]
    if component == "RS":
        return (
            (learned[0],),
            (learned[1],),
            (learned[2],),
            (learned[0], learned[1]),
        )
    rotation = COMPONENTS.index(component)
    return (
        (learned[rotation % 4],),
        (learned[(rotation + 1) % 4],),
        (learned[(rotation + 2) % 4], learned[(rotation + 3) % 4]),
        (learned[(rotation + 3) % 4], learned[rotation % 4]),
    )


def build_confirmation_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for component in COMPONENTS:
        patterns: tuple[tuple[str, ...], ...] = ((), (), (), ()) + _confirmation_faults(component)
        for ordinal, faults in enumerate(patterns):
            values = {factor: True for factor in FACTORS}
            for factor in faults:
                values[factor] = False
            nuisance = {
                "topic_family": TOPICS[(ordinal + 2 * COMPONENTS.index(component)) % len(TOPICS)],
                "history_scale": HISTORY_SCALES[(ordinal + COMPONENTS.index(component)) % 4],
                "prefix_family": PREFIX_FAMILIES[(ordinal * 3 + COMPONENTS.index(component)) % 8],
                "length_direction": LENGTH_DIRECTIONS[(ordinal + COMPONENTS.index(component)) % 2],
            }
            rows.append(
                _row(
                    track="ELIGIBILITY_CONFIRMATION",
                    component=component,
                    ordinal=ordinal,
                    values=values,
                    nuisance=nuisance,
                    fault_pattern=faults,
                )
            )
    return rows


def build_blueprint() -> list[dict[str, Any]]:
    return build_factor_fit_rows() + build_confirmation_rows()


def audit_blueprint(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    ids = [row["blueprint_row_id"] for row in rows]
    users = [row["user_id"] for row in rows]
    signatures = [row["decision_surface_signature"] for row in rows]
    if len(rows) != 96:
        failures.append(f"row_count_{len(rows)}_not_96")
    if len(set(ids)) != len(ids):
        failures.append("duplicate_blueprint_row_id")
    if len(set(users)) != len(users):
        failures.append("duplicate_user_id")
    if len(set(signatures)) != len(signatures):
        failures.append("duplicate_decision_surface")
    if any(row["construction_intent_is_model_input"] for row in rows):
        failures.append("private_intent_exposed_to_model")
    if any(row["construction_intent_is_reviewer_input"] for row in rows):
        failures.append("private_intent_exposed_to_reviewer")
    if any(row["construction_intent_is_gold"] for row in rows):
        failures.append("construction_intent_marked_gold")
    if any(row["external_text_or_outcome_read"] for row in rows):
        failures.append("external_text_or_outcome_read")

    fit = [row for row in rows if row["track"] == "FACTOR_FIT"]
    confirmation = [row for row in rows if row["track"] == "ELIGIBILITY_CONFIRMATION"]
    if len(fit) != 64:
        failures.append(f"factor_fit_count_{len(fit)}_not_64")
    if len(confirmation) != 32:
        failures.append(f"confirmation_count_{len(confirmation)}_not_32")

    cell_counts: dict[str, Any] = {}
    phi_values: dict[str, dict[str, float]] = {}
    nuisance_balance: dict[str, Any] = {}
    semantic_surface_balance: dict[str, Any] = {}
    max_abs_phi = 0.0
    for component in COMPONENTS:
        subset = [row for row in fit if row["target_component"] == component]
        if len(subset) != 16:
            failures.append(f"factor_fit_{component}_count_{len(subset)}_not_16")
        cell_counts[component] = {}
        for factor in FACTORS:
            values = [row["private_factor_plan"][factor]["private_target"] for row in subset]
            counts = Counter(values)
            learned = factor in LEARNED_FACTORS[component]
            cell_counts[component][factor] = {
                "learned": learned,
                "positive": counts[True],
                "negative": counts[False],
            }
            if learned and counts != Counter({False: 8, True: 8}):
                failures.append(f"unbalanced_cell_{component}_{factor}_{dict(counts)}")
            if not learned and (component != "RS" or counts != Counter({True: 16})):
                failures.append(f"invalid_deterministic_cell_{component}_{factor}_{dict(counts)}")

        phi_values[component] = {}
        for left, right in combinations(LEARNED_FACTORS[component], 2):
            value = _phi(
                [row["private_factor_plan"][left]["private_target"] for row in subset],
                [row["private_factor_plan"][right]["private_target"] for row in subset],
            )
            phi_values[component][f"{left}__{right}"] = value
            max_abs_phi = max(max_abs_phi, abs(value))
            if abs(value) > 0.05:
                failures.append(f"phi_too_high_{component}_{left}_{right}_{value}")

        nuisance_balance[component] = {}
        semantic_surface_balance[component] = {}
        for factor in LEARNED_FACTORS[component]:
            by_label = {
                label: Counter(
                    row["private_semantic_surface_family"][factor]
                    for row in subset
                    if row["private_factor_plan"][factor]["private_target"] is label
                )
                for label in (False, True)
            }
            semantic_surface_balance[component][factor] = {
                str(label): dict(counts) for label, counts in by_label.items()
            }
            expected = Counter({f"semantic_surface_{index}": 2 for index in range(4)})
            if any(counts != expected for counts in by_label.values()):
                failures.append(f"semantic_surface_unbalanced_{component}_{factor}")
        for nuisance in (
            "topic_family",
            "history_scale",
            "prefix_family",
            "length_direction",
            "candidate_subtype_target",
            "required_candidate_function",
        ):
            nuisance_balance[component][nuisance] = {}
            for factor in LEARNED_FACTORS[component]:
                cross = defaultdict(Counter)
                for row in subset:
                    cross[row[nuisance]][row["private_factor_plan"][factor]["private_target"]] += 1
                nuisance_balance[component][nuisance][factor] = {
                    str(level): {str(label): count for label, count in counts.items()}
                    for level, counts in cross.items()
                }
                if any(counts[True] != counts[False] for counts in cross.values()):
                    failures.append(f"nuisance_unbalanced_{component}_{nuisance}_{factor}")

    confirmation_balance: dict[str, dict[str, int]] = {}
    for component in COMPONENTS:
        subset = [row for row in confirmation if row["target_component"] == component]
        counts = Counter(row["private_composed_eligibility"] for row in subset)
        confirmation_balance[component] = {
            "eligible": counts[True],
            "ineligible": counts[False],
        }
        if counts != Counter({True: 4, False: 4}):
            failures.append(f"confirmation_not_4_4_{component}_{dict(counts)}")
        failed_factors = Counter(
            factor
            for row in subset
            if not row["private_composed_eligibility"]
            for factor in row["private_fault_pattern"]
        )
        if any(failed_factors[factor] == 0 for factor in LEARNED_FACTORS[component]):
            failures.append(f"confirmation_missing_fault_{component}")
        for factor in LEARNED_FACTORS[component]:
            fit_families = {
                row["private_semantic_surface_family"][factor]
                for row in fit
                if row["target_component"] == component
            }
            confirmation_families = {
                row["private_semantic_surface_family"][factor] for row in subset
            }
            if fit_families & confirmation_families:
                failures.append(f"confirmation_surface_seen_in_fit_{component}_{factor}")

    return {
        "protocol": PROTOCOL,
        "status": "PASS" if not failures else "FAIL",
        "scope": "PRIVATE_BLUEPRINT_STATIC_STRUCTURE_ONLY",
        "rows": len(rows),
        "factor_fit_rows": len(fit),
        "eligibility_confirmation_rows": len(confirmation),
        "unique_row_ids": len(set(ids)),
        "unique_users": len(set(users)),
        "unique_decision_surfaces": len(set(signatures)),
        "factor_cell_counts": cell_counts,
        "pairwise_phi": phi_values,
        "pairwise_phi_absolute_max": max_abs_phi,
        "nuisance_balance": nuisance_balance,
        "semantic_surface_balance": semantic_surface_balance,
        "confirmation_balance": confirmation_balance,
        "failures": failures,
        "api_calls": 0,
        "human_labels_read": 0,
        "external_lockbox_read": False,
        "limitations": [
            "This pass does not prove that realized text preserves the private factor plan.",
            "Text length and model-based nuisance probes require visible-state realization.",
            "Construction intent is a QA target only; human review remains the label source.",
        ],
        "next_step_if_pass": "REALIZE_96_VISIBLE_STATES_THEN_RUN_ALL_PRE_REVIEW_STATIC_GATES",
    }
