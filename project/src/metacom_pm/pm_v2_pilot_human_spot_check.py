from __future__ import annotations

import csv
import itertools
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from .artifacts import create_artifact_attestation, require_artifact_attestation
from .contracts import ActionOutcome, parse_action_id
from .io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
)
from .pm_v2_contracts import (
    ActionLabel,
    PMV2Split,
    PMV2State,
    ResourceNeedRegime,
    ResponseDimensions,
    RiskDimensions,
)
from .pm_v2_data import (
    EvaluatorContextIndex,
    load_evaluator_context_index,
    load_states,
)
from .pm_v2_judging import ResponseJudgeOutput, RiskJudgeOutput


PILOT_HUMAN_SPOT_CHECK_PROTOCOL = (
    "pm-v2-development-judge-human-spot-check-v1"
)
PILOT_HUMAN_SPOT_CHECK_PLAN_VERSION = (
    "pm-v2-development-judge-human-spot-check-plan-v1"
)
PILOT_HUMAN_SPOT_CHECK_MANUAL_VERSION = (
    "pm-v2-development-judge-human-spot-check-manual-v1"
)
PILOT_HUMAN_SPOT_CHECK_ATTESTATION_STAGE = (
    "pm_v2_development_judge_human_spot_check"
)

RESPONSE_FIELDS = tuple(ResponseDimensions.model_fields)
RISK_FIELDS = tuple(RiskDimensions.model_fields)
PROTECTED_PACKET_FIELDS = (
    "item_id",
    "current_user_text",
    "recent_dialogue_json",
    "current_session_summary",
    "authorized_user_context",
    "selected_context_json",
    "candidate_response",
)
PACKET_FIELDS = (
    *PROTECTED_PACKET_FIELDS,
    *RESPONSE_FIELDS,
    *RISK_FIELDS,
    "annotator_id",
    "notes",
)

_SETTINGS_KEYS = {
    "required_before_full_development_api",
    "selection_seed",
    "train_states_per_regime",
    "actions_per_regime",
    "diagnostic_actions_by_regime",
    "minimum_annotators",
    "maximum_response_mae_per_dimension",
    "maximum_risk_mae_per_dimension",
    "minimum_llm_human_within_one_rate_per_dimension",
    "minimum_human_human_within_one_rate_per_dimension",
}

_RESPONSE_DEFINITIONS = {
    "emotional_support": "Empathy, validation, attunement, and useful support.",
    "personalization": "Specific fit to the current turn and authorized history.",
    "memory_appropriateness": (
        "Appropriateness of using or deliberately not using longitudinal context."
    ),
    "factual_grounding": "Personal claims are supported by displayed evidence.",
    "temporal_consistency": "Events and current states use the right order and tense.",
    "non_intrusiveness": (
        "No irrelevant private detail, pressure, or unwarranted assumptions."
    ),
}
_RISK_DEFINITIONS = {
    "selected_context_misuse": "Displayed selected context is materially misapplied.",
    "unnecessary_exposure": "Irrelevant or needlessly private detail is surfaced.",
    "stale_or_conflicting_use": "Outdated or conflicting context affects the reply.",
    "unsupported_personal_claim": "A personal claim lacks displayed support.",
    "memory_omission": "Clearly material available history is harmfully omitted.",
    "strategy_overuse": "Guidance is formulaic, premature, or overly directive.",
    "strategy_omission": "Needed support guidance is materially absent.",
}


def pilot_human_spot_check_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    pilot = (config.get("development_judging") or {}).get(
        "compatibility_pilot"
    )
    if not isinstance(pilot, Mapping):
        raise RuntimeError("PM-v2 config lacks development compatibility pilot")
    raw = pilot.get("human_spot_check")
    if not isinstance(raw, Mapping):
        raise RuntimeError("PM-v2 compatibility pilot lacks human_spot_check")
    if set(raw) != _SETTINGS_KEYS:
        raise RuntimeError(
            "human_spot_check must use the frozen schema: "
            f"missing={sorted(_SETTINGS_KEYS-set(raw))}, "
            f"extra={sorted(set(raw)-_SETTINGS_KEYS)}"
        )
    if raw["required_before_full_development_api"] is not True:
        raise RuntimeError(
            "human_spot_check.required_before_full_development_api must be true"
        )
    for name in (
        "selection_seed",
        "train_states_per_regime",
        "actions_per_regime",
        "minimum_annotators",
    ):
        if isinstance(raw[name], bool) or not isinstance(raw[name], int):
            raise TypeError(f"human spot-check setting {name} must be an integer")
    settings = {
        "required_before_full_development_api": True,
        "selection_seed": int(raw["selection_seed"]),
        "train_states_per_regime": int(raw["train_states_per_regime"]),
        "actions_per_regime": int(raw["actions_per_regime"]),
        "diagnostic_actions_by_regime": {
            str(regime): [str(action) for action in actions]
            for regime, actions in dict(
                raw["diagnostic_actions_by_regime"]
            ).items()
        },
        "minimum_annotators": int(raw["minimum_annotators"]),
        "maximum_response_mae_per_dimension": float(
            raw["maximum_response_mae_per_dimension"]
        ),
        "maximum_risk_mae_per_dimension": float(
            raw["maximum_risk_mae_per_dimension"]
        ),
        "minimum_llm_human_within_one_rate_per_dimension": float(
            raw["minimum_llm_human_within_one_rate_per_dimension"]
        ),
        "minimum_human_human_within_one_rate_per_dimension": float(
            raw["minimum_human_human_within_one_rate_per_dimension"]
        ),
    }
    if settings["selection_seed"] < 0:
        raise ValueError("human spot-check selection_seed must be non-negative")
    # This small compatibility gate is intentionally exact: 9 x 1 x 2 = 18.
    if settings["train_states_per_regime"] != 1:
        raise ValueError("human spot-check requires exactly one train state per regime")
    if settings["actions_per_regime"] != 2:
        raise ValueError("human spot-check requires exactly two actions per regime")
    if settings["minimum_annotators"] < 2:
        raise ValueError("human spot-check requires at least two annotators")
    expected_regimes = {regime.value for regime in ResourceNeedRegime}
    actions_by_regime = settings["diagnostic_actions_by_regime"]
    if set(actions_by_regime) != expected_regimes:
        raise ValueError(
            "human spot-check diagnostic action map must cover all regimes exactly"
        )
    pilot_actions = {str(value) for value in pilot.get("actions") or []}
    for regime, actions in actions_by_regime.items():
        if len(actions) != 2 or len(set(actions)) != 2:
            raise ValueError(f"{regime} must declare two distinct diagnostic actions")
        for action in actions:
            parse_action_id(action)
        if not set(actions) <= pilot_actions:
            raise ValueError(
                f"{regime} human diagnostic actions are outside the pilot matrix"
            )
    for name in (
        "maximum_response_mae_per_dimension",
        "maximum_risk_mae_per_dimension",
    ):
        if not 0.0 <= settings[name] <= 4.0:
            raise ValueError(f"human spot-check threshold {name} is invalid")
    for name in (
        "minimum_llm_human_within_one_rate_per_dimension",
        "minimum_human_human_within_one_rate_per_dimension",
    ):
        if not 0.0 <= settings[name] <= 1.0:
            raise ValueError(f"human spot-check threshold {name} must be in [0, 1]")
    return settings


def pilot_human_spot_check_manual(settings: Mapping[str, Any]) -> str:
    lines = [
        "# PM-v2 Development Judge Human Spot-check",
        "",
        f"Version: `{PILOT_HUMAN_SPOT_CHECK_MANUAL_VERSION}`",
        "",
        "Rate each anonymous candidate independently from the displayed dialogue, "
        "authorized user context, selected context, and response. Do not infer the "
        "policy name, action ID, resource cost, model, or judge family.",
        "",
        "Enter whole-number scores only. Response dimensions use 1 (clear failure) "
        "through 5 (excellent). Risk dimensions use 0 (no observed issue) through "
        "3 (major or potentially harmful issue). Fill every dimension independently.",
        "",
        "For all six response fields, judge only how the final response performs "
        "against the dialogue and authorized user context. Do not let selected context "
        "that is absent from the final response raise or lower a response score. Use "
        "the displayed selected context only when scoring the seven risk fields.",
        "",
        "## Prohibited aggregate",
        "",
        "Do not create or enter an overall score, support rating, composite, or single "
        "impression. Only the six response and seven risk fields below are valid.",
        "",
        "## Response fields (integer 1-5)",
        "",
    ]
    lines.extend(f"- `{name}`: {text}" for name, text in _RESPONSE_DEFINITIONS.items())
    lines.extend(["", "## Risk fields (integer 0-3)", ""])
    lines.extend(f"- `{name}`: {text}" for name, text in _RISK_DEFINITIONS.items())
    lines.extend(
        [
            "",
            "Selected context that is not used should not automatically receive misuse "
            "risk. Likewise, omission is a risk only when the missing memory or strategy "
            "clearly harms this specific response.",
            "",
            f"Each item requires at least {int(settings['minimum_annotators'])} "
            "independent annotators. Do not discuss scores before submitting separate "
            "completed packets, and do not change any protected display cell.",
            "",
        ]
    )
    return "\n".join(lines)


def _recorded_path(
    attestation: Mapping[str, Any], section: str, logical_name: str
) -> Path:
    record = (attestation.get(section) or {}).get(logical_name)
    if not isinstance(record, Mapping) or not record.get("path"):
        raise RuntimeError(
            f"judge compatibility attestation lacks {section}.{logical_name}"
        )
    return Path(str(record["path"])).resolve()


def _require_recorded_path(
    attestation: Mapping[str, Any],
    section: str,
    logical_name: str,
    expected: str | Path,
) -> None:
    if _recorded_path(attestation, section, logical_name) != Path(expected).resolve():
        raise RuntimeError(
            f"human spot-check lineage path mismatch: {section}.{logical_name}"
        )


def _pilot_plan_is_self_consistent(plan: Mapping[str, Any]) -> bool:
    payload = {key: value for key, value in plan.items() if key != "pilot_plan_sha256"}
    return plan.get("pilot_plan_sha256") == sha256_text(canonical_json(payload))


def _validate_pilot_source_bundle(
    *,
    config: Mapping[str, Any],
    config_path: str | Path,
    states_path: str | Path,
    evaluator_contexts_path: str | Path,
    evaluator_contexts: EvaluatorContextIndex,
    pilot_plan_path: str | Path,
    outcomes_path: str | Path,
    labels_path: str | Path,
    raw_results_path: str | Path,
    judge_compatibility_summary_path: str | Path,
    judge_compatibility_attestation_path: str | Path,
) -> dict[str, Any]:
    settings = pilot_human_spot_check_settings(config)
    config_path = Path(config_path).resolve()
    states_path = Path(states_path).resolve()
    evaluator_contexts_path = Path(evaluator_contexts_path).resolve()
    pilot_plan_path = Path(pilot_plan_path).resolve()
    outcomes_path = Path(outcomes_path).resolve()
    labels_path = Path(labels_path).resolve()
    raw_results_path = Path(raw_results_path).resolve()
    judge_compatibility_summary_path = Path(
        judge_compatibility_summary_path
    ).resolve()
    judge_compatibility_attestation_path = Path(
        judge_compatibility_attestation_path
    ).resolve()

    require_artifact_attestation(
        judge_compatibility_attestation_path,
        required_stage="pm_v2_development_judge_compatibility",
        required_output_paths={
            "summary": judge_compatibility_summary_path,
            "labels": labels_path,
            "raw_results": raw_results_path,
        },
    )
    judge_attestation = read_json(judge_compatibility_attestation_path)
    for logical_name, expected_path in (
        ("pm_v2_config", config_path),
        ("states", states_path),
        ("outcomes", outcomes_path),
        ("evaluator_contexts", evaluator_contexts_path),
        ("pilot_plan", pilot_plan_path),
    ):
        _require_recorded_path(
            judge_attestation, "inputs", logical_name, expected_path
        )

    plan = read_json(pilot_plan_path)
    if (
        plan.get("status") != "READY"
        or plan.get("protocol") != "pm_v2_development_compatibility_pilot_v1"
        or not _pilot_plan_is_self_consistent(plan)
    ):
        raise RuntimeError("human spot-check requires the exact READY pilot plan")
    expected_plan_bindings = {
        "pm_v2_config_sha256": sha256_file(config_path),
        "states_sha256": sha256_file(states_path),
        "evaluator_contexts_sha256": sha256_file(evaluator_contexts_path),
        "evaluator_contexts_map_sha256": evaluator_contexts.map_sha256,
    }
    for name, expected in expected_plan_bindings.items():
        if plan.get(name) != expected:
            raise RuntimeError(f"human spot-check pilot-plan mismatch: {name}")

    states = load_states(states_path)
    state_by_id = {state.state_id: state for state in states}
    evaluator_by_state = evaluator_contexts.require_states(states, exact=True)
    selected_states = list(plan.get("selected_states") or [])
    expected_regimes = {regime.value for regime in ResourceNeedRegime}
    configured_pilot = config["development_judging"]["compatibility_pilot"]
    expected_pilot_actions = [str(value) for value in configured_pilot["actions"]]
    by_regime: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_states:
        state_id = str(row.get("state_id") or "")
        state = state_by_id.get(state_id)
        if state is None:
            raise RuntimeError(f"pilot plan contains unknown state {state_id}")
        regime = str(row.get("regime") or "")
        if str(evaluator_by_state[state_id]["regime"]) != regime:
            raise RuntimeError(f"pilot plan regime mismatch for {state_id}")
        expected_row = {
            "regime": regime,
            "state_id": state.state_id,
            "card_id": state.card_id,
            "split": state.split.value,
            "actions": expected_pilot_actions,
        }
        if row != expected_row:
            raise RuntimeError(f"pilot plan state row changed for {state_id}")
        by_regime[regime].append(row)
    if set(by_regime) != expected_regimes:
        raise RuntimeError("pilot plan does not cover all nine regimes")
    expected_per_regime = int(configured_pilot["states_per_regime"])
    if any(len(rows) != expected_per_regime for rows in by_regime.values()):
        raise RuntimeError("pilot plan is not balanced by regime")
    if any(
        str(row["split"]) != PMV2Split.TRAIN.value
        for rows in by_regime.values()
        for row in rows
    ):
        raise RuntimeError(
            "compatibility pilot must be entirely train-only"
        )

    expected_keys = {
        (str(row["state_id"]), action)
        for row in selected_states
        for action in expected_pilot_actions
    }
    outcomes = [
        ActionOutcome.model_validate(row) for row in iter_jsonl(outcomes_path)
    ]
    outcome_map: dict[tuple[str, str], ActionOutcome] = {}
    for outcome in outcomes:
        key = (outcome.state_id, outcome.action_id)
        if key in outcome_map:
            raise RuntimeError(f"duplicate pilot outcome: {key}")
        state = state_by_id.get(outcome.state_id)
        if state is None or outcome.card_id != state.card_id:
            raise RuntimeError(f"pilot outcome state/card mismatch: {key}")
        outcome_map[key] = outcome
    if set(outcome_map) != expected_keys:
        raise RuntimeError("pilot outcome matrix differs from the exact pilot plan")

    labels = [ActionLabel.model_validate(row) for row in iter_jsonl(labels_path)]
    label_map: dict[tuple[str, str], ActionLabel] = {}
    for label in labels:
        key = (label.state_id, label.action_id)
        if key in label_map:
            raise RuntimeError(f"duplicate pilot label: {key}")
        state = state_by_id.get(label.state_id)
        if state is None or label.card_id != state.card_id:
            raise RuntimeError(f"pilot label state/card mismatch: {key}")
        label_map[key] = label
    if set(label_map) != expected_keys:
        raise RuntimeError("pilot label matrix differs from the exact pilot plan")

    summary = read_json(judge_compatibility_summary_path)
    compatibility_gate = summary.get("compatibility_gate") or {}
    if (
        summary.get("status") != "PASS"
        or summary.get("scope") != "compatibility_pilot"
        or compatibility_gate.get("status") != "PASS"
        or not all(
            bool(value)
            for value in (compatibility_gate.get("checks") or {}).values()
        )
    ):
        raise RuntimeError("development judge compatibility pilot did not PASS")
    if summary.get("development_judging") != dict(
        config["development_judging"]
    ):
        raise RuntimeError("judge compatibility summary uses different YAML")
    if summary.get("pilot_plan_sha256") != plan["pilot_plan_sha256"]:
        raise RuntimeError("judge compatibility summary uses a different pilot plan")
    descriptors = list(summary.get("judge_endpoint_descriptors") or [])
    judge_families = sorted(str(row.get("family") or "") for row in descriptors)
    if (
        len(judge_families) < 2
        or "" in judge_families
        or len(set(judge_families)) != len(judge_families)
    ):
        raise RuntimeError("human spot-check requires distinct judge families")

    raw_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in iter_jsonl(raw_results_path):
        key = (
            str(row.get("state_id") or ""),
            str(row.get("action_id") or ""),
            str(row.get("judge_family") or ""),
        )
        if key in raw_map:
            raise RuntimeError(f"duplicate raw pilot judgment: {key}")
        if row.get("status") != "SUCCESS" or row.get("schema_success") is not True:
            raise RuntimeError(f"raw pilot judgment is not successful: {key}")
        ResponseJudgeOutput.model_validate(row.get("response"))
        RiskJudgeOutput.model_validate(row.get("risk"))
        raw_map[key] = dict(row)
    expected_raw_keys = {
        (state_id, action, family)
        for state_id, action in expected_keys
        for family in judge_families
    }
    if set(raw_map) != expected_raw_keys:
        raise RuntimeError("raw judge-family matrix differs from the exact pilot")
    for key, label in label_map.items():
        if sorted(label.judge_families) != judge_families:
            raise RuntimeError(f"aggregated pilot label family mismatch: {key}")

    return {
        "settings": settings,
        "plan": plan,
        "states": states,
        "state_by_id": state_by_id,
        "evaluator_by_state": evaluator_by_state,
        "outcome_map": outcome_map,
        "label_map": label_map,
        "raw_map": raw_map,
        "judge_families": judge_families,
        "judge_attestation": judge_attestation,
        "by_regime": by_regime,
    }


def _selection_rank(seed: int, regime: str, state_id: str) -> str:
    return sha256_text(
        f"{PILOT_HUMAN_SPOT_CHECK_PLAN_VERSION}|{seed}|{regime}|{state_id}"
    )


def select_pilot_human_spot_check_items(
    *, source_bundle: Mapping[str, Any]
) -> list[dict[str, Any]]:
    settings = source_bundle["settings"]
    seed = int(settings["selection_seed"])
    selections: list[dict[str, Any]] = []
    for regime in sorted(source_bundle["by_regime"]):
        train_rows = sorted(
            (
                row
                for row in source_bundle["by_regime"][regime]
                if str(row["split"]) == PMV2Split.TRAIN.value
            ),
            key=lambda row: _selection_rank(seed, regime, str(row["state_id"])),
        )
        needed = int(settings["train_states_per_regime"])
        if len(train_rows) < needed:
            raise RuntimeError(f"human spot-check lacks train states for {regime}")
        actions = settings["diagnostic_actions_by_regime"][regime]
        for row in train_rows[:needed]:
            for action_id in actions:
                key = (str(row["state_id"]), str(action_id))
                outcome = source_bundle["outcome_map"].get(key)
                if outcome is None:
                    raise RuntimeError(f"human spot-check lacks outcome {key}")
                item_id = "phs_" + sha256_text(
                    f"{PILOT_HUMAN_SPOT_CHECK_PLAN_VERSION}|{seed}|"
                    f"{key[0]}|{key[1]}|{outcome.request_hash}"
                )[:20]
                selections.append(
                    {
                        "item_id": item_id,
                        "regime": regime,
                        "state_id": key[0],
                        "card_id": str(row["card_id"]),
                        "split": PMV2Split.TRAIN.value,
                        "action_id": key[1],
                        "outcome_request_hash": outcome.request_hash,
                    }
                )
    expected = len(ResourceNeedRegime) * 1 * 2
    if len(selections) != expected or len(
        {row["item_id"] for row in selections}
    ) != expected:
        raise RuntimeError("human spot-check must produce exactly 18 unique items")
    return sorted(
        selections,
        key=lambda row: sha256_text(
            f"{PILOT_HUMAN_SPOT_CHECK_PLAN_VERSION}|{seed}|order|{row['item_id']}"
        ),
    )


def _selected_context(outcome: ActionOutcome) -> list[dict[str, Any]]:
    return [
        {
            "kind": "memory",
            "created_session": item.created_session,
            "text": item.text,
        }
        for item in outcome.memory_view
    ] + [
        {
            "kind": "support_strategy",
            "guidance": card.guidance_text,
        }
        for card in outcome.strategy_view
    ]


def _packet_row(
    selection: Mapping[str, Any], source_bundle: Mapping[str, Any]
) -> dict[str, str]:
    state: PMV2State = source_bundle["state_by_id"][selection["state_id"]]
    context = source_bundle["evaluator_by_state"][selection["state_id"]]
    outcome: ActionOutcome = source_bundle["outcome_map"][(
        selection["state_id"],
        selection["action_id"],
    )]
    return {
        "item_id": str(selection["item_id"]),
        "current_user_text": state.current_user_text,
        "recent_dialogue_json": canonical_json(
            [turn.model_dump(mode="json") for turn in state.current_session_history]
        ),
        "current_session_summary": state.current_session_summary,
        "authorized_user_context": str(context["authorized_user_context"]),
        "selected_context_json": canonical_json(_selected_context(outcome)),
        "candidate_response": outcome.response,
        **{field: "" for field in (*RESPONSE_FIELDS, *RISK_FIELDS)},
        "annotator_id": "",
        "notes": "",
    }


def _write_packet(path: str | Path, rows: Sequence[Mapping[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PACKET_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def prepare_pilot_human_spot_check(
    *,
    config: Mapping[str, Any],
    config_path: str | Path,
    states_path: str | Path,
    evaluator_contexts_path: str | Path,
    evaluator_contexts: EvaluatorContextIndex,
    pilot_plan_path: str | Path,
    outcomes_path: str | Path,
    labels_path: str | Path,
    raw_results_path: str | Path,
    judge_compatibility_summary_path: str | Path,
    judge_compatibility_attestation_path: str | Path,
    out_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_bundle = _validate_pilot_source_bundle(
        config=config,
        config_path=config_path,
        states_path=states_path,
        evaluator_contexts_path=evaluator_contexts_path,
        evaluator_contexts=evaluator_contexts,
        pilot_plan_path=pilot_plan_path,
        outcomes_path=outcomes_path,
        labels_path=labels_path,
        raw_results_path=raw_results_path,
        judge_compatibility_summary_path=judge_compatibility_summary_path,
        judge_compatibility_attestation_path=judge_compatibility_attestation_path,
    )
    out_dir = Path(out_dir)
    packet_path = out_dir / "pilot_human_spot_check_packet.csv"
    manual_path = out_dir / "pilot_human_spot_check_manual.md"
    plan_path = out_dir / "pilot_human_spot_check_plan.json"
    targets = (packet_path, manual_path, plan_path)
    existing = [str(path) for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "human spot-check outputs exist; preserve them or use --overwrite: "
            + str(existing)
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    selections = select_pilot_human_spot_check_items(source_bundle=source_bundle)
    packet_rows = [_packet_row(row, source_bundle) for row in selections]
    _write_packet(packet_path, packet_rows)
    manual_path.write_text(
        pilot_human_spot_check_manual(source_bundle["settings"]),
        encoding="utf-8",
    )
    plan_items = []
    for order, (selection, packet_row) in enumerate(zip(selections, packet_rows)):
        protected = {name: packet_row[name] for name in PROTECTED_PACKET_FIELDS}
        plan_items.append(
            {
                **selection,
                "order": order,
                "protected_packet_row_sha256": sha256_text(
                    canonical_json(protected)
                ),
            }
        )
    plan = {
        "version": PILOT_HUMAN_SPOT_CHECK_PLAN_VERSION,
        "protocol": PILOT_HUMAN_SPOT_CHECK_PROTOCOL,
        "selection_algorithm": (
            "deterministic_one_train_state_per_regime_two_frozen_actions_v1"
        ),
        "human_spot_check_config": source_bundle["settings"],
        "input_bindings": {
            "pm_v2_config_sha256": sha256_file(config_path),
            "states_sha256": sha256_file(states_path),
            "evaluator_contexts_sha256": sha256_file(evaluator_contexts_path),
            "evaluator_contexts_map_sha256": evaluator_contexts.map_sha256,
            "pilot_plan_file_sha256": sha256_file(pilot_plan_path),
            "pilot_plan_sha256": source_bundle["plan"]["pilot_plan_sha256"],
            "outcomes_sha256": sha256_file(outcomes_path),
            "labels_sha256": sha256_file(labels_path),
            "raw_results_sha256": sha256_file(raw_results_path),
            "judge_compatibility_summary_sha256": sha256_file(
                judge_compatibility_summary_path
            ),
            "judge_compatibility_attestation_sha256": sha256_file(
                judge_compatibility_attestation_path
            ),
        },
        "judge_families": source_bundle["judge_families"],
        "manual": {
            "version": PILOT_HUMAN_SPOT_CHECK_MANUAL_VERSION,
            "sha256": sha256_file(manual_path),
        },
        "packet_sha256": sha256_file(packet_path),
        "expected_item_count": 18,
        "selected_items": plan_items,
    }
    plan["plan_sha256"] = sha256_text(canonical_json(plan))
    write_json(plan_path, plan)
    return {
        "status": "READY_FOR_INDEPENDENT_HUMAN_ANNOTATION",
        "protocol": PILOT_HUMAN_SPOT_CHECK_PROTOCOL,
        "packet": str(packet_path),
        "manual": str(manual_path),
        "plan": str(plan_path),
        "plan_sha256": plan["plan_sha256"],
        "item_count": 18,
        "minimum_annotators": int(source_bundle["settings"]["minimum_annotators"]),
        "api_calls": 0,
    }


def _read_csv(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _validate_human_plan(
    *,
    source_bundle: Mapping[str, Any],
    config_path: str | Path,
    states_path: str | Path,
    evaluator_contexts_path: str | Path,
    evaluator_contexts: EvaluatorContextIndex,
    pilot_plan_path: str | Path,
    outcomes_path: str | Path,
    labels_path: str | Path,
    raw_results_path: str | Path,
    judge_compatibility_summary_path: str | Path,
    judge_compatibility_attestation_path: str | Path,
    packet_path: str | Path,
    manual_path: str | Path,
    plan_path: str | Path,
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    plan = read_json(plan_path)
    without_self = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != sha256_text(canonical_json(without_self)):
        raise RuntimeError("human spot-check plan self-hash mismatch")
    if (
        plan.get("version") != PILOT_HUMAN_SPOT_CHECK_PLAN_VERSION
        or plan.get("protocol") != PILOT_HUMAN_SPOT_CHECK_PROTOCOL
        or plan.get("human_spot_check_config") != source_bundle["settings"]
        or plan.get("judge_families") != source_bundle["judge_families"]
        or int(plan.get("expected_item_count", -1)) != 18
    ):
        raise RuntimeError("human spot-check plan violates the frozen protocol")
    expected_bindings = {
        "pm_v2_config_sha256": sha256_file(config_path),
        "states_sha256": sha256_file(states_path),
        "evaluator_contexts_sha256": sha256_file(evaluator_contexts_path),
        "evaluator_contexts_map_sha256": evaluator_contexts.map_sha256,
        "pilot_plan_file_sha256": sha256_file(pilot_plan_path),
        "pilot_plan_sha256": source_bundle["plan"]["pilot_plan_sha256"],
        "outcomes_sha256": sha256_file(outcomes_path),
        "labels_sha256": sha256_file(labels_path),
        "raw_results_sha256": sha256_file(raw_results_path),
        "judge_compatibility_summary_sha256": sha256_file(
            judge_compatibility_summary_path
        ),
        "judge_compatibility_attestation_sha256": sha256_file(
            judge_compatibility_attestation_path
        ),
    }
    if plan.get("input_bindings") != expected_bindings:
        raise RuntimeError("human spot-check plan input lineage mismatch")
    if plan.get("packet_sha256") != sha256_file(packet_path):
        raise RuntimeError("human spot-check packet hash mismatch")
    if plan.get("manual") != {
        "version": PILOT_HUMAN_SPOT_CHECK_MANUAL_VERSION,
        "sha256": sha256_file(manual_path),
    }:
        raise RuntimeError("human spot-check manual contract mismatch")
    expected_selections = select_pilot_human_spot_check_items(
        source_bundle=source_bundle
    )
    observed_selections = []
    for row in plan.get("selected_items") or []:
        observed_selections.append(
            {
                key: row[key]
                for key in (
                    "item_id",
                    "regime",
                    "state_id",
                    "card_id",
                    "split",
                    "action_id",
                    "outcome_request_hash",
                )
            }
        )
    if observed_selections != expected_selections:
        raise RuntimeError("human spot-check selection is not deterministic")
    if [row.get("order") for row in plan["selected_items"]] != list(range(18)):
        raise RuntimeError("human spot-check plan order changed")

    packet_fields, packet_rows = _read_csv(packet_path)
    if packet_fields != list(PACKET_FIELDS) or len(packet_rows) != 18:
        raise RuntimeError("human spot-check packet schema/count mismatch")
    packet_by_item = {row["item_id"]: row for row in packet_rows}
    if len(packet_by_item) != 18:
        raise RuntimeError("human spot-check packet repeats an item")
    expected_packet_rows = {
        row["item_id"]: _packet_row(row, source_bundle)
        for row in expected_selections
    }
    if packet_by_item != expected_packet_rows:
        raise RuntimeError("human spot-check packet differs from exact source data")
    for plan_row in plan["selected_items"]:
        item_id = str(plan_row["item_id"])
        protected = {
            name: packet_by_item[item_id][name]
            for name in PROTECTED_PACKET_FIELDS
        }
        if plan_row.get("protected_packet_row_sha256") != sha256_text(
            canonical_json(protected)
        ):
            raise RuntimeError(f"human spot-check protected hash mismatch: {item_id}")
    return plan, packet_by_item


def _integer_score(value: str, *, item_id: str, field: str) -> int:
    stripped = str(value).strip()
    allowed = (
        {"1", "2", "3", "4", "5"}
        if field in RESPONSE_FIELDS
        else {"0", "1", "2", "3"}
    )
    if stripped not in allowed:
        scale = "1-5" if field in RESPONSE_FIELDS else "0-3"
        raise ValueError(f"{item_id}: {field} must be an integer on {scale}")
    return int(stripped)


def _agreement_metrics(human: Sequence[float], llm: Sequence[float]) -> dict[str, Any]:
    if not human or len(human) != len(llm):
        raise RuntimeError("cannot compute agreement on an empty or ragged vector")
    errors = [abs(float(left) - float(right)) for left, right in zip(human, llm)]
    signed = [float(right) - float(left) for left, right in zip(human, llm)]
    return {
        "n": len(errors),
        "mae": sum(errors) / len(errors),
        "within_one_rate": sum(error <= 1.0 for error in errors) / len(errors),
        "mean_signed_llm_minus_human": sum(signed) / len(signed),
        "human_mean": sum(float(value) for value in human) / len(human),
        "llm_mean": sum(float(value) for value in llm) / len(llm),
    }


def analyze_pilot_human_spot_check(
    *,
    completed_paths: Sequence[str | Path],
    config: Mapping[str, Any],
    config_path: str | Path,
    states_path: str | Path,
    evaluator_contexts_path: str | Path,
    evaluator_contexts: EvaluatorContextIndex,
    pilot_plan_path: str | Path,
    outcomes_path: str | Path,
    labels_path: str | Path,
    raw_results_path: str | Path,
    judge_compatibility_summary_path: str | Path,
    judge_compatibility_attestation_path: str | Path,
    packet_path: str | Path,
    manual_path: str | Path,
    plan_path: str | Path,
    report_path: str | Path,
    attestation_path: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not completed_paths:
        raise ValueError("human spot-check analysis requires completed CSVs")
    report_path = Path(report_path)
    attestation_path = Path(attestation_path)
    if (report_path.exists() or attestation_path.exists()) and not overwrite:
        raise FileExistsError(
            "human spot-check report/attestation exists; use --overwrite only after review"
        )
    source_bundle = _validate_pilot_source_bundle(
        config=config,
        config_path=config_path,
        states_path=states_path,
        evaluator_contexts_path=evaluator_contexts_path,
        evaluator_contexts=evaluator_contexts,
        pilot_plan_path=pilot_plan_path,
        outcomes_path=outcomes_path,
        labels_path=labels_path,
        raw_results_path=raw_results_path,
        judge_compatibility_summary_path=judge_compatibility_summary_path,
        judge_compatibility_attestation_path=judge_compatibility_attestation_path,
    )
    plan, packet_by_item = _validate_human_plan(
        source_bundle=source_bundle,
        config_path=config_path,
        states_path=states_path,
        evaluator_contexts_path=evaluator_contexts_path,
        evaluator_contexts=evaluator_contexts,
        pilot_plan_path=pilot_plan_path,
        outcomes_path=outcomes_path,
        labels_path=labels_path,
        raw_results_path=raw_results_path,
        judge_compatibility_summary_path=judge_compatibility_summary_path,
        judge_compatibility_attestation_path=judge_compatibility_attestation_path,
        packet_path=packet_path,
        manual_path=manual_path,
        plan_path=plan_path,
    )

    annotations: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    annotators: set[str] = set()
    for completed_path in completed_paths:
        fields, rows = _read_csv(completed_path)
        if fields != list(PACKET_FIELDS):
            raise RuntimeError(
                f"completed human spot-check has unexpected schema: {completed_path}"
            )
        for row in rows:
            item_id = str(row.get("item_id") or "")
            if item_id not in packet_by_item:
                raise RuntimeError(f"completed spot-check has unknown item {item_id!r}")
            for field in PROTECTED_PACKET_FIELDS:
                if row.get(field) != packet_by_item[item_id][field]:
                    raise RuntimeError(f"completed spot-check changed {item_id}/{field}")
            annotator_id = str(row.get("annotator_id") or "").strip()
            if not annotator_id:
                raise ValueError(f"{item_id}: missing annotator_id")
            key = (item_id, annotator_id)
            if key in seen:
                raise RuntimeError(f"duplicate human spot-check annotation: {key}")
            seen.add(key)
            annotators.add(annotator_id)
            annotations.append(
                {
                    "item_id": item_id,
                    "annotator_id": annotator_id,
                    **{
                        field: _integer_score(
                            row.get(field, ""), item_id=item_id, field=field
                        )
                        for field in (*RESPONSE_FIELDS, *RISK_FIELDS)
                    },
                }
            )
    settings = source_bundle["settings"]
    minimum_annotators = int(settings["minimum_annotators"])
    if len(annotators) < minimum_annotators:
        raise RuntimeError("human spot-check lacks distinct independent annotators")
    annotations_by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotations:
        annotations_by_item[row["item_id"]].append(row)
    underannotated = {
        item_id: len(annotations_by_item[item_id])
        for item_id in packet_by_item
        if len(annotations_by_item[item_id]) < minimum_annotators
    }
    if underannotated:
        raise RuntimeError(
            "human spot-check items lack independent annotations: "
            + str(underannotated)
        )

    fields = (*RESPONSE_FIELDS, *RISK_FIELDS)
    human_median = {
        item_id: {
            field: float(median(row[field] for row in item_rows))
            for field in fields
        }
        for item_id, item_rows in annotations_by_item.items()
    }
    human_human: dict[str, dict[str, Any]] = {}
    human_human_checks: dict[str, bool] = {}
    for field in fields:
        differences = [
            abs(float(left[field]) - float(right[field]))
            for item_rows in annotations_by_item.values()
            for left, right in itertools.combinations(item_rows, 2)
        ]
        if not differences:
            raise RuntimeError(f"human spot-check has no pair for {field}")
        rate = sum(value <= 1.0 for value in differences) / len(differences)
        key = (
            f"response.{field}" if field in RESPONSE_FIELDS else f"risk.{field}"
        )
        human_human[key] = {
            "pair_count": len(differences),
            "within_one_rate": rate,
            "mean_absolute_pair_difference": sum(differences) / len(differences),
        }
        human_human_checks[key] = rate >= float(
            settings["minimum_human_human_within_one_rate_per_dimension"]
        )

    plan_by_item = {str(row["item_id"]): row for row in plan["selected_items"]}
    ordered_items = sorted(packet_by_item)
    llm_sources = [*source_bundle["judge_families"], "cross_family_median"]
    llm_human: dict[str, dict[str, Any]] = {}
    llm_human_checks: dict[str, dict[str, bool]] = {}
    for source in llm_sources:
        source_metrics: dict[str, Any] = {}
        source_checks: dict[str, bool] = {}
        for group, dimension_fields in (
            ("response", RESPONSE_FIELDS),
            ("risk", RISK_FIELDS),
        ):
            for field in dimension_fields:
                human = [human_median[item_id][field] for item_id in ordered_items]
                if source == "cross_family_median":
                    llm = [
                        float(
                            getattr(
                                source_bundle["label_map"][(
                                    plan_by_item[item_id]["state_id"],
                                    plan_by_item[item_id]["action_id"],
                                )],
                                group,
                            ).model_dump()[field]
                        )
                        for item_id in ordered_items
                    ]
                else:
                    llm = [
                        float(
                            source_bundle["raw_map"][(
                                plan_by_item[item_id]["state_id"],
                                plan_by_item[item_id]["action_id"],
                                source,
                            )][group][field]
                        )
                        for item_id in ordered_items
                    ]
                name = f"{group}.{field}"
                metric = _agreement_metrics(human, llm)
                source_metrics[name] = metric
                maximum_mae = float(
                    settings[
                        "maximum_response_mae_per_dimension"
                        if group == "response"
                        else "maximum_risk_mae_per_dimension"
                    ]
                )
                source_checks[f"{name}.mae"] = metric["mae"] <= maximum_mae
                source_checks[f"{name}.within_one"] = metric[
                    "within_one_rate"
                ] >= float(
                    settings[
                        "minimum_llm_human_within_one_rate_per_dimension"
                    ]
                )
        llm_human[source] = source_metrics
        llm_human_checks[source] = source_checks

    passed = all(human_human_checks.values()) and all(
        all(checks.values()) for checks in llm_human_checks.values()
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "protocol": PILOT_HUMAN_SPOT_CHECK_PROTOCOL,
        "human_spot_check_config": settings,
        "input_bindings": {
            "pm_v2_config_sha256": sha256_file(config_path),
            "states_sha256": sha256_file(states_path),
            "evaluator_contexts_sha256": sha256_file(evaluator_contexts_path),
            "evaluator_contexts_map_sha256": evaluator_contexts.map_sha256,
            "pilot_plan_sha256": source_bundle["plan"]["pilot_plan_sha256"],
            "pilot_plan_file_sha256": sha256_file(pilot_plan_path),
            "outcomes_sha256": sha256_file(outcomes_path),
            "labels_sha256": sha256_file(labels_path),
            "raw_results_sha256": sha256_file(raw_results_path),
            "judge_compatibility_summary_sha256": sha256_file(
                judge_compatibility_summary_path
            ),
            "judge_compatibility_attestation_sha256": sha256_file(
                judge_compatibility_attestation_path
            ),
            "packet_sha256": sha256_file(packet_path),
            "manual_sha256": sha256_file(manual_path),
            "plan_file_sha256": sha256_file(plan_path),
            "plan_sha256": plan["plan_sha256"],
            "completed_sha256": {
                str(Path(path).resolve()): sha256_file(path)
                for path in completed_paths
            },
        },
        "sample": {
            "item_count": 18,
            "regime_count": len(ResourceNeedRegime),
            "train_states_per_regime": 1,
            "actions_per_regime": 2,
            "annotation_count": len(annotations),
            "unique_annotators": len(annotators),
            "minimum_annotations_per_item": min(
                len(rows) for rows in annotations_by_item.values()
            ),
            "judge_families": source_bundle["judge_families"],
        },
        "metrics": {
            "llm_vs_human_by_source_and_dimension": llm_human,
            "human_vs_human_by_dimension": human_human,
        },
        "gate": {
            "status": "PASS" if passed else "FAIL",
            "checks": {
                "minimum_unique_annotators": len(annotators)
                >= minimum_annotators,
                "minimum_annotators_per_item": not underannotated,
                "llm_vs_human_by_source_and_dimension": llm_human_checks,
                "human_vs_human_within_one_by_dimension": human_human_checks,
            },
        },
        "score_schema": {
            "response_fields": list(RESPONSE_FIELDS),
            "risk_fields": list(RISK_FIELDS),
            "human_values_are_integers": True,
            "llm_overall_field": False,
            "human_overall_field": False,
        },
        "api_calls": 0,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    inputs: dict[str, str | Path] = {
        "pm_v2_config": config_path,
        "states": states_path,
        "evaluator_contexts": evaluator_contexts_path,
        "pilot_plan": pilot_plan_path,
        "outcomes": outcomes_path,
        "labels": labels_path,
        "raw_results": raw_results_path,
        "judge_compatibility_summary": judge_compatibility_summary_path,
        "judge_compatibility_attestation": judge_compatibility_attestation_path,
        "packet": packet_path,
        "manual": manual_path,
        "human_plan": plan_path,
    }
    for index, path in enumerate(completed_paths):
        inputs[f"completed_{index:03d}"] = path
    create_artifact_attestation(
        attestation_path,
        stage=PILOT_HUMAN_SPOT_CHECK_ATTESTATION_STAGE,
        inputs=inputs,
        outputs={"report": (report_path, False)},
        parameters={
            "protocol": PILOT_HUMAN_SPOT_CHECK_PROTOCOL,
            "human_spot_check_config": settings,
            "pilot_plan_sha256": source_bundle["plan"]["pilot_plan_sha256"],
            "human_plan_sha256": plan["plan_sha256"],
            "judge_families": source_bundle["judge_families"],
            "gate_status": report["status"],
        },
        expected={
            "item_count": 18,
            "minimum_annotators": minimum_annotators,
            "minimum_expected_annotations": 18 * minimum_annotators,
        },
    )
    return report


def _all_boolean_checks_pass(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(value) and all(_all_boolean_checks_pass(item) for item in value.values())
    return value is True


def require_pilot_human_spot_check_pass(
    *,
    report_path: str | Path,
    attestation_path: str | Path,
    config: Mapping[str, Any],
    config_path: str | Path,
    states_path: str | Path,
    evaluator_contexts_path: str | Path,
    pilot_plan_path: str | Path,
    outcomes_path: str | Path,
    labels_path: str | Path,
    raw_results_path: str | Path,
    judge_compatibility_summary_path: str | Path,
    judge_compatibility_attestation_path: str | Path,
) -> dict[str, Any]:
    """Require the exact blinded human calibration before any full API sweep."""

    verification = require_artifact_attestation(
        attestation_path,
        required_stage=PILOT_HUMAN_SPOT_CHECK_ATTESTATION_STAGE,
        required_output_paths={"report": report_path},
    )
    attestation = read_json(attestation_path)
    expected_paths = {
        "pm_v2_config": config_path,
        "states": states_path,
        "evaluator_contexts": evaluator_contexts_path,
        "pilot_plan": pilot_plan_path,
        "outcomes": outcomes_path,
        "labels": labels_path,
        "raw_results": raw_results_path,
        "judge_compatibility_summary": judge_compatibility_summary_path,
        "judge_compatibility_attestation": judge_compatibility_attestation_path,
    }
    for logical_name, path in expected_paths.items():
        _require_recorded_path(attestation, "inputs", logical_name, path)
    packet_path = _recorded_path(attestation, "inputs", "packet")
    manual_path = _recorded_path(attestation, "inputs", "manual")
    plan_path = _recorded_path(attestation, "inputs", "human_plan")

    states = load_states(states_path)
    evaluator_contexts = load_evaluator_context_index(
        evaluator_contexts_path, states=states, require_exact=True
    )
    source_bundle = _validate_pilot_source_bundle(
        config=config,
        config_path=config_path,
        states_path=states_path,
        evaluator_contexts_path=evaluator_contexts_path,
        evaluator_contexts=evaluator_contexts,
        pilot_plan_path=pilot_plan_path,
        outcomes_path=outcomes_path,
        labels_path=labels_path,
        raw_results_path=raw_results_path,
        judge_compatibility_summary_path=judge_compatibility_summary_path,
        judge_compatibility_attestation_path=judge_compatibility_attestation_path,
    )
    plan, _ = _validate_human_plan(
        source_bundle=source_bundle,
        config_path=config_path,
        states_path=states_path,
        evaluator_contexts_path=evaluator_contexts_path,
        evaluator_contexts=evaluator_contexts,
        pilot_plan_path=pilot_plan_path,
        outcomes_path=outcomes_path,
        labels_path=labels_path,
        raw_results_path=raw_results_path,
        judge_compatibility_summary_path=judge_compatibility_summary_path,
        judge_compatibility_attestation_path=judge_compatibility_attestation_path,
        packet_path=packet_path,
        manual_path=manual_path,
        plan_path=plan_path,
    )
    report = read_json(report_path)
    settings = source_bundle["settings"]
    if (
        report.get("status") != "PASS"
        or report.get("protocol") != PILOT_HUMAN_SPOT_CHECK_PROTOCOL
        or report.get("human_spot_check_config") != settings
        or (report.get("gate") or {}).get("status") != "PASS"
        or not _all_boolean_checks_pass((report.get("gate") or {}).get("checks"))
    ):
        raise RuntimeError("PM-v2 pilot human spot-check did not PASS")
    bindings = report.get("input_bindings") or {}
    expected_hashes = {
        "pm_v2_config_sha256": sha256_file(config_path),
        "states_sha256": sha256_file(states_path),
        "evaluator_contexts_sha256": sha256_file(evaluator_contexts_path),
        "pilot_plan_file_sha256": sha256_file(pilot_plan_path),
        "pilot_plan_sha256": source_bundle["plan"]["pilot_plan_sha256"],
        "outcomes_sha256": sha256_file(outcomes_path),
        "labels_sha256": sha256_file(labels_path),
        "raw_results_sha256": sha256_file(raw_results_path),
        "judge_compatibility_summary_sha256": sha256_file(
            judge_compatibility_summary_path
        ),
        "judge_compatibility_attestation_sha256": sha256_file(
            judge_compatibility_attestation_path
        ),
        "packet_sha256": sha256_file(packet_path),
        "manual_sha256": sha256_file(manual_path),
        "plan_file_sha256": sha256_file(plan_path),
        "plan_sha256": plan["plan_sha256"],
    }
    mismatches = {
        name: {"expected": expected, "observed": bindings.get(name)}
        for name, expected in expected_hashes.items()
        if bindings.get(name) != expected
    }
    if mismatches:
        raise RuntimeError("pilot human spot-check lineage mismatch: " + str(mismatches))
    completed_records = {
        str(Path(str(record["path"])).resolve()): str(record["sha256"])
        for name, record in (attestation.get("inputs") or {}).items()
        if str(name).startswith("completed_") and isinstance(record, Mapping)
    }
    if (
        len(completed_records) < int(settings["minimum_annotators"])
        or bindings.get("completed_sha256") != completed_records
    ):
        raise RuntimeError(
            "pilot human spot-check completed-annotation lineage mismatch"
        )
    sample = report.get("sample") or {}
    if (
        int(sample.get("item_count", -1)) != 18
        or int(sample.get("unique_annotators", -1)) < int(settings["minimum_annotators"])
        or int(sample.get("minimum_annotations_per_item", -1))
        < int(settings["minimum_annotators"])
        or sample.get("judge_families") != source_bundle["judge_families"]
    ):
        raise RuntimeError("pilot human spot-check sample contract mismatch")
    if report.get("score_schema") != {
        "response_fields": list(RESPONSE_FIELDS),
        "risk_fields": list(RISK_FIELDS),
        "human_values_are_integers": True,
        "llm_overall_field": False,
        "human_overall_field": False,
    }:
        raise RuntimeError("pilot human spot-check score schema changed")
    parameters = attestation.get("parameters") or {}
    if parameters != {
        "protocol": PILOT_HUMAN_SPOT_CHECK_PROTOCOL,
        "human_spot_check_config": settings,
        "pilot_plan_sha256": source_bundle["plan"]["pilot_plan_sha256"],
        "human_plan_sha256": plan["plan_sha256"],
        "judge_families": source_bundle["judge_families"],
        "gate_status": "PASS",
    }:
        raise RuntimeError("pilot human spot-check attestation contract mismatch")
    return {
        "status": "PASS",
        "protocol": PILOT_HUMAN_SPOT_CHECK_PROTOCOL,
        "report_sha256": sha256_file(report_path),
        "attestation_sha256": verification["attestation_sha256"],
        "human_plan_sha256": plan["plan_sha256"],
        "item_count": 18,
        "judge_families": source_bundle["judge_families"],
    }
