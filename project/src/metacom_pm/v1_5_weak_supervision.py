from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import ActionOutcome
from .io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from .pm_v2_contracts import ActionLabel, PMV2Split, PMV2State
from .pm_v2_judging import (
    JudgeResult,
    ResponseJudgeOutput,
    RiskJudgeOutput,
    build_action_label,
    composite_spec_from_config,
    labeling_settings_from_config,
)


WEAK_SUPERVISION_PROTOCOL = "pm-v1.5-llm-weak-supervision-v1"
WEAK_SUPERVISION_STATUS = "LLM_WEAK_SUPERVISION_NOT_GOLD"
WEAK_SUPERVISION_OUTPUT_STATUS = "COMPLETE_WEAK_SUPERVISION_NOT_GOLD"


def require_weak_supervision_contract(path: str | Path) -> dict[str, Any]:
    contract = read_json(path)
    expected_hash = contract.get("contract_sha256")
    body = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if expected_hash != sha256_text(canonical_json(body)):
        raise RuntimeError("weak-supervision contract self-hash mismatch")
    if contract.get("protocol") != WEAK_SUPERVISION_PROTOCOL:
        raise RuntimeError("weak-supervision protocol mismatch")
    if contract.get("status") != WEAK_SUPERVISION_STATUS:
        raise RuntimeError("weak-supervision status mismatch")
    if contract.get("automatic_gold_label_claimed") is not False:
        raise RuntimeError("weak supervision must explicitly reject a gold-label claim")
    if contract.get("human_anchors_used_as_automatic_gold") is not False:
        raise RuntimeError("human anchors must not be promoted to automatic gold")
    return contract


def require_weak_supervision_split(
    contract: Mapping[str, Any], *, domain: str, split: str
) -> None:
    allowed = set((contract.get("scope") or {}).get(domain) or [])
    if split not in allowed:
        raise RuntimeError(
            f"weak-supervision contract does not authorize {domain}/{split}"
        )
    if split == PMV2Split.INTERNAL_TEST.value:
        raise RuntimeError("weak-supervision contract cannot open internal-test outcomes")


def require_weak_supervision_source_hashes(
    contract: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, str]:
    declared = contract.get("source_lineage")
    if not isinstance(declared, Mapping):
        raise RuntimeError("weak-supervision contract lacks source_lineage")
    actual: dict[str, str] = {}
    for name, path in sorted(paths.items()):
        if not path.is_file():
            raise FileNotFoundError(path)
        actual[name] = sha256_file(path)
        if declared.get(f"{name}_sha256") != actual[name]:
            raise RuntimeError(f"weak-supervision source hash mismatch: {name}")
    return actual


def _unique_index(
    rows: Iterable[Any], *, key, label: str
) -> dict[tuple[str, ...], Any]:
    result: dict[tuple[str, ...], Any] = {}
    for row in rows:
        logical_key = tuple(str(value) for value in key(row))
        if logical_key in result:
            raise RuntimeError(f"duplicate {label} key: {logical_key}")
        result[logical_key] = row
    return result


def _load_states_for_splits(
    path: Path, allowed_splits: frozenset[PMV2Split]
) -> list[PMV2State]:
    states: list[PMV2State] = []
    for row in iter_jsonl(path):
        split = PMV2Split(str(row.get("split")))
        if split in allowed_splits:
            # PMV2State is strict but its persisted wire format carries enum
            # strings.  Validate through the JSON entry point, exactly as the
            # canonical load_states() helper does.
            states.append(PMV2State.model_validate_json(canonical_json(row)))
    return states


def _load_judge_results(path: Path) -> dict[tuple[str, str], list[JudgeResult]]:
    grouped: dict[tuple[str, str], list[JudgeResult]] = defaultdict(list)
    raw_keys: set[tuple[str, str, str]] = set()
    for row in iter_jsonl(path):
        if row.get("status") != "SUCCESS" or row.get("schema_success") is not True:
            raise RuntimeError("weak supervision accepts only complete schema-success rows")
        raw_key = (
            str(row["state_id"]),
            str(row["action_id"]),
            str(row["judge_family"]),
        )
        if raw_key in raw_keys:
            raise RuntimeError(f"duplicate raw judge result: {raw_key}")
        raw_keys.add(raw_key)
        grouped[raw_key[:2]].append(
            JudgeResult(
                family=raw_key[2],
                model=str(row["judge_model"]),
                response=ResponseJudgeOutput.model_validate(row["response"]),
                risk=RiskJudgeOutput.model_validate(row["risk"]),
                response_request_hash=str(row["response_request_hash"]),
                risk_request_hash=str(row["risk_request_hash"]),
            )
        )
    return grouped


def _label_report(labels: Sequence[ActionLabel]) -> dict[str, Any]:
    max_mads = [float(label.max_dimension_mad) for label in labels]
    by_family_set = Counter(
        "+".join(sorted(label.judge_families)) for label in labels
    )
    return {
        "rows": len(labels),
        "states": len({label.state_id for label in labels}),
        "actions": sorted({label.action_id for label in labels}),
        "judge_family_sets": dict(sorted(by_family_set.items())),
        "label_reliable_rate_diagnostic_only": (
            sum(bool(label.label_reliable) for label in labels) / len(labels)
        ),
        "maximum_dimension_mad": max(max_mads, default=0.0),
        "mean_max_dimension_mad": sum(max_mads) / len(max_mads),
        "rows_filtered_for_disagreement": 0,
    }


def compile_longitudinal_weak_labels(
    *,
    states_path: Path,
    outcomes_path: Path,
    raw_judges_path: Path,
    pm_config: Mapping[str, Any],
    contract: Mapping[str, Any],
    contract_sha256: str,
) -> list[ActionLabel]:
    states = _load_states_for_splits(
        states_path, frozenset({PMV2Split.TRAIN, PMV2Split.CALIBRATION})
    )
    state_by_id = _unique_index(
        states, key=lambda row: (row.state_id,), label="longitudinal state"
    )
    outcomes = [
        outcome
        for outcome in (
            ActionOutcome.model_validate(row) for row in iter_jsonl(outcomes_path)
        )
        if (outcome.state_id,) in state_by_id
    ]
    outcome_by_key = _unique_index(
        outcomes,
        key=lambda row: (row.state_id, row.action_id),
        label="longitudinal outcome",
    )
    expected_keys = {
        (state.state_id, action_id)
        for state in states
        for action_id in state.allowed_actions
    }
    if set(outcome_by_key) != expected_keys:
        raise RuntimeError("longitudinal outcome matrix is not exact")
    judges_by_key = _load_judge_results(raw_judges_path)
    if set(judges_by_key) != expected_keys:
        raise RuntimeError("longitudinal judge matrix is not exact")

    expected_families = frozenset(contract["judge_aggregation"]["judge_families"])
    composite_spec = composite_spec_from_config(pm_config)
    labeling = labeling_settings_from_config(pm_config)
    labels: list[ActionLabel] = []
    for state_id, action_id in sorted(expected_keys):
        state = state_by_id[(state_id,)]
        outcome = outcome_by_key[(state_id, action_id)]
        results = judges_by_key[(state_id, action_id)]
        if frozenset(result.family for result in results) != expected_families:
            raise RuntimeError(
                f"longitudinal judge families differ for {(state_id, action_id)}"
            )
        if outcome.card_id != state.card_id or outcome.user_id != state.user_id:
            raise RuntimeError("longitudinal outcome provenance differs from state")
        labels.append(
            build_action_label(
                state=state,
                action_id=action_id,
                observed_input_tokens=outcome.cost.total_input_tokens,
                retrieval_calls=outcome.cost.retrieval_calls,
                results=results,
                composite_spec=composite_spec,
                minimum_families=len(expected_families),
                reliable_mad_threshold=labeling["reliable_mad_threshold"],
                provenance={
                    "supervision_kind": WEAK_SUPERVISION_STATUS,
                    "automatic_gold_label": False,
                    "weak_supervision_contract_sha256": contract_sha256,
                    "domain": "longitudinal_synthetic",
                    "split": state.split.value,
                    "prompt_equivalence_id": outcome.prompt_equivalence_id,
                    "label_lineage_id": outcome.label_lineage_id,
                },
            )
        )
    return labels


def compile_auxiliary_weak_labels(
    *,
    states_path: Path,
    outcomes_path: Path,
    labels_path: Path,
    split: PMV2Split,
    contract: Mapping[str, Any],
    contract_sha256: str,
) -> list[ActionLabel]:
    if split not in {PMV2Split.TRAIN, PMV2Split.CALIBRATION}:
        raise RuntimeError("auxiliary weak labels may only use train/calibration")
    require_weak_supervision_split(
        contract, domain="esconv_auxiliary", split=split.value
    )
    states = _load_states_for_splits(states_path, frozenset({split}))
    state_by_id = _unique_index(
        states, key=lambda row: (row.state_id,), label="auxiliary state"
    )
    outcomes = [
        ActionOutcome.model_validate(row) for row in iter_jsonl(outcomes_path)
    ]
    outcome_by_key = _unique_index(
        outcomes,
        key=lambda row: (row.state_id, row.action_id),
        label="auxiliary outcome",
    )
    expected_keys = {
        (state.state_id, action_id)
        for state in states
        for action_id in state.allowed_actions
    }
    if set(outcome_by_key) != expected_keys:
        raise RuntimeError("auxiliary outcome matrix is not exact")

    source_labels = [
        ActionLabel.model_validate(row) for row in iter_jsonl(labels_path)
    ]
    labels_by_key = _unique_index(
        source_labels,
        key=lambda row: (row.state_id, row.action_id),
        label=f"auxiliary {split.value} source label",
    )
    if set(labels_by_key) != expected_keys:
        raise RuntimeError(f"auxiliary {split.value} label matrix is not exact")
    expected_families = frozenset(contract["judge_aggregation"]["judge_families"])
    labels: list[ActionLabel] = []
    for state_id, action_id in sorted(expected_keys):
        state = state_by_id[(state_id,)]
        outcome = outcome_by_key[(state_id, action_id)]
        label = labels_by_key[(state_id, action_id)]
        if (
            label.card_id != state.card_id
            or label.user_id != state.user_id
            or outcome.card_id != state.card_id
            or outcome.user_id != state.user_id
        ):
            raise RuntimeError("auxiliary state/outcome/label provenance mismatch")
        if frozenset(label.judge_families) != expected_families:
            raise RuntimeError("auxiliary label judge-family mismatch")
        if label.provenance.get("esconv_auxiliary_split") != split.value:
            raise RuntimeError(
                f"auxiliary label does not carry {split.value} provenance"
            )
        source_contract_sha = label.provenance.get(
            "weak_supervision_contract_sha256"
        )
        if source_contract_sha not in {None, contract_sha256}:
            raise RuntimeError("auxiliary label weak-supervision contract mismatch")
        payload = label.model_dump(mode="json")
        payload["provenance"] = {
            **payload["provenance"],
            "supervision_kind": WEAK_SUPERVISION_STATUS,
            "automatic_gold_label": False,
            "weak_supervision_contract_sha256": contract_sha256,
            "domain": "esconv_auxiliary",
            "split": split.value,
            "source_label_lineage_preserved": True,
            "offline_recovery_lineage_preserved": (
                split is PMV2Split.TRAIN
            ),
        }
        labels.append(ActionLabel.model_validate(payload))
    return labels


def compile_auxiliary_train_weak_labels(
    *,
    states_path: Path,
    outcomes_path: Path,
    recovered_labels_path: Path,
    contract: Mapping[str, Any],
    contract_sha256: str,
) -> list[ActionLabel]:
    """Backward-compatible train wrapper for the audited recovery artifact."""

    return compile_auxiliary_weak_labels(
        states_path=states_path,
        outcomes_path=outcomes_path,
        labels_path=recovered_labels_path,
        split=PMV2Split.TRAIN,
        contract=contract,
        contract_sha256=contract_sha256,
    )


def write_weak_supervision_bundle(
    *,
    out_dir: Path,
    contract_path: Path,
    source_paths: Mapping[str, Path],
    longitudinal_labels: Sequence[ActionLabel],
    auxiliary_labels: Sequence[ActionLabel],
    auxiliary_calibration_labels: Sequence[ActionLabel] | None = None,
    code_paths: Mapping[str, Path],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    longitudinal_path = out_dir / "longitudinal_train_calibration_weak_labels.jsonl"
    auxiliary_path = out_dir / "esconv_auxiliary_train_weak_labels.jsonl"
    auxiliary_calibration_path = (
        out_dir / "esconv_auxiliary_calibration_weak_labels.jsonl"
    )
    report_path = out_dir / "weak_supervision_report.json"
    attestation_path = out_dir / "weak_supervision_attestation.json"
    write_jsonl(
        longitudinal_path,
        [label.model_dump(mode="json") for label in longitudinal_labels],
    )
    write_jsonl(
        auxiliary_path,
        [label.model_dump(mode="json") for label in auxiliary_labels],
    )
    if auxiliary_calibration_labels is not None:
        write_jsonl(
            auxiliary_calibration_path,
            [
                label.model_dump(mode="json")
                for label in auxiliary_calibration_labels
            ],
        )
    contract = require_weak_supervision_contract(contract_path)
    report = {
        "protocol": WEAK_SUPERVISION_PROTOCOL,
        "status": WEAK_SUPERVISION_OUTPUT_STATUS,
        "automatic_gold_label_claimed": False,
        "training_use": (
            "judge-conditioned weak supervision with continuous per-dimension "
            "MAD weighting and action-applicability masking"
        ),
        "known_measurement_limitations_retained": True,
        "internal_test_outcomes_opened": False,
        "external_outcomes_opened": False,
        "human_anchors_used_as_automatic_gold": False,
        "dual_domain_training_readiness": (
            "READY_TRAIN_CALIBRATION_WEAK_SUPERVISION"
            if auxiliary_calibration_labels is not None
            else "BLOCKED_AUXILIARY_CALIBRATION_WEAK_LABELS_PENDING"
        ),
        "remaining_required_supervision": (
            None
            if auxiliary_calibration_labels is not None
            else {
                "domain": "esconv_auxiliary",
                "split": "calibration",
                "expected_states": 170,
                "expected_action_labels": 340,
                "must_use_same_weak_supervision_estimand": True,
            }
        ),
        "domains": {
            "longitudinal_synthetic": _label_report(longitudinal_labels),
            "esconv_auxiliary": {
                "train": _label_report(auxiliary_labels),
                "calibration": (
                    _label_report(auxiliary_calibration_labels)
                    if auxiliary_calibration_labels is not None
                    else None
                ),
            },
        },
    }
    write_json(report_path, report)
    output_paths = {
        "longitudinal_labels": longitudinal_path,
        "auxiliary_labels": auxiliary_path,
        "report": report_path,
    }
    if auxiliary_calibration_labels is not None:
        output_paths["auxiliary_calibration_labels"] = auxiliary_calibration_path
    attestation_body = {
        "protocol": WEAK_SUPERVISION_PROTOCOL,
        "status": "ATTESTED_DETERMINISTIC_ZERO_API",
        "zero_api": True,
        "contract_sha256": sha256_file(contract_path),
        "contract_record_sha256": contract["contract_sha256"],
        "inputs": {
            name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in sorted(source_paths.items())
        },
        "outputs": {
            name: {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                **(
                    {"rows": sum(1 for _ in iter_jsonl(path))}
                    if path.suffix == ".jsonl"
                    else {}
                ),
            }
            for name, path in sorted(output_paths.items())
        },
        "code": {
            name: {"sha256": sha256_file(path)}
            for name, path in sorted(code_paths.items())
        },
        "internal_test_outcomes_opened": False,
        "external_outcomes_opened": False,
    }
    attestation_body["attestation_sha256"] = sha256_text(
        canonical_json(attestation_body)
    )
    write_json(attestation_path, attestation_body)
    return {
        **report,
        "paths": {
            "longitudinal_labels": str(longitudinal_path),
            "auxiliary_labels": str(auxiliary_path),
            "auxiliary_calibration_labels": (
                str(auxiliary_calibration_path)
                if auxiliary_calibration_labels is not None
                else None
            ),
            "report": str(report_path),
            "attestation": str(attestation_path),
        },
        "attestation_sha256": attestation_body["attestation_sha256"],
    }
