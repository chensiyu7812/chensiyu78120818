#!/usr/bin/env python3
"""Seal and attest the PM-v1.5 dual-domain training inputs without API calls.

The script validates train/calibration labels structurally, but it never
deserializes internal-test ``ActionLabel`` values.  Internal files are opened
only by ``seal_internal_label_bundle``, which records row/schema/state-universe
hashes before candidate fitting begins.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.artifacts import create_artifact_attestation, require_artifact_attestation
from metacom_pm.config import load_config
from metacom_pm.internal_holdout import seal_internal_label_bundle
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
)
from metacom_pm.pm_v2_contracts import ActionLabel, PMV2Split
from metacom_pm.pm_v2_data import load_states
from metacom_pm.v1_5_dual_domain_training import (
    DUAL_DOMAIN_TRAINING_PROTOCOL,
    validate_dual_domain_training_inputs,
)


ROOT = Path(__file__).resolve().parents[2]


def _state_universe_sha256(states) -> str:
    return sha256_text(canonical_json(sorted(state.state_id for state in states)))


def _require_internal_seal_matches_states(seal: dict, states, *, domain: str) -> None:
    internal_states = [
        state for state in states if state.split is PMV2Split.INTERNAL_TEST
    ]
    expected_rows = sum(len(state.allowed_actions) for state in internal_states)
    if int(seal.get("state_count") or 0) != len(internal_states):
        raise RuntimeError(f"{domain} internal seal state count mismatch")
    if int(seal.get("row_count") or 0) != expected_rows:
        raise RuntimeError(f"{domain} internal seal action-matrix row count mismatch")
    if seal.get("state_universe_sha256") != _state_universe_sha256(internal_states):
        raise RuntimeError(f"{domain} internal seal state universe mismatch")


ESCONV_AUXILIARY_ABSOLUTE_LABEL_INSTRUMENT_NOT_SUPPORTED = (
    "ESCONV_AUXILIARY_ABSOLUTE_LABEL_INSTRUMENT_NOT_SUPPORTED"
)
ESCONV_AUXILIARY_PAIRWISE_INSTRUMENT_NOT_SUPPORTED = (
    "ESCONV_AUXILIARY_PAIRWISE_INSTRUMENT_NOT_SUPPORTED"
)


def _require_content_addressed_instrument_freeze_contract(
    contract_path: Path,
    *,
    blocking_status: str,
    contract_label: str,
) -> dict:
    """Fail closed unless a git-tracked instrument-freeze contract is present,

    self-consistent, and does not carry ``blocking_status``.

    Both the ESConv-auxiliary absolute-label instrument (per-dimension
    scoring, data/pm_v1_5_contracts/esconv_auxiliary_absolute_instrument_
    freeze_v1.json) and the pairwise instrument (anonymous A/B preference,
    data/pm_v1_5_contracts/esconv_auxiliary_pairwise_instrument_freeze_v1.
    json) have independently been found NOT_SUPPORTED by their own zero-/
    low-cost measurement studies. Unlike an earlier, more permissive version
    of this check, a *missing* contract is now itself a fail-closed
    condition -- dual-domain training may not silently proceed just because
    a freeze record has not been produced yet -- and each contract's own
    ``contract_sha256`` self-hash is re-verified against its current on-disk
    content (the same self-addressed-hash convention artifact attestations
    use) so a hand-edited or truncated contract is also rejected.
    """

    if not contract_path.is_file():
        raise RuntimeError(
            f"{contract_label} freeze contract is missing: {contract_path}; "
            "dual-domain training requires this contract to be present"
        )
    contract = read_json(contract_path)
    expected_self_hash = contract.get("contract_sha256")
    without_self_hash = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }
    if expected_self_hash != sha256_text(canonical_json(without_self_hash)):
        raise RuntimeError(
            f"{contract_label} freeze contract hash mismatch: {contract_path}"
        )
    if contract.get("status") == blocking_status:
        raise RuntimeError(
            f"{contract_label} is frozen {blocking_status} ({contract_path}); "
            "refusing to use the existing ESConv-auxiliary labels for "
            "dual-domain training"
        )
    return contract


def _require_frozen_esconv_auxiliary_measurement_contract(
    attestation_path: Path, *, expected_train_labels_path: Path
) -> dict:
    """Fail closed unless the frozen ESConv-auxiliary measurement contract holds.

    Requires that the train-split judging attestation
    (``13c_judge_esconv_auxiliary_v1_5.py``'s ``measurement_contract_record``)
    exists, attests exactly the train labels this preflight is about to
    consume, and that the code defining applicability / conservative-utility /
    per-dimension clamp ranges has not changed since that contract was
    frozen. Per the explicit freeze rule, none of these may be tuned from
    calibration/internal-test results once train-split judging attests them.
    """

    require_artifact_attestation(
        attestation_path,
        required_stage="esconv_auxiliary_judging_full_train",
        required_output_paths={"labels": expected_train_labels_path},
    )
    attestation = read_json(attestation_path)
    measurement_contract = (attestation.get("parameters") or {}).get(
        "measurement_contract"
    )
    if not isinstance(measurement_contract, dict) or not measurement_contract:
        raise RuntimeError(
            "ESConv-auxiliary train judging attestation lacks a measurement_contract"
        )
    if measurement_contract.get("conservative_utility_lambda") != 1.0:
        raise RuntimeError(
            "frozen measurement contract lambda is not the fixed 1.0 constant"
        )
    if measurement_contract.get("conservative_utility_lambda_is_fixed_never_tuned") is not True:
        raise RuntimeError(
            "frozen measurement contract does not declare lambda as fixed/never-tuned"
        )
    if measurement_contract.get("response_dimension_clamp_range") != [1.0, 5.0]:
        raise RuntimeError("frozen measurement contract response clamp range mismatch")
    if measurement_contract.get("risk_dimension_clamp_range") != [0.0, 3.0]:
        raise RuntimeError("frozen measurement contract risk clamp range mismatch")
    code_manifest = measurement_contract.get("code_manifest")
    if not isinstance(code_manifest, dict) or not code_manifest:
        raise RuntimeError("frozen measurement contract lacks a code_manifest")
    drifted = [
        name
        for name, record in sorted(code_manifest.items())
        if sha256_file(ROOT / str(record["relative_path"])) != record["sha256"]
    ]
    if drifted:
        raise RuntimeError(
            "code defining the frozen measurement contract has changed since "
            f"the ESConv-auxiliary train judging attestation: {drifted}"
        )
    return measurement_contract


def _require_auxiliary_build_report(report_path: Path, auxiliary_dir: Path) -> dict:
    report = read_json(report_path)
    if (
        report.get("protocol")
        != "pm-v1.5-esconv-auxiliary-bank-disjoint-seed-training-support-v1"
        or report.get("status") != "COMPLETE"
        or report.get("bank_disjoint") is not True
        or int(report.get("strategy_bank_source_dialogue_overlap_count") or -1) != 0
    ):
        raise RuntimeError("ESConv auxiliary build report is absent or invalid")
    for split in ("train", "calibration", "internal_test"):
        state_path = auxiliary_dir / split / "pm_v2_states.jsonl"
        record = (((report.get("outputs") or {}).get(split) or {}).get("pm_v2_states") or {})
        if record.get("sha256") != sha256_file(state_path):
            raise RuntimeError(
                f"ESConv auxiliary build report state hash mismatch for {split}"
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pm-v1-5-config",
        type=Path,
        default=ROOT / "configs" / "pm_v1_5.yaml",
    )
    parser.add_argument(
        "--longitudinal-states",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "pm_v2_states.jsonl",
    )
    parser.add_argument("--longitudinal-train-calibration-labels", type=Path, required=True)
    parser.add_argument("--longitudinal-internal-test-labels", type=Path, required=True)
    parser.add_argument(
        "--auxiliary-dir",
        type=Path,
        default=ROOT / "data" / "esconv_auxiliary_v1_5",
    )
    parser.add_argument("--auxiliary-train-labels", type=Path, required=True)
    parser.add_argument("--auxiliary-calibration-labels", type=Path, required=True)
    parser.add_argument("--auxiliary-internal-test-labels", type=Path, required=True)
    parser.add_argument(
        "--esconv-auxiliary-judging-train-attestation",
        type=Path,
        required=True,
        help=(
            "attestation.json written by 13c_judge_esconv_auxiliary_v1_5.py "
            "for the full-scope train split. Binds and re-verifies the frozen "
            "applicability + conservative-utility measurement contract "
            "(dimension applicability, lambda=1.0, per-dimension clamp "
            "ranges, applicable risk dimensions, code hashes) before "
            "dual-domain training may proceed; fails closed if missing or if "
            "the contract-defining code has drifted since the freeze."
        ),
    )
    parser.add_argument(
        "--esconv-auxiliary-absolute-instrument-freeze-contract",
        type=Path,
        default=ROOT
        / "data"
        / "pm_v1_5_contracts"
        / "esconv_auxiliary_absolute_instrument_freeze_v1.json",
        help=(
            "Git-tracked contract recording whether the ESConv-auxiliary "
            "absolute (per-dimension) label instrument has been frozen "
            "ESCONV_AUXILIARY_ABSOLUTE_LABEL_INSTRUMENT_NOT_SUPPORTED. Must "
            "be present, self-hash-consistent, and not carry that status."
        ),
    )
    parser.add_argument(
        "--esconv-auxiliary-pairwise-instrument-freeze-contract",
        type=Path,
        default=ROOT
        / "data"
        / "pm_v1_5_contracts"
        / "esconv_auxiliary_pairwise_instrument_freeze_v1.json",
        help=(
            "Git-tracked contract recording whether the ESConv-auxiliary "
            "pairwise (anonymous A/B preference) instrument has been frozen "
            "ESCONV_AUXILIARY_PAIRWISE_INSTRUMENT_NOT_SUPPORTED. Must be "
            "present, self-hash-consistent, and not carry that status."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_dual_domain_training_preflight",
    )
    args = parser.parse_args()

    _require_content_addressed_instrument_freeze_contract(
        args.esconv_auxiliary_absolute_instrument_freeze_contract,
        blocking_status=ESCONV_AUXILIARY_ABSOLUTE_LABEL_INSTRUMENT_NOT_SUPPORTED,
        contract_label="ESConv-auxiliary absolute-label instrument freeze",
    )
    _require_content_addressed_instrument_freeze_contract(
        args.esconv_auxiliary_pairwise_instrument_freeze_contract,
        blocking_status=ESCONV_AUXILIARY_PAIRWISE_INSTRUMENT_NOT_SUPPORTED,
        contract_label="ESConv-auxiliary pairwise instrument freeze",
    )

    config = load_config(args.pm_v1_5_config)
    if config.get("version") != "pm-v1.5":
        raise RuntimeError("dual-domain preflight requires pm-v1.5")
    auxiliary_build_report_path = args.auxiliary_dir / "build_report.json"
    auxiliary_build_report = _require_auxiliary_build_report(
        auxiliary_build_report_path, args.auxiliary_dir
    )
    frozen_measurement_contract = _require_frozen_esconv_auxiliary_measurement_contract(
        args.esconv_auxiliary_judging_train_attestation,
        expected_train_labels_path=args.auxiliary_train_labels,
    )
    longitudinal_states = load_states(args.longitudinal_states)
    auxiliary_state_paths = {
        split: args.auxiliary_dir / split / "pm_v2_states.jsonl"
        for split in ("train", "calibration", "internal_test")
    }
    auxiliary_states = [
        state
        for split in ("train", "calibration", "internal_test")
        for state in load_states(auxiliary_state_paths[split])
    ]
    longitudinal_labels = [
        ActionLabel.model_validate(row)
        for row in iter_jsonl(args.longitudinal_train_calibration_labels)
    ]
    auxiliary_label_paths = {
        "train": args.auxiliary_train_labels,
        "calibration": args.auxiliary_calibration_labels,
    }
    auxiliary_labels = [
        ActionLabel.model_validate(row)
        for split in ("train", "calibration")
        for row in iter_jsonl(auxiliary_label_paths[split])
    ]
    validated = validate_dual_domain_training_inputs(
        longitudinal_states=longitudinal_states,
        auxiliary_states=auxiliary_states,
        longitudinal_train_calibration_labels=longitudinal_labels,
        auxiliary_train_calibration_labels=auxiliary_labels,
        pm_config=config,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    longitudinal_seal_path = args.out_dir / "sealed_internal_longitudinal.json"
    auxiliary_seal_path = args.out_dir / "sealed_internal_esconv_auxiliary.json"
    longitudinal_seal = seal_internal_label_bundle(
        longitudinal_seal_path,
        internal_labels_path=args.longitudinal_internal_test_labels,
    )
    auxiliary_seal = seal_internal_label_bundle(
        auxiliary_seal_path,
        internal_labels_path=args.auxiliary_internal_test_labels,
    )
    _require_internal_seal_matches_states(
        longitudinal_seal, longitudinal_states, domain="longitudinal_synthetic"
    )
    _require_internal_seal_matches_states(
        auxiliary_seal, auxiliary_states, domain="esconv_auxiliary"
    )

    report_path = args.out_dir / "dual_domain_training_preflight.json"
    report = {
        **validated.report,
        "protocol": DUAL_DOMAIN_TRAINING_PROTOCOL,
        "status": "PASS",
        "pm_v1_5_config_sha256": sha256_file(args.pm_v1_5_config),
        "auxiliary_build_report_sha256": sha256_file(
            auxiliary_build_report_path
        ),
        "auxiliary_build_report_status": auxiliary_build_report["status"],
        "sealed_internal_bundles": {
            "longitudinal_synthetic": longitudinal_seal,
            "esconv_auxiliary": auxiliary_seal,
        },
        "internal_label_values_deserialized": False,
        "frozen_esconv_auxiliary_measurement_contract": frozen_measurement_contract,
    }
    write_json(report_path, report)
    attestation_path = args.out_dir / "artifact_attestation.json"
    create_artifact_attestation(
        attestation_path,
        stage="pm_v1_5_dual_domain_training_preflight",
        inputs={
            "pm_v1_5_config": args.pm_v1_5_config,
            "longitudinal_states": args.longitudinal_states,
            "longitudinal_train_calibration_labels": (
                args.longitudinal_train_calibration_labels
            ),
            "longitudinal_internal_test_labels": args.longitudinal_internal_test_labels,
            "auxiliary_build_report": auxiliary_build_report_path,
            "esconv_auxiliary_judging_train_attestation": (
                args.esconv_auxiliary_judging_train_attestation
            ),
            **{
                f"auxiliary_{split}_states": path
                for split, path in auxiliary_state_paths.items()
            },
            **{
                f"auxiliary_{split}_labels": path
                for split, path in {
                    **auxiliary_label_paths,
                    "internal_test": args.auxiliary_internal_test_labels,
                }.items()
            },
        },
        outputs={
            "report": (report_path, False),
            "sealed_internal_longitudinal": (longitudinal_seal_path, False),
            "sealed_internal_esconv_auxiliary": (auxiliary_seal_path, False),
        },
        parameters={
            "protocol": DUAL_DOMAIN_TRAINING_PROTOCOL,
            "top_level_domain_weight": validated.report[
                "top_level_domain_weight"
            ],
            "internal_label_values_deserialized": False,
            "frozen_esconv_auxiliary_measurement_contract": frozen_measurement_contract,
        },
    )
    print(
        canonical_json(
            {
                "status": "PASS",
                "report": str(report_path),
                "attestation": str(attestation_path),
            }
        )
    )


if __name__ == "__main__":
    main()
