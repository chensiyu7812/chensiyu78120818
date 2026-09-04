"""Verify the small PM-v1.5 forced-swap judge-sensitivity canary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import require_artifact_attestation
from .io import canonical_json, read_json, sha256_text
from .pm_v2_forced_swap import forced_swap_unit_id, select_forced_swap_units


PROTOCOL = "pm-v1.5-forced-swap-canary-v1"

# Single source of truth for the disposition PM-v1.5 takes toward the two
# distinct "key claim" mechanisms in play, so configs/pm_v1_5.yaml,
# scripts/v1_5_create_freeze.py, and scripts/v1_5/25_eval_pm_v2_external_v1_5.py
# cannot silently drift from one another: the shared PM-v2.2 forced-swap
# *efficacy* verification is deliberately not run (efficacy comes from
# v1_5_external_claims.assess_external_claims's frozen paired CIs instead);
# what IS run is this module's small judge-sensitivity/order-robustness
# canary, which only proves a judge can detect a swapped response, not that
# PM is better than the baseline.
KEY_CLAIM_GATE: dict[str, Any] = {
    "protocol": "pm-v1.5-key-claim-gate-v1",
    "require_v1_5_forced_swap_canary": True,
    "canary_role": "judge_sensitivity_not_pm_efficacy",
    "efficacy_decision": "frozen_paired_quality_and_cost_ci",
    "shared_pmv22_key_claim_verification": False,
}


def require_v1_5_forced_swap_canary(
    summary_path: str | Path,
    attestation_path: str | Path,
    *,
    study_freeze_sha256: str,
    full_expected_units: Sequence[tuple[str, int, int, str, int]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if contract.get("protocol") != PROTOCOL:
        raise RuntimeError("study freeze lacks the PM-v1.5 forced-swap canary")
    verification = require_artifact_attestation(
        attestation_path,
        required_stage="pm_v2_forced_swap_key_claim",
        required_output_paths={"summary": summary_path},
        expected_freeze_sha256=study_freeze_sha256,
    )
    selected = select_forced_swap_units(
        full_expected_units,
        sample_units=int(contract["sample_units"]),
        seed=int(contract["judge_seed"]),
    )
    selected_sha256 = sha256_text(canonical_json(selected))
    excluded_ids = [forced_swap_unit_id(unit) for unit in selected]
    summary = read_json(summary_path)
    compatibility = summary.get("compatibility_gate") or {}
    checks = compatibility.get("checks") or {}
    expected = {
        "status": "PASS",
        "study_freeze_sha256": study_freeze_sha256,
        "treatment": contract["treatment"],
        "baseline": contract["baseline"],
        "sample_units": int(contract["sample_units"]),
        "sample_units_sha256": selected_sha256,
        "excluded_unit_ids": excluded_ids,
        "order_variants": [0, 1],
        "judge_seed": int(contract["judge_seed"]),
        "full_expected_units_sha256": sha256_text(
            canonical_json(full_expected_units)
        ),
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"PM-v1.5 forced-swap canary mismatch: {key}")
    if compatibility.get("status") != "PASS" or not checks or not all(
        value is True for value in checks.values()
    ):
        raise RuntimeError("PM-v1.5 forced-swap judge-sensitivity canary did not PASS")
    attestation = read_json(attestation_path)
    parameters = attestation.get("parameters") or {}
    if parameters.get("compatibility_gate") != compatibility:
        raise RuntimeError("forced-swap canary attestation/summary gate mismatch")
    return {
        "status": "PASS",
        "protocol": PROTOCOL,
        "attestation_sha256": verification["attestation_sha256"],
        "excluded_units": selected,
        "excluded_unit_ids": excluded_ids,
        "sample_units_sha256": selected_sha256,
        "efficacy_is_not_required_from_canary": True,
    }
