"""Zero-API readiness checks for promoting fixed-seeker V3 to formal use."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .fixed_seeker_contract import (
    FIXED_SEEKER_GENERATION_CONTRACT_VERSION_V3,
)


FIXED_SEEKER_V3_PROMOTION_AUDIT_PROTOCOL = (
    "pm-v1.5-fixed-seeker-v3-formal-promotion-readiness-v1"
)


def assess_fixed_seeker_v3_promotion(
    *,
    fixed_seeker_sidecar_contract: Mapping[str, Any],
    pilot_summary: Mapping[str, Any],
    pilot_attestation: Mapping[str, Any],
    consumer_source_text: Mapping[str, str],
    formal_bundle_exists: bool,
) -> dict[str, Any]:
    """Report blockers without changing the config, freeze, or any artifact.

    configs/pm_v1_5.yaml deliberately stays on the historical V2 fixed-seeker
    treatment (editing it directly was shown to invalidate the already-
    qualified V8.19.2 lineage via a pm_v1_5_config hash mismatch). V3 is
    promoted through the separately-tracked configs/pm_v1_5_fixed_seeker_v3.
    json sidecar instead, so readiness checks that sidecar's own declared
    version, not the global config.
    """

    blockers: list[str] = []
    if (
        fixed_seeker_sidecar_contract.get("version")
        != FIXED_SEEKER_GENERATION_CONTRACT_VERSION_V3
    ):
        blockers.append("fixed_seeker_v3_sidecar_contract_is_not_v3")
    if (
        pilot_summary.get("status") != "COMPLETE"
        or int(pilot_summary.get("completed_tracks") or 0) != 2
        or int(pilot_summary.get("expected_tracks") or 0) != 2
        or int(pilot_summary.get("mid_sentence_truncation_count") or 0) != 0
    ):
        blockers.append("v3_compatibility_pilot_is_not_a_clean_2_track_pass")
    if pilot_attestation.get("stage") != "evoemo_fixed_seeker_tracks_v23_bounded_surface":
        blockers.append("v3_compatibility_pilot_attestation_stage_mismatch")

    required_v3_token = "FIXED_SEEKER_V23_STAGE"
    for consumer, source in sorted(consumer_source_text.items()):
        if required_v3_token not in source:
            blockers.append(f"{consumer}_does_not_accept_v3_attestation_stage")
        if (
            'parent.name != "evoemo_fixed_tracks_v1_5"' in source
            or "required_stage=FIXED_SEEKER_V22_STAGE" in source
        ):
            blockers.append(f"{consumer}_still_hard_codes_v2_bundle_contract")
        if 'parent.name != "evoemo_fixed_tracks_v1_5_v3_formal_candidate"' in source:
            blockers.append(
                f"{consumer}_still_hard_codes_formal_bundle_directory_basename"
            )
    if not formal_bundle_exists:
        blockers.append("formal_102_track_v3_bundle_not_generated")

    return {
        "protocol": FIXED_SEEKER_V3_PROMOTION_AUDIT_PROTOCOL,
        "status": "READY" if not blockers else "BLOCKED",
        "zero_api": True,
        "pilot_passed": not any("pilot" in blocker for blocker in blockers),
        "formal_bundle_exists": formal_bundle_exists,
        "blockers": blockers,
        "required_next_order": [
            "atomically_promote_config_and_all_fixed_track_consumers_to_v3",
            "run_zero_api_contract_tests_and_formal_dry_run_twice",
            "request_separate_paid_approval_for_102_track_formal_generation",
            "create_study_freeze_only_from_complete_attested_v3_formal_bundle",
        ],
    }


def load_consumer_sources(root: Path) -> dict[str, str]:
    paths = {
        "study_freeze": root / "scripts" / "v1_5_create_freeze.py",
        "evoemo_generation": root
        / "scripts"
        / "v1_5"
        / "24_run_pm_v2_evoemo_v1_5.py",
        "reference_baselines": root
        / "scripts"
        / "v1_5"
        / "24a_run_pmv22_reference_baselines_v1_5.py",
        "shared_evoemo_runner": root / "src" / "metacom_pm" / "pm_v2_evoemo.py",
    }
    return {
        name: path.read_text(encoding="utf-8") for name, path in sorted(paths.items())
    }
