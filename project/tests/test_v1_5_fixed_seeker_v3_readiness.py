from __future__ import annotations

from pathlib import Path

import pytest

from metacom_pm import fixed_seeker_contract as fixed_seeker_contract_module
from metacom_pm.fixed_seeker_contract import (
    FIXED_SEEKER_V3_FORMAL_BUNDLE_BINDING_PROTOCOL,
    require_fixed_seeker_v3_formal_bundle,
    require_fixed_seeker_v3_sidecar_contract,
)
from metacom_pm.io import canonical_json, read_json, sha256_file, sha256_text, write_json
from metacom_pm.v1_5_fixed_seeker_readiness import (
    assess_fixed_seeker_v3_promotion,
    load_consumer_sources,
)

ROOT = Path(__file__).resolve().parents[1]
SIDECAR_PATH = ROOT / "configs" / "pm_v1_5_fixed_seeker_v3.json"
FORMAL_BUNDLE_V3_STAGE = "evoemo_fixed_seeker_tracks_v23_bounded_surface"
FORMAL_BUNDLE_V2_STAGE = "evoemo_fixed_seeker_tracks_v22"


def _minimal_formal_bundle(
    tmp_path: Path, *, stage: str = FORMAL_BUNDLE_V3_STAGE
) -> Path:
    """A minimal, self-contained (not content-addressed-attestation-valid)

    bundle directory: enough for require_fixed_seeker_v3_formal_bundle's own
    independent checks, which never delegate to
    require_content_addressed_attestation.
    """

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "fixed_seeker_tracks.jsonl").write_text(
        '{"track_id": "t1"}\n', encoding="utf-8"
    )
    write_json(bundle_dir / "artifact_attestation.json", {"stage": stage})
    write_json(
        bundle_dir / "summary.json",
        {
            "expected_tracks": 1,
            "completed_tracks": 1,
            "expected_logical_calls": 1,
            "successful_logical_calls": 1,
        },
    )
    return bundle_dir


def _write_formal_bundle_binding(
    binding_path: Path,
    *,
    bundle_dir: Path,
    stage: str = FORMAL_BUNDLE_V3_STAGE,
    protocol: str = FIXED_SEEKER_V3_FORMAL_BUNDLE_BINDING_PROTOCOL,
) -> None:
    payload = {
        "protocol": protocol,
        "stage": stage,
        "output_directory": str(bundle_dir),
        "approval_identity": "test-identity",
        "sidecar_sha256": "0" * 64,
        "call_plan_sha256": "0" * 64,
        "expected_tracks": 1,
        "expected_logical_calls": 1,
        "fixed_seeker_tracks_sha256": sha256_file(
            bundle_dir / "fixed_seeker_tracks.jsonl"
        ),
        "artifact_attestation_sha256": sha256_file(
            bundle_dir / "artifact_attestation.json"
        ),
    }
    payload["binding_sha256"] = sha256_text(canonical_json(payload))
    write_json(binding_path, payload)


def _summary() -> dict:
    return {
        "status": "COMPLETE",
        "completed_tracks": 2,
        "expected_tracks": 2,
        "mid_sentence_truncation_count": 0,
    }


def _attestation() -> dict:
    return {"stage": "evoemo_fixed_seeker_tracks_v23_bounded_surface"}


def test_current_style_v2_consumers_block_v3_promotion() -> None:
    report = assess_fixed_seeker_v3_promotion(
        fixed_seeker_sidecar_contract={
            "version": "pm-v2.2-fixed-seeker-generation-v2"
        },
        pilot_summary=_summary(),
        pilot_attestation=_attestation(),
        consumer_source_text={
            "freeze": "required_stage=FIXED_SEEKER_V22_STAGE",
            "external": 'parent.name != "evoemo_fixed_tracks_v1_5"',
        },
        formal_bundle_exists=False,
    )
    assert report["status"] == "BLOCKED"
    assert "fixed_seeker_v3_sidecar_contract_is_not_v3" in report["blockers"]
    assert "formal_102_track_v3_bundle_not_generated" in report["blockers"]


def test_ready_requires_v3_sidecar_consumers_and_formal_bundle() -> None:
    report = assess_fixed_seeker_v3_promotion(
        fixed_seeker_sidecar_contract={
            "version": "pm-v2.2-fixed-seeker-generation-v3-bounded-surface"
        },
        pilot_summary=_summary(),
        pilot_attestation=_attestation(),
        consumer_source_text={
            "freeze": "required_stage = FIXED_SEEKER_V23_STAGE",
            "external": "accepted stage: FIXED_SEEKER_V23_STAGE",
        },
        formal_bundle_exists=True,
    )
    assert report["status"] == "READY"
    assert report["blockers"] == []


def test_real_sidecar_and_consumers_clear_every_blocker_but_the_formal_bundle() -> None:
    """End-to-end confirmation of the real atomic migration (not a synthetic

    fixture): the real, git-tracked configs/pm_v1_5_fixed_seeker_v3.json
    sidecar and all four real consumer files (scripts/v1_5_create_freeze.py,
    scripts/v1_5/24_run_pm_v2_evoemo_v1_5.py,
    scripts/v1_5/24a_run_pmv22_reference_baselines_v1_5.py,
    src/metacom_pm/pm_v2_evoemo.py) must now report zero blockers except the
    formal 102-track bundle, which this migration deliberately does not
    generate (that step requires a separately-approved paid run).
    configs/pm_v1_5.yaml itself is untouched -- it stays on the historical V2
    treatment, and is not read by this check at all.
    """

    sidecar_contract = require_fixed_seeker_v3_sidecar_contract(SIDECAR_PATH)
    report = assess_fixed_seeker_v3_promotion(
        fixed_seeker_sidecar_contract=sidecar_contract.payload(),
        # The paid pilot directory is intentionally private/local.  This test
        # checks the real tracked sidecar and real consumer wiring with the
        # minimal already-validated PASS surface rather than requiring raw
        # provider artifacts in a clean Git checkout.
        pilot_summary=_summary(),
        pilot_attestation=_attestation(),
        consumer_source_text=load_consumer_sources(ROOT),
        formal_bundle_exists=False,
    )
    assert report["status"] == "BLOCKED"
    assert report["pilot_passed"] is True
    assert report["blockers"] == ["formal_102_track_v3_bundle_not_generated"]


def test_unclean_pilot_is_rejected() -> None:
    summary = _summary()
    summary["mid_sentence_truncation_count"] = 1
    report = assess_fixed_seeker_v3_promotion(
        fixed_seeker_sidecar_contract={
            "version": "pm-v2.2-fixed-seeker-generation-v3-bounded-surface"
        },
        pilot_summary=summary,
        pilot_attestation=_attestation(),
        consumer_source_text={"consumer": "FIXED_SEEKER_V23_STAGE"},
        formal_bundle_exists=True,
    )
    assert report["status"] == "BLOCKED"


def test_require_fixed_seeker_v3_sidecar_contract_accepts_the_real_frozen_file() -> None:
    contract = require_fixed_seeker_v3_sidecar_contract(SIDECAR_PATH)
    assert contract.version == "pm-v2.2-fixed-seeker-generation-v3-bounded-surface"


def test_require_fixed_seeker_v3_sidecar_contract_fails_closed_on_hash_mismatch(
    tmp_path,
):
    tampered = tmp_path / "pm_v1_5_fixed_seeker_v3.json"
    payload = read_json(SIDECAR_PATH)
    payload["temperature"] = 0.9  # any deviation from the frozen, reviewed file
    tampered.write_text(canonical_json(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        require_fixed_seeker_v3_sidecar_contract(tampered)


def test_require_fixed_seeker_v3_sidecar_contract_fails_closed_on_non_v3_version(
    tmp_path, monkeypatch
):
    downgraded = tmp_path / "pm_v1_5_fixed_seeker_v2.json"
    downgraded.write_text(
        canonical_json(
            {
                "version": "pm-v2.2-fixed-seeker-generation-v2",
                "seeker_endpoint": "seeker",
                "system_prompt_id": "evoemo-fixed-seeker-v1",
                "system_prompt_template_sha256": (
                    "a9138eaf46b46e8f10c89774a9ef51189b5807391a0882dff58c287a4c54eeac"
                ),
                "response_instruction_token_limit": 60,
                "temperature": 0.2,
                "max_output_tokens": 300,
                "output_normalization": "normalize_space_v1",
                "finish_reason_protocol": "pm-v2-finish-reason-v1",
                "accepted_normalized_finish_reasons": ["complete"],
                "maximum_physical_attempts_per_logical_call": 10,
                "seed_protocol": "base-seed-plus-turn-index-v1",
                "elicitation_scaffold_protocol": "deterministic-generic-open-loop-v1",
            }
        ),
        encoding="utf-8",
    )
    # The hash gate would also (correctly) reject this file; monkeypatch the
    # expected hash to isolate and confirm the *version* check's own message.
    monkeypatch.setattr(
        fixed_seeker_contract_module,
        "FIXED_SEEKER_V3_SIDECAR_CONTRACT_SHA256",
        sha256_file(downgraded),
    )
    with pytest.raises(RuntimeError, match="is not V3"):
        fixed_seeker_contract_module.require_fixed_seeker_v3_sidecar_contract(
            downgraded
        )


def test_require_fixed_seeker_v3_formal_bundle_accepts_the_real_bundle(
    tmp_path,
) -> None:
    bundle_dir = _minimal_formal_bundle(tmp_path)
    binding_path = tmp_path / "binding.json"
    _write_formal_bundle_binding(binding_path, bundle_dir=bundle_dir)
    result = require_fixed_seeker_v3_formal_bundle(
        binding_path, root=tmp_path, required_stage=FORMAL_BUNDLE_V3_STAGE
    )
    assert result["output_directory"] == bundle_dir.resolve()
    assert result["fixed_tracks_path"] == bundle_dir / "fixed_seeker_tracks.jsonl"
    assert (
        result["artifact_attestation_path"]
        == bundle_dir / "artifact_attestation.json"
    )


def test_require_fixed_seeker_v3_formal_bundle_fails_closed_on_binding_self_hash_tamper(
    tmp_path,
) -> None:
    bundle_dir = _minimal_formal_bundle(tmp_path)
    binding_path = tmp_path / "binding.json"
    _write_formal_bundle_binding(binding_path, bundle_dir=bundle_dir)
    payload = read_json(binding_path)
    payload["approval_identity"] = "tampered-after-the-fact"  # binding_sha256 now stale
    write_json(binding_path, payload)
    with pytest.raises(RuntimeError, match="binding hash mismatch"):
        require_fixed_seeker_v3_formal_bundle(
            binding_path, root=tmp_path, required_stage=FORMAL_BUNDLE_V3_STAGE
        )


def test_require_fixed_seeker_v3_formal_bundle_fails_closed_on_wrong_tracks_hash(
    tmp_path,
) -> None:
    """A forged same-name directory: the tracks file's real bytes changed

    after the binding was created (whether by tampering or by someone
    recreating a directory with the same path/name but different content) --
    the recorded hash no longer matches, so it must fail closed rather than
    trust the directory name/path alone.
    """

    bundle_dir = _minimal_formal_bundle(tmp_path)
    binding_path = tmp_path / "binding.json"
    _write_formal_bundle_binding(binding_path, bundle_dir=bundle_dir)
    (bundle_dir / "fixed_seeker_tracks.jsonl").write_text(
        '{"track_id": "forged"}\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="tracks file hash mismatch"):
        require_fixed_seeker_v3_formal_bundle(
            binding_path, root=tmp_path, required_stage=FORMAL_BUNDLE_V3_STAGE
        )


def test_require_fixed_seeker_v3_formal_bundle_fails_closed_on_wrong_attestation_hash(
    tmp_path,
) -> None:
    bundle_dir = _minimal_formal_bundle(tmp_path)
    binding_path = tmp_path / "binding.json"
    _write_formal_bundle_binding(binding_path, bundle_dir=bundle_dir)
    write_json(
        bundle_dir / "artifact_attestation.json",
        {"stage": FORMAL_BUNDLE_V3_STAGE, "forged": True},
    )
    with pytest.raises(RuntimeError, match="attestation file hash mismatch"):
        require_fixed_seeker_v3_formal_bundle(
            binding_path, root=tmp_path, required_stage=FORMAL_BUNDLE_V3_STAGE
        )


def test_require_fixed_seeker_v3_formal_bundle_fails_closed_on_wrong_attestation_stage(
    tmp_path,
) -> None:
    """The binding's own stage field is V3, but the real attestation file on

    disk (re-hashed and re-bound, e.g. by a careless regeneration) actually
    records the historical V2 stage -- content-addressing alone is not
    enough; the stage recorded inside the attestation must also match.
    """

    bundle_dir = _minimal_formal_bundle(tmp_path, stage=FORMAL_BUNDLE_V2_STAGE)
    binding_path = tmp_path / "binding.json"
    _write_formal_bundle_binding(
        binding_path, bundle_dir=bundle_dir, stage=FORMAL_BUNDLE_V3_STAGE
    )
    with pytest.raises(RuntimeError, match="attestation stage mismatch"):
        require_fixed_seeker_v3_formal_bundle(
            binding_path, root=tmp_path, required_stage=FORMAL_BUNDLE_V3_STAGE
        )


def test_require_fixed_seeker_v3_formal_bundle_fails_closed_on_old_v2_binding(
    tmp_path,
) -> None:
    """The binding itself honestly declares an old V2 bundle: callers always

    pass required_stage=FIXED_SEEKER_V23_STAGE, so a V2-declared binding must
    never be silently accepted as formal-V3-ready.
    """

    bundle_dir = _minimal_formal_bundle(tmp_path, stage=FORMAL_BUNDLE_V2_STAGE)
    binding_path = tmp_path / "binding.json"
    _write_formal_bundle_binding(
        binding_path, bundle_dir=bundle_dir, stage=FORMAL_BUNDLE_V2_STAGE
    )
    with pytest.raises(RuntimeError, match="binding stage does not match"):
        require_fixed_seeker_v3_formal_bundle(
            binding_path, root=tmp_path, required_stage=FORMAL_BUNDLE_V3_STAGE
        )


def test_require_fixed_seeker_v3_formal_bundle_fails_closed_on_wrong_protocol(
    tmp_path,
) -> None:
    bundle_dir = _minimal_formal_bundle(tmp_path)
    binding_path = tmp_path / "binding.json"
    _write_formal_bundle_binding(
        binding_path, bundle_dir=bundle_dir, protocol="some-other-protocol-v1"
    )
    with pytest.raises(RuntimeError, match="binding protocol"):
        require_fixed_seeker_v3_formal_bundle(
            binding_path, root=tmp_path, required_stage=FORMAL_BUNDLE_V3_STAGE
        )


def test_require_fixed_seeker_v3_formal_bundle_fails_closed_on_wrong_track_counts(
    tmp_path,
) -> None:
    bundle_dir = _minimal_formal_bundle(tmp_path)
    write_json(
        bundle_dir / "summary.json",
        {
            "expected_tracks": 1,
            "completed_tracks": 0,
            "expected_logical_calls": 1,
            "successful_logical_calls": 0,
        },
    )
    binding_path = tmp_path / "binding.json"
    _write_formal_bundle_binding(binding_path, bundle_dir=bundle_dir)
    with pytest.raises(RuntimeError, match="track/call counts do not match"):
        require_fixed_seeker_v3_formal_bundle(
            binding_path, root=tmp_path, required_stage=FORMAL_BUNDLE_V3_STAGE
        )


def test_real_fixed_seeker_v3_formal_bundle_binding_is_well_formed() -> None:
    """The real, tracked formal-bundle binding must at least be internally

    well-formed (self-hash intact, correct protocol/stage/track counts).
    The referenced output directory is a local, untracked paid-run artifact
    (outputs/ is never committed) and is intentionally not required to exist
    in every checkout, so this does not call
    require_fixed_seeker_v3_formal_bundle end-to-end.
    """

    binding_path = (
        ROOT / "data" / "pm_v1_5_contracts" / "fixed_seeker_v3_formal_bundle_v1.json"
    )
    binding = read_json(binding_path)
    payload = {key: value for key, value in binding.items() if key != "binding_sha256"}
    assert binding["binding_sha256"] == sha256_text(canonical_json(payload))
    assert binding["protocol"] == FIXED_SEEKER_V3_FORMAL_BUNDLE_BINDING_PROTOCOL
    assert binding["stage"] == FORMAL_BUNDLE_V3_STAGE
    assert binding["expected_tracks"] == 102
    assert binding["expected_logical_calls"] == 1020
