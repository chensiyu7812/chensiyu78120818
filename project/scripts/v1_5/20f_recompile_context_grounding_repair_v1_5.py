#!/usr/bin/env python3
"""Canonically recompile the 52-user PM-v1.5 development corpus with the
frozen 25-state context-grounding repair overlay applied.

Zero API calls. Takes the original v8_18 pm_v2_bundles.jsonl, patches
exactly the 25 frozen DATA_DEFECT cases via apply_repair_overlay_to_bundles
(fail-closed: classification_sha256 verification, duplicate rejection,
exact-25-state-set enforcement, cross-checked case_id/user_id/repair_mode,
turn-index constraints), then recompiles the WHOLE corpus via the same
write_development_dataset used by the original formal generation -- so
embeddings/inventory/step0/provenance for the 25 repaired states are
recomputed consistently rather than left stale. Writes to a brand-new
output directory; never touches the original.

After recompiling, this script performs its own independent verification:
every one of the 443 untouched states must be byte-for-byte identical to
the original (field-by-field, not just "looks similar"), and every one of
the 25 repaired states must actually differ.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.config import load_config
from metacom_pm.contracts import StrategyCard
from metacom_pm.io import (
    dict_field_diff,
    iter_jsonl,
    read_json,
    sha256_file,
    write_json,
)
from metacom_pm.pm_v1_5_semantic import (
    FrozenTransformerSemanticEncoder,
    require_semantic_runtime_contract,
    semantic_encoder_spec_from_config,
)
from metacom_pm.pm_v2_contracts import PMV2Split
from metacom_pm.pm_v2_data import load_bundles, load_states
from metacom_pm.pm_v2_generation_pilot import (
    CALIBRATION_SEMANTIC_FAMILIES,
    INTERNAL_TEST_SEMANTIC_FAMILIES,
    TRAIN_SEMANTIC_FAMILIES,
)
from metacom_pm.v1_5_context_grounding_repair import (
    DEFAULT_CLASSIFICATION_PATH,
    load_context_grounding_defect_classification,
)
from metacom_pm.v1_5_context_grounding_repair_overlay import (
    RepairOverlayRecord,
    apply_repair_overlay_to_bundles,
    recompile_repaired_development_dataset,
    split_by_user_from_existing_states,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORIGINAL_DIR = (
    ROOT / "data" / "pm_v1_5_formal_v8_18_duplicate_repair_candidate"
)


def _require_frozen_strategy_bank(
    pm_config: dict[str, Any],
    *,
    strategy_bank_path: Path,
    selected_seed_sources_path: Path,
) -> list[StrategyCard]:
    contract = dict(pm_config["strategy_bank_contract"])
    if sha256_file(strategy_bank_path) != str(contract["sha256"]):
        raise RuntimeError("frozen V1.5 Strategy Bank hash mismatch")
    if sha256_file(selected_seed_sources_path) != str(
        contract["selected_seed_sources_sha256"]
    ):
        raise RuntimeError("frozen V1.5 selected-seed manifest hash mismatch")
    cards = [
        StrategyCard.model_validate(row) for row in iter_jsonl(strategy_bank_path)
    ]
    if len(cards) != int(contract["card_count"]):
        raise RuntimeError("frozen V1.5 Strategy Bank card count mismatch")
    return cards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pm-v1-5-config", type=Path, default=ROOT / "configs" / "pm_v1_5.yaml"
    )
    parser.add_argument("--original-out-dir", type=Path, default=DEFAULT_ORIGINAL_DIR)
    parser.add_argument(
        "--classification", type=Path, default=Path(DEFAULT_CLASSIFICATION_PATH)
    )
    parser.add_argument("--repair-overlays", type=Path, required=True)
    parser.add_argument(
        "--strategy-bank",
        type=Path,
        default=ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl",
    )
    parser.add_argument(
        "--selected-seed-sources",
        type=Path,
        default=ROOT / "data" / "strategy" / "pm_v1_5_selected_seed_sources.jsonl",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Skip recompilation (which requires loading the real frozen "
            "semantic encoder and re-encoding the full strategy catalog, "
            "the dominant cost) and just re-run verification against an "
            "--out-dir that a prior invocation already wrote."
        ),
    )
    args = parser.parse_args()

    if args.verify_only:
        if not args.out_dir.is_dir() or not (args.out_dir / "pm_v2_states.jsonl").is_file():
            raise RuntimeError(f"--verify-only requires an already-recompiled --out-dir: {args.out_dir}")
    elif args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise RuntimeError(f"refusing to write into a non-empty directory: {args.out_dir}")

    pm_config = load_config(args.pm_v1_5_config)
    if pm_config.get("version") != "pm-v1.5":
        raise RuntimeError("recompilation requires the pm-v1.5 config")

    original_bundles_path = args.original_out_dir / "pm_v2_bundles.jsonl"
    original_states_path = args.original_out_dir / "pm_v2_states.jsonl"
    original_bundles = load_bundles(original_bundles_path)
    original_states = load_states(original_states_path)
    original_bundles_sha256 = sha256_file(original_bundles_path)

    classification_records = load_context_grounding_defect_classification(
        args.classification
    )
    classification_sha256 = sha256_file(args.classification)
    overlays = [
        RepairOverlayRecord.model_validate(row)
        for row in iter_jsonl(args.repair_overlays)
    ]
    if not overlays:
        raise RuntimeError("repair-overlays file is empty")

    generation_cfg = pm_config["data_generation"]
    expected_split_state_counts = {
        PMV2Split.TRAIN.value: int(generation_cfg["train_users"]) * int(generation_cfg["cases_per_user"]),
        PMV2Split.CALIBRATION.value: int(generation_cfg["calibration_users"]) * int(generation_cfg["cases_per_user"]),
        PMV2Split.INTERNAL_TEST.value: int(generation_cfg["internal_test_users"]) * int(generation_cfg["cases_per_user"]),
    }

    if args.verify_only:
        report = read_json(args.out_dir / "pm_v2_data_report.json")
    else:
        repaired_bundles = apply_repair_overlay_to_bundles(
            bundles=original_bundles,
            overlays=overlays,
            classification_records=classification_records,
            expected_classification_sha256=classification_sha256,
        )

        split_by_user = split_by_user_from_existing_states(original_states_path)

        strategy_cards = _require_frozen_strategy_bank(
            pm_config,
            strategy_bank_path=args.strategy_bank,
            selected_seed_sources_path=args.selected_seed_sources,
        )
        strategy_bank_sha256 = sha256_file(args.strategy_bank)
        retrieval_cfg = pm_config["retrieval"]
        external_cfg = pm_config["external_evaluation"]
        strategy_top_k = int(retrieval_cfg["strategy_top_k"])
        if int(external_cfg["strategy_top_k"]) != strategy_top_k:
            raise RuntimeError("development/external strategy_top_k mismatch in PM-v2 config")
        strategy_estimated_tokens = int(external_cfg["strategy_action_tokens"])

        semantic_encoder = FrozenTransformerSemanticEncoder.load(
            semantic_encoder_spec_from_config(pm_config)
        )
        require_semantic_runtime_contract(pm_config, semantic_encoder)

        report = recompile_repaired_development_dataset(
            bundles=repaired_bundles,
            split_by_user=split_by_user,
            out_dir=args.out_dir,
            strategy_catalog_count=len(strategy_cards),
            strategy_estimated_tokens=strategy_estimated_tokens,
            strategy_top_k=strategy_top_k,
            strategy_bank_sha256=strategy_bank_sha256,
            strategy_cards=strategy_cards,
            memory_min_score=float(retrieval_cfg["memory_min_score"]),
            strategy_min_score=float(retrieval_cfg["strategy_min_score"]),
            expected_semantic_families_by_split={
                PMV2Split.TRAIN: TRAIN_SEMANTIC_FAMILIES,
                PMV2Split.CALIBRATION: CALIBRATION_SEMANTIC_FAMILIES,
                PMV2Split.INTERNAL_TEST: INTERNAL_TEST_SEMANTIC_FAMILIES,
            },
            semantic_encoder=semantic_encoder,
        )

    if report["split_counts"] != expected_split_state_counts:
        raise RuntimeError(
            "recompiled PM-v2 state counts do not match the frozen full design: "
            f"expected={expected_split_state_counts}, observed={report['split_counts']}"
        )
    expected_family_union_counts = {
        PMV2Split.TRAIN.value: 14,
        PMV2Split.CALIBRATION.value: 5,
        PMV2Split.INTERNAL_TEST.value: 5,
    }
    observed_family_union_counts = {
        split: int(row["observed_count"])
        for split, row in report["semantic_family_coverage"]["splits"].items()
    }
    if observed_family_union_counts != expected_family_union_counts:
        raise RuntimeError(
            "recompiled PM-v2 semantic-family union counts do not match 14/5/5: "
            f"{observed_family_union_counts}"
        )

    new_states = load_states(args.out_dir / "pm_v2_states.jsonl")
    original_by_id = {state.state_id: state for state in original_states}
    new_by_id = {state.state_id: state for state in new_states}
    if set(original_by_id) != set(new_by_id):
        raise RuntimeError(
            "recompiled state_id set differs from the original -- "
            f"missing={set(original_by_id) - set(new_by_id) or None}, "
            f"extra={set(new_by_id) - set(original_by_id) or None}"
        )

    # authorized_user_context (the field every FIELD_ONLY_REPAIR overlay
    # touches) is NOT part of PMV2State/pm_v2_states.jsonl -- it lives only
    # in evaluator_contexts.jsonl. Checking states alone would silently
    # report "unchanged" for every FIELD_ONLY_REPAIR state that did not
    # also need its session_summary repaired, even though the repair was
    # correctly applied. Both files must be diffed for a real verification.
    original_evaluator_by_id = {
        str(row["state_id"]): row
        for row in iter_jsonl(args.original_out_dir / "evaluator_contexts.jsonl")
    }
    new_evaluator_by_id = {
        str(row["state_id"]): row
        for row in iter_jsonl(args.out_dir / "evaluator_contexts.jsonl")
    }
    if set(original_evaluator_by_id) != set(original_by_id):
        raise RuntimeError("evaluator_contexts.jsonl state_id set differs from pm_v2_states.jsonl")
    if set(new_evaluator_by_id) != set(new_by_id):
        raise RuntimeError(
            "recompiled evaluator_contexts.jsonl state_id set has extra or "
            f"missing rows vs the recompiled pm_v2_states.jsonl -- "
            f"missing={set(new_by_id) - set(new_evaluator_by_id) or None}, "
            f"extra={set(new_evaluator_by_id) - set(new_by_id) or None}"
        )

    repaired_state_ids = {overlay.state_id for overlay in overlays}
    unexpectedly_changed: dict[str, dict[str, list[str]]] = {}
    unexpectedly_unchanged: list[str] = []
    repaired_diffs: dict[str, dict[str, list[str]]] = {}
    for state_id in sorted(original_by_id):
        state_diff = dict_field_diff(
            original_by_id[state_id].model_dump(mode="json"),
            new_by_id[state_id].model_dump(mode="json"),
        )
        evaluator_diff = dict_field_diff(
            original_evaluator_by_id[state_id], new_evaluator_by_id[state_id]
        )
        combined = {"pm_v2_states": state_diff, "evaluator_contexts": evaluator_diff}
        if state_id in repaired_state_ids:
            repaired_diffs[state_id] = combined
            if not state_diff and not evaluator_diff:
                unexpectedly_unchanged.append(state_id)
        else:
            if state_diff or evaluator_diff:
                unexpectedly_changed[state_id] = combined

    output_file_sha256s = {
        name: sha256_file(args.out_dir / name)
        for name in (
            "pm_v2_states.jsonl",
            "pm_v2_bundles.jsonl",
            "evaluator_contexts.jsonl",
            "memory_backend.jsonl",
            "runtime_states.jsonl",
            "pm_v2_data_report.json",
        )
    }
    memory_backend_unchanged_from_original = sha256_file(
        args.out_dir / "memory_backend.jsonl"
    ) == sha256_file(args.original_out_dir / "memory_backend.jsonl")

    verification = {
        "protocol": "pm-v1.5-context-grounding-repair-recompile-verification-v1",
        "original_out_dir": str(args.original_out_dir),
        "original_bundles_sha256": original_bundles_sha256,
        "classification_sha256": classification_sha256,
        "repair_overlays_path": str(args.repair_overlays),
        "repair_overlays_sha256": sha256_file(args.repair_overlays),
        "output_file_sha256s": output_file_sha256s,
        "memory_backend_byte_identical_to_original": memory_backend_unchanged_from_original,
        "total_states": len(original_by_id),
        "repaired_state_count": len(repaired_state_ids),
        "untouched_state_count": len(original_by_id) - len(repaired_state_ids),
        "unexpectedly_changed_untouched_states": unexpectedly_changed,
        "unexpectedly_unchanged_repaired_states": unexpectedly_unchanged,
        "repaired_state_diffs": repaired_diffs,
        "status": (
            "PASS"
            if not unexpectedly_changed and not unexpectedly_unchanged
            else "FAIL"
        ),
    }
    if verification["status"] != "PASS":
        raise RuntimeError(
            "recompilation verification FAILED: "
            f"{len(unexpectedly_changed)} untouched states changed, "
            f"{len(unexpectedly_unchanged)} repaired states did not change -- "
            "see verification_report.json for exact field paths"
        )

    verification_path = args.out_dir / "context_grounding_repair_recompile_verification.json"
    write_json(verification_path, verification)

    # Downstream zero-API consumers (step0 shortcut audit, rule-grid
    # preflight) require a real artifact_attestation.json binding this
    # directory's states/evaluator_contexts to their real, hash-verified
    # inputs -- the same stage name formal generation uses, since this
    # recompilation stands in for it for the 25 repaired states.
    attestation_path = args.out_dir / "artifact_attestation.json"
    create_artifact_attestation(
        attestation_path,
        stage="pm_v1_5_development_data",
        inputs={
            "original_bundles": original_bundles_path,
            "classification": args.classification,
            "repair_overlays": args.repair_overlays,
            "pm_v1_5_config": args.pm_v1_5_config,
            "strategy_bank": args.strategy_bank,
            "selected_seed_sources": args.selected_seed_sources,
        },
        outputs={
            "states": (args.out_dir / "pm_v2_states.jsonl", True),
            "bundles": (args.out_dir / "pm_v2_bundles.jsonl", True),
            "evaluator_contexts": (args.out_dir / "evaluator_contexts.jsonl", True),
            "backend": (args.out_dir / "memory_backend.jsonl", True),
            "runtime_states": (args.out_dir / "runtime_states.jsonl", True),
            "data_report": (args.out_dir / "pm_v2_data_report.json", False),
        },
        parameters={
            "recompile_script": "scripts/v1_5/20f_recompile_context_grounding_repair_v1_5.py",
            "repaired_state_count": verification["repaired_state_count"],
        },
        expected={"total_states": verification["total_states"]},
    )

    print(
        {
            "status": "PASS",
            "out_dir": str(args.out_dir),
            "total_states": verification["total_states"],
            "repaired_state_count": verification["repaired_state_count"],
            "untouched_state_count": verification["untouched_state_count"],
            "verification_path": str(verification_path),
            "attestation_path": str(attestation_path),
        }
    )


if __name__ == "__main__":
    main()
