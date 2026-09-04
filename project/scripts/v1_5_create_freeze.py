#!/usr/bin/env python3
"""Create the fail-closed PM-v1.5 study freeze.

This fast-track freeze relies on the generic content-hash machinery rather
than PM-v2.2's human-approval wrapper. It nevertheless requires the complete
attested development chain, a reportable learned checkpoint, internal
decision-quality report, derived fixed checkpoints, and an isolated complete
non-truncated fixed-seeker bundle. It then locks every contract consumed by
V1.5 external generation, canary, schema-smoke, judging, and claim assessment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from metacom_pm.artifacts import (
    require_artifact_attestation,
    require_content_addressed_attestation,
)
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.evidence_filter import EvidenceFilterConfig
from metacom_pm.fixed_seeker_contract import FixedSeekerGenerationContract
from metacom_pm.evoemo import (
    FIXED_SEEKER_V22_STAGE,
    fixed_seeker_cost_planning_contract,
    load_evoemo,
)
from metacom_pm.freeze import create_study_freeze
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
)
from metacom_pm.pm_v2_evoemo import EVALUATION_UNIT_CONTRACT_PROTOCOL
from metacom_pm.pm_v2_external_eval import expected_external_units
from metacom_pm.pm_v2_external_schema_smoke import (
    POINTWISE_SCHEMA_SMOKE_PROTOCOL,
    POINTWISE_SCHEMA_SMOKE_PURPOSE,
)
from metacom_pm.pm_v2_forced_swap import select_forced_swap_units
from metacom_pm.pm_v2_judging import (
    composite_spec_from_config,
    labeling_settings_from_config,
    prompt_contract_hash,
)
from metacom_pm.pm_v22_reference_baselines import (
    POLICY_LOCK_TIMING,
    POST_GENERATION_POLICY_TUNING_PROHIBITED,
    REFERENCE_BASELINE_CONDITIONS,
)
from metacom_pm.strategy_bank import find_deterministic_esconv_evoemo_overlaps
from metacom_pm.v1_5_forced_swap_canary import (
    KEY_CLAIM_GATE,
    PROTOCOL as FORCED_SWAP_CANARY_PROTOCOL,
)
from metacom_pm.v1_5_external_batched import (
    PILOT_PROTOCOL as BATCHED_PILOT_PROTOCOL,
    PROTOCOL as BATCHED_EXTERNAL_PROTOCOL,
    RISK_AUDIT_PROTOCOL,
    batched_prompt_contract_hash,
    select_stratified_units,
)
from metacom_pm.v1_5_latency import PROTOCOL as LATENCY_DIAGNOSTIC_PROTOCOL

ROOT = Path(__file__).resolve().parents[1]

# Condition IDs are shared literally with scripts/v1_5/24_run_pm_v2_evoemo_v1_5.py
# (--condition pm_v2 / pm_v2_cost_matched_fixed / pm_v2_me_r0_fixed) and with
# the reusable src/metacom_pm/pm_v22_reference_baselines.py library that
# scripts/v1_5/24a_run_pmv22_reference_baselines_v1_5.py calls unmodified.
# "pm_v2" here is a condition-ID string inherited from that shared library,
# not a claim that this is the PM-v2.2 track -- pm_v1_5_config["version"]
# ("pm-v1.5") is the honest version marker; see configs/pm_v1_5.yaml header.
PM_V1_5_LEARNED_CONDITION = "pm_v2"
PM_V1_5_CONDITIONS = (
    PM_V1_5_LEARNED_CONDITION,
    "pm_v2_cost_matched_fixed",
    "pm_v2_me_r0_fixed",
    *REFERENCE_BASELINE_CONDITIONS,
)
EXPECTED_V1_5_BANK_CARDS = 12_403
EXPECTED_V1_5_EXCLUDED_ESCONV_SOURCES = 84
EXPECTED_V1_5_TRAIN_SEEDS = 875
EXPECTED_V1_5_ACTIONS_PER_STATE = 16


def _require_attestation_record_matches(
    attestation: Mapping[str, Any],
    *,
    section: str,
    logical_name: str,
    expected_path: Path,
) -> dict[str, Any]:
    record = (attestation.get(section) or {}).get(logical_name)
    if not isinstance(record, dict):
        raise RuntimeError(
            f"attestation lacks {section}.{logical_name} lineage binding"
        )
    if record.get("sha256") != sha256_file(expected_path):
        raise RuntimeError(
            f"attestation {section}.{logical_name} does not bind the frozen file"
        )
    return record


def require_v1_5_retrieval_consistency(
    pm_v1_5_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail before paid external generation if development/external RAG drifts."""

    retrieval = dict(pm_v1_5_config.get("retrieval") or {})
    external = dict(pm_v1_5_config.get("external_evaluation") or {})
    development_values = {
        "strategy_top_k": int(retrieval.get("strategy_top_k", -1)),
        "memory_min_score": float(retrieval.get("memory_min_score", -1.0)),
        "strategy_min_score": float(
            retrieval.get("strategy_min_score", -1.0)
        ),
    }
    external_values = {
        "strategy_top_k": int(external.get("strategy_top_k", -1)),
        "memory_min_score": float(external.get("memory_min_score", -1.0)),
        "strategy_min_score": float(
            external.get("strategy_min_score", -1.0)
        ),
    }
    if development_values != external_values:
        raise RuntimeError(
            "PM-v1.5 development/external retrieval settings differ; this "
            "would recreate the V1 configuration-generalization confound"
        )
    return {
        "status": "PASS",
        "protocol": "pm-v1.5-development-external-retrieval-lock-v1",
        **development_values,
    }


def require_v1_5_bank_and_seed_lineage(
    *,
    esconv_path: Path,
    evoemo_path: Path,
    strategy_bank_path: Path,
    split_manifest_path: Path,
    strategy_bank_audit_path: Path,
    seed_dialogues_path: Path,
    seed_audit_path: Path,
    turn_overlap_audit_path: Path,
) -> dict[str, Any]:
    """Verify the clean 12,403-card bank and 875-seed source lineage."""

    bank_audit = read_json(strategy_bank_audit_path)
    if (
        bank_audit.get("esconv_sha256") != sha256_file(esconv_path)
        or bank_audit.get("evoemo_sha256") != sha256_file(evoemo_path)
        or bank_audit.get("strategy_bank_sha256")
        != sha256_file(strategy_bank_path)
        or bank_audit.get("split_manifest_sha256")
        != sha256_file(split_manifest_path)
        or bank_audit.get("deterministic_source_id_rule_enabled") is not True
        or int(bank_audit.get("n_strategy_cards", -1))
        != EXPECTED_V1_5_BANK_CARDS
        or int(bank_audit.get("n_excluded_overlap", -1))
        != EXPECTED_V1_5_EXCLUDED_ESCONV_SOURCES
    ):
        raise RuntimeError(
            "PM-v1.5 Strategy Bank audit is stale or lacks the deterministic "
            "escN source-exclusion rule"
        )

    split_rows = [dict(row) for row in iter_jsonl(split_manifest_path)]
    manifest_by_id = {
        str(row.get("dialogue_id") or ""): row for row in split_rows
    }
    if (
        len(split_rows) != int(bank_audit.get("n_esconv_dialogues", -1))
        or len(manifest_by_id) != len(split_rows)
        or [int(row.get("index", -1)) for row in split_rows]
        != list(range(len(split_rows)))
        or any(
            str(row.get("dialogue_id") or "")
            != f"esconv_{index:04d}"
            for index, row in enumerate(split_rows)
        )
    ):
        raise RuntimeError("PM-v1.5 split manifest is incomplete or reordered")
    excluded_ids = {
        str(row["dialogue_id"])
        for row in split_rows
        if bool(row.get("excluded_for_evoemo_overlap"))
    }
    audited_overlap_ids = set((bank_audit.get("overlaps") or {}).keys())
    deterministic_ids = {
        f"esconv_{index:04d}"
        for index in find_deterministic_esconv_evoemo_overlaps(evoemo_path)
    }
    if (
        len(excluded_ids) != EXPECTED_V1_5_EXCLUDED_ESCONV_SOURCES
        or excluded_ids != audited_overlap_ids
        or not deterministic_ids
        or not deterministic_ids <= excluded_ids
    ):
        raise RuntimeError(
            "PM-v1.5 split manifest does not exclude the exact audited EvoEmo sources"
        )

    cards = [dict(row) for row in iter_jsonl(strategy_bank_path)]
    card_by_id = {str(row.get("strategy_id") or ""): row for row in cards}
    if len(cards) != EXPECTED_V1_5_BANK_CARDS or len(card_by_id) != len(cards):
        raise RuntimeError("PM-v1.5 Strategy Bank card count/IDs are invalid")
    if any(
        str(card.get("source_dialogue_id") or "") not in manifest_by_id
        or str(card.get("source_dialogue_id") or "") in excluded_ids
        or manifest_by_id[str(card["source_dialogue_id"])].get("split")
        != "train"
        for card in cards
    ):
        raise RuntimeError(
            "PM-v1.5 Strategy Bank contains a non-train or EvoEmo-overlap source"
        )

    turn_audit = read_json(turn_overlap_audit_path)
    if (
        turn_audit.get("protocol")
        != "pm-v1.5-strategy-card-evoemo-turn-overlap-audit-v2"
        or turn_audit.get("status") not in {"PASS", "GENERIC_FINDINGS_ONLY"}
        or turn_audit.get("strategy_bank_sha256")
        != sha256_file(strategy_bank_path)
        or turn_audit.get("evoemo_sha256") != sha256_file(evoemo_path)
        or int(turn_audit.get("n_cards_checked", -1)) != len(cards)
        or int(turn_audit.get("n_findings", -1))
        != len(turn_audit.get("findings") or [])
        or int(turn_audit.get("n_deterministic_source_findings", -1)) != 0
        or any(
            finding.get("deterministic_source_matches")
            for finding in (turn_audit.get("findings") or [])
        )
    ):
        raise RuntimeError(
            "PM-v1.5 turn-level overlap audit is stale or contains source leakage"
        )

    seed_rows = [dict(row) for row in iter_jsonl(seed_dialogues_path)]
    seed_ids = {str(row.get("dialogue_id") or "") for row in seed_rows}
    seed_audit = read_json(seed_audit_path)
    train_manifest = seed_audit.get("train_manifest") or {}
    split_audit = seed_audit.get("split_manifest") or {}
    if (
        seed_audit.get("status") != "COMPLETE"
        or seed_audit.get("output_sha256") != sha256_file(seed_dialogues_path)
        or int(seed_audit.get("selected_unique_train_seeds", -1))
        != EXPECTED_V1_5_TRAIN_SEEDS
        or int(seed_audit.get("evoemo_overlap_excluded_total", -1))
        != EXPECTED_V1_5_EXCLUDED_ESCONV_SOURCES
        or train_manifest.get("file_sha256") != sha256_file(split_manifest_path)
        or split_audit.get("sha256") != sha256_file(split_manifest_path)
        or len(seed_rows) != EXPECTED_V1_5_TRAIN_SEEDS
        or len(seed_ids) != len(seed_rows)
        or any(
            seed_id in excluded_ids
            or seed_id not in manifest_by_id
            or manifest_by_id[seed_id].get("split") != "train"
            or bool(row.get("excluded_for_evoemo_overlap"))
            or row.get("source_split") != "train"
            for seed_id, row in (
                (str(item.get("dialogue_id") or ""), item)
                for item in seed_rows
            )
        )
    ):
        raise RuntimeError("PM-v1.5 synthetic seed lineage is stale or contaminated")

    return {
        "status": "PASS",
        "protocol": "pm-v1.5-bank-seed-lineage-freeze-v1",
        "strategy_bank_sha256": sha256_file(strategy_bank_path),
        "strategy_bank_cards": len(cards),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "excluded_esconv_sources": len(excluded_ids),
        "deterministic_esconv_sources": len(deterministic_ids),
        "seed_dialogues_sha256": sha256_file(seed_dialogues_path),
        "seed_dialogues": len(seed_rows),
        "strategy_bank_audit_sha256": sha256_file(strategy_bank_audit_path),
        "seed_audit_sha256": sha256_file(seed_audit_path),
        "turn_overlap_audit_sha256": sha256_file(turn_overlap_audit_path),
        "turn_overlap_findings": int(turn_audit["n_findings"]),
        "deterministic_source_findings": 0,
    }


def require_v1_5_development_chain(
    *,
    pm_v1_5_config: Mapping[str, Any],
    pm_v1_5_config_path: Path,
    strategy_bank_path: Path,
    seed_dialogues_path: Path,
    data_report_path: Path,
    data_attestation_path: Path,
    states_path: Path,
    sweep_summary_path: Path,
    sweep_attestation_path: Path,
    judging_summary_path: Path,
    judging_labels_path: Path,
    judging_attestation_path: Path,
) -> dict[str, Any]:
    """Require the exact full 468-state x 16-action attested chain."""

    generation = dict(pm_v1_5_config.get("data_generation") or {})
    split_users = {
        "train": int(generation.get("train_users", -1)),
        "calibration": int(generation.get("calibration_users", -1)),
        "internal_test": int(generation.get("internal_test_users", -1)),
    }
    cases_per_user = len(generation.get("required_regimes") or [])
    expected_users = sum(split_users.values())
    expected_split_states = {
        split: users * cases_per_user for split, users in split_users.items()
    }
    expected_states = sum(expected_split_states.values())
    expected_outcomes = expected_states * EXPECTED_V1_5_ACTIONS_PER_STATE
    if expected_users != 52 or cases_per_user != 9 or expected_states != 468:
        raise RuntimeError("PM-v1.5 frozen development design is no longer 52x9")

    require_artifact_attestation(
        data_attestation_path,
        required_stage="pm_v1_5_development_data",
        required_output_paths={
            "states": states_path,
            "data_report": data_report_path,
        },
    )
    data_attestation = read_json(data_attestation_path)
    _require_attestation_record_matches(
        data_attestation,
        section="inputs",
        logical_name="pm_v1_5_config",
        expected_path=pm_v1_5_config_path,
    )
    _require_attestation_record_matches(
        data_attestation,
        section="inputs",
        logical_name="seed_dialogues",
        expected_path=seed_dialogues_path,
    )
    _require_attestation_record_matches(
        data_attestation,
        section="inputs",
        logical_name="strategy_bank",
        expected_path=strategy_bank_path,
    )
    data_states_record = _require_attestation_record_matches(
        data_attestation,
        section="outputs",
        logical_name="states",
        expected_path=states_path,
    )
    data_report_record = _require_attestation_record_matches(
        data_attestation,
        section="outputs",
        logical_name="data_report",
        expected_path=data_report_path,
    )
    for logical_name in ("runtime", "backend", "evaluator_contexts"):
        record = (data_attestation.get("outputs") or {}).get(logical_name)
        if not isinstance(record, dict) or not record.get("sha256"):
            raise RuntimeError(
                f"development attestation lacks output lineage: {logical_name}"
            )
    data_report = read_json(data_report_path)
    full_design = data_report.get("full_state_design") or {}
    if (
        data_report.get("status") != "COMPLETE"
        or int(data_report.get("n_users", -1)) != expected_users
        or int(data_report.get("n_states", -1)) != expected_states
        or data_report.get("split_counts") != expected_split_states
        or data_report.get("all_states_have_16_actions") is not True
        or full_design.get("status") != "PASS"
        or int(full_design.get("cases_per_user", -1)) != cases_per_user
        or full_design.get("expected_split_state_counts")
        != expected_split_states
        or int(full_design.get("expected_total_states", -1)) != expected_states
        or int(data_states_record.get("rows", -1)) != expected_states
        or (data_attestation.get("expected") or {}).get("states")
        != expected_states
        or (data_attestation.get("expected") or {}).get("users")
        != expected_users
    ):
        raise RuntimeError("PM-v1.5 development data is partial or off-contract")

    require_artifact_attestation(
        sweep_attestation_path,
        required_stage="action_sweep",
        required_output_paths={"summary": sweep_summary_path},
    )
    sweep_attestation = read_json(sweep_attestation_path)
    for logical_name in ("runtime", "backend"):
        sweep_record = (sweep_attestation.get("inputs") or {}).get(logical_name)
        data_record = (data_attestation.get("outputs") or {}).get(logical_name)
        if not isinstance(sweep_record, dict) or sweep_record.get(
            "sha256"
        ) != data_record.get("sha256"):
            raise RuntimeError(
                f"full sweep is not bound to development output: {logical_name}"
            )
    _require_attestation_record_matches(
        sweep_attestation,
        section="inputs",
        logical_name="strategy_bank",
        expected_path=strategy_bank_path,
    )
    sweep_summary = read_json(sweep_summary_path)
    sweep_parameters = sweep_attestation.get("parameters") or {}
    bindings = sweep_summary.get("contract_bindings") or {}
    full_gate = bindings.get("v1_5_full_sweep_gate") or {}
    semantic_review = bindings.get("semantic_sanity") or {}
    if (
        sweep_summary.get("status") != "COMPLETE"
        or int(sweep_summary.get("n_cards", -1)) != expected_states
        or int(sweep_summary.get("expected_outcomes", -1))
        != expected_outcomes
        or int(sweep_summary.get("completed_outcomes", -1))
        != expected_outcomes
        or bool(sweep_summary.get("failures"))
        or sweep_summary.get("strategy_bank_sha256")
        != sha256_file(strategy_bank_path)
        or bindings.get("scope") != "full"
        or full_gate.get("protocol") != "pm-v1.5-full-sweep-gate-v1"
        or full_gate.get("status") != "PASS"
        or full_gate.get("scope") != "full"
        or full_gate.get("human_calibration_performed") is not False
        or len(str(full_gate.get("automated_review_report_sha256") or ""))
        != 64
        or len(
            str(full_gate.get("automated_review_attestation_sha256") or "")
        )
        != 64
        or full_gate.get("automated_review_report_sha256")
        != semantic_review.get("automated_review_report_sha256")
        or full_gate.get("automated_review_attestation_sha256")
        != semantic_review.get("automated_review_attestation_sha256")
        or bindings.get("pm_v2_config_sha256")
        != sha256_file(pm_v1_5_config_path)
        or bindings.get("pm_v2_version") != "pm-v1.5"
        or bindings.get("retrieval") != pm_v1_5_config.get("retrieval")
        or (bindings.get("evidence_filter") or {}).get("enabled") is not False
        or (bindings.get("evidence_filter_model") or {}).get("mode")
        != "disabled_for_pm_v1_5"
        or sweep_parameters.get("max_cards") is not None
        or sweep_parameters.get("action_filter") is not None
        or sweep_parameters.get("card_filter") is not None
        or sweep_parameters.get("contract_bindings") != bindings
        or int(
            ((sweep_attestation.get("outputs") or {}).get("action_outcomes") or {}).get(
                "rows", -1
            )
        )
        != expected_outcomes
        or (sweep_attestation.get("expected") or {}).get("cards")
        != expected_states
        or (sweep_attestation.get("expected") or {}).get("outcomes")
        != expected_outcomes
    ):
        raise RuntimeError(
            "study freeze requires the honest full PM-v1.5 468x16 action sweep"
        )

    require_artifact_attestation(
        judging_attestation_path,
        required_stage="pm_v2_action_judging",
        required_output_paths={
            "summary": judging_summary_path,
            "labels": judging_labels_path,
        },
    )
    judging_attestation = read_json(judging_attestation_path)
    cross_bindings = {
        "states": data_states_record,
        "outcomes": (sweep_attestation.get("outputs") or {}).get(
            "action_outcomes"
        ),
        "evaluator_contexts": (data_attestation.get("outputs") or {}).get(
            "evaluator_contexts"
        ),
        "sweep_summary": (sweep_attestation.get("outputs") or {}).get(
            "summary"
        ),
    }
    for logical_name, source_record in cross_bindings.items():
        judge_record = (judging_attestation.get("inputs") or {}).get(
            logical_name
        )
        if (
            not isinstance(judge_record, dict)
            or not isinstance(source_record, dict)
            or judge_record.get("sha256") != source_record.get("sha256")
        ):
            raise RuntimeError(
                f"development judging is not bound to full-chain {logical_name}"
            )
    _require_attestation_record_matches(
        judging_attestation,
        section="inputs",
        logical_name="sweep_attestation",
        expected_path=sweep_attestation_path,
    )
    judging_summary = read_json(judging_summary_path)
    judging_parameters = judging_attestation.get("parameters") or {}
    if (
        judging_summary.get("status") != "COMPLETE"
        or judging_summary.get("scope") != "full"
        or judging_summary.get("reportability_status") != "REPORTABLE"
        or int(judging_summary.get("outcomes", -1)) != expected_outcomes
        or int(judging_summary.get("label_rows", -1)) != expected_outcomes
        or int(judging_summary.get("remaining_judge_pairs", -1)) != 0
        or int(judging_summary.get("remaining_api_calls", -1)) != 0
        or (judging_summary.get("quality_gate") or {}).get("status") != "PASS"
        or (judging_summary.get("raw_family_quality_gate") or {}).get(
            "status"
        )
        != "PASS"
        or judging_parameters.get("status") != "COMPLETE"
        or judging_parameters.get("scope") != "full"
        or judging_parameters.get("pm_v2_config_sha256")
        != sha256_file(pm_v1_5_config_path)
        or int(
            ((judging_attestation.get("outputs") or {}).get("labels") or {}).get(
                "rows", -1
            )
        )
        != expected_outcomes
    ):
        raise RuntimeError(
            "study freeze requires complete reportable full-matrix judging"
        )

    return {
        "status": "PASS",
        "protocol": "pm-v1.5-full-development-chain-freeze-v1",
        "users": expected_users,
        "states": expected_states,
        "actions_per_state": EXPECTED_V1_5_ACTIONS_PER_STATE,
        "outcomes": expected_outcomes,
        "labels": expected_outcomes,
        "data_report_sha256": data_report_record["sha256"],
        "data_attestation_sha256": data_attestation["attestation_sha256"],
        "sweep_attestation_sha256": sweep_attestation["attestation_sha256"],
        "judging_attestation_sha256": judging_attestation[
            "attestation_sha256"
        ],
        "automated_review_report_sha256": full_gate[
            "automated_review_report_sha256"
        ],
        "automated_review_attestation_sha256": full_gate[
            "automated_review_attestation_sha256"
        ],
    }


def require_v1_5_fixed_tracks(
    *,
    evoemo_path: Path,
    tracks_path: Path,
    attestation_path: Path,
    simulator_id: str,
    max_turns: int,
    seeds: list[int],
    treatment: dict[str, Any],
    treatment_sha256: str,
    cost_planning: dict[str, Any],
    cost_planning_sha256: str,
) -> dict[str, Any]:
    """Reject stale, truncated, partial, or cross-track seeker bundles."""

    bundle_dir = tracks_path.resolve().parent
    verification = require_content_addressed_attestation(
        attestation_path,
        required_stage=FIXED_SEEKER_V22_STAGE,
        relocated_inputs={
            "evoemo": evoemo_path,
            "run_manifest": bundle_dir / "run_manifest.json",
            "cost_estimate": bundle_dir / "cost_estimate.json",
            "call_plan": bundle_dir / "call_plan.jsonl",
        },
        relocated_outputs={
            "tracks": tracks_path,
            "raw_calls": bundle_dir / "raw_seeker_calls.jsonl",
            "physical_attempt_ledger": bundle_dir
            / "physical_attempt_ledger.jsonl",
            "summary": bundle_dir / "summary.json",
        },
    )
    attestation = read_json(attestation_path)
    expected_parameters = {
        "simulator_id": simulator_id,
        "max_turns": max_turns,
        "seeds": seeds,
        "fixed_seeker_generation_contract": treatment,
        "fixed_seeker_generation_contract_sha256": treatment_sha256,
        "fixed_seeker_cost_planning": cost_planning,
        "fixed_seeker_cost_planning_sha256": cost_planning_sha256,
    }
    parameters = attestation.get("parameters") or {}
    for key, expected in expected_parameters.items():
        if parameters.get(key) != expected:
            raise RuntimeError(f"fixed-track attestation mismatch: {key}")

    rows = list(iter_jsonl(tracks_path))
    expected_keys = {
        (str(user["id"]), int(topic["idx"]), seed, simulator_id)
        for user in load_evoemo(evoemo_path)
        for topic in (user.get("subsequent_topics") or [])
        for seed in seeds
    }
    observed_keys = {
        (
            str(row.get("user_id")),
            int(row.get("topic_index", -1)),
            int(row.get("seed", -1)),
            str(row.get("simulator_id")),
        )
        for row in rows
    }
    if len(rows) != len(observed_keys) or observed_keys != expected_keys:
        raise RuntimeError("fixed seeker tracks do not exactly cover the frozen matrix")
    if any(
        len(row.get("seeker_turns") or []) != max_turns
        or row.get("fixed_seeker_generation_contract") != treatment
        or row.get("fixed_seeker_generation_contract_sha256")
        != treatment_sha256
        or row.get("fixed_seeker_cost_planning") != cost_planning
        or row.get("fixed_seeker_cost_planning_sha256") != cost_planning_sha256
        for row in rows
    ):
        raise RuntimeError("fixed seeker track rows are partial or treatment-stale")

    expected = attestation.get("expected") or {}
    summary = read_json(bundle_dir / "summary.json")
    if (
        int(expected.get("tracks", -1)) != len(expected_keys)
        or int(expected.get("turns_per_track", -1)) != max_turns
        or int(expected.get("completion_truncated_count", -1)) != 0
        or expected.get("planned_budget_gate_status") != "PASS"
        or expected.get("observed_budget_gate_status") != "PASS"
        or summary.get("status") != "COMPLETE"
        or int(summary.get("expected_tracks", -1)) != len(expected_keys)
        or int(summary.get("completed_tracks", -1)) != len(expected_keys)
        or int(summary.get("max_turns", -1)) != max_turns
        or int(summary.get("completion_truncated_count", -1)) != 0
        or bool(summary.get("failures"))
        or summary.get("fixed_seeker_generation_contract") != treatment
        or summary.get("fixed_seeker_generation_contract_sha256")
        != treatment_sha256
        or summary.get("fixed_seeker_cost_planning") != cost_planning
        or summary.get("fixed_seeker_cost_planning_sha256")
        != cost_planning_sha256
        or (summary.get("planned_budget_gate") or {}).get("status") != "PASS"
        or (summary.get("observed_budget_gate") or {}).get("status") != "PASS"
    ):
        raise RuntimeError("fixed seeker bundle is incomplete, truncated, or stale")
    return {
        "status": "PASS",
        "tracks": len(rows),
        "tracks_sha256": sha256_file(tracks_path),
        "attestation_sha256": verification["attestation_sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--pm-v1-5-config", type=Path, default=ROOT / "configs" / "pm_v1_5.yaml")
    parser.add_argument(
        "--esconv",
        type=Path,
        default=ROOT / "data" / "external" / "ESConv.json",
    )
    parser.add_argument("--evoemo", type=Path, default=ROOT / "data" / "external" / "evo_emo.json")
    parser.add_argument(
        "--strategy-bank", type=Path, default=ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl"
    )
    parser.add_argument(
        "--strategy-bank-audit",
        type=Path,
        default=ROOT / "data" / "strategy" / "strategy_bank_audit_v1_5.json",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=ROOT
        / "data"
        / "strategy"
        / "esconv_split_manifest_v1_5.jsonl",
    )
    parser.add_argument(
        "--seed-dialogues",
        type=Path,
        default=ROOT
        / "data"
        / "pm_v2"
        / "train_seed_dialogues_v1_5.jsonl",
    )
    parser.add_argument(
        "--seed-audit",
        type=Path,
        default=ROOT
        / "data"
        / "pm_v2"
        / "train_seed_dialogues_v1_5.jsonl.audit.json",
    )
    parser.add_argument(
        "--turn-overlap-audit",
        type=Path,
        default=ROOT
        / "outputs"
        / "v1_5_strategy_card_evoemo_turn_overlap_audit_v1_5_bank.json",
    )
    parser.add_argument(
        "--fixed-tracks",
        type=Path,
        required=True,
        help="Non-truncated outputs/evoemo_fixed_tracks_v1_5 bundle; never reuse or overwrite V2.2 tracks.",
    )
    parser.add_argument("--fixed-tracks-attestation", type=Path, required=True)
    parser.add_argument("--pm-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--pm-training-report",
        type=Path,
        required=True,
        help="22_train_pm_v2_v1_5.py's training_report.json -- locked here, before any "
        "reference-baseline generation, and re-checked in 25_eval_pm_v2_external_v1_5.py "
        "against both the current checkpoint file and the reference-baseline attestation's "
        "own recorded policy lock.",
    )
    parser.add_argument("--cost-matched-fixed-checkpoint", type=Path, required=True)
    parser.add_argument("--me-r0-fixed-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--development-data-attestation",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "artifact_attestation.json",
    )
    parser.add_argument(
        "--development-data-report",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "pm_v2_data_report.json",
    )
    parser.add_argument(
        "--states",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "pm_v2_states.jsonl",
    )
    parser.add_argument(
        "--sweep-summary",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_sweep" / "summary.json",
    )
    parser.add_argument(
        "--sweep-attestation",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_sweep" / "artifact_attestation.json",
    )
    parser.add_argument(
        "--judging-summary",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_judging" / "summary.json",
    )
    parser.add_argument(
        "--judging-labels",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_judging" / "action_labels.jsonl",
    )
    parser.add_argument(
        "--judging-attestation",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_judging" / "artifact_attestation.json",
    )
    parser.add_argument(
        "--fixed-baselines-report",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_model" / "fixed_baselines.json",
    )
    parser.add_argument(
        "--decision-quality-report",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v1_5_decision_quality"
        / "decision_quality_report.json",
    )
    parser.add_argument(
        "--decision-quality-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v1_5_decision_quality"
        / "artifact_attestation.json",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "pm_v1_5_study_freeze.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists() and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite existing freeze without --overwrite: {args.out}")
    if (
        args.fixed_tracks.parent.name != "evoemo_fixed_tracks_v1_5"
        or args.fixed_tracks_attestation.parent != args.fixed_tracks.parent
    ):
        raise RuntimeError(
            "V1.5 freeze requires its isolated outputs/evoemo_fixed_tracks_v1_5 bundle"
        )

    experiment_config = load_config(args.config)
    pm_v1_5_config = load_config(args.pm_v1_5_config)
    if pm_v1_5_config.get("version") != "pm-v1.5":
        raise RuntimeError("this freeze creator requires a pm-v1.5 config")
    retrieval_consistency = require_v1_5_retrieval_consistency(
        pm_v1_5_config
    )
    bank_seed_lineage = require_v1_5_bank_and_seed_lineage(
        esconv_path=args.esconv,
        evoemo_path=args.evoemo,
        strategy_bank_path=args.strategy_bank,
        split_manifest_path=args.split_manifest,
        strategy_bank_audit_path=args.strategy_bank_audit,
        seed_dialogues_path=args.seed_dialogues,
        seed_audit_path=args.seed_audit,
        turn_overlap_audit_path=args.turn_overlap_audit,
    )

    training_report = read_json(args.pm_training_report)
    checkpoint_sha256 = sha256_file(args.pm_checkpoint)
    reportability_checks = training_report.get("reportability_checks") or {}
    if (
        training_report.get("status") != "COMPLETE"
        or training_report.get("require_learned_routing_advantage_before_external")
        is not True
        or training_report.get("learned_routing_advantage_verified") is not True
        or not reportability_checks
        or not all(value is True for value in reportability_checks.values())
        or training_report.get("checkpoint_sha256") != checkpoint_sha256
        or Path(str(training_report.get("checkpoint") or "")).name
        != args.pm_checkpoint.name
        or training_report.get("pm_v2_config_sha256")
        != sha256_file(args.pm_v1_5_config)
        or training_report.get("states_sha256") != sha256_file(args.states)
        or training_report.get("labels_sha256") != sha256_file(args.judging_labels)
    ):
        raise RuntimeError(
            "study freeze requires the exact reportable PM-v1.5 policy and "
            "a fully passed learned-routing gate"
        )

    development_chain = require_v1_5_development_chain(
        pm_v1_5_config=pm_v1_5_config,
        pm_v1_5_config_path=args.pm_v1_5_config,
        strategy_bank_path=args.strategy_bank,
        seed_dialogues_path=args.seed_dialogues,
        data_report_path=args.development_data_report,
        data_attestation_path=args.development_data_attestation,
        states_path=args.states,
        sweep_summary_path=args.sweep_summary,
        sweep_attestation_path=args.sweep_attestation,
        judging_summary_path=args.judging_summary,
        judging_labels_path=args.judging_labels,
        judging_attestation_path=args.judging_attestation,
    )

    fixed_report = read_json(args.fixed_baselines_report)
    expected_fixed = {
        "cost_matched_fixed": args.cost_matched_fixed_checkpoint,
        "event_memory_r0": args.me_r0_fixed_checkpoint,
    }
    if (
        fixed_report.get("status") != "COMPLETE"
        or fixed_report.get("track") != "pm-v1.5"
        or fixed_report.get("source_checkpoint_sha256") != checkpoint_sha256
        or fixed_report.get("training_report_sha256")
        != sha256_file(args.pm_training_report)
    ):
        raise RuntimeError("fixed baselines were not derived from the frozen PM-v1.5 policy")
    for name, path in expected_fixed.items():
        row = (fixed_report.get("baselines") or {}).get(name) or {}
        if row.get("checkpoint_sha256") != sha256_file(path):
            raise RuntimeError(f"fixed baseline checkpoint hash mismatch: {name}")
    if (fixed_report.get("baselines") or {}).get("event_memory_r0", {}).get(
        "action_id"
    ) != "ME+R0":
        raise RuntimeError("event-memory fixed baseline is not ME+R0")

    decision_quality_verification = require_artifact_attestation(
        args.decision_quality_attestation,
        required_stage="pm_v1_5_decision_quality",
        required_output_paths={"report": args.decision_quality_report},
    )
    decision_quality_report = read_json(args.decision_quality_report)
    if (
        decision_quality_report.get("status") != "COMPLETE"
        or decision_quality_report.get("protocol")
        != "pm-v1.5-internal-decision-quality-v1"
        or decision_quality_report.get("split") != "internal_test"
        or int(decision_quality_report.get("n_states") or 0) < 1
        or decision_quality_report.get("pm_v1_5_config_sha256")
        != sha256_file(args.pm_v1_5_config)
        or decision_quality_report.get("checkpoint_sha256") != checkpoint_sha256
        or decision_quality_report.get("training_report_sha256")
        != sha256_file(args.pm_training_report)
    ):
        raise RuntimeError(
            "study freeze requires a complete, attested PM-v1.5 internal "
            "decision-quality report"
        )

    supporter_generation_contract = SupporterGenerationContract.from_config(pm_v1_5_config)
    fixed_seeker_contract = FixedSeekerGenerationContract.from_mapping(
        pm_v1_5_config["fixed_seeker_generation_treatment"]
    )
    fixed_seeker_endpoint = endpoint_from_config(experiment_config, fixed_seeker_contract.seeker_endpoint)
    bound_fixed_seeker_contract = fixed_seeker_contract.bind_endpoint(
        fixed_seeker_contract.seeker_endpoint, fixed_seeker_endpoint
    )
    fixed_seeker_cost_planning = fixed_seeker_cost_planning_contract(
        pm_v1_5_config.get("fixed_seeker_cost_planning") or {}
    )
    fixed_seeker_cost_planning_sha256 = sha256_text(
        canonical_json(fixed_seeker_cost_planning)
    )

    # PM-v1.5: Evidence Filter forced off everywhere (see scripts/v1_5/06, 20,
    # 24, 24a docstrings). Mirror the same override here so the freeze's
    # evidence_filter binding matches what every generation stage actually
    # used, rather than the pm_v1_5.yaml value alone (belt-and-suspenders,
    # since the yaml already says enabled: false).
    evidence_filter_config = EvidenceFilterConfig.from_mapping(
        {**pm_v1_5_config["evidence_filter"], "enabled": False}
    )
    evidence_filter_model_binding = {
        "mode": "disabled_for_pm_v1_5",
        "reason": "Evidence Filter is out of scope for PM-v1.5; PM is a pure pre-retrieval router.",
    }

    external = dict(pm_v1_5_config["external_evaluation"])
    generator_endpoint_name = supporter_generation_contract.generator_endpoint
    generator_endpoint = endpoint_from_config(experiment_config, generator_endpoint_name)
    generator_endpoint_sha256 = sha256_text(
        canonical_json(
            {
                "model": generator_endpoint.model,
                "family": generator_endpoint.family,
                "base_url": generator_endpoint.base_url,
            }
        )
    )

    # Must match what scripts/v1_5/24a_run_pmv22_reference_baselines_v1_5.py
    # independently reads (protocol.robustness_seeds), not just primary_seed,
    # or PM (scripts/v1_5/24) and the fixed baselines (24a) end up scored
    # over two different evaluation-unit universes.
    seeds = [int(value) for value in experiment_config["protocol"]["robustness_seeds"]]
    if not seeds:
        raise RuntimeError("experiment protocol robustness_seeds is empty")
    simulator_id = str(external["simulator_id"])
    fixed_tracks_verification = require_v1_5_fixed_tracks(
        evoemo_path=args.evoemo,
        tracks_path=args.fixed_tracks,
        attestation_path=args.fixed_tracks_attestation,
        simulator_id=simulator_id,
        max_turns=int(external["max_turns"]),
        seeds=seeds,
        treatment=bound_fixed_seeker_contract.payload(),
        treatment_sha256=bound_fixed_seeker_contract.digest(),
        cost_planning=fixed_seeker_cost_planning,
        cost_planning_sha256=fixed_seeker_cost_planning_sha256,
    )
    turn_indices = [int(v) for v in external["turn_indices"]]
    expected_units = expected_external_units(
        args.evoemo, seeds=seeds, simulator_id=simulator_id, turn_indices=turn_indices
    )
    evaluation_unit_contract = {
        "protocol": EVALUATION_UNIT_CONTRACT_PROTOCOL,
        "evaluation_turn_indices": turn_indices,
        "expected_unit_count": len(expected_units),
        "expected_units_sha256": sha256_text(canonical_json(expected_units)),
    }

    generation_contract: dict[str, Any] = {
        "protocol": supporter_generation_contract.version,
        "supporter_generation_treatment": supporter_generation_contract.payload(),
        "supporter_generation_treatment_sha256": supporter_generation_contract.digest(),
        "fixed_seeker_generation_treatment": bound_fixed_seeker_contract.payload(),
        "fixed_seeker_generation_treatment_sha256": bound_fixed_seeker_contract.digest(),
        "fixed_seeker_cost_planning": fixed_seeker_cost_planning,
        "fixed_seeker_cost_planning_sha256": fixed_seeker_cost_planning_sha256,
        "evidence_filter": evidence_filter_config.payload(),
        "evidence_filter_config_sha256": evidence_filter_config.digest(),
        "evidence_filter_model": evidence_filter_model_binding,
        "generator_endpoint_sha256": generator_endpoint_sha256,
        "generator_pricing_usd_per_mtok": {"input": 0.15, "output": 0.60},
        "simulator_id": simulator_id,
        "max_turns": int(external["max_turns"]),
        "seeds": seeds,
        "evaluation_unit_contract": evaluation_unit_contract,
        "paid_generation_scope": "frozen_evaluation_turns_only",
        "action_preflight_gate_scope": "frozen_evaluation_turns_only",
        "all_turn_action_preflight_is_diagnostic_only": True,
        "strategy_action_tokens": int(external["strategy_action_tokens"]),
        "strategy_top_k": int(external["strategy_top_k"]),
        "session_rag_top_k": int(
            experiment_config["protocol"]["evoemo_session_rag_top_k"]
        ),
        "memory_min_score": external.get("memory_min_score"),
        "strategy_min_score": external.get("strategy_min_score"),
        "input_token_safety_factor": float(
            pm_v1_5_config["api_cost_planning"]["input_token_safety_factor"]
        ),
        "fail_on_reported_input_overrun": bool(
            pm_v1_5_config["api_cost_planning"]["fail_on_reported_input_overrun"]
        ),
        "generator_retries": 1,
        "action_preflight_gates": dict(external["action_preflight"]),
        "maximum_cost_matched_relative_deviation": float(
            external["maximum_cost_matched_relative_deviation"]
        ),
        # Locked here, before any reference-baseline (24a) generation, so a
        # checkpoint swapped in after seeing partial external results is
        # detectable: 25_eval_pm_v2_external_v1_5.py re-hashes the CLI-supplied
        # checkpoint/report and also cross-checks 24a's own recorded policy
        # lock against these exact values.
        "policy_checkpoint_sha256": checkpoint_sha256,
        "policy_training_report_sha256": sha256_file(args.pm_training_report),
        "policy_lock_timing": POLICY_LOCK_TIMING,
        "post_generation_policy_tuning_prohibited": (
            POST_GENERATION_POLICY_TUNING_PROHIBITED
        ),
    }
    labeling = labeling_settings_from_config(pm_v1_5_config)
    api_cost_planning = {
        "input_token_safety_factor": float(
            pm_v1_5_config["api_cost_planning"]["input_token_safety_factor"]
        ),
        "fail_on_reported_input_overrun": bool(
            pm_v1_5_config["api_cost_planning"]["fail_on_reported_input_overrun"]
        ),
    }
    judge_pricing_usd_per_mtok = dict(external["judge_pricing_usd_per_mtok"])
    judge_endpoint_names = [str(v) for v in external["external_judge_endpoints"]]
    judge_endpoints = []
    for name in judge_endpoint_names:
        endpoint = endpoint_from_config(experiment_config, name)
        judge_endpoints.append(
            {
                "name": name,
                "model": endpoint.model,
                "family": endpoint.family,
                "base_url": endpoint.base_url,
                "sha256": sha256_text(
                    canonical_json(
                        {
                            "model": endpoint.model,
                            "family": endpoint.family,
                            "base_url": endpoint.base_url,
                        }
                    )
                ),
            }
        )
    judge_seed = int(external["external_judge_seed"])

    forced_cfg = dict(external["forced_swap"])
    forced_units = select_forced_swap_units(
        expected_units,
        sample_units=int(forced_cfg["sample_units"]),
        seed=int(forced_cfg["judge_seed"]),
    )
    if not forced_units:
        raise RuntimeError("external evaluation universe cannot supply the canary")
    smoke_unit = forced_units[0]
    forced_swap_selected_units_sha256 = sha256_text(canonical_json(forced_units))
    forced_swap_contract = {
        "protocol": FORCED_SWAP_CANARY_PROTOCOL,
        "purpose": "judge_sensitivity_and_order_robustness_not_pm_efficacy",
        "treatment": PM_V1_5_LEARNED_CONDITION,
        "baseline": "pm_v2_cost_matched_fixed",
        "sample_units": int(forced_cfg["sample_units"]),
        "order_variants": [int(value) for value in forced_cfg["order_variants"]],
        "judge_seed": int(forced_cfg["judge_seed"]),
        "estimated_output_tokens_per_call": int(
            forced_cfg["estimated_output_tokens_per_call"]
        ),
        "maximum_order_disagreement_rate": float(
            forced_cfg["maximum_order_disagreement_rate"]
        ),
        "minimum_schema_success_rate": float(
            forced_cfg["minimum_schema_success_rate"]
        ),
        "minimum_cross_family_support_delta_correlation": float(
            forced_cfg["minimum_cross_family_support_delta_correlation"]
        ),
        "require_cross_family_direction_agreement": bool(
            forced_cfg["require_cross_family_direction_agreement"]
        ),
        "minimum_support_delta_ci_upper_for_continuation": float(
            forced_cfg["minimum_support_delta_ci_upper_for_continuation"]
        ),
        "minimum_support_delta_ci_lower_for_advantage": float(
            forced_cfg["minimum_support_delta_ci_lower_for_advantage"]
        ),
        "require_positive_support_delta_every_family": bool(
            forced_cfg["require_positive_support_delta_every_family"]
        ),
        "minimum_resolved_preference_margin": int(
            forced_cfg["minimum_resolved_preference_margin"]
        ),
        "maximum_cost_matched_relative_deviation": float(
            external["maximum_cost_matched_relative_deviation"]
        ),
        "judge_endpoints": judge_endpoints,
        "judge_pricing_usd_per_mtok": judge_pricing_usd_per_mtok,
        "api_cost_planning": api_cost_planning,
        "selected_units_sha256": forced_swap_selected_units_sha256,
    }

    pointwise_schema_smoke = {
        "protocol": POINTWISE_SCHEMA_SMOKE_PROTOCOL,
        "purpose": POINTWISE_SCHEMA_SMOKE_PURPOSE,
        "condition": PM_V1_5_LEARNED_CONDITION,
        "unit": list(smoke_unit),
        "unit_id": sha256_text(canonical_json(smoke_unit))[:24],
        "expected_calls": 4,
        "judge_endpoints": judge_endpoints,
        "judge_pricing_usd_per_mtok": judge_pricing_usd_per_mtok,
        "api_cost_planning": api_cost_planning,
        "judge_seed": judge_seed,
        "judge_prompt_contract_sha256": prompt_contract_hash(),
        "forced_swap_selected_units_sha256": forced_swap_selected_units_sha256,
        "required_before_full_external_client_creation": False,
        "legacy_pointwise_fallback_only": True,
        "estimated_response_output_tokens": 600,
        "estimated_risk_output_tokens": 700,
    }

    batched_cfg = dict(external.get("batched_judging") or {})
    if batched_cfg.get("protocol") != BATCHED_EXTERNAL_PROTOCOL:
        raise RuntimeError("PM-v1.5 config lacks the current batched judging protocol")
    endpoint_by_name = {
        str(row["name"]): row for row in judge_endpoints
    }
    primary_endpoint_name = str(batched_cfg["primary_quality_endpoint"])
    sensitivity_endpoint_name = str(batched_cfg["sensitivity_quality_endpoint"])
    if (
        primary_endpoint_name == sensitivity_endpoint_name
        or {primary_endpoint_name, sensitivity_endpoint_name}
        != set(endpoint_by_name)
    ):
        raise RuntimeError(
            "batched primary/sensitivity endpoints must exactly cover both frozen judges"
        )
    scoring_units = sorted(set(expected_units) - set(forced_units))
    if len(scoring_units) + len(forced_units) != len(expected_units):
        raise RuntimeError("batched scoring/canary partition is not exact")
    sensitivity_units = select_stratified_units(
        scoring_units,
        units_per_user=int(batched_cfg["sensitivity_units_per_user"]),
        seed=int(batched_cfg["sensitivity_selection_seed"]),
    )
    risk_cfg = dict(batched_cfg.get("risk_audit") or {})
    if (
        risk_cfg.get("protocol") != RISK_AUDIT_PROTOCOL
        or risk_cfg.get("role")
        != "preregistered_stratified_audit_not_population_safety_claim"
    ):
        raise RuntimeError("PM-v1.5 config lacks the bounded risk-audit contract")
    risk_units = select_stratified_units(
        scoring_units,
        units_per_user=int(risk_cfg["units_per_user"]),
        seed=int(risk_cfg["selection_seed"]),
    )
    risk_endpoint_names = [str(value) for value in risk_cfg["judge_endpoints"]]
    if set(risk_endpoint_names) != set(endpoint_by_name):
        raise RuntimeError("stratified risk audit must use both frozen judge endpoints")
    risk_families = [str(endpoint_by_name[name]["family"]) for name in risk_endpoint_names]
    batched_quality = {
        "primary_judge_family": str(endpoint_by_name[primary_endpoint_name]["family"]),
        "primary_judge_count_per_unit": 1,
        "full_two_family_score_pooling": False,
        "primary_scope": str(batched_cfg["primary_scope"]),
        "primary_order_variant": int(batched_cfg["primary_order_variant"]),
        "expected_primary_units": len(scoring_units),
        "sensitivity_judge_family": str(
            endpoint_by_name[sensitivity_endpoint_name]["family"]
        ),
        "sensitivity_role": str(batched_cfg["sensitivity_role"]),
        "sensitivity_units_per_user": int(
            batched_cfg["sensitivity_units_per_user"]
        ),
        "sensitivity_selection_seed": int(
            batched_cfg["sensitivity_selection_seed"]
        ),
        "sensitivity_order_variant": int(
            batched_cfg["sensitivity_order_variant"]
        ),
        "sensitivity_units": [list(unit) for unit in sensitivity_units],
        "sensitivity_units_sha256": sha256_text(
            canonical_json([list(unit) for unit in sensitivity_units])
        ),
        "expected_sensitivity_units": len(sensitivity_units),
        "estimated_output_tokens_per_call": int(
            batched_cfg["estimated_quality_output_tokens_per_call"]
        ),
    }
    batched_risk_audit = {
        "protocol": RISK_AUDIT_PROTOCOL,
        "role": str(risk_cfg["role"]),
        "units_per_user": int(risk_cfg["units_per_user"]),
        "selection_seed": int(risk_cfg["selection_seed"]),
        "order_variant": int(risk_cfg["order_variant"]),
        "judge_families": risk_families,
        "units": [list(unit) for unit in risk_units],
        "units_sha256": sha256_text(
            canonical_json([list(unit) for unit in risk_units])
        ),
        "expected_units": len(risk_units),
        "estimated_output_tokens_per_call": int(
            risk_cfg["estimated_output_tokens_per_call"]
        ),
        "primary_comparator": str(risk_cfg["primary_comparator"]),
        "nonincrease_margin": float(risk_cfg["nonincrease_margin"]),
        "reliable_mad_threshold": float(risk_cfg["reliable_mad_threshold"]),
        "minimum_low_mad_coverage": float(
            risk_cfg["minimum_low_mad_coverage"]
        ),
    }
    batched_evaluation = {
        "protocol": BATCHED_EXTERNAL_PROTOCOL,
        "candidate_count": len(PM_V1_5_CONDITIONS),
        "conditions_sha256": sha256_text(
            canonical_json(list(PM_V1_5_CONDITIONS))
        ),
        "llm_overall_requested": False,
        "batched_prompt_contract_sha256": batched_prompt_contract_hash(),
        "judge_seed": judge_seed,
        "expected_scoring_units": len(scoring_units),
        "expected_api_calls": (
            len(scoring_units)
            + len(sensitivity_units)
            + len(risk_units) * len(risk_families)
        ),
        "quality": batched_quality,
        "risk_audit": batched_risk_audit,
        "bootstrap": {
            "replicates": int(batched_cfg["bootstrap_replicates"]),
            "confidence_level": float(
                batched_cfg["bootstrap_confidence_level"]
            ),
            "seed": int(batched_cfg["bootstrap_seed"]),
        },
    }
    pilot_cfg = dict(batched_cfg.get("schema_order_pilot") or {})
    batched_schema_order_pilot = {
        "protocol": BATCHED_PILOT_PROTOCOL,
        "purpose": "schema_transport_for_batched_quality_and_risk_not_efficacy",
        "candidate_count": len(PM_V1_5_CONDITIONS),
        "condition_count": len(PM_V1_5_CONDITIONS),
        "unit": list(smoke_unit),
        "unit_id": sha256_text(canonical_json(smoke_unit))[:24],
        "order_variants": [int(value) for value in pilot_cfg["order_variants"]],
        "judge_types": [str(value) for value in pilot_cfg["judge_types"]],
        "expected_calls": int(pilot_cfg["expected_calls"]),
        "judge_endpoints": judge_endpoints,
        "judge_pricing_usd_per_mtok": judge_pricing_usd_per_mtok,
        "api_cost_planning": api_cost_planning,
        "judge_seed": judge_seed + 5000,
        "batched_prompt_contract_sha256": batched_prompt_contract_hash(),
        "forced_swap_selected_units_sha256": forced_swap_selected_units_sha256,
        "required_before_full_external_client_creation": True,
        "estimated_quality_output_tokens": int(
            pilot_cfg["estimated_quality_output_tokens"]
        ),
        "estimated_risk_output_tokens": int(
            pilot_cfg["estimated_risk_output_tokens"]
        ),
    }

    # Fail closed if configs/pm_v1_5.yaml's key_claim_gate has drifted from
    # the single source of truth in v1_5_forced_swap_canary.py -- this is
    # what previously was a bare, misleading
    # require_forced_swap_or_human_check_for_key_claims: true.
    configured_key_claim_gate = dict(external.get("key_claim_gate") or {})
    if configured_key_claim_gate != KEY_CLAIM_GATE:
        raise RuntimeError(
            "PM-v1.5 config key_claim_gate does not match the frozen "
            "src/metacom_pm/v1_5_forced_swap_canary.py:KEY_CLAIM_GATE contract"
        )
    key_claim_gate = dict(KEY_CLAIM_GATE)

    external_evaluation_contract = {
        "evaluation_unit_contract": evaluation_unit_contract,
        "turn_indices": turn_indices,
        "conditions": list(PM_V1_5_CONDITIONS),
        "treatment": PM_V1_5_LEARNED_CONDITION,
        "judge_seed": judge_seed,
        "judge_endpoints": judge_endpoints,
        "judge_pricing_usd_per_mtok": judge_pricing_usd_per_mtok,
        "api_cost_planning": api_cost_planning,
        "primary_bootstrap_cluster": str(external["primary_bootstrap_cluster"]),
        "sensitivity_bootstrap_cluster": str(external["sensitivity_bootstrap_cluster"]),
        "minimum_low_mad_coverage_per_dimension": float(
            external["minimum_low_mad_coverage_per_dimension"]
        ),
        "minimum_low_mad_coverage_per_condition_dimension": float(
            external["minimum_low_mad_coverage_per_condition_dimension"]
        ),
        "maximum_cost_matched_relative_deviation": float(
            external["maximum_cost_matched_relative_deviation"]
        ),
        "composite_support_exact_match_rate": float(
            labeling["composite_support_exact_match_rate"]
        ),
        "maximum_absolute_composite_support_correlation": float(
            labeling["maximum_absolute_composite_support_correlation"]
        ),
        "quality_composite": {
            "version": composite_spec_from_config(pm_v1_5_config).version,
            "weights": dict(composite_spec_from_config(pm_v1_5_config).weights),
        },
        "claim_assessment": dict(external["claim_assessment"]),
        "latency_diagnostic": {
            "protocol": LATENCY_DIAGNOSTIC_PROTOCOL,
            "role": str(external["latency_role"]),
            "confirmatory_latency_claim_allowed": False,
        },
        "key_claim_gate": key_claim_gate,
        "forced_swap": forced_swap_contract,
        "pointwise_schema_smoke": pointwise_schema_smoke,
        "batched_schema_order_pilot": batched_schema_order_pilot,
        "batched_evaluation": batched_evaluation,
        # Held out of the main scored set below (see smoke_unit above).
        "excluded_units": [list(unit) for unit in forced_units],
    }

    notes = {
        "track": "pm-v1.5-fast-track-v2",
        "human_calibration_performed": False,
        "generation_contract": generation_contract,
        "external_evaluation_contract": external_evaluation_contract,
        "checkpoint_sha256": checkpoint_sha256,
        "fixed_baselines": fixed_report["baselines"],
        "fixed_seeker_tracks": fixed_tracks_verification,
        "retrieval_consistency": retrieval_consistency,
        "bank_seed_lineage": bank_seed_lineage,
        "full_development_chain": development_chain,
        "decision_quality": {
            "report_sha256": sha256_file(args.decision_quality_report),
            "attestation_sha256": decision_quality_verification[
                "attestation_sha256"
            ],
            "protocol": decision_quality_report["protocol"],
            "split": decision_quality_report["split"],
            "claim_boundary": decision_quality_report["claim_boundary"],
        },
        "development_lineage": {
            "data_attestation_sha256": development_chain[
                "data_attestation_sha256"
            ],
            "sweep_attestation_sha256": development_chain[
                "sweep_attestation_sha256"
            ],
            "judging_attestation_sha256": development_chain[
                "judging_attestation_sha256"
            ],
            "states_sha256": sha256_file(args.states),
            "labels_sha256": sha256_file(args.judging_labels),
        },
    }

    frozen = create_study_freeze(
        release_root=ROOT,
        # PM-v2.2's own script 26 binds config_path to the general
        # experiment.yaml (endpoints), not pm_v2.yaml -- required_files below
        # separately covers args.pm_v2_config. Match that convention so
        # scripts/v1_5/24* 's `require_study_freeze(config_path=args.config, ...)`
        # (args.config also defaults to experiment.yaml) matches this freeze.
        config_path=args.config,
        checkpoint_paths=[
            args.pm_checkpoint,
            args.cost_matched_fixed_checkpoint,
            args.me_r0_fixed_checkpoint,
        ],
        data_paths=[
            args.pm_v1_5_config,
            args.esconv,
            args.evoemo,
            args.strategy_bank,
            args.strategy_bank_audit,
            args.split_manifest,
            args.seed_dialogues,
            args.seed_audit,
            args.turn_overlap_audit,
            args.fixed_tracks,
            args.fixed_tracks_attestation,
            args.pm_training_report,
            args.fixed_baselines_report,
            args.decision_quality_report,
            args.decision_quality_attestation,
            args.development_data_attestation,
            args.development_data_report,
            args.states,
            args.sweep_summary,
            args.sweep_attestation,
            args.judging_summary,
            args.judging_labels,
            args.judging_attestation,
        ],
        prompt_files=[ROOT / "src" / "metacom_pm" / "prompts.py"],
        out_path=args.out,
        notes=notes,
    )
    print({"status": frozen["status"], "freeze_sha256": frozen["freeze_sha256"], "out": str(args.out)})


if __name__ == "__main__":
    main()
