from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .artifacts import require_artifact_attestation
from .io import canonical_json, iter_jsonl, read_json, sha256_file, sha256_text
from .v1_5_actual_corpus_review import (
    ACTUAL_CITATION_POLICY,
    ACTUAL_CORPUS_CONTROL_PROTOCOL,
    ACTUAL_CORPUS_REVIEW_PROTOCOL,
    ACTUAL_DETERMINISTIC_FIELDS,
    ACTUAL_PANEL_POLICY,
    ACTUAL_SEMANTIC_FIELDS,
)


ACTUAL_CORPUS_QUALIFICATION_PROTOCOL = (
    "pm-v1.5-actual-468-posthoc-instrument-qualification-v1"
)
ACTUAL_CORPUS_QUALIFICATION_STAGE = (
    "pm_v1_5_actual_corpus_posthoc_instrument_qualification"
)
ACTUAL_CORPUS_QUALIFIED_STATUS = (
    "QUALIFIED_DATA_CORPUS_WITH_DISCLOSED_INSTRUMENT_LIMITATIONS"
)

# These are not tunable thresholds. They identify the two residual incidents
# that were already observed and frozen before this downstream qualification
# contract was introduced.
FROZEN_CONTROL_MISS_ITEM_ID = (
    "state_060702683f2aff4e846fdf77__control_context_grounding_match_1"
    "::context_grounding_match"
)
FROZEN_UNRECOVERABLE_CALL = {
    "case_item_id": "state_9c90150e2b367dfa8ea7f277",
    "field": "context_grounding_match",
    "judge_family": "deepseek_official",
    "kind": "real",
    "retry_class": "output_token_limit",
}
FROZEN_UNRECOVERABLE_CONTRACT_ERRORS = {
    "state_9c90150e2b367dfa8ea7f277::context_grounding_match lacks the exact two-family panel",
    "state_9c90150e2b367dfa8ea7f277::context_grounding_match has invalid semantic verdicts",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _record_sha(record: Any, *, name: str) -> str:
    _require(isinstance(record, Mapping), f"missing attestation record: {name}")
    digest = str(record.get("sha256") or "")
    _require(len(digest) == 64, f"invalid attestation digest: {name}")
    return digest


def build_actual_corpus_posthoc_qualification(
    *,
    recovered_gate_report_path: str | Path,
    recovery_report_path: str | Path,
    incomplete_gate_report_path: str | Path,
    physical_attempt_ledger_path: str | Path,
    cost_estimate_path: str | Path,
    call_plan_path: str | Path,
    recompile_verification_path: str | Path,
    development_data_attestation_path: str | Path,
    classification_path: str | Path,
    repair_overlays_path: str | Path,
    expected_states_path: str | Path,
    expected_evaluator_contexts_path: str | Path,
    expected_backend_path: str | Path,
    expected_runtime_states_path: str | Path,
    expected_bundles_path: str | Path,
    expected_data_report_path: str | Path,
    expected_strategy_bank_path: str | Path,
    expected_pm_config_path: str | Path,
) -> dict[str, Any]:
    """Qualify the repaired corpus without relabelling the failed audit PASS.

    This is deliberately narrower than the original semantic-review gate. It
    authorizes downstream development only for one already-observed corpus and
    only when its two frozen instrument limitations are exactly reproduced.
    """

    recovered = read_json(recovered_gate_report_path)
    recovery = read_json(recovery_report_path)
    incomplete = read_json(incomplete_gate_report_path)
    cost = read_json(cost_estimate_path)
    recompile = read_json(recompile_verification_path)
    data_attestation = read_json(development_data_attestation_path)
    aggregate = recovered.get("aggregated_gate") or {}

    _require(
        recovered.get("protocol")
        == "pm-v1.5-actual-468-offline-length-bound-recovery-v1"
        and recovered.get("status")
        == "RECOVERED_GATE_DECISION_POST_RUN_AMENDMENT",
        "qualification requires the exact recovered actual-468 report",
    )
    _require(
        recovery.get("protocol")
        == "pm-v1.5-actual-468-offline-length-bound-recovery-v1"
        and recovery.get("status") == "PARTIALLY_RECOVERED",
        "qualification requires the exact partial-recovery report",
    )
    _require(
        recovered.get("recovery_report_sha256")
        == sha256_text(canonical_json(recovery)),
        "recovered gate does not bind the recovery report",
    )
    _require(
        incomplete.get("protocol") == ACTUAL_CORPUS_REVIEW_PROTOCOL
        and incomplete.get("status") == "INCOMPLETE_NO_GATE_DECISION"
        and int(incomplete.get("logical_calls_planned") or 0) == 1880,
        "qualification requires the original incomplete 1880-call gate report",
    )
    _require(
        aggregate.get("protocol") == ACTUAL_CORPUS_REVIEW_PROTOCOL
        and aggregate.get("status") == "FAIL",
        "the original recovered audit must remain explicitly FAIL",
    )
    _require(
        int(aggregate.get("n_real_cases") or 0) == 468
        and int(aggregate.get("n_real_semantic_packets") or 0) == 936
        and int(aggregate.get("n_controls") or 0) == 4,
        "actual-468 corpus/control cardinality drifted",
    )
    _require(
        aggregate.get("control_protocol") == ACTUAL_CORPUS_CONTROL_PROTOCOL
        and aggregate.get("panel_policy") == ACTUAL_PANEL_POLICY
        and aggregate.get("citation_policy") == ACTUAL_CITATION_POLICY
        and list(aggregate.get("semantic_fields") or [])
        == list(ACTUAL_SEMANTIC_FIELDS),
        "actual-468 measurement contract drifted",
    )
    _require(
        not list(aggregate.get("real_case_rejections") or []),
        "qualification forbids any unanimous real-case rejection",
    )
    deterministic = aggregate.get("deterministic_gate") or {}
    _require(
        deterministic.get("status") == "PASS"
        and int(deterministic.get("n_states") or 0) == 468
        and int(deterministic.get("n_failures") or 0) == 0
        and not list(deterministic.get("failures") or [])
        and set((deterministic.get("field_pass_counts") or {}).keys())
        == set(ACTUAL_DETERMINISTIC_FIELDS)
        and all(
            int(value) == 468
            for value in (deterministic.get("field_pass_counts") or {}).values()
        ),
        "deterministic actual-468 checks did not all PASS",
    )

    misses = list(aggregate.get("control_misses") or [])
    _require(
        len(misses) == 1
        and misses[0].get("item_id") == FROZEN_CONTROL_MISS_ITEM_ID
        and misses[0].get("field") == "context_grounding_match"
        and set((misses[0].get("verdicts") or {}).values()) == {"supported"},
        "control residual differs from the one frozen disclosed miss",
    )
    _require(
        set(aggregate.get("control_contract_errors") or [])
        == FROZEN_UNRECOVERABLE_CONTRACT_ERRORS,
        "contract errors differ from the frozen provider truncation",
    )
    unrecoverable = list(recovery.get("unrecoverable_calls") or [])
    _require(len(unrecoverable) == 1, "exactly one unrecoverable call is allowed")
    for key, expected in FROZEN_UNRECOVERABLE_CALL.items():
        _require(
            unrecoverable[0].get(key) == expected,
            f"unrecoverable call drifted at {key}",
        )
    _require(
        int(recovered.get("logical_calls_planned") or 0) == 1880
        and int(recovered.get("logical_calls_recovered") or 0) == 11
        and int(recovered.get("logical_calls_unrecoverable") or 0) == 1,
        "offline recovery counts drifted",
    )

    _require(
        cost.get("protocol") == ACTUAL_CORPUS_REVIEW_PROTOCOL
        and cost.get("stage") == "pm_v1_5_actual_corpus_semantic_review"
        and cost.get("review_scope") == "actual_468"
        and int(cost.get("n_logical_calls") or 0) == 1880
        and cost.get("budget_gate", {}).get("status") == "PASS",
        "actual-468 cost/call contract drifted",
    )
    call_plan = list(iter_jsonl(call_plan_path))
    _require(len(call_plan) == 1880, "actual-468 call plan must contain 1880 rows")
    _require(
        cost.get("call_plan_sha256") == sha256_text(canonical_json(call_plan)),
        "actual-468 call plan digest drifted",
    )
    _require(Path(physical_attempt_ledger_path).is_file(), "missing physical ledger")

    _require(
        recompile.get("protocol")
        == "pm-v1.5-context-grounding-repair-recompile-verification-v1"
        and recompile.get("status") == "PASS"
        and int(recompile.get("total_states") or 0) == 468
        and int(recompile.get("repaired_state_count") or 0) == 25
        and int(recompile.get("untouched_state_count") or 0) == 443
        and not dict(recompile.get("unexpectedly_changed_untouched_states") or {})
        and not list(recompile.get("unexpectedly_unchanged_repaired_states") or []),
        "25-state repair recompilation attestation did not PASS exactly",
    )
    _require(
        recompile.get("classification_sha256") == sha256_file(classification_path)
        and recompile.get("repair_overlays_sha256")
        == sha256_file(repair_overlays_path),
        "repair recompilation does not bind classification/overlays",
    )

    classification = list(iter_jsonl(classification_path))
    class_counts = Counter(str(row.get("classification")) for row in classification)
    _require(
        len(classification) == 91
        and class_counts == {"DATA_DEFECT": 25, "INSTRUMENT_AMBIGUITY": 66}
        and len({str(row.get("state_id")) for row in classification}) == 91,
        "frozen defect classification must be exactly 25 defects + 66 ambiguities",
    )
    overlays = list(iter_jsonl(repair_overlays_path))
    _require(
        len(overlays) == 25
        and len({str(row.get("state_id")) for row in overlays}) == 25,
        "repair overlay must contain exactly 25 unique repaired states",
    )

    required_data_outputs = {
        "states": expected_states_path,
        "evaluator_contexts": expected_evaluator_contexts_path,
        "backend": expected_backend_path,
        "runtime_states": expected_runtime_states_path,
        "bundles": expected_bundles_path,
        "data_report": expected_data_report_path,
    }
    require_artifact_attestation(
        development_data_attestation_path,
        required_stage="pm_v1_5_development_data",
        required_output_paths=required_data_outputs,
    )
    outputs = data_attestation.get("outputs") or {}
    output_hashes = recompile.get("output_file_sha256s") or {}
    output_filename = {
        "states": "pm_v2_states.jsonl",
        "evaluator_contexts": "evaluator_contexts.jsonl",
        "backend": "memory_backend.jsonl",
        "runtime_states": "runtime_states.jsonl",
        "bundles": "pm_v2_bundles.jsonl",
        "data_report": "pm_v2_data_report.json",
    }
    for name, path in required_data_outputs.items():
        digest = sha256_file(path)
        _require(_record_sha(outputs.get(name), name=name) == digest, f"data output drift: {name}")
        _require(
            output_hashes.get(output_filename[name]) == digest,
            f"recompile output digest drift: {name}",
        )
    inputs = data_attestation.get("inputs") or {}
    _require(
        _record_sha(inputs.get("classification"), name="classification")
        == sha256_file(classification_path)
        and _record_sha(inputs.get("repair_overlays"), name="repair_overlays")
        == sha256_file(repair_overlays_path)
        and _record_sha(inputs.get("strategy_bank"), name="strategy_bank")
        == sha256_file(expected_strategy_bank_path)
        and _record_sha(inputs.get("pm_v1_5_config"), name="pm_v1_5_config")
        == sha256_file(expected_pm_config_path),
        "development-data attestation input lineage drifted",
    )

    citation = aggregate.get("citation_audit") or {}
    _require(
        citation.get("policy") == ACTUAL_CITATION_POLICY
        and int(citation.get("n_observations") or 0) == 1880
        and int(citation.get("n_valid") or 0) <= 1880,
        "citation report drifted",
    )

    report: dict[str, Any] = {
        "protocol": ACTUAL_CORPUS_QUALIFICATION_PROTOCOL,
        "status": ACTUAL_CORPUS_QUALIFIED_STATUS,
        "post_hoc_instrument_qualification": True,
        "original_gate_status": "FAIL",
        "original_gate_pass_claimed": False,
        "original_gate_report_sha256": sha256_file(recovered_gate_report_path),
        "source_incomplete_gate_report_sha256": sha256_file(incomplete_gate_report_path),
        "recovery_report_sha256": sha256_file(recovery_report_path),
        "physical_attempt_ledger_sha256": sha256_file(physical_attempt_ledger_path),
        "cost_estimate_sha256": sha256_file(cost_estimate_path),
        "call_plan_sha256": sha256_file(call_plan_path),
        "recompile_verification_sha256": sha256_file(recompile_verification_path),
        "development_data_attestation_sha256": sha256_file(
            development_data_attestation_path
        ),
        "classification_sha256": sha256_file(classification_path),
        "repair_overlays_sha256": sha256_file(repair_overlays_path),
        "qualified_corpus": {
            "n_states": 468,
            "n_semantic_packets": 936,
            "unanimous_real_case_rejections": 0,
            "attested_repairs": 25,
            "untouched_states": 443,
            "deterministic_gate_status": "PASS",
        },
        "frozen_qualification_exceptions": {
            "control_miss": misses[0],
            "provider_truncation": unrecoverable[0],
        },
        "reported_nonblocking_measurement_results": {
            "real_case_panel_passes": len(aggregate.get("real_case_panel_passes") or []),
            "real_case_disagreements": len(aggregate.get("real_case_disagreements") or []),
            "citation_policy": citation.get("policy"),
            "citation_observations": int(citation.get("n_observations") or 0),
            "citation_valid": int(citation.get("n_valid") or 0),
            "citation_integrity_rate": float(citation.get("integrity_rate") or 0.0),
        },
        "frozen_stopping_rule": {
            "no_further_prompt_tuning": True,
            "no_further_control_tuning": True,
            "no_threshold_tuning": True,
            "no_further_data_repair": True,
        },
        "authorized_downstream_scope": [
            "longitudinal_action_sweep",
            "longitudinal_action_sweep_judging",
            "dual_domain_training",
        ],
        "forbidden_claims": [
            "the original actual-468 gate passed",
            "the semantic-review instrument is fully validated",
            "citation integrity is an outcome gate",
            "post-hoc qualification is a held-out confirmation",
        ],
        "paper_disclosure_required": (
            "The development corpus was admitted through a post-hoc, frozen "
            "instrument-qualification addendum after the original audit remained "
            "FAIL; one negative-control miss and one provider truncation are disclosed."
        ),
    }
    report["qualification_contract_sha256"] = sha256_text(canonical_json(report))
    return report


def require_actual_corpus_posthoc_qualification(
    report_path: str | Path,
    attestation_path: str | Path,
    *,
    expected_experiment_config_path: str | Path,
    expected_states_path: str | Path,
    expected_evaluator_contexts_path: str | Path,
    expected_backend_path: str | Path,
    expected_strategy_bank_path: str | Path,
    expected_pm_config_path: str | Path,
) -> dict[str, Any]:
    verification = require_artifact_attestation(
        attestation_path,
        required_stage=ACTUAL_CORPUS_QUALIFICATION_STAGE,
        required_output_paths={"qualification_report": report_path},
    )
    report = read_json(report_path)
    without_hash = {k: v for k, v in report.items() if k != "qualification_contract_sha256"}
    _require(
        report.get("qualification_contract_sha256")
        == sha256_text(canonical_json(without_hash)),
        "qualification report self-hash mismatch",
    )
    _require(
        report.get("protocol") == ACTUAL_CORPUS_QUALIFICATION_PROTOCOL
        and report.get("status") == ACTUAL_CORPUS_QUALIFIED_STATUS
        and report.get("post_hoc_instrument_qualification") is True
        and report.get("original_gate_status") == "FAIL"
        and report.get("original_gate_pass_claimed") is False,
        "actual corpus is not post-hoc qualified under the frozen contract",
    )
    _require(
        report.get("qualified_corpus")
        == {
            "n_states": 468,
            "n_semantic_packets": 936,
            "unanimous_real_case_rejections": 0,
            "attested_repairs": 25,
            "untouched_states": 443,
            "deterministic_gate_status": "PASS",
        },
        "qualified corpus cardinality/gates drifted",
    )
    exceptions = report.get("frozen_qualification_exceptions") or {}
    _require(
        (exceptions.get("control_miss") or {}).get("item_id")
        == FROZEN_CONTROL_MISS_ITEM_ID,
        "qualification control exception drifted",
    )
    truncation = exceptions.get("provider_truncation") or {}
    for key, expected in FROZEN_UNRECOVERABLE_CALL.items():
        _require(truncation.get(key) == expected, f"qualification truncation drifted: {key}")
    _require(
        report.get("frozen_stopping_rule")
        == {
            "no_further_prompt_tuning": True,
            "no_further_control_tuning": True,
            "no_threshold_tuning": True,
            "no_further_data_repair": True,
        },
        "post-hoc qualification stopping rule drifted",
    )
    attestation = read_json(attestation_path)
    expected_inputs = {
        "experiment_config": expected_experiment_config_path,
        "states": expected_states_path,
        "evaluator_contexts": expected_evaluator_contexts_path,
        "memory_backend": expected_backend_path,
        "strategy_bank": expected_strategy_bank_path,
        "pm_v1_5_config": expected_pm_config_path,
    }
    for name, path in expected_inputs.items():
        _require(
            _record_sha((attestation.get("inputs") or {}).get(name), name=name)
            == sha256_file(path),
            f"qualification does not bind current {name}",
        )
    return {
        "status": ACTUAL_CORPUS_QUALIFIED_STATUS,
        "mode": "POSTHOC_INSTRUMENT_QUALIFICATION",
        "report": report,
        "report_sha256": sha256_file(report_path),
        "attestation_sha256": verification["attestation_sha256"],
    }
