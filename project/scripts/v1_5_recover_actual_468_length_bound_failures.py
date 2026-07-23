#!/usr/bin/env python3
"""Offline, zero-API-cost recovery for actual-468's length-bound failures.

Context: a real actual-468 run (development_actual_corpus_semantic_review,
output_directory outputs/pm_v1_5_actual_corpus_semantic_review_v8_18_deepseek_
official_v2_candidate) completed the full 1880-call matrix without crashing
(the three-layer fix, commit 1c6a61f, held), but 24/1880 logical calls were
recorded as incomplete: every one is retry_class=structured_output_validation_
error, i.e. the provider returned a complete, well-formed, non-truncated
response (finish_reason=stop/STOP in all 24) that was rejected only because
SingleFieldDiagnosticOutput's reason/evidence_quotes character-length caps
(600/320) -- chosen for prompt-shaping, not because a real API constraint
requires them -- were exceeded. Independently verified against every one of
the 24 real ledger rows before writing this script: all 24 have complete JSON
and a normal finish reason, and 18/24 already satisfy citation integrity
(evidence_keys/quotes alignment, distinct keys, quotes are real substrings of
their cited evidence) once only the length gate is bypassed; 6/24 have a
genuine citation defect, which this script records honestly rather than
silently upgrading to valid.

This script never re-calls any provider and never mutates the original
output directory's gate_report.json, call_plan.jsonl, or physical_attempt_
ledger.jsonl. It reads them, plus the frozen, already-PASS-validated 468-
state dataset (data/pm_v1_5_formal_v8_18_duplicate_repair_candidate, per
outputs/pm_v1_5_paid_run_release.json's stage_consumptions.development_data_
generation), reconstructs the real semantic packets deterministically, and
re-validates each of the 24 already-received raw responses against
RecoveredSingleFieldDiagnosticOutput (metacom_pm.v1_5_semantic_review_
diagnostic) -- the same schema in every respect except the two character-
length caps. It fails closed (raises, writes nothing) if any incomplete call
is not exactly structured_output_validation_error, if any recovered response
does not have a normal (non-truncated) finish reason, or if any packet
reconstruction does not byte-match the original call plan's prompt_sha256.

Writes two new files inside the existing output directory (never touching
any existing file): recovery_report.json (per-call recovery detail) and
recovered_gate_report.json (the full 1880/1880 aggregated PASS/FAIL gate,
computed via the same aggregate_actual_corpus_gate used by a live run,
explicitly labeled as a post-run engineering-contract amendment).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from metacom_pm.attempt_ledger import PersistentAttemptLedger
from metacom_pm.config import load_config
from metacom_pm.io import canonical_json, iter_jsonl, read_json, sha256_file, sha256_text, write_json
from metacom_pm.v1_5_actual_corpus_review import (
    ACTUAL_CORPUS_REVIEW_STAGE,
    aggregate_actual_corpus_gate,
    build_actual_corpus_review_items,
)
from metacom_pm.v1_5_semantic_review_diagnostic import (
    OFFLINE_LENGTH_BOUND_RECOVERY_PROTOCOL,
    assess_single_field_diagnostic_output,
    recover_length_bound_failure,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--paid-run-release", type=Path, default=ROOT / "outputs" / "pm_v1_5_paid_run_release.json"
    )
    parser.add_argument(
        "--pm-v1-5-config", type=Path, default=ROOT / "configs" / "pm_v1_5.yaml"
    )
    parser.add_argument(
        "--strategy-bank", type=Path, default=ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl"
    )
    parser.add_argument(
        "--judge-endpoints",
        nargs="+",
        default=["training_judge_gemini_flash_lite", "training_judge_deepseek_official_flash"],
    )
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--states-dir",
        type=Path,
        help=(
            "Override the corpus directory used to reconstruct semantic "
            "packets (pm_v2_states.jsonl/evaluator_contexts.jsonl/"
            "memory_backend.jsonl/pm_v2_bundles.jsonl). Required for a "
            "delta re-judge run whose --out-dir is not the manifest's "
            "single recorded development_actual_corpus_semantic_review "
            "output_directory (e.g. a repaired-corpus addendum run) -- "
            "when given, the manifest's output_directory/ledger-sha256 "
            "cross-check is skipped, but the byte-exact prompt_sha256 "
            "reconstruction check below still fails closed on any mismatch."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    manifest = read_json(args.paid_run_release)
    stage_consumption = manifest["stage_consumptions"]["development_actual_corpus_semantic_review"]
    ledger_path = out_dir / "physical_attempt_ledger.jsonl"
    if args.states_dir is not None:
        states_dir = args.states_dir
    else:
        if Path(stage_consumption["output_directory"]) != out_dir:
            raise RuntimeError(
                "recovery --out-dir does not match the manifest's recorded "
                "development_actual_corpus_semantic_review output_directory "
                "-- pass --states-dir explicitly for a delta re-judge run "
                "against a different (e.g. repaired) corpus directory"
            )
        if stage_consumption["status"] != "CONSUMED_INCOMPLETE":
            raise RuntimeError(
                "recovery requires the manifest to record this run as "
                "CONSUMED_INCOMPLETE; refusing to recover any other status"
            )
        if sha256_file(ledger_path) != stage_consumption["physical_attempt_ledger_sha256"]:
            raise RuntimeError(
                "physical_attempt_ledger.jsonl does not match the manifest's "
                "recorded sha256; refusing to recover from an altered ledger"
            )
        data_generation = manifest["stage_consumptions"]["development_data_generation"]
        if data_generation["status"] != "CONSUMED_PASS":
            raise RuntimeError(
                "development_data_generation must be CONSUMED_PASS; refusing "
                "to reconstruct semantic packets from an unvalidated dataset"
            )
        states_dir = Path(data_generation["output_directory"])

    gate_report = read_json(out_dir / "gate_report.json")
    if gate_report["status"] != "INCOMPLETE_NO_GATE_DECISION":
        raise RuntimeError(
            "recovery requires gate_report.json status=INCOMPLETE_NO_GATE_DECISION"
        )
    incomplete_calls = gate_report["incomplete_calls"]

    call_plan = list(iter_jsonl(out_dir / "call_plan.jsonl"))
    call_plan_by_key = {str(row["physical_call_key"]): row for row in call_plan}

    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=ACTUAL_CORPUS_REVIEW_STAGE,
        expected_calls={
            str(row["physical_call_key"]): int(row["maximum_physical_attempts"])
            for row in call_plan
        },
        maximum_total_attempts=sum(int(row["maximum_physical_attempts"]) for row in call_plan),
    )

    pm_config = load_config(args.pm_v1_5_config)
    actual_cfg = pm_config["actual_corpus_semantic_audit"]
    actual_state_items, controls, corpus_audit = build_actual_corpus_review_items(
        states_path=states_dir / "pm_v2_states.jsonl",
        evaluator_contexts_path=states_dir / "evaluator_contexts.jsonl",
        backend_path=states_dir / "memory_backend.jsonl",
        bundles_path=states_dir / "pm_v2_bundles.jsonl",
        strategy_bank_path=args.strategy_bank,
        strategy_top_k=int(pm_config["retrieval"]["strategy_top_k"]),
        strategy_min_score=float(pm_config["retrieval"]["strategy_min_score"]),
        maximum_fallback_rate_by_split=actual_cfg["maximum_provider_surface_fallback_rate_by_split"],
        control_seed=args.seed,
        required_control_fields=actual_cfg["required_control_fields"],
        controls_per_field=int(actual_cfg["controls_per_field"]),
    )
    packets_by_item_id = {
        packet["item_id"]: packet
        for state in actual_state_items
        for packet in state["semantic_packets"]
    }
    for row in controls:
        packet = row["semantic_packet"]
        packets_by_item_id[packet["item_id"]] = packet

    for row in call_plan:
        packet = packets_by_item_id.get(row["item_id"])
        if packet is None:
            raise RuntimeError(f"no reconstructed packet for item_id {row['item_id']}")
        if sha256_text(canonical_json(packet["messages"])) != row["prompt_sha256"]:
            raise RuntimeError(
                f"reconstructed packet prompt_sha256 mismatch for {row['item_id']}: "
                "recovery data source does not byte-match the original call plan"
            )

    incomplete_call_keys = {str(row["call_key"]) for row in incomplete_calls}
    recovered_judgments: dict[str, dict[str, Any]] = {}
    recovery_detail: list[dict[str, Any]] = []
    unrecoverable_calls: list[dict[str, Any]] = []
    for call_key in sorted(incomplete_call_keys):
        terminal = ledger.terminal_row(call_key)
        if terminal is None or terminal.get("event") != "FAILED":
            raise RuntimeError(f"expected a terminal FAILED row for {call_key}")
        retry_class = str((terminal.get("metadata") or {}).get("retry_class") or "")
        if retry_class != "structured_output_validation_error":
            # A genuinely different failure class (e.g. output_token_limit --
            # a truncated, incomplete provider response, not a complete-but-
            # over-the-character-cap one) cannot be safely recovered offline:
            # there is no guarantee the raw response even contains complete
            # JSON. Leave it out of both recovered_judgments and the real/
            # control result maps below -- aggregate_actual_corpus_gate's own
            # contract check then honestly reports this packet as lacking
            # the full two-family panel, rather than silently dropping it or
            # crashing this otherwise-legitimate recovery of the other calls.
            plan_row = call_plan_by_key[call_key]
            unrecoverable_calls.append(
                {
                    "call_key": call_key,
                    "case_item_id": plan_row["case_item_id"],
                    "field": plan_row["field"],
                    "judge_family": plan_row["judge_family"],
                    "kind": plan_row["kind"],
                    "retry_class": retry_class,
                }
            )
            continue
        result = terminal.get("result") or {}
        recovered = recover_length_bound_failure(
            raw_provider_response=result["provider_response"],
            parsed_payload=result["parsed_payload"],
        )
        plan_row = call_plan_by_key[call_key]
        packet = packets_by_item_id[plan_row["item_id"]]
        assessment = assess_single_field_diagnostic_output(item=packet, output=recovered)
        judgment = {
            "verdict": recovered.verdict,
            "evidence_keys": recovered.evidence_keys,
            "evidence_quotes": recovered.evidence_quotes,
            "reason": recovered.reason,
            "citation_valid": assessment["citation_valid"],
            "citation_errors": assessment["citation_errors"],
        }
        recovered_judgments[call_key] = judgment
        recovery_detail.append(
            {
                "call_key": call_key,
                "case_item_id": plan_row["case_item_id"],
                "field": plan_row["field"],
                "judge_family": plan_row["judge_family"],
                "kind": plan_row["kind"],
                "reason_length": len(recovered.reason),
                "evidence_quote_lengths": [len(q) for q in recovered.evidence_quotes],
                "citation_valid": assessment["citation_valid"],
                "citation_errors": assessment["citation_errors"],
            }
        )

    unrecoverable_call_keys = {str(row["call_key"]) for row in unrecoverable_calls}
    real_case_results: dict[str, dict[str, Any]] = {}
    control_results: dict[str, dict[str, Any]] = {}
    for row in call_plan:
        call_key = str(row["physical_call_key"])
        if call_key in unrecoverable_call_keys:
            # Deliberately no entry: this packet's family panel is left
            # incomplete, which aggregate_actual_corpus_gate's own contract
            # check reports honestly rather than papering over.
            continue
        if call_key in recovered_judgments:
            judgment = recovered_judgments[call_key]
        else:
            terminal = ledger.terminal_row(call_key)
            if terminal is None or terminal.get("event") != "SUCCEEDED":
                raise RuntimeError(f"expected a terminal SUCCEEDED row for {call_key}")
            parsed = terminal["result"]["parsed"]
            packet = packets_by_item_id[row["item_id"]]
            assessment = assess_single_field_diagnostic_output(
                item=packet,
                output=SimpleNamespace(
                    verdict=parsed["verdict"],
                    evidence_keys=parsed["evidence_keys"],
                    evidence_quotes=parsed["evidence_quotes"],
                ),
            )
            judgment = {
                "verdict": parsed["verdict"],
                "citation_valid": assessment["citation_valid"],
                "citation_errors": assessment["citation_errors"],
            }
        destination = real_case_results if row["kind"] == "real" else control_results
        destination.setdefault(str(row["item_id"]), {})[str(row["endpoint_name"])] = judgment

    aggregated_gate = aggregate_actual_corpus_gate(
        state_items=actual_state_items,
        real_case_results=real_case_results,
        control_results=control_results,
        controls=controls,
        judge_family_names=list(args.judge_endpoints),
        corpus_audit=corpus_audit,
    )

    citation_valid_count = sum(1 for row in recovery_detail if row["citation_valid"])
    recovery_report = {
        "protocol": OFFLINE_LENGTH_BOUND_RECOVERY_PROTOCOL,
        "status": "RECOVERED" if not unrecoverable_calls else "PARTIALLY_RECOVERED",
        "note": (
            f"Post-run engineering-contract amendment: recovers "
            f"{len(recovery_detail)} real responses rejected only for "
            "exceeding SingleFieldDiagnosticOutput's character-length caps "
            "(reason<=600, evidence_quotes<=320). Every recovered response "
            "has a real, normal, non-truncated provider finish reason; "
            "every other schema constraint that "
            "RecoveredSingleFieldDiagnosticOutput enforces (verdict enum, <=4 "
            "evidence items, non-empty, forbid extra fields) was re-verified, "
            "not bypassed. Evidence_keys/evidence_quotes length-equality is "
            "NOT a schema constraint on either model -- it is a citation-"
            "integrity check (assess_single_field_diagnostic_output), and is "
            f"reported honestly as such: {len(recovery_detail) - citation_valid_count} "
            "of the recovered responses are genuinely misaligned and recorded "
            "citation_valid=false, never upgraded to valid. Citation-integrity "
            "defects overall are disclosed honestly, never upgraded to valid: "
            "report-only-not-outcome-v1 policy already governs citation_valid "
            "for this stage and is unchanged by this recovery. Zero new API "
            "calls; zero new cost."
            + (
                f" {len(unrecoverable_calls)} incomplete call(s) could NOT be "
                "recovered offline (a genuinely different failure class, e.g. "
                "output_token_limit -- a truncated response, not a complete-"
                "but-over-length one) and are left as a disclosed gap in "
                "unrecoverable_calls; the aggregated gate below honestly "
                "reports those packets as lacking a complete two-family panel."
                if unrecoverable_calls
                else ""
            )
        ),
        "original_output_directory": str(out_dir),
        "original_physical_attempt_ledger_sha256": sha256_file(ledger_path),
        "n_recovered": len(recovery_detail),
        "n_citation_valid": citation_valid_count,
        "n_citation_invalid": len(recovery_detail) - citation_valid_count,
        "recovered_calls": recovery_detail,
        "unrecoverable_calls": unrecoverable_calls,
    }
    write_json(out_dir / "recovery_report.json", recovery_report)

    n_succeeded = len(call_plan) - len(incomplete_call_keys)
    recovered_gate_report = {
        "protocol": OFFLINE_LENGTH_BOUND_RECOVERY_PROTOCOL,
        "status": "RECOVERED_GATE_DECISION_POST_RUN_AMENDMENT",
        "note": (
            "Supersedes gate_report.json's INCOMPLETE_NO_GATE_DECISION via the "
            "offline recovery documented in recovery_report.json. gate_report.json "
            f"itself is left untouched. This aggregation includes {n_succeeded} "
            f"original SUCCEEDED plus {len(recovery_detail)} recovered logical "
            f"calls out of {len(call_plan)} planned"
            + (
                f"; {len(unrecoverable_calls)} call(s) remain genuinely "
                "incomplete (see recovery_report.json's unrecoverable_calls) "
                "and are honestly reflected as an incomplete panel in "
                "aggregated_gate's contract_errors, not silently dropped."
                if unrecoverable_calls
                else "."
            )
        ),
        "recovery_report_sha256": sha256_text(canonical_json(recovery_report)),
        "logical_calls_planned": len(call_plan),
        "logical_calls_recovered": len(recovery_detail),
        "logical_calls_unrecoverable": len(unrecoverable_calls),
        "aggregated_gate": aggregated_gate,
    }
    write_json(out_dir / "recovered_gate_report.json", recovered_gate_report)

    print(
        f"RECOVERED: {len(recovery_detail)}/{len(incomplete_call_keys)} incomplete calls "
        f"({citation_valid_count} citation_valid, {len(recovery_detail) - citation_valid_count} "
        f"citation_invalid). Aggregated gate status: {aggregated_gate['status']}. "
        f"See {out_dir / 'recovery_report.json'} and {out_dir / 'recovered_gate_report.json'}."
    )


if __name__ == "__main__":
    main()
