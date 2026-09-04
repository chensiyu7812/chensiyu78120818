#!/usr/bin/env python3
"""Run the exact full PM-v1.5 state-by-action sweep.

The fast track replaces PM-v2.2's human-only development gates with the
attested multi-family automated semantic review. A V1.5 run must explicitly
pass ``--v1-5-full-sweep-scope``; filtered/ad-hoc and legacy compatibility-
pilot sweeps are rejected so the resulting artifact is honestly labelled
``full`` and contains every legal state-action pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metacom_pm.artifacts import require_artifact_attestation
from metacom_pm.attempt_ledger import forbid_overwrite_of_spent_attempts
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import parse_action_id
from metacom_pm.evidence_filter import EvidenceFilterConfig
from metacom_pm.evidence_filter_model import require_evidence_filter_artifacts
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.sweep import plan_action_sweep, run_action_sweep
from metacom_pm.pm_v2_data import load_evaluator_context_index, load_states
from metacom_pm.pm_v2_audit import audit_deployable_feature_observability
from metacom_pm.pm_v2_development_gate import require_development_pilot_gate
from metacom_pm.pm_v2_judge_schema_smoke import (
    require_development_judge_schema_smoke_pass,
)
from metacom_pm.pm_v1_5_shortcut_audit import (
    require_step0_shortcut_audit_pass,
)
from metacom_pm.pm_v1_5_rule_router import RULE_GRID_DIAGNOSTIC_PROTOCOL
from metacom_pm.pm_v2_semantic_audit import (
    require_pmv2_runtime_state_lineage,
    require_semantic_sanity_pass,
)
from metacom_pm.v1_5_automated_semantic_review import (
    require_automated_semantic_review_pass,
)
from metacom_pm.paid_run_release import require_paid_run_release
from metacom_pm.v1_5_actual_corpus_review import (
    require_actual_corpus_semantic_review_pass,
)


ROOT = Path(__file__).resolve().parents[2]


def _budget_gate(
    estimate: dict,
    *,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
) -> dict:
    checks = {
        "physical_api_attempts": int(estimate["maximum_physical_api_attempts"])
        <= int(max_api_calls),
        "estimated_cost_usd": float(estimate["estimated_cost_usd"])
        <= float(max_estimated_usd),
        "max_input_tokens_per_call": int(estimate["max_input_tokens"])
        <= int(max_input_tokens_per_call),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "limits": {
            "max_physical_api_attempts": int(max_api_calls),
            "max_estimated_usd": float(max_estimated_usd),
            "max_input_tokens_per_call": int(max_input_tokens_per_call),
        },
    }


def _persist_or_validate_dry_run(
    out_dir: Path, current: dict, rows: list[dict]
) -> str:
    """Freeze the exact accepted estimate/plan once an HTTP attempt exists."""

    estimate_path = out_dir / "cost_estimate.json"
    plan_path = out_dir / "call_plan.jsonl"
    ledger_path = out_dir / "physical_attempt_ledger.jsonl"
    spent = ledger_path.is_file() and ledger_path.stat().st_size > 0
    if spent:
        if not estimate_path.is_file() or not plan_path.is_file():
            raise RuntimeError(
                "spent action-sweep ledger freezes the accepted dry-run, but its "
                "estimate or full call plan is missing"
            )
        if read_json(estimate_path) != current or list(iter_jsonl(plan_path)) != rows:
            raise RuntimeError(
                "spent action-sweep ledger freezes the original exact estimate "
                "and full call plan; current dry-run drift is rejected"
            )
        return "VALIDATED_EXISTING"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(estimate_path, current)
    write_jsonl(plan_path, rows)
    return "WRITTEN"


def _require_saved_dry_run(
    out_dir: Path, current: dict, rows: list[dict]
) -> None:
    estimate_path = out_dir / "cost_estimate.json"
    plan_path = out_dir / "call_plan.jsonl"
    if not estimate_path.is_file() or not plan_path.is_file():
        raise RuntimeError(
            "API mode requires a completed matching --dry-run in the same output "
            "directory"
        )
    saved = read_json(estimate_path)
    if saved != current:
        raise RuntimeError(
            "saved action-sweep dry-run does not match current inputs/configuration; "
            "run --dry-run again"
        )
    if list(iter_jsonl(plan_path)) != rows:
        raise RuntimeError("saved action-sweep full call plan is stale")
    if saved.get("budget_gate", {}).get("status") != "PASS":
        raise RuntimeError("saved action-sweep dry-run did not pass its budget gate")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed action sweep: exact dry-run and accepted cost hash before API use."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "experiment.yaml"
    )
    parser.add_argument(
        "--pm-v2-config",
        type=Path,
        default=ROOT / "configs" / "pm_v1_5.yaml",
        help=(
            "Bind a PM-v2 sweep to its preregistered generator and retrieval "
            "settings. Required for data/pm_v1_5 runtime inputs."
        ),
    )
    parser.add_argument(
        "--evidence-filter-checkpoint",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_evidence_filter" / "evidence_filter.joblib",
    )
    parser.add_argument(
        "--evidence-filter-report",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_evidence_filter" / "training_report.json",
    )
    parser.add_argument(
        "--evidence-filter-attestation",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_evidence_filter" / "artifact_attestation.json",
    )
    parser.add_argument(
        "--endpoint",
        "--generator-endpoint",
        dest="endpoint",
        default=None,
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "runtime_states.jsonl",
    )
    parser.add_argument(
        "--backend",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "memory_backend.jsonl",
    )
    parser.add_argument(
        "--strategy-bank",
        type=Path,
        default=ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "outputs" / "pm_v1_5_sweep"
    )
    parser.add_argument("--max-cards", type=int)
    parser.add_argument(
        "--v1-5-full-sweep-scope",
        action="store_true",
        help=(
            "Required PM-v1.5 protocol flag: sweep every state and every "
            "legal action under the automated-review gate and record the "
            "artifact with scope=full."
        ),
    )
    parser.add_argument(
        "--actions",
        help="Optional comma-separated pilot action IDs; full confirmatory runs omit this.",
    )
    parser.add_argument(
        "--pilot-plan",
        type=Path,
        help="Frozen balanced PM-v2 compatibility pilot plan from script 31.",
    )
    parser.add_argument(
        "--completed-pilot-plan",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_development_pilot" / "pilot_plan.json",
        help="Pilot plan whose completed action/judge chain gates a full PM-v2 sweep.",
    )
    parser.add_argument(
        "--pilot-sweep-summary",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_development_pilot" / "summary.json",
    )
    parser.add_argument(
        "--pilot-sweep-attestation",
        type=Path,
        default=(
            ROOT / "outputs" / "pm_v1_5_development_pilot" / "artifact_attestation.json"
        ),
    )
    parser.add_argument(
        "--judge-compatibility-summary",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_development_pilot_judging"
            / "summary.json"
        ),
    )
    parser.add_argument(
        "--judge-compatibility-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_development_pilot_judging"
            / "artifact_attestation.json"
        ),
    )
    parser.add_argument(
        "--pilot-human-spot-check-report",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_development_pilot_human_spot_check"
            / "pilot_human_spot_check_report.json"
        ),
    )
    parser.add_argument(
        "--pilot-human-spot-check-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_development_pilot_human_spot_check"
            / "artifact_attestation.json"
        ),
    )
    parser.add_argument(
        "--judge-schema-smoke-summary",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_development_judge_schema_smoke"
            / "summary.json"
        ),
    )
    parser.add_argument(
        "--judge-schema-smoke-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_development_judge_schema_smoke"
            / "artifact_attestation.json"
        ),
    )
    parser.add_argument(
        "--pm-v2-states",
        type=Path,
        help=(
            "PM-v2 states bound by the human semantic-sanity attestation; defaults "
            "to pm_v2_states.jsonl beside --runtime."
        ),
    )
    parser.add_argument(
        "--evaluator-contexts",
        type=Path,
        help=(
            "Evaluator-only PM-v2 contexts bound by the semantic-sanity "
            "attestation; defaults to evaluator_contexts.jsonl beside --runtime."
        ),
    )
    parser.add_argument(
        "--semantic-sanity-report",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v1_5_semantic_sanity"
        / "semantic_sanity_report.json",
    )
    parser.add_argument(
        "--semantic-sanity-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v1_5_semantic_sanity"
        / "artifact_attestation.json",
    )
    parser.add_argument(
        "--automated-semantic-review-report",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_automated_semantic_review" / "gate_report.json",
        help=(
            "PM-v1.5 replacement for the human semantic-sanity and pilot "
            "spot-check gates: output of "
            "scripts/v1_5_run_automated_semantic_review.py, must show "
            "status=PASS."
        ),
    )
    parser.add_argument(
        "--automated-semantic-review-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v1_5_automated_semantic_review"
        / "artifact_attestation.json",
    )
    parser.add_argument(
        "--generation-pilot-attestation",
        type=Path,
        help=(
            "Exact paid nine-case pilot bound by the automated semantic review; "
            "required for the PM-v1.5 full sweep."
        ),
    )
    parser.add_argument(
        "--actual-corpus-semantic-review-report",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_actual_corpus_semantic_review"
            / "gate_report.json"
        ),
    )
    parser.add_argument(
        "--actual-corpus-semantic-review-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_actual_corpus_semantic_review"
            / "artifact_attestation.json"
        ),
    )
    parser.add_argument(
        "--step0-shortcut-audit-report",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_step0_shortcut_audit"
            / "step0_shortcut_audit.json"
        ),
    )
    parser.add_argument(
        "--step0-shortcut-audit-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_step0_shortcut_audit"
            / "artifact_attestation.json"
        ),
    )
    parser.add_argument(
        "--development-data-attestation",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "artifact_attestation.json",
    )
    parser.add_argument(
        "--rule-grid-preflight-report",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_rule_grid_preflight"
            / "rule_grid_report.json"
        ),
    )
    parser.add_argument(
        "--rule-grid-preflight-attestation",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_rule_grid_preflight"
            / "artifact_attestation.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--request-retries",
        type=int,
        help=(
            "Maximum physical attempts per logical generation call. PM-v2 binds "
            "this to development_sweep.request_retries=1."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=None,
        help="Stop the sweep after the first failed logical generation call.",
    )
    parser.add_argument("--strategy-top-k", type=int)
    parser.add_argument(
        "--memory-min-score",
        type=float,
        help="Exclusive lexical relevance threshold; 0 rejects zero-overlap memory items.",
    )
    parser.add_argument(
        "--strategy-min-score",
        type=float,
        help="Exclusive lexical relevance threshold; 0 rejects zero-overlap strategy cards.",
    )
    parser.add_argument(
        "--input-usd-per-mtok",
        type=float,
        help="Optional exact assertion against PM-v2 frozen sweep pricing.",
    )
    parser.add_argument(
        "--output-usd-per-mtok",
        type=float,
        help="Optional exact assertion against PM-v2 frozen sweep pricing.",
    )
    parser.add_argument(
        "--max-api-calls",
        type=int,
        default=6000,
        help="Maximum physical HTTP attempts, including retries.",
    )
    parser.add_argument("--max-estimated-usd", type=float, default=10.0)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=12000)
    parser.add_argument("--accept-cost-estimate-sha256")
    args = parser.parse_args()

    if args.run and args.overwrite:
        raise RuntimeError("paid PM-v1.5 action-sweep runs prohibit --overwrite")
    if args.pilot_plan is not None:
        raise RuntimeError(
            "PM-v1.5 does not run the PM-v2.2 compatibility-pilot branch; "
            "use --v1-5-full-sweep-scope"
        )
    if args.v1_5_full_sweep_scope and (
        args.max_cards is not None or bool(args.actions)
    ):
        raise RuntimeError(
            "--v1-5-full-sweep-scope cannot be combined with --max-cards "
            "or --actions"
        )

    if (
        args.input_usd_per_mtok is not None
        and args.input_usd_per_mtok < 0
    ) or (
        args.output_usd_per_mtok is not None
        and args.output_usd_per_mtok < 0
    ):
        raise ValueError("pricing must be non-negative")
    if args.max_api_calls <= 0 or args.max_estimated_usd < 0:
        raise ValueError("budget limits must be positive")

    experiment_config = load_config(args.config)
    pm_v2_root = (ROOT / "data" / "pm_v1_5").resolve()
    runtime_has_pm_v2_provenance = any(
        bool((row.get("provenance") or {}).get("pm_v2_state_id"))
        for row in iter_jsonl(args.runtime)
    )
    runtime_is_pm_v2 = (
        pm_v2_root in args.runtime.resolve().parents
        or runtime_has_pm_v2_provenance
    )
    if runtime_is_pm_v2 and not args.v1_5_full_sweep_scope:
        raise RuntimeError(
            "PM-v1.5 runtime requires --v1-5-full-sweep-scope; filtered and "
            "legacy compatibility-pilot sweeps are outside this protocol"
        )
    if runtime_is_pm_v2:
        forbid_overwrite_of_spent_attempts(
            args.out_dir / "physical_attempt_ledger.jsonl",
            overwrite=args.overwrite,
            stage="PM-v2 action sweep",
        )
        if args.pilot_plan is None and (
            args.max_cards is not None or bool(args.actions)
        ):
            raise RuntimeError(
                "PM-v2 filtered sweeps require the exact frozen --pilot-plan; "
                "ad-hoc --actions/--max-cards cannot bypass the full-sweep "
                "development pilot gate"
            )
    if runtime_is_pm_v2 and args.pm_v2_config is None:
        raise RuntimeError(
            "PM-v2 action sweeps must pass --pm-v2-config so generator and "
            "retrieval settings are immutable and included in the cost hash"
        )
    if args.pilot_plan is not None and args.pm_v2_config is None:
        raise RuntimeError("--pilot-plan requires --pm-v2-config")
    if (args.pm_v2_states is not None or args.evaluator_contexts is not None) and (
        args.pm_v2_config is None
    ):
        raise RuntimeError(
            "--pm-v2-states/--evaluator-contexts require --pm-v2-config"
        )
    if args.pilot_plan is not None and (args.max_cards is not None or args.actions):
        raise RuntimeError("--pilot-plan cannot be combined with ad-hoc pilot filters")
    pilot_plan = read_json(args.pilot_plan) if args.pilot_plan is not None else None

    contract_bindings: dict = {}
    pm_v2_states_path = None
    evaluator_contexts_path = None
    deployable_feature_observability = None
    judge_schema_smoke = None
    evidence_filter_config = None
    memory_helpfulness_model = None
    evidence_filter_model_binding = None
    supporter_generation_contract = None
    if args.pm_v2_config is not None:
        pm_config = load_config(args.pm_v2_config)
        require_paid_run_release(
            pm_config,
            config_path=args.pm_v2_config,
            stage="development_action_sweep_generation",
            run=bool(args.run),
            run_identity=args.accept_cost_estimate_sha256,
        )
        if pm_config.get("version") != "pm-v1.5":
            raise RuntimeError("PM-v1.5 action sweep requires a pm-v1.5 config")
        supporter_generation_contract = SupporterGenerationContract.from_config(
            pm_config
        )
        sweep_config = dict(pm_config["development_sweep"])
        if set(sweep_config) != {
            "pricing_usd_per_mtok",
            "seed",
            "request_retries",
            "fail_fast",
        }:
            raise RuntimeError(
                "development_sweep must not duplicate the PM-v2.2 supporter "
                "generation treatment"
            )
        retrieval_config = dict(pm_config["retrieval"])
        # PM-v1.5: Evidence Filter is out of scope and must be disabled
        # identically here and in the external-generation fork
        # (scripts/v1_5/24_run_pm_v2_evoemo_v1_5.py). Leaving this branch's
        # original "must be enabled" requirement in place while the external
        # fork force-disables it would silently make the "ME+RS" trained here
        # a different mechanism (post-retrieval-filtered) from the "ME+RS"
        # evaluated externally (raw retrieval) -- the same class of
        # treatment mismatch this whole V1.5 effort exists to fix.
        evidence_filter_config = EvidenceFilterConfig.from_mapping(
            {**pm_config["evidence_filter"], "enabled": False}
        )
        memory_helpfulness_model = None
        evidence_filter_model_binding = {
            "mode": "disabled_for_pm_v1_5",
            "reason": "Evidence Filter is out of scope for PM-v1.5; PM is a pure pre-retrieval router.",
        }
        sweep_pricing = dict(sweep_config["pricing_usd_per_mtok"])
        if set(sweep_pricing) != {"input", "output"}:
            raise RuntimeError(
                "development_sweep pricing must contain exactly input/output"
            )
        frozen_input_price = float(sweep_pricing["input"])
        frozen_output_price = float(sweep_pricing["output"])
        if frozen_input_price <= 0.0 or frozen_output_price <= 0.0:
            raise RuntimeError("PM-v2 sweep budget-accounting prices must be positive")
        semantic_sanity = None
        runtime_state_lineage = None
        if runtime_is_pm_v2 or pilot_plan is not None:
            pm_v2_states_path = args.pm_v2_states or (
                args.runtime.parent / "pm_v2_states.jsonl"
            )
            evaluator_contexts_path = args.evaluator_contexts or (
                args.runtime.parent / "evaluator_contexts.jsonl"
            )
            audited_states = load_states(pm_v2_states_path)
            runtime_state_lineage = require_pmv2_runtime_state_lineage(
                args.runtime, audited_states
            )
            automated_review_verification = require_automated_semantic_review_pass(
                args.automated_semantic_review_report,
                args.automated_semantic_review_attestation,
                expected_experiment_config_path=args.config,
                expected_pm_config_path=args.pm_v2_config,
                expected_strategy_bank_path=args.strategy_bank,
                expected_generation_pilot_attestation_path=(
                    args.generation_pilot_attestation
                ),
            )
            automated_review_report = automated_review_verification["report"]
            actual_corpus_verification = require_actual_corpus_semantic_review_pass(
                args.actual_corpus_semantic_review_report,
                args.actual_corpus_semantic_review_attestation,
                expected_experiment_config_path=args.config,
                expected_states_path=pm_v2_states_path,
                expected_evaluator_contexts_path=evaluator_contexts_path,
                expected_backend_path=args.backend,
                expected_strategy_bank_path=args.strategy_bank,
                expected_pm_config_path=args.pm_v2_config,
            )
            shortcut_audit_verification = require_step0_shortcut_audit_pass(
                args.step0_shortcut_audit_report,
                args.step0_shortcut_audit_attestation,
                expected_states_path=pm_v2_states_path,
                expected_evaluator_contexts_path=evaluator_contexts_path,
                expected_pm_config_path=args.pm_v2_config,
                expected_generation_attestation_path=(
                    args.development_data_attestation
                ),
            )
            rule_grid_attestation = require_artifact_attestation(
                args.rule_grid_preflight_attestation,
                required_stage="pm_v1_5_pre_training_rule_grid_diagnostic",
                required_output_paths={
                    "rule_grid_report": args.rule_grid_preflight_report
                },
            )
            rule_grid_report = read_json(args.rule_grid_preflight_report)
            if (
                rule_grid_report.get("protocol")
                != RULE_GRID_DIAGNOSTIC_PROTOCOL
                or rule_grid_report.get("status") != "PASS"
                or rule_grid_report.get("outcome_labels_used") is not False
                or rule_grid_report.get("internal_states_used") is not False
                or rule_grid_report.get("selection_or_retuning_authorized")
                is not False
                or rule_grid_report.get("pm_v1_5_config_sha256")
                != sha256_file(args.pm_v2_config)
                or rule_grid_report.get("states_sha256")
                != sha256_file(pm_v2_states_path)
            ):
                raise RuntimeError(
                    "action sweep requires exact PASS outcome-free rule-grid preflight"
                )
            semantic_sanity = {
                "protocol": (
                    "pm-v1.5-pilot-plus-actual-corpus-and-shortcut-gate-v2"
                ),
                "status": "PASS",
                "human_calibration_performed": False,
                "automated_review_report_sha256": sha256_text(
                    canonical_json(automated_review_report)
                ),
                "automated_review_attestation_sha256": (
                    automated_review_verification["attestation_sha256"]
                ),
                "actual_corpus_review_report_sha256": (
                    actual_corpus_verification["report_sha256"]
                ),
                "actual_corpus_review_attestation_sha256": (
                    actual_corpus_verification["attestation_sha256"]
                ),
                "step0_shortcut_audit_report_sha256": (
                    shortcut_audit_verification["report_sha256"]
                ),
                "step0_shortcut_audit_attestation_sha256": (
                    shortcut_audit_verification["attestation_sha256"]
                ),
                "rule_grid_preflight_report_sha256": sha256_file(
                    args.rule_grid_preflight_report
                ),
                "rule_grid_preflight_attestation_sha256": (
                    rule_grid_attestation["attestation_sha256"]
                ),
            }
            if pilot_plan is not None:
                evaluator_index = load_evaluator_context_index(
                    evaluator_contexts_path,
                    states=audited_states,
                    require_exact=True,
                )
                deployable_feature_observability = (
                    audit_deployable_feature_observability(
                        audited_states,
                        evaluator_contexts=evaluator_index,
                        settings=dict(
                            pm_config["development_judging"][
                                "compatibility_pilot"
                            ]["deployable_feature_observability"]
                        ),
                    )
                )
                if (
                    deployable_feature_observability.get("status") != "PASS"
                    or pilot_plan.get("deployable_feature_observability")
                    != deployable_feature_observability
                ):
                    raise RuntimeError(
                        "PM-v2 pilot deployable-feature observability gate "
                        "is absent, stale, or failed"
                    )
                judge_schema_smoke = (
                    require_development_judge_schema_smoke_pass(
                        summary_path=args.judge_schema_smoke_summary,
                        attestation_path=args.judge_schema_smoke_attestation,
                        experiment_config_path=args.config,
                        pm_v2_config_path=args.pm_v2_config,
                        states_path=pm_v2_states_path,
                        backend_path=args.backend,
                        evaluator_contexts_path=evaluator_contexts_path,
                        pilot_plan_path=args.pilot_plan,
                        semantic_sanity_report_path=args.semantic_sanity_report,
                        semantic_sanity_attestation_path=(
                            args.semantic_sanity_attestation
                        ),
                    )
                )
        api_cost_config = dict(pm_config["api_cost_planning"])
        if set(api_cost_config) != {
            "input_token_safety_factor",
            "fail_on_reported_input_overrun",
        }:
            raise RuntimeError("api_cost_planning keys do not match PM-v2")
        input_token_safety_factor = float(
            api_cost_config["input_token_safety_factor"]
        )
        fail_on_reported_input_overrun = bool(
            api_cost_config["fail_on_reported_input_overrun"]
        )
        if input_token_safety_factor < 1.0 or not fail_on_reported_input_overrun:
            raise RuntimeError(
                "PM-v2 requires input token safety factor >=1 and fail-on-overrun"
            )

        frozen_values = {
            "endpoint": supporter_generation_contract.generator_endpoint,
            "temperature": supporter_generation_contract.temperature,
            "max_output_tokens": (
                supporter_generation_contract.max_output_tokens
            ),
            "seed": int(sweep_config["seed"]),
            "request_retries": int(sweep_config["request_retries"]),
            "fail_fast": bool(sweep_config["fail_fast"]),
            "strategy_top_k": int(retrieval_config["strategy_top_k"]),
            "memory_min_score": float(retrieval_config["memory_min_score"]),
            "strategy_min_score": float(retrieval_config["strategy_min_score"]),
        }
        requested_values = {
            "endpoint": args.endpoint,
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
            "seed": args.seed,
            "request_retries": args.request_retries,
            "fail_fast": args.fail_fast,
            "strategy_top_k": args.strategy_top_k,
            "memory_min_score": args.memory_min_score,
            "strategy_min_score": args.strategy_min_score,
        }
        for name, requested in requested_values.items():
            if requested is not None and requested != frozen_values[name]:
                raise RuntimeError(
                    f"{name} override differs from the PM-v2 config; edit and "
                    "review the preregistration before generating any labels"
                )
        endpoint_name = frozen_values["endpoint"]
        temperature = frozen_values["temperature"]
        max_output_tokens = frozen_values["max_output_tokens"]
        seed = frozen_values["seed"]
        request_retries = frozen_values["request_retries"]
        if request_retries != 1:
            raise RuntimeError(
                "PM-v2 development_sweep.request_retries must equal 1 so the "
                "dry-run is a physical-attempt cap"
            )
        fail_fast = frozen_values["fail_fast"]
        if not fail_fast:
            raise RuntimeError(
                "PM-v2 development_sweep.fail_fast must be true so an exact "
                "action-matrix failure cannot spend the remaining budget"
            )
        strategy_top_k = frozen_values["strategy_top_k"]
        memory_min_score = frozen_values["memory_min_score"]
        strategy_min_score = frozen_values["strategy_min_score"]
        if (
            args.input_usd_per_mtok is not None
            and float(args.input_usd_per_mtok) != frozen_input_price
        ):
            raise RuntimeError("input pricing override differs from PM-v2 YAML")
        if (
            args.output_usd_per_mtok is not None
            and float(args.output_usd_per_mtok) != frozen_output_price
        ):
            raise RuntimeError("output pricing override differs from PM-v2 YAML")
        args.input_usd_per_mtok = frozen_input_price
        args.output_usd_per_mtok = frozen_output_price
        contract_bindings = {
            "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
            "pm_v2_version": str(pm_config["version"]),
            "supporter_generation_treatment": (
                supporter_generation_contract.payload()
            ),
            "supporter_generation_treatment_sha256": (
                supporter_generation_contract.digest()
            ),
            "development_sweep": sweep_config,
            "retrieval": retrieval_config,
            "evidence_filter": evidence_filter_config.payload(),
            "evidence_filter_config_sha256": evidence_filter_config.digest(),
            "evidence_filter_model": evidence_filter_model_binding,
            **(
                {
                    "semantic_sanity": semantic_sanity,
                    "runtime_state_lineage": runtime_state_lineage,
                }
                if semantic_sanity is not None
                else {}
            ),
            **(
                {
                    "deployable_feature_observability": (
                        deployable_feature_observability
                    ),
                    "judge_schema_smoke": judge_schema_smoke,
                }
                if pilot_plan is not None
                else {}
            ),
            "api_cost_planning": api_cost_config,
            "scope": (
                "compatibility_pilot"
                if pilot_plan is not None
                else "pilot"
                if args.max_cards is not None or args.actions
                else "full"
            ),
        }
    else:
        if args.input_usd_per_mtok is None or args.output_usd_per_mtok is None:
            raise RuntimeError(
                "non-PM-v2 sweeps require explicit input/output pricing"
            )
        endpoint_name = args.endpoint or "generator"
        temperature = 0.0 if args.temperature is None else float(args.temperature)
        max_output_tokens = (
            300 if args.max_output_tokens is None else int(args.max_output_tokens)
        )
        seed = 4311 if args.seed is None else int(args.seed)
        request_retries = (
            3 if args.request_retries is None else int(args.request_retries)
        )
        fail_fast = bool(args.fail_fast) if args.fail_fast is not None else False
        strategy_top_k = 3 if args.strategy_top_k is None else int(args.strategy_top_k)
        memory_min_score = args.memory_min_score
        strategy_min_score = args.strategy_min_score
        evidence_filter_config = None
        memory_helpfulness_model = None
        input_token_safety_factor = 1.0
        fail_on_reported_input_overrun = False

    endpoint = endpoint_from_config(experiment_config, endpoint_name)
    action_filter = None
    card_filter = None
    if pilot_plan is not None:
        assert supporter_generation_contract is not None
        if pilot_plan.get("status") != "READY":
            raise RuntimeError("PM-v2 compatibility pilot plan is not READY")
        if pilot_plan.get("protocol") != (
            "pm_v2_development_compatibility_pilot_v2_treatment_bound"
        ):
            raise RuntimeError("PM-v2 compatibility pilot protocol is stale")
        if (
            pilot_plan.get("supporter_generation_treatment")
            != supporter_generation_contract.payload()
            or pilot_plan.get("supporter_generation_treatment_sha256")
            != supporter_generation_contract.digest()
        ):
            raise RuntimeError(
                "PM-v2 pilot plan supporter-generation treatment mismatch"
            )
        expected_hashes = {
            "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
            "runtime_sha256": sha256_file(args.runtime),
            "backend_sha256": sha256_file(args.backend),
            "strategy_bank_sha256": sha256_file(args.strategy_bank),
        }
        mismatches = {
            name: {"expected": expected, "observed": pilot_plan.get(name)}
            for name, expected in expected_hashes.items()
            if pilot_plan.get(name) != expected
        }
        if mismatches:
            raise RuntimeError(f"PM-v2 pilot plan lineage mismatch: {mismatches}")
        if pilot_plan.get("semantic_sanity") != contract_bindings.get(
            "semantic_sanity"
        ):
            raise RuntimeError(
                "PM-v2 pilot plan semantic-sanity attestation mismatch"
            )
        if pilot_plan.get("runtime_state_lineage") != contract_bindings.get(
            "runtime_state_lineage"
        ):
            raise RuntimeError("PM-v2 pilot plan runtime/state lineage mismatch")
        pilot_payload = {
            key: value for key, value in pilot_plan.items() if key != "pilot_plan_sha256"
        }
        if pilot_plan.get("pilot_plan_sha256") != sha256_text(
            canonical_json(pilot_payload)
        ):
            raise RuntimeError("PM-v2 pilot plan self-hash mismatch")
        action_filter = {str(value) for value in pilot_plan["actions"]}
        card_filter = {
            str(row["card_id"]) for row in pilot_plan["selected_states"]
        }
        contract_bindings["pilot_plan_sha256"] = str(
            pilot_plan["pilot_plan_sha256"]
        )
        contract_bindings["pilot_expected_keys_sha256"] = str(
            pilot_plan["expected_keys_sha256"]
        )
    elif args.actions:
        action_filter = {value.strip() for value in args.actions.split(",") if value.strip()}
        if not action_filter:
            raise ValueError("--actions did not contain any action IDs")
        for action_id in action_filter:
            parse_action_id(action_id)
    elif runtime_is_pm_v2 and args.max_cards is None:
        if (
            not args.v1_5_full_sweep_scope
            or semantic_sanity is None
            or runtime_state_lineage is None
        ):
            raise RuntimeError("PM-v1.5 full sweep lacks its frozen review gate")
        contract_bindings["v1_5_full_sweep_gate"] = {
            "protocol": "pm-v1.5-full-sweep-gate-v2",
            "status": "PASS",
            "scope": "full",
            "human_calibration_performed": False,
            "automated_review_attestation_sha256": semantic_sanity[
                "automated_review_attestation_sha256"
            ],
            "automated_review_report_sha256": semantic_sanity[
                "automated_review_report_sha256"
            ],
            "actual_corpus_review_attestation_sha256": semantic_sanity[
                "actual_corpus_review_attestation_sha256"
            ],
            "actual_corpus_review_report_sha256": semantic_sanity[
                "actual_corpus_review_report_sha256"
            ],
            "step0_shortcut_audit_attestation_sha256": semantic_sanity[
                "step0_shortcut_audit_attestation_sha256"
            ],
            "step0_shortcut_audit_report_sha256": semantic_sanity[
                "step0_shortcut_audit_report_sha256"
            ],
        }
    estimate, rows = plan_action_sweep(
        args.runtime,
        args.backend,
        args.strategy_bank,
        endpoint=endpoint,
        max_cards=args.max_cards,
        card_filter=card_filter,
        action_filter=action_filter,
        temperature=temperature,
        max_tokens=max_output_tokens,
        seed=seed,
        request_retries=request_retries,
        fail_fast=fail_fast,
        input_token_safety_factor=input_token_safety_factor,
        fail_on_reported_input_overrun=fail_on_reported_input_overrun,
        strategy_top_k=strategy_top_k,
        memory_min_score=memory_min_score,
        strategy_min_score=strategy_min_score,
        evidence_filter_config=evidence_filter_config,
        memory_helpfulness_model=memory_helpfulness_model,
        supporter_generation_contract=supporter_generation_contract,
        input_usd_per_mtok=args.input_usd_per_mtok,
        output_usd_per_mtok=args.output_usd_per_mtok,
        contract_bindings=contract_bindings,
    )
    gate = _budget_gate(
        estimate,
        max_api_calls=args.max_api_calls,
        max_estimated_usd=args.max_estimated_usd,
        max_input_tokens_per_call=args.max_input_tokens_per_call,
    )
    result = {**estimate, "budget_gate": gate}

    if args.dry_run:
        _persist_or_validate_dry_run(args.out_dir, result, rows)
        print(result)
        if gate["status"] != "PASS":
            raise RuntimeError("action-sweep dry-run failed the frozen budget gate")
        return

    if gate["status"] != "PASS":
        raise RuntimeError("action-sweep API run blocked by budget gate")
    _require_saved_dry_run(args.out_dir, result, rows)
    expected_hash = str(result["cost_estimate_sha256"])
    if not args.accept_cost_estimate_sha256:
        raise RuntimeError(
            "API mode is fail-closed: pass --accept-cost-estimate-sha256 "
            f"{expected_hash} from the matching dry-run"
        )
    if args.accept_cost_estimate_sha256 != expected_hash:
        raise RuntimeError(
            "accepted cost estimate hash does not match current action-sweep plan"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = run_action_sweep(
        args.runtime,
        args.backend,
        args.strategy_bank,
        args.out_dir / "action_outcomes.jsonl",
        args.out_dir / "raw_api_calls.jsonl",
        args.out_dir / "summary.json",
        endpoint=endpoint,
        max_cards=args.max_cards,
        card_filter=card_filter,
        action_filter=action_filter,
        temperature=temperature,
        max_tokens=max_output_tokens,
        seed=seed,
        request_retries=request_retries,
        fail_fast=fail_fast,
        input_token_safety_factor=input_token_safety_factor,
        fail_on_reported_input_overrun=fail_on_reported_input_overrun,
        strategy_top_k=strategy_top_k,
        memory_min_score=memory_min_score,
        strategy_min_score=strategy_min_score,
        evidence_filter_config=evidence_filter_config,
        memory_helpfulness_model=memory_helpfulness_model,
        supporter_generation_contract=supporter_generation_contract,
        overwrite=args.overwrite,
        max_physical_api_attempts=args.max_api_calls,
        contract_bindings={
            **contract_bindings,
            "accepted_cost_estimate_sha256": expected_hash,
            "pricing": result["pricing"],
        },
    )
    print(summary)


if __name__ == "__main__":
    main()
