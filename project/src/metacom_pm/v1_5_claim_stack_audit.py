"""Fail-closed audit of the V1.5 V3 claim and same-stack transition.

The historical V1.5 response-mechanism contract correctly bound one raw
Strategy Bank, retriever, prompt compiler, generator treatment, and Evidence
Filter setting.  Strategy Bank V2 intentionally changes that treatment.  This
module prevents the scientifically invalid shortcut of retaining historical
responses or checkpoints while describing them as evidence for the new V2
RAG mechanism.

This is a readiness audit, not a study freeze.  It identifies local work that
may continue and blocks response generation/formal fitting until a new
same-stack identity is implemented and bound across every policy condition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import load_config
from .io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
)
from .response_mechanism_contract import (
    require_matching_response_mechanism_contract,
)
from .v1_5_strategy_bank import (
    validate_strategy_bank_v2_human_annotations,
)


PROTOCOL = "pm-v1.5-v3-claim-same-stack-readiness-audit-v1"
REQUIRED_CONSUMER_ROLES = (
    "clean_component_pair_generation",
    "train_sweep",
    "transparent_rule",
    "fixed_comparators",
    "internal_evaluation",
    "external_esconv",
    "external_evoemo",
)


def _within_root(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute():
        raise RuntimeError(f"contract path must be relative: {relative}")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"contract path escapes project root: {relative}") from exc
    return path


def _verify_self_hash(
    value: Mapping[str, Any],
    *,
    hash_field: str,
    context: str,
) -> str:
    declared = str(value.get(hash_field) or "")
    body = {key: item for key, item in value.items() if key != hash_field}
    actual = sha256_text(canonical_json(body))
    if declared != actual:
        raise RuntimeError(f"{context} self hash is invalid")
    return declared


def _require_file(
    root: Path,
    binding: Mapping[str, Any],
    *,
    context: str,
) -> Path:
    path = _within_root(root, str(binding["path"]))
    if not path.is_file():
        raise RuntimeError(f"{context} is missing: {path}")
    expected = str(binding["file_sha256"])
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{context} hash drifted: expected {expected}, observed {actual}"
        )
    return path


def validate_same_stack_consumer_matrix(
    consumers: Sequence[Mapping[str, Any]],
    *,
    required_roles: Sequence[str] = REQUIRED_CONSUMER_ROLES,
) -> dict[str, Any]:
    """Require every scientific condition to bind one exact mechanism.

    Different policies may request different actions.  They may not use a
    different Bank, query builder, retriever, filter, prompt compiler,
    generator endpoint, decoding treatment, or normalization.
    """

    roles = [str(row.get("role") or "") for row in consumers]
    required = list(required_roles)
    if len(roles) != len(set(roles)):
        raise RuntimeError("same-stack consumer matrix contains duplicate roles")
    if set(roles) != set(required):
        missing = sorted(set(required) - set(roles))
        extra = sorted(set(roles) - set(required))
        raise RuntimeError(
            "same-stack consumer roles are not exact: "
            f"missing={missing}, extra={extra}"
        )
    by_role = {str(row["role"]): row for row in consumers}
    reference = dict(
        by_role[required[0]].get("response_mechanism_contract") or {}
    )
    if not reference:
        raise RuntimeError(
            f"same-stack consumer {required[0]} lacks a response mechanism contract"
        )
    for role in required:
        actual = dict(
            by_role[role].get("response_mechanism_contract") or {}
        )
        if not actual:
            raise RuntimeError(
                f"same-stack consumer {role} lacks a response mechanism contract"
            )
        require_matching_response_mechanism_contract(
            expected=reference,
            actual=actual,
            context=f"V1.5 V3 same-stack consumer {role}",
        )
    return {
        "status": "PASS",
        "protocol": "pm-v1.5-v3-same-stack-consumer-matrix-v1",
        "required_roles": required,
        "response_mechanism_contract_sha256": reference["contract_sha256"],
        "consumer_matrix_sha256": sha256_text(canonical_json(list(consumers))),
    }


def _audit_bank_v2(
    root: Path,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    binding_path = _require_file(
        root, spec["candidate_binding"], context="Strategy Bank V2 binding"
    )
    binding = read_json(binding_path)
    if (
        binding.get("protocol")
        != "pm-v1.5-strategy-bank-v2-candidate-binding-v2"
        or binding.get("candidate_eligible_for_formal_rs") is not False
    ):
        raise RuntimeError("Strategy Bank V2 binding has an unexpected role/status")

    output_dir = _within_root(root, str(binding["output_directory"]))
    if not output_dir.is_dir():
        raise RuntimeError("Strategy Bank V2 candidate output directory is missing")
    for name, digest in dict(binding["files"]).items():
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != str(digest):
            raise RuntimeError(f"Strategy Bank V2 candidate file drifted: {name}")

    build_report = read_json(output_dir / "build_report.json")
    _verify_self_hash(
        build_report,
        hash_field="report_sha256",
        context="Strategy Bank V2 build report",
    )
    if (
        build_report["report_sha256"] != binding["build_report_sha256"]
        or build_report.get("candidate_card_count") != 5
        or build_report.get("candidate_packet_source_overlap_count") != 0
        or build_report.get("candidate_raw_examples_exposed_to_generator")
        is not False
        or build_report.get("candidate_eligible_for_formal_rs") is not False
    ):
        raise RuntimeError("Strategy Bank V2 candidate invariants failed")

    cards = [
        dict(row)
        for row in iter_jsonl(output_dir / "strategy_cards_v2_candidate.jsonl")
    ]
    if (
        len(cards) != 5
        or len({str(row["card_id"]) for row in cards}) != 5
        or any(row.get("eligible_for_formal_rs") is not False for row in cards)
        or any(row.get("content_scope") != "technique_only" for row in cards)
        or any("example_response" in row for row in cards)
    ):
        raise RuntimeError("Strategy Bank V2 card surface is unsafe or malformed")

    reproduction_dir = _within_root(root, str(spec["reproduction_directory"]))
    reproduction_checks: dict[str, bool] = {}
    for name, digest in dict(binding["files"]).items():
        candidate = output_dir / name
        reproduction = reproduction_dir / name
        reproduction_checks[name] = bool(
            reproduction.is_file()
            and sha256_file(reproduction) == str(digest)
            and reproduction.read_bytes() == candidate.read_bytes()
        )
    if not all(reproduction_checks.values()):
        raise RuntimeError(
            "Strategy Bank V2 exact reproduction is missing or differs"
        )

    annotation_path = _within_root(
        root, str(spec["human_annotation_path"])
    )
    if annotation_path.is_file():
        human_review = validate_strategy_bank_v2_human_annotations(
            candidate_dir=output_dir,
            annotations_path=annotation_path,
        )
        human_review_complete = (
            human_review["status"]
            == "HUMAN_REVIEW_PASS_PENDING_LLM_WEAK_AUDIT"
            and int(human_review["approved_count"]) == 5
        )
    else:
        human_review = {
            "status": "NOT_COMPLETED",
            "review_count": 0,
            "approved_count": 0,
            "formal_rs_promoted": False,
        }
        human_review_complete = False

    llm_audit_path = _within_root(root, str(spec["llm_weak_audit_path"]))
    llm_weak_audit_complete = False
    llm_weak_audit = {"status": "NOT_COMPLETED"}
    if llm_audit_path.is_file():
        llm_weak_audit = read_json(llm_audit_path)
        if "binding_sha256" in llm_weak_audit:
            _verify_self_hash(
                llm_weak_audit,
                hash_field="binding_sha256",
                context="Strategy Bank V2 LLM weak-audit binding",
            )
        llm_weak_audit_complete = (
            llm_weak_audit.get("status") == "PASS"
            and llm_weak_audit.get("formal_rs_promoted") is False
        )

    return {
        "status": "REPRODUCIBLE_CANDIDATE_NOT_FORMAL",
        "binding_file_sha256": sha256_file(binding_path),
        "build_report_sha256": build_report["report_sha256"],
        "candidate_files_exact": True,
        "candidate_reproduction_exact": True,
        "reproduction_file_checks": reproduction_checks,
        "card_count": len(cards),
        "source_overlap_count": 0,
        "raw_examples_exposed_to_generator": False,
        "human_review": human_review,
        "human_review_complete": human_review_complete,
        "llm_weak_audit": llm_weak_audit,
        "llm_weak_audit_complete": llm_weak_audit_complete,
        "formal_rs_promoted": False,
        "formal_rs_ready": False,
    }


def _audit_support_need(
    root: Path,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    binding_path = _require_file(
        root, spec["binding"], context="expanded SupportNeed binding"
    )
    binding = read_json(binding_path)
    _verify_self_hash(
        binding,
        hash_field="binding_sha256",
        context="expanded SupportNeed binding",
    )
    candidate = _within_root(root, str(binding["candidate_report"]))
    reproduction = _within_root(root, str(binding["repro_report"]))
    analysis = _within_root(root, str(binding["analysis_report"]))
    analysis_reproduction = _within_root(
        root, str(binding["analysis_repro_report"])
    )
    if (
        not candidate.is_file()
        or not reproduction.is_file()
        or candidate.read_bytes() != reproduction.read_bytes()
        or sha256_file(candidate) != binding["bakeoff_report_file_sha256"]
        or not analysis.is_file()
        or not analysis_reproduction.is_file()
        or analysis.read_bytes() != analysis_reproduction.read_bytes()
        or sha256_file(analysis) != binding["analysis_report_file_sha256"]
    ):
        raise RuntimeError("expanded SupportNeed evidence is missing or drifted")
    if (
        binding.get("formal_factorized_fit_authorized") is not False
        or binding.get("representation_promotion_authorized") is not False
        or binding.get("flat_support_mode_fit_authorized") is not False
        or binding.get("confirmation_opening_authorized") is not False
        or binding.get("internal_test_outcomes_opened") is not False
        or binding.get("external_outcomes_opened") is not False
    ):
        raise RuntimeError("expanded SupportNeed gate unexpectedly opened")
    return {
        "status": str(binding["status"]),
        "binding_file_sha256": sha256_file(binding_path),
        "independent_dialogue_groups": int(
            binding["independent_dialogue_groups"]
        ),
        "confirmation_groups_opened": int(
            binding["confirmation_groups_opened"]
        ),
        "nli_stable_axis_signal": list(
            binding["nli_best_and_ci_better_than_lexical_factor_ids"]
        ),
        "collapsed_primary_factors": list(
            binding["collapsed_best_primary_factor_ids"]
        ),
        "formal_factorized_fit_authorized": False,
        "representation_promotion_authorized": False,
        "confirmation_opening_authorized": False,
    }


def _audit_historical_same_stack(
    root: Path,
    spec: Mapping[str, Any],
    *,
    bank_v2: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = _require_file(
        root, spec["sweep_manifest"], context="historical sweep manifest"
    )
    manifest = read_json(manifest_path)
    mechanism = dict(
        (manifest.get("contract_bindings") or {}).get(
            "response_mechanism_contract"
        )
        or {}
    )
    _verify_self_hash(
        mechanism,
        hash_field="contract_sha256",
        context="historical response mechanism contract",
    )
    code_checks: dict[str, bool] = {}
    for relative, digest in dict(mechanism["shared_code_manifest"]).items():
        path = _within_root(root, relative)
        code_checks[relative] = path.is_file() and sha256_file(path) == digest
    current_code_exact = all(code_checks.values())

    config_path = _require_file(
        root, spec["pm_config"], context="PM-v1.5 config"
    )
    config = load_config(config_path)
    retrieval = dict(config.get("retrieval") or {})
    external = dict(config.get("external_evaluation") or {})
    keys = ("memory_min_score", "strategy_min_score", "strategy_top_k")
    development_settings = {key: retrieval.get(key) for key in keys}
    external_settings = {key: external.get(key) for key in keys}
    if development_settings != external_settings:
        raise RuntimeError("historical development/external RAG settings differ")
    if development_settings != dict(mechanism["retrieval_settings"]):
        raise RuntimeError("historical sweep/config RAG settings differ")

    current_v2_cards_sha = str(
        read_json(
            _within_root(
                root,
                str(spec["strategy_bank_v2_binding_path"]),
            )
        )["files"]["strategy_cards_v2_candidate.jsonl"]
    )
    historical_bank_sha = str(mechanism["strategy_bank_sha256"])
    if historical_bank_sha == current_v2_cards_sha:
        raise RuntimeError("historical raw Bank unexpectedly equals Bank V2")

    return {
        "status": (
            "HISTORICAL_CONTRACT_VALID_CURRENT_CODE_EXACT_BUT_SUPERSEDED"
            if current_code_exact
            else (
                "HISTORICAL_CONTRACT_VALID_CURRENT_CODE_DRIFTED_AND_"
                "SUPERSEDED"
            )
        ),
        "manifest_file_sha256": sha256_file(manifest_path),
        "response_mechanism_contract_sha256": mechanism["contract_sha256"],
        "shared_code_checks": code_checks,
        "current_code_exact_to_historical_contract": current_code_exact,
        "current_code_drifted_paths": sorted(
            path for path, exact in code_checks.items() if not exact
        ),
        "development_external_scalar_settings_equal": True,
        "retrieval_settings": development_settings,
        "historical_strategy_bank_sha256": historical_bank_sha,
        "strategy_bank_v2_cards_file_sha256": current_v2_cards_sha,
        "bank_changed": True,
        "historical_response_outcomes_reusable_for_v3": False,
        "historical_checkpoint_reusable_for_v3": False,
        "reason": (
            "Strategy Bank V2 changes the response treatment; old raw-Bank "
            "responses and fitted checkpoints cannot evidence the V3 method"
        ),
        "bank_v2_candidate_reproducible": bool(
            bank_v2["candidate_reproduction_exact"]
        ),
    }


def _audit_runtime_integration(
    root: Path,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    references: dict[str, bool] = {}
    needles = tuple(str(value) for value in spec["v2_runtime_needles"])
    for relative in spec["consumer_source_paths"]:
        path = _within_root(root, str(relative))
        if not path.is_file():
            raise RuntimeError(f"same-stack consumer source is missing: {relative}")
        source = path.read_text(encoding="utf-8")
        references[str(relative)] = any(needle in source for needle in needles)
    integrated_paths = sorted(
        path for path, present in references.items() if present
    )
    return {
        "status": (
            "PARTIAL_OR_PRESENT"
            if integrated_paths
            else "NOT_IMPLEMENTED"
        ),
        "consumer_source_reference_checks": references,
        "integrated_consumer_paths": integrated_paths,
        "all_required_consumers_integrated": (
            len(integrated_paths) == len(references)
        ),
    }


def audit_v1_5_v3_claim_stack(
    *,
    project_root: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    """Audit current evidence without opening held-out outcomes."""

    root = Path(project_root).resolve()
    contract_path = Path(contract_path).resolve()
    contract = read_json(contract_path)
    if contract.get("protocol") != PROTOCOL:
        raise RuntimeError("V1.5 V3 claim/same-stack contract is missing or stale")
    contract_sha256 = _verify_self_hash(
        contract,
        hash_field="contract_sha256",
        context="V1.5 V3 claim/same-stack contract",
    )
    bank_v2 = _audit_bank_v2(root, contract["strategy_bank_v2"])
    support_need = _audit_support_need(root, contract["support_need"])
    historical = _audit_historical_same_stack(
        root,
        contract["historical_same_stack"],
        bank_v2=bank_v2,
    )
    runtime = _audit_runtime_integration(
        root, contract["runtime_integration"]
    )

    consumer_manifest_path = _within_root(
        root, str(contract["future_same_stack"]["consumer_matrix_path"])
    )
    if consumer_manifest_path.is_file():
        consumer_matrix = read_json(consumer_manifest_path)
        consumer_matrix_result = validate_same_stack_consumer_matrix(
            list(consumer_matrix["consumers"]),
            required_roles=tuple(
                contract["future_same_stack"]["required_consumer_roles"]
            ),
        )
    else:
        consumer_matrix_result = {
            "status": "NOT_FROZEN",
            "required_roles": list(
                contract["future_same_stack"]["required_consumer_roles"]
            ),
        }

    blockers: list[str] = []
    if not support_need["formal_factorized_fit_authorized"]:
        blockers.append("SUPPORT_NEED_FORMAL_FIT_NOT_AUTHORIZED")
    if not bank_v2["human_review_complete"]:
        blockers.append("BANK_V2_HUMAN_REVIEW_INCOMPLETE")
    if not bank_v2["llm_weak_audit_complete"]:
        blockers.append("BANK_V2_LLM_WEAK_AUDIT_INCOMPLETE")
    if not bank_v2["formal_rs_promoted"]:
        blockers.append("BANK_V2_NOT_PROMOTED")
    if not runtime["all_required_consumers_integrated"]:
        blockers.append("BANK_V2_RUNTIME_NOT_IMPLEMENTED_FOR_ALL_CONSUMERS")
    if consumer_matrix_result["status"] != "PASS":
        blockers.append("V3_SAME_STACK_CONSUMER_MATRIX_NOT_FROZEN")

    claim_assessment = {
        "abstract_estimand_changed": False,
        "operational_treatment_changed": True,
        "historical_v1_5_same_stack": (
            "SUPPORTED_AS_HISTORICAL_IDENTITY_ONLY"
        ),
        "support_need_diagnostic_module": "NOT_SUPPORTED",
        "support_need_axis_signal": (
            "TRAIN_ONLY_DIAGNOSTIC_FOR_HEARD_AND_CONTAINMENT"
        ),
        "strategy_bank_v2_formal_rag": "NOT_READY",
        "v3_learned_router": "NOT_YET_TESTED",
        "v3_quality_cost_claim": "NOT_YET_TESTED",
        "allowed_current_statement": contract["claims"][
            "allowed_current_statement"
        ],
        "future_primary_claim_template": contract["claims"][
            "future_primary_claim_template"
        ],
        "prohibited_interpretations": list(
            contract["claims"]["prohibited_interpretations"]
        ),
    }

    report_core = {
        "protocol": PROTOCOL,
        "status": "BLOCKED_BEFORE_V3_RAG_TREATMENT_GENERATION",
        "contract_file_sha256": sha256_file(contract_path),
        "contract_sha256": contract_sha256,
        "claim_assessment": claim_assessment,
        "support_need": support_need,
        "strategy_bank_v2": bank_v2,
        "historical_same_stack": historical,
        "runtime_integration": runtime,
        "future_same_stack_consumer_matrix": consumer_matrix_result,
        "blockers": blockers,
        "allowed_now": list(contract["allowed_now"]),
        "prohibited_until_ready": list(contract["prohibited_until_ready"]),
        "same_stack_definition": dict(contract["same_stack_definition"]),
        "internal_test_outcomes_opened": False,
        "external_outcomes_opened": False,
        "api_calls_made": 0,
    }
    return {
        **report_core,
        "report_sha256": sha256_text(canonical_json(report_core)),
    }
