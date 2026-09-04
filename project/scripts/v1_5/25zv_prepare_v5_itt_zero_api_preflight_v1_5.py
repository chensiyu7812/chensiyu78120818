#!/usr/bin/env python3
"""Write the V5 zero-API preflight report; never calls a model provider."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

from metacom_pm.v1_5_itt_policy import validate_pre_injection_feature_names
from metacom_pm.v1_5_typed_resource_adapter import (
    TypedResourceCandidate,
    compile_typed_bundle,
)
from metacom_pm.v1_5b_policy_runtime import COMPONENTS, compile_component_bits


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def synthetic_candidate(component: str) -> TypedResourceCandidate:
    base = {
        "component": component,
        "resource_id": f"preflight-{component.lower()}",
        "candidate_version": "preflight-v1",
    }
    if component == "MP":
        return TypedResourceCandidate(
            **base,
            subtype="MP_PROFILE",
            source_kind="profile",
            owner_id="preflight-user",
            profile_fact="has a limited private time window",
        )
    if component == "MS":
        return TypedResourceCandidate(
            **base,
            subtype="MS_SESSION_OBSERVATION",
            source_kind="session",
            owner_id="preflight-user",
            strictly_prior=True,
            age_sessions=1,
            prior_observation="the transition, rather than sleep in general, was difficult",
        )
    if component == "ME":
        return TypedResourceCandidate(
            **base,
            subtype="ME_REUSABLE_OUTCOME",
            source_kind="event",
            owner_id="preflight-user",
            strictly_prior=True,
            age_sessions=1,
            past_action="wrote one sentence",
            observed_outcome="the conversation stayed focused",
        )
    return TypedResourceCandidate(
        **base,
        subtype="RS_ATOMIC_MOVE",
        source_kind="strategy",
        support_move="ask one focused question",
        when_to_use="one question is welcomed",
        when_not_to_use="questions are declined",
    )


def build_report() -> dict:
    contracts = {
        "architecture": "data/pm_v1_5_contracts/staged_policy_architecture_v4.json",
        "execution": "data/pm_v1_5_contracts/final_execution_plan_v5.json",
        "fit_design": "data/pm_v1_5_contracts/v5_itt_fit_design_v1.json",
        "registry": "data/pm_v1_5_contracts/method_version_registry_v1.json",
    }
    loaded = {key: load_json(path) for key, path in contracts.items()}
    if loaded["execution"]["status"] not in {
        "ACTIVE_SINGLE_SOURCE_ZERO_API_IMPLEMENTATION_IN_PROGRESS",
        "ACTIVE_S0_PASS_S1_FIT_CALL_PLAN_FROZEN",
    }:
        raise RuntimeError("V5 execution contract is not active")
    if loaded["registry"]["canonical_next"] != "V1_5_V5_ITT_SEPARATED_POLICY":
        raise RuntimeError("method registry does not point to V5")

    compiled_actions: list[str] = []
    for values in product((False, True), repeat=4):
        bits = dict(zip(COMPONENTS, values, strict=True))
        action = compile_component_bits(bits)
        compile_typed_bundle(
            requested_action_id=action,
            candidates={
                component: synthetic_candidate(component)
                for component, enabled in bits.items()
                if enabled
            },
            current_user_id="preflight-user",
        )
        compiled_actions.append(action)

    transparent_features = [
        "candidate_present",
        "owner_time_valid",
        "goal_function_fit",
        "boundary_compatible",
        "specific_increment",
        "incremental_tokens",
    ]
    validate_pre_injection_feature_names(transparent_features)

    opportunity_assets = {
        "MP_MS_internal": "outputs/pm_v1_5b_mp_ms_opportunity_heads_v1/fit_report.json",
        "MP_MS_external_transport": "outputs/pm_v1_5b_evoemo_mp_ms_transport_v1/transport_report.json",
        "ME": "outputs/pm_v1_5_scale_stable_memory_opportunity_heads_v1/fit_report.json",
        "RS": "outputs/pm_v1_5_same_bank_rs_opportunity_router_fit_v1/fit_report.json",
    }
    asset_status = {}
    for key, relative in opportunity_assets.items():
        data = load_json(relative)
        asset_status[key] = {"path": relative, "status": data.get("status")}

    implementation = [
        "src/metacom_pm/v1_5_typed_resource_adapter.py",
        "src/metacom_pm/v1_5_itt_policy.py",
    ]
    fit_manifest_path = ROOT / "outputs/pm_v1_5_v5_itt_fit_plan_v2/freeze_manifest.json"
    fit_manifest = (
        json.loads(fit_manifest_path.read_text(encoding="utf-8"))
        if fit_manifest_path.is_file()
        else None
    )
    fit_ready = bool(
        fit_manifest
        and fit_manifest.get("status") == "READY_FOR_SINGLE_V5_ITT_FIT_EXECUTION"
        and fit_manifest.get("planned_calls") == 1024
    )
    return {
        "protocol": "pm-v1.5-v5-itt-zero-api-preflight-v1",
        "status": (
            "S0_MACHINE_INVARIANTS_PASS_S1_FIT_CALL_PLAN_FROZEN"
            if fit_ready
            else "S0_MACHINE_INVARIANTS_PASS_S1_FULL_PLAN_MATERIALIZATION_REQUIRED"
        ),
        "api_calls": 0,
        "formal_fit_generation_scientifically_ready": fit_ready,
        "method_version": "V1_5_V5_ITT_SEPARATED_POLICY",
        "checks": {
            "active_contract_consistent": True,
            "all_16_actions_compile": len(set(compiled_actions)) == 16,
            "m0_r0_empty_bundle_legal": "M0+R0" in compiled_actions,
            "pre_injection_feature_schema_has_no_post_action_fields": True,
            "opportunity_assets_present": True,
            "typed_adapter_and_itt_validator_present": True,
            "new_step2_human_packet_required": False,
        },
        "opportunity_layer_assets": asset_status,
        "contract_sha256": {key: sha256(ROOT / path) for key, path in contracts.items()},
        "implementation_sha256": {path: sha256(ROOT / path) for path in implementation},
        "fit_shape": loaded["fit_design"]["fit_panel"],
        "frozen_fit_manifest": (
            "outputs/pm_v1_5_v5_itt_fit_plan_v2/freeze_manifest.json"
            if fit_ready
            else None
        ),
        "next_required_action": (
            "Execute the frozen resumable FIT response plan once, then build the single "
            "pre-registered full outcome review; do not create a Step2 pilot packet."
            if fit_ready
            else "Materialize and hash the complete 256-state/1024-response FIT call plan."
        ),
        "explicitly_not_next": [
            "another 10-32 item Step2 human review",
            "training on old generator self-reports",
            "calling fallback or nonuse an invalid treatment",
            "reading ESConv/EvoEmo outcomes to modify the plan",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="outputs/pm_v1_5_v5_itt_zero_api_preflight_v1",
    )
    args = parser.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report()
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "api_calls": 0}))


if __name__ == "__main__":
    main()
