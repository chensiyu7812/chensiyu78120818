#!/usr/bin/env python3
"""Generate PM-v1.5 synthetic development data on the frozen fast track.

The V1.5 branch requires an exact PASS casewise generation-compatibility
attestation before either the formal dry-run identity or a paid full run can be
created.  Semantic validation is intentionally deferred to actual-468
structured QA after generation and before the action sweep; the consumed V4
calibration review is not an authorization gate.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.api import (
    CallResult,
    ProviderRequestError,
    RetryableProviderError,
    StructuredOutputValidationError,
    chat_request_payload,
    make_client,
    require_reported_usage,
)
from metacom_pm.attempt_ledger import (
    PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
    PersistentAttemptLedger,
    forbid_overwrite_of_spent_attempts,
    physical_call_key,
)
from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.bounded_retry import (
    BOUNDED_PROVIDER_OUTPUT_RETRY_CLASSES,
    RETRY_CONTRACT_PROTOCOL,
    TERMINAL_DISPOSITION,
    call_retry_blocker,
    execute_with_bounded_retry,
    failure_metadata,
    retry_ledger_summary,
)
from metacom_pm.contracts import StrategyCard
from metacom_pm.io import (
    append_jsonl,
    canonical_json,
    dict_field_diff,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.pm_v2_contracts import PMV2Split, ResourceNeedRegime
from metacom_pm.pm_v2_data import (
    GENERATION_CASE_FIELDS,
    GENERATION_TEMPERATURE,
    OBSERVABLE_HISTORY_TURN_TARGETS,
    OBSERVABLE_STATE_SUPPORT_PROTOCOL,
    SURFACE_GENERATION_MAX_OUTPUT_TOKENS,
    SURFACE_GENERATION_MAX_REPAIRS,
    GeneratedSurfaceOnlyCaseDraft,
    GeneratedUserBundle,
    audit_cross_split_near_duplicates,
    audit_current_user_text_diversity,
    bind_bundle_to_generation_run,
    compile_surface_only_user_bundle,
    deterministic_memory_blueprint_preflight,
    generation_case_family_assignments,
    generation_case_messages,
    lint_generation_surface_case,
    load_bundles,
    load_states,
    normalize_text,
    require_bundle_generation_binding,
    surface_generation_contract_hash,
    validate_bundle,
    validate_successful_generation_trace,
    write_development_dataset,
)
from metacom_pm.pm_v2_generation_pilot import (
    CALIBRATION_SEMANTIC_FAMILIES,
    INTERNAL_TEST_SEMANTIC_FAMILIES,
    SEMANTIC_FAMILY_COHORTS_BY_SPLIT,
    TRAIN_SEMANTIC_FAMILIES,
    GENERATION_PILOT_MAX_TRANSPORT_ATTEMPTS_PER_CONTENT_ATTEMPT,
    GENERATION_PILOT_TRANSPORT_BACKOFF_SECONDS,
    build_generation_compatibility_contract,
    generation_family_schedule,
    require_generation_compatibility_attestation,
)
from metacom_pm.text import conservative_token_bound, estimate_tokens
from metacom_pm.paid_run_release import require_paid_run_release
from metacom_pm.pm_v1_5_semantic import (
    FrozenTransformerSemanticEncoder,
    require_semantic_runtime_contract,
    require_unified_semantic_query_contract,
    semantic_encoder_spec_from_config,
)
from metacom_pm.pm_v1_5_step0 import readiness_natural_language_challenge

ROOT = Path(__file__).resolve().parents[2]

GENERATION_REQUEST_RETRIES = 1
GENERATION_COST_PROTOCOL = (
    "pm_v2_generation_cost_v3_casewise_surface_bounded_repair_ledger"
)
GENERATION_STAGE = "pm_v2_synthetic_surface_generation"
# Comfortably above the maximum seed the normal per-case formula can ever
# reach (base_seed + up to 51*1000 + 8*10 + 1), so a duplicate-specific
# repair attempt never reuses the exact seed that produced the colliding
# text, without touching the normal formula or its budget for anyone else.
DUPLICATE_REPAIR_SEED_OFFSET = 900_000


class ReportedInputTokenOverrun(RuntimeError):
    pass


_CONTENT_ATTEMPT_FAILURE_CLASSES = frozenset(
    {
        *BOUNDED_PROVIDER_OUTPUT_RETRY_CLASSES,
        "structured_output_validation_error",
        "content_lint_failure",
    }
)


def _persisted_content_attempt_failed(
    ledger: PersistentAttemptLedger, call_key: str
) -> bool:
    terminal = ledger.terminal_row(call_key)
    if terminal is None or terminal.get("event") != "FAILED":
        return False
    metadata = terminal.get("metadata") or {}
    return str(metadata.get("retry_class") or "") in (
        _CONTENT_ATTEMPT_FAILURE_CLASSES
    )


def _load_successful_surface_attempt(
    *,
    ledger: PersistentAttemptLedger,
    case_plan: dict[str, Any],
) -> tuple[dict[str, Any], GeneratedSurfaceOnlyCaseDraft, CallResult] | None:
    successful = [
        attempt
        for attempt in case_plan["attempts"]
        if ledger.succeeded(str(attempt["call_key"]))
    ]
    if len(successful) > 1:
        raise RuntimeError(
            "casewise generation has multiple successful calls for "
            f"{case_plan['user_id']}/{case_plan['case_field']}"
        )
    if not successful:
        return None
    attempt = successful[0]
    terminal = ledger.terminal_row(str(attempt["call_key"])) or {}
    result = terminal.get("result") or {}
    surface_payload = result.get("surface")
    provider_response = result.get("provider_response")
    if not isinstance(surface_payload, dict) or not isinstance(
        provider_response, dict
    ):
        raise RuntimeError("successful surface attempt lacks recoverable trace")
    usage = require_reported_usage(
        terminal.get("usage"), stage="recovered PM-v1.5 surface generation"
    )
    request_hash = str(terminal.get("request_hash") or "")
    if not request_hash:
        raise RuntimeError("successful surface attempt lacks request hash")
    surface = GeneratedSurfaceOnlyCaseDraft.model_validate(surface_payload)
    call = CallResult(
        text=canonical_json(surface_payload),
        raw_response=provider_response,
        usage=usage,
        latency_ms=0.0,
        request_hash=request_hash,
    )
    return attempt, surface, call


def _load_successful_surface_from_old_ledger(
    old_ledger_path: Path,
    *,
    user_id: str,
    case_field: str,
) -> tuple[GeneratedSurfaceOnlyCaseDraft, CallResult] | None:
    """Read-only extraction of one already-paid surface from a DIFFERENT run's
    immutable ledger, by (user_id, case_field) rather than by call_key.

    This intentionally does not go through the normal call-plan/call-key
    lookup (_load_successful_surface_attempt): the request payload -- and
    therefore the call_key -- is derived from the exact prompt text, which
    legitimately changes across code fixes (e.g. the V8.13-V8.16 anchor
    fixes). The underlying paid provider response is unaffected by a local
    prompt-wording or lint change, so it is still valid raw material to
    recompile under current code; only the cache key changes.
    """
    matches: list[dict[str, Any]] = []
    for row in iter_jsonl(old_ledger_path):
        if row.get("event") != "SUCCEEDED":
            continue
        record_ids = row.get("record_ids") or {}
        if record_ids.get("user_id") == user_id and record_ids.get("case_field") == case_field:
            matches.append(row)
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(
            f"carry-forward source ledger has multiple SUCCEEDED rows for "
            f"{user_id}/{case_field}: {old_ledger_path}"
        )
    result = matches[0].get("result") or {}
    surface_payload = result.get("surface")
    provider_response = result.get("provider_response")
    if not isinstance(surface_payload, dict) or not isinstance(provider_response, dict):
        raise RuntimeError(
            f"carry-forward source lacks a recoverable trace for {user_id}/{case_field}"
        )
    usage = require_reported_usage(
        matches[0].get("usage"), stage="carried-forward PM-v1.5 surface generation"
    )
    request_hash = str(matches[0].get("request_hash") or "")
    if not request_hash:
        raise RuntimeError(
            f"carry-forward source lacks a request hash for {user_id}/{case_field}"
        )
    surface = GeneratedSurfaceOnlyCaseDraft.model_validate(surface_payload)
    call = CallResult(
        text=canonical_json(surface_payload),
        raw_response=provider_response,
        usage=usage,
        latency_ms=0.0,
        request_hash=request_hash,
    )
    return surface, call


def _load_successful_surface_from_any_source(
    old_out_dirs: list[Path],
    *,
    user_id: str,
    case_field: str,
) -> tuple[GeneratedSurfaceOnlyCaseDraft, CallResult, Path] | None:
    """Try each source directory in priority order; first match wins.

    Different source directories can legitimately hold the authoritative
    real content for different users -- e.g. a later directory that only
    regenerated a handful of previously excluded/colliding users, while an
    earlier, full-cohort directory remains authoritative for everyone else.
    Carried-forward users never touch their OWN run's physical attempt
    ledger (they are written straight to the compiled bundle file), so a
    later directory's ledger alone cannot recover users it merely inherited
    from an earlier one; the caller must supply that earlier directory too.
    """
    for old_out_dir in old_out_dirs:
        old_ledger_path = old_out_dir / "_generation_physical_attempt_ledger.jsonl"
        if not old_ledger_path.is_file():
            continue
        loaded = _load_successful_surface_from_old_ledger(
            old_ledger_path, user_id=user_id, case_field=case_field
        )
        if loaded is not None:
            surface, call = loaded
            return surface, call, old_out_dir
    return None


def _carry_forward_bundle_from_old_directory(
    *,
    old_out_dirs: list[Path],
    user_id: str,
    families: list[str],
    seed_dialogue: str,
    seed_dialogue_source_id: str,
    endpoint,
    generation_binding: dict[str, Any],
) -> tuple[GeneratedUserBundle, dict[str, Any]] | tuple[None, str]:
    """Recompile one user's bundle from one or more older runs' paid, real
    provider responses, under CURRENT code, rebinding it to the CURRENT
    generation run. Read-only on every old_out_dir. All-or-nothing per
    user: any case missing from EVERY source directory means the whole
    user must be regenerated for real, not partially carried forward here
    (see the casewise-repair path in main() for partial carry-forward of a
    user whose OTHER cases are fine but one specific case is not).

    Directories are tried in the given priority order per case: a later
    directory that only regenerated a handful of users is consulted first,
    falling back to an earlier, full-cohort directory for users/cases it
    never touched.
    """
    existing_dirs = [
        d for d in old_out_dirs
        if (d / "_generation_physical_attempt_ledger.jsonl").is_file()
    ]
    if not existing_dirs:
        return None, f"no carry-forward source directory has a physical attempt ledger: {old_out_dirs}"
    surfaces: dict[str, GeneratedSurfaceOnlyCaseDraft] = {}
    calls: dict[str, CallResult] = {}
    messages: dict[str, list[dict[str, str]]] = {}
    attempt_kinds: dict[str, str] = {}
    accepted_call_keys: dict[str, str] = {}
    case_sources: dict[str, str] = {}
    for case_field, _ in GENERATION_CASE_FIELDS:
        loaded = _load_successful_surface_from_any_source(
            old_out_dirs, user_id=user_id, case_field=case_field
        )
        if loaded is None:
            return None, f"no carried-forward surface available for case {case_field}"
        surface, call, source_dir = loaded
        surfaces[case_field] = surface
        calls[case_field] = call
        messages[case_field] = []
        attempt_kinds[case_field] = "carried_forward"
        accepted_call_keys[case_field] = f"carried_forward:{source_dir}:{user_id}:{case_field}"
        case_sources[case_field] = str(source_dir)
    try:
        bundle = compile_surface_only_user_bundle(
            surfaces=surfaces,
            accepted_calls=calls,
            accepted_messages=messages,
            accepted_attempt_kinds=attempt_kinds,
            seed_dialogue=seed_dialogue,
            user_id=user_id,
            semantic_families=families,
            regimes=list(ResourceNeedRegime),
            generator_model=endpoint.model,
            generator_family=endpoint.family,
        )
    except (ValueError, RuntimeError) as exc:
        return None, f"failed current recompilation/lint: {exc}"
    bundle.provenance.update({"accepted_surface_call_keys": accepted_call_keys})
    used_dirs = sorted(set(case_sources.values()))
    audit = {
        "source_output_directories": used_dirs,
        "case_sources": case_sources,
        "source_physical_attempt_ledger_sha256_by_directory": {
            d: sha256_file(Path(d) / "_generation_physical_attempt_ledger.jsonl")
            for d in used_dirs
        },
    }
    bundle.provenance["carried_forward_from"] = audit
    bind_bundle_to_generation_run(bundle, generation_binding)
    return bundle, audit


def _seed_casewise_carry_forward_into_ledger(
    *,
    ledger: PersistentAttemptLedger,
    all_user_attempts: dict[str, dict[str, Any]],
    casewise_repair_plan: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Pre-seed THIS run's own ledger with the carried-forward (non-flagged)
    case fields of any partially-excluded user, recorded under this run's
    own call keys so the normal pending/call-plan/execution machinery sees
    them as already succeeded and queues a real API call for only the
    flagged field(s), never the whole user. Read-only on every source
    directory named in casewise_repair_plan; this run's own ledger is the
    only thing written to. Idempotent: a second invocation against an
    already-seeded ledger (e.g. a repeated --dry-run) makes no new writes.

    Returns per-user casewise-carry provenance for attaching to the
    eventually compiled bundle.
    """
    casewise_carry_provenance: dict[str, dict[str, Any]] = {}
    for user_id, plan in casewise_repair_plan.items():
        user_plan = all_user_attempts[user_id]
        carried_fields = set(plan["carried_case_fields"])
        carried_case_provenance: dict[str, Any] = {}
        for case_plan in user_plan["cases"]:
            case_field = str(case_plan["case_field"])
            if case_field not in carried_fields:
                continue
            source_dir_str = plan["carried_case_sources"][case_field]
            source_dir = Path(source_dir_str)
            loaded = _load_successful_surface_from_old_ledger(
                source_dir / "_generation_physical_attempt_ledger.jsonl",
                user_id=user_id,
                case_field=case_field,
            )
            if loaded is None:
                raise RuntimeError(
                    "casewise repair lost its previously located "
                    f"carried-forward surface for {user_id}/{case_field} "
                    f"in {source_dir}"
                )
            surface, call = loaded
            attempt = case_plan["attempts"][0]
            call_key = str(attempt["call_key"])
            if not ledger.succeeded(call_key):
                reservation = ledger.reserve(
                    call_key,
                    record_ids={
                        "user_id": user_id,
                        "case_field": case_field,
                        "attempt_kind": "carried_forward",
                        "generation_seed": int(attempt["seed"]),
                    },
                    prompt_sha256=str(attempt["prompt_sha256"]),
                )
                ledger.finish(
                    reservation,
                    succeeded=True,
                    request_hash=call.request_hash,
                    usage=call.usage,
                    error=None,
                    result={
                        "surface": surface.model_dump(mode="json"),
                        "provider_response": call.raw_response,
                        "attempt_kind": "carried_forward",
                    },
                    metadata={
                        "casewise_carried_forward_from": {
                            "source_output_directory": source_dir_str,
                        }
                    },
                )
            carried_case_provenance[case_field] = {
                "source_output_directory": source_dir_str,
                "source_physical_attempt_ledger_sha256": sha256_file(
                    source_dir / "_generation_physical_attempt_ledger.jsonl"
                ),
            }
        casewise_carry_provenance[user_id] = {
            "regenerated_case_fields": plan["regenerated_case_fields"],
            "carried_case_fields": carried_case_provenance,
        }
    return casewise_carry_provenance


def _compute_cross_user_duplicate_repair_manifest(
    *,
    planned_users: list[str],
    bundles: dict[str, GeneratedUserBundle],
    split_by_user: dict[str, PMV2Split],
) -> dict[str, Any]:
    """Deterministic, no-API: which (user_id, case_field) pairs must be
    regenerated because their current_user_text collides with an earlier
    case (by frozen plan order) belonging to a DIFFERENT user.

    This is exactly the failure validate_split_manifests raises at final
    corpus assembly (same-split, cross-user exact duplicates are never
    permitted; only same-user/same-family/same-split counterfactual pairs
    are), computed here so a carry-forward candidate set can be checked
    -- and repaired -- BEFORE spending anything on the remaining pending
    users, instead of discovering it only after paying for them.

    Policy is fixed and auditable, never a human's choice of which user to
    redo: within each cross-user duplicate group, keep the occurrence
    appearing earliest in planned_users (ties broken by
    GENERATION_CASE_FIELDS order) and mark every other member of the group
    for regeneration.
    """
    regime_to_case_field = {
        regime: field for field, regime in GENERATION_CASE_FIELDS
    }
    case_field_rank = {
        field: index for index, (field, _) in enumerate(GENERATION_CASE_FIELDS)
    }
    plan_rank = {user_id: index for index, user_id in enumerate(planned_users)}

    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for user_id, bundle in bundles.items():
        for case in bundle.cases:
            case_field = regime_to_case_field[case.regime]
            normalized = normalize_text(case.current_user_text)
            groups[normalized].append((user_id, case_field))

    repair_cases: list[dict[str, Any]] = []
    for normalized, members in sorted(groups.items()):
        if len({user_id for user_id, _ in members}) < 2:
            continue  # same-user reuse is the allowed counterfactual design
        ordered = sorted(
            members,
            key=lambda member: (plan_rank[member[0]], case_field_rank[member[1]]),
        )
        keep_user_id, keep_case_field = ordered[0]
        for user_id, case_field in ordered[1:]:
            repair_cases.append(
                {
                    "user_id": user_id,
                    "case_field": case_field,
                    "reason": "cross_user_exact_duplicate",
                    "duplicate_of": {
                        "user_id": keep_user_id,
                        "case_field": keep_case_field,
                    },
                    "normalized_text_sha256": sha256_text(normalized),
                }
            )
    repair_cases.sort(
        key=lambda row: (plan_rank[row["user_id"]], case_field_rank[row["case_field"]])
    )
    manifest = {
        "protocol": "pm-v1.5-cross-user-duplicate-repair-manifest-v1",
        "policy": "keep_earliest_frozen_plan_position",
        "repair_cases": repair_cases,
    }
    manifest["manifest_sha256"] = sha256_text(canonical_json(manifest))
    return manifest


def _build_casewise_repair_plan(
    *,
    candidates: dict[str, GeneratedUserBundle],
    repair_manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """For each candidate with a case flagged by the automatic cross-user
    duplicate-repair manifest, derive everything a casewise repair needs
    directly from the manifest and the (about-to-be-discarded) candidate
    bundle -- never from operator-supplied free text. An operator can
    approve the resulting plan the same way as any other stage here (via
    the dry-run's cost_estimate_sha256) but cannot name or alter which
    text is forbidden or which user it collided with.
    """
    regime_to_case_field = {regime: field for field, regime in GENERATION_CASE_FIELDS}
    case_fields_requiring_regeneration: dict[str, set[str]] = defaultdict(set)
    duplicate_of_by_case: dict[tuple[str, str], dict[str, str]] = {}
    for row in repair_manifest["repair_cases"]:
        case_fields_requiring_regeneration[row["user_id"]].add(row["case_field"])
        duplicate_of_by_case[(row["user_id"], row["case_field"])] = dict(
            row["duplicate_of"]
        )

    plan: dict[str, dict[str, Any]] = {}
    for user_id, bundle in candidates.items():
        flagged = case_fields_requiring_regeneration.get(user_id)
        if not flagged:
            continue
        case_sources = bundle.provenance["carried_forward_from"]["case_sources"]
        non_flagged = [
            field for field, _ in GENERATION_CASE_FIELDS if field not in flagged
        ]
        current_text_by_case_field = {
            regime_to_case_field[case.regime]: case.current_user_text
            for case in bundle.cases
        }
        duplicate_repair = {
            case_field: {
                "forbidden_current_user_text": current_text_by_case_field[case_field],
                "forbidden_current_user_text_sha256": sha256_text(
                    current_text_by_case_field[case_field]
                ),
                "duplicate_of": duplicate_of_by_case[(user_id, case_field)],
            }
            for case_field in flagged
        }
        plan[user_id] = {
            "carried_case_fields": non_flagged,
            "regenerated_case_fields": sorted(flagged),
            "carried_case_sources": {
                field: case_sources[field] for field in non_flagged
            },
            "duplicate_repair_manifest_sha256": repair_manifest["manifest_sha256"],
            "duplicate_repair": duplicate_repair,
        }
    return plan


def _find_cross_user_duplicate(
    *,
    normalized_text: str,
    user_id: str,
    accepted_normalized_texts: dict[str, tuple[str, str]],
) -> tuple[str, str] | None:
    """Return the (user_id, case_field) this normalized text collides with,
    if any DIFFERENT user has already had it accepted this run (including
    ones carried forward or recovered from an earlier invocation) --
    excluding the allowed same-user counterfactual reuse validate_split_
    manifests itself permits.
    """
    match = accepted_normalized_texts.get(normalized_text)
    if match is not None and match[0] != user_id:
        return match
    return None


def _is_carried_forward_ledger_row(row: dict[str, Any]) -> bool:
    return (row.get("record_ids") or {}).get("attempt_kind") == "carried_forward"


def _real_physical_attempt_count(ledger: PersistentAttemptLedger) -> int:
    """Physical HTTP attempts actually made against the provider, excluding
    casewise carry-forward entries seeded directly into this same ledger
    (which reuse already-paid content and never issue a real request).

    A casewise-repaired user's carried case fields are recorded as
    STARTED/SUCCEEDED events in this run's OWN physical-attempt ledger --
    the only way the existing pending/call-plan machinery recognizes them
    as already done without a larger refactor -- but that means
    ``ledger.started_attempts`` alone conflates real spend with reused
    history. Every call site that reports cost, budget consumption, or a
    physical-attempt count for THIS run should use this instead.
    """
    seen: set[tuple[str, int]] = set()
    for row in ledger.event_rows:
        if row.get("event") != "STARTED" or _is_carried_forward_ledger_row(row):
            continue
        seen.add((str(row["call_key"]), int(row["attempt_index"])))
    return len(seen)


def _carried_forward_ledger_entry_count(ledger: PersistentAttemptLedger) -> int:
    """Companion to _real_physical_attempt_count: how many of this ledger's
    entries are seeded carry-forward records rather than real attempts, so
    the two numbers are always reported side by side, never one silently
    standing in for the other.
    """
    seen: set[tuple[str, int]] = set()
    for row in ledger.event_rows:
        if row.get("event") == "STARTED" and _is_carried_forward_ledger_row(row):
            seen.add((str(row["call_key"]), int(row["attempt_index"])))
    return len(seen)


def _ledger_usage_for_keys(
    ledger: PersistentAttemptLedger, call_keys: list[str]
) -> dict[str, int]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for call_key in call_keys:
        terminal = ledger.terminal_row(call_key)
        if terminal is None or terminal.get("usage") is None:
            continue
        usage = require_reported_usage(
            terminal.get("usage"), stage="completed PM-v1.5 surface call"
        )
        for key in totals:
            totals[key] += usage[key]
    return totals


def _compile_casewise_user_from_ledger(
    *,
    ledger: PersistentAttemptLedger,
    user_plan: dict[str, Any],
    seed_dialogue: str,
    seed_dialogue_source_id: str,
    endpoint,
    generation_binding: dict[str, Any],
    casewise_carried_forward: dict[str, Any] | None = None,
) -> GeneratedUserBundle | None:
    surfaces: dict[str, GeneratedSurfaceOnlyCaseDraft] = {}
    calls: dict[str, CallResult] = {}
    messages: dict[str, list[dict[str, str]]] = {}
    attempt_kinds: dict[str, str] = {}
    accepted_call_keys: dict[str, str] = {}
    for case_plan in user_plan["cases"]:
        loaded = _load_successful_surface_attempt(
            ledger=ledger, case_plan=case_plan
        )
        if loaded is None:
            return None
        attempt, surface, call = loaded
        case_field = str(case_plan["case_field"])
        surfaces[case_field] = surface
        calls[case_field] = call
        messages[case_field] = attempt["messages"]
        attempt_kinds[case_field] = str(attempt["attempt_kind"])
        accepted_call_keys[case_field] = str(attempt["call_key"])
    bundle = compile_surface_only_user_bundle(
        surfaces=surfaces,
        accepted_calls=calls,
        accepted_messages=messages,
        accepted_attempt_kinds=attempt_kinds,
        seed_dialogue=seed_dialogue,
        user_id=str(user_plan["user_id"]),
        semantic_families=[str(value) for value in user_plan["semantic_families"]],
        regimes=list(ResourceNeedRegime),
        generator_model=endpoint.model,
        generator_family=endpoint.family,
    )
    bundle.provenance.update(
        {
            "seed_dialogue_source_id": seed_dialogue_source_id,
            "accepted_surface_call_keys": accepted_call_keys,
            "all_physical_attempt_usage": _ledger_usage_for_keys(
                ledger,
                [
                    str(attempt["call_key"])
                    for case_plan in user_plan["cases"]
                    for attempt in case_plan["attempts"]
                ],
            ),
        }
    )
    if casewise_carried_forward is not None:
        bundle.provenance["casewise_carried_forward_from"] = casewise_carried_forward
    bind_bundle_to_generation_run(bundle, generation_binding)
    return bundle


def _recover_casewise_bundles_from_ledger(
    *,
    ledger: PersistentAttemptLedger,
    all_user_attempts: dict[str, dict[str, Any]],
    existing: dict[str, GeneratedUserBundle],
    work_path: Path,
    generation_binding: dict[str, Any],
    family_by_user: dict[str, list[str]],
    seeds: list[dict[str, str]],
    endpoint,
    casewise_carry_provenance: dict[str, dict[str, Any]] | None = None,
) -> int:
    casewise_carry_provenance = casewise_carry_provenance or {}
    recovered = 0
    for user_id, user_plan in all_user_attempts.items():
        if user_id in existing:
            continue
        seed_record = seeds[int(user_plan["seed_dialogue_index"])]
        bundle = _compile_casewise_user_from_ledger(
            ledger=ledger,
            user_plan=user_plan,
            seed_dialogue=seed_record["dialogue_text"],
            seed_dialogue_source_id=seed_record["dialogue_id"],
            endpoint=endpoint,
            generation_binding=generation_binding,
            casewise_carried_forward=casewise_carry_provenance.get(user_id),
        )
        if bundle is None:
            continue
        strict_bundle_check(bundle, family_by_user[user_id])
        append_jsonl(work_path, bundle.model_dump(mode="json"))
        existing[user_id] = bundle
        recovered += 1
    return recovered


def _budget_gate(
    estimate: dict[str, Any],
    *,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
) -> dict[str, Any]:
    checks = {
        "maximum_api_calls": int(
            estimate["maximum_physical_api_attempts_including_history"]
        )
        <= max_api_calls,
        "pending_users_have_attempt_capacity": not bool(
            estimate["blocked_pending_users"]
        ),
        "maximum_estimated_cost_usd": float(
            estimate["maximum_estimated_cost_usd"]
        )
        <= max_estimated_usd,
        "max_input_tokens_per_call": int(estimate["max_input_tokens"])
        <= max_input_tokens_per_call,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "limits": {
            "max_api_calls": int(max_api_calls),
            "max_estimated_usd": float(max_estimated_usd),
            "max_input_tokens_per_call": int(max_input_tokens_per_call),
        },
    }


def _strict_existing_bundles(path: Path) -> dict[str, GeneratedUserBundle]:
    result: dict[str, GeneratedUserBundle] = {}
    if not path.exists():
        return result
    for bundle in load_bundles(path):
        if bundle.user_id in result:
            raise RuntimeError(
                f"duplicate resumable generation bundle for user {bundle.user_id}"
            )
        result[bundle.user_id] = bundle
    return result


def _estimate_field_diff(
    saved: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    """Pinpoint which field(s) made two cost-estimate dicts hash differently.

    A bare hash mismatch (the only thing the caller previously reported) is not
    diagnosable after the fact: nothing about which of the ~20 hashed inputs
    (themselves nesting things like a 14-file code manifest) changed survives
    past the raised exception. This recurses through nested dicts
    (code_manifest, generation_run_binding, budget_gate, ...) so a real
    mismatch is a direct dotted-path lookup instead of a multi-hour forensic
    reconstruction (see PM_V1_TO_V1_5_GLOBAL_FAILURE_LEDGER_ZH.md V15-REL-15
    for the prior, narrower instance of this same "CLI/identity
    reproducibility" failure class).
    """
    return {"differing_field_paths": dict_field_diff(saved, current)}


def _require_saved_dry_run(
    out_dir: Path,
    current: dict[str, Any],
) -> None:
    estimate_path = out_dir / "generation_cost_estimate.json"
    plan_path = out_dir / "generation_call_plan.jsonl"
    if not estimate_path.is_file() or not plan_path.is_file():
        raise RuntimeError(
            "API mode requires a completed matching --dry-run in the same output "
            "directory"
        )
    saved = read_json(estimate_path)
    if saved.get("cost_estimate_sha256") != current.get("cost_estimate_sha256"):
        diagnostic_path = out_dir / (
            "generation_dry_run_mismatch_diagnostic_"
            f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
        )
        write_json(
            diagnostic_path,
            {
                "protocol": "pm-v1.5-dry-run-mismatch-diagnostic-v1",
                "saved_cost_estimate_sha256": saved.get("cost_estimate_sha256"),
                "current_cost_estimate_sha256": current.get("cost_estimate_sha256"),
                **_estimate_field_diff(saved, current),
            },
        )
        raise RuntimeError(
            "saved generation dry-run does not match the current generator, seed, "
            "configuration, prompt, code, resume state, pricing, or budget limits; "
            f"see {diagnostic_path} for the exact differing field(s), then run "
            "--dry-run again"
        )
    if saved.get("call_plan_sha256") != current.get("call_plan_sha256"):
        raise RuntimeError("saved generation call-plan hash is stale")


def _require_exact_generation_pilot_binding(
    attestation_path: Path | None,
    *,
    expected_contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and content-bind the exact pilot before formal identity creation."""

    if attestation_path is None:
        raise RuntimeError(
            "formal generation dry-run and paid run require an explicit fresh "
            "--generation-pilot-attestation"
        )
    resolved = attestation_path.resolve()
    try:
        relative_path = str(resolved.relative_to(ROOT.resolve()))
    except ValueError as exc:
        raise RuntimeError(
            "generation pilot attestation must be inside the project root so "
            "the reviewed identity is machine-independent"
        ) from exc
    stale_pilot_directories = {
        "pm_v1_5_generation_compatibility_pilot_v8_candidate",
        "pm_v1_5_generation_compatibility_pilot_v8_1_candidate",
        "pm_v1_5_generation_compatibility_pilot_v8_2_candidate",
        "pm_v1_5_generation_compatibility_pilot_v8_3_candidate",
        "pm_v1_5_generation_compatibility_pilot_post_repair_candidate",
        "pm_v1_5_generation_compatibility_pilot_v8_4_release_candidate",
        "pm_v1_5_generation_compatibility_pilot_v8_5_release_candidate",
        "pm_v1_5_generation_compatibility_pilot_v8_5_final_release_candidate",
        "pm_v1_5_generation_compatibility_pilot_v8_6_release_candidate",
        "pm_v1_5_generation_compatibility_pilot_v8_7_frozen_budget_candidate",
        "pm_v1_5_generation_compatibility_pilot_v8_8_1_final_contract_candidate",
    }
    if resolved.parent.name in stale_pilot_directories:
        raise RuntimeError(
            "formal generation refuses a known consumed/stale compatibility "
            f"pilot directory: {resolved.parent.name}"
        )
    verification = require_generation_compatibility_attestation(
        resolved,
        expected_contract=expected_contract,
    )
    attestation = read_json(resolved)
    internal_sha256 = str(attestation.get("attestation_sha256") or "")
    if len(internal_sha256) != 64:
        raise RuntimeError("generation pilot attestation lacks its internal digest")
    binding = {
        "protocol": "pm-v1.5-exact-generation-pilot-attestation-binding-v1",
        "relative_path": relative_path,
        "artifact_attestation_file_sha256": sha256_file(resolved),
        "attestation_sha256": internal_sha256,
        "compatibility_contract_sha256": str(
            expected_contract["contract_sha256"]
        ),
        "verification_status": str(verification.get("status") or ""),
    }
    if binding["verification_status"] != "PASS":
        raise RuntimeError("generation pilot attestation verification is not PASS")
    return binding, verification


def read_seed_dialogues(path: Path) -> list[dict[str, str]]:
    """Read strict dialogue-level seed records while preserving provenance.

    V1.5_1 no longer accepts anonymous strings: losing ``dialogue_id`` made it
    impossible to prove instance disjointness from the Strategy Bank.
    """

    seeds: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(
                    "PM-v1.5 seed rows must be objects with dialogue_id and text"
                )
            dialogue_id = str(row.get("dialogue_id") or "").strip()
            if not dialogue_id or dialogue_id in seen_ids:
                raise RuntimeError("seed dialogue IDs must be non-empty and unique")
            text = row.get("dialogue_text") or row.get("text") or row.get("content")
            if not text and isinstance(row.get("dialogue"), list):
                text = "\n".join(
                    f"{turn.get('role','unknown')}: {turn.get('content','')}"
                    for turn in row["dialogue"]
                )
            if not text or not str(text).strip():
                raise RuntimeError(f"seed dialogue {dialogue_id} has no usable text")
            normalized_text = str(text).strip()
            declared_hash = str(row.get("seed_text_sha256") or "")
            if declared_hash and declared_hash != sha256_text(normalized_text):
                raise RuntimeError(f"seed dialogue text hash mismatch: {dialogue_id}")
            seeds.append(
                {"dialogue_id": dialogue_id, "dialogue_text": normalized_text}
            )
            seen_ids.add(dialogue_id)
    if not seeds:
        raise ValueError("seed dialogue file contains no usable text")
    return seeds


def require_instance_disjoint_seed_strategy_sources(
    *,
    seeds: list[dict[str, str]],
    planned_user_count: int,
    selected_seed_sources_path: Path,
    strategy_cards: list[StrategyCard],
) -> dict[str, Any]:
    selected_rows = [dict(row) for row in iter_jsonl(selected_seed_sources_path)]
    selected_ids = [str(row.get("dialogue_id") or "") for row in selected_rows]
    actual_ids = [
        str(seeds[index % len(seeds)]["dialogue_id"])
        for index in range(planned_user_count)
    ]
    if (
        len(selected_ids) != planned_user_count
        or len(set(selected_ids)) != len(selected_ids)
        or selected_ids != actual_ids
    ):
        raise RuntimeError(
            "actual 52 seed sources differ from the frozen selected-source manifest"
        )
    bank_ids = {card.source_dialogue_id for card in strategy_cards}
    intersection = sorted(set(actual_ids) & bank_ids)
    if intersection:
        raise RuntimeError(
            "synthetic seed sources overlap the primary Strategy Bank: "
            + str(intersection[:20])
        )
    family_counts: dict[str, int] = {}
    for card in strategy_cards:
        family_counts[card.strategy_label] = family_counts.get(card.strategy_label, 0) + 1
    if len(family_counts) != 8 or any(count < 1 for count in family_counts.values()):
        raise RuntimeError("source-disjoint Strategy Bank does not cover all 8 families")
    return {
        "protocol": "pm-v1.5-dialogue-instance-disjoint-strategy-bank-v1",
        "status": "PASS",
        "selected_seed_sources_path": str(selected_seed_sources_path.resolve()),
        "selected_seed_sources_sha256": sha256_file(selected_seed_sources_path),
        "selected_seed_source_ids": actual_ids,
        "selected_seed_source_ids_sha256": sha256_text(canonical_json(actual_ids)),
        "strategy_source_dialogues": len(bank_ids),
        "strategy_bank_cards": len(strategy_cards),
        "intersection": intersection,
        "strategy_family_card_counts": dict(sorted(family_counts.items())),
    }


def require_frozen_strategy_bank_binding(
    *,
    pm_config: dict[str, Any],
    strategy_bank_path: Path,
    selected_seed_sources_path: Path,
    strategy_cards: list[StrategyCard],
) -> dict[str, Any]:
    """Fail before API use if V1.5 is pointed at a different method input."""

    contract = dict(pm_config.get("strategy_bank_contract") or {})
    required_keys = {
        "protocol",
        "relative_path",
        "sha256",
        "card_count",
        "source_dialogue_count",
        "audit_relative_path",
        "audit_sha256",
        "selected_seed_sources_relative_path",
        "selected_seed_sources_sha256",
        "selected_seed_source_count",
    }
    if set(contract) != required_keys:
        raise RuntimeError("PM-v1.5 Strategy Bank contract keys are missing or stale")
    if contract["protocol"] != "pm-v1.5-frozen-strategy-bank-binding-v1":
        raise RuntimeError("PM-v1.5 Strategy Bank protocol is stale")
    expected_bank = (ROOT / str(contract["relative_path"])).resolve()
    expected_selected = (
        ROOT / str(contract["selected_seed_sources_relative_path"])
    ).resolve()
    audit_path = (ROOT / str(contract["audit_relative_path"])).resolve()
    if strategy_bank_path.resolve() != expected_bank:
        raise RuntimeError("alternate Strategy Bank path is outside the frozen V1.5 method")
    if selected_seed_sources_path.resolve() != expected_selected:
        raise RuntimeError(
            "alternate selected-seed manifest is outside the frozen V1.5 method"
        )
    if sha256_file(expected_bank) != str(contract["sha256"]):
        raise RuntimeError("frozen V1.5 Strategy Bank hash mismatch")
    if not audit_path.is_file() or sha256_file(audit_path) != str(
        contract["audit_sha256"]
    ):
        raise RuntimeError("frozen V1.5 Strategy Bank audit hash mismatch")
    if sha256_file(expected_selected) != str(
        contract["selected_seed_sources_sha256"]
    ):
        raise RuntimeError("frozen V1.5 selected-seed manifest hash mismatch")
    source_ids = {card.source_dialogue_id for card in strategy_cards}
    selected_rows = list(iter_jsonl(expected_selected))
    observed = {
        "card_count": len(strategy_cards),
        "source_dialogue_count": len(source_ids),
        "selected_seed_source_count": len(selected_rows),
    }
    expected = {
        "card_count": int(contract["card_count"]),
        "source_dialogue_count": int(contract["source_dialogue_count"]),
        "selected_seed_source_count": int(contract["selected_seed_source_count"]),
    }
    if observed != expected:
        raise RuntimeError(
            f"frozen V1.5 Strategy Bank counts mismatch: {observed} != {expected}"
        )
    return {
        "protocol": contract["protocol"],
        "status": "PASS",
        "strategy_bank_path": str(expected_bank),
        "strategy_bank_sha256": contract["sha256"],
        "strategy_bank_audit_path": str(audit_path),
        "strategy_bank_audit_sha256": contract["audit_sha256"],
        "selected_seed_sources_path": str(expected_selected),
        "selected_seed_sources_sha256": contract[
            "selected_seed_sources_sha256"
        ],
        **observed,
    }


def strict_bundle_check(bundle: GeneratedUserBundle, allowed_families: list[str]) -> None:
    validate_bundle(bundle)
    expected_regimes = set(ResourceNeedRegime)
    observed_regimes = {case.regime for case in bundle.cases}
    if observed_regimes != expected_regimes or len(bundle.cases) != len(expected_regimes):
        raise ValueError(
            "bundle must contain exactly one case per PM-v2 regime; "
            f"missing={sorted(x.value for x in expected_regimes-observed_regimes)}, "
            f"extra={len(bundle.cases)-len(observed_regimes)}"
        )
    expected_families = set(allowed_families)
    observed_families = {case.semantic_family for case in bundle.cases}
    invalid_families = sorted(observed_families - expected_families)
    missing_families = sorted(expected_families - observed_families)
    if invalid_families or missing_families:
        raise ValueError(
            "bundle semantic families must exactly cover its frozen assignment: "
            f"missing={missing_families}, extra={invalid_families}"
        )
    for case in bundle.cases:
        missing_sources = [
            name
            for name, memories in (
                ("MP", case.profile_memories),
                ("MS", case.summary_memories),
                ("ME", case.event_memories),
            )
            if not memories
        ]
        if missing_sources:
            raise ValueError(
                f"case {case.case_id} is missing sources {missing_sources}; "
                "all development states must expose the complete 16-action space"
            )
        source_sizes = {
            "MP": len(case.profile_memories),
            "MS": len(case.summary_memories),
            "ME": len(case.event_memories),
        }
        if set(source_sizes.values()) != {2}:
            raise ValueError(
                f"case {case.case_id} has label-revealing source inventory sizes: "
                f"{source_sizes}; every source must expose exactly two items"
            )
    validate_successful_generation_trace(bundle)


def enforce_full_state_design(
    report: dict,
    *,
    train_users: int,
    calibration_users: int,
    internal_test_users: int,
) -> dict:
    """Fail closed on the YAML-frozen full-state and 14/5/5 family design."""

    cases_per_user = len(ResourceNeedRegime)
    expected_split_state_counts = {
        PMV2Split.TRAIN.value: train_users * cases_per_user,
        PMV2Split.CALIBRATION.value: calibration_users * cases_per_user,
        PMV2Split.INTERNAL_TEST.value: internal_test_users * cases_per_user,
    }
    if report["split_counts"] != expected_split_state_counts:
        raise RuntimeError(
            "generated PM-v2 state counts do not match the frozen full design: "
            f"expected={expected_split_state_counts}, observed={report['split_counts']}"
        )
    expected_total_states = sum(expected_split_state_counts.values())
    minimum_unique_current_texts = (expected_total_states * 7 + 8) // 9
    split_manifest_report = report["split_manifest"]
    if (
        report["n_states"] != expected_total_states
        or split_manifest_report["total_states"] != expected_total_states
        or split_manifest_report["unique_normalized_current_user_texts"]
        < minimum_unique_current_texts
    ):
        raise RuntimeError(
            "generated PM-v2 current-user texts violate the frozen controlled "
            "counterfactual-diversity floor"
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
            "generated PM-v2 semantic-family union counts do not match 14/5/5: "
            f"{observed_family_union_counts}"
        )
    result = {
        "status": "PASS",
        "cases_per_user": cases_per_user,
        "expected_split_state_counts": expected_split_state_counts,
        "expected_total_states": expected_total_states,
        "minimum_unique_normalized_current_user_texts": (
            minimum_unique_current_texts
        ),
        "unique_normalized_current_user_texts": split_manifest_report[
            "unique_normalized_current_user_texts"
        ],
        "normalized_current_user_text_unique_rate": split_manifest_report[
            "normalized_current_user_text_unique_rate"
        ],
        "current_user_text_diversity_policy": (
            "same-user/same-family/same-split pairs only; max group 2; "
            "minimum 7 unique per 9-case bundle"
        ),
        "semantic_family_union_counts": observed_family_union_counts,
    }
    report["full_state_design"] = result
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed PM-v2 synthetic generation: save an exact prompt plan "
            "and accepted conservative cost bound before API use."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--pm-v2-config", type=Path, default=ROOT / "configs" / "pm_v1_5.yaml")
    parser.add_argument("--endpoint")
    parser.add_argument(
        "--seed-dialogues",
        type=Path,
        default=ROOT / "data" / "pm_v2" / "train_seed_dialogues_v1_5.jsonl",
    )
    parser.add_argument(
        "--strategy-bank",
        type=Path,
        default=ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl",
    )
    parser.add_argument(
        "--selected-seed-sources",
        type=Path,
        default=(
            ROOT
            / "data"
            / "strategy"
            / "pm_v1_5_selected_seed_sources.jsonl"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "pm_v1_5")
    parser.add_argument(
        "--carry-forward-from",
        type=Path,
        nargs="+",
        help=(
            "One or more older output directories whose already-paid, real "
            "provider responses should be recompiled under current code and "
            "carried into this run instead of being regenerated. When "
            "multiple directories are given, each (user_id, case_field) is "
            "recovered from the first directory (in the given order) whose "
            "immutable ledger has a successful surface for it -- e.g. a "
            "later directory that only regenerated a handful of previously "
            "excluded/colliding users, falling back to the original "
            "full-cohort directory for everyone else. Read-only on every "
            "source directory. A user whose carried-forward content no "
            "longer passes current lint/schema checks, in full, is left "
            "pending for real regeneration of ONLY the case field(s) that "
            "no longer pass -- other case fields for that user are still "
            "carried forward, never silently dropped or forced through."
        ),
    )
    parser.add_argument("--train-users", type=int)
    parser.add_argument("--calibration-users", type=int)
    parser.add_argument("--internal-test-users", type=int)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--max-generation-attempts", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--input-usd-per-mtok",
        type=float,
        help="Optional exact-match assertion against frozen data-generation pricing.",
    )
    parser.add_argument(
        "--output-usd-per-mtok",
        type=float,
        help="Optional exact-match assertion against frozen data-generation pricing.",
    )
    parser.add_argument("--max-api-calls", type=int)
    parser.add_argument("--max-estimated-usd", type=float)
    parser.add_argument("--max-input-tokens-per-call", type=int)
    parser.add_argument("--accept-cost-estimate-sha256")
    parser.add_argument(
        "--generation-pilot-attestation",
        type=Path,
        help=(
            "Explicit fresh PASS casewise surface-only pilot attestation. "
            "Required for full --run; there is deliberately no default because "
            "consumed or stale pilot identities must never be inherited."
        ),
    )
    parser.add_argument(
        "--automated-semantic-review-report",
        type=Path,
        help=(
            "Deprecated and forbidden for formal generation. The consumed V4 "
            "calibration review cannot authorize new data; actual structured QA "
            "runs after the 468-state corpus exists and before the action sweep."
        ),
    )
    parser.add_argument(
        "--automated-semantic-review-attestation",
        type=Path,
        help="Deprecated companion to --automated-semantic-review-report; forbidden.",
    )
    args = parser.parse_args()

    if args.run and args.overwrite:
        raise RuntimeError("paid PM-v1.5 data-generation runs prohibit --overwrite")

    if (
        args.input_usd_per_mtok is not None
        and args.input_usd_per_mtok < 0
    ) or (
        args.output_usd_per_mtok is not None
        and args.output_usd_per_mtok < 0
    ):
        raise ValueError("pricing must be non-negative")
    if (
        (args.max_api_calls is not None and args.max_api_calls <= 0)
        or (args.max_estimated_usd is not None and args.max_estimated_usd < 0)
        or (
            args.max_input_tokens_per_call is not None
            and args.max_input_tokens_per_call <= 0
        )
    ):
        raise ValueError("budget limits must be positive (USD may be zero)")

    pm_config = load_config(args.pm_v2_config)
    if pm_config.get("version") != "pm-v1.5":
        raise ValueError("PM-v1.5 data generation requires a pm-v1.5 config")
    require_unified_semantic_query_contract(pm_config)
    formal_execution_cfg = dict(pm_config["formal_generation_execution"])
    formal_transport_cfg = dict(formal_execution_cfg["transport_retry"])
    if (
        formal_transport_cfg.get("protocol") != RETRY_CONTRACT_PROTOCOL
        or int(
            formal_transport_cfg.get(
                "maximum_physical_attempts_per_content_attempt", 0
            )
        )
        != GENERATION_PILOT_MAX_TRANSPORT_ATTEMPTS_PER_CONTENT_ATTEMPT
        or tuple(
            float(value)
            for value in formal_transport_cfg.get("transport_backoff_seconds", [])
        )
        != tuple(GENERATION_PILOT_TRANSPORT_BACKOFF_SECONDS)
        or set(formal_transport_cfg.get("retryable_classes", []))
        != {
            "http_5xx",
            "network_timeout",
            "rate_limited_429",
            "request_timeout_408",
        }
    ):
        raise ValueError(
            "formal generation transport retry contract differs from the "
            "successful compatibility pilot"
        )
    formal_budget_cfg = dict(formal_execution_cfg["budget_limits"])
    frozen_budget = {
        "max_api_calls": int(formal_budget_cfg["max_api_calls"]),
        "max_estimated_usd": float(formal_budget_cfg["max_estimated_usd"]),
        "max_input_tokens_per_call": int(
            formal_budget_cfg["max_input_tokens_per_call"]
        ),
    }
    for name, frozen_value in frozen_budget.items():
        requested_value = getattr(args, name)
        if requested_value is not None and requested_value != frozen_value:
            raise RuntimeError(
                f"{name} override differs from the frozen formal-generation budget"
            )
        setattr(args, name, frozen_value)
    # Fail before any paid generation if the exact deployable semantic
    # observation mechanism cannot be reconstructed locally.
    semantic_encoder = FrozenTransformerSemanticEncoder.load(
        semantic_encoder_spec_from_config(pm_config)
    )
    semantic_runtime_verification = require_semantic_runtime_contract(
        pm_config, semantic_encoder
    )
    readiness_challenge = readiness_natural_language_challenge(semantic_encoder)
    require_paid_run_release(
        pm_config,
        config_path=args.pm_v2_config,
        stage="development_data_generation",
        run=bool(args.run),
        run_identity=args.accept_cost_estimate_sha256,
    )
    generation_cfg = pm_config["data_generation"]
    generation_pricing = dict(generation_cfg["pricing_usd_per_mtok"])
    if set(generation_pricing) != {"input", "output"}:
        raise ValueError("data_generation pricing must contain input/output")
    frozen_input_price = float(generation_pricing["input"])
    frozen_output_price = float(generation_pricing["output"])
    if (frozen_input_price, frozen_output_price) != (0.15, 0.60):
        raise ValueError("PM-v2 gpt-4o-mini generation pricing must be 0.15/0.60")
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
    api_cost_cfg = dict(pm_config["api_cost_planning"])
    if set(api_cost_cfg) != {
        "input_token_safety_factor",
        "fail_on_reported_input_overrun",
    }:
        raise ValueError("api_cost_planning keys do not match the PM-v2 contract")
    input_token_safety_factor = float(api_cost_cfg["input_token_safety_factor"])
    fail_on_reported_input_overrun = bool(
        api_cost_cfg["fail_on_reported_input_overrun"]
    )
    if input_token_safety_factor < 1.0:
        raise ValueError("input_token_safety_factor must be at least 1")
    if not fail_on_reported_input_overrun:
        raise ValueError("PM-v2 requires fail_on_reported_input_overrun=true")
    retrieval_cfg = pm_config["retrieval"]
    external_cfg = pm_config["external_evaluation"]
    strategy_top_k = int(retrieval_cfg["strategy_top_k"])
    if int(external_cfg["strategy_top_k"]) != strategy_top_k:
        raise ValueError(
            "development/external strategy_top_k mismatch in PM-v2 config"
        )
    strategy_estimated_tokens = int(external_cfg["strategy_action_tokens"])
    strategy_cards = [
        StrategyCard.model_validate(row) for row in iter_jsonl(args.strategy_bank)
    ]
    if not strategy_cards:
        raise ValueError("strategy bank is empty")
    frozen_strategy_bank_binding = require_frozen_strategy_bank_binding(
        pm_config=pm_config,
        strategy_bank_path=args.strategy_bank,
        selected_seed_sources_path=args.selected_seed_sources,
        strategy_cards=strategy_cards,
    )
    strategy_catalog_count = len(strategy_cards)
    strategy_bank_sha256 = sha256_file(args.strategy_bank)
    frozen_generation_values = {
        "endpoint": str(generation_cfg["generator_endpoint"]),
        "train_users": int(generation_cfg["train_users"]),
        "calibration_users": int(generation_cfg["calibration_users"]),
        "internal_test_users": int(generation_cfg["internal_test_users"]),
        "max_generation_attempts": int(generation_cfg["max_generation_attempts"]),
        "pricing_usd_per_mtok": {
            "input": frozen_input_price,
            "output": frozen_output_price,
        },
        "seed": int(generation_cfg["base_seed"]),
    }
    requested_generation_values = {
        "endpoint": args.endpoint,
        "train_users": args.train_users,
        "calibration_users": args.calibration_users,
        "internal_test_users": args.internal_test_users,
        "max_generation_attempts": args.max_generation_attempts,
        "pricing_usd_per_mtok": {
            "input": args.input_usd_per_mtok,
            "output": args.output_usd_per_mtok,
        },
        "seed": args.seed,
    }
    for name, requested in requested_generation_values.items():
        if requested is not None and requested != frozen_generation_values[name]:
            raise RuntimeError(
                f"{name} override differs from the PM-v2 config; data-generation "
                "protocol changes must be reviewed in YAML before dry-run"
            )
    args.endpoint = frozen_generation_values["endpoint"]
    args.train_users = frozen_generation_values["train_users"]
    args.calibration_users = frozen_generation_values["calibration_users"]
    args.internal_test_users = frozen_generation_values["internal_test_users"]
    args.max_generation_attempts = frozen_generation_values[
        "max_generation_attempts"
    ]
    args.seed = frozen_generation_values["seed"]
    required_regimes = {str(value) for value in generation_cfg["required_regimes"]}
    if required_regimes != {regime.value for regime in ResourceNeedRegime}:
        raise ValueError("PM-v2 config required_regimes does not match code contract")
    if int(generation_cfg["cases_per_user"]) != len(ResourceNeedRegime):
        raise ValueError("PM-v2 config cases_per_user must equal the regime count")
    observable_cfg = generation_cfg.get("observable_state_support") or {}
    if (
        observable_cfg.get("protocol") != OBSERVABLE_STATE_SUPPORT_PROTOCOL
        or observable_cfg.get("history_turn_targets")
        != list(OBSERVABLE_HISTORY_TURN_TARGETS)
        or observable_cfg.get("summary_treatments") != ["present", "absent"]
        or observable_cfg.get("evoemo_instance_text_or_outcomes_used") is not False
    ):
        raise ValueError("PM-v1.5 observable-state support config is stale")
    if min(args.train_users, args.calibration_users, args.internal_test_users) < 0:
        raise ValueError("user counts must be non-negative")
    if args.max_users is not None and args.max_users < 0:
        raise ValueError("max-users must be non-negative")
    if args.run and args.max_users is not None:
        raise RuntimeError(
            "API generation forbids --max-users: a partial synthetic bundle cannot "
            "satisfy the frozen train/calibration/internal-test design"
        )
    if args.max_generation_attempts < 1:
        raise ValueError("max-generation-attempts must be positive")
    if args.max_generation_attempts != 1 + SURFACE_GENERATION_MAX_REPAIRS:
        raise ValueError(
            "max-generation-attempts must equal initial plus the single frozen repair"
        )

    seeds = read_seed_dialogues(args.seed_dialogues)
    experiment_config = load_config(args.config)
    endpoint = endpoint_from_config(experiment_config, args.endpoint)
    split_specs = [
        (PMV2Split.TRAIN, args.train_users, TRAIN_SEMANTIC_FAMILIES),
        (
            PMV2Split.CALIBRATION,
            args.calibration_users,
            CALIBRATION_SEMANTIC_FAMILIES,
        ),
        (
            PMV2Split.INTERNAL_TEST,
            args.internal_test_users,
            INTERNAL_TEST_SEMANTIC_FAMILIES,
        ),
    ]
    split_by_user: dict[str, PMV2Split] = {}
    family_by_user: dict[str, list[str]] = {}
    planned_users: list[str] = []
    global_index = 0
    for split, count, family_pool in split_specs:
        cohorts = SEMANTIC_FAMILY_COHORTS_BY_SPLIT[split.value]
        family_schedule = generation_family_schedule(split.value, count)
        cohort_union = {family for cohort in cohorts for family in cohort}
        if cohort_union != set(family_pool):
            raise RuntimeError(
                f"semantic-family cohorts do not exactly cover {split.value}: "
                f"expected={sorted(family_pool)}, observed={sorted(cohort_union)}"
            )
        if any(len(cohort) != 3 or len(set(cohort)) != 3 for cohort in cohorts):
            raise RuntimeError(
                f"semantic-family cohorts for {split.value} must be unique triads"
            )
        for local_index, scheduled_families in enumerate(family_schedule):
            if args.max_users is not None and global_index >= args.max_users:
                break
            user_id = f"pmv2_{split.value}_u{local_index + 1:03d}"
            families = list(scheduled_families)
            split_by_user[user_id] = split
            family_by_user[user_id] = families
            planned_users.append(user_id)
            global_index += 1

    seed_strategy_disjointness = require_instance_disjoint_seed_strategy_sources(
        seeds=seeds,
        planned_user_count=len(planned_users),
        selected_seed_sources_path=args.selected_seed_sources,
        strategy_cards=strategy_cards,
    )

    code_paths = [
        Path(__file__).resolve(),
        ROOT / "src" / "metacom_pm" / "pm_v2_data.py",
        ROOT / "src" / "metacom_pm" / "pm_v2_contracts.py",
        ROOT / "src" / "metacom_pm" / "pm_v1_5_semantic.py",
        ROOT / "src" / "metacom_pm" / "pm_v1_5_step0.py",
        ROOT / "src" / "metacom_pm" / "pm_v2_generation_pilot.py",
        ROOT / "src" / "metacom_pm" / "pm_v2_generation_review_v8.py",
        ROOT / "src" / "metacom_pm" / "attempt_ledger.py",
        ROOT / "src" / "metacom_pm" / "bounded_retry.py",
        ROOT / "src" / "metacom_pm" / "paid_run_release.py",
        ROOT / "src" / "metacom_pm" / "api.py",
        ROOT / "src" / "metacom_pm" / "config.py",
        ROOT / "src" / "metacom_pm" / "io.py",
        ROOT / "src" / "metacom_pm" / "text.py",
    ]
    code_manifest = {
        str(path.relative_to(ROOT)): sha256_file(path) for path in code_paths
    }
    endpoint_descriptor = {
        "endpoint_name": args.endpoint,
        "base_url": endpoint.base_url,
        "model": endpoint.model,
        "family": endpoint.family,
        "api_key_env": endpoint.api_key_env,
        "timeout_seconds": endpoint.timeout_seconds,
        "temperature": GENERATION_TEMPERATURE,
        "max_output_tokens": SURFACE_GENERATION_MAX_OUTPUT_TOKENS,
        # One HTTP request per high-level attempt makes the accepted maximum
        # enforceable by this script instead of hiding client-internal retries.
        "request_retries": GENERATION_REQUEST_RETRIES,
        "max_generation_attempts": args.max_generation_attempts,
    }
    user_plan = [
        {
            "user_id": user_id,
            "split": split_by_user[user_id].value,
            "semantic_families": family_by_user[user_id],
            "seed_dialogue_index": index % len(seeds),
            "seed_dialogue_source_id": seeds[index % len(seeds)]["dialogue_id"],
            "seed_dialogue_sha256": sha256_text(
                seeds[index % len(seeds)]["dialogue_text"]
            ),
        }
        for index, user_id in enumerate(planned_users)
    ]
    data_plan = {
        "requested_user_counts": {
            "train": args.train_users,
            "calibration": args.calibration_users,
            "internal_test": args.internal_test_users,
        },
        "max_users": args.max_users,
        "users": user_plan,
    }
    full_user_count = (
        int(args.train_users)
        + int(args.calibration_users)
        + int(args.internal_test_users)
    )
    generation_compatibility_contract = build_generation_compatibility_contract(
        project_root=ROOT,
        experiment_config_path=args.config,
        pm_v2_config_path=args.pm_v2_config,
        seed_dialogues_path=args.seed_dialogues,
        endpoint=endpoint,
        base_generation_seed=int(args.seed),
        full_user_count=full_user_count,
        input_token_safety_factor=input_token_safety_factor,
        fail_on_reported_input_overrun=fail_on_reported_input_overrun,
        input_usd_per_mtok=args.input_usd_per_mtok,
        output_usd_per_mtok=args.output_usd_per_mtok,
    )
    memory_blueprint_preflight = deterministic_memory_blueprint_preflight()
    (
        generation_pilot_attestation_binding,
        generation_pilot_verification,
    ) = _require_exact_generation_pilot_binding(
        args.generation_pilot_attestation,
        expected_contract=generation_compatibility_contract,
    )
    generation_binding = {
        "protocol": "pm_v2_generation_resume_binding_v3_casewise_surface_ledger",
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        "api_cost_planning": {
            "input_token_safety_factor": input_token_safety_factor,
            "fail_on_reported_input_overrun": fail_on_reported_input_overrun,
        },
        "formal_transport_retry": formal_transport_cfg,
        "formal_budget_limits": frozen_budget,
        "experiment_config_sha256": sha256_file(args.config),
        "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
        "seed_dialogues_sha256": sha256_file(args.seed_dialogues),
        "strategy_catalog": {
            "path": str(args.strategy_bank.resolve()),
            "sha256": strategy_bank_sha256,
            "count": strategy_catalog_count,
            "estimated_action_tokens": strategy_estimated_tokens,
            "top_k": strategy_top_k,
        },
        "frozen_strategy_bank_binding": frozen_strategy_bank_binding,
        "seed_strategy_instance_disjointness": seed_strategy_disjointness,
        "base_generation_seed": args.seed,
        "generator": endpoint_descriptor,
        "generator_config_sha256": sha256_text(canonical_json(endpoint_descriptor)),
        "prompt_contract_sha256": surface_generation_contract_hash(),
        "code_manifest": code_manifest,
        "code_manifest_sha256": sha256_text(canonical_json(code_manifest)),
        "data_plan_sha256": sha256_text(canonical_json(data_plan)),
        "generation_compatibility_contract_version": (
            generation_compatibility_contract["version"]
        ),
        "generation_compatibility_contract_sha256": (
            generation_compatibility_contract["contract_sha256"]
        ),
        "generation_pilot_attestation": generation_pilot_attestation_binding,
        "deterministic_memory_blueprint_preflight": memory_blueprint_preflight,
    }
    generation_binding_sha256 = sha256_text(canonical_json(generation_binding))

    work_path = args.out_dir / "_generated_bundles_work.jsonl"
    error_path = args.out_dir / "_generation_errors.jsonl"
    attempt_ledger_path = args.out_dir / "_generation_physical_attempt_ledger.jsonl"
    forbid_overwrite_of_spent_attempts(
        attempt_ledger_path,
        overwrite=args.overwrite,
        stage="PM-v2 synthetic bundle generation",
    )
    if args.overwrite:
        for path in (work_path, error_path, attempt_ledger_path):
            if path.exists():
                path.unlink()
    existing = _strict_existing_bundles(work_path)
    unexpected_existing = sorted(set(existing) - set(planned_users))
    if unexpected_existing:
        raise RuntimeError(
            "resumable work file contains users outside the active immutable data "
            f"plan: {unexpected_existing}; use a new output directory or --overwrite"
        )

    carry_forward_report: dict[str, Any] = {
        "source_output_directories": None,
        "carried_users": [],
        "skipped_users": {},
        "duplicate_repair_manifest": None,
        "casewise_repair_plan": {},
    }
    casewise_repair_plan: dict[str, dict[str, Any]] = {}
    if args.carry_forward_from is not None:
        carry_forward_report["source_output_directories"] = [
            str(d.resolve()) for d in args.carry_forward_from
        ]
        # Pass 1: collect every carry-forward candidate WITHOUT writing it
        # yet, so the whole candidate set can be checked together for the
        # cross-user duplicate failure validate_split_manifests would only
        # otherwise catch at final assembly -- after paying for the
        # remaining pending users too.
        candidates: dict[str, GeneratedUserBundle] = {}
        for user_index, user_id in enumerate(planned_users):
            if user_id in existing:
                continue
            seed_record = seeds[user_index % len(seeds)]
            bundle, outcome = _carry_forward_bundle_from_old_directory(
                old_out_dirs=args.carry_forward_from,
                user_id=user_id,
                families=family_by_user[user_id],
                seed_dialogue=seed_record["dialogue_text"],
                seed_dialogue_source_id=seed_record["dialogue_id"],
                endpoint=endpoint,
                generation_binding=generation_binding,
            )
            if bundle is None:
                carry_forward_report["skipped_users"][user_id] = outcome
                continue
            strict_bundle_check(bundle, family_by_user[user_id])
            candidates[user_id] = bundle

        # Checked against the full would-be-included set (already-`existing`
        # bundles from a PRIOR invocation of this same out_dir, plus this
        # invocation's own candidates) -- not candidates alone. On a second
        # --dry-run against an already-populated directory, a duplicate's
        # earlier-plan-position partner may already be sitting in `existing`
        # (persisted by the first invocation) and absent from `candidates`,
        # which would otherwise make the check silently miss the exact same
        # collision it just found, producing another unstable identity.
        repair_manifest = _compute_cross_user_duplicate_repair_manifest(
            planned_users=planned_users,
            bundles={**existing, **candidates},
            split_by_user=split_by_user,
        )
        carry_forward_report["duplicate_repair_manifest"] = repair_manifest
        # Fixed, deterministic policy (never a human's choice of which user
        # to redo): a user with a case flagged by the repair manifest keeps
        # every OTHER case carried forward from the same source(s) and only
        # has the flagged case field(s) regenerated for real (casewise
        # repair) -- not a whole-user regeneration. Whole-user regeneration
        # was tried first and its residual risk materialized for real
        # twice in a row (V8.16 dedup-repair regenerated calibration_u012
        # whole, which then collided with a DIFFERENT user each time,
        # first u009 then u010): touching all 9 fresh cases instead of just
        # the 1 that actually needs it needlessly re-exposes the other 8,
        # already-fine cases to a brand new chance of colliding with
        # someone else.
        casewise_repair_plan = _build_casewise_repair_plan(
            candidates=candidates, repair_manifest=repair_manifest
        )

        # Pass 2: write clean candidates whole; partially-flagged candidates
        # are held back here (not written to existing) so their carried
        # case fields can be re-attached to a real, freshly generated
        # surface for the flagged field(s) further below, once this run's
        # own call keys exist.
        for user_id, bundle in candidates.items():
            if user_id in casewise_repair_plan:
                continue
            append_jsonl(work_path, bundle.model_dump(mode="json"))
            existing[user_id] = bundle
        carry_forward_report["casewise_repair_plan"] = {
            user_id: {
                key: value
                for key, value in plan.items()
                if key != "duplicate_repair"
            }
            | {
                "duplicate_repair": {
                    case_field: {
                        "forbidden_current_user_text_sha256": entry[
                            "forbidden_current_user_text_sha256"
                        ],
                        "duplicate_of": entry["duplicate_of"],
                    }
                    for case_field, entry in plan["duplicate_repair"].items()
                }
            }
            for user_id, plan in casewise_repair_plan.items()
        }
        # Derived from the final state of `existing`, not from which
        # invocation happened to append each bundle: --dry-run itself
        # persists carried-forward bundles into this out_dir's own work
        # file, so a later --run invocation would otherwise see them as
        # already-`existing` and silently omit them from this list, making
        # the saved dry-run and the freshly recomputed estimate disagree
        # over an identical, unchanged set of completed users.
        carry_forward_report["carried_users"] = sorted(
            user_id
            for user_id, bundle in existing.items()
            if bundle.provenance.get("carried_forward_from") is not None
        )

    all_user_attempts: dict[str, dict[str, Any]] = {}
    for user_index, user_id in enumerate(planned_users):
        seed_dialogue_index = user_index % len(seeds)
        seed_record = seeds[seed_dialogue_index]
        seed_dialogue = seed_record["dialogue_text"]
        families = family_by_user[user_id]
        assignments = generation_case_family_assignments(
            families, list(ResourceNeedRegime)
        )
        cases: list[dict[str, Any]] = []
        for case_index, (case_field, regime) in enumerate(
            GENERATION_CASE_FIELDS
        ):
            semantic_family = assignments[case_field]
            forbidden_families = [
                family for family in families if family != semantic_family
            ]
            duplicate_repair = casewise_repair_plan.get(user_id, {}).get(
                "duplicate_repair", {}
            ).get(case_field)
            attempts: list[dict[str, Any]] = []
            for attempt_index in range(args.max_generation_attempts):
                attempt_kind = "initial" if attempt_index == 0 else "repair"
                messages = generation_case_messages(
                    seed_dialogue=seed_dialogue,
                    user_id=user_id,
                    case_field=case_field,
                    regime=regime,
                    semantic_family=semantic_family,
                    forbidden_families=forbidden_families,
                    repair=attempt_kind == "repair",
                    forbidden_exact_texts=(
                        [duplicate_repair["forbidden_current_user_text"]]
                        if duplicate_repair is not None
                        else ()
                    ),
                )
                attempt_seed = (
                    int(args.seed)
                    + user_index * 1000
                    + case_index * 10
                    + attempt_index
                    + (DUPLICATE_REPAIR_SEED_OFFSET if duplicate_repair is not None else 0)
                )
                request_contract = chat_request_payload(
                    endpoint,
                    messages,
                    temperature=GENERATION_TEMPERATURE,
                    max_tokens=SURFACE_GENERATION_MAX_OUTPUT_TOKENS,
                    seed=attempt_seed,
                    response_schema=GeneratedSurfaceOnlyCaseDraft,
                )
                request_contract_json = canonical_json(request_contract)
                request_contract_sha256 = sha256_text(request_contract_json)
                prompt_sha256 = sha256_text(canonical_json(messages))
                record_ids = {
                    "user_id": user_id,
                    "case_field": case_field,
                    "attempt_kind": (
                        f"duplicate_repair_{attempt_kind}"
                        if duplicate_repair is not None
                        else attempt_kind
                    ),
                    "generation_seed": attempt_seed,
                }
                call_key = physical_call_key(
                    stage=GENERATION_STAGE,
                    record_ids=record_ids,
                    prompt_sha256=prompt_sha256,
                    endpoint=endpoint,
                    request_parameters={
                        "temperature": GENERATION_TEMPERATURE,
                        "max_tokens": SURFACE_GENERATION_MAX_OUTPUT_TOKENS,
                        "seed": attempt_seed,
                        "response_schema": GeneratedSurfaceOnlyCaseDraft.__name__,
                        "request_contract_sha256": request_contract_sha256,
                    },
                )
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "attempt_kind": attempt_kind,
                        "seed": attempt_seed,
                        "messages": messages,
                        "prompt_sha256": prompt_sha256,
                        "request_contract_sha256": request_contract_sha256,
                        "call_key": call_key,
                        "raw_estimated_input_tokens": estimate_tokens(
                            request_contract_json
                        ),
                        "estimated_input_tokens": conservative_token_bound(
                            request_contract_json,
                            safety_factor=input_token_safety_factor,
                        ),
                    }
                )
            cases.append(
                {
                    "user_id": user_id,
                    "case_field": case_field,
                    "regime": regime.value,
                    "semantic_family": semantic_family,
                    "attempts": attempts,
                }
            )
        all_user_attempts[user_id] = {
            "user_id": user_id,
            "split": split_by_user[user_id].value,
            "semantic_families": families,
            "seed_dialogue_index": seed_dialogue_index,
            "seed_dialogue_source_id": seed_record["dialogue_id"],
            "seed_dialogue_sha256": sha256_text(seed_dialogue),
            "maximum_output_tokens_per_call": (
                SURFACE_GENERATION_MAX_OUTPUT_TOKENS
            ),
            "maximum_attempts_per_case": args.max_generation_attempts,
            "cases": cases,
        }

    all_call_keys = [
        str(attempt["call_key"])
        for row in all_user_attempts.values()
        for case_plan in row["cases"]
        for attempt in case_plan["attempts"]
    ]
    if len(all_call_keys) != len(set(all_call_keys)):
        raise RuntimeError("synthetic generation produced duplicate physical call keys")
    ledger = PersistentAttemptLedger(
        attempt_ledger_path,
        stage=GENERATION_STAGE,
        expected_calls={
            call_key: GENERATION_PILOT_MAX_TRANSPORT_ATTEMPTS_PER_CONTENT_ATTEMPT
            for call_key in all_call_keys
        },
        maximum_total_attempts=max(
            len(all_call_keys)
            * GENERATION_PILOT_MAX_TRANSPORT_ATTEMPTS_PER_CONTENT_ATTEMPT,
            1,
        ),
    )
    # Casewise repair: seed THIS run's own ledger with the carried-forward
    # (non-flagged) case fields of any partially-excluded user, recorded
    # under this run's own call keys so the normal pending/call-plan/
    # execution machinery below sees them as already succeeded and queues
    # a real API call for only the flagged field(s) -- never the whole
    # user. Read-only on every source directory; this run's own ledger is
    # the only thing written to.
    casewise_carry_provenance = _seed_casewise_carry_forward_into_ledger(
        ledger=ledger,
        all_user_attempts=all_user_attempts,
        casewise_repair_plan=casewise_repair_plan,
    )

    recovered_successful_bundles = _recover_casewise_bundles_from_ledger(
        ledger=ledger,
        all_user_attempts=all_user_attempts,
        existing=existing,
        work_path=work_path,
        generation_binding=generation_binding,
        family_by_user=family_by_user,
        seeds=seeds,
        endpoint=endpoint,
        casewise_carry_provenance=casewise_carry_provenance,
    )
    for user_id, bundle in existing.items():
        require_bundle_generation_binding(bundle, generation_binding)
        strict_bundle_check(bundle, family_by_user[user_id])
        carried_from = bundle.provenance.get("carried_forward_from")
        if carried_from is not None:
            # This user's "proof of payment" lives in one or more DIFFERENT
            # runs' ledgers (already verified once at carry-forward time);
            # re-check that every source ledger this bundle actually drew
            # from has not changed or vanished since, rather than requiring
            # it to appear in this run's own (mostly-empty) ledger, which
            # does not apply to carried-forward users.
            for (
                dir_str,
                expected_hash,
            ) in carried_from["source_physical_attempt_ledger_sha256_by_directory"].items():
                source_dir = Path(dir_str)
                source_ledger = source_dir / "_generation_physical_attempt_ledger.jsonl"
                if not source_ledger.is_file() or sha256_file(source_ledger) != str(
                    expected_hash
                ):
                    raise RuntimeError(
                        f"carried-forward bundle {user_id} source ledger changed or "
                        f"is missing since carry-forward: {source_dir}"
                    )
            continue
        accepted_call_keys = bundle.provenance.get(
            "accepted_surface_call_keys"
        )
        valid_call_keys = {
            str(attempt["call_key"])
            for case_plan in all_user_attempts[user_id]["cases"]
            for attempt in case_plan["attempts"]
        }
        if (
            not isinstance(accepted_call_keys, dict)
            or set(accepted_call_keys)
            != {field for field, _ in GENERATION_CASE_FIELDS}
            or any(
                str(value) not in valid_call_keys
                or not ledger.succeeded(str(value))
                for value in accepted_call_keys.values()
            )
        ):
            raise RuntimeError(
                f"resumable bundle lacks nine successful surface bindings: {user_id}"
            )

    pending_users = [user_id for user_id in planned_users if user_id not in existing]
    call_plan: list[dict[str, Any]] = []
    blocked_pending_users: list[dict[str, Any]] = []
    for user_id in pending_users:
        user_plan = all_user_attempts[user_id]
        for case_plan in user_plan["cases"]:
            successful = [
                attempt
                for attempt in case_plan["attempts"]
                if ledger.succeeded(str(attempt["call_key"]))
            ]
            if successful:
                continue
            remaining_attempts = []
            blocked_attempt_reasons: list[str] = []
            for attempt in case_plan["attempts"]:
                call_key = str(attempt["call_key"])
                if _persisted_content_attempt_failed(ledger, call_key):
                    continue
                blocker = call_retry_blocker(
                    ledger,
                    call_key,
                    max_provider_output_attempts=1,
                )
                if blocker is None:
                    remaining_attempts.append(attempt)
                elif ledger.attempts_for(call_key):
                    blocked_attempt_reasons.append(
                        f"{attempt['attempt_kind']}: {blocker}"
                    )
            if not remaining_attempts:
                blocked_pending_users.append(
                    {
                        "user_id": user_id,
                        "case_field": case_plan["case_field"],
                        "reason": (
                            "persisted_noncontent_retry_blocker"
                            if blocked_attempt_reasons
                            else "maximum_case_attempts_exhausted"
                        ),
                        "blockers": blocked_attempt_reasons,
                        "historical_attempts": sum(
                            ledger.attempts_for(str(attempt["call_key"]))
                            for attempt in case_plan["attempts"]
                        ),
                    }
                )
                continue
            public_attempts = [
                {
                    key: value
                    for key, value in attempt.items()
                    if key != "messages"
                }
                for attempt in remaining_attempts
            ]
            for public_attempt in public_attempts:
                call_key = str(public_attempt["call_key"])
                public_attempt["maximum_physical_attempts"] = (
                    GENERATION_PILOT_MAX_TRANSPORT_ATTEMPTS_PER_CONTENT_ATTEMPT
                )
                public_attempt["historical_physical_attempts"] = (
                    ledger.attempts_for(call_key)
                )
                public_attempt["remaining_physical_attempts"] = (
                    GENERATION_PILOT_MAX_TRANSPORT_ATTEMPTS_PER_CONTENT_ATTEMPT
                    - ledger.attempts_for(call_key)
                )
            call_plan.append(
                {
                    "user_id": user_id,
                    "split": user_plan["split"],
                    "semantic_families": user_plan["semantic_families"],
                    "seed_dialogue_index": user_plan["seed_dialogue_index"],
                    "seed_dialogue_source_id": user_plan[
                        "seed_dialogue_source_id"
                    ],
                    "seed_dialogue_sha256": user_plan["seed_dialogue_sha256"],
                    "case_field": case_plan["case_field"],
                    "regime": case_plan["regime"],
                    "semantic_family": case_plan["semantic_family"],
                    "estimated_input_tokens": public_attempts[0][
                        "estimated_input_tokens"
                    ],
                    "maximum_estimated_input_tokens": max(
                        int(attempt["estimated_input_tokens"])
                        for attempt in public_attempts
                    ),
                    "remaining_attempts": len(public_attempts),
                    "remaining_physical_api_attempts": sum(
                        int(attempt["remaining_physical_attempts"])
                        for attempt in public_attempts
                    ),
                    "attempts": public_attempts,
                }
            )

    input_tokens = [int(row["estimated_input_tokens"]) for row in call_plan]
    expected_input_tokens = sum(input_tokens)
    potential_input_tokens = [
        int(attempt["estimated_input_tokens"])
        for row in call_plan
        for attempt in row["attempts"]
        for _ in range(int(attempt["remaining_physical_attempts"]))
    ]
    maximum_input_tokens = sum(potential_input_tokens)
    expected_output_tokens = (
        len(call_plan) * SURFACE_GENERATION_MAX_OUTPUT_TOKENS
    )
    maximum_output_tokens = (
        len(potential_input_tokens) * SURFACE_GENERATION_MAX_OUTPUT_TOKENS
    )
    expected_cost = (
        expected_input_tokens / 1_000_000 * args.input_usd_per_mtok
        + expected_output_tokens / 1_000_000 * args.output_usd_per_mtok
    )
    maximum_cost = (
        maximum_input_tokens / 1_000_000 * args.input_usd_per_mtok
        + maximum_output_tokens / 1_000_000 * args.output_usd_per_mtok
    )
    completed_bundle_hashes = {
        user_id: sha256_text(canonical_json(existing[user_id].model_dump(mode="json")))
        for user_id in sorted(existing)
    }
    estimate: dict[str, Any] = {
        "protocol": GENERATION_COST_PROTOCOL,
        "input_token_safety_factor": input_token_safety_factor,
        "fail_on_reported_input_overrun": fail_on_reported_input_overrun,
        "generation_run_binding_sha256": generation_binding_sha256,
        "generation_run_binding": generation_binding,
        "carry_forward": carry_forward_report,
        "completed_bundle_hashes": completed_bundle_hashes,
        "completed_bundle_manifest_sha256": sha256_text(
            canonical_json(completed_bundle_hashes)
        ),
        "planned_users": len(planned_users),
        "completed_users": len(existing),
        "pending_users": len(pending_users),
        "blocked_pending_users": blocked_pending_users,
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        "physical_attempt_ledger_sha256": (
            sha256_file(attempt_ledger_path)
            if attempt_ledger_path.exists()
            else sha256_text("")
        ),
        "historical_physical_api_attempts": _real_physical_attempt_count(ledger),
        "historical_carried_forward_ledger_entries": (
            _carried_forward_ledger_entry_count(ledger)
        ),
        "recovered_successful_bundles_from_ledger": recovered_successful_bundles,
        # Exact one-attempt success-path count plus an accepted hard upper bound.
        "expected_api_calls": len(call_plan),
        "maximum_api_calls": len(potential_input_tokens),
        "maximum_new_physical_api_attempts": len(potential_input_tokens),
        "maximum_physical_api_attempts_including_history": (
            _real_physical_attempt_count(ledger) + len(potential_input_tokens)
        ),
        "expected_total_input_tokens": expected_input_tokens,
        "maximum_total_input_tokens": maximum_input_tokens,
        "expected_maximum_output_tokens": expected_output_tokens,
        "maximum_total_output_tokens": maximum_output_tokens,
        "mean_input_tokens": (
            float(expected_input_tokens / len(input_tokens)) if input_tokens else 0.0
        ),
        "max_input_tokens": max(potential_input_tokens, default=0),
        "pricing": {
            "input_usd_per_mtok": args.input_usd_per_mtok,
            "output_usd_per_mtok": args.output_usd_per_mtok,
        },
        "expected_estimated_cost_usd": expected_cost,
        "maximum_estimated_cost_usd": maximum_cost,
        "call_plan_sha256": sha256_text(canonical_json(call_plan)),
    }
    estimate["budget_gate"] = _budget_gate(
        estimate,
        max_api_calls=args.max_api_calls,
        max_estimated_usd=args.max_estimated_usd,
        max_input_tokens_per_call=args.max_input_tokens_per_call,
    )
    estimate["cost_estimate_sha256"] = sha256_text(canonical_json(estimate))

    if args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.out_dir / "generation_cost_estimate.json", estimate)
        write_jsonl(args.out_dir / "generation_call_plan.jsonl", call_plan)
        print({**estimate, "status": "DRY_RUN"})
        if estimate["budget_gate"]["status"] != "PASS":
            raise RuntimeError("generation dry-run failed the frozen budget gate")
        return

    if estimate["budget_gate"]["status"] != "PASS":
        raise RuntimeError("generation API run blocked by the budget gate")
    _require_saved_dry_run(args.out_dir, estimate)
    expected_hash = str(estimate["cost_estimate_sha256"])
    if not args.accept_cost_estimate_sha256:
        raise RuntimeError(
            "API mode is fail-closed: pass --accept-cost-estimate-sha256 "
            f"{expected_hash} from the matching dry-run"
        )
    if args.accept_cost_estimate_sha256 != expected_hash:
        raise RuntimeError(
            "accepted cost estimate hash does not match the current generation plan"
        )

    if (
        args.automated_semantic_review_report is not None
        or args.automated_semantic_review_attestation is not None
    ):
        raise RuntimeError(
            "formal generation refuses the consumed V4 semantic-review artifacts; "
            "run actual-468 structured QA v3 after generation and before sweep"
        )
    generation_pilot_semantic_verification = {
        "protocol": "pm-v1.5-post-generation-actual-structured-qa-required-v1",
        "status": "DEFERRED_UNTIL_ACTUAL_468_EXISTS",
        "human_calibration_performed": False,
        "historical_v4_status": "CONSUMED_FAILED_CLOSED_CALIBRATION_ONLY",
        "historical_v4_2_status": "CONSUMED_INSTRUMENT_NOT_READY_CALIBRATION_ONLY",
        "required_next_gate": "pm-v1.5-actual-468-structured-qa-v3",
        "required_before": "7488_action_sweep",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Resolve credentials before reserving the first paid attempt.  Every
    # reservation after this point is fsynced before its HTTP request.
    if call_plan:
        _ = endpoint.api_key
    authorized_call_keys = {
        str(attempt["call_key"])
        for row in call_plan
        for attempt in row["attempts"]
    }
    historical_api_calls = _real_physical_attempt_count(ledger)
    api_calls_used = 0
    client = make_client(endpoint) if call_plan else None
    # Immediate, cheap (exact-normalized-text) cross-user duplicate check,
    # updated as each case is accepted. validate_split_manifests still runs
    # the authoritative final check at whole-corpus assembly, but that only
    # happens after every pending user's cases are generated and the
    # expensive semantic near-duplicate audit runs; catching a new
    # cross-user collision right after the response that caused it is
    # generated fails closed immediately, before paying for -- or waiting
    # on -- the rest of the pipeline.
    regime_to_case_field_for_dedup = {
        regime: field for field, regime in GENERATION_CASE_FIELDS
    }
    accepted_normalized_texts: dict[str, tuple[str, str]] = {
        normalize_text(case.current_user_text): (
            existing_user_id,
            regime_to_case_field_for_dedup[case.regime],
        )
        for existing_user_id, existing_bundle in existing.items()
        for case in existing_bundle.cases
    }
    try:
        for user_id in planned_users:
            if user_id in existing:
                continue
            user_plan = all_user_attempts[user_id]
            prior_current_user_texts: list[str] = []
            prior_current_user_families: list[str] = []
            for case_plan in user_plan["cases"]:
                loaded = _load_successful_surface_attempt(
                    ledger=ledger, case_plan=case_plan
                )
                if loaded is None:
                    last_error: str | None = None
                    for attempt in case_plan["attempts"]:
                        call_key = str(attempt["call_key"])
                        if call_key not in authorized_call_keys:
                            continue
                        if _real_physical_attempt_count(ledger) >= args.max_api_calls:
                            raise RuntimeError(
                                "generation API call cap exhausted before completion"
                            )
                        record_ids = {
                            "user_id": user_id,
                            "case_field": case_plan["case_field"],
                            "attempt_kind": attempt["attempt_kind"],
                            "generation_seed": int(attempt["seed"]),
                        }
                        reservation = None
                        call = None
                        try:
                            assert client is not None
                            reservation, call, surface = execute_with_bounded_retry(
                                ledger,
                                call_key,
                                record_ids=record_ids,
                                prompt_sha256=str(attempt["prompt_sha256"]),
                                call_fn=lambda attempt=attempt: client.chat(
                                    attempt["messages"],
                                    temperature=GENERATION_TEMPERATURE,
                                    max_tokens=(
                                        SURFACE_GENERATION_MAX_OUTPUT_TOKENS
                                    ),
                                    seed=int(attempt["seed"]),
                                    response_schema=GeneratedSurfaceOnlyCaseDraft,
                                    retries=GENERATION_REQUEST_RETRIES,
                                ),
                                max_provider_output_attempts=1,
                                backoff_seconds=(
                                    GENERATION_PILOT_TRANSPORT_BACKOFF_SECONDS
                                ),
                                sleep=time.sleep,
                                capture_provider_output_text=True,
                            )
                            assert surface is not None
                            usage = require_reported_usage(
                                call.usage,
                                stage="PM-v1.5 full surface generation",
                            )
                            if usage["prompt_tokens"] > int(
                                attempt["estimated_input_tokens"]
                            ):
                                result = {
                                    "surface": surface.model_dump(mode="json"),
                                    "provider_response": call.raw_response,
                                }
                                last_error = (
                                    "reported prompt_tokens exceed the frozen "
                                    "conservative bound"
                                )
                                ledger.finish(
                                    reservation,
                                    succeeded=False,
                                    request_hash=call.request_hash,
                                    usage=usage,
                                    error=last_error,
                                    result=result,
                                    metadata=failure_metadata(
                                        retry_class="input_token_bound_overrun",
                                        retry_disposition=TERMINAL_DISPOSITION,
                                    ),
                                )
                                raise ReportedInputTokenOverrun(last_error)
                            lint = lint_generation_surface_case(
                                user_id=str(user_plan["user_id"]),
                                case_field=str(case_plan["case_field"]),
                                regime=ResourceNeedRegime(
                                    str(case_plan["regime"])
                                ),
                                family=str(case_plan["semantic_family"]),
                                forbidden_families=[
                                    family
                                    for family in user_plan["semantic_families"]
                                    if family != case_plan["semantic_family"]
                                ],
                                surface=surface,
                                prior_current_user_texts=(
                                    prior_current_user_texts
                                ),
                                prior_current_user_families=(
                                    prior_current_user_families
                                ),
                            )
                            result = {
                                "surface": surface.model_dump(mode="json"),
                                "provider_response": call.raw_response,
                                "lint": lint,
                                "attempt_kind": attempt["attempt_kind"],
                            }
                            if lint["status"] != "PASS":
                                last_error = "surface lint failed: " + canonical_json(
                                    lint["errors"]
                                )
                                ledger.finish(
                                    reservation,
                                    succeeded=False,
                                    request_hash=call.request_hash,
                                    usage=usage,
                                    error=last_error,
                                    result=result,
                                    metadata=failure_metadata(
                                        retry_class="content_lint_failure",
                                        retry_disposition=TERMINAL_DISPOSITION,
                                    ),
                                )
                                append_jsonl(
                                    error_path,
                                    {
                                        "user_id": user_id,
                                        "case_field": case_plan["case_field"],
                                        "attempt_kind": attempt["attempt_kind"],
                                        "generation_seed": int(attempt["seed"]),
                                        "physical_call_key": call_key,
                                        "error": last_error,
                                        "provider_response_preserved": True,
                                        "accepted_cost_estimate_sha256": expected_hash,
                                    },
                                )
                                continue
                            normalized_text = normalize_text(surface.current_user_text)
                            cross_user_duplicate = _find_cross_user_duplicate(
                                normalized_text=normalized_text,
                                user_id=user_id,
                                accepted_normalized_texts=accepted_normalized_texts,
                            )
                            if cross_user_duplicate is not None:
                                last_error = (
                                    "cross-user exact-duplicate current_user_text "
                                    "detected immediately after generation (before "
                                    "final corpus assembly): collides with "
                                    f"{cross_user_duplicate[0]}/{cross_user_duplicate[1]}"
                                )
                                ledger.finish(
                                    reservation,
                                    succeeded=False,
                                    request_hash=call.request_hash,
                                    usage=usage,
                                    error=last_error,
                                    result=result,
                                    metadata=failure_metadata(
                                        retry_class="cross_user_duplicate_detected_immediately",
                                        retry_disposition=TERMINAL_DISPOSITION,
                                    ),
                                )
                                append_jsonl(
                                    error_path,
                                    {
                                        "user_id": user_id,
                                        "case_field": case_plan["case_field"],
                                        "attempt_kind": attempt["attempt_kind"],
                                        "generation_seed": int(attempt["seed"]),
                                        "physical_call_key": call_key,
                                        "error": last_error,
                                        "provider_response_preserved": True,
                                        "accepted_cost_estimate_sha256": expected_hash,
                                    },
                                )
                                continue
                            ledger.finish(
                                reservation,
                                succeeded=True,
                                request_hash=call.request_hash,
                                usage=usage,
                                error=None,
                                result=result,
                            )
                            accepted_normalized_texts[normalized_text] = (
                                user_id,
                                str(case_plan["case_field"]),
                            )
                            loaded = (attempt, surface, call)
                            last_error = None
                            break
                        except StructuredOutputValidationError as exc:
                            last_error = f"{type(exc).__name__}: {exc}"
                            append_jsonl(
                                error_path,
                                {
                                    "user_id": user_id,
                                    "case_field": case_plan["case_field"],
                                    "attempt_kind": attempt["attempt_kind"],
                                    "generation_seed": int(attempt["seed"]),
                                    "physical_call_key": call_key,
                                    "error": last_error,
                                    "provider_response_preserved": True,
                                    "validation_errors": exc.validation_errors,
                                    "accepted_cost_estimate_sha256": expected_hash,
                                },
                            )
                            continue
                        except RetryableProviderError as exc:
                            last_error = f"{type(exc).__name__}: {exc}"
                            if (
                                exc.last_retry_class
                                in BOUNDED_PROVIDER_OUTPUT_RETRY_CLASSES
                            ):
                                append_jsonl(
                                    error_path,
                                    {
                                        "user_id": user_id,
                                        "case_field": case_plan["case_field"],
                                        "attempt_kind": attempt["attempt_kind"],
                                        "generation_seed": int(attempt["seed"]),
                                        "physical_call_key": call_key,
                                        "error": last_error,
                                        "provider_response_preserved": True,
                                        "accepted_cost_estimate_sha256": (
                                            expected_hash
                                        ),
                                    },
                                )
                                continue
                            raise
                        except ProviderRequestError:
                            raise
                        except ReportedInputTokenOverrun:
                            raise
                        except Exception as exc:
                            last_error = f"{type(exc).__name__}: {exc}"
                            if (
                                reservation is not None
                                and ledger.terminal_event(
                                    call_key, reservation.attempt_index
                                )
                                is None
                            ):
                                ledger.finish(
                                    reservation,
                                    succeeded=False,
                                    request_hash=getattr(call, "request_hash", None),
                                    usage=getattr(call, "usage", None),
                                    error=last_error,
                                    metadata=failure_metadata(
                                        retry_class="local_postcondition_error",
                                        retry_disposition=TERMINAL_DISPOSITION,
                                    ),
                                )
                            raise RuntimeError(last_error) from exc
                    if loaded is None:
                        raise RuntimeError(
                            "surface generation exhausted initial+repair for "
                            f"{user_id}/{case_plan['case_field']}: {last_error}"
                        )
                _, surface, _ = loaded
                prior_current_user_texts.append(surface.current_user_text)
                prior_current_user_families.append(
                    str(case_plan["semantic_family"])
                )
                # Covers the ledger-recovered path too (a case already
                # accepted in an EARLIER invocation of this same run):
                # harmless to repeat for the freshly-generated path, which
                # already registered the identical key/value above.
                accepted_normalized_texts[
                    normalize_text(surface.current_user_text)
                ] = (user_id, str(case_plan["case_field"]))
            seed_record = seeds[int(user_plan["seed_dialogue_index"])]
            bundle = _compile_casewise_user_from_ledger(
                ledger=ledger,
                user_plan=user_plan,
                seed_dialogue=seed_record["dialogue_text"],
                seed_dialogue_source_id=seed_record["dialogue_id"],
                endpoint=endpoint,
                generation_binding=generation_binding,
                casewise_carried_forward=casewise_carry_provenance.get(user_id),
            )
            if bundle is None:
                raise RuntimeError(f"user {user_id} lacks nine accepted surfaces")
            strict_bundle_check(bundle, family_by_user[user_id])
            append_jsonl(work_path, bundle.model_dump(mode="json"))
            existing[user_id] = bundle
    finally:
        if client is not None:
            client.close()
    api_calls_used = _real_physical_attempt_count(ledger) - historical_api_calls
    bundles = [existing[user_id] for user_id in planned_users]
    report = write_development_dataset(
        bundles=bundles,
        split_by_user=split_by_user,
        out_dir=args.out_dir,
        strategy_catalog_count=strategy_catalog_count,
        strategy_estimated_tokens=strategy_estimated_tokens,
        strategy_top_k=strategy_top_k,
        strategy_bank_sha256=strategy_bank_sha256,
        strategy_cards=strategy_cards,
        enforce_required_hit_preflight=True,
        memory_min_score=float(retrieval_cfg["memory_min_score"]),
        strategy_min_score=float(retrieval_cfg["strategy_min_score"]),
        expected_semantic_families_by_split={
            PMV2Split.TRAIN: TRAIN_SEMANTIC_FAMILIES,
            PMV2Split.CALIBRATION: CALIBRATION_SEMANTIC_FAMILIES,
            PMV2Split.INTERNAL_TEST: INTERNAL_TEST_SEMANTIC_FAMILIES,
        },
        semantic_encoder=semantic_encoder,
    )
    enforce_full_state_design(
        report,
        train_users=args.train_users,
        calibration_users=args.calibration_users,
        internal_test_users=args.internal_test_users,
    )
    generated_states = load_states(args.out_dir / "pm_v2_states.jsonl")
    states_by_split = {
        split: [state for state in generated_states if state.split == split]
        for split in (
            PMV2Split.TRAIN,
            PMV2Split.CALIBRATION,
            PMV2Split.INTERNAL_TEST,
        )
    }
    near_duplicate_cfg = pm_config["splits"]["near_duplicate_audit"]
    if near_duplicate_cfg.get("method") != (
        "fixed_hash_word_1_2_and_char_3_5_cosine"
    ):
        raise ValueError("unsupported PM-v2 near-duplicate audit method")
    near_duplicate_audit = audit_cross_split_near_duplicates(
        states_by_split,
        maximum_word_hash_cosine=float(
            near_duplicate_cfg["maximum_word_hash_cosine"]
        ),
        maximum_char_hash_cosine=float(
            near_duplicate_cfg["maximum_char_hash_cosine"]
        ),
        report_top_pairs=int(near_duplicate_cfg["report_top_pairs"]),
    )
    generation_repair_cases = [
        {"user_id": bundle.user_id, "case_field": case_field}
        for bundle in bundles
        for case_field, attempt_kind in (
            bundle.provenance.get("provider_surface_attempt_kinds") or {}
        ).items()
        if attempt_kind == "repair"
    ]
    report.update(
        {
            "work_path": str(work_path),
            "error_path": str(error_path),
            "pm_v2_config": str(args.pm_v2_config),
            "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
            "seed_dialogues": str(args.seed_dialogues),
            "seed_dialogues_sha256": sha256_file(args.seed_dialogues),
            "frozen_strategy_bank_binding": frozen_strategy_bank_binding,
            "seed_strategy_instance_disjointness": seed_strategy_disjointness,
            "generation_run_binding": generation_binding,
            "generation_run_binding_sha256": generation_binding_sha256,
            "semantic_runtime": semantic_runtime_verification,
            "readiness_natural_language_challenge": readiness_challenge,
            "accepted_cost_estimate_sha256": expected_hash,
            "generation_api_calls_used": api_calls_used,
            "generation_historical_api_calls": historical_api_calls,
            "generation_total_physical_api_attempts": _real_physical_attempt_count(
                ledger
            ),
            "generation_total_carried_forward_ledger_entries": (
                _carried_forward_ledger_entry_count(ledger)
            ),
            "generation_total_ledger_entries": ledger.started_attempts,
            "generation_transport_retry_summary": retry_ledger_summary(
                ledger, list(ledger.expected_calls)
            ),
            "generation_attempt_ledger_path": str(attempt_ledger_path),
            "generation_attempt_ledger_sha256": sha256_file(attempt_ledger_path),
            "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
            "generation_compatibility_pilot": generation_pilot_verification,
            "generation_pilot_semantic_review": (
                generation_pilot_semantic_verification
            ),
            "generation_expected_api_calls": estimate["expected_api_calls"],
            "generation_maximum_api_calls": estimate["maximum_api_calls"],
            "generation_surface_protocol": "surface_only_casewise",
            "generation_repair_case_count": len(generation_repair_cases),
            "generation_repair_case_rate": (
                len(generation_repair_cases) / len(generated_states)
                if generated_states
                else 0.0
            ),
            "generation_repair_cases": generation_repair_cases,
            "cross_split_near_duplicate_audit": near_duplicate_audit,
            "all_states_have_16_actions": all(
                len(state.allowed_actions) == 16 for state in generated_states
            ),
        }
    )
    if not report["all_states_have_16_actions"]:
        raise RuntimeError("generated PM-v2 dataset does not expose all 16 actions")
    write_json(args.out_dir / "pm_v2_data_report.json", report)
    create_artifact_attestation(
        args.out_dir / "artifact_attestation.json",
        stage="pm_v1_5_development_data",
        inputs={
            "experiment_config": args.config,
            "pm_v1_5_config": args.pm_v2_config,
            "seed_dialogues": args.seed_dialogues,
            "selected_seed_sources": args.selected_seed_sources,
            "strategy_bank": args.strategy_bank,
            "generation_pilot_attestation": args.generation_pilot_attestation,
            "cost_estimate": args.out_dir / "generation_cost_estimate.json",
            "call_plan": args.out_dir / "generation_call_plan.jsonl",
        },
        outputs={
            "states": (args.out_dir / "pm_v2_states.jsonl", True),
            "runtime": (args.out_dir / "runtime_states.jsonl", True),
            "backend": (args.out_dir / "memory_backend.jsonl", True),
            "evaluator_contexts": (
                args.out_dir / "evaluator_contexts.jsonl",
                True,
            ),
            "data_report": (args.out_dir / "pm_v2_data_report.json", False),
            "physical_attempt_ledger": (attempt_ledger_path, True),
        },
        parameters={
            "track": "pm-v1.5",
            "generation_run_binding_sha256": generation_binding_sha256,
            "accepted_cost_estimate_sha256": expected_hash,
            "human_semantic_review_performed": False,
        },
        expected={
            "states": len(generated_states),
            "users": len(planned_users),
            "physical_attempts": ledger.started_attempts,
        },
    )
    print(report)


if __name__ == "__main__":
    main()
