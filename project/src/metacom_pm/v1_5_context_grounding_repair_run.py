"""Content-bound planning and materialization for context-grounding repairs.

The six-state pilot certifies only the provider-facing repair prompt/schema and
the per-record postconditions.  It is deliberately unable to publish a repaired
development corpus.  The full scope is the only scope that may materialize the
exact-25 repair overlay consumed by the canonical recompilation path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .api import Endpoint, chat_request_payload, endpoint_transport
from .attempt_ledger import physical_call_key
from .io import canonical_json, sha256_file, sha256_text
from .text import conservative_token_bound, estimate_tokens, normalize_space, tokens
from .v1_5_context_grounding_data_repair import (
    FIELD_ONLY_REPAIR_SUMMARY_ALSO_NEEDED,
    FieldOnlyRepairOutput,
    VISIBLE_SURFACE_REPAIR_TURN_INDICES,
    VisibleSurfaceRepairOutput,
    maximum_legal_field_only_repair_output_tokens,
    maximum_legal_visible_surface_repair_output_tokens,
    repair_call_plan_row,
)
from .v1_5_context_grounding_repair_overlay import build_repair_overlay_record


REPAIR_RUN_PROTOCOL = "pm-v1.5-context-grounding-repair-run-v1"
REPAIR_PILOT_STAGE = "context_grounding_repair_pilot"
REPAIR_FULL_STAGE = "context_grounding_repair_full"
REPAIR_TEMPERATURE = 0.0
REPAIR_BASE_SEED = 913_700
REPAIR_MAX_PHYSICAL_ATTEMPTS_PER_CALL = 3
REPAIR_BACKOFF_SECONDS = (10.0, 30.0)

# Frozen before looking at repair outcomes. Covers: context-only with a clean
# summary that must remain byte-identical; context+summary; visible referent
# repair with present/absent summaries; multi-turn pronoun cascade; temporal
# contradiction. Selection is not an outcome sample and cannot be changed by
# a CLI flag.
REPAIR_PILOT_STATE_IDS = (
    "state_062b3dd61638973f95199d76",
    "state_1d327ed7b58f3b13298b968c",
    "state_481fdb962d3774c3e0d720bb",
    "state_44550214bf7c9fa22284a731",
    "state_d0eee004b44a562c57a0e6c6",
    "state_ba1e526156d1cf3810fa8e98",
)


def repair_stage(scope: str) -> str:
    if scope == "pilot":
        return REPAIR_PILOT_STAGE
    if scope == "full":
        return REPAIR_FULL_STAGE
    raise ValueError(f"unsupported repair scope: {scope}")


def select_repair_records(records: Sequence[Any], *, scope: str) -> list[Any]:
    defects = [record for record in records if record.classification == "DATA_DEFECT"]
    by_state = {record.state_id: record for record in defects}
    if len(by_state) != len(defects) or len(defects) != 25:
        raise RuntimeError("repair planning requires exactly 25 unique DATA_DEFECT records")
    if scope == "full":
        return sorted(defects, key=lambda record: record.state_id)
    if scope != "pilot":
        raise ValueError(f"unsupported repair scope: {scope}")
    missing = set(REPAIR_PILOT_STATE_IDS) - set(by_state)
    if missing:
        raise RuntimeError(f"frozen repair pilot state_ids are missing: {sorted(missing)}")
    return [by_state[state_id] for state_id in REPAIR_PILOT_STATE_IDS]


def response_schema_for_mode(repair_mode: str):
    if repair_mode == "FIELD_ONLY_REPAIR":
        return FieldOnlyRepairOutput
    if repair_mode == "VISIBLE_SURFACE_REPAIR":
        return VisibleSurfaceRepairOutput
    raise RuntimeError(f"unsupported repair_mode: {repair_mode}")


def maximum_output_tokens_for_mode(repair_mode: str) -> int:
    if repair_mode == "FIELD_ONLY_REPAIR":
        return maximum_legal_field_only_repair_output_tokens()
    if repair_mode == "VISIBLE_SURFACE_REPAIR":
        return maximum_legal_visible_surface_repair_output_tokens()
    raise RuntimeError(f"unsupported repair_mode: {repair_mode}")


def build_repair_run_plan(
    *,
    records: Sequence[Any],
    scope: str,
    endpoint: Endpoint,
    original_bundles_path: str | Path,
    classification_path: str | Path,
    experiment_config_path: str | Path,
    pm_v1_5_config_path: str | Path,
    input_token_safety_factor: float,
    input_usd_per_mtok: float,
    output_usd_per_mtok: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, str]]]]:
    selected = select_repair_records(records, scope=scope)
    stage = repair_stage(scope)
    rows: list[dict[str, Any]] = []
    messages_by_call: dict[str, list[dict[str, str]]] = {}
    for sequence_index, record in enumerate(selected, 1):
        base = repair_call_plan_row(record)
        messages = list(base["messages"])
        schema = response_schema_for_mode(record.repair_mode)
        maximum_output_tokens = maximum_output_tokens_for_mode(record.repair_mode)
        seed = REPAIR_BASE_SEED + sequence_index
        request_payload = chat_request_payload(
            endpoint,
            messages,
            temperature=REPAIR_TEMPERATURE,
            max_tokens=maximum_output_tokens,
            seed=seed,
            response_schema=schema,
        )
        request_json = canonical_json(request_payload)
        prompt_sha256 = sha256_text(canonical_json(messages))
        request_payload_sha256 = sha256_text(request_json)
        record_ids = {
            "state_id": record.state_id,
            "user_id": record.user_id,
            "repair_mode": record.repair_mode,
            "scope": scope,
        }
        call_key = physical_call_key(
            stage=stage,
            record_ids=record_ids,
            prompt_sha256=prompt_sha256,
            endpoint=endpoint,
            request_parameters={
                "temperature": REPAIR_TEMPERATURE,
                "max_tokens": maximum_output_tokens,
                "seed": seed,
                "response_schema": schema.__name__,
                "request_payload_sha256": request_payload_sha256,
            },
        )
        row = {
            **record_ids,
            "sequence_index": sequence_index,
            "defect_type": record.defect_type,
            "repair_summary": bool(base["repair_summary"]),
            "response_schema": schema.__name__,
            "seed": seed,
            "prompt_sha256": prompt_sha256,
            "request_payload_sha256": request_payload_sha256,
            "raw_estimated_input_tokens": estimate_tokens(request_json),
            "input_token_upper_bound": conservative_token_bound(
                request_json, safety_factor=input_token_safety_factor
            ),
            "maximum_output_tokens": maximum_output_tokens,
            "maximum_physical_attempts": REPAIR_MAX_PHYSICAL_ATTEMPTS_PER_CALL,
            "physical_call_key": call_key,
        }
        rows.append(row)
        messages_by_call[call_key] = messages

    project_root = Path(pm_v1_5_config_path).resolve().parent.parent
    shared_code_paths = {
        "api": project_root / "src" / "metacom_pm" / "api.py",
        "attempt_ledger": project_root / "src" / "metacom_pm" / "attempt_ledger.py",
        "bounded_retry": project_root / "src" / "metacom_pm" / "bounded_retry.py",
        "repair_classification": project_root
        / "src"
        / "metacom_pm"
        / "v1_5_context_grounding_repair.py",
        "repair_generation": project_root
        / "src"
        / "metacom_pm"
        / "v1_5_context_grounding_data_repair.py",
        "repair_overlay": project_root
        / "src"
        / "metacom_pm"
        / "v1_5_context_grounding_repair_overlay.py",
        "repair_run": Path(__file__).resolve(),
        "repair_runner": project_root
        / "scripts"
        / "v1_5"
        / "20e_run_context_grounding_repair_v1_5.py",
    }
    shared_code_manifest = {
        name: {
            "relative_path": str(path.resolve().relative_to(project_root)),
            "sha256": sha256_file(path),
        }
        for name, path in sorted(shared_code_paths.items())
    }
    def relative_input(path: str | Path) -> str:
        resolved = Path(path).resolve()
        try:
            return str(resolved.relative_to(project_root))
        except ValueError:
            return str(resolved)

    contract = {
        "protocol": REPAIR_RUN_PROTOCOL,
        "scope": scope,
        "stage": stage,
        "selected_state_ids": [row["state_id"] for row in rows],
        "pilot_state_ids": list(REPAIR_PILOT_STATE_IDS),
        "original_bundles_path": relative_input(original_bundles_path),
        "original_bundles_sha256": sha256_file(original_bundles_path),
        "classification_path": relative_input(classification_path),
        "classification_sha256": sha256_file(classification_path),
        "experiment_config_sha256": sha256_file(experiment_config_path),
        "pm_v1_5_config_sha256": sha256_file(pm_v1_5_config_path),
        "endpoint": {
            "base_url": endpoint.base_url,
            "model": endpoint.model,
            "family": endpoint.family,
            "transport": endpoint_transport(endpoint),
            "supports_strict_json_schema": endpoint.supports_strict_json_schema,
            "thinking_mode": endpoint.thinking_mode,
        },
        "temperature": REPAIR_TEMPERATURE,
        "base_seed": REPAIR_BASE_SEED,
        "maximum_physical_attempts_per_call": REPAIR_MAX_PHYSICAL_ATTEMPTS_PER_CALL,
        "input_token_safety_factor": input_token_safety_factor,
        "pricing_usd_per_mtok": {
            "input": input_usd_per_mtok,
            "output": output_usd_per_mtok,
        },
        "schema_sha256s": {
            FieldOnlyRepairOutput.__name__: sha256_text(
                canonical_json(FieldOnlyRepairOutput.model_json_schema())
            ),
            VisibleSurfaceRepairOutput.__name__: sha256_text(
                canonical_json(VisibleSurfaceRepairOutput.model_json_schema())
            ),
        },
        "shared_code_manifest": shared_code_manifest,
        "shared_code_manifest_sha256": sha256_text(
            canonical_json(shared_code_manifest)
        ),
        "call_plan_sha256": sha256_text(canonical_json(rows)),
    }
    contract["contract_sha256"] = sha256_text(canonical_json(contract))

    # Sum exact integer token counts first, then convert to dollars in a single
    # fixed float expression (matching plan_action_sweep's own convention in
    # sweep.py). Summing many already-converted float dollar terms is not
    # reproducible across Python versions: CPython 3.12 changed the built-in
    # sum() to use Neumaier compensated summation for floats, so the same
    # per-row terms in the same order can round to a different last bit on
    # 3.10 (this project's mandated distress_build env) vs 3.12+, which would
    # silently change cost_estimate_sha256 depending on interpreter version.
    total_input_token_upper_bound = sum(
        int(row["input_token_upper_bound"]) for row in rows
    )
    total_maximum_output_tokens = sum(
        int(row["maximum_output_tokens"]) for row in rows
    )
    one_attempt_cost = (
        total_input_token_upper_bound / 1_000_000 * input_usd_per_mtok
        + total_maximum_output_tokens / 1_000_000 * output_usd_per_mtok
    )
    maximum_cost = one_attempt_cost * REPAIR_MAX_PHYSICAL_ATTEMPTS_PER_CALL
    cost_payload = {
        "protocol": REPAIR_RUN_PROTOCOL,
        "scope": scope,
        "stage": stage,
        "contract_sha256": contract["contract_sha256"],
        "call_plan_sha256": contract["call_plan_sha256"],
        "minimum_logical_calls": len(rows),
        "maximum_physical_api_attempts": len(rows)
        * REPAIR_MAX_PHYSICAL_ATTEMPTS_PER_CALL,
        "estimated_cost_usd": one_attempt_cost,
        "maximum_estimated_cost_usd": maximum_cost,
        "max_input_tokens": max(int(row["input_token_upper_bound"]) for row in rows),
        "max_output_tokens": max(int(row["maximum_output_tokens"]) for row in rows),
        "pricing_usd_per_mtok": contract["pricing_usd_per_mtok"],
        "contract": contract,
    }
    cost_payload["cost_estimate_sha256"] = sha256_text(canonical_json(cost_payload))
    return cost_payload, rows, messages_by_call


def validate_repair_result(record: Any, parsed: Any) -> dict[str, Any]:
    """Validate the provider result without weakening the exact-25 publisher."""

    def require_substantive(value: str, *, field: str) -> str:
        normalized = normalize_space(value)
        if len(tokens(normalized)) < 3:
            raise RuntimeError(
                f"repair for {record.state_id} returned non-substantive {field}"
            )
        return normalized

    if record.repair_mode == "FIELD_ONLY_REPAIR":
        output = FieldOnlyRepairOutput.model_validate(parsed)
        context = require_substantive(
            output.authorized_user_context, field="authorized_user_context"
        )
        if context == normalize_space(record.authorized_user_context):
            raise RuntimeError(
                f"repair for {record.state_id} left authorized_user_context unchanged"
            )
        summary = None
        if record.state_id in FIELD_ONLY_REPAIR_SUMMARY_ALSO_NEEDED:
            summary = require_substantive(output.session_summary, field="session_summary")
            if summary == normalize_space(record.session_summary):
                raise RuntimeError(
                    f"repair for {record.state_id} left required session_summary unchanged"
                )
        return {
            "state_id": record.state_id,
            "repair_mode": record.repair_mode,
            "authorized_user_context": context,
            "session_summary": summary,
            "recent_dialogue_patch": {},
        }
    output = VisibleSurfaceRepairOutput.model_validate(parsed)
    indices = VISIBLE_SURFACE_REPAIR_TURN_INDICES[record.state_id]
    if len(output.repaired_turn_contents) != len(indices):
        raise RuntimeError(
            f"repair for {record.state_id} returned {len(output.repaired_turn_contents)} "
            f"turns; frozen contract requires {len(indices)}"
        )
    repaired_contents: list[str] = []
    for index, content in zip(indices, output.repaired_turn_contents):
        normalized = require_substantive(content, field=f"history turn {index}")
        if normalized == normalize_space(record.history[index]["content"]):
            raise RuntimeError(
                f"repair for {record.state_id} left frozen repair turn {index} unchanged"
            )
        repaired_contents.append(normalized)
    context = require_substantive(
        output.authorized_user_context, field="authorized_user_context"
    )
    if context == normalize_space(record.authorized_user_context):
        raise RuntimeError(
            f"repair for {record.state_id} left authorized_user_context unchanged"
        )
    original_summary_present = bool(normalize_space(record.session_summary))
    summary = (
        require_substantive(output.session_summary, field="session_summary")
        if original_summary_present
        else ""
    )
    return {
        "state_id": record.state_id,
        "repair_mode": record.repair_mode,
        "authorized_user_context": context,
        "session_summary": summary,
        "recent_dialogue_patch": {
            index: content
            for index, content in zip(indices, repaired_contents)
        },
    }


def materialize_full_repair_overlays(
    *,
    records: Sequence[Any],
    bundles: Sequence[Any],
    classification_sha256: str,
    validated_results: Mapping[str, Mapping[str, Any]],
) -> list[Any]:
    """Build exact-25 overlays. Pilot outputs are structurally rejected."""

    selected = select_repair_records(records, scope="full")
    expected = {record.state_id for record in selected}
    if set(validated_results) != expected:
        raise RuntimeError(
            "formal overlay publication requires results for exactly all 25 "
            "frozen DATA_DEFECT states"
        )
    overlays = []
    for record in selected:
        result = dict(validated_results[record.state_id])
        overlays.append(
            build_repair_overlay_record(
                record=record,
                bundles=bundles,
                classification_sha256=classification_sha256,
                authorized_user_context=str(result["authorized_user_context"]),
                session_summary=result.get("session_summary"),
                recent_dialogue_patch={
                    int(index): str(content)
                    for index, content in dict(
                        result.get("recent_dialogue_patch") or {}
                    ).items()
                },
            )
        )
    return overlays
