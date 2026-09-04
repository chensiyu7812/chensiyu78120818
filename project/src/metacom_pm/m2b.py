from __future__ import annotations

from pathlib import Path
from typing import Any

from .api import Endpoint, OpenAICompatibleClient, make_client
from .artifacts import create_artifact_attestation, require_artifact_attestation
from .contracts import (
    ActionOutcome,
    MemorySelectedSetOmissionJudgment,
    MemorySource,
    RuntimeState,
)
from .io import (
    append_jsonl,
    canonical_json,
    ensure_run_manifest,
    iter_jsonl,
    load_done_keys,
    sha256_file,
    sha256_text,
    write_json,
)
from .judging import (
    _call_with_semantic_retry,
    _key_inventory,
    _load_outcomes,
    _validate_exact_output,
)
from .prompts import memory_selected_set_omission_messages
from .sweep import load_backends, load_states


def _m2b_protocol_hash() -> str:
    root = Path(__file__).resolve().parent
    paths = [
        root / "m2b.py",
        root / "prompts.py",
        root / "contracts.py",
    ]
    return sha256_text(canonical_json({p.name: sha256_file(p) for p in paths}))


def _semantic_validate_m2b(
    parsed: MemorySelectedSetOmissionJudgment,
    state: RuntimeState,
    outcome: ActionOutcome,
) -> None:
    selected_sources = {item.source for item in outcome.memory_view}
    available_sources = {
        source for source, catalog in state.inventory.items() if catalog.available
    }
    missed = set(parsed.missed_useful_sources)
    if missed & selected_sources:
        raise ValueError(
            "missed_useful_sources cannot include selected sources: "
            + ",".join(sorted(x.value for x in missed & selected_sources))
        )
    if missed - available_sources:
        raise ValueError(
            "missed_useful_sources contains unavailable sources: "
            + ",".join(sorted(x.value for x in missed - available_sources))
        )
    if parsed.selected_set_omission_severity == 0 and missed:
        raise ValueError("severity 0 requires missed_useful_sources=[]")
    if parsed.selected_set_omission_severity > 0 and not missed:
        raise ValueError("nonzero omission severity requires missed_useful_sources")
    expected_sufficiency_max = 2 - parsed.selected_set_omission_severity
    if parsed.selected_set_sufficiency > expected_sufficiency_max:
        raise ValueError(
            "selected_set_sufficiency contradicts selected_set_omission_severity"
        )


def _selected_memory_items(outcome: ActionOutcome):
    return list(outcome.memory_view)


def run_m2b_audit(
    runtime_path: str | Path,
    backend_path: str | Path,
    outcomes_path: str | Path,
    full_judging_attestation_path: str | Path,
    out_dir: str | Path,
    *,
    endpoint: Endpoint,
    outcomes_attestation_path: str | Path | None = None,
    max_rows: int | None = None,
    overwrite: bool = False,
    study_freeze_sha256: str | None = None,
) -> dict[str, Any]:
    """Run selected-set omission auditing for non-M0 memory actions.

    This is intentionally separate from the frozen full judging artifact.  It
    adds the missing supervision for non-M0 actions that selected some memory
    but may have omitted another materially useful source.
    """
    runtime_path = Path(runtime_path).resolve()
    backend_path = Path(backend_path).resolve()
    outcomes_path = Path(outcomes_path).resolve()
    full_judging_attestation_path = Path(full_judging_attestation_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    outcomes_attestation_path = Path(
        outcomes_attestation_path
        or outcomes_path.parent / "artifact_attestation.json"
    ).resolve()
    outcomes_verification = require_artifact_attestation(
        outcomes_attestation_path,
        required_stage="action_sweep",
        required_output_paths={"action_outcomes": outcomes_path},
        expected_freeze_sha256=study_freeze_sha256,
    )
    full_judging_verification = require_artifact_attestation(
        full_judging_attestation_path,
        required_stage="full_judging",
        expected_freeze_sha256=study_freeze_sha256,
    )

    states = load_states(runtime_path)
    backends = load_backends(backend_path)
    outcomes = _load_outcomes(outcomes_path)
    if set(states) != set(backends):
        raise ValueError("runtime/backend card IDs differ")

    jobs: list[tuple[str, str]] = []
    for card_id in sorted(states):
        state = states[card_id]
        for action_id in sorted(state.allowed_actions):
            outcome = outcomes.get((card_id, action_id))
            if outcome is None:
                raise ValueError(f"missing action outcome: {(card_id, action_id)}")
            if not outcome.memory_view:
                continue
            jobs.append((card_id, action_id))
    if max_rows is not None:
        jobs = jobs[:max_rows]

    expected = set(jobs)
    out_path = out_dir / "memory_selected_set_omission_judgments.jsonl"
    raw_path = out_dir / "raw_m2b_judge_calls.jsonl"
    manifest_path = out_dir / "run_manifest.json"
    summary_path = out_dir / "m2b_summary.json"
    attestation_path = out_dir / "artifact_attestation.json"
    if overwrite:
        for path in (out_path, raw_path, manifest_path, summary_path, attestation_path):
            if path.exists():
                path.unlink()

    protocol_hash = _m2b_protocol_hash()
    run_metadata = {
        "stage": "m2b_selected_set_omission",
        "runtime_sha256": sha256_file(runtime_path),
        "backend_sha256": sha256_file(backend_path),
        "outcomes_sha256": sha256_file(outcomes_path),
        "outcomes_attestation_sha256": outcomes_verification["attestation_sha256"],
        "full_judging_attestation_sha256": full_judging_verification["attestation_sha256"],
        "endpoint_model": endpoint.model,
        "endpoint_family": endpoint.family,
        "endpoint_base_url": endpoint.base_url,
        "max_rows": max_rows,
        "selected_job_count": len(jobs),
        "selected_job_ids_sha256": sha256_text(canonical_json(sorted(jobs))),
        "m2b_protocol_sha256": protocol_hash,
        "study_freeze_sha256": study_freeze_sha256,
    }
    ensure_run_manifest(manifest_path, run_metadata, overwrite=False)

    actual, duplicates, _ = _key_inventory(out_path, ("card_id", "action_id"))
    extras = actual - expected
    if duplicates or extras:
        raise RuntimeError(
            "M2b output is incompatible with immutable run manifest; "
            f"duplicates={duplicates[:5]}, extras={sorted(extras)[:5]}."
        )

    client = (
        make_client(endpoint)
        if "anthropic.com" in endpoint.base_url
        else OpenAICompatibleClient(endpoint)
    )
    failures: list[dict[str, Any]] = []
    try:
        done = load_done_keys(out_path, ("card_id", "action_id"))
        for card_id, action_id in jobs:
            if (card_id, action_id) in done:
                continue
            state = states[card_id]
            backend = backends[card_id]
            outcome = outcomes[(card_id, action_id)]
            selected_items = _selected_memory_items(outcome)
            if not selected_items:
                raise RuntimeError(f"M2b job unexpectedly has no selected memory: {card_id} {action_id}")
            messages = memory_selected_set_omission_messages(
                state,
                selected_items,
                outcome.response,
                all_items=backend.items,
            )
            try:
                parsed = _call_with_semantic_retry(
                    client,
                    endpoint,
                    messages,
                    MemorySelectedSetOmissionJudgment,
                    lambda value, state=state, outcome=outcome: _semantic_validate_m2b(
                        value, state, outcome
                    ),
                    stage="memory_selected_set_omission",
                    record_ids={"card_id": card_id, "action_id": action_id},
                    raw_log_path=raw_path,
                )
                append_jsonl(
                    out_path,
                    {
                        "card_id": card_id,
                        "action_id": action_id,
                        **parsed.model_dump(mode="json"),
                    },
                )
            except Exception as exc:
                failures.append({
                    "stage": "m2b",
                    "card_id": card_id,
                    "action_id": action_id,
                    "error": str(exc),
                })
    finally:
        client.close()

    validation = _validate_exact_output(
        name="m2b",
        path=out_path,
        fields=("card_id", "action_id"),
        expected=expected,
    )
    status = "COMPLETE" if validation["ok"] and not failures else "INCOMPLETE"
    summary = {
        "status": status,
        "expected_m2b": len(expected),
        "validation": validation,
        "failures": failures,
        "input_hashes": {
            "runtime": sha256_file(runtime_path),
            "backend": sha256_file(backend_path),
            "outcomes": sha256_file(outcomes_path),
        },
        "endpoint_model": endpoint.model,
        "endpoint_family": endpoint.family,
        "endpoint_base_url": endpoint.base_url,
        "m2b_protocol_sha256": protocol_hash,
        "run_manifest_sha256": sha256_file(manifest_path),
        "outcomes_attestation_verification": outcomes_verification,
        "full_judging_attestation_verification": full_judging_verification,
        "study_freeze_sha256": study_freeze_sha256,
    }
    write_json(summary_path, summary)
    if status != "COMPLETE":
        raise RuntimeError(
            f"M2b audit incomplete: failures={len(failures)}, validation_ok={validation['ok']}"
        )

    create_artifact_attestation(
        attestation_path,
        stage="m2b_selected_set_omission",
        inputs={
            "runtime": runtime_path,
            "backend": backend_path,
            "outcomes": outcomes_path,
            "outcomes_attestation": outcomes_attestation_path,
            "full_judging_attestation": full_judging_attestation_path,
        },
        outputs={
            "memory_selected_set_omission": (out_path, True),
            "m2b_summary": (summary_path, False),
            "run_manifest": (manifest_path, False),
        },
        parameters={
            "endpoint_model": endpoint.model,
            "endpoint_family": endpoint.family,
            "endpoint_base_url": endpoint.base_url,
            "max_rows": max_rows,
            "m2b_protocol_sha256": protocol_hash,
        },
        expected={"m2b_rows": len(expected)},
        study_freeze_sha256=study_freeze_sha256,
    )
    summary["artifact_attestation"] = str(attestation_path)
    return summary
