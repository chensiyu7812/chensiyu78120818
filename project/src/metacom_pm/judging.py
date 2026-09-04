from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Type
import json

from pydantic import BaseModel

from .api import Endpoint, OpenAICompatibleClient, make_client, request_log
from .artifacts import create_artifact_attestation, require_artifact_attestation
from .contracts import (
    ActionOutcome,
    MemoryBackendRecord,
    MemoryOmissionJudgment,
    MemoryOpportunityJudgment,
    MemorySource,
    MemoryUseJudgment,
    PairRecord,
    ResponsePairJudgment,
    RuntimeState,
    StrategyOmissionJudgment,
    StrategyUseJudgment,
)
from .controls import heldout_controls
from .io import (
    append_jsonl,
    canonical_json,
    iter_jsonl,
    load_done_keys,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    ensure_run_manifest,
)
from .prompts import (
    memory_omission_messages,
    memory_opportunity_messages,
    memory_use_messages,
    response_pair_messages,
    strategy_omission_messages,
    strategy_use_messages,
)
from .sweep import load_backends, load_states
from .sampling import select_stratified_card_ids


def _load_outcomes(path: str | Path) -> dict[tuple[str, str], ActionOutcome]:
    out = {}
    for row in iter_jsonl(path):
        value = ActionOutcome.model_validate(row)
        key = (value.card_id, value.action_id)
        if key in out:
            raise ValueError(f"duplicate outcome {key}")
        out[key] = value
    return out


def _semantic_validate_m1(
    parsed: MemoryOpportunityJudgment,
    backend: MemoryBackendRecord,
) -> None:
    expected = {x.memory_id for x in backend.items}
    actual = [x.memory_id for x in parsed.items]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f"M1 IDs mismatch: expected={sorted(expected)}, actual={sorted(actual)}")


def _semantic_validate_m2(parsed: MemoryUseJudgment, outcome: ActionOutcome) -> None:
    expected = {x.source for x in outcome.memory_view}
    actual = [x.source for x in parsed.source_assessments]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(
            f"M2 source set mismatch: expected={sorted(x.value for x in expected)}, "
            f"actual={sorted(x.value for x in actual)}"
        )


def _semantic_validate_strategy_use(parsed: StrategyUseJudgment) -> None:
    if parsed.strategy_utilization > 0 and parsed.strategy_relevance == 0:
        raise ValueError("strategy_utilization > 0 requires strategy_relevance > 0")
    reason = parsed.reason.casefold()
    positive_alignment_markers = (
        "aligned with",
        "aligns with",
        "followed",
        "follows the guidance",
        "reflected",
        "reflects the user's feeling",
        "mirrored",
        "matches the guidance",
        "matched the guidance",
        "used the guidance",
        "utilized",
        "directly addresses",
        "effectively uses",
    )
    if (
        parsed.strategy_relevance == 0
        and parsed.strategy_utilization == 0
        and any(marker in reason for marker in positive_alignment_markers)
    ):
        raise ValueError(
            "strategy scores contradict the reason: the reason describes "
            "alignment/use/reflection but relevance and utilization are both 0"
        )


RESPONSE_LABEL_FIELDS = (
    "preference",
    "empathy",
    "contextual_fit",
    "guidance_fit",
    "non_intrusiveness",
    "coherence",
)


def _invert_ab_label(value: str) -> str:
    if value == "A":
        return "B"
    if value == "B":
        return "A"
    return "tie"


def _winner_from_preference(pair: PairRecord, preference: str) -> str:
    if preference == "A":
        return pair.action_a
    if preference == "B":
        return pair.action_b
    return "tie"


def _winner_from_reversed_preference(pair: PairRecord, preference: str) -> str:
    if preference == "A":
        return pair.action_b
    if preference == "B":
        return pair.action_a
    return "tie"


def _dual_order_response_row(
    pair: PairRecord,
    fwd: ResponsePairJudgment,
    rev: ResponsePairJudgment,
) -> dict[str, Any]:
    fwd_dump = fwd.model_dump(mode="json")
    rev_dump = rev.model_dump(mode="json")
    rev_original_labels = {
        field: _invert_ab_label(str(rev_dump[field]))
        for field in RESPONSE_LABEL_FIELDS
    }
    row: dict[str, Any] = pair.model_dump(mode="json")
    disagreements: list[str] = []
    for field in RESPONSE_LABEL_FIELDS:
        fwd_value = str(fwd_dump[field])
        rev_value = rev_original_labels[field]
        if fwd_value == rev_value:
            row[field] = fwd_value
        else:
            row[field] = "tie"
            disagreements.append(field)

    fwd_winner = _winner_from_preference(pair, fwd.preference)
    rev_winner = _winner_from_reversed_preference(pair, rev.preference)
    winner = fwd_winner if fwd_winner == rev_winner else "tie"
    row["preference"] = (
        "A" if winner == pair.action_a
        else "B" if winner == pair.action_b
        else "tie"
    )
    row["winner_action"] = winner
    row["reason"] = (
        "Dual-order response judgment. "
        f"Forward preference={fwd.preference}; reversed preference={rev.preference}. "
        + (
            "Both orders selected the same underlying response."
            if fwd_winner == rev_winner
            else "Orders disagreed, so the final preference was resolved to tie."
        )
    )
    row["fwd_preference"] = fwd.preference
    row["rev_preference"] = rev.preference
    row["fwd_winner_action"] = fwd_winner
    row["rev_winner_action"] = rev_winner
    row["dual_order_debiased"] = True
    row["dual_order_agreed"] = fwd_winner == rev_winner
    row["dual_order_disagreement_fields"] = disagreements
    row["fwd_judgment"] = fwd_dump
    row["rev_judgment"] = rev_dump
    row["rev_judgment_original_order_labels"] = rev_original_labels
    return row


def _call_with_semantic_retry(
    client: OpenAICompatibleClient,
    endpoint: Endpoint,
    messages: list[dict[str, str]],
    schema: Type[BaseModel],
    validator: Callable[[BaseModel], None] | None,
    *,
    stage: str,
    record_ids: dict[str, Any],
    raw_log_path: str | Path,
    max_tokens: int = 800,
    outer_retries: int = 3,
) -> BaseModel:
    prompt_hash = sha256_text(canonical_json(messages))
    current_messages = list(messages)
    last_error = None
    for outer in range(1, outer_retries + 1):
        result = None
        parsed = None
        try:
            result, parsed = client.chat(
                current_messages,
                temperature=0.0,
                max_tokens=max_tokens,
                response_schema=schema,
                retries=3,
            )
            assert parsed is not None
            if validator is not None:
                validator(parsed)
            append_jsonl(
                raw_log_path,
                request_log(
                    stage=stage,
                    endpoint=endpoint,
                    messages=current_messages,
                    result=result,
                    parsed=parsed,
                    error=None,
                    prompt_hash=prompt_hash,
                    record_ids={**record_ids, "outer_attempt": outer},
                ),
            )
            return parsed
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            append_jsonl(
                raw_log_path,
                request_log(
                    stage=stage,
                    endpoint=endpoint,
                    messages=current_messages,
                    result=result,
                    parsed=parsed,
                    error=last_error,
                    prompt_hash=prompt_hash,
                    record_ids={**record_ids, "outer_attempt": outer},
                ),
            )
            current_messages = list(messages) + [{
                "role": "user",
                "content": (
                    "Your previous output failed strict validation: "
                    + last_error
                    + ". Return a corrected JSON object matching the schema exactly."
                ),
            }]
    raise RuntimeError(f"{stage} failed semantic validation: {last_error}")


def _run_controls(
    client: OpenAICompatibleClient,
    endpoint: Endpoint,
    out_path: str | Path,
    raw_log_path: str | Path,
) -> list[dict[str, Any]]:
    done = load_done_keys(out_path, ("control_id",))
    rows: list[dict[str, Any]] = []
    for control in heldout_controls():
        if (control["control_id"],) in done:
            continue
        kind = control["kind"]
        if kind == "response":
            schema = ResponsePairJudgment
        elif kind == "memory_use":
            schema = MemoryUseJudgment
        elif kind == "memory_omission":
            schema = MemoryOmissionJudgment
        elif kind == "strategy":
            schema = StrategyUseJudgment
        elif kind == "strategy_omission":
            schema = StrategyOmissionJudgment
        else:
            raise ValueError(kind)
        validator = _semantic_validate_strategy_use if kind == "strategy" else None
        parsed = _call_with_semantic_retry(
            client, endpoint, control["messages"], schema, validator,
            stage=f"control_{kind}",
            record_ids={"control_id": control["control_id"]},
            raw_log_path=raw_log_path,
        )
        row = {
            "control_id": control["control_id"],
            "kind": kind,
            "control_tier": control.get("gate_tier", "hard"),
            "control_rationale": control.get("gate_rationale", ""),
            "expected": control["expected"],
            "validated": parsed.model_dump(mode="json"),
        }
        if "selected_sources" in control:
            row["selected_sources"] = control["selected_sources"]
        append_jsonl(out_path, row)
        rows.append(row)
    return rows



def _key_inventory(
    path: str | Path,
    fields: tuple[str, ...],
) -> tuple[set[tuple[Any, ...]], list[tuple[Any, ...]], int]:
    p = Path(path)
    if not p.exists():
        return set(), [], 0
    seen: set[tuple[Any, ...]] = set()
    duplicates: list[tuple[Any, ...]] = []
    n = 0
    for row in iter_jsonl(p):
        n += 1
        key = tuple(row.get(field) for field in fields)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    return seen, duplicates, n


def _judge_protocol_hash() -> str:
    root = Path(__file__).resolve().parent
    paths = [
        root / "judging.py",
        root / "prompts.py",
        root / "contracts.py",
        root / "controls.py",
    ]
    return sha256_text(canonical_json({p.name: sha256_file(p) for p in paths}))


def _validate_exact_output(
    *,
    name: str,
    path: str | Path,
    fields: tuple[str, ...],
    expected: set[tuple[Any, ...]],
) -> dict[str, Any]:
    actual, duplicates, n_rows = _key_inventory(path, fields)
    missing = expected - actual
    extras = actual - expected
    return {
        "name": name,
        "path": str(Path(path).resolve()),
        "expected": len(expected),
        "rows": n_rows,
        "unique": len(actual),
        "missing": len(missing),
        "extra": len(extras),
        "duplicates": len(duplicates),
        "missing_examples": sorted(missing)[:10],
        "extra_examples": sorted(extras)[:10],
        "duplicate_examples": duplicates[:10],
        "ok": not missing and not extras and not duplicates and n_rows == len(expected),
    }


def run_judging(
    runtime_path: str | Path,
    backend_path: str | Path,
    outcomes_path: str | Path,
    pairs_path: str | Path,
    out_dir: str | Path,
    *,
    endpoint: Endpoint,
    mode: str = "pilot",
    max_cards: int | None = 20,
    pilot_gate_path: str | Path | None = None,
    outcomes_attestation_path: str | Path | None = None,
    pilot_gate_attestation_path: str | Path | None = None,
    overwrite: bool = False,
    study_freeze_sha256: str | None = None,
) -> dict[str, Any]:
    """Run schema-validated judging with immutable provenance and exact coverage.

    The function fails closed on stale/extra rows, duplicate keys, an action
    sweep not bound by an attestation, or a full run that is not compatible
    with the exact pilot measurement protocol that passed the gate.
    """
    runtime_path = Path(runtime_path).resolve()
    backend_path = Path(backend_path).resolve()
    outcomes_path = Path(outcomes_path).resolve()
    pairs_path = Path(pairs_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode not in {"pilot", "full"}:
        raise ValueError("mode must be pilot or full")
    if mode == "full" and max_cards is not None:
        raise ValueError("full judging must use max_cards=None")

    # Never trust an outcomes filename by itself. Bind it to the exact sweep
    # inputs/configuration and reject tampered or partially resumed output.
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

    states = load_states(runtime_path)
    backends = load_backends(backend_path)
    if set(states) != set(backends):
        raise ValueError("runtime/backend card IDs differ")
    outcomes = _load_outcomes(outcomes_path)
    all_pairs = [PairRecord.model_validate(row) for row in iter_jsonl(pairs_path)]
    if len({x.pair_id for x in all_pairs}) != len(all_pairs):
        raise ValueError("pair graph contains duplicate pair_id values")

    pair_card_ids = {x.card_id for x in all_pairs}
    unknown_pair_cards = pair_card_ids - set(states)
    if unknown_pair_cards:
        raise ValueError(f"pair graph references unknown cards: {sorted(unknown_pair_cards)[:5]}")
    card_ids = sorted(pair_card_ids)
    if max_cards is not None:
        card_ids = select_stratified_card_ids(states, max_cards, eligible=pair_card_ids)
    card_ids = sorted(card_ids)
    card_set = set(card_ids)
    pairs = [x for x in all_pairs if x.card_id in card_set]

    expected_outcomes = {
        (card_id, action_id)
        for card_id in card_ids
        for action_id in states[card_id].allowed_actions
    }
    missing_outcomes = expected_outcomes - set(outcomes)
    if missing_outcomes:
        raise ValueError(f"missing outcomes for selected judging jobs: {sorted(missing_outcomes)[:10]}")

    input_hashes = {
        "runtime": sha256_file(runtime_path),
        "backend": sha256_file(backend_path),
        "outcomes": sha256_file(outcomes_path),
        "pairs": sha256_file(pairs_path),
    }
    protocol_hash = _judge_protocol_hash()

    gate: dict[str, Any] | None = None
    gate_verification: dict[str, Any] | None = None
    if mode == "full":
        if pilot_gate_path is None:
            raise ValueError("full judging requires pilot_gate_path")
        pilot_gate_path = Path(pilot_gate_path).resolve()
        gate = read_json(pilot_gate_path)
        if gate.get("status") != "PILOT_GO_FULL_JUDGING":
            raise RuntimeError("pilot gate did not authorize full judging")
        pilot_gate_attestation_path = Path(
            pilot_gate_attestation_path
            or pilot_gate_path.parent / "pilot_gate_attestation.json"
        ).resolve()
        gate_verification = require_artifact_attestation(
            pilot_gate_attestation_path,
            required_stage="judge_pilot_gate",
            required_output_paths={"pilot_gate": pilot_gate_path},
        )
        signature = gate.get("measurement_signature") or {}
        incompatibilities: list[str] = []
        if signature.get("input_hashes") != input_hashes:
            incompatibilities.append("pilot/full input hashes differ")
        if signature.get("judge_protocol_sha256") != protocol_hash:
            incompatibilities.append("judge prompts/schemas/code changed after pilot")
        if signature.get("endpoint_model") != endpoint.model:
            incompatibilities.append("judge model differs from pilot")
        if signature.get("endpoint_family") != endpoint.family:
            incompatibilities.append("judge family differs from pilot")
        if signature.get("endpoint_base_url") != endpoint.base_url:
            incompatibilities.append("judge endpoint differs from pilot")
        if incompatibilities:
            raise RuntimeError(
                "full judging is incompatible with the passed pilot:\n- "
                + "\n- ".join(incompatibilities)
            )

    paths = {
        "response": out_dir / "response_pair_judgments.jsonl",
        "m1": out_dir / "memory_opportunity_judgments.jsonl",
        "m2": out_dir / "memory_use_judgments.jsonl",
        "m0": out_dir / "memory_omission_judgments.jsonl",
        "strategy": out_dir / "strategy_use_judgments.jsonl",
        "strategy_omission": out_dir / "strategy_omission_judgments.jsonl",
        "controls": out_dir / "heldout_control_judgments.jsonl",
        "raw": out_dir / "raw_judge_calls.jsonl",
    }
    manifest_path = out_dir / "run_manifest.json"
    summary_path = out_dir / "judging_summary.json"
    attestation_path = out_dir / "artifact_attestation.json"
    if overwrite:
        for path in [*paths.values(), manifest_path, summary_path, attestation_path]:
            if path.exists():
                path.unlink()

    run_metadata = {
        "stage": "judging",
        "mode": mode,
        **{f"{key}_sha256": value for key, value in input_hashes.items()},
        "outcomes_attestation_sha256": outcomes_verification["attestation_sha256"],
        "endpoint_model": endpoint.model,
        "endpoint_family": endpoint.family,
        "endpoint_base_url": endpoint.base_url,
        "max_cards": max_cards,
        "selected_card_count": len(card_ids),
        "selected_card_ids_sha256": sha256_text(canonical_json(card_ids)),
        "selected_pair_count": len(pairs),
        "selected_pair_ids_sha256": sha256_text(canonical_json(sorted(x.pair_id for x in pairs))),
        "judge_protocol_sha256": protocol_hash,
        "pilot_gate_sha256": (sha256_file(pilot_gate_path) if pilot_gate_path else None),
        "pilot_gate_attestation_sha256": (
            gate_verification["attestation_sha256"] if gate_verification else None
        ),
        "study_freeze_sha256": study_freeze_sha256,
    }
    ensure_run_manifest(manifest_path, run_metadata, overwrite=False)

    expected_m1 = {(card_id,) for card_id in card_ids}
    expected_m0 = {
        (card_id, action_id)
        for card_id, action_id in expected_outcomes
        if not outcomes[(card_id, action_id)].memory_view
    }
    expected_m2 = expected_outcomes - expected_m0
    expected_strategy = {
        (card_id, action_id)
        for card_id, action_id in expected_outcomes
        if outcomes[(card_id, action_id)].strategy_view
    }
    expected_strategy_omission = expected_outcomes - expected_strategy
    expected_response = {(pair.pair_id,) for pair in pairs}
    expected_controls = (
        {(x["control_id"],) for x in heldout_controls()} if mode == "pilot" else set()
    )

    # Reject stale/extra rows before making any new API calls.
    pre_specs = [
        ("m1", paths["m1"], ("card_id",), expected_m1),
        ("m0", paths["m0"], ("card_id", "action_id"), expected_m0),
        ("m2", paths["m2"], ("card_id", "action_id"), expected_m2),
        ("strategy", paths["strategy"], ("card_id", "action_id"), expected_strategy),
        ("strategy_omission", paths["strategy_omission"], ("card_id", "action_id"), expected_strategy_omission),
        ("response", paths["response"], ("pair_id",), expected_response),
    ]
    if mode == "pilot":
        pre_specs.append(("controls", paths["controls"], ("control_id",), expected_controls))
    for name, path, fields, expected in pre_specs:
        actual, duplicates, _ = _key_inventory(path, fields)
        extras = actual - expected
        if duplicates or extras:
            raise RuntimeError(
                f"{name} output is incompatible with immutable run manifest; "
                f"duplicates={duplicates[:5]}, extras={sorted(extras)[:5]}. "
                "Use a new directory or --overwrite."
            )

    client = (
        make_client(endpoint)
        if "anthropic.com" in endpoint.base_url
        else OpenAICompatibleClient(endpoint)
    )
    failures: list[dict[str, Any]] = []
    try:
        if mode == "pilot":
            _run_controls(client, endpoint, paths["controls"], paths["raw"])

        done_m1 = load_done_keys(paths["m1"], ("card_id",))
        for card_id in card_ids:
            if (card_id,) in done_m1:
                continue
            state, backend = states[card_id], backends[card_id]
            messages = memory_opportunity_messages(state, backend.items)
            try:
                parsed = _call_with_semantic_retry(
                    client, endpoint, messages, MemoryOpportunityJudgment,
                    lambda value, backend=backend: _semantic_validate_m1(value, backend),
                    stage="memory_opportunity",
                    record_ids={"card_id": card_id},
                    raw_log_path=paths["raw"],
                    max_tokens=1200,
                )
                append_jsonl(paths["m1"], {"card_id": card_id, **parsed.model_dump(mode="json")})
            except Exception as exc:
                failures.append({"stage": "m1", "card_id": card_id, "error": str(exc)})

        done_m0 = load_done_keys(paths["m0"], ("card_id", "action_id"))
        done_m2 = load_done_keys(paths["m2"], ("card_id", "action_id"))
        done_s = load_done_keys(paths["strategy"], ("card_id", "action_id"))
        done_s0 = load_done_keys(paths["strategy_omission"], ("card_id", "action_id"))
        for card_id in card_ids:
            state, backend = states[card_id], backends[card_id]
            for action_id in state.allowed_actions:
                outcome = outcomes[(card_id, action_id)]
                if not outcome.memory_view and (card_id, action_id) not in done_m0:
                    messages = memory_omission_messages(state, backend.items, outcome.response)
                    try:
                        parsed = _call_with_semantic_retry(
                            client, endpoint, messages, MemoryOmissionJudgment, None,
                            stage="memory_omission",
                            record_ids={"card_id": card_id, "action_id": action_id},
                            raw_log_path=paths["raw"],
                        )
                        append_jsonl(paths["m0"], {"card_id": card_id, "action_id": action_id, **parsed.model_dump(mode="json")})
                    except Exception as exc:
                        failures.append({"stage": "m0", "card_id": card_id, "action_id": action_id, "error": str(exc)})
                elif outcome.memory_view and (card_id, action_id) not in done_m2:
                    messages = memory_use_messages(
                        state, outcome.memory_view, outcome.response, all_items=backend.items
                    )
                    try:
                        parsed = _call_with_semantic_retry(
                            client, endpoint, messages, MemoryUseJudgment,
                            lambda value, outcome=outcome: _semantic_validate_m2(value, outcome),
                            stage="memory_use",
                            record_ids={"card_id": card_id, "action_id": action_id},
                            raw_log_path=paths["raw"],
                        )
                        append_jsonl(paths["m2"], {"card_id": card_id, "action_id": action_id, **parsed.model_dump(mode="json")})
                    except Exception as exc:
                        failures.append({"stage": "m2", "card_id": card_id, "action_id": action_id, "error": str(exc)})

                if outcome.strategy_view and (card_id, action_id) not in done_s:
                    messages = strategy_use_messages(state, outcome.strategy_view, outcome.response)
                    try:
                        parsed = _call_with_semantic_retry(
                            client, endpoint, messages, StrategyUseJudgment,
                            _semantic_validate_strategy_use,
                            stage="strategy_use",
                            record_ids={"card_id": card_id, "action_id": action_id},
                            raw_log_path=paths["raw"],
                        )
                        append_jsonl(paths["strategy"], {"card_id": card_id, "action_id": action_id, **parsed.model_dump(mode="json")})
                    except Exception as exc:
                        failures.append({"stage": "strategy", "card_id": card_id, "action_id": action_id, "error": str(exc)})
                elif not outcome.strategy_view and (card_id, action_id) not in done_s0:
                    messages = strategy_omission_messages(state, outcome.response)
                    try:
                        parsed = _call_with_semantic_retry(
                            client, endpoint, messages, StrategyOmissionJudgment, None,
                            stage="strategy_omission",
                            record_ids={"card_id": card_id, "action_id": action_id},
                            raw_log_path=paths["raw"],
                        )
                        append_jsonl(paths["strategy_omission"], {"card_id": card_id, "action_id": action_id, **parsed.model_dump(mode="json")})
                    except Exception as exc:
                        failures.append({"stage": "strategy_omission", "card_id": card_id, "action_id": action_id, "error": str(exc)})

        done_response = load_done_keys(paths["response"], ("pair_id",))
        for pair in pairs:
            if (pair.pair_id,) in done_response:
                continue
            state = states[pair.card_id]
            outcome_a = outcomes[(pair.card_id, pair.action_a)]
            outcome_b = outcomes[(pair.card_id, pair.action_b)]
            try:
                messages = response_pair_messages(
                    state, outcome_a.response, outcome_b.response
                )
                if pair.training_eligible:
                    # Dual-order debiasing for labels that may train the PM:
                    # run forward (A=action_a, B=action_b) and reversed
                    # (A=action_b, B=action_a). Only trust verdicts that
                    # select the same underlying response; resolve conflicts to
                    # tie. Audit-only reversal/repeat rows stay single-order so
                    # they continue to measure raw judge stability.
                    fwd = _call_with_semantic_retry(
                        client, endpoint, messages, ResponsePairJudgment, None,
                        stage="response_pair_fwd",
                        record_ids={"pair_id": pair.pair_id, "card_id": pair.card_id},
                        raw_log_path=paths["raw"],
                    )
                    rev_messages = response_pair_messages(
                        state, outcome_b.response, outcome_a.response
                    )
                    rev = _call_with_semantic_retry(
                        client, endpoint, rev_messages, ResponsePairJudgment, None,
                        stage="response_pair_rev",
                        record_ids={"pair_id": pair.pair_id, "card_id": pair.card_id},
                        raw_log_path=paths["raw"],
                    )
                    append_jsonl(paths["response"], _dual_order_response_row(pair, fwd, rev))
                else:
                    parsed = _call_with_semantic_retry(
                        client, endpoint, messages, ResponsePairJudgment, None,
                        stage="response_pair",
                        record_ids={"pair_id": pair.pair_id, "card_id": pair.card_id},
                        raw_log_path=paths["raw"],
                    )
                    winner = _winner_from_preference(pair, parsed.preference)
                    append_jsonl(paths["response"], {
                        **pair.model_dump(mode="json"),
                        **parsed.model_dump(mode="json"),
                        "winner_action": winner,
                        "fwd_preference": parsed.preference,
                        "fwd_winner_action": winner,
                        "dual_order_debiased": False,
                    })
            except Exception as exc:
                failures.append({"stage": "response", "pair_id": pair.pair_id, "card_id": pair.card_id, "error": str(exc)})
    finally:
        client.close()

    validations = {
        name: _validate_exact_output(name=name, path=path, fields=fields, expected=expected)
        for name, path, fields, expected in pre_specs
    }
    exact_complete = all(value["ok"] for value in validations.values())
    status = "COMPLETE" if exact_complete and not failures else "INCOMPLETE"
    summary = {
        "mode": mode,
        "status": status,
        "n_cards": len(card_ids),
        "card_ids": card_ids,
        "n_pairs": len(pairs),
        "expected_m1": len(expected_m1),
        "expected_m0": len(expected_m0),
        "expected_m2": len(expected_m2),
        "expected_strategy": len(expected_strategy),
        "expected_strategy_omission": len(expected_strategy_omission),
        "expected_controls": len(expected_controls),
        "validations": validations,
        "failures": failures,
        "input_hashes": input_hashes,
        "model": endpoint.model,
        "model_family": endpoint.family,
        "endpoint_base_url": endpoint.base_url,
        "judge_protocol_sha256": protocol_hash,
        "outcomes_attestation_verification": outcomes_verification,
        "pilot_gate_attestation_verification": gate_verification,
        "run_manifest_sha256": sha256_file(manifest_path),
        "study_freeze_sha256": study_freeze_sha256,
    }
    write_json(summary_path, summary)
    if status != "COMPLETE":
        raise RuntimeError(
            f"judging incomplete: failures={len(failures)}, "
            f"invalid_outputs={[k for k,v in validations.items() if not v['ok']]}"
        )

    output_records = {
        "response_pairs": (paths["response"], True),
        "memory_opportunity": (paths["m1"], True),
        "memory_use": (paths["m2"], True),
        "memory_omission": (paths["m0"], True),
        "strategy_use": (paths["strategy"], True),
        "strategy_omission": (paths["strategy_omission"], True),
        "raw_calls": (paths["raw"], True),
        "summary": (summary_path, False),
    }
    if mode == "pilot":
        output_records["heldout_controls"] = (paths["controls"], True)
    input_records: dict[str, str | Path] = {
        "runtime": runtime_path,
        "backend": backend_path,
        "outcomes": outcomes_path,
        "pairs": pairs_path,
        "run_manifest": manifest_path,
        "outcomes_attestation": outcomes_attestation_path,
    }
    if mode == "full":
        assert pilot_gate_path is not None and pilot_gate_attestation_path is not None
        input_records["pilot_gate"] = pilot_gate_path
        input_records["pilot_gate_attestation"] = pilot_gate_attestation_path
    create_artifact_attestation(
        attestation_path,
        stage="judge_pilot" if mode == "pilot" else "full_judging",
        inputs=input_records,
        outputs=output_records,
        parameters={
            "mode": mode,
            "endpoint_model": endpoint.model,
            "endpoint_family": endpoint.family,
            "endpoint_base_url": endpoint.base_url,
            "max_cards": max_cards,
            "judge_protocol_sha256": protocol_hash,
        },
        expected={
            "cards": len(card_ids),
            "response_pairs": len(expected_response),
            "m1": len(expected_m1),
            "m0": len(expected_m0),
            "m2": len(expected_m2),
            "strategy": len(expected_strategy),
            "strategy_omission": len(expected_strategy_omission),
            "controls": len(expected_controls),
        },
        study_freeze_sha256=study_freeze_sha256,
    )
    return summary
