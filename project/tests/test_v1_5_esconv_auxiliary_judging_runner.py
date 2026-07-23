"""No-API wiring tests for scripts/v1_5/13c_judge_esconv_auxiliary_v1_5.py.

Only exercises --dry-run (zero-cost, no network): cost-estimate shape (one
judge pair per (state, action), four physical calls per pair), fail-closed
behavior on a missing/incomplete generation source, and the frozen call-plan
shuffle's determinism/independence from input row order.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest

from metacom_pm.api import ProviderRequestError, RetryableProviderError
from metacom_pm.contracts import CostRecord, DialogueTurn, MemorySource, SourceCatalog
from metacom_pm.esconv_v1_5 import ESCONV_V1_5_ALLOWED_ACTIONS
from metacom_pm.io import write_json, write_jsonl
from metacom_pm.pm_v2_contracts import ObservableSourceSummary, PMV2Split, PMV2State
from metacom_pm.pm_v2_judging import ResponseJudgeOutput, RiskJudgeOutput

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "outputs" / "v1_5_test_fixtures"


@pytest.fixture()
def workdir() -> Path:
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="esconv_auxiliary_judging_", dir=FIXTURE_ROOT
    ) as directory:
        yield Path(directory)


def _load_module(relpath: str, name: str):
    path = ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _make_state(state_id: str) -> PMV2State:
    return PMV2State(
        state_id=state_id,
        card_id=f"card_{state_id}",
        user_id=f"dialogue_{state_id}",
        split=PMV2Split.TRAIN,
        semantic_family="esconv_auxiliary_strategy_routing",
        surface_form_id=f"surface_{state_id}",
        current_user_text="I've been feeling overwhelmed lately.",
        current_session_history=[
            DialogueTurn(role="user", content="Work has been a lot recently.")
        ],
        current_session_summary="",
        session_index=1,
        inventory={
            source: ObservableSourceSummary(available=False, count=0)
            for source in MemorySource
        },
        allowed_actions=list(ESCONV_V1_5_ALLOWED_ACTIONS),
    )


def _write_fixture(
    aux_dir: Path, gen_dir: Path, split: str, *, n_states: int, shuffled: bool = False
) -> None:
    split_dir = aux_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    states = [_make_state(f"state_{split}_{i}") for i in range(n_states)]
    write_jsonl(
        split_dir / "pm_v2_states.jsonl", [s.model_dump(mode="json") for s in states]
    )
    outcomes = []
    for state in states:
        for action_id in ESCONV_V1_5_ALLOWED_ACTIONS:
            outcomes.append(
                {
                    "card_id": state.card_id,
                    "state_id": state.state_id,
                    "user_id": state.user_id,
                    "action_id": action_id,
                    "response": f"A supportive reply for {state.state_id}/{action_id}.",
                    "selected_memory_ids": [],
                    "selected_strategy_ids": [],
                    "memory_view": [],
                    "strategy_view": [],
                    "cost": CostRecord(
                        pm_input_tokens_est=5,
                        retrieval_calls=0,
                        reranker_calls=0,
                        memory_tokens=0,
                        strategy_tokens=0,
                        base_prompt_tokens=50,
                        total_input_tokens=60,
                        output_tokens=20,
                        latency_ms=1.0,
                    ).model_dump(mode="json"),
                    "model_name": "mock",
                    "prompt_hash": "hash",
                    "request_hash": "request",
                    "provenance": {},
                }
            )
    if shuffled:
        outcomes = list(reversed(outcomes))
    write_jsonl(gen_dir / "action_outcomes.jsonl", outcomes)
    write_json(
        gen_dir / "summary.json",
        {"status": "COMPLETE", "split": split, "completed": len(outcomes)},
    )


def _run_dry_run(
    aux_dir: Path,
    generation_root: Path,
    out_root: Path,
    split: str,
    *,
    pilot: bool = False,
    carry_forward_from: Path | None = None,
) -> dict:
    module = _load_module(
        "scripts/v1_5/13c_judge_esconv_auxiliary_v1_5.py",
        f"v1_5_esconv_auxiliary_judging_runner_test_{id(aux_dir)}_{split}_{pilot}",
    )
    scope = "pilot" if pilot else "full"
    argv = [
        "13c_judge_esconv_auxiliary_v1_5.py",
        "--dry-run",
        "--split",
        split,
        "--scope",
        scope,
        "--auxiliary-dir",
        str(aux_dir),
        "--generation-root",
        str(generation_root),
        "--out-root",
        str(out_root),
        "--max-api-calls",
        "1000",
        "--max-estimated-usd",
        "5.0",
        "--max-input-tokens-per-call",
        "12000",
    ]
    if pilot:
        argv.append("--pilot")
    if carry_forward_from is not None:
        argv.extend(["--carry-forward-from", str(carry_forward_from)])
    previous = sys.argv
    try:
        sys.argv = argv
        module.main()
    finally:
        sys.argv = previous
    out_dir = out_root / f"esconv_auxiliary_judging_v1_5_{scope}_{split}"
    return json.loads((out_dir / "cost_estimate.json").read_text(encoding="utf-8"))


def test_dry_run_has_one_pair_per_state_action_and_four_calls_per_pair(workdir):
    aux_dir = workdir / "aux"
    gen_root = workdir / "gen_root"
    gen_dir = gen_root / "esconv_auxiliary_generation_v1_5_full_train"
    gen_dir.mkdir(parents=True)
    _write_fixture(aux_dir, gen_dir, "train", n_states=3)
    estimate = _run_dry_run(aux_dir, gen_root, workdir / "outputs", "train")
    # 3 states x 2 actions = 6 judge pairs; 6 x 2 families x 2 judge types = 24
    # physical calls.
    assert estimate["expected_judge_pairs"] == 6
    assert estimate["full_logical_api_calls"] == 24
    assert estimate["remaining_new_logical_calls"] == 24
    assert estimate["maximum_physical_attempts_per_logical_call"] == 10
    assert estimate["maximum_physical_http_attempts"] == 240
    assert estimate["planned_new_api_calls"] == 240
    assert estimate["total_input_tokens_est"] == 10 * estimate["logical_input_tokens_est"]
    assert estimate["total_output_tokens_est"] == 10 * estimate["logical_output_tokens_est"]
    assert estimate["estimated_cost_usd"] == pytest.approx(
        10 * estimate["logical_single_attempt_estimated_cost_usd"]
    )
    assert estimate["budget_gate"]["status"] == "PASS"


def test_call_plan_shuffle_is_independent_of_outcomes_file_row_order(workdir):
    aux_dir = workdir / "aux"
    gen_root = workdir / "gen_root"
    gen_dir_a = gen_root / "esconv_auxiliary_generation_v1_5_full_train"
    gen_dir_a.mkdir(parents=True)
    _write_fixture(aux_dir, gen_dir_a, "train", n_states=3, shuffled=False)
    first = _run_dry_run(aux_dir, gen_root, workdir / "outputs_a", "train")

    gen_root_b = workdir / "gen_root_b"
    gen_dir_b = gen_root_b / "esconv_auxiliary_generation_v1_5_full_train"
    gen_dir_b.mkdir(parents=True)
    _write_fixture(aux_dir, gen_dir_b, "train", n_states=3, shuffled=True)
    second = _run_dry_run(aux_dir, gen_root_b, workdir / "outputs_b", "train")

    assert first["call_plan_sha256"] == second["call_plan_sha256"]


def test_missing_generation_outcomes_fails_closed(workdir):
    aux_dir = workdir / "aux"
    gen_root = workdir / "gen_root"
    split_dir = aux_dir / "train"
    split_dir.mkdir(parents=True)
    states = [_make_state("state_only")]
    write_jsonl(
        split_dir / "pm_v2_states.jsonl", [s.model_dump(mode="json") for s in states]
    )
    with pytest.raises(RuntimeError, match="missing ESConv-auxiliary generation"):
        _run_dry_run(aux_dir, gen_root, workdir / "outputs", "train")


def test_incomplete_generation_summary_fails_closed(workdir):
    aux_dir = workdir / "aux"
    gen_root = workdir / "gen_root"
    gen_dir = gen_root / "esconv_auxiliary_generation_v1_5_full_train"
    gen_dir.mkdir(parents=True)
    _write_fixture(aux_dir, gen_dir, "train", n_states=1)
    write_json(gen_dir / "summary.json", {"status": "STARTING"})
    with pytest.raises(RuntimeError, match="not COMPLETE"):
        _run_dry_run(aux_dir, gen_root, workdir / "outputs", "train")


class _ConstantJudgeClient:
    """Every response/risk judgment is identical -- deliberately degenerate,
    to test the raw-family-health constant-dimension gate."""

    calls = 0

    def __init__(self, endpoint):
        pass

    def close(self):
        pass

    def chat(self, messages, *, response_schema, **kwargs):
        type(self).calls += 1
        if response_schema is ResponseJudgeOutput:
            parsed = ResponseJudgeOutput(
                emotional_support=3.0,
                personalization=3.0,
                memory_appropriateness=3.0,
                factual_grounding=3.0,
                temporal_consistency=3.0,
                non_intrusiveness=3.0,
                rationale="constant",
            )
        else:
            parsed = RiskJudgeOutput(
                selected_context_misuse=0.0,
                unnecessary_exposure=0.0,
                stale_or_conflicting_use=0.0,
                unsupported_personal_claim=0.0,
                memory_omission=0.0,
                strategy_overuse=0.0,
                strategy_omission=0.0,
                rationale="constant",
            )
        from metacom_pm.api import CallResult

        result = CallResult(
            text="{}",
            raw_response={"fixture": True},
            usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
            latency_ms=1.0,
            request_hash=f"request-{type(self).calls}",
        )
        return result, parsed


class _OneTerminalOutputThenSuccessClient(_ConstantJudgeClient):
    """One deterministic provider-surface failure must not kill the matrix."""

    def chat(self, messages, *, response_schema, **kwargs):
        if type(self).calls == 0:
            type(self).calls += 1
            raise RetryableProviderError(
                "fixture output hit its token ceiling",
                last_retry_class="output_token_limit",
                last_status_code=None,
                attempts_tried=1,
                request_hash="truncated-request",
                usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            )
        return super().chat(messages, response_schema=response_schema, **kwargs)


class _AlwaysTerminalOutputClient(_ConstantJudgeClient):
    def chat(self, messages, *, response_schema, **kwargs):
        type(self).calls += 1
        raise RetryableProviderError(
            "fixture output hit its token ceiling",
            last_retry_class="output_token_limit",
            last_status_code=None,
            attempts_tried=1,
            request_hash=f"truncated-request-{type(self).calls}",
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )


class _TerminalRequestClient(_ConstantJudgeClient):
    def chat(self, messages, *, response_schema, **kwargs):
        type(self).calls += 1
        raise ProviderRequestError(
            status_code=400,
            detail="fixture request contract rejected",
            schema_mode=True,
            request_hash="bad-request",
        )


def _run_run(
    aux_dir: Path,
    generation_root: Path,
    out_root: Path,
    split: str,
    *,
    accept_cost_estimate_sha256: str,
    pilot: bool = False,
    carry_forward_from: Path | None = None,
    client_cls=_ConstantJudgeClient,
) -> dict:
    module = _load_module(
        "scripts/v1_5/13c_judge_esconv_auxiliary_v1_5.py",
        f"v1_5_esconv_auxiliary_judging_run_test_{id(aux_dir)}_{split}_{pilot}",
    )
    scope = "pilot" if pilot else "full"
    argv = [
        "13c_judge_esconv_auxiliary_v1_5.py",
        "--run",
        "--split",
        split,
        "--scope",
        scope,
        "--auxiliary-dir",
        str(aux_dir),
        "--generation-root",
        str(generation_root),
        "--out-root",
        str(out_root),
        "--max-api-calls",
        "1000",
        "--max-estimated-usd",
        "5.0",
        "--max-input-tokens-per-call",
        "12000",
        "--accept-cost-estimate-sha256",
        accept_cost_estimate_sha256,
    ]
    if pilot:
        argv.append("--pilot")
    if carry_forward_from is not None:
        argv.extend(["--carry-forward-from", str(carry_forward_from)])
    client_cls.calls = 0
    module.make_client = lambda endpoint: client_cls(endpoint)
    # The real paid-run-release gate needs a matching approval in the real
    # project manifest; that gate itself is tested elsewhere
    # (test_central_paid_release_is_fail_closed_and_identity_bound). Here we
    # only care about judging behavior, so bypass it.
    module.require_paid_run_release = lambda *args, **kwargs: {
        "status": "PAID_RUN_RELEASED"
    }
    previous = sys.argv
    try:
        sys.argv = argv
        module.main()
    finally:
        sys.argv = previous
    out_dir = out_root / f"esconv_auxiliary_judging_v1_5_{scope}_{split}"
    return json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))


def test_non_pilot_run_raises_on_constant_risk_dimensions(workdir):
    aux_dir = workdir / "aux"
    gen_root = workdir / "gen_root"
    gen_dir = gen_root / "esconv_auxiliary_generation_v1_5_full_train"
    gen_dir.mkdir(parents=True)
    _write_fixture(aux_dir, gen_dir, "train", n_states=3)
    out_root = workdir / "outputs"
    dry_run_estimate = _run_dry_run(aux_dir, gen_root, out_root, "train")
    with pytest.raises(RuntimeError, match="quality gate failed"):
        _run_run(
            aux_dir,
            gen_root,
            out_root,
            "train",
            accept_cost_estimate_sha256=dry_run_estimate["cost_estimate_sha256"],
        )


def test_pilot_run_completes_as_diagnostic_only_despite_constant_risk_dimensions(
    workdir,
):
    aux_dir = workdir / "aux"
    gen_root = workdir / "gen_root"
    gen_dir = gen_root / "esconv_auxiliary_generation_v1_5_pilot_train"
    gen_dir.mkdir(parents=True)
    _write_fixture(aux_dir, gen_dir, "train", n_states=3)
    pilot_out_root = workdir / "outputs_pilot"
    pilot_estimate = _run_dry_run(
        aux_dir, gen_root, pilot_out_root, "train", pilot=True
    )
    summary = _run_run(
        aux_dir,
        gen_root,
        pilot_out_root,
        "train",
        accept_cost_estimate_sha256=pilot_estimate["cost_estimate_sha256"],
        pilot=True,
    )
    assert summary["status"] == "COMPLETE"
    assert summary["pilot_mode"] is True
    assert summary["reportability_status"] == "PILOT_DIAGNOSTIC_ONLY"
    assert summary["raw_family_quality_gate"]["status"] == "FAIL"
    assert summary["completed_judge_pairs"] == 6


def test_carry_forward_makes_zero_new_client_calls(workdir):
    aux_dir = workdir / "aux"
    gen_root = workdir / "gen_root"
    gen_dir = gen_root / "esconv_auxiliary_generation_v1_5_pilot_train"
    gen_dir.mkdir(parents=True)
    _write_fixture(aux_dir, gen_dir, "train", n_states=2)
    out_root = workdir / "outputs"
    dry_run_estimate = _run_dry_run(aux_dir, gen_root, out_root, "train", pilot=True)
    _run_run(
        aux_dir,
        gen_root,
        out_root,
        "train",
        accept_cost_estimate_sha256=dry_run_estimate["cost_estimate_sha256"],
        pilot=True,
    )
    original_out_dir = out_root / "esconv_auxiliary_judging_v1_5_pilot_train"
    assert _ConstantJudgeClient.calls == 16  # 2 states x 2 actions x 2 families x 2 types

    # Fresh output directory + fresh identity (pilot flag unchanged here, but
    # a real reprocessing would typically change something in cost_payload;
    # carry-forward correctness only requires a byte-identical call plan).
    new_out_root = workdir / "outputs_v2"
    cf_estimate = _run_dry_run(
        aux_dir,
        gen_root,
        new_out_root,
        "train",
        pilot=True,
        carry_forward_from=original_out_dir,
    )
    assert cf_estimate["historical_carried_forward_calls"] == 16
    assert cf_estimate["remaining_new_logical_calls"] == 0
    assert cf_estimate["maximum_physical_http_attempts"] == 0
    assert cf_estimate["logical_single_attempt_estimated_cost_usd"] == 0
    assert cf_estimate["estimated_cost_usd"] == 0

    summary = _run_run(
        aux_dir,
        gen_root,
        new_out_root,
        "train",
        accept_cost_estimate_sha256=cf_estimate["cost_estimate_sha256"],
        pilot=True,
        carry_forward_from=original_out_dir,
    )
    # _run_run resets the call counter to 0 before invoking main(); zero here
    # means the carry-forward path made no new client calls at all.
    assert _ConstantJudgeClient.calls == 0
    assert summary["carried_forward_physical_calls"] == 16
    assert summary["new_physical_calls"] == 0
    assert summary["completed_judge_pairs"] == 4


def test_one_terminal_provider_output_is_isolated_and_matrix_is_nonreportable(workdir):
    aux_dir = workdir / "aux"
    gen_root = workdir / "gen_root"
    gen_dir = gen_root / "esconv_auxiliary_generation_v1_5_pilot_train"
    gen_dir.mkdir(parents=True)
    _write_fixture(aux_dir, gen_dir, "train", n_states=2)
    out_root = workdir / "outputs"
    estimate = _run_dry_run(aux_dir, gen_root, out_root, "train", pilot=True)
    with pytest.raises(RuntimeError, match="missing logical calls"):
        _run_run(
            aux_dir,
            gen_root,
            out_root,
            "train",
            accept_cost_estimate_sha256=estimate["cost_estimate_sha256"],
            pilot=True,
            client_cls=_OneTerminalOutputThenSuccessClient,
        )
    out_dir = out_root / "esconv_auxiliary_judging_v1_5_pilot_train"
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert _OneTerminalOutputThenSuccessClient.calls == 16
    assert summary["status"] == "INCOMPLETE"
    assert summary["reportability_status"] == "NONREPORTABLE_INCOMPLETE_MATRIX"
    assert summary["new_physical_calls"] == 16
    assert len(summary["isolated_failures"]) == 1
    assert (out_dir / "action_labels.jsonl").read_text(encoding="utf-8") == ""

    continuation_root = workdir / "continuation"
    continuation = _run_dry_run(
        aux_dir,
        gen_root,
        continuation_root,
        "train",
        pilot=True,
        carry_forward_from=out_dir,
    )
    assert continuation["historical_carried_forward_calls"] == 15
    assert continuation["remaining_new_logical_calls"] == 1
    assert continuation["maximum_physical_http_attempts"] == 10
    assert continuation["estimated_cost_usd"] == pytest.approx(
        10 * continuation["logical_single_attempt_estimated_cost_usd"]
    )


def test_five_adjacent_terminal_provider_outputs_trip_circuit_breaker(workdir):
    aux_dir = workdir / "aux"
    gen_root = workdir / "gen_root"
    gen_dir = gen_root / "esconv_auxiliary_generation_v1_5_pilot_train"
    gen_dir.mkdir(parents=True)
    _write_fixture(aux_dir, gen_dir, "train", n_states=2)
    out_root = workdir / "outputs"
    estimate = _run_dry_run(aux_dir, gen_root, out_root, "train", pilot=True)
    with pytest.raises(RuntimeError, match="circuit breaker"):
        _run_run(
            aux_dir,
            gen_root,
            out_root,
            "train",
            accept_cost_estimate_sha256=estimate["cost_estimate_sha256"],
            pilot=True,
            client_cls=_AlwaysTerminalOutputClient,
        )
    assert _AlwaysTerminalOutputClient.calls == 5


def test_terminal_http_400_stops_immediately_without_blind_retry(workdir):
    aux_dir = workdir / "aux"
    gen_root = workdir / "gen_root"
    gen_dir = gen_root / "esconv_auxiliary_generation_v1_5_pilot_train"
    gen_dir.mkdir(parents=True)
    _write_fixture(aux_dir, gen_dir, "train", n_states=2)
    out_root = workdir / "outputs"
    estimate = _run_dry_run(aux_dir, gen_root, out_root, "train", pilot=True)
    with pytest.raises(ProviderRequestError):
        _run_run(
            aux_dir,
            gen_root,
            out_root,
            "train",
            accept_cost_estimate_sha256=estimate["cost_estimate_sha256"],
            pilot=True,
            client_cls=_TerminalRequestClient,
        )
    assert _TerminalRequestClient.calls == 1


def test_output_directory_guard_is_wired_in_before_any_expensive_work(workdir):
    """Proves main() calls resolve_first_unconsumed_output_directory with the
    real, scope-qualified out_dir, before any judge call plan is built. See
    test_output_directory_previously_consumed_is_permanently_protected in
    tests/test_v1_5_latest_protocol_repairs.py for the underlying guard
    function's own correctness, and test_resolve_first_unconsumed_output_
    directory_falls_back_to_a_retry_sibling in the same file for the
    __retryN fallback behavior."""

    aux_dir = workdir / "aux"
    gen_root = workdir / "gen_root"
    gen_dir = gen_root / "esconv_auxiliary_generation_v1_5_full_train"
    gen_dir.mkdir(parents=True)
    _write_fixture(aux_dir, gen_dir, "train", n_states=2)
    module = _load_module(
        "scripts/v1_5/13c_judge_esconv_auxiliary_v1_5.py",
        f"v1_5_esconv_auxiliary_judging_guard_wiring_test_{id(workdir)}",
    )
    calls: list[Path] = []

    def fake_resolver(out_dir, *, config, config_path):
        calls.append(Path(out_dir))
        raise RuntimeError("guard invoked -- stopping before any real work")

    module.resolve_first_unconsumed_output_directory = fake_resolver
    argv = [
        "13c_judge_esconv_auxiliary_v1_5.py",
        "--dry-run",
        "--split",
        "train",
        "--scope",
        "full",
        "--auxiliary-dir",
        str(aux_dir),
        "--generation-root",
        str(gen_root),
        "--out-root",
        str(workdir / "outputs"),
        "--max-api-calls",
        "1000",
        "--max-estimated-usd",
        "5.0",
        "--max-input-tokens-per-call",
        "12000",
    ]
    previous = sys.argv
    try:
        sys.argv = argv
        with pytest.raises(RuntimeError, match="guard invoked"):
            module.main()
    finally:
        sys.argv = previous
    assert len(calls) == 1
    assert calls[0].name == "esconv_auxiliary_judging_v1_5_full_train"


def test_script_never_references_gold_fields_or_action_id_in_authorized_context():
    source = (
        ROOT / "scripts" / "v1_5" / "13c_judge_esconv_auxiliary_v1_5.py"
    ).read_text(encoding="utf-8")
    _, _, code_after_docstring = source.partition('"""')
    _, _, code = code_after_docstring.partition('"""')
    assert "audit_only" not in code
    assert "gold_response" not in code
    assert "gold_strategy" not in code
    # authorized_user_context is always empty -- ESConv has no cross-session
    # memory bank to authorize beyond the current session, already visible in
    # the prompt.
    assert 'authorized_user_context = ""' in code
