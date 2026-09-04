"""End-to-end, no-API wiring tests for scripts/v1_5/14_run_esconv_generation_v1_5.py.

Proves the action-first fix on real, small-scale (not 2,112-state) data: the
dry-run call plan equals len(states) * len(legal_actions), never
len(states) * 4, and the script fails closed when the frozen
esconv_generation_contract drifts from what it independently recomputes.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import (
    DialogueTurn,
    MemoryBackendRecord,
    MemorySource,
    RuntimeState,
    SourceCatalog,
)
from metacom_pm.esconv_v1_5 import ESCONV_V1_5_ALLOWED_ACTIONS
from metacom_pm.evidence_filter import EvidenceFilterConfig
from metacom_pm.freeze import create_study_freeze
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import canonical_json, sha256_file, sha256_text, write_json, write_jsonl
from metacom_pm.response_mechanism_contract import build_response_mechanism_contract

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "outputs" / "v1_5_test_fixtures"


@pytest.fixture()
def workdir() -> Path:
    """Isolated directory inside the release root (create_study_freeze's
    data_hashes require every frozen path to resolve inside release_root)."""

    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="esconv_runner_", dir=FIXTURE_ROOT
    ) as directory:
        yield Path(directory)


def _load_module(relpath: str, name: str):
    path = ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_esconv_fixture(workdir: Path, *, n_states: int) -> dict[str, Path]:
    runtime_path = workdir / "runtime_states.jsonl"
    backend_path = workdir / "memory_backend.jsonl"
    policy_path = workdir / "policy_choices.jsonl"

    runtime_rows = []
    backend_rows = []
    policy_rows = []
    for index in range(n_states):
        state = RuntimeState(
            state_id=f"state_esconv_test_{index}",
            card_id=f"card_esconv_test_{index}",
            user_id=f"esconv_dialogue_{index}",
            split="esconv_test",
            semantic_family="esconv_single_session_strategy_routing",
            current_user_text=f"I've been feeling overwhelmed lately, turn {index}.",
            current_session_history=[
                DialogueTurn(role="user", content="Work has been a lot recently."),
                DialogueTurn(
                    role="assistant", content="That does sound like a lot to carry."
                ),
            ],
            current_session_summary="",
            session_index=1,
            inventory={
                source: SourceCatalog(available=False, count=0)
                for source in MemorySource
            },
            allowed_actions=list(ESCONV_V1_5_ALLOWED_ACTIONS),
        )
        backend = MemoryBackendRecord(card_id=state.card_id, items=[])
        runtime_rows.append(state.model_dump(mode="json"))
        backend_rows.append(backend.model_dump(mode="json"))
        policy_rows.append(
            {
                "state_id": state.state_id,
                "card_id": state.card_id,
                "learned_action": "M0+R0",
                "transparent_rule_action": "M0+RS",
                "always_r0_action": "M0+R0",
                "always_rs_action": "M0+RS",
                "esconv_outcome_used_for_choice": False,
            }
        )
    write_jsonl(runtime_path, runtime_rows)
    write_jsonl(backend_path, backend_rows)
    write_jsonl(policy_path, policy_rows)
    return {
        "runtime_states": runtime_path,
        "memory_backend": backend_path,
        "policy_choices": policy_path,
    }


def _build_minimal_freeze(
    workdir: Path,
    *,
    n_states: int = 3,
    perturb_esconv_contract: dict | None = None,
) -> tuple[Path, dict[str, Path]]:
    fixture = _write_esconv_fixture(workdir, n_states=n_states)
    pm_v1_5_config_path = ROOT / "configs" / "pm_v1_5.yaml"
    experiment_config_path = ROOT / "configs" / "experiment.yaml"
    strategy_bank_path = ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl"
    pm_v1_5_config = load_config(pm_v1_5_config_path)
    experiment_config = load_config(experiment_config_path)

    supporter_generation_contract = SupporterGenerationContract.from_config(
        pm_v1_5_config
    )
    generator_endpoint = endpoint_from_config(
        experiment_config, supporter_generation_contract.generator_endpoint
    )
    generator_endpoint_sha256 = sha256_text(
        canonical_json(
            {
                "model": generator_endpoint.model,
                "family": generator_endpoint.family,
                "base_url": generator_endpoint.base_url,
            }
        )
    )
    evidence_filter_config = EvidenceFilterConfig.from_mapping(
        {**pm_v1_5_config["evidence_filter"], "enabled": False}
    )
    external = pm_v1_5_config["external_evaluation"]
    response_mechanism_contract = build_response_mechanism_contract(
        project_root=ROOT,
        supporter_generation_contract=supporter_generation_contract,
        generator_endpoint_sha256=generator_endpoint_sha256,
        strategy_bank_sha256=sha256_file(strategy_bank_path),
        memory_min_score=external.get("memory_min_score"),
        strategy_min_score=external.get("strategy_min_score"),
        strategy_top_k=int(external["strategy_top_k"]),
        evidence_filter_enabled=bool(evidence_filter_config.enabled),
    )

    freeze_module = _load_module(
        "scripts/v1_5_create_freeze.py", "v1_5_esconv_runner_freeze_helper"
    )
    esconv_generation_contract = freeze_module.build_v1_5_esconv_generation_contract(
        response_mechanism_contract=response_mechanism_contract,
        runtime_states_path=fixture["runtime_states"],
        policy_choices_path=fixture["policy_choices"],
        legal_actions=ESCONV_V1_5_ALLOWED_ACTIONS,
    )
    if perturb_esconv_contract:
        esconv_generation_contract = {
            **esconv_generation_contract,
            **perturb_esconv_contract,
        }

    # verify_study_freeze requires non-empty checkpoint_hashes/prompt_hashes
    # sections; content is irrelevant to this test, so use tiny placeholders.
    placeholder_checkpoint = workdir / "placeholder_checkpoint.txt"
    placeholder_checkpoint.write_text("placeholder", encoding="utf-8")
    placeholder_prompt = workdir / "placeholder_prompt.txt"
    placeholder_prompt.write_text("placeholder", encoding="utf-8")

    freeze_path = workdir / "study_freeze.json"
    create_study_freeze(
        release_root=ROOT,
        config_path=experiment_config_path,
        checkpoint_paths=[placeholder_checkpoint],
        data_paths=[
            pm_v1_5_config_path,
            strategy_bank_path,
            fixture["runtime_states"],
            fixture["memory_backend"],
            fixture["policy_choices"],
        ],
        prompt_files=[placeholder_prompt],
        out_path=freeze_path,
        notes={"esconv_generation_contract": esconv_generation_contract},
    )
    return freeze_path, fixture


def _run_dry_run(workdir: Path, freeze_path: Path, fixture: dict[str, Path]) -> dict:
    module = _load_module(
        "scripts/v1_5/14_run_esconv_generation_v1_5.py",
        f"v1_5_esconv_generation_runner_test_{id(workdir)}",
    )
    out_dir = workdir / "out"
    argv = [
        "14_run_esconv_generation_v1_5.py",
        "--dry-run",
        "--freeze",
        str(freeze_path),
        "--esconv-runtime-states",
        str(fixture["runtime_states"]),
        "--esconv-memory-backend",
        str(fixture["memory_backend"]),
        "--esconv-policy-choices",
        str(fixture["policy_choices"]),
        "--out-dir",
        str(out_dir),
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
        module.main()
    finally:
        sys.argv = previous
    import json

    return json.loads((out_dir / "cost_estimate.json").read_text(encoding="utf-8"))


def test_esconv_generation_dry_run_is_action_first_not_condition_first(workdir):
    freeze_path, fixture = _build_minimal_freeze(workdir, n_states=3)
    estimate = _run_dry_run(workdir, freeze_path, fixture)
    # 3 states x 2 legal actions = 6, never 3 x 4 = 12.
    assert estimate["logical_api_calls"] == 6
    assert estimate["action_filter"] == ["M0+R0", "M0+RS"]
    assert estimate["budget_gate"]["status"] == "PASS"


def test_esconv_generation_rejects_stale_legal_actions(workdir):
    freeze_path, fixture = _build_minimal_freeze(
        workdir, n_states=3, perturb_esconv_contract={"legal_actions": ["M0+R0"]}
    )
    with pytest.raises(RuntimeError, match="legal actions"):
        _run_dry_run(workdir, freeze_path, fixture)


def test_esconv_generation_rejects_stale_expected_action_keys(workdir):
    freeze_path, fixture = _build_minimal_freeze(
        workdir,
        n_states=3,
        perturb_esconv_contract={"expected_logical_action_outcomes": 999},
    )
    with pytest.raises(RuntimeError, match="action-first plan"):
        _run_dry_run(workdir, freeze_path, fixture)


def test_esconv_generation_rejects_stale_response_mechanism_contract(workdir):
    freeze_path, fixture = _build_minimal_freeze(
        workdir,
        n_states=3,
        perturb_esconv_contract={
            "response_mechanism_contract": {"contract_sha256": "f" * 64}
        },
    )
    with pytest.raises(RuntimeError, match="response mechanism contract"):
        _run_dry_run(workdir, freeze_path, fixture)


def test_esconv_generation_never_imports_audit_only():
    source = (
        ROOT / "scripts" / "v1_5" / "14_run_esconv_generation_v1_5.py"
    ).read_text(encoding="utf-8")
    # The module docstring documents the audit_only boundary by name; strip
    # it before checking the actual code never references the file.
    _, _, code_after_docstring = source.partition('"""')
    _, _, code = code_after_docstring.partition('"""')
    assert "audit_only" not in code
