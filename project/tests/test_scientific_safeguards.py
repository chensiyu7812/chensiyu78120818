from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from metacom_pm.artifacts import (
    create_artifact_attestation,
    verify_artifact_attestation,
)
from metacom_pm.config import confirmatory_model_independence
from metacom_pm.evoemo import (
    NEUTRAL_INITIAL_GREETING,
    _fixed_context_before_turn,
    make_evo_runtime_state,
)
from metacom_pm.features import FeatureBuilder
from metacom_pm.io import iter_jsonl, sha256_file
from metacom_pm.policies import LearnedPMPolicy
from metacom_pm.prompts import (
    fixed_context_pair_messages,
    memory_use_messages,
    observation_usage_messages,
    official_dialogue_score_messages,
)
from metacom_pm.release import run_release_preflight
from metacom_pm.sweep import load_states
from metacom_pm.variance import select_variance_card_ids

ROOT = Path(__file__).resolve().parents[1]


class StubPM:
    feature_mode = "full"
    last_ood_report = {"severe_scalar_ood": False, "severe_catalog_ood": False}

    def __init__(self, scores):
        self.scores = scores

    def score_actions(self, state, action_ids):
        return {action_id: dict(self.scores[action_id]) for action_id in action_ids}


def _full_state():
    states = load_states(ROOT / "data/synthetic/runtime_states.jsonl")
    return next(state for state in states.values() if len(state.allowed_actions) == 16)


def _base_score(**updates):
    value = {
        "response_score": 0.90,
        "misuse_risk": 0.10,
        "memory_omission_risk": 0.90,
        "strategy_decision_risk": 0.10,
        "memory_decision_quality": 0.50,
        "strategy_decision_quality": 0.50,
    }
    value.update(updates)
    return value


def test_no_memory_cannot_win_on_cost_when_omission_is_unsafe():
    state = _full_state()
    scores = {action: _base_score() for action in state.allowed_actions}
    scores["M0+R0"] = _base_score(response_score=0.99, memory_omission_risk=1.0)
    scores["ME+R0"] = _base_score(
        response_score=0.82,
        memory_omission_risk=0.10,
        memory_decision_quality=0.90,
    )
    policy = LearnedPMPolicy(
        StubPM(scores),
        epsilon=0.05,
        tau_misuse=0.35,
        tau_omission=0.35,
        tau_strategy=0.35,
        allow_constraint_fallback=False,
    )
    assert policy.choose(state) == "ME+R0"
    assert policy.last_decision_report["constraint_fallback_used"] is False
    assert "M0+R0" not in policy.last_decision_report["safe_actions"]


def test_confirmatory_policy_fails_closed_when_every_action_is_unsafe():
    state = _full_state()
    scores = {
        action: _base_score(
            misuse_risk=0.9,
            memory_omission_risk=0.9,
            strategy_decision_risk=0.9,
        )
        for action in state.allowed_actions
    }
    policy = LearnedPMPolicy(
        StubPM(scores), epsilon=0.05, allow_constraint_fallback=False
    )
    with pytest.raises(RuntimeError, match="No action satisfies"):
        policy.choose(state)


def test_m2_prompt_contains_complete_timeline_and_selected_marker():
    state = _full_state()
    backend = {
        row["card_id"]: row
        for row in iter_jsonl(ROOT / "data/synthetic/memory_backend.jsonl")
    }[state.card_id]
    from metacom_pm.contracts import MemoryItem

    items = [MemoryItem.model_validate(x) for x in backend["items"]]
    selected = items[:1]
    content = memory_use_messages(
        state, selected, "I remember that update.", all_items=items
    )[-1]["content"]
    assert "Complete authorized memory timeline" in content
    assert '"selected": true' in content
    if len(items) > 1:
        assert '"selected": false' in content
    assert all(item.memory_id in content for item in items)


def test_grounded_external_judges_receive_truth_but_not_policy_identity():
    context = {
        "user_profile": {"name": "Alex"},
        "past_session_timeline": [{"summary": "A newer update"}],
        "current_topic": {"topic": "work"},
        "evaluator_only": True,
    }
    dialogue = [{"role": "seeker", "content": "I feel uncertain."}]
    prompt = official_dialogue_score_messages(dialogue, context)[-1]["content"]
    assert "authorized_ground_truth" in prompt
    assert "A newer update" in prompt
    assert "selected_memory" not in prompt
    assert "action_id" not in prompt


def test_observation_attribution_sees_current_session_history():
    messages = observation_usage_messages(
        "It is still hard.",
        "You said earlier today that the move was exhausting.",
        "The move was exhausting.",
        current_session_history=[
            {"role": "seeker", "content": "The move was exhausting."}
        ],
    )
    assert "current_session_before_response" in messages[-1]["content"]
    assert "The move was exhausting" in messages[-1]["content"]


def test_scalar_ood_is_not_diluted_by_catalog_dimensions():
    state = _full_state()
    builder = FeatureBuilder(mode="metadata_only").fit([state])
    shifted = state.model_copy(deep=True)
    shifted.inventory[next(iter(shifted.inventory))] = shifted.inventory[
        next(iter(shifted.inventory))
    ].model_copy(update={"count": 1000, "estimated_tokens": 100000})
    report = builder.ood_report([(shifted, "M0+R0")])
    assert report["severe_scalar_ood"] is True
    assert report["recommendation"] == "ABORT_OR_USE_OOD_BASELINE"


def test_artifact_attestation_detects_posthoc_tampering(tmp_path):
    source = tmp_path / "source.txt"
    output = tmp_path / "output.jsonl"
    source.write_text("source", encoding="utf-8")
    output.write_text('{"x":1}\n', encoding="utf-8")
    attestation = tmp_path / "attestation.json"
    create_artifact_attestation(
        attestation,
        stage="unit_test",
        inputs={"source": source},
        outputs={"output": (output, True)},
        parameters={"seed": 1},
    )
    assert verify_artifact_attestation(
        attestation, required_stage="unit_test"
    )["ok"]
    output.write_text('{"x":2}\n', encoding="utf-8")
    result = verify_artifact_attestation(attestation, required_stage="unit_test")
    assert result["ok"] is False
    assert any("hash mismatch" in error for error in result["errors"])


def test_model_role_gate_rejects_self_play_and_same_judge_family():
    endpoint = lambda family: {
        "base_url": "https://example.invalid",
        "model": family,
        "family": family,
        "api_key_env": "X",
    }
    config = {
        "endpoints": {
            "generator": endpoint("same"),
            "training_judge": endpoint("judge"),
            "final_judge": endpoint("judge"),
            "seeker": endpoint("same"),
            "seeker_alt": endpoint("same"),
        },
        "protocol": {"confirmatory_seeker_endpoints": ["seeker", "seeker_alt"]},
    }
    result = confirmatory_model_independence(config)
    assert result["ok"] is False
    assert len(result["errors"]) >= 3


def test_release_preflight_includes_model_family_independence(tmp_path):
    manifest_before = sha256_file(ROOT / "release_manifest.json")
    report = run_release_preflight(
        ROOT,
        tmp_path / "preflight.json",
        run_tests=False,
        manifest_out_path=tmp_path / "release_manifest.json",
    )
    assert "model_family_independence" in report["checks"]
    assert report["checks"]["model_family_independence"]["passed"] is True
    assert report["confirmatory_checks"]["model_family_independence"] is True
    assert report["status"] == "API_PILOT_READY"
    assert report["confirmatory_ready"] is False
    assert report["freeze_verification"]["status"] == "STALE_HISTORICAL_FREEZE"
    assert report["freeze_verification"]["blocking_scope"] == "confirmatory_only"
    assert sha256_file(ROOT / "release_manifest.json") == manifest_before


def test_esconv_confirmatory_sweep_forbids_max_cards():
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/13_run_esconv_sweep.py"),
            "--max-cards",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env={"PYTHONPATH": str(ROOT / "src")},
    )
    assert proc.returncode != 0
    assert "--max-cards is forbidden" in (proc.stderr + proc.stdout)


def test_fixed_track_exogenous_id_is_shared_but_runtime_history_is_not():
    user = {"id": "u1"}
    topic = {"idx": 2}
    conversation_a = [
        {"role": "supporter", "content": "Hello"},
        {"role": "seeker", "content": "First disclosure"},
        {"role": "supporter", "content": "Response A"},
    ]
    conversation_b = [
        {"role": "supporter", "content": "Hello"},
        {"role": "seeker", "content": "First disclosure"},
        {"role": "supporter", "content": "Response B"},
    ]
    a = make_evo_runtime_state(
        user, topic, conversation_a, "Same fixed next turn", [], 2, "pm", track_id="track_x"
    )
    b = make_evo_runtime_state(
        user, topic, conversation_b, "Same fixed next turn", [], 2, "baseline", track_id="track_x"
    )
    assert a.provenance["exogenous_state_id"] == b.provenance["exogenous_state_id"]
    assert a.state_id != b.state_id
    assert a.provenance["runtime_state_includes_prior_treatment_history"] is True



def test_fixed_open_loop_uses_identical_complete_state_across_conditions():
    track = {
        "initial_greeting": NEUTRAL_INITIAL_GREETING,
        "seeker_turns": ["First disclosure", "Same fixed next turn"],
    }
    context = _fixed_context_before_turn(track, 2)
    assert context == [
        {"role": "supporter", "content": NEUTRAL_INITIAL_GREETING},
        {"role": "seeker", "content": "First disclosure"},
        {
            "role": "supporter",
            "content": "I'm listening. What feels most important to share right now?",
        },
    ]
    user = {"id": "u1"}
    topic = {"idx": 2}
    a = make_evo_runtime_state(
        user, topic, context, "Same fixed next turn", [], 2, "pm",
        track_id="track_x", fixed_open_loop=True,
    )
    b = make_evo_runtime_state(
        user, topic, context, "Same fixed next turn", [], 2, "baseline",
        track_id="track_x", fixed_open_loop=True,
    )
    assert a.state_id == b.state_id
    assert a.card_id == b.card_id
    assert a.provenance["exogenous_state_id"] == b.provenance["exogenous_state_id"]
    assert a.provenance["runtime_state_includes_prior_treatment_history"] is False
    assert a.provenance["condition_label_not_present_in_pm_state"] is True


def test_fixed_context_pair_judge_rejects_nonidentical_inputs():
    left = [{
        "turn_index": 1,
        "context_before_turn": [
            {"role": "supporter", "content": NEUTRAL_INITIAL_GREETING}
        ],
        "seeker_message": "I feel uncertain.",
        "supporter_message": "Response A",
    }]
    right = [{
        **left[0],
        "context_before_turn": [
            {"role": "supporter", "content": "A treatment-dependent prompt"}
        ],
        "supporter_message": "Response B",
    }]
    with pytest.raises(ValueError, match="inputs differ"):
        fixed_context_pair_messages(left, right, {"evaluator_only": True})

def test_variance_sampling_counts_independent_states_not_inventory_siblings():
    runtime = ROOT / "data/synthetic/runtime_states.jsonl"
    card_ids = select_variance_card_ids(runtime, n_states=8)
    states = load_states(runtime)
    assert len(card_ids) == 8
    assert len({states[x].state_id for x in card_ids}) == 8
    assert all(len(states[x].allowed_actions) == 16 for x in card_ids)
