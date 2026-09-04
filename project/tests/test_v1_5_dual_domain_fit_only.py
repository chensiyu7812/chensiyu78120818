from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from metacom_pm.pm_v2_contracts import PMV2Split


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = (
    ROOT / "scripts" / "v1_5" / "21a_preflight_dual_domain_training_v1_5.py"
)
TRAINER = (
    ROOT / "scripts" / "v1_5" / "22a_train_pm_v2_dual_domain_v1_5.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _states():
    return [
        SimpleNamespace(
            state_id="train",
            split=PMV2Split.TRAIN,
            allowed_actions=["M0+R0"],
        ),
        SimpleNamespace(
            state_id="internal_b",
            split=PMV2Split.INTERNAL_TEST,
            allowed_actions=["M0+RS", "M0+R0"],
        ),
        SimpleNamespace(
            state_id="internal_a",
            split=PMV2Split.INTERNAL_TEST,
            allowed_actions=["M0+R0"],
        ),
    ]


def test_trainer_accepts_exact_outcome_blind_commitment(tmp_path):
    preflight = _load(PREFLIGHT, "fit_only_preflight")
    trainer = _load(TRAINER, "fit_only_trainer")
    path = tmp_path / "commitment.json"
    written = preflight._write_internal_state_commitment(
        path, _states(), domain="example"
    )
    required = trainer._require_internal_state_commitment(
        path, reversed(_states()), domain="example"
    )
    assert required == written
    assert required["internal_label_values_deserialized"] is False


def test_trainer_rejects_state_or_action_drift(tmp_path):
    preflight = _load(PREFLIGHT, "fit_only_preflight_drift")
    trainer = _load(TRAINER, "fit_only_trainer_drift")
    path = tmp_path / "commitment.json"
    preflight._write_internal_state_commitment(
        path, _states(), domain="example"
    )
    drifted = _states()
    drifted[-1].allowed_actions.append("M0+RS")
    with pytest.raises(RuntimeError, match="commitment is invalid"):
        trainer._require_internal_state_commitment(
            path, drifted, domain="example"
        )


def test_selector_inference_cache_reuses_state_predictions_and_restores_methods():
    trainer = _load(TRAINER, "fit_only_trainer_cache")

    class FakeModel:
        def __init__(self):
            self.bundle_calls = 0
            self.routing_calls = 0

        def _prediction_bundle(self, state):
            self.bundle_calls += 1
            return ("bundle", state.state_id)

        def _routing_scores(self, state):
            self.routing_calls += 1
            return ("routing", state.state_id)

    model = FakeModel()
    state = SimpleNamespace(state_id="same")
    with trainer._cache_selection_inference(model) as cache:
        assert model._prediction_bundle(state) == ("bundle", "same")
        assert model._prediction_bundle(state) == ("bundle", "same")
        assert model._routing_scores(state) == ("routing", "same")
        assert model._routing_scores(state) == ("routing", "same")
        assert len(cache["prediction_bundle_cache"]) == 1
        assert len(cache["routing_score_cache"]) == 1
    assert model.bundle_calls == 1
    assert model.routing_calls == 1
    assert model._prediction_bundle(state) == ("bundle", "same")
    assert model.bundle_calls == 2


def test_algorithm_selection_checkpoint_is_content_bound(tmp_path):
    trainer = _load(TRAINER, "fit_only_trainer_checkpoint")
    inputs = {}
    for name in (
        "preflight",
        "states",
        "long_labels",
        "aux_labels",
        "config",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(f'{{"name":"{name}"}}\\n', encoding="utf-8")
        inputs[name] = path
    args = SimpleNamespace(
        dual_preflight_report=inputs["preflight"],
        longitudinal_states=inputs["states"],
        longitudinal_train_calibration_labels=inputs["long_labels"],
        auxiliary_train_labels=inputs["aux_labels"],
        run_identity="identity",
        seed=1701,
    )
    binding = trainer._algorithm_selection_checkpoint_binding(
        args=args, config_path=inputs["config"]
    )
    core = {
        "binding": binding,
        "selected_algorithm": "absolute_outcome_factorized_hgb",
        "algorithm_selection": {
            "selection_data_role": "train_only",
            "selected_algorithm": "absolute_outcome_factorized_hgb",
        },
    }
    record = {
        **core,
        "binding_sha256": trainer.sha256_text(
            trainer.canonical_json(core)
        ),
    }
    path = tmp_path / "checkpoint.json"
    path.write_text(
        json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
    )
    selected, report = trainer._require_algorithm_selection_checkpoint(
        path, expected_binding=binding
    )
    assert selected == "absolute_outcome_factorized_hgb"
    assert report["selection_data_role"] == "train_only"
    inputs["aux_labels"].write_text('{"drifted":true}\n', encoding="utf-8")
    drifted_binding = trainer._algorithm_selection_checkpoint_binding(
        args=args, config_path=inputs["config"]
    )
    with pytest.raises(RuntimeError, match="input binding drifted"):
        trainer._require_algorithm_selection_checkpoint(
            path, expected_binding=drifted_binding
        )


def test_trainer_wires_pre_internal_viability_before_consumption():
    source = TRAINER.read_text(encoding="utf-8")
    viability = source.index("pre_internal_action_viability =")
    rejection = source.index(
        "dual-domain candidate failed the pre-internal calibration"
    )
    consumption = source.index("    starts = {")
    assert viability < rejection < consumption
    assert "begin_internal_test_consumption(" in source[consumption:]
    assert '"internal_test_outcomes_opened": False' in source
    assert "CANDIDATE_NOT_SUPPORTED_BEFORE_INTERNAL_TEST" in source
