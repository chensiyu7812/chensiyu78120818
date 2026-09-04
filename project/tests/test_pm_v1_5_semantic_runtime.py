from __future__ import annotations

import site
import sys

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from metacom_pm.pm_v1_5_semantic import (
    FrozenSemanticEncoderSpec,
    FrozenTransformerSemanticEncoder,
    SEMANTIC_CANARY_TEXTS,
    SemanticEncoderBinding,
    require_recorded_semantic_runtime,
    require_unified_semantic_query_contract,
    semantic_runtime_attestation,
    summarize_semantic_truncation_audits,
)


class _CanaryEncoder:
    spec = FrozenSemanticEncoderSpec(
        model_id="fixture/canary",
        revision="1" * 40,
        snapshot_tree_sha256="2" * 64,
        max_length=64,
        output_dimension=16,
    )
    binding = SemanticEncoderBinding(
        spec_sha256=spec.digest(),
        snapshot_tree_sha256="2" * 64,
        snapshot_file_count=1,
        implementation="transformers-auto-model-cls-float32",
    )
    model = torch.nn.Linear(2, 2, bias=False)

    def encode(self, texts):
        rows = []
        for index, _ in enumerate(texts):
            row = np.arange(1, 17, dtype=float) + index
            rows.append(row / np.linalg.norm(row))
        return np.vstack(rows)


class _CountingTokenizer:
    truncation_side = "right"

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": [101, *range(len(text.split())), 102]}


class _ReversibleTokenizer:
    truncation_side = "right"

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def __call__(self, text, *, add_special_tokens=False, **kwargs):
        del kwargs
        ids = []
        for token in str(text).split():
            if token not in self._token_to_id:
                token_id = 1000 + len(self._token_to_id)
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            ids.append(self._token_to_id[token])
        if add_special_tokens:
            ids = [101, *ids, 102]
        return {"input_ids": ids}

    def decode(self, ids, **kwargs):
        del kwargs
        return " ".join(self._id_to_token[value] for value in ids if value >= 1000)


def test_semantic_runtime_record_is_exact_and_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(site, "ENABLE_USER_SITE", False)
    monkeypatch.setattr(
        site,
        "getusersitepackages",
        lambda: "/__pm_v1_5_disabled_user_site__",
    )
    observed = semantic_runtime_attestation(_CanaryEncoder())
    assert observed.canary_shape == [len(SEMANTIC_CANARY_TEXTS), 16]
    assert observed.user_site_enabled is False
    assert observed.user_site_on_sys_path is False
    config = {"semantic_runtime": observed.model_dump(mode="json")}
    recorded = {
        "status": "PASS",
        "contract": observed.model_dump(mode="json"),
        "contract_sha256": observed.digest(),
    }
    assert require_recorded_semantic_runtime(config, recorded)["status"] == "PASS"
    tampered = {**recorded, "contract_sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="digest mismatch"):
        require_recorded_semantic_runtime(config, tampered)


@pytest.mark.parametrize(
    ("user_site_enabled", "user_site_on_sys_path", "rejected_field"),
    [
        (True, False, "user_site_enabled"),
        (False, True, "user_site_on_sys_path"),
    ],
)
def test_semantic_runtime_attestation_rejects_user_site_contamination(
    monkeypatch,
    user_site_enabled: bool,
    user_site_on_sys_path: bool,
    rejected_field: str,
) -> None:
    user_site = (
        sys.path[0]
        if user_site_on_sys_path
        else "/__pm_v1_5_disabled_user_site__"
    )
    monkeypatch.setattr(site, "ENABLE_USER_SITE", user_site_enabled)
    monkeypatch.setattr(site, "getusersitepackages", lambda: user_site)
    with pytest.raises(ValidationError, match=rejected_field):
        semantic_runtime_attestation(_CanaryEncoder())


def test_unified_semantic_query_config_fails_closed() -> None:
    config = {
        "step0_observation": {
            "protocol": (
                "pm-v1.5-step0-semantic-source-observation-v3-unified-state-query"
            ),
            "stage": "pre_item_retrieval",
            "full_state_semantic_query": (
                "pm-v1.5-unified-step0-state-semantic-query-v1"
            ),
        },
        "semantic_diagnostics": {
            "unified_step0_state_semantic_query_required": True,
            "section_allocation_required_for_every_state": True,
        },
    }
    assert require_unified_semantic_query_contract(config)[
        "semantic_query_protocol"
    ].endswith("v1")
    stale = {
        **config,
        "step0_observation": {
            **config["step0_observation"],
            "full_state_semantic_query": "legacy-unbounded-query",
        },
    }
    with pytest.raises(RuntimeError, match="unified section-aware"):
        require_unified_semantic_query_contract(stale)


def test_tokenization_telemetry_reports_loss_without_text_or_token_ids() -> None:
    encoder = _CanaryEncoder()
    encoder.tokenizer = _CountingTokenizer()
    telemetry = FrozenTransformerSemanticEncoder.tokenization_telemetry(
        encoder,
        ["short input", " ".join(["long"] * 100)],
        view_names=["current_user_text", "visible_dialogue_state"],
    )
    assert telemetry["views"]["current_user_text"]["truncated"] is False
    assert telemetry["views"]["visible_dialogue_state"]["truncated"] is True
    assert telemetry["views"]["visible_dialogue_state"]["truncated_token_count"] == 38
    assert "input_ids" not in str(telemetry)
    assert "long long" not in str(telemetry)

    summary = summarize_semantic_truncation_audits(
        [{"tokenization": telemetry}]
    )
    assert summary["current_user_text_complete"] is True
    assert summary["views"]["visible_dialogue_state"]["truncation_rate"] == 1.0


def test_section_aware_visible_state_preserves_recent_history_without_implicit_loss() -> None:
    encoder = _CanaryEncoder()
    encoder.tokenizer = _ReversibleTokenizer()
    state_text, allocation = (
        FrozenTransformerSemanticEncoder.assemble_visible_dialogue_state(
            encoder,
            current_user_text=" ".join(f"current{index}" for index in range(70)),
            current_session_history=[
                {
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": " ".join(
                        f"history{index}_{token}" for token in range(20)
                    ),
                }
                for index in range(10)
            ],
            current_session_summary=" ".join(
                f"summary{index}" for index in range(80)
            ),
        )
    )
    encoded = encoder.tokenizer(state_text, add_special_tokens=True)["input_ids"]
    assert len(encoded) <= encoder.spec.max_length
    assert allocation["protocol"].endswith("v2-section-aware")
    assert allocation["implicit_full_state_truncation"] == "forbidden"
    assert allocation["sections"]["current_user"]["retained_token_count"] == 32
    assert allocation["sections"]["session_summary"]["retained_token_count"] == 16
    assert allocation["sections"]["recent_dialogue"]["dropped_token_count"] > 0
    assert "history9_19" in state_text
    assert "history0_0" not in state_text
    assert "history9_19" not in str(allocation)
