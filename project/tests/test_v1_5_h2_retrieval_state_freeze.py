from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v1_5/24an_freeze_h2_retrieval_states_v1_5.py"


def _module():
    spec = importlib.util.spec_from_file_location("h2_state_freeze", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strata_keep_hard_off_separate_from_positive_cues() -> None:
    module = _module()
    hard = {
        "pure_phatic": False,
        "explicit_stop": False,
        "expanded_routine_closing": False,
        "ordinary_rag_hard_off": True,
        "effort_or_progress_visible": True,
        "advice_welcome": True,
        "uncertainty_or_multi_concern": True,
        "emotion_visible": True,
    }
    assert module._strata(hard) == ["ordinary_hard_off"]
    hard["pure_phatic"] = True
    assert module._strata(hard) == ["phatic_stop_closing"]


def test_selection_uses_distinct_dialogues() -> None:
    module = _module()
    pools = {}
    targets = {"a": 2, "b": 2}
    for stratum in targets:
        pools[stratum] = [
            {
                "source_dialogue_id": f"d{index}",
                "visible_dialogue_sha256": f"h{index}",
                "problem_type": f"p{index}",
            }
            for index in range(6)
        ]
    selected = module._select(pools, targets=targets)
    dialogues = [row["source_dialogue_id"] for row in selected]
    assert len(dialogues) == len(set(dialogues)) == 4
