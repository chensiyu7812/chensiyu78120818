from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts/v1_5/24ag_audit_candidate_admission_data_v1_5.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "candidate_admission_audit", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_marker_detection_is_explicit_and_case_insensitive() -> None:
    module = _load()
    assert module._has_explicit_negative_marker(
        "This was an OUTSIDE TOPIC, not the user's experience."
    )
    assert not module._has_explicit_negative_marker(
        "The user prefers one gentle question at a time."
    )


def test_candidate_data_gate_is_stricter_than_old_point_nine_gate() -> None:
    module = _load()
    assert module.MAX_METADATA_BALANCED_ACCURACY == 0.70
