from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BLUEPRINT = (
    ROOT / "data/pm_v1_5_v3_effect_blueprint_v1/private/construction_blueprint.jsonl"
)


@pytest.mark.skipif(
    not BLUEPRINT.is_file(),
    reason="private V3 construction blueprint is not included in a clean checkout",
)
def test_current_v3_me_construction_support_and_observer_gap(tmp_path: Path) -> None:
    import importlib.util

    script = (
            ROOT
        / "scripts/v1_5/61_audit_v5_3_me_construction_support_v1_5.py"
    )
    spec = importlib.util.spec_from_file_location("me_support_audit", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run(tmp_path)

    assert report["counts"]["me_effect_rows"] == 128
    assert report["counts"]["effect_rank1_compiler_valid"] == 128
    assert report["counts"]["effect_blueprint_action_ready"] == 128
    # This is the open problem the audit is meant to preserve, not hide:
    # current natural "open to one optional idea" surfaces are not detected
    # by the old explicit-advice regex.
    assert report["counts"]["effect_legacy_observer_action_invitation"] == 0
    assert report["counts"]["effect_v53_observer_action_invitation"] == 128
    assert report["interpretation"][
        "current_v3_effect_construction_has_me_action_result_support"
    ] is True
    assert report["interpretation"][
        "legacy_action_invitation_observer_failed_development_replay"
    ] is True
    assert report["interpretation"][
        "v53_action_readiness_observer_passed_development_replay"
    ] is True
    assert report["interpretation"][
        "v53_action_readiness_observer_fresh_qualified"
    ] is False
    assert report["interpretation"]["construction_intent_is_worth_opening_gold"] is False
