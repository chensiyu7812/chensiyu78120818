import importlib.util
from pathlib import Path


def _module():
    path = Path("scripts/v1_5/25s_prepare_final_h1r_review_v1_5.py")
    spec = importlib.util.spec_from_file_location("h1r_packet", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_h1r_overlap_is_24_states_8_per_split_and_all_actions() -> None:
    module = _module()
    bindings = []
    for action_index in range(16):
        for split in module.SPLITS:
            for repeat in range(2):
                bindings.append(
                    {
                        "blind_state_id": f"b_{action_index}_{split}_{repeat}",
                        "state_id": f"s_{action_index}_{split}_{repeat}",
                        "split": split,
                        "construction_action_private_not_gold": f"a_{action_index}",
                    }
                )
    selected = module._overlap_ids(bindings)
    assert len(selected) == 24
    assert {
        row["construction_action_private_not_gold"]
        for row in bindings
        if row["blind_state_id"] in selected
    } == {f"a_{index}" for index in range(16)}
    for split in module.SPLITS:
        assert sum(
            row["split"] == split and row["blind_state_id"] in selected
            for row in bindings
        ) == 8
