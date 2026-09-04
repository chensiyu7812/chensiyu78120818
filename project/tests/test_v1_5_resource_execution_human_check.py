from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v1_5/24dt_aggregate_resource_execution_human_check_v1_5.py"
SPEC = importlib.util.spec_from_file_location("resource_execution_aggregate", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fixture(*, bad: int = 0, misuse: int = 0):
    packet = []
    private = []
    annotations = []
    for index in range(10):
        review_id = f"review_{index}"
        decision = "ignore" if index < 2 else "use"
        response = f"response evidence {index}"
        packet.append(
            {
                "review_item_id": review_id,
                "component": ("RS", "MP", "MS", "ME")[index % 4],
                "resource_subtype": "test",
                "declared_decision": decision,
                "response": response,
            }
        )
        private.append({"review_item_id": review_id, "state_id": f"state_{index}"})
        ok = index >= bad
        annotations.append(
            {
                "protocol": MODULE.PROTOCOL,
                "review_item_id": review_id,
                "declaration_supported": "yes" if ok else "no",
                "functionally_used": "yes" if ok else "no",
                "material_misuse": "yes" if index < misuse else "no",
                "literal_response_excerpt": "[none]" if decision == "ignore" else response,
                "review_notes": "auditable note",
                "annotator_id": "reviewer_1",
            }
        )
    return annotations, packet, private


def test_gate_passes_at_eight_of_ten_with_one_misuse() -> None:
    annotations, packet, private = _fixture(bad=2, misuse=1)
    report, frozen = MODULE.aggregate(
        annotations=annotations, packet=packet, private_key=private
    )
    assert len(frozen) == 10
    assert report["passed"] is True
    assert report["supported_and_functional_n"] == 8
    assert report["material_misuse_n"] == 1


def test_gate_repairs_generator_without_changing_pm() -> None:
    annotations, packet, private = _fixture(bad=3, misuse=0)
    report, _ = MODULE.aggregate(
        annotations=annotations, packet=packet, private_key=private
    )
    assert report["passed"] is False
    assert report["status"] == MODULE.STATUS_FAIL
    assert "do not relabel PM routing" in report["next_action"]
