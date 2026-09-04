from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not (
        ROOT
        / "outputs/pm_v1_5b_final_external_panel_v2/panel_freeze_report.json"
    ).is_file(),
    reason="corrected external-split artifact bundle is not included in a clean checkout",
)


def _module():
    path = ROOT / "scripts/v1_5/24eu_freeze_corrected_external_split_v1_5.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_user_grain_external_split_preserves_two_untouched_evoemo_sets() -> None:
    report = _module().build_split(root=ROOT)
    assert report["status"] == "PASS_CORRECTED_EXTERNAL_USER_SPLIT_FROZEN"
    assert report["partitions"]["evoemo_diagnostic"]["states"] == 66
    assert report["partitions"]["evoemo_qualification"]["states"] == 60
    assert report["partitions"]["evoemo_lockbox"]["states"] == 78
    assert report["partitions"]["evoemo_qualification"]["saved_v2_calls"] == 0
    assert report["partitions"]["evoemo_lockbox"]["saved_v2_calls"] == 0
    assert report["checks"]["every_p1_p6_state_is_development_contaminated"]
    assert report["checks"]["all_esconv_panels_were_exposed_under_objectively_broken_treatment"]
    assert report["v2_diagnostic_closeout"]["empty_requested_resource_counts_in_v2_plan"]["RS"] > 0


def test_current_broken_v2_plan_cannot_be_rematerialized_as_corrected(
    tmp_path: Path,
) -> None:
    module = _module()
    module.build_split(root=ROOT)
    broken = ROOT / "outputs/pm_v1_5b_final_system_generation_v2_candidate/call_plan_private.jsonl"
    with pytest.raises(RuntimeError, match="empty requested resources"):
        module.validate_and_partition_corrected_plan(
            corrected_plan_path=broken,
            output_dir=tmp_path,
            root=ROOT,
        )


def test_materialization_firewall_accepts_only_full_nonempty_resource_plan(
    tmp_path: Path,
) -> None:
    module = _module()
    module.build_split(root=ROOT)
    source = ROOT / "outputs/pm_v1_5b_final_system_generation_v2_candidate/call_plan_private.jsonl"
    rows = _read_jsonl(source)
    replacement = "Use one grounded, low-burden support move that fits the visible exchange."
    for row in rows:
        if "RS" not in row["requested_components"]:
            continue
        row["selected_resources_private"]["RS"] = replacement
        row["messages"][-1]["content"] += "\n\nCorrected strategy resource:\n" + replacement
        row["messages_sha256"] = module.sha256_text(module.canonical_json(row["messages"]))
    corrected = tmp_path / "corrected.jsonl"
    corrected.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = module.validate_and_partition_corrected_plan(
        corrected_plan_path=corrected,
        output_dir=tmp_path / "partitioned",
        root=ROOT,
    )
    assert report["status"] == "PASS_CORRECTED_CALL_PLAN_MATERIALIZED_AND_PARTITIONED"
    assert report["checks"]["all_requested_resource_bodies_nonempty"]
    assert report["checks"]["all_requested_resource_bodies_present_in_messages"]
    assert sum(report["partition_calls"].values()) == len(rows)
