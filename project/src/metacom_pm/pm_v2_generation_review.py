from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

from .artifacts import create_artifact_attestation, require_artifact_attestation
from .io import canonical_json, read_json, sha256_file, sha256_text, write_json
from .pm_v2_data import GENERATION_CASE_FIELDS, GeneratedUserBundle
from .pm_v2_generation_pilot import require_generation_compatibility_attestation


GENERATION_REVIEW_STAGE = "pm_v2_generation_pilot_semantic_review"
GENERATION_REVIEW_PROTOCOL = (
    "pm-v2-generation-pilot-semantic-review-v2-readable-two-independent-"
    "all-affirmative"
)
MINIMUM_ANNOTATORS = 2

PROTECTED_FIELDS = (
    "item_id",
    "case_id",
    "candidate_semantic_family",
    "candidate_regime",
    "candidate_needed_memory_sources_json",
    "session_index",
    "memory_age_sessions_json",
    "current_user_text",
    "dialogue_before_current_json",
    "session_summary",
    "authorized_user_context",
    "coverage_rationale",
    "profile_memories_json",
    "summary_memories_json",
    "event_memories_json",
    "surface_origin",
    "provider_surface_draft_json",
    "evidence_blueprint_sha256",
)
RATING_FIELDS = (
    "semantic_family_match",
    "regime_match",
    "needed_memory_sources_match",
    "memory_item_utility_match",
    "source_type_match",
    "dialogue_temporal_order_match",
    "memory_age_design_match",
    "strategy_need_match",
)
PACKET_FIELDS = (*PROTECTED_FIELDS, *RATING_FIELDS, "annotator_id", "notes")
SIMPLE_REVIEW_FIELDS = ("item_id", *RATING_FIELDS, "annotator_id", "notes")


def _pilot_inputs(
    pilot_attestation_path: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], GeneratedUserBundle, Path]:
    attestation_path = Path(pilot_attestation_path).resolve()
    if not attestation_path.is_file():
        raise RuntimeError(
            f"generation semantic review requires pilot attestation: {attestation_path}"
        )
    attestation = read_json(attestation_path)
    contract = (attestation.get("parameters") or {}).get(
        "compatibility_contract"
    )
    if not isinstance(contract, dict):
        raise RuntimeError("pilot attestation lacks its compatibility contract")
    verification = require_generation_compatibility_attestation(
        attestation_path,
        expected_contract=contract,
    )
    bundle_path = attestation_path.parent / "pilot_bundle.json"
    bundle = GeneratedUserBundle.model_validate(read_json(bundle_path))
    return attestation_path, contract, verification, bundle, bundle_path


def _packet_rows(bundle: GeneratedUserBundle) -> list[dict[str, str]]:
    provider_draft = bundle.provenance.get("provider_draft")
    if not isinstance(provider_draft, dict):
        raise RuntimeError("pilot bundle lacks provider_draft for semantic review")
    selection = bundle.provenance.get("surface_selection")
    if not isinstance(selection, dict):
        raise RuntimeError("pilot bundle lacks surface-selection provenance")
    fallback_cases = set(selection.get("fallback_cases") or [])
    evidence_blueprint_sha256 = str(
        bundle.provenance.get("evidence_blueprint_sha256") or ""
    )
    if not evidence_blueprint_sha256:
        raise RuntimeError("pilot bundle lacks deterministic evidence blueprint")
    rows: list[dict[str, str]] = []
    if len(bundle.cases) != len(GENERATION_CASE_FIELDS):
        raise RuntimeError("generation review requires the exact nine-case pilot")
    for index, ((case_field, expected_regime), case) in enumerate(
        zip(GENERATION_CASE_FIELDS, bundle.cases, strict=True),
        1,
    ):
        if case.regime is not expected_regime:
            raise RuntimeError("pilot case order differs from frozen generation slots")
        row = {
            "item_id": f"generation_case_{index:02d}_{case_field}",
            "case_id": case.case_id,
            "candidate_semantic_family": case.semantic_family,
            "candidate_regime": case.regime.value,
            "candidate_needed_memory_sources_json": canonical_json(
                [source.value for source in case.needed_memory_sources]
            ),
            "session_index": str(case.session_index),
            "memory_age_sessions_json": canonical_json(
                {
                    "MP": [
                        case.session_index - item.created_session
                        for item in case.profile_memories
                    ],
                    "MS": [
                        case.session_index - item.created_session
                        for item in case.summary_memories
                    ],
                    "ME": [
                        case.session_index - item.created_session
                        for item in case.event_memories
                    ],
                }
            ),
            "current_user_text": case.current_user_text,
            "dialogue_before_current_json": canonical_json(
                [turn.model_dump(mode="json") for turn in case.recent_dialogue]
            ),
            "session_summary": case.session_summary,
            "authorized_user_context": case.authorized_user_context,
            "coverage_rationale": case.coverage_rationale,
            "profile_memories_json": canonical_json(
                [item.model_dump(mode="json") for item in case.profile_memories]
            ),
            "summary_memories_json": canonical_json(
                [item.model_dump(mode="json") for item in case.summary_memories]
            ),
            "event_memories_json": canonical_json(
                [item.model_dump(mode="json") for item in case.event_memories]
            ),
            "surface_origin": (
                "deterministic_fallback"
                if case_field in fallback_cases
                else "provider_surface"
            ),
            "provider_surface_draft_json": canonical_json(
                {
                    field: provider_draft[case_field][field]
                    for field in (
                        "current_user_text",
                        "dialogue_before_current",
                        "session_summary",
                        "authorized_user_context",
                        "coverage_rationale",
                    )
                }
            ),
            "evidence_blueprint_sha256": evidence_blueprint_sha256,
            **{field: "" for field in RATING_FIELDS},
            "annotator_id": "",
            "notes": "",
        }
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PACKET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_simple_review_csv(
    path: Path,
    rows: Sequence[dict[str, str]],
    *,
    annotator_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SIMPLE_REVIEW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "item_id": row["item_id"],
                    **{field: "" for field in RATING_FIELDS},
                    "annotator_id": annotator_id,
                    "notes": "",
                }
            )


def _manual() -> str:
    dimensions = "\n".join(f"- `{field}`" for field in RATING_FIELDS)
    return f"""# PM V2 generation pilot 双人语义审查

协议：`{GENERATION_REVIEW_PROTOCOL}`

目的：在任何全量 synthetic-generation API 调用前，人工核验单次 pilot 的九个
case 是否真的符合候选语义族、resource-need regime、item utility、来源类型与
时间顺序。结构校验 PASS 不能替代本审查。

Provider 只负责自然语言 surface。MP/MS/ME evidence 与 helpful/irrelevant/harmful
标签、coverage rationale 与 evidence 来自冻结的 deterministic blueprint；provider
输出中的 rationale 和 memory 占位字段被明确丢弃。`surface_origin=deterministic_fallback` 表示该 case 的 provider surface
越界后由本地模板替换，仍须按相同标准审查，不能因为来源是 fallback 自动给 1。

操作：审查者 A 阅读中文材料并填写精简的 `reviewer_a.csv`，审查者 B 独立阅读
同一材料并填写 `reviewer_b.csv`。两份表都只有九行；只填写八个评分列和可选
`notes`。所有评分只能填 `0`（不符合或无法确定）或 `1`（符合）。两份表中的
`annotator_id` 已预填为不同标识，不要交换。两人完成前不得讨论答案。

评分列：

{dimensions}

判定是 fail-closed：九个 case、全部评分维度、每名审查者都必须为 1，且同一
case 至少有两名独立审查者。任一 0、缺失、保护列改动或审查者不足都会 FAIL，
并继续禁止全量 API。`strategy_need_match` 对非 strategy regime 也要核验其
strategy 条件没有反向暗示。

`memory_age_design_match` 必须结合 `session_index`、各 item 的
`created_session` 与 `memory_age_sessions_json` 判断：所有 memory 必须严格早于
当前 session，三类 source 的年龄应有变化，但年龄/轮次本身不得充当 regime 或
item utility 标签。
"""


REVIEW_QUESTIONS_ZH = {
    "semantic_family_match": (
        "话题是否匹配：当前话语、前文和 summary 是否只围绕候选 semantic family，"
        "没有混入其他主题？"
    ),
    "regime_match": (
        "regime 是否成立：结合当前对话和所有 evidence，这一例是否真的符合候选"
        "资源情形？"
    ),
    "needed_memory_sources_match": (
        "memory source 是否必要且充分：列出的 needed sources 是否确实增加有用信息，"
        "未列出的 source 是否不应使用？"
    ),
    "memory_item_utility_match": (
        "item utility 是否正确：helpful 真有帮助、irrelevant 确实无关、harmful 确实"
        "过时/冲突/冒犯？"
    ),
    "source_type_match": (
        "来源类型是否正确：MP 是稳定偏好/属性，MS 是跨 session 模式，ME 是具体过去"
        "事件？"
    ),
    "dialogue_temporal_order_match": (
        "对话时间是否正确：dialogue 全部早于 current turn、角色交替、以 assistant "
        "结束，而且没有复制当前话语或未来信息？"
    ),
    "memory_age_design_match": (
        "memory 时间是否合理：created_session 均早于 session_index，age 有变化，且"
        "看不出用 age 直接编码 helpful/harmful/regime？"
    ),
    "strategy_need_match": (
        "strategy 条件是否正确：需要 RS 时用户明确要步骤；不应使用时没有暗示必须"
        "给计划；strategy_harmful 时用户明确只要倾听？"
    ),
}

REGIME_DEFINITIONS_ZH = {
    "context_only": "当前/近期对话已足够；不需要 MP/MS/ME，也不需要 strategy card。",
    "profile_needed": "只有 MP 中的稳定个人偏好能实质改善回复。",
    "summary_needed": "只有 MS 中的跨 session 重复模式能实质改善回复。",
    "event_needed": "只有 ME 中的具体过去事件能实质改善回复。",
    "multi_source_needed": "MP、MS、ME 提供互不重复且都必要的补充信息。",
    "memory_harmful": "旧 memory 过时、冲突或冒犯；应避免使用 memory。",
    "strategy_helpful": "用户明确需要步骤/计划；strategy card 有帮助，memory 不需要。",
    "strategy_harmful": "用户明确只想被倾听；结构化计划会过早或冒犯。",
    "ambiguous": "证据不足以确定唯一资源方案；多个低成本 action 都合理。",
}


def _readable_review(rows: Sequence[dict[str, str]]) -> str:
    parts = [
        "# PM V2 generation pilot 中文可读审查材料",
        "",
        "这不是只检查话题。每个 case 需要回答下面固定的 8 个问题。符合填 `1`；",
        "不符合或无法确定填 `0`，并在 notes 简短说明。精简的 `reviewer_a.csv`",
        "和 `reviewer_b.csv` 已预填不同 reviewer ID；每名审查者只填写自己的文件，",
        "完成前不得讨论答案。",
        "",
        "## 八个评分问题",
        "",
    ]
    for field in RATING_FIELDS:
        parts.append(f"- `{field}`：{REVIEW_QUESTIONS_ZH[field]}")
    parts.extend(["", "## 九类 regime 的简明定义", ""])
    for regime, definition in REGIME_DEFINITIONS_ZH.items():
        parts.append(f"- `{regime}`：{definition}")
    for index, row in enumerate(rows, 1):
        dialogue = json.loads(row["dialogue_before_current_json"])
        ages = json.loads(row["memory_age_sessions_json"])
        parts.extend(
            [
                "",
                f"## Case {index}: {row['candidate_regime']}",
                "",
                f"- 话题：`{row['candidate_semantic_family']}`",
                f"- 候选 needed sources：`{row['candidate_needed_memory_sources_json']}`",
                f"- surface 来源：`{row['surface_origin']}`",
                f"- 当前 session：`{row['session_index']}`；memory ages：`{canonical_json(ages)}`",
                "",
                f"当前用户：{row['current_user_text']}",
                "",
                "此前对话：",
                "",
            ]
        )
        for turn in dialogue:
            parts.append(f"- {turn['role']}: {turn['content']}")
        parts.extend(
            [
                "",
                f"Session summary：{row['session_summary']}",
                "",
                f"Authorized context：{row['authorized_user_context']}",
                "",
                f"冻结 rationale：{row['coverage_rationale']}",
                "",
                "Memory evidence：",
                "",
                "| Source | Utility | Age | Stale | Conflict | Sensitive | Text |",
                "|---|---|---:|---|---|---|---|",
            ]
        )
        for source, field in (
            ("MP", "profile_memories_json"),
            ("MS", "summary_memories_json"),
            ("ME", "event_memories_json"),
        ):
            memories = json.loads(row[field])
            for item_index, item in enumerate(memories):
                text = str(item["text"]).replace("|", "\\|").replace("\n", " ")
                parts.append(
                    f"| {source} | {item['item_utility']} | "
                    f"{ages[source][item_index]} | {item['stale']} | "
                    f"{item['conflicts_with_current_state']} | "
                    f"{item['private_sensitivity']} | {text} |"
                )
        parts.extend(
            [
                "",
                "本例填写：",
                "",
                "| CSV 字段 | 你的判断（0/1） | 判断问题 |",
                "|---|---:|---|",
            ]
        )
        for field in RATING_FIELDS:
            parts.append(f"| `{field}` |  | {REVIEW_QUESTIONS_ZH[field]} |")
        parts.extend(["", "备注："])
    return "\n".join(parts) + "\n"


def prepare_generation_pilot_semantic_review(
    *,
    pilot_attestation_path: str | Path,
    out_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    (
        pilot_attestation_path,
        contract,
        pilot_verification,
        bundle,
        bundle_path,
    ) = _pilot_inputs(pilot_attestation_path)
    out_dir = Path(out_dir).resolve()
    packet_path = out_dir / "generation_pilot_semantic_review_packet.csv"
    manual_path = out_dir / "generation_pilot_semantic_review_manual.md"
    readable_path = out_dir / "generation_pilot_semantic_review_readable_ZH.md"
    reviewer_a_path = out_dir / "reviewer_a.csv"
    reviewer_b_path = out_dir / "reviewer_b.csv"
    plan_path = out_dir / "generation_pilot_semantic_review_plan.json"
    for path in (
        packet_path,
        manual_path,
        readable_path,
        reviewer_a_path,
        reviewer_b_path,
        plan_path,
    ):
        if path.exists() and not overwrite:
            raise RuntimeError(f"refusing to overwrite existing review artifact: {path}")

    rows = _packet_rows(bundle)
    _write_csv(packet_path, rows)
    _write_simple_review_csv(reviewer_a_path, rows, annotator_id="reviewer_a")
    _write_simple_review_csv(reviewer_b_path, rows, annotator_id="reviewer_b")
    manual_path.write_text(_manual(), encoding="utf-8")
    readable_path.write_text(_readable_review(rows), encoding="utf-8")
    item_rows = [
        {
            "item_id": row["item_id"],
            "protected_packet_row_sha256": sha256_text(
                canonical_json({field: row[field] for field in PROTECTED_FIELDS})
            ),
        }
        for row in rows
    ]
    plan = {
        "protocol": GENERATION_REVIEW_PROTOCOL,
        "pilot_contract_sha256": contract["contract_sha256"],
        "pilot_attestation_path": str(pilot_attestation_path),
        "pilot_attestation_sha256": sha256_file(pilot_attestation_path),
        "pilot_bundle_path": str(bundle_path),
        "pilot_bundle_sha256": sha256_file(bundle_path),
        "packet_path": str(packet_path),
        "packet_sha256": sha256_file(packet_path),
        "manual_path": str(manual_path),
        "manual_sha256": sha256_file(manual_path),
        "readable_path": str(readable_path),
        "readable_sha256": sha256_file(readable_path),
        "minimum_annotators": MINIMUM_ANNOTATORS,
        "required_all_affirmative": True,
        "rating_fields": list(RATING_FIELDS),
        "items": item_rows,
    }
    plan["plan_sha256"] = sha256_text(canonical_json(plan))
    write_json(plan_path, plan)
    return {
        "status": "READY_FOR_TWO_INDEPENDENT_REVIEWS",
        "packet": str(packet_path),
        "manual": str(manual_path),
        "readable": str(readable_path),
        "reviewer_a": str(reviewer_a_path),
        "reviewer_b": str(reviewer_b_path),
        "plan": str(plan_path),
        "item_count": len(rows),
        "minimum_annotators": MINIMUM_ANNOTATORS,
        "pilot_verification": pilot_verification,
    }


def _read_completed(
    *,
    completed_paths: Sequence[str | Path],
    packet_rows: Sequence[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[Path]]:
    packet_by_item = {row["item_id"]: row for row in packet_rows}
    annotations: list[dict[str, Any]] = []
    paths: list[Path] = []
    seen_keys: set[tuple[str, str]] = set()
    file_annotators: set[str] = set()
    for raw_path in completed_paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise RuntimeError(f"completed review CSV is missing: {path}")
        paths.append(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            completed_fields = tuple(reader.fieldnames or ())
            if completed_fields not in (PACKET_FIELDS, SIMPLE_REVIEW_FIELDS):
                raise RuntimeError(f"completed review header differs: {path}")
            rows = list(reader)
        if len(rows) != len(packet_rows):
            raise RuntimeError(f"completed review row count differs: {path}")
        file_ids: set[str] = set()
        seen_items: set[str] = set()
        for row in rows:
            item_id = str(row.get("item_id") or "")
            if item_id not in packet_by_item or item_id in seen_items:
                raise RuntimeError(f"invalid/duplicate completed review item: {item_id}")
            seen_items.add(item_id)
            reference = packet_by_item[item_id]
            if completed_fields == PACKET_FIELDS:
                for field in PROTECTED_FIELDS:
                    if row.get(field) != reference[field]:
                        raise RuntimeError(
                            f"completed review changed protected cell {item_id}/{field}"
                        )
            annotator_id = str(row.get("annotator_id") or "").strip()
            if not annotator_id:
                raise ValueError(f"{item_id}: missing annotator_id")
            file_ids.add(annotator_id)
            key = (item_id, annotator_id)
            if key in seen_keys:
                raise RuntimeError(f"duplicate annotation {item_id}/{annotator_id}")
            seen_keys.add(key)
            ratings: dict[str, int] = {}
            for field in RATING_FIELDS:
                value = str(row.get(field) or "").strip()
                if value not in {"0", "1"}:
                    raise ValueError(f"{item_id}/{annotator_id}: {field} must be 0 or 1")
                ratings[field] = int(value)
            annotations.append(
                {
                    "item_id": item_id,
                    "annotator_id": annotator_id,
                    "ratings": ratings,
                    "notes": str(row.get("notes") or ""),
                    "completed_path": str(path),
                }
            )
        if len(file_ids) != 1:
            raise RuntimeError(f"each completed packet must use one annotator_id: {path}")
        file_annotators.update(file_ids)
    if len(file_annotators) < MINIMUM_ANNOTATORS:
        raise RuntimeError(
            f"generation review requires {MINIMUM_ANNOTATORS} independent annotators"
        )
    return annotations, paths


def analyze_generation_pilot_semantic_review(
    *,
    completed_paths: Sequence[str | Path],
    pilot_attestation_path: str | Path,
    packet_path: str | Path,
    manual_path: str | Path,
    plan_path: str | Path,
    report_path: str | Path,
    attestation_path: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    (
        pilot_attestation_path,
        contract,
        _,
        bundle,
        bundle_path,
    ) = _pilot_inputs(pilot_attestation_path)
    packet_path = Path(packet_path).resolve()
    manual_path = Path(manual_path).resolve()
    plan_path = Path(plan_path).resolve()
    report_path = Path(report_path).resolve()
    attestation_path = Path(attestation_path).resolve()
    for path in (packet_path, manual_path, plan_path):
        if not path.is_file():
            raise RuntimeError(f"generation review input is missing: {path}")
    for path in (report_path, attestation_path):
        if path.exists() and not overwrite:
            raise RuntimeError(f"refusing to overwrite existing review result: {path}")

    plan = read_json(plan_path)
    stored_plan_hash = plan.get("plan_sha256")
    plan_payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if stored_plan_hash != sha256_text(canonical_json(plan_payload)):
        raise RuntimeError("generation semantic review plan hash mismatch")
    exact_lineage = (
        plan.get("protocol") == GENERATION_REVIEW_PROTOCOL
        and plan.get("pilot_contract_sha256") == contract["contract_sha256"]
        and plan.get("pilot_attestation_sha256")
        == sha256_file(pilot_attestation_path)
        and plan.get("pilot_bundle_sha256") == sha256_file(bundle_path)
        and plan.get("packet_sha256") == sha256_file(packet_path)
        and plan.get("manual_sha256") == sha256_file(manual_path)
        and Path(str(plan.get("readable_path") or "")).resolve().is_file()
        and plan.get("readable_sha256")
        == sha256_file(Path(str(plan["readable_path"])).resolve())
    )
    if not exact_lineage:
        raise RuntimeError("generation semantic review lineage is stale")

    with packet_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PACKET_FIELDS:
            raise RuntimeError("generation review packet header differs")
        packet_rows = list(reader)
    expected_rows = _packet_rows(bundle)
    if packet_rows != expected_rows:
        raise RuntimeError("generation review packet differs from the pilot bundle")
    for row, plan_item in zip(packet_rows, plan["items"], strict=True):
        protected_hash = sha256_text(
            canonical_json({field: row[field] for field in PROTECTED_FIELDS})
        )
        if (
            plan_item.get("item_id") != row["item_id"]
            or plan_item.get("protected_packet_row_sha256") != protected_hash
        ):
            raise RuntimeError("generation review protected-row plan mismatch")

    annotations, completed = _read_completed(
        completed_paths=completed_paths,
        packet_rows=packet_rows,
    )
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotations:
        by_item[row["item_id"]].append(row)
    underannotated = sorted(
        item_id
        for item_id in (row["item_id"] for row in packet_rows)
        if len(by_item[item_id]) < MINIMUM_ANNOTATORS
    )
    negative_ratings = [
        {
            "item_id": row["item_id"],
            "annotator_id": row["annotator_id"],
            "field": field,
        }
        for row in annotations
        for field, value in row["ratings"].items()
        if value != 1
    ]
    agreement_values: list[int] = []
    for item_rows in by_item.values():
        for left, right in combinations(item_rows, 2):
            agreement_values.extend(
                int(left["ratings"][field] == right["ratings"][field])
                for field in RATING_FIELDS
            )
    exact_agreement = (
        sum(agreement_values) / len(agreement_values) if agreement_values else 0.0
    )
    checks = {
        "exact_pilot_lineage": exact_lineage,
        "exact_nine_items": len(packet_rows) == 9 and len(by_item) == 9,
        "minimum_two_independent_annotators_per_item": not underannotated,
        "all_semantic_ratings_affirmative": not negative_ratings,
        "pairwise_exact_agreement": exact_agreement == 1.0,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": GENERATION_REVIEW_PROTOCOL,
        "checks": checks,
        "pilot_contract_sha256": contract["contract_sha256"],
        "plan_sha256": stored_plan_hash,
        "item_count": len(packet_rows),
        "annotation_count": len(annotations),
        "annotators": sorted({row["annotator_id"] for row in annotations}),
        "underannotated_items": underannotated,
        "negative_ratings": negative_ratings,
        "pairwise_exact_agreement": exact_agreement,
        "decision_rule": "all nine cases x all dimensions x all annotators must equal 1",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    input_paths: dict[str, Path] = {
        "pilot_attestation": pilot_attestation_path,
        "pilot_bundle": bundle_path,
        "packet": packet_path,
        "manual": manual_path,
        "readable": Path(str(plan["readable_path"])).resolve(),
        "plan": plan_path,
    }
    for index, path in enumerate(completed, 1):
        input_paths[f"completed_review_{index:02d}"] = path
    create_artifact_attestation(
        attestation_path,
        stage=GENERATION_REVIEW_STAGE,
        inputs=input_paths,
        outputs={"report": (report_path, False)},
        parameters={
            "protocol": GENERATION_REVIEW_PROTOCOL,
            "pilot_contract_sha256": contract["contract_sha256"],
            "plan_sha256": stored_plan_hash,
            "minimum_annotators": MINIMUM_ANNOTATORS,
            "required_all_affirmative": True,
        },
        expected={
            "status": "PASS",
            "items": 9,
            "minimum_annotations": 9 * MINIMUM_ANNOTATORS,
        },
    )
    return report


def require_generation_pilot_semantic_review(
    attestation_path: str | Path,
    *,
    expected_pilot_attestation_path: str | Path,
    expected_contract: dict[str, Any],
) -> dict[str, Any]:
    attestation_path = Path(attestation_path).resolve()
    if not attestation_path.is_file():
        raise RuntimeError(
            "full PM-v2 generation requires a PASS two-annotator semantic review: "
            f"{attestation_path}"
        )
    attestation = read_json(attestation_path)
    report_record = (attestation.get("outputs") or {}).get("report") or {}
    report_path = Path(str(report_record.get("path") or "")).resolve()
    if not report_path.is_file():
        raise RuntimeError("generation semantic review report is missing")
    verification = require_artifact_attestation(
        attestation_path,
        required_stage=GENERATION_REVIEW_STAGE,
        required_output_paths={"report": report_path},
    )
    parameters = attestation.get("parameters") or {}
    if (
        parameters.get("protocol") != GENERATION_REVIEW_PROTOCOL
        or parameters.get("pilot_contract_sha256")
        != expected_contract["contract_sha256"]
    ):
        raise RuntimeError("generation semantic review contract lineage differs")
    expected_pilot_attestation_path = Path(
        expected_pilot_attestation_path
    ).resolve()
    input_rows = attestation.get("inputs") or {}
    pilot_input = input_rows.get("pilot_attestation") or {}
    if (
        Path(str(pilot_input.get("path") or "")).resolve()
        != expected_pilot_attestation_path
        or pilot_input.get("sha256") != sha256_file(expected_pilot_attestation_path)
    ):
        raise RuntimeError("generation semantic review uses a different pilot")
    report = read_json(report_path)
    if report.get("status") != "PASS" or not all(
        (report.get("checks") or {}).values()
    ):
        raise RuntimeError("generation pilot semantic review is not PASS")
    return {
        **verification,
        "status": "PASS",
        "protocol": GENERATION_REVIEW_PROTOCOL,
        "pilot_contract_sha256": expected_contract["contract_sha256"],
        "review_report_sha256": sha256_file(report_path),
        "annotators": report["annotators"],
        "annotation_count": report["annotation_count"],
    }
