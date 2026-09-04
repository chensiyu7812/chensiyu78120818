#!/usr/bin/env python3
"""Freeze and build the final E7 external human-review packets (zero API).

The sample is selected entirely from frozen E2/E3 plan metadata before this
script reads E4 response text.  It never reads an E6 directory.  One
protocol-hashed representative seed is used per state; exact aliases and
exact public surfaces are reviewed once and propagated through private keys.

Public panels:
  * anonymous pairwise response quality;
  * evidence-aware, single-response interaction-and-grounding material risk;
  * an independently assigned 20% overlap for each construct.

ES-MemEval QA is intentionally absent.  Its primary outcome is the official
objective scoring surface plus the separately frozen E6-QA robustness judge.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    stable_hex,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5.2-external-e7-human-review-v1"
QUALITY_PROTOCOL = "pm-v1.5-v5.2-external-e7-quality-blind-v1"
RISK_PROTOCOL = "pm-v1.5-v5.2-external-e7-grounding-risk-v1"

CONTRACT = ROOT / "data/pm_v1_5_contracts/v5_2_external_e7_human_review_v1.json"
E2 = ROOT / "outputs/pm_v1_5_v5_2_external_e2_plan_v1"
E3 = ROOT / "outputs/pm_v1_5_v5_2_external_e3_retrieval_v1"
E4 = ROOT / "outputs/pm_v1_5_v5_2_external_e4_degenerate_exclusion_continuation_execution_v1"
CORE_PLAN = E2 / "response_core_call_plan_private.jsonl"
RAW_PLAN = E3 / "response_raw_call_plan_private.jsonl"
RAW_AUDIT = E3 / "response_raw_retrieval_audit_private.jsonl"
CORE_OUTCOMES = E4 / "response_core_outcomes_private.jsonl"
RAW_OUTCOMES = E4 / "response_raw_outcomes_private.jsonl"
E4_SUMMARY = E4 / "execution_summary.json"
INVALID_REGISTRY = E4 / "registered_invalid_generation.json"

QUALITY_HELPER = ROOT / "scripts/v1_5/25zy_prepare_v5_full_fit_outcome_review_v1_5.py"
RISK_HELPER = ROOT / "scripts/v1_5/26i_prepare_v5_1_confirmation_review_v1_5.py"

ESCONV_PARTITION = "esconv_corrected_test_panel"
EVO_PARTITIONS = {"evoemo_qualification_p7_p12", "evoemo_lockbox_p13_p18"}
CORE_BASELINES = (
    "always_off",
    "fixed_high_eligible",
    "transparent_rule",
    "cost_matched_fixed",
)
RAW_CONDITIONS = (
    "raw_session_top4_plus_frozen_strategy",
    "all_raw_sessions_plus_frozen_strategy",
)
VISIBLE_MARKER = "Visible current conversation:"
RAW_PRIOR_MARKER = "Strictly prior same-user dialogue sessions"


def canonical_visible(text: str) -> str:
    """Normalize only whitespace and speaker-boundary formatting."""
    collapsed = " ".join(str(text).split())
    return re.sub(r"\s+(?=(?:User|Assistant):\s)", "\n", collapsed).strip()


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load review helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hashed_take(values: list[str], count: int, *salt: str) -> list[str]:
    if count > len(values):
        raise RuntimeError(f"cannot select {count} from {len(values)} values: {salt}")
    return sorted(
        values,
        key=lambda value: stable_hex(PROTOCOL, *salt, value, n=32),
    )[:count]


def representative_seed(state_id: str, available: set[str]) -> str:
    if available != {"seed-a", "seed-b"}:
        raise RuntimeError(f"state lacks the two frozen seeds: {state_id}: {available}")
    return ("seed-a", "seed-b")[
        int(stable_hex(PROTOCOL, "representative-seed", state_id, n=8), 16) % 2
    ]


def core_visible(user_content: str, call_id: str) -> str:
    if not user_content.startswith(VISIBLE_MARKER):
        raise RuntimeError(f"unexpected core user prompt: {call_id}")
    return canonical_visible(user_content[len(VISIBLE_MARKER) :])


def raw_visible_and_authorization(user_content: str, call_id: str) -> tuple[str, str]:
    if not user_content.startswith(VISIBLE_MARKER):
        raise RuntimeError(f"unexpected raw user prompt: {call_id}")
    marker = f"\n\n{RAW_PRIOR_MARKER}"
    if marker not in user_content:
        raise RuntimeError(f"raw prompt missing prior-session marker: {call_id}")
    visible, remainder = user_content[len(VISIBLE_MARKER) :].split(marker, 1)
    return canonical_visible(visible), (RAW_PRIOR_MARKER + remainder).strip()


def core_authorization(plan_row: dict[str, Any]) -> str:
    comp = dict(plan_row.get("composition_plan") or {})
    parts: list[str] = []
    for clause in comp.get("locked_clauses") or []:
        parts.append(
            f"[Authorized past evidence] {clause['text']}"
        )
    preference = str(comp.get("response_preference") or "").strip()
    if preference:
        parts.append(f"[Authorized response-format preference] {preference}")
    strategy = str(comp.get("strategy_instruction") or "").strip()
    if strategy:
        parts.append(f"[Authorized atomic support instruction] {strategy}")
    if not parts:
        return (
            "Visible current conversation only; no prior user-specific memory "
            "or strategy resource was authorized."
        )
    public = "\n\n".join(parts)
    if any(marker in public for marker in ("mem_", "strategy_v", "card_")):
        raise RuntimeError(f"opaque resource ID leaked: {plan_row['call_id']}")
    return public


def deduplicate(
    records: list[dict[str, Any]],
    *,
    id_field: str,
    signature: Callable[[dict[str, Any]], str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[signature(record)].append(record)
    representatives: list[dict[str, Any]] = []
    representative_by_id: dict[str, str] = {}
    for members in grouped.values():
        representative = min(members, key=lambda row: str(row[id_field]))
        representatives.append(representative)
        for member in members:
            representative_by_id[str(member[id_field])] = str(representative[id_field])
    representatives.sort(key=lambda row: str(row[id_field]))
    return representatives, representative_by_id


def stratified_overlap(
    records: list[dict[str, Any]],
    *,
    id_field: str,
    stratum_by_id: dict[str, str],
    fraction: float,
    salt: str,
) -> set[str]:
    """Largest-remainder proportional allocation with an exact global target."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        item_id = str(record[id_field])
        grouped[stratum_by_id[item_id]].append(item_id)
    target = round(len(records) * fraction)
    allocation: dict[str, int] = {
        stratum: math.floor(len(ids) * fraction) for stratum, ids in grouped.items()
    }
    remaining = target - sum(allocation.values())
    remainders = sorted(
        grouped,
        key=lambda stratum: (
            -(len(grouped[stratum]) * fraction - allocation[stratum]),
            stable_hex(PROTOCOL, salt, "remainder", stratum, n=32),
        ),
    )
    for stratum in remainders[:remaining]:
        allocation[stratum] += 1
    selected: set[str] = set()
    for stratum, ids in grouped.items():
        ordered = sorted(
            ids,
            key=lambda item_id: stable_hex(
                PROTOCOL, salt, "item", stratum, item_id, n=32
            ),
        )
        selected.update(ordered[: allocation[stratum]])
    if len(selected) != target:
        raise RuntimeError(f"overlap allocation drift: {len(selected)} != {target}")
    return selected


def make_quality_html(template: str) -> str:
    return (
        template.replace("PM V1.5 V5 FIT 质量盲评", "PM V1.5 外部 E7 质量盲评")
        .replace(
            "本页属于一次完整冻结panel，不是开发小包。",
            "本页属于唯一最终外部分层人评，不是开发小包。",
        )
    )


def make_risk_html(template: str) -> str:
    return (
        template.replace(
            "PM V1.5 V5.1 Confirmation Grounding Risk",
            "PM V1.5 外部 E7 Grounding Risk",
        )
        .replace(
            "审核ON与OFF双方回复。",
            "逐条审核冻结外部系统回复。",
        )
    )


def write_html(
    path: Path,
    *,
    template: str,
    protocol: str,
    items: list[dict[str, Any]],
    role: str,
    safe_json: Callable[[Any], str],
) -> None:
    payload = {
        "manifest": {"protocol": PROTOCOL, "panel_role": role, "items": len(items)},
        "items": items,
    }
    path.write_text(
        template.replace("__DATA__", safe_json(payload)).replace(
            "__PROTOCOL__", protocol
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_v5_2_external_e7_human_review_v1_candidate",
    )
    args = parser.parse_args()

    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal E7 requires {FORMAL_PYTHON}; got {sys.executable}")
    contract = read_json(CONTRACT)
    if contract.get("status") != "FROZEN_BEFORE_E6_OR_E7_OUTCOMES":
        raise RuntimeError("E7 contract is not outcome-blind frozen")
    if not all(value is False for value in (
        contract["attestations"]["e6_judge_outcomes_read"],
        contract["attestations"]["sampling_used_llm_judge_result"],
        contract["attestations"]["sampling_used_quality_or_risk_outcome"],
        contract["attestations"]["pm_route_generator_or_resource_changed"],
    )):
        raise RuntimeError("E7 outcome-blind attestation failed")

    execution = read_json(E4_SUMMARY)
    if execution.get("status") != (
        "COMPLETE_WITH_ONE_REGISTERED_INVALID_RAW_GENERATION_READY_FOR_"
        "VERSIONED_E5_ZERO_API_SCORING"
    ):
        raise RuntimeError("final E4 execution is incomplete")
    if execution.get("valid_completed_calls") != 4217:
        raise RuntimeError("final E4 valid call denominator drifted")
    invalid = read_json(INVALID_REGISTRY)
    excluded_raw_calls = set(invalid["raw_secondary_table_excluded_call_ids"])
    excluded_raw_state = str(invalid["raw_secondary_table_excluded_state_id"])

    # Plan-only materialization and sample selection.  No outcome file has
    # been opened above this line.
    core_plan_rows = rows(CORE_PLAN)
    raw_plan_rows = rows(RAW_PLAN)
    raw_audit_rows = rows(RAW_AUDIT)
    if len(core_plan_rows) != 1576 or len(raw_plan_rows) != 552:
        raise RuntimeError("external response plan denominator drifted")
    core_plan = {str(row["call_id"]): row for row in core_plan_rows}
    raw_plan = {str(row["call_id"]): row for row in raw_plan_rows}
    raw_audit = {
        (str(row["state_id"]), str(row["condition"])): row
        for row in raw_audit_rows
    }

    core_by_state_seed: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    state_meta: dict[str, dict[str, str]] = {}
    seeds_by_state: dict[str, set[str]] = defaultdict(set)
    for row in core_plan_rows:
        state_id, seed_label = str(row["state_id"]), str(row["seed_label"])
        seeds_by_state[state_id].add(seed_label)
        state_meta.setdefault(
            state_id,
            {
                "partition": str(row["partition"]),
                "cluster_id": str(row["user_id_private_analysis_only"]),
            },
        )
        for alias in row.get("policy_aliases") or []:
            existing = core_by_state_seed[(state_id, seed_label)].get(str(alias))
            if existing is not None and existing != row["call_id"]:
                raise RuntimeError(f"policy alias maps to two calls: {state_id}/{alias}")
            core_by_state_seed[(state_id, seed_label)][str(alias)] = str(row["call_id"])
    if any(meta["partition"] not in {ESCONV_PARTITION, *EVO_PARTITIONS} for meta in state_meta.values()):
        raise RuntimeError("unexpected external partition")
    selected_seed = {
        state_id: representative_seed(state_id, seeds)
        for state_id, seeds in seeds_by_state.items()
    }

    raw_by_state_seed: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for row in raw_plan_rows:
        if str(row["call_id"]) in excluded_raw_calls or str(row["state_id"]) == excluded_raw_state:
            continue
        raw_by_state_seed[(str(row["state_id"]), str(row["seed_label"]))][
            str(row["condition"])
        ] = str(row["call_id"])

    esconv_states = sorted(
        state_id
        for state_id, meta in state_meta.items()
        if meta["partition"] == ESCONV_PARTITION
    )
    evo_by_user: dict[str, list[str]] = defaultdict(list)
    for state_id, meta in state_meta.items():
        if meta["partition"] in EVO_PARTITIONS:
            evo_by_user[meta["cluster_id"]].append(state_id)
    if len(esconv_states) != 122 or len(evo_by_user) != 12:
        raise RuntimeError("external cluster denominator drifted")

    esconv_quality_states = hashed_take(esconv_states, 64, "quality", "esconv")
    evo_quality_by_user: dict[str, list[str]] = {}
    evo_raw_quality_by_user: dict[str, list[str]] = {}
    for user, values in sorted(evo_by_user.items()):
        selected = hashed_take(sorted(values), 4, "quality", "evoemo", user)
        raw_eligible = [
            state_id
            for state_id in selected
            if set(raw_by_state_seed.get((state_id, selected_seed[state_id]), {}))
            == set(RAW_CONDITIONS)
        ]
        # If the nested four contain the one registered-invalid raw state,
        # replace only the raw-secondary selection from the user's complete
        # valid pool; core selection remains unchanged.
        if len(raw_eligible) < 2:
            raw_eligible = [
                state_id
                for state_id in sorted(values)
                if set(raw_by_state_seed.get((state_id, selected_seed[state_id]), {}))
                == set(RAW_CONDITIONS)
            ]
        evo_quality_by_user[user] = selected
        evo_raw_quality_by_user[user] = hashed_take(
            raw_eligible, 2, "quality", "evoemo-raw", user
        )
    evo_quality_states = sorted(
        state_id for values in evo_quality_by_user.values() for state_id in values
    )
    evo_raw_quality_states = sorted(
        state_id
        for values in evo_raw_quality_by_user.values()
        for state_id in values
    )
    if len(evo_quality_states) != 48 or len(evo_raw_quality_states) != 24:
        raise RuntimeError("EvoEmo quality sampling shape drifted")

    esconv_risk_states = hashed_take(
        esconv_quality_states, 32, "risk", "esconv"
    )
    evo_risk_states: list[str] = []
    evo_raw_risk_states: list[str] = []
    for user in sorted(evo_quality_by_user):
        evo_risk_states.extend(
            hashed_take(evo_quality_by_user[user], 2, "risk", "evoemo", user)
        )
        evo_raw_risk_states.extend(
            hashed_take(
                evo_raw_quality_by_user[user], 1, "risk", "evoemo-raw", user
            )
        )
    if len(evo_risk_states) != 24 or len(evo_raw_risk_states) != 12:
        raise RuntimeError("EvoEmo risk sampling shape drifted")

    # Build pair/call identities without response text.
    quality_specs: list[dict[str, Any]] = []

    def add_core_quality(state_id: str, domain: str) -> None:
        seed = selected_seed[state_id]
        aliases = core_by_state_seed[(state_id, seed)]
        learned = aliases.get("learned_pm")
        if learned is None:
            raise RuntimeError(f"learned_pm absent: {state_id}/{seed}")
        comparator_aliases: dict[str, list[str]] = defaultdict(list)
        for baseline in CORE_BASELINES:
            comparator = aliases.get(baseline)
            if comparator is not None:
                comparator_aliases[comparator].append(baseline)
        for comparator, names in sorted(comparator_aliases.items()):
            quality_specs.append(
                {
                    "table": "core",
                    "domain": domain,
                    "state_id": state_id,
                    "cluster_id": state_meta[state_id]["cluster_id"],
                    "partition": state_meta[state_id]["partition"],
                    "seed_label": seed,
                    "learned_call_id": learned,
                    "comparator_call_id": comparator,
                    "comparators": sorted(names),
                    "stratum": f"{domain}:core:{'+'.join(sorted(names))}",
                }
            )

    for state_id in esconv_quality_states:
        add_core_quality(state_id, "esconv")
    for state_id in evo_quality_states:
        add_core_quality(state_id, "evoemo")
    for state_id in evo_raw_quality_states:
        seed = selected_seed[state_id]
        learned = core_by_state_seed[(state_id, seed)].get("learned_pm")
        if learned is None:
            raise RuntimeError(f"learned_pm absent for raw pair: {state_id}/{seed}")
        for condition in RAW_CONDITIONS:
            comparator = raw_by_state_seed[(state_id, seed)].get(condition)
            if comparator is None:
                raise RuntimeError(f"raw comparator absent: {state_id}/{seed}/{condition}")
            quality_specs.append(
                {
                    "table": "raw",
                    "domain": "evoemo",
                    "state_id": state_id,
                    "cluster_id": state_meta[state_id]["cluster_id"],
                    "partition": state_meta[state_id]["partition"],
                    "seed_label": seed,
                    "learned_call_id": learned,
                    "comparator_call_id": comparator,
                    "comparators": [condition],
                    "stratum": f"evoemo:raw:{condition}",
                }
            )

    risk_call_ids: set[str] = set()
    for state_id in esconv_risk_states + evo_risk_states:
        risk_call_ids.update(
            core_by_state_seed[(state_id, selected_seed[state_id])].values()
        )
    for state_id in evo_raw_risk_states:
        risk_call_ids.update(
            raw_by_state_seed[(state_id, selected_seed[state_id])].values()
        )

    # A/B assignment is fixed from plan identities, before outcomes are read.
    learned_as_a: set[str] = set()
    quality_specs_by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in quality_specs:
        comparison_id = "e7cmp_" + stable_hex(
            PROTOCOL,
            spec["table"],
            spec["state_id"],
            spec["seed_label"],
            spec["comparator_call_id"],
            n=24,
        )
        spec["comparison_id"] = comparison_id
        quality_specs_by_stratum[spec["stratum"]].append(spec)
    for stratum, members in quality_specs_by_stratum.items():
        ordered = sorted(
            members,
            key=lambda spec: stable_hex(
                PROTOCOL, "ab-position", stratum, spec["comparison_id"], n=32
            ),
        )
        learned_as_a.update(
            str(spec["comparison_id"]) for spec in ordered[: len(ordered) // 2]
        )

    sample_identity = {
        "protocol": PROTOCOL,
        "selected_seed_by_state": selected_seed,
        "esconv_quality_states": sorted(esconv_quality_states),
        "evoemo_quality_states_by_user": evo_quality_by_user,
        "evoemo_raw_quality_states_by_user": evo_raw_quality_by_user,
        "esconv_risk_states": sorted(esconv_risk_states),
        "evoemo_risk_states": sorted(evo_risk_states),
        "evoemo_raw_risk_states": sorted(evo_raw_risk_states),
        "quality_comparison_ids": sorted(
            str(spec["comparison_id"]) for spec in quality_specs
        ),
        "risk_call_ids": sorted(risk_call_ids),
        "selection_inputs": "E2/E3 plan metadata only; generated response text not read",
    }
    sample_identity_sha256 = sha256_text(canonical_json(sample_identity))

    # Only now read generated response outcomes.
    core_outcome_rows = rows(CORE_OUTCOMES)
    raw_outcome_rows_all = rows(RAW_OUTCOMES)
    raw_outcome_rows = [
        row for row in raw_outcome_rows_all if str(row["call_id"]) not in excluded_raw_calls
    ]
    if len(core_outcome_rows) != 1576 or len(raw_outcome_rows) != 548:
        raise RuntimeError("valid external response outcome denominator drifted")
    outcomes = {
        str(row["call_id"]): row for row in core_outcome_rows + raw_outcome_rows
    }

    raw_audit_by_key = {
        (str(row["state_id"]), str(row["condition"])): row
        for row in raw_audit_rows
    }
    call_visible: dict[str, str] = {}
    call_auth: dict[str, str] = {}
    call_response: dict[str, str] = {}
    for row in core_outcome_rows:
        call_id = str(row["call_id"])
        plan = core_plan.get(call_id)
        if plan is None or plan["messages_sha256"] != row["messages_sha256"]:
            raise RuntimeError(f"core plan/outcome identity mismatch: {call_id}")
        if row.get("guard_errors"):
            raise RuntimeError(f"core outcome has guard errors: {call_id}")
        comp = dict(plan.get("composition_plan") or {})
        clauses = list(comp.get("locked_clauses") or [])
        expected = " ".join(
            [str(row["primary_response"]).strip()]
            + [str(clause["text"]) for clause in clauses]
        )
        if " ".join(expected.split()) != " ".join(str(row["final_output"]).split()):
            raise RuntimeError(f"core locked composition mismatch: {call_id}")
        for clause in clauses:
            if str(clause["text"]) not in str(row["final_output"]):
                raise RuntimeError(f"core locked clause absent: {call_id}")
        user = next(
            str(message["content"])
            for message in plan["messages"]
            if message["role"] == "user"
        )
        call_visible[call_id] = core_visible(user, call_id)
        call_auth[call_id] = core_authorization(plan)
        call_response[call_id] = str(row["final_output"])
    for row in raw_outcome_rows:
        call_id = str(row["call_id"])
        plan = raw_plan.get(call_id)
        if plan is None or plan["messages_sha256"] != row["messages_sha256"]:
            raise RuntimeError(f"raw plan/outcome identity mismatch: {call_id}")
        if row.get("guard_errors"):
            raise RuntimeError(f"raw outcome has guard errors: {call_id}")
        audit = raw_audit_by_key.get((str(plan["state_id"]), str(plan["condition"])))
        if audit is None or list(plan["selected_session_ids"]) != list(
            audit["selected_session_ids"]
        ):
            raise RuntimeError(f"raw retrieval identity mismatch: {call_id}")
        user = next(
            str(message["content"])
            for message in plan["messages"]
            if message["role"] == "user"
        )
        visible, auth = raw_visible_and_authorization(user, call_id)
        call_visible[call_id] = visible
        call_auth[call_id] = auth
        call_response[call_id] = str(row["final_output"])

    quality_rows: list[dict[str, Any]] = []
    quality_key: list[dict[str, Any]] = []
    automatic_ties: list[dict[str, Any]] = []
    for spec in sorted(quality_specs, key=lambda row: str(row["comparison_id"])):
        learned_id = str(spec["learned_call_id"])
        comparator_id = str(spec["comparator_call_id"])
        if call_visible[learned_id] != call_visible[comparator_id]:
            raise RuntimeError(
                "visible conversation differs inside pair: "
                f"{spec['comparison_id']} table={spec['table']} state={spec['state_id']} "
                f"learned={learned_id}:{sha256_text(call_visible[learned_id])} "
                f"comparator={comparator_id}:{sha256_text(call_visible[comparator_id])}"
            )
        key_row = {
            "protocol": PROTOCOL,
            "comparison_id": spec["comparison_id"],
            "domain": spec["domain"],
            "table": spec["table"],
            "partition": spec["partition"],
            "state_id": spec["state_id"],
            "cluster_id": spec["cluster_id"],
            "seed_label": spec["seed_label"],
            "comparators": spec["comparators"],
            "stratum": spec["stratum"],
            "learned_call_id": learned_id,
            "comparator_call_id": comparator_id,
        }
        if call_response[learned_id] == call_response[comparator_id]:
            tie_id = "e7tie_" + stable_hex(PROTOCOL, spec["comparison_id"], n=24)
            automatic_ties.append(
                {
                    "protocol": PROTOCOL,
                    "automatic_tie_id": tie_id,
                    **key_row,
                    "quality_preference": "tie",
                    "decisive_criterion": "materially_equivalent",
                    "reason": "Byte-identical responses; deterministic exact-identity tie.",
                }
            )
            continue
        learned_first = str(spec["comparison_id"]) in learned_as_a
        call_a, call_b = (
            (learned_id, comparator_id)
            if learned_first
            else (comparator_id, learned_id)
        )
        blind_id = "e7q_" + stable_hex(PROTOCOL, spec["comparison_id"], n=24)
        quality_rows.append(
            {
                "protocol": QUALITY_PROTOCOL,
                "blind_item_id": blind_id,
                "visible_conversation": call_visible[learned_id],
                "response_a": call_response[call_a],
                "response_b": call_response[call_b],
            }
        )
        quality_key.append(
            {
                **key_row,
                "blind_item_id": blind_id,
                "call_id_a": call_a,
                "call_id_b": call_b,
                "learned_is_a": learned_first,
            }
        )

    quality_manual, quality_representative = deduplicate(
        quality_rows,
        id_field="blind_item_id",
        signature=lambda row: canonical_json(
            [row["visible_conversation"], row["response_a"], row["response_b"]]
        ),
    )
    quality_stratum_by_rep: dict[str, str] = {}
    for row in quality_key:
        representative = quality_representative[str(row["blind_item_id"])]
        row["manual_representative_id"] = representative
        quality_stratum_by_rep.setdefault(representative, str(row["stratum"]))

    risk_underlying: list[dict[str, Any]] = []
    for call_id in sorted(risk_call_ids):
        plan = core_plan.get(call_id) or raw_plan.get(call_id)
        if plan is None:
            raise RuntimeError(f"sampled risk call absent from plan: {call_id}")
        domain = "esconv" if str(plan["partition"]) == ESCONV_PARTITION else "evoemo"
        if call_id in core_plan:
            aliases = sorted(str(value) for value in plan.get("policy_aliases") or [])
            condition = "+".join(aliases)
            table = "core"
        else:
            aliases = []
            condition = str(plan["condition"])
            table = "raw"
        risk_underlying.append(
            {
                "call_id": call_id,
                "domain": domain,
                "table": table,
                "partition": str(plan["partition"]),
                "state_id": str(plan["state_id"]),
                "cluster_id": str(plan["user_id_private_analysis_only"]),
                "seed_label": str(plan["seed_label"]),
                "policy_aliases": aliases,
                "condition": condition,
                "stratum": f"{domain}:{table}:{condition}",
                "visible_conversation": call_visible[call_id],
                "authorized_evidence_and_instruction": call_auth[call_id],
                "candidate_response": call_response[call_id],
            }
        )

    surface_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in risk_underlying:
        signature = canonical_json(
            [
                row["visible_conversation"],
                row["authorized_evidence_and_instruction"],
                row["candidate_response"],
            ]
        )
        surface_groups[signature].append(row)
    risk_rows: list[dict[str, Any]] = []
    risk_key: list[dict[str, Any]] = []
    risk_stratum_by_id: dict[str, str] = {}
    for signature, members in sorted(surface_groups.items()):
        representative = min(members, key=lambda row: str(row["call_id"]))
        risk_id = "e7r_" + stable_hex(PROTOCOL, signature, n=24)
        risk_rows.append(
            {
                "protocol": RISK_PROTOCOL,
                "risk_item_id": risk_id,
                "visible_conversation": representative["visible_conversation"],
                "authorized_evidence_and_instruction": representative[
                    "authorized_evidence_and_instruction"
                ],
                "candidate_response": representative["candidate_response"],
            }
        )
        strata = sorted({str(member["stratum"]) for member in members})
        risk_stratum_by_id[risk_id] = strata[0]
        risk_key.append(
            {
                "protocol": PROTOCOL,
                "risk_item_id": risk_id,
                "public_surface_sha256": sha256_text(signature),
                "propagates_to_call_ids": sorted(
                    str(member["call_id"]) for member in members
                ),
                "propagates_to_rows": [
                    {
                        key: member[key]
                        for key in (
                            "call_id",
                            "domain",
                            "table",
                            "partition",
                            "state_id",
                            "cluster_id",
                            "seed_label",
                            "policy_aliases",
                            "condition",
                            "stratum",
                        )
                    }
                    for member in members
                ],
                "strata": strata,
            }
        )

    overlap_quality_ids = stratified_overlap(
        quality_manual,
        id_field="blind_item_id",
        stratum_by_id=quality_stratum_by_rep,
        fraction=0.2,
        salt="quality-overlap",
    )
    overlap_risk_ids = stratified_overlap(
        risk_rows,
        id_field="risk_item_id",
        stratum_by_id=risk_stratum_by_id,
        fraction=0.2,
        salt="risk-overlap",
    )
    overlap_quality = [
        row for row in quality_manual if str(row["blind_item_id"]) in overlap_quality_ids
    ]
    overlap_risk = [
        row for row in risk_rows if str(row["risk_item_id"]) in overlap_risk_ids
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "sample_identity_preoutcome.json", sample_identity)
    write_jsonl(args.out_dir / "primary_quality_packet.jsonl", quality_manual)
    write_jsonl(args.out_dir / "primary_risk_packet.jsonl", risk_rows)
    write_jsonl(args.out_dir / "overlap_quality_packet.jsonl", overlap_quality)
    write_jsonl(args.out_dir / "overlap_risk_packet.jsonl", overlap_risk)
    write_jsonl(args.out_dir / "private_quality_key.jsonl", quality_key)
    write_jsonl(args.out_dir / "private_risk_key.jsonl", risk_key)
    write_jsonl(args.out_dir / "automatic_exact_identity_ties.jsonl", automatic_ties)
    write_json(
        args.out_dir / "private_overlap_selection.json",
        {
            "protocol": PROTOCOL,
            "selection": "protocol-hashed proportional allocation within analysis strata",
            "quality_ids": sorted(overlap_quality_ids),
            "risk_ids": sorted(overlap_risk_ids),
        },
    )

    quality_helper = load_module(QUALITY_HELPER, "external_e7_quality_helper")
    risk_helper = load_module(RISK_HELPER, "external_e7_risk_helper")
    quality_html = make_quality_html(quality_helper.QUALITY_HTML)
    risk_html = make_risk_html(risk_helper.RISK_HTML)
    write_html(
        args.out_dir / "human_quality_primary.html",
        template=quality_html,
        protocol=QUALITY_PROTOCOL,
        items=quality_manual,
        role="primary",
        safe_json=quality_helper._safe_script_json,
    )
    write_html(
        args.out_dir / "human_risk_primary.html",
        template=risk_html,
        protocol=RISK_PROTOCOL,
        items=risk_rows,
        role="primary",
        safe_json=quality_helper._safe_script_json,
    )
    write_html(
        args.out_dir / "human_quality_overlap.html",
        template=quality_html,
        protocol=QUALITY_PROTOCOL,
        items=overlap_quality,
        role="independent_overlap",
        safe_json=quality_helper._safe_script_json,
    )
    write_html(
        args.out_dir / "human_risk_overlap.html",
        template=risk_html,
        protocol=RISK_PROTOCOL,
        items=overlap_risk,
        role="independent_overlap",
        safe_json=quality_helper._safe_script_json,
    )

    readme = f"""# PM V1.5 外部 E7 人评顺序\n\n状态：冻结、零 API、等待人工。\n\n1. 主评先完整完成 `human_quality_primary.html`；期间不要打开 risk、private key、E6 或执行目录。\n2. 质量导出后，再完整完成 `human_risk_primary.html`。risk 只评 material interaction-and-grounding risk，不重评质量。\n3. 第二位评审独立完成两个 overlap 页面；质量优先、risk 后做。\n4. 不看中途比例，不按部分结果改样本、PM、generator、资源、阈值或 rubric。\n5. 两位评审完成后只允许一次集中分歧裁决。\n\n主评数量：quality={len(quality_manual)}，risk={len(risk_rows)}。\n独立 overlap：quality={len(overlap_quality)}，risk={len(overlap_risk)}。\n自动 exact-identity ties={len(automatic_ties)}，无需人工但保留在统计分母。\n"""
    (args.out_dir / "README_REVIEW_ORDER_ZH.md").write_text(readme, encoding="utf-8")

    quality_position = Counter(
        "learned_a" if row["learned_is_a"] else "learned_b" for row in quality_key
    )
    quality_manual_by_comparator = Counter(
        f"{row['domain']}:{row['table']}:{comparator}"
        for row in quality_key
        for comparator in row["comparators"]
    )
    quality_ties_by_comparator = Counter(
        f"{row['domain']}:{row['table']}:{comparator}"
        for row in automatic_ties
        for comparator in row["comparators"]
    )
    quality_total_by_comparator = quality_manual_by_comparator.copy()
    quality_total_by_comparator.update(quality_ties_by_comparator)
    risk_by_policy_or_condition = Counter()
    for row in risk_underlying:
        conditions = row["policy_aliases"] or [row["condition"]]
        for condition in conditions:
            risk_by_policy_or_condition[
                f"{row['domain']}:{row['table']}:{condition}"
            ] += 1
    report = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_FINAL_EXTERNAL_E7_PRIMARY_AND_INDEPENDENT_REVIEW",
        "api_calls": 0,
        "sample_identity_preoutcome_sha256": sample_identity_sha256,
        "selected_clusters": {
            "quality": {
                "esconv_dialogues": len(esconv_quality_states),
                "evoemo_users": len(evo_quality_by_user),
                "evoemo_core_states": len(evo_quality_states),
                "evoemo_raw_states": len(evo_raw_quality_states),
            },
            "risk": {
                "esconv_dialogues": len(esconv_risk_states),
                "evoemo_users": len(evo_quality_by_user),
                "evoemo_core_states": len(evo_risk_states),
                "evoemo_raw_states": len(evo_raw_risk_states),
            },
        },
        "quality": {
            "plan_level_comparisons_after_alias_collapse": len(quality_specs),
            "automatic_exact_identity_ties": len(automatic_ties),
            "underlying_nonidentity_pairs": len(quality_rows),
            "primary_public_surfaces": len(quality_manual),
            "exact_duplicate_surfaces_propagated": len(quality_rows)
            - len(quality_manual),
            "position_distribution_underlying": dict(quality_position),
            "strata_underlying": dict(Counter(row["stratum"] for row in quality_key)),
            "per_comparator_manual_human_pairs": dict(
                sorted(quality_manual_by_comparator.items())
            ),
            "per_comparator_automatic_ties": dict(
                sorted(quality_ties_by_comparator.items())
            ),
            "per_comparator_including_automatic_ties": dict(
                sorted(quality_total_by_comparator.items())
            ),
        },
        "risk": {
            "sampled_distinct_physical_calls": len(risk_call_ids),
            "primary_public_surfaces": len(risk_rows),
            "exact_duplicate_surfaces_propagated": len(risk_underlying)
            - len(risk_rows),
            "strata_underlying": dict(
                Counter(row["stratum"] for row in risk_underlying)
            ),
            "per_policy_or_condition_underlying": dict(
                sorted(risk_by_policy_or_condition.items())
            ),
        },
        "independent_overlap": {
            "fraction_target": 0.2,
            "quality": len(overlap_quality),
            "quality_fraction": len(overlap_quality) / len(quality_manual),
            "risk": len(overlap_risk),
            "risk_fraction": len(overlap_risk) / len(risk_rows),
        },
        "attestations": {
            "e6_judge_outcomes_read": False,
            "sampling_used_llm_judge_result": False,
            "sampling_used_quality_or_risk_outcome": False,
            "pm_route_generator_or_resource_changed": False,
            "quality_public_packet_blind_to_policy_component_resource_domain": True,
            "risk_and_quality_constructs_separate": True,
            "es_memeval_qa_excluded_from_e7": True,
        },
        "input_sha256": {
            "contract": sha256_file(CONTRACT),
            "e4_execution_summary": sha256_file(E4_SUMMARY),
            "core_plan": sha256_file(CORE_PLAN),
            "raw_plan": sha256_file(RAW_PLAN),
            "raw_audit": sha256_file(RAW_AUDIT),
            "core_outcomes": sha256_file(CORE_OUTCOMES),
            "raw_outcomes": sha256_file(RAW_OUTCOMES),
            "registered_invalid_generation": sha256_file(INVALID_REGISTRY),
        },
        "output_sha256": {
            name: sha256_file(args.out_dir / name)
            for name in (
                "sample_identity_preoutcome.json",
                "primary_quality_packet.jsonl",
                "primary_risk_packet.jsonl",
                "overlap_quality_packet.jsonl",
                "overlap_risk_packet.jsonl",
                "private_quality_key.jsonl",
                "private_risk_key.jsonl",
                "automatic_exact_identity_ties.jsonl",
                "private_overlap_selection.json",
                "human_quality_primary.html",
                "human_risk_primary.html",
                "human_quality_overlap.html",
                "human_risk_overlap.html",
                "README_REVIEW_ORDER_ZH.md",
            )
        },
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    write_json(args.out_dir / "review_manifest.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
