"""Outcome-blind static asset bindings for the V5.3 release.

This is deliberately a *partial* release identity.  It freezes assets that
are already decided (the six-card Strategy Bank, BGE-M3 for MS, the retained
production lexical/typed-tier ranker for ME, and the shared Step2/
accountability/baseline implementations) while leaving the Step2 recovery
policy visibly unresolved.  A formal P2 release must fill that field and
produce a new identity; callers cannot silently substitute another Bank,
retriever, encoder snapshot, or implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .contracts import StrictModel
from .io import canonical_json, read_json, sha256_file, stable_hex
from .pm_v1_5_semantic import semantic_snapshot_tree_sha256
from .v1_5_v5_3_response_baselines import POLICIES
from .v1_5_v5_3_semantic_ms_retrieval import DEFAULT_BGE_M3_SNAPSHOT


STATIC_RELEASE_PROTOCOL = "pm-v1.5-v5.3-static-release-bindings-v1"
EXPECTED_STRATEGY_BANK_SHA256 = (
    "04c3af44d54ae9875cd817364ff46e90aa954e0bee81b295af279b1b38964d24"
)
EXPECTED_STRATEGY_CARD_COUNT = 6
EXPECTED_BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


class FileBinding(StrictModel):
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class StrategyBankBinding(FileBinding):
    card_count: int = Field(gt=0)
    # JSON artifacts necessarily encode this collection as an array.  Keep the
    # contract round-trippable under StrictModel instead of requiring callers
    # to silently coerce an on-disk list back to a tuple.
    move_ids: list[str]


class SemanticRetrieverBinding(StrictModel):
    component: Literal["MS"] = "MS"
    method: Literal["BGE_M3_COSINE_FULL_CAUSAL_POOL"] = (
        "BGE_M3_COSINE_FULL_CAUSAL_POOL"
    )
    snapshot_path: str = Field(min_length=1)
    snapshot_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    snapshot_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation: FileBinding
    candidate_discovery_implementation: FileBinding
    local_files_only: Literal[True] = True
    formal_runner_wiring_required: Literal[True] = True


class MeRetrieverBinding(StrictModel):
    component: Literal["ME"] = "ME"
    method: Literal[
        "PRODUCTION_LEXICAL_TYPED_TIER_EXACT_RANK1_COMPILE_OR_OFF"
    ] = "PRODUCTION_LEXICAL_TYPED_TIER_EXACT_RANK1_COMPILE_OR_OFF"
    candidate_discovery_implementation: FileBinding
    atomic_compiler_implementation: FileBinding
    qualification_report: FileBinding
    rank2_promotion_allowed: Literal[False] = False
    bge_reranker_adopted: Literal[False] = False
    formal_runner_wiring_required: Literal[True] = True


class StaticReleaseBindings(StrictModel):
    protocol: Literal["pm-v1.5-v5.3-static-release-bindings-v1"] = (
        STATIC_RELEASE_PROTOCOL
    )
    status: Literal["STATIC_BINDINGS_FROZEN_RECOVERY_PENDING"] = (
        "STATIC_BINDINGS_FROZEN_RECOVERY_PENDING"
    )
    strategy_bank: StrategyBankBinding
    ms_retriever: SemanticRetrieverBinding
    me_retriever: MeRetrieverBinding
    step2_recovery_policy_status: Literal[
        "PENDING_DEVELOPMENT_ONLY_REWRITE_VS_DIRECT_FALLBACK_COMPARISON"
    ]
    shared_implementations: dict[str, FileBinding]
    response_baselines: list[str]
    generated_response_or_quality_risk_outcome_read: Literal[False] = False
    api_calls: Literal[0] = 0
    release_identity: str = Field(pattern=r"^v53static_[0-9a-f]{24}$")

    @model_validator(mode="after")
    def coherent(self):
        if self.response_baselines != list(POLICIES):
            raise ValueError("response baseline order differs from the shared planner")
        if self.strategy_bank.card_count != EXPECTED_STRATEGY_CARD_COUNT:
            raise ValueError("V5.3 primary must use the frozen six-card Strategy Bank")
        if self.strategy_bank.sha256 != EXPECTED_STRATEGY_BANK_SHA256:
            raise ValueError("strategy bank hash differs from the leader decision")
        data = self.model_dump(mode="json", exclude={"release_identity"})
        expected = "v53static_" + stable_hex(canonical_json(data), n=24)
        if self.release_identity != expected:
            raise ValueError("release_identity does not match the bound assets")
        return self


def _file_binding(root: Path, relative_path: str) -> FileBinding:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return FileBinding(relative_path=relative_path, sha256=sha256_file(path))


def build_static_release_bindings(root: str | Path) -> StaticReleaseBindings:
    project_root = Path(root).resolve()
    bank_relative = "data/strategy/strategy_cards_v1_5_minimal.jsonl"
    bank_path = project_root / bank_relative
    freeze = read_json(project_root / "data/strategy/strategy_rag_v1_5_minimal_freeze.json")
    observed_bank_sha = sha256_file(bank_path)
    configured_bank = freeze["bank"]
    move_ids = [str(value) for value in configured_bank["move_ids"]]
    card_count = sum(1 for line in bank_path.open(encoding="utf-8") if line.strip())
    if configured_bank["sha256"] != observed_bank_sha:
        raise ValueError("minimal Strategy Bank differs from its existing freeze")
    if configured_bank["card_count"] != card_count:
        raise ValueError("minimal Strategy Bank row count differs from its freeze")

    snapshot = DEFAULT_BGE_M3_SNAPSHOT.resolve()
    if not snapshot.is_dir() or snapshot.name != EXPECTED_BGE_M3_REVISION:
        raise RuntimeError("the decided local BGE-M3 snapshot is unavailable or drifted")

    implementations = {
        "typed_step2": _file_binding(
            project_root, "src/metacom_pm/v1_5_v5_3_typed_response_program.py"
        ),
        "accountability_schema": _file_binding(
            project_root, "src/metacom_pm/v1_5_v5_3_accountability.py"
        ),
        "accountability_runner": _file_binding(
            project_root, "src/metacom_pm/v1_5_v5_3_accountability_runner.py"
        ),
        "response_baselines": _file_binding(
            project_root, "src/metacom_pm/v1_5_v5_3_response_baselines.py"
        ),
        "candidate_layer_responsibility": _file_binding(
            project_root,
            "src/metacom_pm/v1_5_v5_3_candidate_layer_responsibility.py",
        ),
    }
    payload = {
        "protocol": STATIC_RELEASE_PROTOCOL,
        "status": "STATIC_BINDINGS_FROZEN_RECOVERY_PENDING",
        "strategy_bank": StrategyBankBinding(
            relative_path=bank_relative,
            sha256=observed_bank_sha,
            card_count=card_count,
            move_ids=move_ids,
        ),
        "ms_retriever": SemanticRetrieverBinding(
            snapshot_path=str(snapshot),
            snapshot_revision=snapshot.name,
            snapshot_tree_sha256=semantic_snapshot_tree_sha256(snapshot),
            implementation=_file_binding(
                project_root,
                "src/metacom_pm/v1_5_v5_3_semantic_ms_retrieval.py",
            ),
            candidate_discovery_implementation=_file_binding(
                project_root, "src/metacom_pm/v1_5_candidate_discovery.py"
            ),
        ),
        "me_retriever": MeRetrieverBinding(
            candidate_discovery_implementation=_file_binding(
                project_root, "src/metacom_pm/v1_5_candidate_discovery.py"
            ),
            atomic_compiler_implementation=_file_binding(
                project_root, "src/metacom_pm/v1_5_v5_2_atomic_memory.py"
            ),
            qualification_report=_file_binding(
                project_root,
                "docs/PM_V1_5_V5_3_ME_RERANKER_QUALIFICATION_20260806_ZH.md",
            ),
        ),
        "step2_recovery_policy_status": (
            "PENDING_DEVELOPMENT_ONLY_REWRITE_VS_DIRECT_FALLBACK_COMPARISON"
        ),
        "shared_implementations": implementations,
        "response_baselines": list(POLICIES),
        "generated_response_or_quality_risk_outcome_read": False,
        "api_calls": 0,
    }
    payload["release_identity"] = "v53static_" + stable_hex(
        canonical_json(
            {
                key: value.model_dump(mode="json")
                if isinstance(value, StrictModel)
                else {
                    item_key: item_value.model_dump(mode="json")
                    for item_key, item_value in value.items()
                }
                if key == "shared_implementations"
                else value
                for key, value in payload.items()
            }
        ),
        n=24,
    )
    return StaticReleaseBindings.model_validate(payload)
