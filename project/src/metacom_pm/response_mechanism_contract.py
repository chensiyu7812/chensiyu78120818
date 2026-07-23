"""Unified response-mechanism contract binding shared generation code + config.

Historically, ``require_v1_5_retrieval_consistency``
(``scripts/v1_5_create_freeze.py``) only compared three scalar config values
(``strategy_top_k``, ``memory_min_score``, ``strategy_min_score``) between the
development and external retrieval config blocks. That does not bind: the
query-construction code (``retrieval.context_query``), the lexical scorer
implementation (``text.lexical_score``), the MP/MS/ME top-k values
(``retrieval.DEFAULT_MEMORY_TOP_K``), the Strategy Bank content, the Evidence
Filter disabled state, the action-execution code (``action_execution.py``), or
the prompt-compiler code (``prompts.generation_messages``) -- a change to any
of these could silently drift between the internal sweep and an external
(EvoEmo/ESConv) run without being caught by that narrow check. Nor did any
existing check directly compare the internal sweep's actual generator
endpoint against the one an external freeze resolves for itself -- both
happened to be computed from the same config, but nothing ever compared them.

``build_response_mechanism_contract`` fixes this: it produces one payload
binding all of the above, including four deterministic canary prompts
(M0+R0, M0+RS, ME+R0, ME+RS) whose exact ``generation_messages()`` output is
hashed -- so a change to the prompt-construction *code* itself (not just its
config inputs) changes the contract hash, the same way
``pm_v1_5_semantic.py``'s ``SEMANTIC_CANARY_TEXTS`` catches drift in the
semantic-encoder runtime rather than just its declared spec.

Every real stage that generates a supporter response (the internal action
sweep, EvoEmo external generation, and the ESConv external generation runner
once it exists) must record this same contract's ``contract_sha256`` in its
attestation; the freeze must verify all recorded values are identical to the
one it computes for itself, and fail closed otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    DialogueTurn,
    MemoryItem,
    MemorySource,
    RuntimeState,
    SourceCatalog,
    StrategyCard,
    StrategyMode,
)
from .generation_contract import SupporterGenerationContract
from .io import canonical_json, sha256_file, sha256_text
from .prompts import generation_messages
from .retrieval import DEFAULT_MEMORY_TOP_K

RESPONSE_MECHANISM_CONTRACT_PROTOCOL = "pm-v1.5-response-mechanism-contract-v1"

# Files whose exact content determines what a supporter prompt contains,
# independent of config -- a change to any of these must change the contract
# hash even when every config value stays the same.
MECHANISM_CODE_RELATIVE_PATHS = (
    "src/metacom_pm/prompts.py",
    "src/metacom_pm/retrieval.py",
    "src/metacom_pm/action_execution.py",
    "src/metacom_pm/text.py",
    "src/metacom_pm/generation_contract.py",
    "src/metacom_pm/contracts.py",
    "src/metacom_pm/api.py",
)

_CANARY_MEMORY_ITEM = MemoryItem(
    memory_id="mem_" + "ab" * 10,
    source=MemorySource.ME,
    created_session=3,
    timestamp="2026-01-01T00:00:00Z",
    text="canary: user once said the weekend trip helped them reset.",
)

_CANARY_STRATEGY_CARD = StrategyCard(
    strategy_id="strat_" + "cd" * 10,
    strategy_label="reflect",
    retrieval_text="canary retrieval text",
    guidance_text="Reflect the user's stated feeling before offering any suggestion.",
    example_response="It sounds like this week has been really draining for you.",
    source_dialogue_id="canary_dialogue",
    source_turn_index=1,
)

_CANARY_STATE = RuntimeState(
    state_id="state_" + "11" * 10,
    card_id="card_" + "22" * 10,
    user_id="canary_user",
    split="development",
    semantic_family="response_mechanism_canary",
    current_user_text="I've been feeling overwhelmed by everything lately.",
    current_session_history=[
        DialogueTurn(role="user", content="Work has been a lot recently."),
        DialogueTurn(role="assistant", content="That does sound like a lot to carry."),
    ],
    current_session_summary="User has been under sustained stress from work.",
    session_index=1,
    inventory={
        source: SourceCatalog(
            available=source is MemorySource.ME,
            count=1 if source is MemorySource.ME else 0,
        )
        for source in MemorySource
    },
    allowed_actions=["M0+R0", "M0+RS", "ME+R0", "ME+RS"],
)

# (action_id, memory_sources_present, strategy_mode) -- matches the plan's
# "R0、RS、Memory-only、Memory+RS" four-canary requirement.
_CANARY_ACTIONS: tuple[tuple[str, frozenset, StrategyMode], ...] = (
    ("M0+R0", frozenset(), StrategyMode.R0),
    ("M0+RS", frozenset(), StrategyMode.RS),
    ("ME+R0", frozenset({MemorySource.ME}), StrategyMode.R0),
    ("ME+RS", frozenset({MemorySource.ME}), StrategyMode.RS),
)


def _canary_prompt_hashes(*, system_prompt: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for action_id, memory_sources, strategy_mode in _CANARY_ACTIONS:
        memories = [_CANARY_MEMORY_ITEM] if memory_sources else []
        strategies = [_CANARY_STRATEGY_CARD] if strategy_mode is StrategyMode.RS else []
        messages = generation_messages(
            _CANARY_STATE, memories, strategies, system_prompt=system_prompt
        )
        hashes[action_id] = sha256_text(canonical_json(messages))
    return hashes


def build_response_mechanism_contract(
    *,
    project_root: str | Path,
    supporter_generation_contract: SupporterGenerationContract,
    generator_endpoint_sha256: str,
    strategy_bank_sha256: str,
    memory_min_score: float | None,
    strategy_min_score: float | None,
    strategy_top_k: int,
    evidence_filter_enabled: bool,
) -> dict[str, Any]:
    """Build the frozen mechanism contract for one generation stage.

    Every caller (internal sweep, EvoEmo external generation, the ESConv
    external generation runner) must call this with its own resolved
    ``generator_endpoint_sha256``/config values, then record
    ``contract_sha256`` in its own attestation. The freeze must verify all
    recorded values are equal (see ``require_matching_response_mechanism_contract``).

    ``memory_min_score``/``strategy_min_score`` of ``None`` is not the same
    thing as ``0.0``: ``MemoryRetriever``/``StrategyRetriever`` treat ``None``
    as "no score floor at all" (a zero-score candidate is still retrieved),
    while a floor of ``0.0`` excludes zero-score candidates (``score >
    threshold``). Callers must pass the real value through unchanged rather
    than coercing ``None`` to ``0.0``, or this contract would silently claim a
    different retrieval mechanism than the one actually running.
    """

    root = Path(project_root).resolve()
    code_manifest = {
        relative: sha256_file(root / relative)
        for relative in MECHANISM_CODE_RELATIVE_PATHS
    }
    payload: dict[str, Any] = {
        "protocol": RESPONSE_MECHANISM_CONTRACT_PROTOCOL,
        "supporter_generation_treatment_sha256": supporter_generation_contract.digest(),
        "generator_endpoint_sha256": generator_endpoint_sha256,
        "strategy_bank_sha256": strategy_bank_sha256,
        "retrieval_settings": {
            "memory_min_score": (
                float(memory_min_score) if memory_min_score is not None else None
            ),
            "strategy_min_score": (
                float(strategy_min_score) if strategy_min_score is not None else None
            ),
            "strategy_top_k": int(strategy_top_k),
        },
        "memory_top_k_by_source": {
            source.value: k
            for source, k in sorted(
                DEFAULT_MEMORY_TOP_K.items(), key=lambda pair: pair[0].value
            )
        },
        "evidence_filter_enabled": bool(evidence_filter_enabled),
        "shared_code_manifest": code_manifest,
        "canary_prompts": _canary_prompt_hashes(
            system_prompt=supporter_generation_contract.system_prompt
        ),
    }
    payload["contract_sha256"] = sha256_text(canonical_json(payload))
    return payload


def _self_hash(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "contract_sha256"}
    return sha256_text(canonical_json(body))


def require_matching_response_mechanism_contract(
    *,
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    context: str,
) -> None:
    """Fail closed unless two mechanism contracts are byte-for-byte identical.

    Recomputes each side's own hash from its full payload (excluding the
    self-referential ``contract_sha256`` field) rather than trusting the
    declared ``contract_sha256`` field at face value -- a payload whose
    declared hash is stale, tampered, or copy-pasted from elsewhere would
    otherwise pass a check that only compared the two declared hash
    strings. Also compares the full canonical payloads directly, so a
    difference in any field is caught even if hash computation itself ever
    had some quirk that happened to coincide on both sides.
    """

    for label, payload in (("expected", expected), ("actual", actual)):
        declared = payload.get("contract_sha256")
        recomputed = _self_hash(payload)
        if declared != recomputed:
            raise RuntimeError(
                f"{context}: {label} response mechanism contract's declared "
                "contract_sha256 does not match its own recomputed payload "
                "hash -- the contract is stale, tampered, or was not built "
                "by build_response_mechanism_contract"
            )
    if canonical_json(expected) != canonical_json(actual):
        raise RuntimeError(
            f"{context}: response mechanism contract does not match the frozen "
            "value -- development, EvoEmo, and ESConv generation must share "
            "the identical retrieval/prompt/action-execution mechanism, not "
            "just the same config values"
        )
