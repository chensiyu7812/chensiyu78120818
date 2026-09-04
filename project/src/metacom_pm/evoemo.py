from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import math
import time

from .api import (
    Endpoint,
    OpenAICompatibleClient,
    chat_request_payload,
    make_client,
    request_log,
    require_reported_usage,
)
from .artifacts import create_artifact_attestation, require_artifact_attestation
from .attempt_ledger import PersistentAttemptLedger
from .contracts import (
    CostRecord,
    DialogueTurn,
    MemoryItem,
    MemorySource,
    RuntimeState,
    SourceCatalog,
    StrategyCard,
    StrategyMode,
    parse_action_id,
)
from .io import (
    append_jsonl,
    canonical_json,
    ensure_run_manifest,
    index_jsonl_unique,
    iter_jsonl,
    load_done_keys,
    read_json,
    sha256_file,
    sha256_text,
    stable_hex,
    utc_now,
    write_json,
    write_jsonl,
)
from .fixed_seeker_contract import FixedSeekerGenerationContract
from .policies import FixedPolicy, LearnedPMPolicy, RuleConfig, StrongRulePolicy
from .prompts import OFFICIAL_ESMEM_SYSTEM, SELECTIVE_ESMEM_SYSTEM, generation_messages
from .retrieval import MemoryRetriever, StrategyRetriever, context_query
from .text import estimate_tokens, lexical_score, normalize_space
from .training import PMModel


NEUTRAL_INITIAL_GREETING = "Hi, I'm here with you. What would you like to talk about today?"
_NEUTRAL_TRACK_PROBES = (
    "I'm listening. What feels most important to share right now?",
    "What has that experience been like for you?",
    "What part of it has been weighing on you most?",
    "How has this been affecting you lately?",
    "What do you wish felt different at this point?",
    "What kind of support would feel most useful right now?",
)

FIXED_SEEKER_V22_STAGE = "evoemo_fixed_seeker_tracks_v22"
FIXED_SEEKER_V22_DRY_RUN_PROTOCOL = "pm-v2.2-fixed-seeker-dry-run-v1"
FIXED_SEEKER_V22_LOGICAL_CALL_PROTOCOL = "pm-v2.2-fixed-seeker-logical-call-v1"
FIXED_SEEKER_COST_PLANNING_PROTOCOL = (
    "pm-v2.2-fixed-seeker-cost-planning-v1"
)
FIXED_SEEKER_INPUT_BOUND_FORMULA = (
    "ceil(input_token_safety_factor * "
    "(static_request_tokens_with_empty_prior_seeker_content + "
    "prior_turn_count * max_output_tokens))"
)


def fixed_seeker_cost_planning_contract(
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize the frozen positive-price seeker cost contract."""

    data = dict(raw)
    expected_keys = {
        "protocol",
        "pricing_usd_per_mtok",
        "input_token_safety_factor",
        "fail_on_reported_input_overrun",
    }
    normalized_keys = expected_keys | {"per_call_input_bound_formula"}
    if frozenset(data) not in {
        frozenset(expected_keys),
        frozenset(normalized_keys),
    }:
        raise ValueError(
            "fixed_seeker_cost_planning keys differ from the frozen contract"
        )
    if (
        "per_call_input_bound_formula" in data
        and data["per_call_input_bound_formula"]
        != FIXED_SEEKER_INPUT_BOUND_FORMULA
    ):
        raise ValueError("fixed-seeker input-bound formula changed")
    if data["protocol"] != FIXED_SEEKER_COST_PLANNING_PROTOCOL:
        raise ValueError("unsupported fixed-seeker cost-planning protocol")
    prices_raw = data["pricing_usd_per_mtok"]
    if not isinstance(prices_raw, Mapping) or set(prices_raw) != {
        "input",
        "output",
    }:
        raise ValueError("fixed-seeker pricing must contain input and output")
    prices = {
        "input": float(prices_raw["input"]),
        "output": float(prices_raw["output"]),
    }
    if prices != {"input": 0.15, "output": 0.60}:
        raise ValueError(
            "fixed-seeker pricing must equal the frozen positive 0.15/0.60 proxy"
        )
    safety_factor = float(data["input_token_safety_factor"])
    if not math.isfinite(safety_factor) or safety_factor != 1.50:
        raise ValueError("fixed-seeker input-token safety factor must equal 1.50")
    if data["fail_on_reported_input_overrun"] is not True:
        raise ValueError("fixed-seeker reported input-token overrun must fail closed")
    return {
        "protocol": FIXED_SEEKER_COST_PLANNING_PROTOCOL,
        "pricing_usd_per_mtok": prices,
        "input_token_safety_factor": safety_factor,
        "fail_on_reported_input_overrun": True,
        "per_call_input_bound_formula": FIXED_SEEKER_INPUT_BOUND_FORMULA,
    }


def load_evoemo(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 18:
        raise ValueError("expected EvoEmo list with exactly 18 users")
    total_sessions = sum(len(x.get("dialog_history") or []) for x in data)
    total_topics = sum(len(x.get("subsequent_topics") or []) for x in data)
    if total_sessions != 401 or total_topics != 34:
        raise ValueError(
            f"unexpected EvoEmo statistics: sessions={total_sessions}, topics={total_topics}"
        )
    return data


def _opaque_memory_id(user_id: str, source: str, key: str) -> str:
    return f"mem_{stable_hex('evo', user_id, source, key, n=20)}"


def build_evo_memory(user: dict[str, Any]) -> tuple[list[MemoryItem], list[dict[str, Any]]]:
    """Build deployable memory only from profile and past dialogue history.

    Event timelines, observation annotations, related-session labels, QA
    evidence, reference answers and future topics are evaluator-only and never
    enter the policy/generator view.
    """
    user_id = str(user["id"])
    items: list[MemoryItem] = []
    basic = user.get("basic_info") or {}
    # Preserve profile fields as separate memory units.  A single concatenated
    # profile made external MP-count=1 while training MP contained many units,
    # creating an avoidable representation/domain shortcut.
    for key, value in basic.items():
        if value in (None, ""):
            continue
        items.append(MemoryItem(
            memory_id=_opaque_memory_id(user_id, "MP", str(key)),
            source=MemorySource.MP,
            created_session=0,
            text=f"{str(key).replace('_', ' ').title()}: {value}",
        ))

    session_docs: list[dict[str, Any]] = []
    sessions = user.get("dialog_history") or []
    for index, session in enumerate(sessions, 1):
        timestamp = str(session.get("timestamp") or "")
        session_id = str(session.get("id") or f"session_{index}")
        summary = normalize_space(session.get("summary") or "")
        if summary:
            items.append(MemoryItem(
                memory_id=_opaque_memory_id(user_id, "MS", session_id),
                source=MemorySource.MS,
                created_session=index,
                timestamp=timestamp or None,
                text=summary,
            ))
        seeker_turns = [
            normalize_space(turn.get("content") or "")
            for turn in (session.get("dialogue") or [])
            if turn.get("role") == "seeker" and normalize_space(turn.get("content") or "")
        ]
        if seeker_turns:
            items.append(MemoryItem(
                memory_id=_opaque_memory_id(user_id, "ME", session_id),
                source=MemorySource.ME,
                created_session=index,
                timestamp=timestamp or None,
                text=" ".join(seeker_turns),
            ))
        dialogue_text = "\n".join(
            f"{turn.get('role')}: {normalize_space(turn.get('content') or '')}"
            for turn in (session.get("dialogue") or [])
        )
        session_docs.append({
            "session_id": session_id,
            "session_index": index,
            "timestamp": timestamp,
            "summary": summary,
            "text": dialogue_text,
        })
    return items, session_docs


def evaluator_context(user: dict[str, Any], topic: dict[str, Any]) -> dict[str, Any]:
    """Ground truth available only to post-hoc evaluators.

    The evaluator sees enough authorized history to verify facts and temporal
    updates, but no policy name, action, selected evidence or model score.
    """
    session_by_id = {
        str(row.get("id")): row for row in (user.get("dialog_history") or [])
    }
    timeline = [
        {
            "session_id": str(row.get("id")),
            "timestamp": row.get("timestamp"),
            "summary": row.get("summary"),
        }
        for row in (user.get("dialog_history") or [])
    ]
    related_sessions = []
    for session_id in topic.get("related_sessions") or []:
        session = session_by_id.get(str(session_id))
        if not session:
            continue
        related_sessions.append({
            "session_id": str(session.get("id")),
            "timestamp": session.get("timestamp"),
            "summary": session.get("summary"),
            "dialogue": session.get("dialogue") or [],
            "observations": session.get("observation") or [],
        })
    return {
        "user_profile": user.get("basic_info") or {},
        "past_session_timeline": timeline,
        "detailed_related_sessions": related_sessions,
        "current_topic": {
            "topic": topic.get("topic"),
            "psychological_condition": topic.get("psychological_condition"),
            "physical_condition": topic.get("physical_condition"),
            "more_details": topic.get("more_details"),
        },
        "evaluator_only": True,
    }


def _catalog(items: Sequence[MemoryItem], source: MemorySource, session_index: int) -> SourceCatalog:
    selected = [x for x in items if x.source is source]
    # Imported lazily so the legacy EvoEmo module and PM-v2 data adapter share one
    # deployable source-catalog contract without introducing an import cycle.
    from .pm_v2_data import build_deployable_catalog_statistics

    statistics = build_deployable_catalog_statistics(
        texts=[item.text for item in selected],
        created_sessions=[item.created_session for item in selected],
        session_index=session_index,
    )
    return SourceCatalog(
        available=bool(statistics["available"]),
        count=int(statistics["count"]),
        min_age_sessions=statistics["min_age_sessions"],
        max_age_sessions=statistics["max_age_sessions"],
        estimated_tokens=int(statistics["estimated_tokens"]),
        catalog_fingerprint=list(statistics["catalog_fingerprint"]),
    )


def make_evo_runtime_state(
    user: dict[str, Any],
    topic: dict[str, Any],
    conversation: list[dict[str, str]],
    current_user_text: str,
    items: list[MemoryItem],
    turn_index: int,
    condition: str,
    *,
    track_id: str | None = None,
    fixed_open_loop: bool = False,
) -> RuntimeState:
    session_index = max((x.created_session for x in items), default=0) + 1
    # The literal condition label is excluded.  The runtime state ID still
    # includes the actually observed dialogue history, so after turn 1 it may
    # legitimately differ across policies because prior supporter responses
    # are part of the treatment trajectory.  A separate exogenous state ID
    # below identifies the common fixed seeker input for paired analyses.
    transcript_hash = sha256_text(canonical_json(conversation))
    state_id = f"state_{stable_hex('evo-state', user['id'], topic['idx'], track_id or '', turn_index, transcript_hash, current_user_text, n=20)}"
    card_id = f"card_{stable_hex(state_id, n=20)}"
    exogenous_history = [
        normalize_space(row.get("content") or "")
        for row in conversation if row.get("role") == "seeker"
    ]
    exogenous_state_id = f"state_{stable_hex('evo-exogenous-state', user['id'], topic['idx'], track_id or '', turn_index, exogenous_history, current_user_text, n=20)}"
    inventory = {source: _catalog(items, source, session_index) for source in MemorySource}
    available = {src for src, cat in inventory.items() if cat.available}
    from .contracts import ACTION_MEMORY_MAP, canonical_action_id
    allowed = sorted(
        canonical_action_id(sources, strategy)
        for sources in ACTION_MEMORY_MAP.values()
        if sources <= available
        for strategy in StrategyMode
    )
    prior = list(conversation)
    if (
        prior
        and prior[-1].get("role") == "seeker"
        and normalize_space(prior[-1].get("content") or "")
        == normalize_space(current_user_text)
    ):
        prior = prior[:-1]
    history = [
        DialogueTurn(
            role="user" if row["role"] == "seeker" else "assistant",
            content=row["content"],
        )
        for row in prior[-8:]
    ]
    return RuntimeState(
        state_id=state_id,
        card_id=card_id,
        user_id=str(user["id"]),
        split="evoemo_test",
        semantic_family="evoemo_dialogue_generation",
        current_user_text=current_user_text,
        current_session_history=history,
        current_session_summary="",
        session_index=session_index,
        inventory=inventory,
        allowed_actions=allowed,
        provenance={
            "topic_index": int(topic["idx"]),
            "turn_index": int(turn_index),
            "track_id": track_id,
            "condition_label_not_present_in_pm_state": True,
            "runtime_state_includes_prior_treatment_history": not fixed_open_loop,
            "fixed_open_loop_context": fixed_open_loop,
            "fixed_context_sha256": (
                transcript_hash if fixed_open_loop else None
            ),
            "exogenous_state_id": exogenous_state_id,
        },
    )


def seeker_system_prompt(user: dict[str, Any], topic: dict[str, Any]) -> str:
    private = evaluator_context(user, topic)
    return f"""Role-play the emotional-support seeker. Stay faithful to the
private profile and topic. Do not mention that this is a benchmark, do not
reveal the entire hidden card at once, and do not discuss retrieval, policies,
or experimental conditions. Respond naturally to the supporter's latest
message in at most 60 tokens. Do not become artificially agreeable merely
because the supporter suggests something.

Private scenario:
{json.dumps(private, ensure_ascii=False, indent=2)}
"""


def _seeker_next(
    client: OpenAICompatibleClient,
    endpoint: Endpoint,
    system_prompt: str,
    conversation: list[dict[str, str]],
    *,
    seed: int,
    raw_log_path: Path,
    record_ids: dict[str, Any],
) -> str:
    if not conversation or conversation[-1].get("role") != "supporter":
        raise ValueError("seeker generation requires a transcript ending in a supporter turn")
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    # For the seeker model, supporter utterances are user messages and seeker
    # utterances are assistant messages. Every transcript row is included once.
    for row in conversation:
        messages.append({
            "role": "assistant" if row["role"] == "seeker" else "user",
            "content": row["content"],
        })
    result, _ = client.chat(
        messages,
        temperature=0.2,
        max_tokens=60,
        seed=seed,
        response_schema=None,
        retries=3,
    )
    append_jsonl(raw_log_path, request_log(
        stage="evoemo_seeker",
        endpoint=endpoint,
        messages=messages,
        result=result,
        parsed=None,
        error=None,
        prompt_hash=sha256_text(canonical_json(messages)),
        record_ids=record_ids,
    ))
    return normalize_space(result.text)


def _session_rag(query: str, session_docs: list[dict[str, Any]], top_k: int = 4) -> list[MemoryItem]:
    ranked = sorted(
        session_docs,
        key=lambda row: (
            lexical_score(query, row["summary"] + "\n" + row["text"]),
            row["session_index"],
        ),
        reverse=True,
    )[:top_k]
    return [
        MemoryItem(
            memory_id=_opaque_memory_id("session-rag", "ME", row["session_id"]),
            source=MemorySource.ME,
            created_session=row["session_index"],
            timestamp=row["timestamp"] or None,
            text=row["text"],
        )
        for row in ranked
    ]


def _track_key(user_id: str, topic_index: int, seed: int, simulator_id: str) -> tuple[str, int, int, str]:
    return str(user_id), int(topic_index), int(seed), str(simulator_id)


def build_fixed_seeker_tracks(
    evoemo_path: str | Path,
    out_dir: str | Path,
    *,
    seeker_endpoint: Endpoint,
    simulator_id: str,
    max_turns: int = 10,
    seeds: Sequence[int] = (101,),
    max_scenarios: int | None = None,
    overwrite: bool = False,
    study_freeze_sha256: str | None = None,
) -> dict[str, Any]:
    """Generate legacy policy-independent tracks for historical reproduction.

    This pre-V2.2 path retains the historical 60-token provider cap and retry
    behavior. New PM-v2 runs must use :func:`build_fixed_seeker_tracks_v22`,
    whose treatment, dry-run acceptance, finish gate, and physical-attempt
    ledger are frozen explicitly.
    """
    users = load_evoemo(evoemo_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = out_dir / "fixed_seeker_tracks.jsonl"
    raw_path = out_dir / "raw_seeker_calls.jsonl"
    summary_path = out_dir / "summary.json"
    manifest_path = out_dir / "run_manifest.json"
    attestation_path = out_dir / "artifact_attestation.json"
    if overwrite:
        for path in (tracks_path, raw_path, summary_path, manifest_path, attestation_path):
            if path.exists():
                path.unlink()
    ensure_run_manifest(manifest_path, {
        "stage": "evoemo_fixed_seeker_tracks",
        "evoemo_sha256": sha256_file(evoemo_path),
        "seeker_model": seeker_endpoint.model,
        "seeker_family": seeker_endpoint.family,
        "seeker_base_url": seeker_endpoint.base_url,
        "simulator_id": simulator_id,
        "max_turns": int(max_turns),
        "seeds": [int(x) for x in seeds],
        "max_scenarios": max_scenarios,
        "scaffold_sha256": sha256_text(canonical_json([NEUTRAL_INITIAL_GREETING, *_NEUTRAL_TRACK_PROBES])),
        "study_freeze_sha256": study_freeze_sha256,
    })
    done = load_done_keys(tracks_path, ("user_id", "topic_index", "seed", "simulator_id"))
    client = OpenAICompatibleClient(seeker_endpoint)
    failures: list[dict[str, Any]] = []
    scenarios: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for user in users:
        for topic in user.get("subsequent_topics") or []:
            scenarios.append((user, topic))
    if max_scenarios is not None:
        scenarios = scenarios[:max_scenarios]
    try:
        for user, topic in scenarios:
            for seed in seeds:
                key = _track_key(str(user["id"]), int(topic["idx"]), int(seed), simulator_id)
                if key in done:
                    continue
                conversation = [{"role": "supporter", "content": NEUTRAL_INITIAL_GREETING}]
                seeker_turns: list[str] = []
                try:
                    for turn_index in range(1, max_turns + 1):
                        message = _seeker_next(
                            client,
                            seeker_endpoint,
                            seeker_system_prompt(user, topic),
                            conversation,
                            seed=int(seed) + turn_index,
                            raw_log_path=raw_path,
                            record_ids={
                                "user_id": str(user["id"]),
                                "topic_index": int(topic["idx"]),
                                "seed": int(seed),
                                "simulator_id": simulator_id,
                                "turn_index": turn_index,
                                "track_generation": True,
                            },
                        )
                        seeker_turns.append(message)
                        conversation.append({"role": "seeker", "content": message})
                        if turn_index < max_turns:
                            probe = _NEUTRAL_TRACK_PROBES[(turn_index - 1) % len(_NEUTRAL_TRACK_PROBES)]
                            conversation.append({"role": "supporter", "content": probe})
                    track_id = f"track_{stable_hex(user['id'], topic['idx'], seed, simulator_id, canonical_json(seeker_turns), n=20)}"
                    append_jsonl(tracks_path, {
                        "track_id": track_id,
                        "user_id": str(user["id"]),
                        "topic_index": int(topic["idx"]),
                        "seed": int(seed),
                        "simulator_id": simulator_id,
                        "seeker_model": seeker_endpoint.model,
                        "seeker_family": seeker_endpoint.family,
                        "initial_greeting": NEUTRAL_INITIAL_GREETING,
                        "seeker_turns": seeker_turns,
                        "elicitation_scaffold": "deterministic_generic_open_loop",
                        "causal_use": "replayed unchanged to all policies",
                    })
                except Exception as exc:
                    failures.append({
                        "user_id": str(user["id"]),
                        "topic_index": int(topic["idx"]),
                        "seed": int(seed),
                        "simulator_id": simulator_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
    finally:
        client.close()
    expected = len(scenarios) * len(seeds)
    completed = len(load_done_keys(tracks_path, ("user_id", "topic_index", "seed", "simulator_id")))
    summary = {
        "status": "COMPLETE" if completed == expected and not failures else "INCOMPLETE",
        "simulator_id": simulator_id,
        "seeker_model": seeker_endpoint.model,
        "seeker_family": seeker_endpoint.family,
        "expected_tracks": expected,
        "completed_tracks": completed,
        "max_turns": max_turns,
        "failures": failures,
        "interpretation": "fixed-input causal track; later seeker turns do not react to evaluated policy replies",
    }
    write_json(summary_path, summary)
    if summary["status"] != "COMPLETE":
        raise RuntimeError(f"fixed seeker track generation incomplete: {len(failures)} failures")
    create_artifact_attestation(
        attestation_path,
        stage="evoemo_fixed_seeker_tracks",
        inputs={"evoemo": evoemo_path, "run_manifest": manifest_path},
        outputs={
            "tracks": (tracks_path, True),
            "raw_calls": (raw_path, True),
            "summary": (summary_path, False),
        },
        parameters={
            "simulator_id": simulator_id,
            "max_turns": max_turns,
            "seeds": [int(x) for x in seeds],
        },
        expected={"tracks": expected, "turns_per_track": max_turns},
        study_freeze_sha256=study_freeze_sha256,
    )
    return summary


def _fixed_seeker_v22_scenarios(
    evoemo_path: str | Path, *, max_scenarios: int | None
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    scenarios = [
        (user, topic)
        for user in load_evoemo(evoemo_path)
        for topic in (user.get("subsequent_topics") or [])
    ]
    if max_scenarios is not None:
        if max_scenarios < 1:
            raise ValueError("max_scenarios must be positive when provided")
        scenarios = scenarios[: int(max_scenarios)]
    return scenarios


def plan_fixed_seeker_tracks_v22(
    evoemo_path: str | Path,
    *,
    seeker_endpoint: Endpoint,
    contract: FixedSeekerGenerationContract,
    cost_planning: Mapping[str, Any],
    simulator_id: str,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
    max_turns: int = 10,
    seeds: Sequence[int] = (101,),
    max_scenarios: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build a deterministic, no-client call plan and exact output budget.

    Later prompts contain earlier generated seeker turns, so their provider
    request hashes cannot honestly be predicted before generation.  The dry
    run instead freezes every logical coordinate, endpoint/treatment identity,
    one-attempt bound, and exact maximum output-token budget.  Each actual
    prompt hash is fsynced by the attempt ledger immediately before its HTTP
    request.
    """

    if max_turns < 1:
        raise ValueError("fixed-seeker max_turns must be positive")
    normalized_seeds = [int(value) for value in seeds]
    if not normalized_seeds or len(normalized_seeds) != len(set(normalized_seeds)):
        raise ValueError("fixed-seeker seeds must be non-empty and unique")
    if not str(simulator_id).strip():
        raise ValueError("fixed-seeker simulator_id must be non-empty")
    normalized_cost_planning = fixed_seeker_cost_planning_contract(cost_planning)
    if int(max_api_calls) < 1 or int(max_input_tokens_per_call) < 1:
        raise ValueError("fixed-seeker API and input-token caps must be positive")
    if not math.isfinite(float(max_estimated_usd)) or float(
        max_estimated_usd
    ) <= 0.0:
        raise ValueError("fixed-seeker max_estimated_usd must be finite and positive")
    prices = normalized_cost_planning["pricing_usd_per_mtok"]
    safety_factor = float(
        normalized_cost_planning["input_token_safety_factor"]
    )
    cost_planning_sha256 = sha256_text(
        canonical_json(normalized_cost_planning)
    )

    evoemo_sha256 = sha256_file(evoemo_path)
    scaffold_sha256 = sha256_text(
        canonical_json([NEUTRAL_INITIAL_GREETING, *_NEUTRAL_TRACK_PROBES])
    )
    bound = contract.bind_endpoint(contract.seeker_endpoint, seeker_endpoint)
    bound_payload = bound.payload()
    bound_sha256 = bound.digest()
    rows: list[dict[str, Any]] = []
    scenarios = _fixed_seeker_v22_scenarios(
        evoemo_path, max_scenarios=max_scenarios
    )
    for user, topic in scenarios:
        system_prompt = contract.render_system_prompt(
            evaluator_context(user, topic)
        )
        private_scenario_sha256 = sha256_text(
            canonical_json(evaluator_context(user, topic))
        )
        for seed in normalized_seeds:
            for turn_index in range(1, int(max_turns) + 1):
                record_ids = {
                    "user_id": str(user["id"]),
                    "topic_index": int(topic["idx"]),
                    "seed": int(seed),
                    "simulator_id": str(simulator_id),
                    "turn_index": int(turn_index),
                }
                static_conversation: list[dict[str, str]] = [
                    {"role": "supporter", "content": NEUTRAL_INITIAL_GREETING}
                ]
                for prior_index in range(turn_index - 1):
                    # Historical seeker text is unknown at dry-run time. Its
                    # provider-token contribution is added separately using
                    # the exact 300-token completion ceiling.
                    static_conversation.append(
                        {"role": "seeker", "content": ""}
                    )
                    static_conversation.append(
                        {
                            "role": "supporter",
                            "content": _NEUTRAL_TRACK_PROBES[
                                prior_index % len(_NEUTRAL_TRACK_PROBES)
                            ],
                        }
                    )
                static_messages = _fixed_seeker_v22_messages(
                    system_prompt, static_conversation
                )
                static_request_payload = chat_request_payload(
                    seeker_endpoint,
                    static_messages,
                    temperature=contract.temperature,
                    max_tokens=contract.max_output_tokens,
                    seed=int(seed) + int(turn_index),
                    response_schema=None,
                )
                static_request_tokens = estimate_tokens(
                    canonical_json(static_request_payload)
                )
                history_completion_token_cap = (
                    (int(turn_index) - 1) * contract.max_output_tokens
                )
                maximum_input_tokens = int(
                    math.ceil(
                        safety_factor
                        * (
                            static_request_tokens
                            + history_completion_token_cap
                        )
                    )
                )
                maximum_cost_usd = (
                    maximum_input_tokens / 1_000_000 * prices["input"]
                    + contract.max_output_tokens
                    / 1_000_000
                    * prices["output"]
                )
                call_identity = {
                    "protocol": FIXED_SEEKER_V22_LOGICAL_CALL_PROTOCOL,
                    "record_ids": record_ids,
                    "evoemo_sha256": evoemo_sha256,
                    "private_scenario_sha256": private_scenario_sha256,
                    "scaffold_sha256": scaffold_sha256,
                    "fixed_seeker_generation_contract": bound_payload,
                    "fixed_seeker_generation_contract_sha256": bound_sha256,
                    "fixed_seeker_cost_planning": normalized_cost_planning,
                    "fixed_seeker_cost_planning_sha256": (
                        cost_planning_sha256
                    ),
                    "maximum_input_tokens": maximum_input_tokens,
                    "turn_seed": int(seed) + int(turn_index),
                }
                rows.append(
                    {
                        **record_ids,
                        "turn_seed": int(seed) + int(turn_index),
                        "static_request_tokens_with_empty_prior_seeker_content": (
                            static_request_tokens
                        ),
                        "history_completion_token_cap": (
                            history_completion_token_cap
                        ),
                        "maximum_input_tokens": maximum_input_tokens,
                        "maximum_output_tokens": contract.max_output_tokens,
                        "maximum_cost_usd": maximum_cost_usd,
                        "maximum_physical_attempts": (
                            contract.maximum_physical_attempts_per_logical_call
                        ),
                        "fixed_seeker_generation_contract": bound_payload,
                        "fixed_seeker_generation_contract_sha256": bound_sha256,
                        "fixed_seeker_cost_planning_sha256": (
                            cost_planning_sha256
                        ),
                        "logical_call_key": sha256_text(
                            canonical_json(call_identity)
                        ),
                    }
                )
    logical_keys = [str(row["logical_call_key"]) for row in rows]
    if len(logical_keys) != len(set(logical_keys)):
        raise RuntimeError("fixed-seeker V2.2 call plan contains duplicate keys")
    call_plan_sha256 = sha256_text(canonical_json(rows))
    maximum_input_tokens = [
        int(row["maximum_input_tokens"]) for row in rows
    ]
    maximum_total_input_tokens = sum(maximum_input_tokens)
    maximum_total_output_tokens = len(rows) * contract.max_output_tokens
    maximum_estimated_cost_usd = sum(
        float(row["maximum_cost_usd"]) for row in rows
    )
    budget_limits = {
        "max_api_calls": int(max_api_calls),
        "max_estimated_usd": float(max_estimated_usd),
        "max_input_tokens_per_call": int(max_input_tokens_per_call),
    }
    budget_checks = {
        "api_calls": len(rows) <= int(max_api_calls),
        "estimated_cost_usd": maximum_estimated_cost_usd
        <= float(max_estimated_usd),
        "max_input_tokens_per_call": max(maximum_input_tokens, default=0)
        <= int(max_input_tokens_per_call),
    }
    budget_gate = {
        "status": "PASS" if all(budget_checks.values()) else "FAIL",
        "checks": budget_checks,
        "limits": budget_limits,
    }
    estimate_payload = {
        "protocol": FIXED_SEEKER_V22_DRY_RUN_PROTOCOL,
        "stage": FIXED_SEEKER_V22_STAGE,
        "evoemo_sha256": evoemo_sha256,
        "scaffold_sha256": scaffold_sha256,
        "simulator_id": str(simulator_id),
        "max_turns": int(max_turns),
        "seeds": normalized_seeds,
        "max_scenarios": max_scenarios,
        "scenario_count": len(scenarios),
        "expected_tracks": len(scenarios) * len(normalized_seeds),
        "maximum_physical_api_attempts": len(rows),
        "maximum_input_tokens_per_call": max(
            maximum_input_tokens, default=0
        ),
        "maximum_total_input_tokens": maximum_total_input_tokens,
        "maximum_output_tokens_per_call": contract.max_output_tokens,
        "maximum_total_output_tokens": maximum_total_output_tokens,
        "maximum_estimated_usd": maximum_estimated_cost_usd,
        "maximum_physical_attempts_per_logical_call": (
            contract.maximum_physical_attempts_per_logical_call
        ),
        "call_plan_sha256": call_plan_sha256,
        "fixed_seeker_generation_contract": bound_payload,
        "fixed_seeker_generation_contract_sha256": bound_sha256,
        "fixed_seeker_cost_planning": normalized_cost_planning,
        "fixed_seeker_cost_planning_sha256": cost_planning_sha256,
        "budget_limits": budget_limits,
        "budget_gate": budget_gate,
        "cost_scope": "conservative sequential input/output/USD upper bound",
    }
    estimate = {
        **estimate_payload,
        "dry_run_acceptance_sha256": sha256_text(
            canonical_json(estimate_payload)
        ),
    }
    return estimate, rows


def persist_fixed_seeker_tracks_v22_dry_run(
    out_dir: str | Path,
    estimate: Mapping[str, Any],
    call_plan: Sequence[Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> str:
    """Persist or validate the exact no-API plan without creating a client."""

    out_dir = Path(out_dir)
    estimate_path = out_dir / "cost_estimate.json"
    plan_path = out_dir / "call_plan.jsonl"
    ledger_path = out_dir / "physical_attempt_ledger.jsonl"
    if overwrite and ledger_path.is_file() and ledger_path.stat().st_size > 0:
        raise RuntimeError(
            "fixed-seeker V2.2 refuses to overwrite a dry run after any physical "
            "attempt was reserved"
        )
    normalized_estimate = dict(estimate)
    normalized_plan = [dict(row) for row in call_plan]
    if estimate_path.is_file() or plan_path.is_file():
        if not estimate_path.is_file() or not plan_path.is_file():
            raise RuntimeError("fixed-seeker dry-run bundle is partial")
        existing_estimate = read_json(estimate_path)
        existing_plan = list(iter_jsonl(plan_path))
        if (
            existing_estimate == normalized_estimate
            and existing_plan == normalized_plan
        ):
            return "VALIDATED_EXISTING"
        if not overwrite:
            raise RuntimeError(
                "fixed-seeker dry-run differs from the saved plan; use a new "
                "output directory or explicit --overwrite before any attempt"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(estimate_path, normalized_estimate)
    write_jsonl(plan_path, normalized_plan)
    return "WRITTEN"


def require_fixed_seeker_tracks_v22_dry_run(
    out_dir: str | Path,
    estimate: Mapping[str, Any],
    call_plan: Sequence[Mapping[str, Any]],
    *,
    accepted_dry_run_sha256: str,
) -> None:
    """Fail closed unless the caller accepts this exact saved dry-run hash."""

    out_dir = Path(out_dir)
    estimate_path = out_dir / "cost_estimate.json"
    plan_path = out_dir / "call_plan.jsonl"
    if not estimate_path.is_file() or not plan_path.is_file():
        raise RuntimeError(
            "fixed-seeker API mode requires a completed matching --dry-run"
        )
    saved_estimate = read_json(estimate_path)
    saved_plan = list(iter_jsonl(plan_path))
    if saved_estimate != dict(estimate) or saved_plan != [
        dict(row) for row in call_plan
    ]:
        raise RuntimeError(
            "saved fixed-seeker dry run is stale; run --dry-run again"
        )
    expected_hash = str(estimate.get("dry_run_acceptance_sha256") or "")
    if not expected_hash or accepted_dry_run_sha256 != expected_hash:
        raise RuntimeError(
            "--accepted-dry-run-sha256 must exactly equal the current fixed-seeker "
            f"dry-run hash: {expected_hash}"
        )
    if (estimate.get("budget_gate") or {}).get("status") != "PASS":
        raise RuntimeError("fixed-seeker accepted dry-run budget gate did not PASS")


def _fixed_seeker_reported_usage(
    usage: Mapping[str, Any] | None,
    *,
    plan_row: Mapping[str, Any],
) -> tuple[dict[str, int] | None, str | None]:
    """Apply the accepted per-call input/output token upper bounds."""

    try:
        normalized = require_reported_usage(
            usage, stage="PM-v2.2 fixed-seeker generation"
        )
    except RuntimeError as exc:
        return None, str(exc)
    if normalized["prompt_tokens"] > int(plan_row["maximum_input_tokens"]):
        return normalized, (
            "fixed-seeker reported prompt_tokens exceed the accepted conservative "
            f"bound: reported={normalized['prompt_tokens']}, "
            f"bound={plan_row['maximum_input_tokens']}"
        )
    if normalized["completion_tokens"] > int(
        plan_row["maximum_output_tokens"]
    ):
        return normalized, (
            "fixed-seeker reported completion_tokens exceed the accepted output "
            f"bound: reported={normalized['completion_tokens']}, "
            f"bound={plan_row['maximum_output_tokens']}"
        )
    return normalized, None


def _fixed_seeker_v22_messages(
    system_prompt: str, conversation: Sequence[Mapping[str, str]]
) -> list[dict[str, str]]:
    if not conversation or conversation[-1].get("role") != "supporter":
        raise ValueError(
            "fixed-seeker generation requires a transcript ending in a "
            "supporter turn"
        )
    return [
        {"role": "system", "content": system_prompt},
        *[
            {
                "role": (
                    "assistant" if row["role"] == "seeker" else "user"
                ),
                "content": str(row["content"]),
            }
            for row in conversation
        ],
    ]


def build_fixed_seeker_tracks_v22(
    evoemo_path: str | Path,
    out_dir: str | Path,
    *,
    seeker_endpoint: Endpoint,
    contract: FixedSeekerGenerationContract,
    cost_planning: Mapping[str, Any],
    simulator_id: str,
    accepted_dry_run_sha256: str,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
    max_turns: int = 10,
    seeds: Sequence[int] = (101,),
    max_scenarios: int | None = None,
    overwrite: bool = False,
    study_freeze_sha256: str | None = None,
) -> dict[str, Any]:
    """Generate PM-v2.2 fixed tracks under an accepted, fail-closed plan.

    This is deliberately separate from :func:`build_fixed_seeker_tracks`, which
    remains the legacy 60-output-token path for historical reproduction only.
    """

    estimate, call_plan = plan_fixed_seeker_tracks_v22(
        evoemo_path,
        seeker_endpoint=seeker_endpoint,
        contract=contract,
        cost_planning=cost_planning,
        simulator_id=simulator_id,
        max_api_calls=max_api_calls,
        max_estimated_usd=max_estimated_usd,
        max_input_tokens_per_call=max_input_tokens_per_call,
        max_turns=max_turns,
        seeds=seeds,
        max_scenarios=max_scenarios,
    )
    out_dir = Path(out_dir)
    require_fixed_seeker_tracks_v22_dry_run(
        out_dir,
        estimate,
        call_plan,
        accepted_dry_run_sha256=accepted_dry_run_sha256,
    )

    tracks_path = out_dir / "fixed_seeker_tracks.jsonl"
    raw_path = out_dir / "raw_seeker_calls.jsonl"
    ledger_path = out_dir / "physical_attempt_ledger.jsonl"
    summary_path = out_dir / "summary.json"
    manifest_path = out_dir / "run_manifest.json"
    attestation_path = out_dir / "artifact_attestation.json"
    if overwrite:
        if ledger_path.is_file() and ledger_path.stat().st_size > 0:
            raise RuntimeError(
                "fixed-seeker V2.2 refuses --overwrite after a physical attempt "
                "was reserved"
            )
        for path in (
            tracks_path,
            raw_path,
            ledger_path,
            summary_path,
            manifest_path,
            attestation_path,
        ):
            if path.exists():
                path.unlink()

    bound = contract.bind_endpoint(contract.seeker_endpoint, seeker_endpoint)
    bound_payload = bound.payload()
    bound_sha256 = bound.digest()
    normalized_cost_planning = fixed_seeker_cost_planning_contract(
        cost_planning
    )
    cost_planning_sha256 = sha256_text(
        canonical_json(normalized_cost_planning)
    )
    scaffold_sha256 = str(estimate["scaffold_sha256"])
    ensure_run_manifest(
        manifest_path,
        {
            "stage": FIXED_SEEKER_V22_STAGE,
            "evoemo_sha256": str(estimate["evoemo_sha256"]),
            "simulator_id": simulator_id,
            "max_turns": int(max_turns),
            "seeds": [int(value) for value in seeds],
            "max_scenarios": max_scenarios,
            "scaffold_sha256": scaffold_sha256,
            "dry_run_acceptance_sha256": str(
                estimate["dry_run_acceptance_sha256"]
            ),
            "call_plan_sha256": str(estimate["call_plan_sha256"]),
            "fixed_seeker_generation_contract": bound_payload,
            "fixed_seeker_generation_contract_sha256": bound_sha256,
            "fixed_seeker_cost_planning": normalized_cost_planning,
            "fixed_seeker_cost_planning_sha256": cost_planning_sha256,
            "planned_budget_gate": estimate["budget_gate"],
            "study_freeze_sha256": study_freeze_sha256,
        },
    )
    expected_calls = {
        str(row["logical_call_key"]): 1 for row in call_plan
    }
    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=FIXED_SEEKER_V22_STAGE,
        expected_calls=expected_calls,
        maximum_total_attempts=len(expected_calls),
    )
    plan_index = {
        (
            str(row["user_id"]),
            int(row["topic_index"]),
            int(row["seed"]),
            str(row["simulator_id"]),
            int(row["turn_index"]),
        ): dict(row)
        for row in call_plan
    }
    existing_tracks = (
        index_jsonl_unique(
            tracks_path, ("user_id", "topic_index", "seed", "simulator_id")
        )
        if tracks_path.is_file()
        else {}
    )
    expected_track_keys = {
        (key[0], key[1], key[2], key[3]) for key in plan_index
    }
    unexpected_tracks = sorted(set(existing_tracks) - expected_track_keys)
    if unexpected_tracks:
        raise RuntimeError(
            f"fixed-seeker output contains unplanned tracks: {unexpected_tracks[:3]}"
        )

    client = None
    failures: list[dict[str, Any]] = []
    scenarios = _fixed_seeker_v22_scenarios(
        evoemo_path, max_scenarios=max_scenarios
    )
    work_units = [
        (user, topic, int(seed))
        for user, topic in scenarios
        for seed in seeds
    ]
    try:
        for user, topic, seed in work_units:
            track_key = _track_key(
                str(user["id"]), int(topic["idx"]), seed, simulator_id
            )
            if track_key in existing_tracks:
                existing = existing_tracks[track_key]
                if (
                    existing.get("fixed_seeker_generation_contract")
                    != bound_payload
                    or existing.get(
                        "fixed_seeker_generation_contract_sha256"
                    )
                    != bound_sha256
                    or existing.get("fixed_seeker_cost_planning")
                    != normalized_cost_planning
                    or existing.get("fixed_seeker_cost_planning_sha256")
                    != cost_planning_sha256
                    or len(existing.get("seeker_turns") or []) != max_turns
                ):
                    raise RuntimeError(
                        f"existing fixed-seeker V2.2 track is stale: {track_key}"
                    )
                for turn_index in range(1, max_turns + 1):
                    plan_row = plan_index[(*track_key, turn_index)]
                    logical_key = str(plan_row["logical_call_key"])
                    terminal = ledger.terminal_row(logical_key)
                    _, usage_error = _fixed_seeker_reported_usage(
                        (terminal or {}).get("usage"), plan_row=plan_row
                    )
                    if (
                        not ledger.succeeded(logical_key)
                        or terminal is None
                        or (terminal.get("result") or {}).get(
                            "normalized_finish_reason"
                        )
                        != "complete"
                        or usage_error is not None
                    ):
                        raise RuntimeError(
                            "persisted fixed track lacks a matching successful "
                            f"attempt: {track_key}, turn={turn_index}"
                        )
                continue

            conversation: list[dict[str, str]] = [
                {"role": "supporter", "content": NEUTRAL_INITIAL_GREETING}
            ]
            seeker_turns: list[str] = []
            turn_provenance: list[dict[str, Any]] = []
            try:
                system_prompt = contract.render_system_prompt(
                    evaluator_context(user, topic)
                )
                for turn_index in range(1, max_turns + 1):
                    plan_row = plan_index[(*track_key, turn_index)]
                    logical_key = str(plan_row["logical_call_key"])
                    terminal = ledger.terminal_row(logical_key)
                    if ledger.succeeded(logical_key):
                        result_payload = dict((terminal or {}).get("result") or {})
                        reported_usage, usage_error = _fixed_seeker_reported_usage(
                            (terminal or {}).get("usage"), plan_row=plan_row
                        )
                        message = contract.normalize_output(
                            str(result_payload.get("seeker_message") or "")
                        )
                        if (
                            not message
                            or result_payload.get("normalized_finish_reason")
                            != "complete"
                            or usage_error is not None
                        ):
                            raise RuntimeError(
                                "successful fixed-seeker ledger result is invalid: "
                                f"{logical_key}"
                            )
                        provider_finish_reason = result_payload.get(
                            "provider_finish_reason"
                        )
                        normalized_finish_reason = result_payload.get(
                            "normalized_finish_reason"
                        )
                        request_hash = terminal.get("request_hash")
                    elif ledger.attempts_for(logical_key):
                        raise RuntimeError(
                            "fixed-seeker logical call already spent its one "
                            f"physical attempt without success: {logical_key}"
                        )
                    else:
                        messages = _fixed_seeker_v22_messages(
                            system_prompt, conversation
                        )
                        prompt_hash = sha256_text(canonical_json(messages))
                        record_ids = {
                            "user_id": str(user["id"]),
                            "topic_index": int(topic["idx"]),
                            "seed": seed,
                            "simulator_id": simulator_id,
                            "turn_index": turn_index,
                            "track_generation": True,
                            "logical_call_key": logical_key,
                            "fixed_seeker_generation_contract_sha256": (
                                bound_sha256
                            ),
                            "fixed_seeker_cost_planning_sha256": (
                                cost_planning_sha256
                            ),
                            "maximum_input_tokens": int(
                                plan_row["maximum_input_tokens"]
                            ),
                            "maximum_output_tokens": int(
                                plan_row["maximum_output_tokens"]
                            ),
                            "dry_run_acceptance_sha256": accepted_dry_run_sha256,
                        }
                        if client is None:
                            client = make_client(seeker_endpoint)
                        reservation = ledger.reserve(
                            logical_key,
                            record_ids=record_ids,
                            prompt_sha256=prompt_hash,
                        )
                        try:
                            result, _ = client.chat(
                                messages,
                                temperature=contract.temperature,
                                max_tokens=contract.max_output_tokens,
                                seed=int(plan_row["turn_seed"]),
                                response_schema=None,
                                retries=1,
                            )
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                            append_jsonl(
                                raw_path,
                                request_log(
                                    stage=FIXED_SEEKER_V22_STAGE,
                                    endpoint=seeker_endpoint,
                                    messages=messages,
                                    result=None,
                                    parsed=None,
                                    error=error,
                                    prompt_hash=prompt_hash,
                                    record_ids=record_ids,
                                ),
                            )
                            ledger.finish(
                                reservation,
                                succeeded=False,
                                request_hash=None,
                                usage=None,
                                error=error,
                                result={
                                    "fixed_seeker_generation_contract_sha256": (
                                        bound_sha256
                                    )
                                },
                            )
                            raise

                        completion_error = contract.completion_gate_error(
                            normalized_finish_reason=(
                                result.normalized_finish_reason
                            ),
                            provider_finish_reason=result.provider_finish_reason,
                        )
                        reported_usage, accounting_error = (
                            _fixed_seeker_reported_usage(
                                result.usage, plan_row=plan_row
                            )
                        )
                        message = contract.normalize_output(result.text)
                        empty_error = (
                            None
                            if message
                            else (
                                "fixed-seeker completion is empty after frozen "
                                "normalization"
                            )
                        )
                        gate_error = (
                            completion_error or accounting_error or empty_error
                        )
                        result_payload = {
                            "seeker_message": message,
                            "provider_finish_reason": (
                                result.provider_finish_reason
                            ),
                            "normalized_finish_reason": (
                                result.normalized_finish_reason
                            ),
                            "fixed_seeker_generation_contract": bound_payload,
                            "fixed_seeker_generation_contract_sha256": (
                                bound_sha256
                            ),
                            "fixed_seeker_cost_planning_sha256": (
                                cost_planning_sha256
                            ),
                            "reported_usage": reported_usage,
                            "observed_cost_usd": (
                                (
                                    reported_usage["prompt_tokens"]
                                    * normalized_cost_planning[
                                        "pricing_usd_per_mtok"
                                    ]["input"]
                                    + reported_usage["completion_tokens"]
                                    * normalized_cost_planning[
                                        "pricing_usd_per_mtok"
                                    ]["output"]
                                )
                                / 1_000_000
                                if reported_usage is not None
                                else None
                            ),
                        }
                        append_jsonl(
                            raw_path,
                            request_log(
                                stage=FIXED_SEEKER_V22_STAGE,
                                endpoint=seeker_endpoint,
                                messages=messages,
                                result=result,
                                parsed=None,
                                error=gate_error,
                                prompt_hash=prompt_hash,
                                record_ids=record_ids,
                            ),
                        )
                        if gate_error:
                            ledger.finish(
                                reservation,
                                succeeded=False,
                                request_hash=result.request_hash,
                                usage=result.usage,
                                error=gate_error,
                                result=result_payload,
                            )
                            raise RuntimeError(gate_error)
                        ledger.finish(
                            reservation,
                            succeeded=True,
                            request_hash=result.request_hash,
                            usage=result.usage,
                            error=None,
                            result=result_payload,
                        )
                        provider_finish_reason = result.provider_finish_reason
                        normalized_finish_reason = (
                            result.normalized_finish_reason
                        )
                        request_hash = result.request_hash

                    seeker_turns.append(message)
                    turn_provenance.append(
                        {
                            "turn_index": turn_index,
                            "logical_call_key": logical_key,
                            "request_hash": request_hash,
                            "provider_finish_reason": provider_finish_reason,
                            "normalized_finish_reason": (
                                normalized_finish_reason
                            ),
                            "reported_usage": reported_usage,
                            "observed_cost_usd": (
                                (
                                    reported_usage["prompt_tokens"]
                                    * normalized_cost_planning[
                                        "pricing_usd_per_mtok"
                                    ]["input"]
                                    + reported_usage["completion_tokens"]
                                    * normalized_cost_planning[
                                        "pricing_usd_per_mtok"
                                    ]["output"]
                                )
                                / 1_000_000
                                if reported_usage is not None
                                else None
                            ),
                            "fixed_seeker_generation_contract": bound_payload,
                            "fixed_seeker_generation_contract_sha256": (
                                bound_sha256
                            ),
                            "fixed_seeker_cost_planning_sha256": (
                                cost_planning_sha256
                            ),
                        }
                    )
                    conversation.append({"role": "seeker", "content": message})
                    if turn_index < max_turns:
                        probe = _NEUTRAL_TRACK_PROBES[
                            (turn_index - 1) % len(_NEUTRAL_TRACK_PROBES)
                        ]
                        conversation.append(
                            {"role": "supporter", "content": probe}
                        )

                track_id = (
                    "track_"
                    + stable_hex(
                        user["id"],
                        topic["idx"],
                        seed,
                        simulator_id,
                        bound_sha256,
                        canonical_json(seeker_turns),
                        n=20,
                    )
                )
                append_jsonl(
                    tracks_path,
                    {
                        "track_id": track_id,
                        "user_id": str(user["id"]),
                        "topic_index": int(topic["idx"]),
                        "seed": seed,
                        "simulator_id": simulator_id,
                        "seeker_model": seeker_endpoint.model,
                        "seeker_family": seeker_endpoint.family,
                        "initial_greeting": NEUTRAL_INITIAL_GREETING,
                        "seeker_turns": seeker_turns,
                        "turn_provenance": turn_provenance,
                        "elicitation_scaffold": (
                            contract.elicitation_scaffold_protocol
                        ),
                        "causal_use": "replayed unchanged to all policies",
                        "fixed_seeker_generation_contract": bound_payload,
                        "fixed_seeker_generation_contract_sha256": (
                            bound_sha256
                        ),
                        "fixed_seeker_cost_planning": (
                            normalized_cost_planning
                        ),
                        "fixed_seeker_cost_planning_sha256": (
                            cost_planning_sha256
                        ),
                        "dry_run_acceptance_sha256": (
                            accepted_dry_run_sha256
                        ),
                    },
                )
            except Exception as exc:
                failures.append(
                    {
                        "user_id": str(user["id"]),
                        "topic_index": int(topic["idx"]),
                        "seed": seed,
                        "simulator_id": simulator_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                break
    finally:
        if client is not None:
            client.close()

    completed = (
        len(
            index_jsonl_unique(
                tracks_path,
                ("user_id", "topic_index", "seed", "simulator_id"),
            )
        )
        if tracks_path.is_file()
        else 0
    )
    expected_tracks = int(estimate["expected_tracks"])
    normalized_finish_reason_counts = {
        reason: 0
        for reason in ("complete", "length", "tool_call", "content_filter", "unknown")
    }
    for call_key in expected_calls:
        terminal = ledger.terminal_row(call_key)
        if terminal is None:
            continue
        reason = str(
            (terminal.get("result") or {}).get(
                "normalized_finish_reason", "unknown"
            )
        )
        normalized_finish_reason_counts[reason] = (
            normalized_finish_reason_counts.get(reason, 0) + 1
        )
    observed_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    observed_prompt_tokens_per_call: list[int] = []
    observed_completion_tokens_per_call: list[int] = []
    usage_accounting_complete = True
    for plan_row in call_plan:
        terminal = ledger.terminal_row(str(plan_row["logical_call_key"]))
        if terminal is None:
            continue
        normalized_usage, usage_error = _fixed_seeker_reported_usage(
            terminal.get("usage"), plan_row=plan_row
        )
        if usage_error is not None or normalized_usage is None:
            usage_accounting_complete = False
            continue
        for key in observed_usage:
            observed_usage[key] += int(normalized_usage[key])
        observed_prompt_tokens_per_call.append(
            int(normalized_usage["prompt_tokens"])
        )
        observed_completion_tokens_per_call.append(
            int(normalized_usage["completion_tokens"])
        )
    observed_cost_usd = (
        observed_usage["prompt_tokens"]
        * normalized_cost_planning["pricing_usd_per_mtok"]["input"]
        + observed_usage["completion_tokens"]
        * normalized_cost_planning["pricing_usd_per_mtok"]["output"]
    ) / 1_000_000
    observed_budget_checks = {
        "planned_budget_gate_passed": estimate["budget_gate"]["status"]
        == "PASS",
        "physical_attempts": ledger.started_attempts <= int(max_api_calls),
        "reported_usage_complete": usage_accounting_complete
        and len(observed_prompt_tokens_per_call) == ledger.started_attempts,
        "observed_cost_usd": observed_cost_usd
        <= float(max_estimated_usd),
        "observed_cost_within_planned_upper_bound": observed_cost_usd
        <= float(estimate["maximum_estimated_usd"]),
        "max_reported_input_tokens_per_call": max(
            observed_prompt_tokens_per_call, default=0
        )
        <= int(max_input_tokens_per_call),
        "max_reported_completion_tokens_per_call": max(
            observed_completion_tokens_per_call, default=0
        )
        <= int(contract.max_output_tokens),
    }
    observed_budget_gate = {
        "status": (
            "PASS" if all(observed_budget_checks.values()) else "FAIL"
        ),
        "checks": observed_budget_checks,
        "limits": dict(estimate["budget_limits"]),
        "reported_usage": observed_usage,
        "observed_cost_usd": observed_cost_usd,
        "max_reported_input_tokens_per_call": max(
            observed_prompt_tokens_per_call, default=0
        ),
        "max_reported_completion_tokens_per_call": max(
            observed_completion_tokens_per_call, default=0
        ),
    }
    status = (
        "COMPLETE"
        if completed == expected_tracks
        and not failures
        and ledger.started_attempts == len(expected_calls)
        and all(ledger.succeeded(key) for key in expected_calls)
        and observed_budget_gate["status"] == "PASS"
        else "INCOMPLETE"
    )
    summary = {
        "status": status,
        "stage": FIXED_SEEKER_V22_STAGE,
        "simulator_id": simulator_id,
        "seeker_model": seeker_endpoint.model,
        "seeker_family": seeker_endpoint.family,
        "expected_tracks": expected_tracks,
        "completed_tracks": completed,
        "max_turns": max_turns,
        "expected_logical_calls": len(expected_calls),
        "started_physical_attempts": ledger.started_attempts,
        "successful_logical_calls": sum(
            int(ledger.succeeded(key)) for key in expected_calls
        ),
        "normalized_finish_reason_counts": normalized_finish_reason_counts,
        "completion_truncated_count": normalized_finish_reason_counts.get(
            "length", 0
        ),
        "failures": failures,
        "dry_run_acceptance_sha256": accepted_dry_run_sha256,
        "call_plan_sha256": str(estimate["call_plan_sha256"]),
        "fixed_seeker_generation_contract": bound_payload,
        "fixed_seeker_generation_contract_sha256": bound_sha256,
        "fixed_seeker_cost_planning": normalized_cost_planning,
        "fixed_seeker_cost_planning_sha256": cost_planning_sha256,
        "planned_budget_gate": estimate["budget_gate"],
        "observed_budget_gate": observed_budget_gate,
        "interpretation": (
            "fixed-input causal track; later seeker turns do not react to "
            "evaluated policy replies"
        ),
    }
    write_json(summary_path, summary)
    if status != "COMPLETE":
        raise RuntimeError(
            "fixed seeker V2.2 generation incomplete; no failed logical call "
            "may be retried under this accepted plan"
        )
    create_artifact_attestation(
        attestation_path,
        stage=FIXED_SEEKER_V22_STAGE,
        inputs={
            "evoemo": evoemo_path,
            "run_manifest": manifest_path,
            "cost_estimate": out_dir / "cost_estimate.json",
            "call_plan": out_dir / "call_plan.jsonl",
        },
        outputs={
            "tracks": (tracks_path, True),
            "raw_calls": (raw_path, True),
            "physical_attempt_ledger": (ledger_path, True),
            "summary": (summary_path, False),
        },
        parameters={
            "simulator_id": simulator_id,
            "max_turns": max_turns,
            "seeds": [int(value) for value in seeds],
            "dry_run_acceptance_sha256": accepted_dry_run_sha256,
            "call_plan_sha256": str(estimate["call_plan_sha256"]),
            "fixed_seeker_generation_contract": bound_payload,
            "fixed_seeker_generation_contract_sha256": bound_sha256,
            "fixed_seeker_cost_planning": normalized_cost_planning,
            "fixed_seeker_cost_planning_sha256": cost_planning_sha256,
            "planned_budget_gate": estimate["budget_gate"],
            "observed_budget_gate": observed_budget_gate,
        },
        expected={
            "tracks": expected_tracks,
            "turns_per_track": max_turns,
            "logical_calls": len(expected_calls),
            "accepted_normalized_finish_reasons": ["complete"],
            "completion_truncated_count": 0,
            "planned_budget_gate_status": "PASS",
            "observed_budget_gate_status": "PASS",
            "planned_budget_gate": estimate["budget_gate"],
            "observed_budget_gate": observed_budget_gate,
        },
        study_freeze_sha256=study_freeze_sha256,
    )
    return summary


def _fixed_context_before_turn(
    track: dict[str, Any], turn_index: int
) -> list[dict[str, str]]:
    """Return the policy-independent context used for a one-step comparison.

    Evaluated supporter replies are deliberately absent.  Each turn is a
    separate fixed-context decision point elicited by the neutral scaffold.
    This is a causal one-step stress test, not a simulated closed-loop dialogue.
    """
    turns = [normalize_space(x) for x in (track.get("seeker_turns") or [])]
    if turn_index < 1 or turn_index > len(turns):
        raise ValueError(f"turn_index {turn_index} outside fixed track")
    greeting = normalize_space(
        track.get("initial_greeting") or NEUTRAL_INITIAL_GREETING
    )
    if greeting != NEUTRAL_INITIAL_GREETING:
        raise ValueError("fixed track initial greeting is not the frozen neutral greeting")
    context: list[dict[str, str]] = [
        {"role": "supporter", "content": NEUTRAL_INITIAL_GREETING}
    ]
    for prior_index in range(turn_index - 1):
        context.append({"role": "seeker", "content": turns[prior_index]})
        context.append({
            "role": "supporter",
            "content": _NEUTRAL_TRACK_PROBES[
                prior_index % len(_NEUTRAL_TRACK_PROBES)
            ],
        })
    return context


def _load_fixed_tracks(path: str | Path) -> dict[tuple[str, int, int, str], dict[str, Any]]:
    tracks: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        key = _track_key(row["user_id"], row["topic_index"], row["seed"], row["simulator_id"])
        if key in tracks:
            raise ValueError(f"duplicate fixed track key: {key}")
        tracks[key] = row
    return tracks


def _policy_from_selection(model: PMModel, selection: dict[str, Any]) -> LearnedPMPolicy:
    pm = selection["pm"]
    return LearnedPMPolicy(
        model,
        epsilon=float(pm["epsilon"]),
        tau_misuse=float(pm["tau_misuse"]),
        tau_omission=float(pm["tau_omission"]),
        tau_strategy=float(pm["tau_strategy"]),
        allow_constraint_fallback=bool(pm.get("confirmatory_allow_constraint_fallback", False)),
    )


def _external_ood_preflight(
    model: PMModel,
    users: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for user in users:
        items, _ = build_evo_memory(user)
        for topic in user.get("subsequent_topics") or []:
            state = make_evo_runtime_state(
                user,
                topic,
                [{"role": "supporter", "content": NEUTRAL_INITIAL_GREETING}],
                "I want to talk about what has been happening.",
                items,
                1,
                "ood_preflight",
                track_id="ood_preflight",
            )
            rows = [(state, action) for action in state.allowed_actions]
            report = model.feature_builder.ood_report(rows)
            reports.append({
                "user_id": str(user["id"]),
                "topic_index": int(topic["idx"]),
                **report,
            })
    severe = [
        row for row in reports
        if row.get("severe_scalar_ood") or row.get("severe_catalog_ood")
    ]
    return {
        "ok": not severe,
        "n_scenarios": len(reports),
        "n_severe": len(severe),
        "severe_examples": severe[:5],
        "required_action": (
            "retrain_on_matched_longitudinal_development_data_or_use_preregistered_ood_baseline"
            if severe else "none"
        ),
    }


def run_evoemo_dialogues(
    evoemo_path: str | Path,
    strategy_bank_path: str | Path,
    checkpoint_path: str | Path,
    selection_path: str | Path,
    out_dir: str | Path,
    *,
    generator_endpoint: Endpoint,
    seeker_endpoint: Endpoint | None,
    simulator_id: str,
    protocol: str = "selective",
    interaction_mode: str = "fixed",
    fixed_tracks_path: str | Path | None = None,
    fixed_tracks_attestation_path: str | Path | None = None,
    conditions: Sequence[str] = (
        "no_memory_r0",
        "no_memory_rs",
        "session_rag_rs",
        "full_history_rs",
        "all_structured_rs",
        "best_fixed",
        "strong_rule",
        "pm",
    ),
    max_turns: int = 10,
    seeds: Sequence[int] = (101,),
    max_scenarios: int | None = None,
    overwrite: bool = False,
    study_freeze_sha256: str | None = None,
) -> dict[str, Any]:
    if protocol not in {"official", "selective"}:
        raise ValueError("protocol must be official or selective")
    if interaction_mode not in {"fixed", "interactive"}:
        raise ValueError("interaction_mode must be fixed or interactive")
    if interaction_mode == "fixed" and fixed_tracks_path is None:
        raise ValueError("fixed_tracks_path is required for fixed-input evaluation")
    fixed_tracks_verification = None
    if interaction_mode == "fixed":
        fixed_tracks_attestation_path = Path(
            fixed_tracks_attestation_path
            or Path(fixed_tracks_path).parent / "artifact_attestation.json"
        )
        fixed_tracks_verification = require_artifact_attestation(
            fixed_tracks_attestation_path,
            required_stage="evoemo_fixed_seeker_tracks",
            required_output_paths={"tracks": fixed_tracks_path},
        )
    if interaction_mode == "interactive" and seeker_endpoint is None:
        raise ValueError("seeker_endpoint is required for interactive evaluation")

    users = load_evoemo(evoemo_path)
    strategy_cards = [StrategyCard.model_validate(row) for row in iter_jsonl(strategy_bank_path)]
    if not strategy_cards:
        raise ValueError("strategy bank is empty")
    strategy_retriever = StrategyRetriever(strategy_cards, top_k=3)
    memory_retriever = MemoryRetriever()
    model = PMModel.load(checkpoint_path)
    selection = read_json(selection_path)
    pm_policy = _policy_from_selection(model, selection)
    best_fixed = FixedPolicy(selection["best_fixed_action"])
    strong_rule = StrongRulePolicy(RuleConfig(**selection["strong_rule"]["config"]), strategy_retriever)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dialogue_path = out_dir / "dialogues.jsonl"
    turn_path = out_dir / "turns.jsonl"
    raw_path = out_dir / "raw_api_calls.jsonl"
    summary_path = out_dir / "generation_summary.json"
    ood_path = out_dir / "external_ood_preflight.json"
    manifest_path = out_dir / "run_manifest.json"
    attestation_path = out_dir / "artifact_attestation.json"
    if overwrite:
        for path in (dialogue_path, turn_path, raw_path, summary_path, ood_path, manifest_path, attestation_path):
            if path.exists():
                path.unlink()

    tracks = _load_fixed_tracks(fixed_tracks_path) if fixed_tracks_path else {}
    ensure_run_manifest(manifest_path, {
        "stage": "evoemo_generation",
        "evoemo_sha256": sha256_file(evoemo_path),
        "strategy_bank_sha256": sha256_file(strategy_bank_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "selection_sha256": sha256_file(selection_path),
        "fixed_tracks_sha256": sha256_file(fixed_tracks_path) if fixed_tracks_path else None,
        "fixed_tracks_attestation_sha256": (
            fixed_tracks_verification["attestation_sha256"]
            if fixed_tracks_verification else None
        ),
        "generator_model": generator_endpoint.model,
        "generator_family": generator_endpoint.family,
        "generator_base_url": generator_endpoint.base_url,
        "seeker_model": seeker_endpoint.model if seeker_endpoint else None,
        "seeker_family": seeker_endpoint.family if seeker_endpoint else None,
        "seeker_base_url": seeker_endpoint.base_url if seeker_endpoint else None,
        "simulator_id": simulator_id,
        "protocol": protocol,
        "interaction_mode": interaction_mode,
        "fixed_track_design": (
            "policy_independent_open_loop_one_step_v2"
            if interaction_mode == "fixed" else None
        ),
        "conditions": list(conditions),
        "max_turns": int(max_turns),
        "seeds": [int(x) for x in seeds],
        "max_scenarios": max_scenarios,
        "study_freeze_sha256": study_freeze_sha256,
    })

    if "pm" in conditions:
        ood = _external_ood_preflight(model, users)
        write_json(ood_path, ood)
        if not ood["ok"]:
            raise RuntimeError(
                "EvoEmo PM evaluation blocked before API calls by severe feature-domain shift. "
                "Create a matched longitudinal development split and retrain, or run only a "
                "preregistered non-PM OOD baseline. Details: " + str(ood)
            )
    else:
        write_json(ood_path, {"ok": True, "not_applicable": True, "reason": "pm condition not requested"})

    scenarios: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for user in users:
        for topic in user.get("subsequent_topics") or []:
            scenarios.append((user, topic))
    if max_scenarios is not None:
        scenarios = scenarios[:max_scenarios]

    done_fields = ("user_id", "topic_index", "condition", "seed", "simulator_id", "interaction_mode")
    done = load_done_keys(dialogue_path, done_fields)
    generator = OpenAICompatibleClient(generator_endpoint)
    seeker = OpenAICompatibleClient(seeker_endpoint) if seeker_endpoint is not None and interaction_mode == "interactive" else None
    failures: list[dict[str, Any]] = []
    try:
        for user, topic in scenarios:
            items, session_docs = build_evo_memory(user)
            private_prompt = seeker_system_prompt(user, topic)
            for seed in seeds:
                fixed_track = None
                if interaction_mode == "fixed":
                    track_key = _track_key(str(user["id"]), int(topic["idx"]), int(seed), simulator_id)
                    fixed_track = tracks.get(track_key)
                    if fixed_track is None:
                        raise RuntimeError(f"missing fixed seeker track: {track_key}")
                    if len(fixed_track.get("seeker_turns") or []) != max_turns:
                        raise RuntimeError(
                            f"fixed track turn count mismatch for {track_key}: "
                            f"expected {max_turns}, got {len(fixed_track.get('seeker_turns') or [])}"
                        )
                for condition in conditions:
                    key = (
                        str(user["id"]), int(topic["idx"]), condition, int(seed),
                        simulator_id, interaction_mode,
                    )
                    if key in done:
                        continue
                    conversation: list[dict[str, str]] = [
                        {"role": "supporter", "content": NEUTRAL_INITIAL_GREETING}
                    ]
                    turn_records: list[dict[str, Any]] = []
                    evaluation_cases: list[dict[str, Any]] = []
                    track_id = (
                        str(fixed_track["track_id"])
                        if fixed_track is not None
                        else f"interactive_{stable_hex(user['id'], topic['idx'], condition, seed, simulator_id, n=20)}"
                    )
                    try:
                        for turn_index in range(1, max_turns + 1):
                            if fixed_track is not None:
                                seeker_message = normalize_space(
                                    fixed_track["seeker_turns"][turn_index - 1]
                                )
                                state_conversation = _fixed_context_before_turn(
                                    fixed_track, turn_index
                                )
                            else:
                                assert seeker is not None and seeker_endpoint is not None
                                seeker_message = _seeker_next(
                                    seeker,
                                    seeker_endpoint,
                                    private_prompt,
                                    conversation,
                                    seed=int(seed) + turn_index,
                                    raw_log_path=raw_path,
                                    record_ids={
                                        "user_id": str(user["id"]),
                                        "topic_index": int(topic["idx"]),
                                        "condition": condition,
                                        "seed": int(seed),
                                        "simulator_id": simulator_id,
                                        "turn_index": turn_index,
                                        "interaction_mode": interaction_mode,
                                    },
                                )
                                state_conversation = list(conversation)

                            state_start = time.perf_counter()
                            state = make_evo_runtime_state(
                                user,
                                topic,
                                state_conversation,
                                seeker_message,
                                items,
                                turn_index,
                                condition,
                                track_id=track_id,
                                fixed_open_loop=fixed_track is not None,
                            )
                            pre_evidence_ms = (time.perf_counter() - state_start) * 1000.0
                            query = context_query(
                                state.current_user_text,
                                [x.model_dump(mode="json") for x in state.current_session_history],
                                state.current_session_summary,
                            )

                            pm_ood_report = None
                            pm_decision_report = None
                            pm_ms = 0.0
                            retrieval_start = time.perf_counter()
                            catalog_reads = 0
                            if condition == "no_memory_r0":
                                action_id = "M0+R0"
                                memory_view, strategy_view = [], []
                            elif condition == "no_memory_rs":
                                action_id = "M0+RS"
                                memory_view, strategy_view = [], strategy_retriever.retrieve(query)
                            elif condition == "session_rag_rs":
                                action_id = "SESSION_RAG+RS"
                                memory_view = _session_rag(query, session_docs, top_k=4)
                                strategy_view = strategy_retriever.retrieve(query)
                            elif condition == "full_history_rs":
                                action_id = "FULL_HISTORY+RS"
                                memory_view = _session_rag(query, session_docs, top_k=len(session_docs))
                                strategy_view = strategy_retriever.retrieve(query)
                            elif condition == "all_structured_rs":
                                action_id = "ALL_STRUCTURED+RS"
                                memory_view = list(items)
                                strategy_view = strategy_retriever.retrieve(query)
                            elif condition in {"best_fixed", "strong_rule", "pm"}:
                                if condition == "best_fixed":
                                    action_id = best_fixed.choose(state)
                                elif condition == "strong_rule":
                                    catalog_reads = len(state.inventory)
                                    action_id = strong_rule.choose(state)
                                else:
                                    catalog_reads = len(state.inventory)
                                    pm_start = time.perf_counter()
                                    action_id = pm_policy.choose(state)
                                    pm_ms = (time.perf_counter() - pm_start) * 1000.0
                                    pm_ood_report = model.last_ood_report
                                    pm_decision_report = pm_policy.last_decision_report
                                sources, strategy = parse_action_id(action_id)
                                memory_view = memory_retriever.retrieve(query, items, sources)
                                strategy_view = strategy_retriever.retrieve(query) if strategy is StrategyMode.RS else []
                            else:
                                raise ValueError(f"unknown condition: {condition}")
                            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

                            system = OFFICIAL_ESMEM_SYSTEM if protocol == "official" else SELECTIVE_ESMEM_SYSTEM
                            messages = generation_messages(state, memory_view, strategy_view, system_prompt=system)
                            result, _ = generator.chat(
                                messages,
                                temperature=0.0,
                                max_tokens=(60 if protocol == "official" else 100),
                                seed=int(seed) + turn_index,
                                response_schema=None,
                                retries=3,
                            )
                            append_jsonl(raw_path, request_log(
                                stage="evoemo_supporter",
                                endpoint=generator_endpoint,
                                messages=messages,
                                result=result,
                                parsed=None,
                                error=None,
                                prompt_hash=sha256_text(canonical_json(messages)),
                                record_ids={
                                    "user_id": str(user["id"]),
                                    "topic_index": int(topic["idx"]),
                                    "condition": condition,
                                    "seed": int(seed),
                                    "simulator_id": simulator_id,
                                    "turn_index": turn_index,
                                    "interaction_mode": interaction_mode,
                                    "track_id": track_id,
                                },
                            ))
                            supporter_message = normalize_space(result.text)
                            if fixed_track is not None:
                                # Display-only stitched sequence. It is never fed
                                # into a later PM/generator decision or scored as
                                # an interactive dialogue.
                                conversation.extend([
                                    {"role": "seeker", "content": seeker_message},
                                    {"role": "supporter", "content": supporter_message},
                                ])
                                evaluation_cases.append({
                                    "turn_index": turn_index,
                                    "context_before_turn": state_conversation,
                                    "current_seeker_message": seeker_message,
                                    "supporter_response": supporter_message,
                                    "context_sha256": sha256_text(
                                        canonical_json(state_conversation)
                                    ),
                                })
                            else:
                                conversation.extend([
                                    {"role": "seeker", "content": seeker_message},
                                    {"role": "supporter", "content": supporter_message},
                                ])
                            sources_count = len({x.source for x in memory_view})
                            retrieval_calls = sources_count + (1 if strategy_view else 0)
                            memory_tokens = sum(estimate_tokens(x.text) for x in memory_view)
                            strategy_tokens = sum(
                                estimate_tokens(x.guidance_text + x.example_response)
                                for x in strategy_view
                            )
                            base_tokens = estimate_tokens(
                                system
                                + state.current_user_text
                                + state.current_session_summary
                                + "\n".join(x.content for x in state.current_session_history)
                            )
                            cost = CostRecord(
                                pm_input_tokens_est=estimate_tokens(query) + sum(
                                    len(cat.catalog_fingerprint) for cat in state.inventory.values()
                                ),
                                catalog_reads=catalog_reads,
                                pre_evidence_compute_ms=pre_evidence_ms,
                                pm_inference_ms=pm_ms,
                                retrieval_latency_ms=retrieval_ms,
                                generation_latency_ms=result.latency_ms,
                                retrieval_calls=retrieval_calls,
                                reranker_calls=0,
                                memory_tokens=memory_tokens,
                                strategy_tokens=strategy_tokens,
                                base_prompt_tokens=base_tokens,
                                total_input_tokens=(
                                    result.usage["prompt_tokens"]
                                    or base_tokens + memory_tokens + strategy_tokens
                                ),
                                output_tokens=(
                                    result.usage["completion_tokens"]
                                    or estimate_tokens(supporter_message)
                                ),
                                latency_ms=pre_evidence_ms + pm_ms + retrieval_ms + result.latency_ms,
                                api_cost_usd=None,
                            )
                            turn_record = {
                                "user_id": str(user["id"]),
                                "topic_index": int(topic["idx"]),
                                "condition": condition,
                                "protocol": protocol,
                                "interaction_mode": interaction_mode,
                                "trajectory_comparability": (
                                    "causal_fixed_context_one_step"
                                    if interaction_mode == "fixed"
                                    else "associational_divergent_trajectory"
                                ),
                                "simulator_id": simulator_id,
                                "track_id": track_id,
                                "seed": int(seed),
                                "turn_index": turn_index,
                                "state_id": state.state_id,
                                "exogenous_state_id": state.provenance["exogenous_state_id"],
                                "card_id": state.card_id,
                                "context_before_turn": state_conversation,
                                "context_sha256": sha256_text(
                                    canonical_json(state_conversation)
                                ),
                                "seeker_message": seeker_message,
                                "supporter_message": supporter_message,
                                "action_id": action_id,
                                "selected_memory": [x.model_dump(mode="json") for x in memory_view],
                                "selected_strategy": [x.model_dump(mode="json") for x in strategy_view],
                                "cost": cost.model_dump(mode="json"),
                                "input_tokens": cost.total_input_tokens,
                                "output_tokens": cost.output_tokens,
                                "latency_ms": cost.latency_ms,
                                "pm_ood_report": pm_ood_report,
                                "pm_decision_report": pm_decision_report,
                            }
                            append_jsonl(turn_path, turn_record)
                            turn_records.append(turn_record)

                        append_jsonl(dialogue_path, {
                            "user_id": str(user["id"]),
                            "topic_index": int(topic["idx"]),
                            "condition": condition,
                            "protocol": protocol,
                            "interaction_mode": interaction_mode,
                            "trajectory_comparability": (
                                "causal_fixed_context_one_step"
                                if interaction_mode == "fixed"
                                else "associational_divergent_trajectory"
                            ),
                            "simulator_id": simulator_id,
                            "track_id": track_id,
                            "seed": int(seed),
                            "initial_greeting": NEUTRAL_INITIAL_GREETING,
                            "dialogue": conversation,
                            "dialogue_semantics": (
                                "display_only_stitched_open_loop_cases"
                                if interaction_mode == "fixed"
                                else "actual_interactive_trajectory"
                            ),
                            "evaluation_cases": evaluation_cases,
                            "turns": turn_records,
                        })
                    except Exception as exc:
                        failures.append({
                            "user_id": str(user["id"]),
                            "topic_index": int(topic["idx"]),
                            "condition": condition,
                            "seed": int(seed),
                            "simulator_id": simulator_id,
                            "interaction_mode": interaction_mode,
                            "error": f"{type(exc).__name__}: {exc}",
                        })
    finally:
        generator.close()
        if seeker is not None:
            seeker.close()

    expected = len(scenarios) * len(conditions) * len(seeds)
    completed_keys = load_done_keys(dialogue_path, done_fields)
    completed = len(completed_keys)
    malformed = []
    for row in iter_jsonl(dialogue_path):
        if len(row.get("turns") or []) != max_turns:
            malformed.append({
                "user_id": row.get("user_id"),
                "topic_index": row.get("topic_index"),
                "condition": row.get("condition"),
                "seed": row.get("seed"),
                "n_turns": len(row.get("turns") or []),
            })
        if not row.get("dialogue") or row["dialogue"][0] != {
            "role": "supporter", "content": NEUTRAL_INITIAL_GREETING
        }:
            malformed.append({
                "key": [row.get(x) for x in done_fields],
                "error": "missing neutral initial greeting",
            })
        if row.get("interaction_mode") == "fixed":
            cases = row.get("evaluation_cases") or []
            if len(cases) != max_turns:
                malformed.append({
                    "key": [row.get(x) for x in done_fields],
                    "error": "fixed track lacks one evaluation case per turn",
                })
            for turn in row.get("turns") or []:
                context = turn.get("context_before_turn") or []
                if not context or context[0] != {
                    "role": "supporter", "content": NEUTRAL_INITIAL_GREETING
                }:
                    malformed.append({
                        "key": [row.get(x) for x in done_fields],
                        "turn": turn.get("turn_index"),
                        "error": "bad fixed context",
                    })

    if interaction_mode == "fixed":
        # Every condition must see the exact same full context and seeker turn
        # at a comparison point. This catches accidental treatment-history
        # feedback or condition leakage before any judging begins.
        state_groups: dict[tuple[Any, ...], set[tuple[Any, ...]]] = {}
        for row in iter_jsonl(dialogue_path):
            for turn in row.get("turns") or []:
                group_key = (
                    row.get("user_id"), row.get("topic_index"), row.get("seed"),
                    row.get("simulator_id"), turn.get("turn_index"),
                )
                signature = (
                    turn.get("state_id"), turn.get("exogenous_state_id"),
                    turn.get("context_sha256"), turn.get("seeker_message"),
                )
                state_groups.setdefault(group_key, set()).add(signature)
        for key, signatures in state_groups.items():
            if len(signatures) != 1:
                malformed.append({
                    "key": key,
                    "error": "fixed-context state differs across conditions",
                    "n_signatures": len(signatures),
                })
    status = "COMPLETE" if completed == expected and not failures and not malformed else "INCOMPLETE"
    summary = {
        "status": status,
        "protocol": protocol,
        "interaction_mode": interaction_mode,
        "trajectory_interpretation": (
            "causal one-step policy comparison on identical full contexts; "
            "evaluated replies are not fed into later turns"
            if interaction_mode == "fixed"
            else "secondary associational simulation; policy worlds diverge"
        ),
        "simulator_id": simulator_id,
        "conditions": list(conditions),
        "seeds": [int(x) for x in seeds],
        "max_turns": max_turns,
        "scenarios_attempted": len(scenarios),
        "expected_dialogues": expected,
        "completed_dialogues": completed,
        "malformed": malformed,
        "failures": failures,
        "evoemo_sha256": sha256_file(evoemo_path),
        "strategy_bank_sha256": sha256_file(strategy_bank_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "selection_sha256": sha256_file(selection_path),
        "fixed_tracks_sha256": sha256_file(fixed_tracks_path) if fixed_tracks_path else None,
        "fixed_tracks_attestation_verification": fixed_tracks_verification,
        "generated_at": utc_now(),
    }
    write_json(summary_path, summary)
    if status != "COMPLETE":
        raise RuntimeError(
            f"EvoEmo generation incomplete: failures={len(failures)}, "
            f"malformed={len(malformed)}, completed={completed}/{expected}"
        )
    create_artifact_attestation(
        attestation_path,
        stage="evoemo_generation",
        inputs={
            "evoemo": evoemo_path,
            "strategy_bank": strategy_bank_path,
            "checkpoint": checkpoint_path,
            "selection": selection_path,
            "run_manifest": manifest_path,
            **({"fixed_tracks": fixed_tracks_path} if fixed_tracks_path else {}),
            **({"fixed_tracks_attestation": fixed_tracks_attestation_path} if fixed_tracks_verification else {}),
        },
        outputs={
            "dialogues": (dialogue_path, True),
            "turns": (turn_path, True),
            "raw_calls": (raw_path, True),
            "summary": (summary_path, False),
            "ood_preflight": (ood_path, False),
        },
        parameters={
            "protocol": protocol,
            "interaction_mode": interaction_mode,
            "fixed_track_design": (
                "policy_independent_open_loop_one_step_v2"
                if interaction_mode == "fixed" else None
            ),
            "simulator_id": simulator_id,
            "conditions": list(conditions),
            "max_turns": max_turns,
            "seeds": [int(x) for x in seeds],
        },
        expected={
            "dialogues": expected,
            "turns": expected * max_turns,
        },
        study_freeze_sha256=study_freeze_sha256,
    )
    return summary
