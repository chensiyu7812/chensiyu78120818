from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

SOURCE_ORDER: tuple[str, ...] = ("MP", "MS", "ME")
STRATEGY_ACTIONS: tuple[str, ...] = ("R0", "RS")

ACTION_TO_SOURCES: dict[str, frozenset[str]] = {
    "M0": frozenset(),
    "MP": frozenset({"MP"}),
    "MS": frozenset({"MS"}),
    "ME": frozenset({"ME"}),
    "MPMS": frozenset({"MP", "MS"}),
    "MPE": frozenset({"MP", "ME"}),
    "MSE": frozenset({"MS", "ME"}),
    "MPMSME": frozenset({"MP", "MS", "ME"}),
}

SOURCES_TO_ACTION: dict[frozenset[str], str] = {
    value: key for key, value in ACTION_TO_SOURCES.items()
}
MEMORY_ACTIONS: tuple[str, ...] = tuple(ACTION_TO_SOURCES.keys())
ACTION_IDS: tuple[str, ...] = tuple(
    f"{memory}+{strategy}"
    for memory in MEMORY_ACTIONS
    for strategy in STRATEGY_ACTIONS
)

FORBIDDEN_MEMORY_FIELDS: frozenset[str] = frozenset(
    {
        "memory_id",
        "id",
        "score",
        "retrieval_score",
        "is_stale",
        "is_conflicting",
        "is_sensitive",
        "catalog_tags",
        "tags",
        "expected_source",
        "gold_source",
        "audit_expected_source_family",
    }
)


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    memory_action: str
    strategy_action: str
    memory_sources: frozenset[str]


class V32ContractError(ValueError):
    """Raised when a V3.2 contract or schema invariant is violated."""


def parse_action(action_id: str) -> ActionSpec:
    if not isinstance(action_id, str) or "+" not in action_id:
        raise V32ContractError(f"Invalid action id: {action_id!r}")
    memory_action, strategy_action = action_id.split("+", 1)
    if memory_action not in ACTION_TO_SOURCES:
        raise V32ContractError(f"Unknown memory action: {memory_action!r}")
    if strategy_action not in STRATEGY_ACTIONS:
        raise V32ContractError(f"Unknown strategy action: {strategy_action!r}")
    return ActionSpec(
        action_id=action_id,
        memory_action=memory_action,
        strategy_action=strategy_action,
        memory_sources=ACTION_TO_SOURCES[memory_action],
    )


def canonical_memory_action(sources: Iterable[str]) -> str:
    source_set = frozenset(sources)
    unknown = source_set - set(SOURCE_ORDER)
    if unknown:
        raise V32ContractError(f"Unknown memory sources: {sorted(unknown)}")
    try:
        return SOURCES_TO_ACTION[source_set]
    except KeyError as exc:
        raise V32ContractError(
            f"Unsupported memory source set: {sorted(source_set)}"
        ) from exc


def canonical_action(sources: Iterable[str], strategy_action: str) -> str:
    if strategy_action not in STRATEGY_ACTIONS:
        raise V32ContractError(f"Unknown strategy action: {strategy_action!r}")
    return f"{canonical_memory_action(sources)}+{strategy_action}"


def required_sources(memory_action_or_action_id: str) -> tuple[str, ...]:
    if "+" in memory_action_or_action_id:
        sources = parse_action(memory_action_or_action_id).memory_sources
    else:
        if memory_action_or_action_id not in ACTION_TO_SOURCES:
            raise V32ContractError(
                f"Unknown memory action: {memory_action_or_action_id!r}"
            )
        sources = ACTION_TO_SOURCES[memory_action_or_action_id]
    return tuple(source for source in SOURCE_ORDER if source in sources)


def require_exact_keys(
    obj: Mapping[str, Any],
    required_keys: Iterable[str],
    context: str = "object",
) -> None:
    if not isinstance(obj, Mapping):
        raise V32ContractError(f"{context} must be a JSON object")
    required = set(required_keys)
    actual = set(obj.keys())
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    if missing or extra:
        raise V32ContractError(
            f"{context} keys mismatch: missing={missing} extra={extra}"
        )


def strict_int_012(value: Any, context: str = "value") -> int:
    if type(value) is not int or value not in (0, 1, 2):
        raise V32ContractError(
            f"{context} must be strict int 0|1|2, got {value!r}"
        )
    return value


def strict_int_range(value: Any, lo: int, hi: int, context: str = "value") -> int:
    if type(value) is not int or not (lo <= value <= hi):
        raise V32ContractError(
            f"{context} must be strict int in [{lo}, {hi}], got {value!r}"
        )
    return value


def strict_bool(value: Any, context: str = "value") -> bool:
    if type(value) is not bool:
        raise V32ContractError(f"{context} must be strict bool, got {value!r}")
    return value


def strict_choice(value: Any, choices: Sequence[str], context: str = "value") -> str:
    if type(value) is not str or value not in set(choices):
        raise V32ContractError(
            f"{context} must be one of {list(choices)}, got {value!r}"
        )
    return value


def strict_source_list(value: Any, context: str = "sources") -> list[str]:
    if not isinstance(value, list):
        raise V32ContractError(f"{context} must be a list")
    output: list[str] = []
    seen: set[str] = set()
    for index, source in enumerate(value):
        if type(source) is not str or source not in SOURCE_ORDER:
            raise V32ContractError(
                f"{context}[{index}] invalid source: {source!r}"
            )
        if source in seen:
            raise V32ContractError(f"{context} duplicate source: {source!r}")
        seen.add(source)
        output.append(source)
    return output


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def sanitize_memory_store(
    memory_store: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Return evaluator-safe memory items and original-id -> opaque-id map."""

    clean: dict[str, list[dict[str, Any]]] = {
        source: [] for source in SOURCE_ORDER
    }
    id_map: dict[str, str] = {}
    index = 0
    for source in SOURCE_ORDER:
        items = memory_store.get(source) or []
        if not isinstance(items, Sequence):
            raise V32ContractError(
                f"memory_store[{source}] must be a sequence"
            )
        for item in items:
            if not isinstance(item, Mapping):
                raise V32ContractError(
                    f"memory_store[{source}] item must be object"
                )
            original_id = str(
                item.get("memory_id") or item.get("id") or f"{source}_{index}"
            )
            opaque_id = f"mem_{index:03d}"
            index += 1
            text = normalize_text(item.get("text"))
            if not text:
                raise V32ContractError(
                    f"memory item {original_id!r} has empty text"
                )
            sanitized = {
                "opaque_id": opaque_id,
                "source": source,
                "session_index": item.get("session_index"),
                "age_sessions": item.get("age_sessions"),
                "text": text,
            }
            leaked = set(sanitized) & FORBIDDEN_MEMORY_FIELDS
            if leaked:
                raise V32ContractError(
                    f"sanitized item leaked forbidden fields: {sorted(leaked)}"
                )
            clean[source].append(sanitized)
            id_map[original_id] = opaque_id
    return clean, id_map


def sanitize_selected_memory_ids(
    ids: Sequence[Any], id_map: Mapping[str, str]
) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in ids:
        key = str(raw)
        if key not in id_map:
            raise V32ContractError(
                f"selected memory id not present in sanitized map: {key!r}"
            )
        opaque = id_map[key]
        if opaque in seen:
            raise V32ContractError(
                f"duplicate selected memory id: {opaque!r}"
            )
        seen.add(opaque)
        output.append(opaque)
    return output
