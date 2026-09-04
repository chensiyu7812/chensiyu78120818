"""Deterministic, fail-closed sharding for frozen PM-v1.5 call plans.

This module only partitions an already materialized call plan.  It does not
construct prompts, call providers, change retry policy, or merge scientific
results.  A shard executor must still bind the original full-plan hash and the
shard contract before it is permitted to make API calls.
"""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io import canonical_json, sha256_text


DETERMINISTIC_SHARDING_PROTOCOL = (
    "pm-v1.5-deterministic-physical-call-key-sharding-v1"
)
DEFAULT_SHARD_COUNT = 4


def _physical_call_key(row: Mapping[str, Any]) -> str:
    key = row.get("physical_call_key")
    if not isinstance(key, str) or not key:
        raise ValueError("every call-plan row must have a non-empty physical_call_key")
    return key


def shard_index_for_key(key: str, *, shard_count: int) -> int:
    """Assign one stable shard using the already frozen physical call key."""

    if shard_count < 2:
        raise ValueError("shard_count must be at least 2")
    if not key:
        raise ValueError("physical call key must be non-empty")
    # The key is itself a SHA-256 in current runners.  Hashing its text again
    # keeps the protocol well-defined even if a future key format changes.
    digest = sha256_text(key)
    return int(digest[:16], 16) % shard_count


def partition_call_plan(
    rows: Sequence[Mapping[str, Any]], *, shard_count: int = DEFAULT_SHARD_COUNT
) -> list[list[dict[str, Any]]]:
    """Partition exactly once, preserving the original order within a shard."""

    keys = [_physical_call_key(row) for row in rows]
    duplicates = sorted(key for key, count in Counter(keys).items() if count != 1)
    if duplicates:
        raise ValueError(f"call plan has duplicate physical keys: {duplicates[:3]}")
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for row, key in zip(rows, keys, strict=True):
        index = shard_index_for_key(key, shard_count=shard_count)
        shards[index].append(dict(row))
    validate_shard_partition(rows, shards, shard_count=shard_count)
    return shards


def validate_shard_partition(
    full_rows: Sequence[Mapping[str, Any]],
    shards: Sequence[Sequence[Mapping[str, Any]]],
    *,
    shard_count: int,
) -> None:
    """Reject missing, extra, duplicate, modified, or misassigned shard rows."""

    if len(shards) != shard_count:
        raise ValueError("number of shard collections does not match shard_count")
    full_by_key = {_physical_call_key(row): dict(row) for row in full_rows}
    if len(full_by_key) != len(full_rows):
        raise ValueError("full call plan contains duplicate physical keys")
    observed: dict[str, dict[str, Any]] = {}
    for shard_index, rows in enumerate(shards):
        for row in rows:
            key = _physical_call_key(row)
            if key in observed:
                raise ValueError(f"physical key appears in more than one shard: {key}")
            if key not in full_by_key:
                raise ValueError(f"shard contains an unplanned physical key: {key}")
            if dict(row) != full_by_key[key]:
                raise ValueError(f"shard row differs from frozen full-plan row: {key}")
            expected = shard_index_for_key(key, shard_count=shard_count)
            if shard_index != expected:
                raise ValueError(
                    f"physical key is assigned to shard {shard_index}, expected {expected}: {key}"
                )
            observed[key] = dict(row)
    missing = sorted(set(full_by_key) - set(observed))
    if missing:
        raise ValueError(f"shard partition is missing physical keys: {missing[:3]}")


def sharding_contract(
    full_rows: Sequence[Mapping[str, Any]],
    shards: Sequence[Sequence[Mapping[str, Any]]],
    *,
    shard_count: int,
    source_call_plan_path: str | Path,
) -> dict[str, Any]:
    """Return the content-addressed execution-partition contract."""

    validate_shard_partition(full_rows, shards, shard_count=shard_count)
    full_plan_sha256 = sha256_text(canonical_json(list(full_rows)))
    shard_records = []
    for index, rows in enumerate(shards):
        rows_list = list(rows)
        shard_records.append(
            {
                "shard_index": index,
                "row_count": len(rows_list),
                "call_plan_sha256": sha256_text(canonical_json(rows_list)),
                "logical_single_attempt_cost_usd": math.fsum(
                    float(row.get("maximum_cost_usd") or 0.0) for row in rows_list
                ),
                "maximum_physical_attempts": sum(
                    int(row.get("max_http_attempts") or 1) for row in rows_list
                ),
            }
        )
    payload: dict[str, Any] = {
        "protocol": DETERMINISTIC_SHARDING_PROTOCOL,
        "assignment": "sha256_text(physical_call_key)[:16] modulo shard_count",
        "shard_count": shard_count,
        "source_call_plan_path": str(source_call_plan_path),
        "source_call_plan_sha256": full_plan_sha256,
        "source_row_count": len(full_rows),
        "partition_exact_coverage": True,
        "scientific_request_rows_unchanged": True,
        "shards": shard_records,
    }
    payload["contract_sha256"] = sha256_text(canonical_json(payload))
    return payload


def validate_sharding_contract(
    full_rows: Sequence[Mapping[str, Any]],
    shards: Sequence[Sequence[Mapping[str, Any]]],
    contract: Mapping[str, Any],
) -> None:
    """Require a self-consistent contract for one exact full-plan partition."""

    payload = {
        key: value for key, value in dict(contract).items() if key != "contract_sha256"
    }
    if contract.get("contract_sha256") != sha256_text(canonical_json(payload)):
        raise ValueError("sharding contract self-hash mismatch")
    if contract.get("protocol") != DETERMINISTIC_SHARDING_PROTOCOL:
        raise ValueError("unexpected deterministic-sharding protocol")
    shard_count = int(contract.get("shard_count") or 0)
    validate_shard_partition(full_rows, shards, shard_count=shard_count)
    if contract.get("source_call_plan_sha256") != sha256_text(
        canonical_json(list(full_rows))
    ):
        raise ValueError("sharding contract full call-plan hash mismatch")
    if int(contract.get("source_row_count") or -1) != len(full_rows):
        raise ValueError("sharding contract full row count mismatch")
    observed_records = []
    for index, rows in enumerate(shards):
        rows_list = list(rows)
        observed_records.append(
            {
                "shard_index": index,
                "row_count": len(rows_list),
                "call_plan_sha256": sha256_text(canonical_json(rows_list)),
                "logical_single_attempt_cost_usd": math.fsum(
                    float(row.get("maximum_cost_usd") or 0.0) for row in rows_list
                ),
                "maximum_physical_attempts": sum(
                    int(row.get("max_http_attempts") or 1) for row in rows_list
                ),
            }
        )
    if contract.get("shards") != observed_records:
        raise ValueError("sharding contract per-shard records mismatch")


def load_validated_sharding_contract(
    *,
    full_rows: Sequence[Mapping[str, Any]],
    contract_path: str | Path,
) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    """Load a contract and all sibling shard plans, then verify exact coverage."""

    import json

    path = Path(contract_path)
    if not path.is_file():
        raise ValueError(f"sharding contract does not exist: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("sharding contract must be a JSON object")
    shard_count = int(contract.get("shard_count") or 0)
    if shard_count < 2:
        raise ValueError("sharding contract shard_count must be at least 2")
    shards: list[list[dict[str, Any]]] = []
    for index in range(shard_count):
        shard_path = path.parent / f"call_plan_shard_{index:02d}.jsonl"
        if not shard_path.is_file():
            raise ValueError(f"sharding contract lacks shard plan: {shard_path}")
        rows = []
        for line in shard_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"shard row is not an object: {shard_path}")
                rows.append(row)
        shards.append(rows)
    validate_sharding_contract(full_rows, shards, contract)
    return dict(contract), shards


def verify_result_key_coverage(
    planned_rows: Iterable[Mapping[str, Any]],
    result_rows: Iterable[Mapping[str, Any]],
) -> None:
    """Require exactly one terminal result row for every planned physical key."""

    planned = [_physical_call_key(row) for row in planned_rows]
    observed = [_physical_call_key(row) for row in result_rows]
    if len(planned) != len(set(planned)):
        raise ValueError("planned rows contain duplicate physical keys")
    if len(observed) != len(set(observed)):
        raise ValueError("result rows contain duplicate physical keys")
    missing = sorted(set(planned) - set(observed))
    extra = sorted(set(observed) - set(planned))
    if missing or extra:
        raise ValueError(
            f"result key coverage mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )
