from __future__ import annotations

import copy
import json

import pytest

from metacom_pm.v1_5_deterministic_sharding import (
    load_validated_sharding_contract,
    partition_call_plan,
    shard_index_for_key,
    sharding_contract,
    validate_sharding_contract,
    validate_shard_partition,
    verify_result_key_coverage,
)


def _rows(n: int = 40) -> list[dict]:
    return [
        {
            "physical_call_key": f"physical-{index:03d}",
            "maximum_cost_usd": index / 1000.0,
            "max_http_attempts": 4,
            "payload": {"index": index},
        }
        for index in range(n)
    ]


def test_partition_is_deterministic_and_exact() -> None:
    rows = _rows()
    first = partition_call_plan(rows, shard_count=4)
    second = partition_call_plan(rows, shard_count=4)
    assert first == second
    assert sum(map(len, first)) == len(rows)
    assert {
        row["physical_call_key"] for shard in first for row in shard
    } == {row["physical_call_key"] for row in rows}
    for index, shard in enumerate(first):
        assert all(
            shard_index_for_key(row["physical_call_key"], shard_count=4) == index
            for row in shard
        )


def test_partition_assignment_is_independent_of_input_order() -> None:
    rows = _rows()
    forward = partition_call_plan(rows, shard_count=4)
    reverse = partition_call_plan(list(reversed(rows)), shard_count=4)
    assert [
        {row["physical_call_key"] for row in shard} for shard in forward
    ] == [{row["physical_call_key"] for row in shard} for shard in reverse]


@pytest.mark.parametrize("failure", ["missing", "extra", "duplicate", "modified"])
def test_partition_validator_rejects_corruption(failure: str) -> None:
    rows = _rows()
    shards = partition_call_plan(rows, shard_count=4)
    broken = copy.deepcopy(shards)
    if failure == "missing":
        next(shard for shard in broken if shard).pop()
    elif failure == "extra":
        broken[0].append(
            {"physical_call_key": "extra", "maximum_cost_usd": 0, "max_http_attempts": 4}
        )
    elif failure == "duplicate":
        row = next(shard for shard in broken if shard)[0]
        broken[shard_index_for_key(row["physical_call_key"], shard_count=4)].append(row)
    else:
        next(shard for shard in broken if shard)[0]["payload"]["index"] = 999
    with pytest.raises(ValueError):
        validate_shard_partition(rows, broken, shard_count=4)


def test_contract_binds_full_plan_and_each_shard() -> None:
    rows = _rows()
    shards = partition_call_plan(rows, shard_count=4)
    contract = sharding_contract(
        rows,
        shards,
        shard_count=4,
        source_call_plan_path="/frozen/call_plan.jsonl",
    )
    assert contract["source_row_count"] == 40
    assert contract["partition_exact_coverage"] is True
    assert sum(record["row_count"] for record in contract["shards"]) == 40
    assert len(contract["contract_sha256"]) == 64
    validate_sharding_contract(rows, shards, contract)


def test_load_contract_rejects_tampering(tmp_path) -> None:
    rows = _rows()
    shards = partition_call_plan(rows, shard_count=4)
    contract = sharding_contract(
        rows,
        shards,
        shard_count=4,
        source_call_plan_path="/frozen/call_plan.jsonl",
    )
    contract_path = tmp_path / "sharding_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    for index, shard in enumerate(shards):
        path = tmp_path / f"call_plan_shard_{index:02d}.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in shard),
            encoding="utf-8",
        )
    loaded, loaded_shards = load_validated_sharding_contract(
        full_rows=rows,
        contract_path=contract_path,
    )
    assert loaded == contract
    assert loaded_shards == shards

    loaded_shards[0][0]["payload"]["index"] = 999
    (tmp_path / "call_plan_shard_00.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in loaded_shards[0]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_validated_sharding_contract(
            full_rows=rows,
            contract_path=contract_path,
        )


def test_result_key_coverage_is_fail_closed() -> None:
    rows = _rows(4)
    verify_result_key_coverage(rows, rows)
    with pytest.raises(ValueError):
        verify_result_key_coverage(rows, rows[:-1])
    with pytest.raises(ValueError):
        verify_result_key_coverage(rows, [*rows, {"physical_call_key": "extra"}])
