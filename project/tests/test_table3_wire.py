import importlib.util
import json
from pathlib import Path
import pytest
from metacom_pm.io import canonical_json, sha256_text
from metacom_pm.table3_execution import SCHEMAS, StopRun

spec = importlib.util.spec_from_file_location('wire_runner', Path(__file__).resolve().parents[1] / 'scripts/28_run_table3_wire.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('stage', ['quality', 'risk', 'omission'])
def test_native_schema_order_survives_ledger_round_trip(stage):
    cls = SCHEMAS[stage]
    original = {'model': 'frozen', 'messages': [{'role': 'user', 'content': 'fixture'}], 'temperature': 0,
                'seed': 1, 'max_tokens': 100, 'response_format': {'type': 'json_schema',
                'json_schema': {'name': cls.__name__, 'strict': True, 'schema': cls.model_json_schema()}}}
    stored = json.loads(canonical_json({'stage': stage, 'body': original}))
    restored = module.ordered_body(stored)
    assert canonical_json(restored) == canonical_json(original)
    wire = json.loads(module.wire_json(restored))
    actual = wire['response_format']['json_schema']['schema']
    if stage == 'quality':
        actual = actual['$defs']['EvoResponseV4CandidateScore']
    assert list(actual['properties']) == actual['required']
    assert list(actual['properties'])[-1] == 'reason'
    assert sha256_text(module.wire_json(restored)) != sha256_text(canonical_json(original))


def test_wire_repair_does_not_silently_change_schema_constraints():
    cls = SCHEMAS['omission']
    body = {'model': 'frozen', 'messages': [], 'temperature': 0, 'seed': 1, 'max_tokens': 100,
            'response_format': {'type': 'json_schema', 'json_schema': {'name': cls.__name__, 'strict': True, 'schema': cls.model_json_schema()}}}
    body['response_format']['json_schema']['schema']['additionalProperties'] = True
    with pytest.raises(StopRun):
        module.ordered_body({'stage': 'omission', 'body': body})


def test_actual_batch_upload_preserves_native_order(tmp_path, monkeypatch):
    import httpx
    from metacom_pm.table3_execution import Ledger
    cls = SCHEMAS['quality']
    body = {'model': 'frozen', 'messages': [], 'temperature': 0, 'seed': 1, 'max_tokens': 100,
            'response_format': {'type': 'json_schema', 'json_schema': {'name': cls.__name__, 'strict': True, 'schema': cls.model_json_schema()}}}
    job = {'id': 'fixture', 'phase': 'main', 'stage': 'quality', 'body': body, 'reserve_nano': 100,
           'wire_sha256': sha256_text(module.wire_json(body))}
    inspected = []

    def handle(request):
        if request.url.path == '/v1/files':
            payload = request.read().decode()
            start = payload.index('{"custom_id"')
            wire = json.loads(payload[start:].splitlines()[0])
            schema = wire['body']['response_format']['json_schema']['schema']['$defs']['EvoResponseV4CandidateScore']
            assert list(schema['properties']) == schema['required']
            assert list(schema['properties'])[-1] == 'reason'
            inspected.append(wire)
            return httpx.Response(200, json={'id': 'file-fixture'})
        assert request.url.path == '/v1/batches'
        return httpx.Response(200, json={'id': 'batch-fixture'})

    runner = module.WireRunner.__new__(module.WireRunner)
    runner.root = runner.frozen = runner.run_dir = tmp_path
    runner.ledger = Ledger(tmp_path / 'ledger.sqlite')
    runner.execution = {'maximum_attempts_per_job': 2}
    runner.config = {'judge': {'base_url': 'https://fixture.invalid'}}
    runner.manifest = {'freeze_sha256': 'fixture'}
    runner.headers = lambda _: {}
    runner.http = httpx.Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(module, 'verify_freeze', lambda *_: runner.manifest)
    state = runner.submit_batch('fixture', [job])
    assert state['status'] == 'submitted' and len(inspected) == 1
    saved = json.loads((tmp_path / 'fixture.batch_input.jsonl').read_text())
    assert module.wire_json(saved['body']) == module.wire_json(body)
    assert runner.ledger.rows()[0]['status'] == 'queued'
    runner.http.close()
