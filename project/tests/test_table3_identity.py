import importlib.util
import json
from pathlib import Path
import pytest
from metacom_pm.io import canonical_json, sha256_file, sha256_text
from metacom_pm.table3_execution import Ledger, StopRun

spec = importlib.util.spec_from_file_location('identity_runner', Path(__file__).resolve().parents[1] / 'scripts/29_run_table3_identity.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path, omission=0):
    body = {'model': 'gpt-4.1-mini-2025-04-14', 'messages': [{'role': 'system', 'content': 'fixture'},
            {'role': 'user', 'content': json.dumps({'audit_call_id': 'expected-id'})}]}
    job = {'id': 'fixture', 'phase': 'main', 'stage': 'risk', 'body': body,
           'reserve_nano': 1_000_000_000, 'batch': True,
           'judge': {'input_usd_per_million': .4, 'output_usd_per_million': 1.6},
           'wire_sha256': sha256_text(module.wire.wire_json(body))}
    score = {'audit_call_id': 'typo-id', 'verdict': 'acceptable', 'selected_evidence_misuse': 0,
             'unnecessary_exposure': 0, 'stale_or_conflict': 0, 'unsupported_personal_claim': 0,
             'source_set_appropriateness': 5, 'strategy_overuse': 0, 'strategy_omission': 0,
             'omission_severity': omission, 'response_support_sufficiency': 5, 'overall_risk': 0, 'reason': 'fixture'}
    response = {'model': body['model'], 'usage': {'prompt_tokens': 100, 'completion_tokens': 100},
                'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(score)}}]}
    ledger = Ledger(tmp_path / 'ledger.sqlite')
    aid = ledger.reserve_many([job], 2)[0][0]
    ledger.update(aid, status='queued')
    source = tmp_path / 'fixture.batch_input.jsonl'
    source.write_text(module.wire.wire_json({'custom_id': aid, 'method': 'POST', 'url': '/v1/chat/completions', 'body': body}) + '\n')
    out = {'custom_id': aid, 'response': {'status_code': 200, 'request_id': 'fixture-provider-id', 'body': response}}
    (tmp_path / 'fixture.output_file_id.jsonl').write_text(canonical_json(out) + '\n')
    ledger.batch_save('fixture', {'attempts': [aid], 'batch_id': 'fixture-batch', 'input_sha256': sha256_file(source)})
    runner = module.IdentityRunner.__new__(module.IdentityRunner)
    runner.run_dir, runner.ledger = tmp_path, ledger
    return runner, aid, job, response, score


def test_typo_is_accepted_only_with_proof_and_original_values_preserved(tmp_path):
    runner, aid, job, response, score = fixture(tmp_path)
    assert runner.record_response(aid, job, response)
    row = runner.ledger.successful(job['id'])
    assert json.loads(row['parsed_json']) == score
    assert json.loads(row['response_json']) == response
    assert row['actual_nano'] > 0
    annotation = json.loads((tmp_path / 'identity_annotations' / f'{aid}.json').read_text())
    assert annotation['expected_audit_call_id'] == 'expected-id'
    assert annotation['model_echo_audit_call_id'] == 'typo-id'


def test_score_validity_is_not_relaxed(tmp_path):
    runner, aid, job, response, _ = fixture(tmp_path, omission=1)
    assert not runner.record_response(aid, job, response)
    assert not runner.ledger.successful(job['id'])
    assert not (tmp_path / 'identity_annotations').exists()


def test_wrong_transport_custom_id_is_rejected(tmp_path):
    runner, aid, job, response, _ = fixture(tmp_path)
    path = tmp_path / 'fixture.output_file_id.jsonl'
    value = json.loads(path.read_text())
    value['custom_id'] = 'different-request'
    path.write_text(canonical_json(value) + '\n')
    with pytest.raises(StopRun):
        runner.record_response(aid, job, response)
    assert not runner.ledger.successful(job['id'])


def test_changed_submitted_input_is_rejected(tmp_path):
    runner, aid, job, response, _ = fixture(tmp_path)
    path = tmp_path / 'fixture.batch_input.jsonl'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(StopRun):
        runner.record_response(aid, job, response)
