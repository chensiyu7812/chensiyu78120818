import importlib.util
import json
from pathlib import Path
import pytest
from metacom_pm.io import canonical_json, sha256_file, sha256_text
from metacom_pm.table3_execution import Ledger, StopRun

spec = importlib.util.spec_from_file_location('risk_omission_exception_v7', Path(__file__).resolve().parents[1] / 'scripts/31_run_table3_risk_omission_exception_v7.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path, omission=1, finish_reason='stop'):
    body = {'model': 'gpt-4.1-mini-2025-04-14', 'messages': [{'role': 'system', 'content': 'fixture'},
            {'role': 'user', 'content': json.dumps({'audit_call_id': 'expected-id'})}]}
    job = {'id': module.EXCEPTION_JOB_ID, 'phase': 'main', 'stage': 'risk', 'body': body,
           'reserve_nano': 1_000_000_000, 'batch': True,
           'judge': {'input_usd_per_million': .4, 'output_usd_per_million': 1.6},
           'wire_sha256': sha256_text(module.identity.wire.wire_json(body))}
    score = {'audit_call_id': 'expected-id', 'verdict': 'acceptable', 'selected_evidence_misuse': 0,
             'unnecessary_exposure': 0, 'stale_or_conflict': 0, 'unsupported_personal_claim': 0,
             'source_set_appropriateness': 5, 'strategy_overuse': 0, 'strategy_omission': 1,
             'omission_severity': omission, 'response_support_sufficiency': 4, 'overall_risk': 1, 'reason': 'fixture'}
    response = {'model': body['model'], 'usage': {'prompt_tokens': 100, 'completion_tokens': 100},
                'choices': [{'finish_reason': finish_reason, 'message': {'content': json.dumps(score)}}]}
    ledger = Ledger(tmp_path / 'ledger.sqlite')
    aid = ledger.reserve_many([job], 5)[0][0]
    assert aid == module.EXCEPTION_ATTEMPT_ID
    ledger.update(aid, status='queued')
    source = tmp_path / 'fixture.batch_input.jsonl'
    source.write_text(module.identity.wire.wire_json({'custom_id': aid, 'method': 'POST', 'url': '/v1/chat/completions', 'body': body}) + '\n')
    out = {'custom_id': aid, 'response': {'status_code': 200, 'request_id': 'fixture-provider-id', 'body': response}}
    (tmp_path / 'fixture.output_file_id.jsonl').write_text(canonical_json(out) + '\n')
    ledger.batch_save('fixture', {'attempts': [aid], 'batch_id': 'fixture-batch', 'input_sha256': sha256_file(source)})
    ledger.update(aid, status='invalid', actual_nano=805800, response_json=canonical_json(response),
                  error='ValueError: Audit ID mismatch or selected-only omission score')
    runner = module.identity.IdentityRunner.__new__(module.identity.IdentityRunner)
    runner.run_dir, runner.ledger = tmp_path, ledger
    return runner, aid, score


def test_documented_exception_preserves_raw_omission_severity_unmodified(tmp_path):
    runner, aid, score = fixture(tmp_path)
    proof = module.apply_documented_exception(runner)
    assert proof is not None
    row = runner.ledger.successful(module.EXCEPTION_JOB_ID)
    assert json.loads(row['parsed_json']) == score
    assert json.loads(row['parsed_json'])['omission_severity'] == 1
    annotation = json.loads((tmp_path / 'risk_omission_exceptions' / f'{aid}.json').read_text())
    assert annotation['omission_severity_returned'] == 1


def test_is_idempotent_once_already_resolved(tmp_path):
    runner, aid, score = fixture(tmp_path)
    assert module.apply_documented_exception(runner) is not None
    assert module.apply_documented_exception(runner) is None


def test_refuses_when_response_is_actually_compliant(tmp_path):
    # If omission_severity were 0, this would not be the documented incident at all;
    # the exception must never be used to paper over a different, unreviewed case.
    runner, aid, score = fixture(tmp_path, omission=0)
    with pytest.raises(StopRun):
        module.apply_documented_exception(runner)


def test_refuses_when_response_no_longer_matches_the_reviewed_incident(tmp_path):
    runner, aid, score = fixture(tmp_path, finish_reason='length')
    with pytest.raises(StopRun):
        module.apply_documented_exception(runner)


def test_scope_is_a_single_named_job_not_a_general_rule():
    assert module.EXCEPTION_JOB_ID in module.build_config({'protocol': 'p', 'execution': {}})['risk_omission_exception']['scope']
