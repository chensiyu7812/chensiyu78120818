"""Synthetic transport/accounting fixtures only; no research observations or network."""
import copy
import json
from pathlib import Path

import httpx
import pytest

from metacom_pm.io import canonical_json, write_json, sha256_file, sha256_text
from metacom_pm.table3 import ARMS
from metacom_pm.table3_execution import Ledger, NANO, Runner, StopRun, judge_job, nano_cost, parse_response, verify_freeze


JUDGE = {'model': 'gpt-4.1-mini-2025-04-14', 'base_url': 'https://api.openai.com', 'api_key_env': 'OPENAI_API_KEY',
         'temperature': 0, 'seed': 20260910, 'input_usd_per_million': 0.4, 'output_usd_per_million': 1.6,
         'quality_max_tokens': 1600, 'risk_max_tokens': 600, 'omission_max_tokens': 300}
EXECUTION = {'input_reservation_multiplier': 1.25, 'input_reservation_extra_tokens': 64, 'maximum_attempts_per_job': 2}


def job(identifier='test', reserve=1_000_000):
    return {'id': identifier, 'phase': 'test', 'stage': 'generation', 'reserve_nano': reserve, 'batch': False,
            'body': {'model': 'test-model', 'messages': [{'role': 'user', 'content': 'synthetic fixture'}]}}


def bare_runner(tmp_path, transport=None):
    runner = Runner.__new__(Runner)
    runner.run_dir = tmp_path
    runner.ledger = Ledger(tmp_path/'ledger.sqlite')
    runner.execution = EXECUTION
    runner.http = httpx.Client(transport=transport)
    return runner


def test_reservations_are_atomic_and_include_queued_cost(tmp_path):
    ledger = Ledger(tmp_path/'ledger.sqlite', limit=9)
    ledger.reserve_many([job('first', 8*NANO)], 2)
    with pytest.raises(StopRun, match='Budget'):
        ledger.reserve_many([job('second', NANO), job('third', 1)], 2)
    assert len(ledger.rows()) == 1
    assert ledger.summary()['charged_plus_reserved_usd'] == 8


def test_unknown_usage_cannot_be_zeroed_or_retried(tmp_path):
    ledger = Ledger(tmp_path/'ledger.sqlite')
    aid, _ = ledger.reserve_many([job()], 2)[0]
    ledger.update(aid, status='unknown')
    assert ledger.summary()['charged_plus_reserved_usd'] == .001
    with pytest.raises(StopRun, match='Unresolved'):
        ledger.reserve_many([job()], 2)


def test_every_failed_paid_attempt_stays_in_total(tmp_path):
    ledger = Ledger(tmp_path/'ledger.sqlite')
    aid, _ = ledger.reserve_many([job()], 2)[0]
    ledger.update(aid, status='invalid', actual_nano=500_000)
    aid2, _ = ledger.reserve_many([job()], 2)[0]
    ledger.update(aid2, status='done', actual_nano=600_000)
    assert ledger.summary()['charged_upper_usd'] == .0011
    assert ledger.reserve_many([job()], 2) == []


def test_retry_with_changed_payload_is_rejected(tmp_path):
    ledger = Ledger(tmp_path/'ledger.sqlite')
    aid, _ = ledger.reserve_many([job()], 2)[0]
    ledger.update(aid, status='invalid', actual_nano=1)
    changed = job()
    changed['body']['model'] = 'other-model'
    with pytest.raises(StopRun, match='Request changed'):
        ledger.reserve_many([changed], 2)


def test_model_drift_and_truncation_are_not_valid_judge_outputs():
    j = {'stage':'omission', 'body':{'model':JUDGE['model']}}
    body = {'model':'different-model','choices':[{'finish_reason':'stop','message':{'content':'{}'}}]}
    with pytest.raises(ValueError, match='model'):
        parse_response(j, body)
    body['model'] = JUDGE['model']
    body['choices'][0]['finish_reason'] = 'length'
    with pytest.raises(ValueError, match='truncated'):
        parse_response(j, body)


def test_quality_requires_all_six_unique_ids_and_correct_unblinding():
    u = {'unit_id':'synthetic','ordinal':2,'turn_index':3,'context_before_turn':[],
         'seeker_message':'synthetic work concern','authorized_ground_truth':{}}
    j = judge_job(u, [], {a:'synthetic reply' for a in ARMS}, JUDGE, 'quality', 'test', EXECUTION)
    assert set(x['condition'] for x in j['mapping'].values()) == set(ARMS)
    scores = [{'candidate_id':f'C{i}', **{k:3 for k in ('emotional_support','personalization','memory_appropriateness',
               'factual_grounding','temporal_consistency','non_intrusiveness','overall')}, 'reason':'synthetic fixture'} for i in range(1,7)]
    body = {'model':JUDGE['model'],'choices':[{'finish_reason':'stop','message':{'content':canonical_json({'candidates':scores})}}]}
    assert len(parse_response(j, body)['candidates']) == 6
    scores[-1]['candidate_id'] = 'C1'
    body['choices'][0]['message']['content'] = canonical_json({'candidates':scores})
    with pytest.raises(ValueError, match='duplicate'):
        parse_response(j, body)


def test_timeout_keeps_reservation_and_does_not_resubmit(tmp_path, monkeypatch):
    calls = []
    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout('synthetic timeout')
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-test-key')
    runner = bare_runner(tmp_path, httpx.MockTransport(handler))
    with pytest.raises(StopRun, match='uncertain'):
        runner.sync_job(job(), JUDGE)
    with pytest.raises(StopRun, match='Unresolved'):
        runner.sync_job(job(), JUDGE)
    assert len(calls) == 1
    assert runner.ledger.rows()[0]['actual_nano'] is None
    runner.close()


def test_completed_request_is_not_sent_again(tmp_path, monkeypatch):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={'model':'test-model','choices':[{'finish_reason':'length','message':{'content':'synthetic generated response'}}]})
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-test-key')
    runner = bare_runner(tmp_path, httpx.MockTransport(handler))
    j = job(reserve=0)
    first = runner.sync_job(j, JUDGE)
    assert runner.sync_job(j, JUDGE) == first
    assert len(calls) == 1
    assert first['finish_reason'] == 'length'
    runner.close()


def test_usage_is_charged_even_when_judge_output_is_invalid(tmp_path):
    runner = bare_runner(tmp_path)
    j = job(reserve=NANO)
    j.update(stage='omission', judge=JUDGE)
    j['body']['model'] = JUDGE['model']
    aid, _ = runner.ledger.reserve_many([j], 2)[0]
    body = {'model':JUDGE['model'],'usage':{'prompt_tokens':1000,'completion_tokens':300},
            'choices':[{'finish_reason':'length','message':{'content':'{"omission_severity":'}}]}
    assert runner.record_response(aid,j,body) is False
    assert runner.ledger.rows()[0]['actual_nano'] == nano_cost(1000,300,JUDGE)
    runner.close()


def test_freeze_rejects_modified_code(tmp_path):
    p = tmp_path/'code.py'
    p.write_text('original')
    m = {'files':{'code.py':sha256_file(p)},'packages':{}}
    m['freeze_sha256'] = sha256_text(canonical_json(m))
    write_json(tmp_path/'manifest.json',m)
    assert verify_freeze(tmp_path,tmp_path) == m
    p.write_text('modified')
    with pytest.raises(StopRun,match='Frozen file changed'):
        verify_freeze(tmp_path,tmp_path)


def test_batch_half_price_and_integer_accounting():
    assert nano_cost(1_000_000,1_000_000,JUDGE) == 2*NANO
    assert nano_cost(1_000_000,1_000_000,JUDGE,.5) == NANO


def test_batch_reversed_outputs_are_joined_by_custom_id(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','synthetic-test-key')
    runner = bare_runner(tmp_path)
    runner.config = {'judge':JUDGE}
    runner.manifest = {'freeze_sha256':'synthetic-fixture'}
    jobs = []
    for i in range(2):
        j = job(f'batch-test-{i}',NANO)
        j.update(stage='omission',judge=JUDGE,batch=True)
        j['body']['model'] = JUDGE['model']
        jobs.append(j)
    pending = runner.ledger.reserve_many(jobs,2)
    state = {'batch_id':'batch-synthetic','status':'submitted','attempts':[a for a,_ in pending]}
    lines = []
    for i,(aid,j) in reversed(list(enumerate(pending))):
        body = {'model':JUDGE['model'],'usage':{'prompt_tokens':100,'completion_tokens':50},
                'choices':[{'finish_reason':'stop','message':{'content':canonical_json({
                    'omission_severity':i,'response_support_sufficiency':3,'reason':'synthetic fixture'})}}]}
        lines.append(canonical_json({'custom_id':aid,'response':{'status_code':200,'body':body}}))
    def handler(request):
        if '/batches/' in request.url.path:
            return httpx.Response(200,json={'status':'completed','output_file_id':'file-synthetic'})
        return httpx.Response(200,text='\n'.join(lines))
    runner.http.close()
    runner.http = httpx.Client(transport=httpx.MockTransport(handler))
    runner.wait_batch('synthetic',state)
    for i,j in enumerate(jobs):
        row = runner.ledger.successful(j['id'])
        assert json.loads(row['parsed_json'])['omission_severity'] == i
    assert runner.ledger.summary()['charged_plus_reserved_usd'] == 2*nano_cost(100,50,JUDGE,.5)/NANO
    runner.close()
