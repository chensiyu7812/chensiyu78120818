"""Local backend isolation and attestation tests, with synthetic responses only."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from metacom_pm.io import canonical_json, sha256_text
from metacom_pm.table3_execution import Ledger, StopRun


def script(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / name
    spec = spec_from_file_location(name.removesuffix('.py'), path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SERVER = script('24_local_llama_server.py')
LOCAL = script('25_run_table3_local.py')


def test_plain_text_format_has_exactly_one_bos_and_assistant_prefix():
    messages = [{'role':'system','content':' system '},{'role':'user','content':'hello\n'}]
    expected = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nsystem<|eot_id|>'
                '<|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|>'
                '<|start_header_id|>assistant<|end_header_id|>\n\n')
    assert SERVER.prompt_text(messages) == expected
    with pytest.raises(ValueError):
        SERVER.prompt_text([{'role':'tool','content':'unsupported'}])


def test_cloud_credentials_never_sent_to_local_server(monkeypatch):
    monkeypatch.setenv('NVIDIA_API_KEY','synthetic-private-key')
    runner = LOCAL.LocalRunner.__new__(LOCAL.LocalRunner)
    assert runner.headers({'local_generation':True,'api_key_env':'NVIDIA_API_KEY'}) == {}


def local_case(tmp_path):
    runner = LOCAL.LocalRunner.__new__(LOCAL.LocalRunner)
    runner.ledger = Ledger(tmp_path/'ledger.sqlite')
    runner.assets = {'asset_sha256':'expected-weights-and-code'}
    messages = [{'role':'user','content':'synthetic fixture'}]
    runner.local_inputs = {(sha256_text(canonical_json(messages)),104):'expected-token-ids'}
    job = {'id':'synthetic-local','phase':'generation','stage':'generation','reserve_nano':0,'batch':False,
           'body':{'model':'meta/llama-3.1-8b-instruct','messages':messages,'seed':104}}
    aid,_ = runner.ledger.reserve_many([job],2)[0]
    body = {'model':job['body']['model'],'choices':[{'finish_reason':'length','message':{'content':'synthetic response'}}],
            'local_backend':{'asset_sha256':'expected-weights-and-code','input_ids_sha256':'expected-token-ids'}}
    return runner,job,aid,body


def test_unexpected_local_token_ids_are_rejected_and_raw_response_kept(tmp_path):
    runner,job,aid,body = local_case(tmp_path)
    body['local_backend']['input_ids_sha256'] = 'wrong-prompt'
    with pytest.raises(StopRun,match='attestation'):
        runner.record_response(aid,job,body)
    row = runner.ledger.rows()[0]
    assert row['status'] == 'invalid' and row['actual_nano'] == 0
    assert 'wrong-prompt' in row['response_json']


def test_attested_local_response_is_accepted_without_api_charge(tmp_path):
    runner,job,aid,body = local_case(tmp_path)
    assert runner.record_response(aid,job,body)
    assert runner.ledger.successful(job['id'])
    assert runner.ledger.summary()['charged_plus_reserved_usd'] == 0
