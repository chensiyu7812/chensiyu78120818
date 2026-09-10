"""Validate auxiliary risk IDs by attested transport, preserving every score."""
from __future__ import annotations
import argparse
from copy import copy, deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3

from metacom_pm.io import canonical_json, read_json, sha256_file, sha256_text, utc_now, write_json
from metacom_pm.table3_execution import Ledger, StopRun, SCHEMAS, verify_freeze

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/table3_under10/frozen_wire_v4'
FROZEN = ROOT / 'outputs/table3_under10/frozen_identity_v5'
ASSETS = ROOT / 'outputs/table3_under10/identity_amendment_v5'
RUN = ROOT / 'outputs/table3_under10/run_identity_v5'
spec = importlib.util.spec_from_file_location('frozen_wire_v4', ROOT / 'scripts/28_run_table3_wire.py')
wire = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wire)


def prove_transport(run_dir, ledger, aid, job, body):
    """Independent input/output lookup; never infer identity from model text."""
    matches = [(key, state) for key, state in ledger.batches().items() if aid in state['attempts']]
    if len(matches) != 1:
        raise StopRun('Transport identity has no unique owning Batch')
    key, state = matches[0]
    source = run_dir / f'{key}.batch_input.jsonl'
    if sha256_file(source) != state['input_sha256']:
        raise StopRun('Submitted Batch input hash differs')
    inputs = [json.loads(s) for s in source.read_text().splitlines() if json.loads(s)['custom_id'] == aid]
    outputs = []
    for field in ('output_file_id', 'error_file_id'):
        path = run_dir / f'{key}.{field}.jsonl'
        if path.exists():
            outputs.extend(json.loads(s) for s in path.read_text().splitlines() if json.loads(s)['custom_id'] == aid)
    if len(inputs) != 1 or len(outputs) != 1:
        raise StopRun('Duplicate or missing transport request/response')
    sent, received = inputs[0], outputs[0]
    if (sent['method'] != 'POST' or sent['url'] != '/v1/chat/completions'
        or canonical_json(sent['body']) != canonical_json(job['body'])
        or sha256_text(wire.wire_json(sent['body'])) != job['wire_sha256']
        or received.get('response', {}).get('status_code') != 200
        or canonical_json(received['response']['body']) != canonical_json(body)):
        raise StopRun('Transport request/response proof mismatch')
    return {'batch_key': key, 'batch_id': state['batch_id'], 'custom_id': aid,
            'input_file_sha256': sha256_file(source), 'provider_request_id': received['response'].get('request_id'),
            'response_sha256': sha256_text(canonical_json(body))}


class IdentityRunner(wire.WireRunner):
    def replies(self):
        # Apply the unchanged parent generation verifier with its real, pinned
        # source manifest. Only the new run's derived summary is written.
        source_view = copy(self)
        source_view.manifest = read_json(PARENT / 'manifest.json')
        return wire.WireRunner.replies(source_view)

    def record_response(self, aid, job, body, http_status=200):
        valid = super().record_response(aid, job, body, http_status)
        if valid or job['phase'] != 'main' or job['stage'] != 'risk' or http_status != 200:
            return valid
        try:
            choice = body['choices'][0]
            if body['model'] != job['body']['model'] or choice['finish_reason'] != 'stop' or choice['message'].get('refusal'):
                return False
            parsed = SCHEMAS['risk'].model_validate_json(choice['message']['content']).model_dump(mode='json')
            expected = json.loads(job['body']['messages'][1]['content'])['audit_call_id']
            if parsed['omission_severity'] != 0 or parsed['audit_call_id'] == expected:
                return False
            if not all(isinstance(body['usage'][k], int) and body['usage'][k] >= 0 for k in ('prompt_tokens', 'completion_tokens')):
                return False
        except (ValueError, KeyError, TypeError, IndexError):
            return False
        proof = prove_transport(self.run_dir, self.ledger, aid, job, body)
        proof.update(expected_audit_call_id=expected, model_echo_audit_call_id=parsed['audit_call_id'],
                     timestamp=utc_now(), rule='Transport identity authoritative; model echo preserved; all scores unchanged.')
        destination = self.run_dir / 'identity_annotations'
        destination.mkdir(exist_ok=True)
        write_json(destination / f'{aid}.json', proof)
        self.ledger.update(aid, status='done', parsed_json=canonical_json(parsed),
                           error='Auxiliary ID echo mismatch; independently verified Batch transport identity')
        return True

    def recover_existing(self):
        active = {j['id'] for j in self.jobs('main', self.config['judge'], pilot=False, batch=True)}
        recovered = []
        # Choose the earliest schema-valid, transport-attested response. Never
        # select among scores; later attempts remain charged and preserved.
        for row in self.ledger.rows('main'):
            if row['job_id'] not in active or row['status'] != 'invalid' or self.ledger.successful(row['job_id']):
                continue
            if self.record_response(row['id'], json.loads(row['job_json']), json.loads(row['response_json'])):
                recovered.append(row['id'])
        write_json(self.run_dir / 'identity_recovery.json', {'timestamp': utc_now(), 'recovered_without_api_calls': recovered,
            'policy': 'Earliest valid response; raw model ID, metrics, all prior attempts and costs preserved.'})

    def run(self):
        with self.lock():
            self.status('IDENTITY_RECOVERY')
            self.replies()
            self.recover_existing()
        super().run()
        path = self.run_dir / 'TABLE3_CN.md'
        with path.open('a') as stream:
            stream.write('\n\n辅助 audit ID 的模型回写可能漏抄字符。本版本以保存的 Batch custom_id、实际提交文件 hash、请求 body 和原始响应完成独立关联；回写差异原样留档，不改分数。相同输入有多次响应时采用最早满足完整性与关联规则的响应，所有费用保留。\n')


def initialize(manifest):
    RUN.mkdir(parents=True, exist_ok=True)
    target = RUN / 'ledger.sqlite'
    if not target.exists():
        with sqlite3.connect(f'file:{ASSETS / "parent_ledger.sqlite"}?mode=ro', uri=True) as src, sqlite3.connect(target) as dst:
            src.backup(dst)
            old = dict(dst.execute('SELECT key,value FROM meta'))
            if old['freeze'] != manifest['parent_freeze_sha256']:
                raise StopRun('Identity parent ledger mismatch')
            dst.execute("INSERT INTO meta VALUES ('identity_parent_meta',?)", (canonical_json(old),))
            dst.execute("UPDATE meta SET value=? WHERE key='freeze'", (manifest['freeze_sha256'],))
    Ledger(target).bind(manifest['freeze_sha256'])
    for source in (ASSETS / 'parent_runtime').iterdir():
        dest = RUN / source.name
        if dest.exists():
            if sha256_file(dest) != sha256_file(source):
                raise StopRun('Copied parent evidence changed')
        else:
            shutil.copyfile(source, dest)


def freeze():
    if FROZEN.exists():
        raise StopRun('Identity freeze already exists')
    parent = verify_freeze(ROOT, PARENT)
    config = deepcopy(read_json(PARENT / 'config.json'))
    config['protocol'] += '-transport-identity-v5'
    config['eos_amendment']['parent_ledger'] = str((ASSETS / 'parent_ledger.sqlite').relative_to(ROOT))
    config['identity_amendment'] = {'scope': 'Auxiliary model-echo ID check only; do not change any score, input, schema, metric, budget or attempt limit.',
        'authoritative_identity': 'Unique Batch custom_id plus immutable exact submitted wire body and saved raw response.',
        'response_choice': 'Earliest response passing schema, model, natural stop, usage, selected-only omission and transport identity checks.',
        'retention': 'Raw model echo, mismatch annotation, superseded attempts and every prior fee remain preserved.'}
    FROZEN.mkdir(parents=True)
    write_json(FROZEN / 'config.json', config)
    write_json(FROZEN / 'pilot_units.json', read_json(PARENT / 'pilot_units.json'))
    files = dict(parent['files'])
    extra = [PARENT / 'manifest.json', FROZEN / 'config.json', FROZEN / 'pilot_units.json',
             ROOT / 'scripts/29_run_table3_identity.py', ROOT / 'tests/test_table3_identity.py',
             ROOT / 'docs/TABLE3_IDENTITY_REPAIR_CN.md', ASSETS / 'parent_ledger.sqlite', ASSETS / 'incident.json']
    extra.extend(sorted((ASSETS / 'parent_runtime').iterdir()))
    files.update({str(p.relative_to(ROOT)): sha256_file(p) for p in extra})
    manifest = {'status': 'FROZEN_AUXILIARY_ID_VALIDATION_REPAIR', 'frozen_at': utc_now(),
                'parent_freeze_sha256': parent['freeze_sha256'], 'files': files, 'packages': parent['packages'],
                'budget_usd': 9, 'planned_states': 204, 'arms': 6}
    manifest['freeze_sha256'] = sha256_text(canonical_json(manifest))
    write_json(FROZEN / 'manifest.json', manifest)
    print(canonical_json({'status': 'FROZEN', 'freeze_sha256': manifest['freeze_sha256']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'verify', 'run'])
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze()
        return
    manifest = verify_freeze(ROOT, FROZEN)
    if args.command == 'verify':
        print(canonical_json({'status': 'PASS', 'freeze_sha256': manifest['freeze_sha256']}))
        return
    initialize(manifest)
    runner = IdentityRunner(ROOT, FROZEN, RUN)
    try:
        runner.run()
    except Exception as exc:
        runner.status('STOPPED', reason=str(exc) if isinstance(exc, StopRun) else type(exc).__name__)
        raise
    finally:
        runner.close()


if __name__ == '__main__':
    main()
