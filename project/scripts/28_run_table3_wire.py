"""Repair Batch schema field order and uniformly re-evaluate the main table."""
from __future__ import annotations
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sqlite3

import httpx
from metacom_pm.io import canonical_json, read_json, sha256_file, sha256_text, utc_now, write_json
from metacom_pm.table3_execution import Runner, Ledger, StopRun, SCHEMAS, verify_freeze, NANO

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/table3_under10/frozen_eos_v3'
FROZEN = ROOT / 'outputs/table3_under10/frozen_wire_v4'
ASSETS = ROOT / 'outputs/table3_under10/wire_amendment_v4'
RUN = ROOT / 'outputs/table3_under10/run_wire_v4'
spec = importlib.util.spec_from_file_location('frozen_eos_v3', ROOT / 'scripts/27_run_table3_eos.py')
eos = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eos)


def wire_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=False)


def ordered_body(job):
    """Restore declaration order even after sorted ledger JSON round trips."""
    body = job['body']
    schema = SCHEMAS[job['stage']]
    result = {k: deepcopy(body[k]) for k in ('model', 'messages', 'temperature', 'seed', 'max_tokens')}
    result['response_format'] = {'type': 'json_schema', 'json_schema': {
        'name': schema.__name__, 'strict': True, 'schema': schema.model_json_schema()}}
    if canonical_json(result) != canonical_json(body):
        raise StopRun('Ordered serialization changed schema semantics or request values')
    return result


def initialize_ledger(manifest):
    target = RUN / 'ledger.sqlite'
    if target.exists():
        Ledger(target).bind(manifest['freeze_sha256'])
        return
    RUN.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f'file:{ASSETS / "parent_ledger.sqlite"}?mode=ro', uri=True) as src, sqlite3.connect(target) as dst:
        src.backup(dst)
        prior = dict(dst.execute('SELECT key,value FROM meta'))
        if prior['freeze'] != manifest['parent_freeze_sha256']:
            raise StopRun('Parent ledger binding mismatch')
        dst.execute("INSERT INTO meta VALUES ('wire_parent_meta',?)", (canonical_json(prior),))
        dst.execute("UPDATE meta SET value=? WHERE key='freeze'", (manifest['freeze_sha256'],))


class WireRunner(eos.EOSRunner):
    def replies(self):
        replies = Runner.replies(self)
        originals = {r['job_id']: r for r in self.ledger.rows('generation') if r['status'] == 'done'}
        required = {k for k, r in originals.items() if json.loads(r['parsed_json'])['finish_reason'] == 'length'}
        seen, records = set(), []
        for path in (ASSETS / 'extended_replies').glob('*.json'):
            item = read_json(path)
            key = item['original_job_id']
            if key not in required or key in seen:
                raise StopRun('Unexpected EOS source reply')
            seen.add(key)
            old = json.loads(originals[key]['response_json'])
            job = json.loads(originals[key]['job_json'])
            if (item['freeze_sha256'] != self.manifest['parent_freeze_sha256']
                or item['completion_token_ids'][:100] != old['local_backend']['completion_token_ids']
                or item['input_ids_sha256'] != old['local_backend']['input_ids_sha256']
                or item['asset_sha256'] != old['local_backend']['asset_sha256']
                or item['completion_token_ids'][-1] not in (128001, 128008, 128009)
                or (item['unit_id'], item['arm']) != (job['unit_id'], job['arm'])):
                raise StopRun('EOS source provenance mismatch')
            changed = replies[item['unit_id']][item['arm']] != item['text']
            replies[item['unit_id']][item['arm']] = item['text']
            records.append({'unit_id': item['unit_id'], 'arm': item['arm'], 'text_changed': changed,
                            'tokens': item['completion_tokens'], 'sha256': sha256_file(path)})
        if seen != required or len(seen) != 28:
            raise StopRun('Incomplete EOS source replies')
        write_json(self.run_dir / 'eos_generation_summary.json', {'reused_natural_replies': 1196, 'completed_cap_hits': 28,
            'text_changed': sum(r['text_changed'] for r in records), 'all_prefixes_match': True,
            'all_natural_stop': True, 'records': records, 'source_freeze_sha256': self.manifest['parent_freeze_sha256']})
        return replies

    def jobs(self, phase, judge, *, pilot, batch):
        jobs = Runner.jobs(self, phase, judge, pilot=pilot, batch=batch)
        result = []
        for job in jobs:
            if phase == 'main':
                job['body'] = ordered_body(job)
                job['wire_sha256'] = sha256_text(wire_json(job['body']))
                job['id'] = sha256_text('wire-v4:' + job['identity'] + ':' + job['wire_sha256'])[:40]
                job['wire_revision'] = 'native-schema-order-v4'
            else:
                job = eos.amended_job(job, self.source_attempts.get(job['id']))
            result.append(job)
        Runner.save_jobs(self, phase, result)
        return result

    def submit_batch(self, key, jobs):
        verify_freeze(self.root, self.frozen)
        existing = self.ledger.batches().get(key)
        if existing:
            return existing
        queued = self.ledger.reserve_many(jobs, self.execution['maximum_attempts_per_job'])
        if not queued:
            return {'status': 'imported'}
        lines = []
        for aid, job in queued:
            body = ordered_body(job)
            if sha256_text(wire_json(body)) != job['wire_sha256']:
                raise StopRun('Exact ordered request hash mismatch')
            lines.append(wire_json({'custom_id': aid, 'method': 'POST', 'url': '/v1/chat/completions', 'body': body}))
        payload = '\n'.join(lines) + '\n'
        path = self.run_dir / f'{key}.batch_input.jsonl'
        path.write_text(payload)
        state = {'status': 'prepared', 'attempts': [a for a, _ in queued], 'input_sha256': sha256_file(path), 'batch_id': None}
        self.ledger.batch_save(key, state)
        base = self.config['judge']['base_url'].rstrip('/')
        headers = self.headers(self.config['judge'])
        try:
            upload = self.http.post(base + '/v1/files', headers=headers, data={'purpose': 'batch'},
                                    files={'file': (path.name, payload.encode(), 'application/jsonl')})
            if upload.status_code != 200:
                state.update(status='upload_failed', http_status=upload.status_code)
                self.ledger.batch_save(key, state)
                raise StopRun('Batch upload failed before creation')
            state.update(input_file_id=upload.json()['id'], status='creating')
            self.ledger.batch_save(key, state)
            response = self.http.post(base + '/v1/batches', headers=headers, json={
                'input_file_id': state['input_file_id'], 'endpoint': '/v1/chat/completions', 'completion_window': '24h',
                'metadata': {'experiment': self.manifest['freeze_sha256'][:40], 'local_batch': key}})
            body = response.json()
            write_json(self.run_dir / f'{key}.create.json', body)
            if response.status_code != 200:
                state.update(status='creation_failed', http_status=response.status_code)
                self.ledger.batch_save(key, state)
                raise StopRun('Batch creation failed; reservations retained')
            state.update(status='submitted', batch_id=body['id'])
            self.ledger.batch_save(key, state)
            for aid, _ in queued:
                self.ledger.update(aid, status='queued')
            return state
        except httpx.HTTPError:
            raise StopRun('Batch network outcome uncertain; reservations retained; no blind retry') from None

    def full_scoring(self):
        jobs = self.jobs('main', self.config['judge'], pilot=False, batch=True)
        chunks, current, tokens = [], [], 0
        for job in jobs:
            if current and tokens + job['input_tokens_est'] > self.execution['batch_max_input_tokens']:
                chunks.append(current)
                current, tokens = [], 0
            current.append(job)
            tokens += job['input_tokens_est']
        if current:
            chunks.append(current)
        for i, chunk in enumerate(chunks):
            while True:
                pending = [j for j in chunk if not self.ledger.successful(j['id'])]
                if not pending:
                    break
                self.preflight()
                rows = self.ledger.rows('main')
                previous = [[r for r in rows if r['job_id'] == j['id']] for j in pending]
                if any(len(rs) >= self.execution['maximum_attempts_per_job'] for rs in previous):
                    raise StopRun('Frozen ordered-request attempt limit exhausted')
                if any(r['status'] != 'invalid' or r['actual_nano'] is None for rs in previous for r in rs):
                    raise StopRun('Unresolved ordered request; no automatic retry')
                attempt = 1 + max(map(len, previous))
                key = f'wire-main-{i:03d}-a{attempt}'
                state = self.submit_batch(key, pending)
                try:
                    if state['status'] != 'imported':
                        self.wait_batch(key, state)
                except StopRun as exc:
                    if str(exc) != 'Batch has incomplete or invalid results; reservations retained; no silent sample removal':
                        raise
                    # Next iteration checks known usage and total attempts, with
                    # every previous charge still included in the same ledger.

    def run(self):
        with self.lock():
            self.status('WIRE_ORDER_PREFLIGHT')
            self.replies()
            for key, state in self.ledger.batches().items():
                if state['status'] != 'imported':
                    self.wait_batch(key, state)
            self.preflight()
            self.pilot()
            self.full_scoring()
            self.export()
            from metacom_pm.table3_eos_analysis import analyze
            analyze(self)
            verify_wire_outputs(self)
            self.status('COMPLETE', output='TABLE3_CN.md')


def verify_wire_outputs(runner):
    jobs = runner.jobs('main', runner.config['judge'], pilot=False, batch=True)
    expected = {j['id']: j for j in jobs}
    verified = set()
    for path in runner.run_dir.glob('wire-main-*.batch_input.jsonl'):
        for line in path.read_text().splitlines():
            item = json.loads(line)
            jid = item['custom_id'].rsplit('-', 1)[0]
            job = expected[jid]
            if sha256_text(wire_json(item['body'])) != job['wire_sha256']:
                raise StopRun('Saved Batch ordered body mismatch')
            verified.add(jid)
    if verified != set(expected):
        raise StopRun('Missing ordered wire evidence for active main jobs')
    write_json(runner.run_dir / 'wire_completion_qa.json', {'status': 'PASS', 'active_main_wire_bodies': len(verified),
        'all_native_schema_order': True, 'prior_main_scores_excluded': True, 'cumulative_budget': runner.ledger.summary()})
    with (runner.run_dir / 'TABLE3_CN.md').open('a') as stream:
        stream.write('\n\n本次还修复了 Batch 序列化重排评分字段的问题。正式评分全部使用与同步 pilot 一致的原始 schema 字段顺序；旧批处理评分保留但未混入本表，其费用计入总额。字段顺序不一致已确认，但不能将它断言为先前空白循环的唯一原因。\n')


def freeze():
    if FROZEN.exists():
        raise StopRun('Wire freeze already exists')
    parent = verify_freeze(ROOT, PARENT)
    config = deepcopy(read_json(PARENT / 'config.json'))
    config['protocol'] += '-native-wire-order-v4'
    config['eos_amendment']['parent_ledger'] = str((ASSETS / 'parent_ledger.sqlite').relative_to(ROOT))
    config['wire_amendment'] = {'reason': 'Repair observed Batch schema-property reordering relative to synchronous pilot.',
        'wire_order': 'Native Pydantic schema property declaration order, preserved in exact submitted JSON.',
        'main_score_reuse': 'Re-evaluate all 2652 main jobs uniformly; exclude every older main score.',
        'pilot_score_reuse': 'Reuse identical synchronous pilot requests, which already used native order.',
        'other_changes': 'None to replies, model, rubric, score fields, thresholds, sampling, bootstrap, maximum attempts or cumulative budget.'}
    FROZEN.mkdir(parents=True)
    write_json(FROZEN / 'config.json', config)
    write_json(FROZEN / 'pilot_units.json', read_json(PARENT / 'pilot_units.json'))
    files = dict(parent['files'])
    extra = [PARENT / 'manifest.json', FROZEN / 'config.json', FROZEN / 'pilot_units.json',
             ROOT / 'scripts/28_run_table3_wire.py', ROOT / 'tests/test_table3_wire.py',
             ROOT / 'docs/TABLE3_WIRE_REPAIR_CN.md', ASSETS / 'parent_ledger.sqlite', ASSETS / 'incident.json']
    extra.extend(sorted((ASSETS / 'extended_replies').glob('*.json')))
    files.update({str(p.relative_to(ROOT)): sha256_file(p) for p in extra})
    manifest = {'status': 'FROZEN_WIRE_REPAIR_BEFORE_MAIN_RESUBMISSION', 'frozen_at': utc_now(),
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
    initialize_ledger(manifest)
    runner = WireRunner(ROOT, FROZEN, RUN)
    try:
        runner.run()
    except Exception as exc:
        runner.status('STOPPED', reason=str(exc) if isinstance(exc, StopRun) else type(exc).__name__)
        raise
    finally:
        runner.close()


if __name__ == '__main__':
    main()
