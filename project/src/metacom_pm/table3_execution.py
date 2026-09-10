"""Frozen Table III execution with durable request accounting and no hidden retries."""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
import csv
import fcntl
import importlib.metadata
import json
import math
import os
import sqlite3
import time

import httpx
import numpy as np

from .io import canonical_json, iter_jsonl, read_json, sha256_file, sha256_text, utc_now, write_json
from .table3 import (ARMS, EvoResponseV4Judgment, EvoSampledAuditJudgment,
                    OmissionJudgment, input_token_estimate, quality_messages,
                    risk_messages, omission_messages)
from .evo_response_v4 import SCORE_FIELDS, _summarize_pilot

SCHEMAS = {'quality': EvoResponseV4Judgment, 'risk': EvoSampledAuditJudgment, 'omission': OmissionJudgment}
NANO = 1_000_000_000


class StopRun(RuntimeError):
    pass


def nano_cost(inp, out, judge, multiplier=1):
    value = (Decimal(str(inp)) * Decimal(str(judge['input_usd_per_million']))
             + Decimal(str(out)) * Decimal(str(judge['output_usd_per_million'])))
    return int((value * Decimal(str(multiplier)) * 1000).to_integral_value(rounding=ROUND_CEILING))


def load_keys(path):
    """Read only the two expected literal dotenv assignments; never evaluate shell."""
    allowed = {'OPENAI_API_KEY', 'NVIDIA_API_KEY'}
    if path and Path(path).exists():
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[7:]
            key, sep, value = line.partition('=')
            key, value = key.strip(), value.strip()
            if sep and key in allowed:
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]
                os.environ.setdefault(key, value)
    return {key: bool(os.environ.get(key)) for key in sorted(allowed)}


def verify_freeze(root, frozen):
    manifest = read_json(frozen / 'manifest.json')
    unsigned = {k: v for k, v in manifest.items() if k != 'freeze_sha256'}
    if sha256_text(canonical_json(unsigned)) != manifest['freeze_sha256']:
        raise StopRun('Freeze manifest hash mismatch')
    for name, expected in manifest['files'].items():
        if sha256_file(root / name) != expected:
            raise StopRun(f'Frozen file changed: {name}')
    for name, version in manifest['packages'].items():
        if importlib.metadata.version(name) != version:
            raise StopRun(f'Frozen dependency changed: {name}')
    return manifest


class Ledger:
    """Integer nanodollar reservations, retained for unknown or queued usage."""
    def __init__(self, path, limit=9):
        self.path, self.limit = Path(path), int(Decimal(str(limit)) * NANO)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            db.execute('''CREATE TABLE IF NOT EXISTS attempts (
                id TEXT PRIMARY KEY, job_id TEXT NOT NULL, number INTEGER NOT NULL,
                phase TEXT NOT NULL, status TEXT NOT NULL, request_hash TEXT NOT NULL,
                job_json TEXT NOT NULL, reserve_nano INTEGER NOT NULL, actual_nano INTEGER,
                response_json TEXT, parsed_json TEXT, error TEXT, created_at TEXT NOT NULL,
                UNIQUE(job_id,number))''')
            db.execute('CREATE TABLE IF NOT EXISTS batches (id TEXT PRIMARY KEY, state_json TEXT NOT NULL)')

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA synchronous=FULL')
        return db

    def bind(self, freeze_hash):
        with self.connect() as db:
            old = db.execute("SELECT value FROM meta WHERE key='freeze'").fetchone()
            if old and old[0] != freeze_hash:
                raise StopRun('Ledger belongs to a different frozen experiment')
            db.execute("INSERT OR IGNORE INTO meta VALUES ('freeze',?)", (freeze_hash,))

    @staticmethod
    def committed(db):
        return db.execute('SELECT COALESCE(SUM(COALESCE(actual_nano,reserve_nano)),0) FROM attempts').fetchone()[0]

    def summary(self):
        with self.connect() as db:
            spent = db.execute('SELECT COALESCE(SUM(actual_nano),0) FROM attempts').fetchone()[0]
            return {'charged_upper_usd': spent / NANO,
                    'charged_plus_reserved_usd': self.committed(db) / NANO,
                    'limit_usd': self.limit / NANO,
                    'attempts_by_status': dict(db.execute('SELECT status,COUNT(*) FROM attempts GROUP BY status').fetchall())}

    def rows(self, phase=None):
        with self.connect() as db:
            query, args = ('SELECT * FROM attempts', ()) if phase is None else ('SELECT * FROM attempts WHERE phase=?', (phase,))
            return [dict(x) for x in db.execute(query + ' ORDER BY created_at,id', args)]

    def successful(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM attempts WHERE job_id=? AND status='done'", (job_id,)).fetchone()
            return dict(row) if row else None

    def reserve_many(self, jobs, maximum_attempts):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            queued, total = [], self.committed(db)
            for job in jobs:
                existing = db.execute('SELECT * FROM attempts WHERE job_id=? ORDER BY number', (job['id'],)).fetchall()
                expected_hash = sha256_text(canonical_json(job['body']))
                if any(x['request_hash'] != expected_hash for x in existing):
                    raise StopRun('Request changed across attempts')
                if any(x['status'] == 'done' for x in existing):
                    continue
                if any(x['status'] in ('reserved', 'in_flight', 'queued', 'unknown') for x in existing):
                    raise StopRun(f"Unresolved request must be reconciled before resubmission: {job['id']}")
                number = len(existing) + 1
                if number > maximum_attempts:
                    raise StopRun(f"Attempt limit reached: {job['id']}")
                total += job['reserve_nano']
                if total > self.limit:
                    raise StopRun('Budget reservation would exceed 9 USD; no new requests submitted')
                aid = f"{job['id']}-{number}"
                db.execute('INSERT INTO attempts (id,job_id,number,phase,status,request_hash,job_json,reserve_nano,created_at) VALUES (?,?,?,?,?,?,?,?,?)',
                           (aid, job['id'], number, job['phase'], 'reserved', expected_hash, canonical_json(job), job['reserve_nano'], utc_now()))
                queued.append((aid, job))
            return queued

    def update(self, aid, **values):
        allowed = {'status','actual_nano','response_json','parsed_json','error'}
        if not set(values) <= allowed:
            raise ValueError('Unexpected ledger update')
        with self.connect() as db:
            db.execute('UPDATE attempts SET ' + ','.join(f'{k}=?' for k in values) + ' WHERE id=?', (*values.values(), aid))

    def batch_save(self, key, state):
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO batches VALUES (?,?)', (key, canonical_json(state)))

    def batches(self):
        with self.connect() as db:
            return {r[0]: json.loads(r[1]) for r in db.execute('SELECT * FROM batches ORDER BY id')}


def judge_job(unit, arms, responses, judge, stage, phase, execution, *, arm=None, variant=0, batch=False):
    mapping = None
    if stage == 'quality':
        messages, mapping = quality_messages(unit, arms, responses, variant=variant)
    elif stage == 'risk':
        messages = risk_messages(unit, arm, responses[arm['arm']])
    else:
        messages = omission_messages(unit, responses[arm['arm']])
    schema = SCHEMAS[stage]
    body = {'model': judge['model'], 'messages': messages,
            'temperature': judge['temperature'], 'seed': judge['seed'],
            'max_tokens': judge[f'{stage}_max_tokens'],
            'response_format': {'type': 'json_schema', 'json_schema': {
                'name': schema.__name__, 'strict': True, 'schema': schema.model_json_schema()}}}
    inp = input_token_estimate(messages, schema)
    reserved_input = math.ceil(inp * execution['input_reservation_multiplier']) + execution['input_reservation_extra_tokens']
    identifier = f"{phase}:{unit['unit_id']}:{stage}:{arm['arm'] if arm else 'all'}:{variant}"
    return {'id': sha256_text(identifier)[:40], 'identity': identifier, 'phase': phase, 'stage': stage,
            'unit_id': unit['unit_id'], 'arm': arm['arm'] if arm else None, 'variant': variant,
            'mapping': mapping, 'body': body, 'judge': judge, 'batch': batch,
            'input_tokens_est': inp,
            'reserve_nano': nano_cost(reserved_input, body['max_tokens'], judge, 0.5 if batch else 1)}


def parse_response(job, body):
    if body.get('model') != job['body']['model']:
        raise ValueError('Returned model differs from frozen snapshot')
    choice = body['choices'][0]
    text = choice['message'].get('content')
    if choice['message'].get('refusal') or not isinstance(text, str) or not text.strip():
        raise ValueError('Refusal or empty response')
    if job['stage'] == 'generation':
        # Original v1 max_tokens=100 is preserved; length stops are observations.
        if choice.get('finish_reason') not in ('stop', 'length'):
            raise ValueError('Unexpected generation finish reason')
        return {'text': text.strip(), 'finish_reason': choice.get('finish_reason')}
    if choice.get('finish_reason') != 'stop':
        raise ValueError('Judge response truncated or unfinished')
    parsed = SCHEMAS[job['stage']].model_validate_json(text).model_dump(mode='json')
    if job['stage'] == 'quality':
        ids = [x['candidate_id'] for x in parsed['candidates']]
        if Counter(ids) != Counter(job['mapping'].keys()):
            raise ValueError('Missing, duplicate or extra candidate IDs')
    if job['stage'] == 'risk':
        payload = json.loads(job['body']['messages'][1]['content'])
        if parsed['audit_call_id'] != payload['audit_call_id'] or parsed['omission_severity'] != 0:
            raise ValueError('Audit ID mismatch or selected-only omission score')
    return parsed


class Runner:
    def __init__(self, root, frozen, run_dir, *, transport=None):
        self.root, self.frozen, self.run_dir = Path(root), Path(frozen), Path(run_dir)
        self.manifest = verify_freeze(self.root, self.frozen)
        self.config = read_json(self.frozen / 'config.json')
        self.execution = self.config['execution']
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(self.run_dir / 'ledger.sqlite', self.config['budget']['execution_reservation_limit_usd'])
        self.ledger.bind(self.manifest['freeze_sha256'])
        self.units = list(iter_jsonl(self.root / self.config['source_plan'] / 'units.jsonl'))
        self.arms = list(iter_jsonl(self.root / self.config['source_plan'] / 'arms.jsonl'))
        self.by_unit = defaultdict(list)
        for arm in self.arms:
            self.by_unit[arm['unit_id']].append(arm)
        self.pilot_ids = {u['unit_id'] for u in read_json(self.frozen / 'pilot_units.json')}
        self.http = httpx.Client(timeout=httpx.Timeout(180, connect=30), transport=transport)

    def close(self):
        self.http.close()

    @contextmanager
    def lock(self):
        with (self.run_dir / 'run.lock').open('a') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise StopRun('Another runner holds this experiment lock')
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def status(self, stage, **details):
        value = {'timestamp': utc_now(), 'stage': stage, 'freeze_sha256': self.manifest['freeze_sha256'],
                 'budget': self.ledger.summary(), **details}
        write_json(self.run_dir / 'status.json', value)
        print(canonical_json(value), flush=True)

    def headers(self, provider):
        key = os.environ.get(provider['api_key_env'])
        if not key:
            raise StopRun(f"Missing environment variable: {provider['api_key_env']}")
        return {'Authorization': 'Bearer ' + key}

    def record_response(self, aid, job, body, http_status=200):
        values = {'response_json': canonical_json(body)}
        usage = body.get('usage') or {}
        actual = None
        if job['stage'] == 'generation':
            actual = 0
        elif all(isinstance(usage.get(k), int) and usage[k] >= 0 for k in ('prompt_tokens', 'completion_tokens')):
            actual = nano_cost(usage['prompt_tokens'], usage['completion_tokens'], job['judge'], 0.5 if job['batch'] else 1)
        if actual is not None:
            values['actual_nano'] = actual
        try:
            if http_status != 200:
                raise ValueError(f'HTTP {http_status}')
            parsed = parse_response(job, body)
            if actual is None:
                raise ValueError('Provider usage missing')
            values.update(status='done', parsed_json=canonical_json(parsed))
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            values.update(status='invalid' if actual is not None else 'unknown', error=type(exc).__name__ + ': ' + str(exc).split('\n')[0][:180])
        self.ledger.update(aid, **values)
        if actual is not None and actual > job['reserve_nano']:
            raise StopRun('Actual cost exceeded its reservation; stop and reconcile before continuing')
        if self.ledger.summary()['charged_plus_reserved_usd'] > 9:
            raise StopRun('Budget limit reached')
        return values['status'] == 'done'

    def sync_job(self, job, provider):
        old = self.ledger.successful(job['id'])
        if old:
            if old['request_hash'] != sha256_text(canonical_json(job['body'])):
                raise StopRun('Completed request changed')
            return json.loads(old['parsed_json'])
        headers = self.headers(provider)
        for _ in range(self.execution['maximum_attempts_per_job']):
            aid, _ = self.ledger.reserve_many([job], self.execution['maximum_attempts_per_job'])[0]
            self.ledger.update(aid, status='in_flight')
            started = time.monotonic()
            try:
                response = self.http.post(provider['base_url'].rstrip('/') + '/v1/chat/completions', headers=headers, json=job['body'])
                try:
                    body = response.json()
                except ValueError:
                    body = {'error': {'type': 'non_json_response'}, 'http_status': response.status_code}
                write_json(self.run_dir / f'{aid}.response.json', {'body': body, 'http_status': response.status_code, 'latency_seconds': time.monotonic() - started})
                done = self.record_response(aid, job, body, response.status_code)
                if done:
                    return json.loads(self.ledger.successful(job['id'])['parsed_json'])
                # Explicit responses are retained even if usage is unknown; no automatic retry.
                if response.status_code in (400, 401, 403, 404) or not body.get('usage'):
                    raise StopRun(f"Provider request failed: HTTP {response.status_code}; see saved response {aid}")
            except httpx.HTTPError as exc:
                self.ledger.update(aid, status='unknown', error=type(exc).__name__)
                raise StopRun('Network result uncertain; reservation retained and automatic retry stopped') from None
        raise StopRun('Valid response not obtained within frozen attempt limit')

    def generation(self, pilot_only):
        verify_freeze(self.root, self.frozen)
        rows = [a for a in self.arms if not pilot_only or a['unit_id'] in self.pilot_ids]
        provider = self.config['generator']
        total = len(rows)
        for i, arm in enumerate(rows):
            body = {'model': provider['model'], 'messages': arm['messages'], 'temperature': provider['temperature'],
                    'max_tokens': provider['max_tokens'], 'seed': arm['generation_seed']}
            job = {'id': sha256_text('generation:' + arm['call_id'])[:40], 'phase': 'generation', 'stage': 'generation',
                   'unit_id': arm['unit_id'], 'arm': arm['arm'], 'body': body, 'reserve_nano': 0, 'batch': False}
            self.sync_job(job, provider)
            if i % 6 == 0 or i + 1 == total:
                self.status('PILOT_GENERATION' if pilot_only else 'FULL_GENERATION', completed=i + 1, total=total)

    def replies(self):
        replies = defaultdict(dict)
        for row in self.ledger.rows('generation'):
            if row['status'] == 'done':
                job = json.loads(row['job_json'])
                replies[job['unit_id']][job['arm']] = json.loads(row['parsed_json'])['text']
        return replies

    def jobs(self, phase, judge, *, pilot, batch):
        replies = self.replies()
        jobs = []
        for unit in self.units:
            if pilot and unit['unit_id'] not in self.pilot_ids:
                continue
            responses = replies[unit['unit_id']]
            if set(responses) != set(ARMS):
                raise StopRun('Generation is incomplete; all six responses are required')
            for variant in range(2 if pilot else 1):
                jobs.append(judge_job(unit, self.by_unit[unit['unit_id']], responses, judge, 'quality', phase, self.execution, variant=variant, batch=batch))
            for arm in self.by_unit[unit['unit_id']]:
                for stage in ('risk', 'omission'):
                    jobs.append(judge_job(unit, self.by_unit[unit['unit_id']], responses, judge, stage, phase, self.execution, arm=arm, batch=batch))
        self.save_jobs(phase, jobs)
        return jobs

    def save_jobs(self, phase, jobs):
        path = self.run_dir / f'{phase}.jobs.jsonl'
        content = ''.join(canonical_json(j) + '\n' for j in jobs)
        if path.exists():
            if path.read_text() != content:
                raise StopRun('Previously prepared jobs differ from current inputs')
        else:
            path.write_text(content)

    def pilot(self):
        for phase, judge in [('pilot_mini', self.config['judge']), ('pilot_reference', self.config['pilot']['reference_judge'])]:
            for i, job in enumerate(self.jobs(phase, judge, pilot=True, batch=False)):
                self.sync_job(job, judge)
                if i % 8 == 0:
                    self.status(phase.upper(), completed=i + 1, total=168)
        report = self.pilot_gate()
        write_json(self.run_dir / 'pilot_gate.json', report)
        if report['status'] != 'PASS':
            raise StopRun('Frozen pilot gate failed; full paid evaluation has not been submitted')
        self.status('PILOT_PASS', pilot_gate=report['checks'])

    def observations(self, phase):
        result = {}
        for row in self.ledger.rows(phase):
            if row['status'] != 'done':
                continue
            job, parsed = json.loads(row['job_json']), json.loads(row['parsed_json'])
            if job['stage'] == 'quality':
                for score in parsed['candidates']:
                    mapped = job['mapping'][score['candidate_id']]
                    result[(job['unit_id'], mapped['condition'], 'quality', job['variant'])] = {
                        **score, 'position': mapped['position'], 'condition': mapped['condition'],
                        'unit_id': job['unit_id'], 'order_variant': job['variant']}
            else:
                result[(job['unit_id'], job['arm'], job['stage'], 0)] = parsed
        return result

    def pilot_gate(self):
        mini, ref = self.observations('pilot_mini'), self.observations('pilot_reference')
        checks = {'complete': len(mini) == len(ref) == 288 and set(mini) == set(ref)}
        reports = {}
        gate = self.config['pilot_gate']
        for name, obs in [('mini', mini), ('reference', ref)]:
            reports[name] = _summarize_pilot([v for k, v in obs.items() if k[2] == 'quality'],
                max_order_mean_abs_diff=gate['max_order_mean_abs_diff'], max_position_mean_shift=gate['max_position_mean_shift'])
            checks[name + '_order_stability'] = reports[name]['status'] == 'PASS'
        overall, risk, omission, disagreements = [], [], [], []
        for key in sorted(set(mini) & set(ref)):
            a, b = mini[key], ref[key]
            stage = key[2]
            if stage == 'quality' and key[3] == 0:
                overall.append(abs(a['overall'] - b['overall']))
            elif stage == 'risk':
                risk.append((a['selected_evidence_misuse'] >= 2) == (b['selected_evidence_misuse'] >= 2))
            elif stage == 'omission':
                omission.append((a['omission_severity'] >= 2) == (b['omission_severity'] >= 2))
            if {k: v for k, v in a.items() if k != 'reason'} != {k: v for k, v in b.items() if k != 'reason'}:
                disagreements.append({'key': list(key), 'mini': a, 'reference': b})
        comparison = {'overall_mae': float(np.mean(overall)) if overall else None,
                      'clear_misuse_agreement': float(np.mean(risk)) if risk else None,
                      'clear_omission_agreement': float(np.mean(omission)) if omission else None}
        checks.update(overall_agreement=comparison['overall_mae'] is not None and comparison['overall_mae'] <= gate['max_reference_overall_mae'],
                      misuse_agreement=comparison['clear_misuse_agreement'] is not None and comparison['clear_misuse_agreement'] >= gate['minimum_reference_binary_agreement'],
                      omission_agreement=comparison['clear_omission_agreement'] is not None and comparison['clear_omission_agreement'] >= gate['minimum_reference_binary_agreement'])
        write_json(self.run_dir / 'pilot_disagreements.json', disagreements)
        return {'status': 'PASS' if all(checks.values()) else 'FAILED', 'checks': checks, 'order_reports': reports,
                'reference_comparison': comparison, 'thresholds': gate,
                'limitation': 'Technical screening only; same-family model reference is not human ground truth or proof of judge equivalence.'}

    def get(self, path):
        response = self.http.get(self.config['judge']['base_url'].rstrip('/') + path, headers=self.headers(self.config['judge']))
        response.raise_for_status()
        return response

    def submit_batch(self, key, jobs):
        """Reserve the entire batch before the potentially billable create request."""
        verify_freeze(self.root, self.frozen)
        prior = self.ledger.batches().get(key)
        if prior:
            return prior
        queued = self.ledger.reserve_many(jobs, self.execution['maximum_attempts_per_job'])
        if not queued:
            return {'status': 'imported'}
        payload = ''.join(canonical_json({'custom_id': aid, 'method': 'POST', 'url': '/v1/chat/completions', 'body': job['body']}) + '\n' for aid, job in queued)
        file_path = self.run_dir / f'{key}.batch_input.jsonl'
        file_path.write_text(payload)
        state = {'status': 'prepared', 'attempts': [a for a, _ in queued], 'input_sha256': sha256_file(file_path), 'batch_id': None}
        self.ledger.batch_save(key, state)
        base = self.config['judge']['base_url'].rstrip('/')
        headers = self.headers(self.config['judge'])
        try:
            upload = self.http.post(base + '/v1/files', headers=headers, data={'purpose': 'batch'}, files={'file': (file_path.name, payload.encode(), 'application/jsonl')})
            if upload.status_code != 200:
                state.update(status='upload_failed', http_status=upload.status_code)
                self.ledger.batch_save(key, state)
                raise StopRun('Batch file upload failed; no batch creation attempted')
            state['input_file_id'] = upload.json()['id']
            state['status'] = 'creating'
            self.ledger.batch_save(key, state)
            created = self.http.post(base + '/v1/batches', headers=headers, json={
                'input_file_id': state['input_file_id'], 'endpoint': '/v1/chat/completions', 'completion_window': '24h',
                'metadata': {'experiment': self.manifest['freeze_sha256'][:40], 'local_batch': key}})
            body = created.json()
            write_json(self.run_dir / f'{key}.create.json', body)
            if created.status_code != 200:
                state.update(status='creation_failed', http_status=created.status_code)
                self.ledger.batch_save(key, state)
                raise StopRun('Batch creation failed; reservations retained for reconciliation')
            state.update(status='submitted', batch_id=body['id'])
            self.ledger.batch_save(key, state)
            for aid, _ in queued:
                self.ledger.update(aid, status='queued')
            return state
        except httpx.HTTPError:
            raise StopRun('Batch network result uncertain; saved reservations prevent duplicate submission') from None

    def reconcile_batch_id(self, key, state):
        if state.get('batch_id'):
            return state
        if state['status'] == 'creating':
            after = None
            while True:
                path = '/v1/batches?limit=100' + ('&after=' + after if after else '')
                page = self.get(path).json()
                matches = [b for b in page['data'] if b.get('metadata', {}).get('local_batch') == key
                           and b.get('metadata', {}).get('experiment') == self.manifest['freeze_sha256'][:40]]
                if len(matches) == 1:
                    state.update(status='submitted', batch_id=matches[0]['id'])
                    self.ledger.batch_save(key, state)
                    return state
                if len(matches) > 1:
                    raise StopRun('Duplicate remote batches detected; reconcile before further work')
                if not page.get('has_more'):
                    break
                after = page['last_id']
        raise StopRun('Unresolved batch creation; no duplicate creation will be attempted')

    def wait_batch(self, key, state):
        state = self.reconcile_batch_id(key, state)
        while True:
            body = self.get('/v1/batches/' + state['batch_id']).json()
            write_json(self.run_dir / f'{key}.status.json', body)
            self.status('BATCH_WAIT', batch_key=key, batch_id=state['batch_id'], batch_status=body['status'], request_counts=body.get('request_counts'))
            if body['status'] in ('completed', 'failed', 'cancelled', 'expired'):
                break
            time.sleep(self.execution['batch_poll_seconds'])
        attempts = {r['id']: r for r in self.ledger.rows() if r['id'] in state['attempts']}
        seen = set()
        for field in ('output_file_id', 'error_file_id'):
            if not body.get(field):
                continue
            text = self.get('/v1/files/' + body[field] + '/content').text
            (self.run_dir / f'{key}.{field}.jsonl').write_text(text)
            for line in text.splitlines():
                result = json.loads(line)
                aid = result['custom_id']
                if aid not in attempts or aid in seen:
                    raise StopRun('Batch returned unexpected or duplicate custom_id')
                seen.add(aid)
                row = attempts[aid]
                if row['status'] == 'done':
                    continue
                response = result.get('response')
                if response:
                    self.record_response(aid, json.loads(row['job_json']), response.get('body') or {}, response['status_code'])
                else:
                    self.ledger.update(aid, status='unknown', response_json=canonical_json(result), error='Batch per-request error; reservation retained')
        for aid in set(attempts) - seen:
            self.ledger.update(aid, status='unknown', error='Batch ended without a result; reservation retained')
        state.update(status='imported', remote_status=body['status'])
        self.ledger.batch_save(key, state)
        if body['status'] != 'completed' or any(r['status'] != 'done' for r in self.ledger.rows() if r['id'] in state['attempts']):
            raise StopRun('Batch has incomplete or invalid results; reservations retained; no silent sample removal')

    def full_scoring(self):
        jobs = self.jobs('main', self.config['judge'], pilot=False, batch=True)
        total_reserved = sum(j['reserve_nano'] for j in jobs if not self.ledger.successful(j['id']))
        if total_reserved / NANO + self.ledger.summary()['charged_plus_reserved_usd'] > 9 and not self.ledger.batches():
            raise StopRun('Full scoring preflight exceeds remaining budget; no partial table submitted')
        chunks, current, inp = [], [], 0
        for job in jobs:
            if current and inp + job['input_tokens_est'] > self.execution['batch_max_input_tokens']:
                chunks.append(current)
                current, inp = [], 0
            current.append(job)
            inp += job['input_tokens_est']
        if current:
            chunks.append(current)
        for index, chunk in enumerate(chunks):
            key = f'main-{index:03d}'
            state = self.submit_batch(key, chunk)
            if state['status'] != 'imported':
                self.wait_batch(key, state)
            if any(not self.ledger.successful(j['id']) for j in chunk):
                raise StopRun('Full scoring incomplete')

    def run(self):
        with self.lock():
            keys = load_keys(None)
            self.status('PREFLIGHT', keys_present=keys)
            if not all(keys.values()):
                raise StopRun('Missing provider credentials; no paid calls made')
            self.generation(pilot_only=True)
            self.pilot()
            self.generation(pilot_only=False)
            self.full_scoring()
            self.export()
            self.status('COMPLETE', output='table3.csv')

    def export(self):
        verify_freeze(self.root, self.frozen)
        obs = self.observations('main')
        if len(obs) != 204 * 6 * 3:
            raise StopRun('Cannot export complete Table III: observations missing')
        unit_map = {u['unit_id']: u for u in self.units}
        rows = []
        for arm in self.arms:
            uid, name = arm['unit_id'], arm['arm']
            quality, risk, omission = (obs[(uid, name, s, 0)] for s in ('quality', 'risk', 'omission'))
            rows.append({'unit_id': uid, 'arm': name, 'user_id': unit_map[uid]['user_id'],
                         'scenario_id': f"{unit_map[uid]['user_id']}:{unit_map[uid]['topic_index']}",
                         **{k: quality[k] for k in SCORE_FIELDS},
                         'misuse_any': int(risk['selected_evidence_misuse'] >= 1),
                         'misuse_clear': int(risk['selected_evidence_misuse'] >= 2),
                         'misuse_severity': risk['selected_evidence_misuse'],
                         'major_issue_rate': int(risk['verdict'] == 'major_issue'),
                         'overall_risk': risk['overall_risk'],
                         'omission_clear': int(omission['omission_severity'] >= 2),
                         'omission_severity': omission['omission_severity'],
                         'source_invocations': arm['requested_source_invocations'],
                         'memory_tokens_est': arm['kept_memory_tokens_est'],
                         'candidate_memory_tokens_est': arm['candidate_memory_tokens_est']})
        (self.run_dir / 'observations.jsonl').write_text(''.join(canonical_json(r) + '\n' for r in rows))
        metrics = [k for k in rows[0] if k not in ('unit_id', 'arm', 'user_id', 'scenario_id')]
        table = [{'arm': arm, 'n': 204, **{k: float(np.mean([r[k] for r in rows if r['arm'] == arm])) for k in metrics}} for arm in ARMS]
        with (self.run_dir / 'table3.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        contrasts = [(f'{p}_on', f'{p}_off') for p in ('pm', 'rule', 'fixed')]
        contrasts += [(f'pm_{flag}', f'{p}_{flag}') for flag in ('off', 'on') for p in ('rule', 'fixed')]
        intervals = []
        by_pair = {(r['unit_id'], r['arm']): r for r in rows}
        for cluster in ('user_id', 'scenario_id'):
            clusters = sorted({r[cluster] for r in rows})
            rng = np.random.default_rng(self.config['analysis']['bootstrap_seed'])
            sampled = rng.integers(0, len(clusters), (self.config['analysis']['bootstrap_resamples'], len(clusters)))
            for left, right in contrasts:
                for metric in metrics:
                    groups = defaultdict(list)
                    for unit in self.units:
                        a, b = by_pair[(unit['unit_id'], left)], by_pair[(unit['unit_id'], right)]
                        groups[a[cluster]].append(a[metric] - b[metric])
                    sums = np.array([sum(groups[c]) for c in clusters])
                    sizes = np.array([len(groups[c]) for c in clusters])
                    boot = sums[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)
                    lo, hi = np.quantile(boot, [0.025, 0.975])
                    intervals.append({'left': left, 'right': right, 'metric': metric, 'cluster': cluster,
                                      'delta_left_minus_right': float(sums.sum() / sizes.sum()),
                                      'lower': float(lo), 'upper': float(hi), 'clusters': len(clusters)})
        write_json(self.run_dir / 'paired_intervals.json', intervals)
        columns = ['arm','overall','misuse_clear','misuse_severity','omission_clear','memory_tokens_est','source_invocations']
        lines = ['# Table III — frozen six-arm rerun', '', 'Main judge: gpt-4.1-mini-2025-04-14; n=204 per arm. Rates are fractions.', '',
                 '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |']
        for row in table:
            lines.append('| ' + ' | '.join(row[k] if k == 'arm' else f'{row[k]:.4f}' for k in columns) + ' |')
        lines += ['', 'All planned arms and states retained. Clear misuse/omission means severity ≥2; any misuse (≥1) is also exported in CSV.',
                  'Paired confidence intervals use a state-weighted mean and cluster resampling (user primary; scenario sensitivity).',
                  'This run changes the main judge and does not claim identical historical Table III numbers.', '']
        (self.run_dir / 'TABLE3.md').write_text('\n'.join(lines))
        write_json(self.run_dir / 'result_manifest.json', {'freeze_sha256': self.manifest['freeze_sha256'],
            'budget': self.ledger.summary(), 'hashes': {n: sha256_file(self.run_dir / n) for n in
                ('observations.jsonl','table3.csv','paired_intervals.json','TABLE3.md','pilot_gate.json')}})
