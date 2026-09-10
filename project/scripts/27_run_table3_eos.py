"""Authorized EOS amendment, preserving every old charge and reusable score."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

from metacom_pm.io import canonical_json, read_json, sha256_file, sha256_text, utc_now, write_json
from metacom_pm.table3_execution import Ledger, Runner, StopRun, verify_freeze, NANO

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'outputs/table3_under10/frozen_eos_v3'
RUN = ROOT / 'outputs/table3_under10/run_eos_v3'
PARENT = ROOT / 'outputs/table3_under10/frozen_local_v2'
SNAPSHOT = ROOT / 'outputs/table3_under10/eos_amendment_v3/parent_ledger.sqlite'


def amended_job(job, previous):
    """Only changed existing requests get a new ID; their old charges stay put."""
    if previous and previous['request_hash'] != sha256_text(canonical_json(job['body'])):
        job = deepcopy(job)
        job['id'] = sha256_text('eos-v3:' + job['identity'] + ':' + sha256_text(canonical_json(job['body'])))[:40]
    return job


def import_ledger(target, parent, freeze_hash, parent_hash):
    if target.exists():
        Ledger(target).bind(freeze_hash)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f'file:{parent}?mode=ro', uri=True) as source, sqlite3.connect(target) as dest:
        source.backup(dest)
        old = dest.execute("SELECT value FROM meta WHERE key='freeze'").fetchone()[0]
        if old != parent_hash:
            raise StopRun('Parent ledger binding differs')
        # This is a new database copied from a hash-pinned, immutable parent.
        # All attempts, fees and remote reservations survive byte-for-byte.
        dest.execute("INSERT INTO meta VALUES ('parent_freeze',?)", (old,))
        dest.execute("INSERT INTO meta VALUES ('parent_ledger_sha256',?)", (sha256_file(parent),))
        dest.execute("UPDATE meta SET value=? WHERE key='freeze'", (freeze_hash,))


class EOSRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_attempts = {}
        for row in self.ledger.rows():
            self.source_attempts.setdefault(row['job_id'], row)

    def replies(self):
        replies = super().replies()
        originals = {r['job_id']: r for r in self.ledger.rows('generation') if r['status'] == 'done'}
        required = {jid for jid, r in originals.items() if json.loads(r['parsed_json'])['finish_reason'] == 'length'}
        extended = list((self.run_dir / 'extended_replies').glob('*.json'))
        if any(p.name.endswith('.failure.json') for p in extended):
            raise StopRun('EOS generation has a recorded failure')
        seen, diagnostics = set(), []
        for path in extended:
            item = read_json(path)
            jid = item['original_job_id']
            if jid not in required or jid in seen:
                raise StopRun('Unexpected or duplicate EOS replacement')
            seen.add(jid)
            original = json.loads(originals[jid]['response_json'])
            old_job = json.loads(originals[jid]['job_json'])
            if (item['freeze_sha256'] != self.manifest['freeze_sha256'] or not item['natural_stop'] or not item['prefix_matches']
                or item['completion_token_ids'][:100] != original['local_backend']['completion_token_ids']
                or item['input_ids_sha256'] != original['local_backend']['input_ids_sha256']
                or item['asset_sha256'] != original['local_backend']['asset_sha256']
                or (item['unit_id'], item['arm']) != (old_job['unit_id'], old_job['arm'])
                or item['completion_token_ids'][-1] not in (128001, 128008, 128009)):
                raise StopRun('EOS replacement provenance differs')
            changed = replies[item['unit_id']][item['arm']] != item['text']
            replies[item['unit_id']][item['arm']] = item['text']
            diagnostics.append({'unit_id': item['unit_id'], 'arm': item['arm'], 'text_changed': changed,
                                'tokens': item['completion_tokens'], 'sha256': sha256_file(path)})
        if seen != required or len(seen) != 28:
            raise StopRun('EOS replacements are incomplete')
        write_json(self.run_dir / 'eos_generation_summary.json', {'reused_natural_replies': 1196, 'completed_cap_hits': len(seen),
            'text_changed': sum(r['text_changed'] for r in diagnostics), 'all_prefixes_match': True,
            'all_natural_stop': True, 'records': diagnostics})
        return replies

    def jobs(self, phase, judge, *, pilot, batch):
        # Parent implementation calls our save_jobs, which is disabled until IDs
        # are amended. Do not persist conflicting intermediate job IDs.
        raw = super().jobs(phase, judge, pilot=pilot, batch=batch)
        jobs = [amended_job(j, self.source_attempts.get(j['id'])) for j in raw]
        Runner.save_jobs(self, phase, jobs)
        return jobs

    def save_jobs(self, phase, jobs):
        pass

    def observations(self, phase):
        judge = self.config['pilot']['reference_judge'] if phase == 'pilot_reference' else self.config['judge']
        desired = self.jobs(phase, judge, pilot=phase != 'main', batch=phase == 'main')
        result = {}
        for job in desired:
            row = self.ledger.successful(job['id'])
            if not row:
                continue
            if row['request_hash'] != sha256_text(canonical_json(job['body'])):
                raise StopRun('Active score request hash differs')
            parsed = json.loads(row['parsed_json'])
            if job['stage'] == 'quality':
                for score in parsed['candidates']:
                    mapped = job['mapping'][score['candidate_id']]
                    result[(job['unit_id'], mapped['condition'], 'quality', job['variant'])] = {
                        **score, 'position': mapped['position'], 'condition': mapped['condition'],
                        'unit_id': job['unit_id'], 'order_variant': job['variant']}
            else:
                result[(job['unit_id'], job['arm'], job['stage'], 0)] = parsed
        return result

    def preflight(self):
        missing = []
        for phase, judge, pilot, batch in [('pilot_mini', self.config['judge'], True, False),
            ('pilot_reference', self.config['pilot']['reference_judge'], True, False), ('main', self.config['judge'], False, True)]:
            missing.extend(j for j in self.jobs(phase, judge, pilot=pilot, batch=batch) if not self.ledger.successful(j['id']))
        projected = self.ledger.summary()['charged_plus_reserved_usd'] + sum(j['reserve_nano'] for j in missing) / NANO
        write_json(self.run_dir / 'amended_budget_preflight.json', {'charged': self.ledger.summary(),
            'pending_by_phase': dict(Counter(j['phase'] for j in missing)), 'all_remaining_reserved_upper_usd': projected,
            'limit_usd': 9})
        if projected > 9:
            raise StopRun('Amended full-run preflight exceeds cumulative 9 USD limit')

    def full_scoring(self):
        pending = [j for j in self.jobs('main', self.config['judge'], pilot=False, batch=True) if not self.ledger.successful(j['id'])]
        # The immutable complete jobs file determines stable chunk membership on
        # resume; include done jobs in those chunks and let reservations skip them.
        jobs = self.jobs('main', self.config['judge'], pilot=False, batch=True)
        total = self.ledger.summary()['charged_plus_reserved_usd'] + sum(j['reserve_nano'] for j in pending) / NANO
        if total > 9:
            raise StopRun('Remaining full scoring exceeds cumulative budget')
        chunks, current, inp = [], [], 0
        for job in jobs:
            if current and inp + job['input_tokens_est'] > self.execution['batch_max_input_tokens']:
                chunks.append(current)
                current, inp = [], 0
            current.append(job)
            inp += job['input_tokens_est']
        if current:
            chunks.append(current)
        for i, chunk in enumerate(chunks):
            key = f'eos-main-{i:03d}'
            state = self.submit_batch(key, chunk)
            if state['status'] != 'imported':
                self.wait_batch(key, state)
            if any(not self.ledger.successful(j['id']) for j in chunk):
                raise StopRun('Amended full scoring incomplete')

    def run(self):
        with self.lock():
            self.status('EOS_GENERATION_START')
            env = {'PATH': os.environ['PATH'], 'PYTHONNOUSERSITE': '1', 'HF_HUB_OFFLINE': '1',
                   'CUDA_VISIBLE_DEVICES': 'GPU-ba5f65be-ecfa-7fda-b239-af9efd468db4'}
            subprocess.run(['/home/tokkio/miniconda3/envs/sim_eval/bin/python', '-u',
                            str(self.root / 'scripts/26_complete_table3_eos.py')], env=env, check=True)
            self.replies()
            for key, state in self.ledger.batches().items():
                if state['status'] != 'imported':
                    self.wait_batch(key, state)
            self.preflight()
            self.pilot()
            self.full_scoring()
            self.export()
            # Analysis runs only after all 204 x 6 x 3 valid active observations.
            from metacom_pm.table3_eos_analysis import analyze
            analyze(self)
            self.status('COMPLETE', output='TABLE3_CN.md')


def freeze():
    if FROZEN.exists():
        raise StopRun('EOS freeze already exists')
    parent = verify_freeze(ROOT, PARENT)
    config = deepcopy(read_json(PARENT / 'config.json'))
    config['protocol'] += '-natural-eos-v3'
    config['generator']['max_tokens'] = None
    config['local_generator']['generation_max_tokens'] = None
    config['eos_amendment'] = {'authorization': 'User: 好的，解除限制，重跑后，开始表3的分析。',
        'parent_freeze_sha256': parent['freeze_sha256'], 'parent_ledger': str(SNAPSHOT.relative_to(ROOT)),
        'cap_hit_count': 28, 'reuse_naturally_ended': 1196,
        'generation_rule': 'Greedy generation until EOS; no experiment-level output-token cutoff. Native model context ceiling exhaustion is an error, not a valid truncated response.',
        'selection': 'All 28 length-stopped responses, all six arms, selected solely by saved stop reason.',
        'integrity': 'Same inputs, weights, tokenizer, dtype, decoder and seeds; exact original 100-token prefix required.',
        'scores': 'Reuse only byte-identical request bodies; retain superseded scores and all prior charges; update both pilot judges and apply original gates.',
        'main': 'All 204 states and six arms; same judge, schema, metrics, bootstrap and budget as parent.'}
    FROZEN.mkdir(parents=True)
    write_json(FROZEN / 'config.json', config)
    write_json(FROZEN / 'pilot_units.json', read_json(PARENT / 'pilot_units.json'))
    files = dict(parent['files'])
    extra = [PARENT / 'manifest.json', SNAPSHOT, FROZEN / 'config.json', FROZEN / 'pilot_units.json',
             ROOT / 'scripts/26_complete_table3_eos.py', ROOT / 'scripts/27_run_table3_eos.py',
             ROOT / 'src/metacom_pm/table3_eos_analysis.py', ROOT / 'tests/test_table3_eos.py',
             ROOT / 'docs/TABLE3_EOS_AMENDMENT_CN.md']
    files.update({str(p.relative_to(ROOT)): sha256_file(p) for p in extra})
    manifest = {'frozen_at': utc_now(), 'status': 'AUTHORIZED_EOS_AMENDMENT_BEFORE_REGENERATION',
                'parent_freeze_sha256': parent['freeze_sha256'], 'files': files, 'packages': parent['packages'],
                'budget_usd': 9, 'planned_states': 204, 'arms': 6}
    manifest['freeze_sha256'] = sha256_text(canonical_json(manifest))
    write_json(FROZEN / 'manifest.json', manifest)
    print(canonical_json({'status': 'FROZEN', 'freeze_sha256': manifest['freeze_sha256']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'verify', 'run', 'export'])
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze()
        return
    manifest = verify_freeze(ROOT, FROZEN)
    if args.command == 'verify':
        print(canonical_json({'status': 'PASS', 'freeze_sha256': manifest['freeze_sha256']}))
        return
    import_ledger(RUN / 'ledger.sqlite', SNAPSHOT, manifest['freeze_sha256'], manifest['parent_freeze_sha256'])
    runner = EOSRunner(ROOT, FROZEN, RUN)
    try:
        if args.command == 'run':
            runner.run()
        else:
            with runner.lock():
                runner.export()
                from metacom_pm.table3_eos_analysis import analyze
                analyze(runner)
    except Exception as exc:
        runner.status('STOPPED', reason=str(exc) if isinstance(exc, StopRun) else type(exc).__name__)
        raise
    finally:
        runner.close()


if __name__ == '__main__':
    main()
