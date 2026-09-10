"""Approved local-generator amendment; original v1 freeze and runner stay intact."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import httpx

from metacom_pm.io import canonical_json, read_json, sha256_file, sha256_text, utc_now, write_json
from metacom_pm.table3_execution import Ledger, Runner, StopRun, load_keys, verify_freeze


class LocalRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.assets = read_json(self.root / self.config['local_generator']['model_manifest'])
        if self.assets['asset_sha256'] != self.config['local_generator']['asset_sha256']:
            raise StopRun('Model asset manifest differs from frozen configuration')
        records_path = (self.root / self.config['local_generator']['model_manifest']).parent / 'generation_inputs.jsonl'
        self.local_inputs = {(r['prompt_messages_sha256'], r['seed']): r['input_ids_sha256'] for r in
                             (json.loads(line) for line in records_path.read_text().splitlines())}

    def headers(self, provider):
        if provider.get('local_generation'):
            return {}  # Cloud credentials are never sent to the local server.
        return super().headers(provider)

    def record_response(self, aid, job, body, http_status=200):
        if job['stage'] == 'generation' and http_status == 200:
            expected_ids = self.local_inputs[(sha256_text(canonical_json(job['body']['messages'])), job['body']['seed'])]
            attestation = body.get('local_backend') or {}
            if (attestation.get('asset_sha256') != self.assets['asset_sha256']
                    or attestation.get('input_ids_sha256') != expected_ids):
                self.ledger.update(aid, status='invalid', actual_nano=0, response_json=canonical_json(body), error='Local backend or prompt attestation differs')
                raise StopRun('Local backend or prompt attestation differs; response excluded')
        return super().record_response(aid, job, body, http_status)

    def generation(self, pilot_only):
        rows = [a for a in self.arms if not pilot_only or a['unit_id'] in self.pilot_ids]
        pending = any(not self.ledger.successful(sha256_text('generation:' + a['call_id'])[:40]) for a in rows)
        if pending:
            try:
                response = self.http.get(self.config['generator']['base_url'] + '/health')
                response.raise_for_status()
                health = response.json()
            except (httpx.HTTPError, ValueError):
                raise StopRun('Local model service is not ready') from None
            if (health.get('asset_sha256') != self.assets['asset_sha256'] or health.get('model') != self.config['generator']['model']
                    or health.get('gpu') != self.config['local_generator']['gpu'] or health.get('dtype') != 'torch.bfloat16'):
                raise StopRun('Local model health attestation differs')
        super().generation(pilot_only)

    def full_scoring(self):
        # All generation is durable at this point; release this experiment's GPU.
        subprocess.run(['systemctl', '--user', 'stop', self.config['local_generator']['service_unit']], check=True)
        super().full_scoring()

    def run(self):
        with self.lock():
            present = bool(os.environ.get('OPENAI_API_KEY'))
            self.status('PREFLIGHT_LOCAL', openai_key_present=present, local_asset_sha256=self.assets['asset_sha256'])
            if not present:
                raise StopRun('OPENAI_API_KEY missing; no paid request made')
            self.generation(pilot_only=True)
            self.pilot()
            self.generation(pilot_only=False)
            self.full_scoring()
            self.export()
            self.status('COMPLETE', output='table3.csv')


def freeze_local(root, target):
    parent_dir = root / 'outputs/table3_under10/frozen_v1'
    parent = verify_freeze(root, parent_dir)
    if target.exists():
        raise StopRun('Local freeze already exists; it must not be overwritten')
    old_ledger = Ledger(root / 'outputs/table3_under10/run_v1/ledger.sqlite')
    if old_ledger.summary()['charged_plus_reserved_usd'] != 0:
        raise StopRun('Parent has nonzero fees/reservations; must carry them forward explicitly')
    if any(r['phase'] != 'generation' for r in old_ledger.rows()):
        raise StopRun('Unexpected parent scoring observations')
    asset_path = root / 'outputs/table3_under10/local_model_v2/model_manifest.json'
    assets = read_json(asset_path)
    if sha256_text(canonical_json({k: v for k, v in assets.items() if k != 'asset_sha256'})) != assets['asset_sha256']:
        raise StopRun('Local asset manifest hash differs')
    if assets['server_code_sha256'] != sha256_file(root / 'scripts/24_local_llama_server.py'):
        raise StopRun('Server code differs from verified model package')
    config = deepcopy(read_json(parent_dir / 'config.json'))
    config['protocol'] += '-local-generator-v2'
    config['generator'].update(base_url='http://127.0.0.1:18081', api_key_env='LOCAL_NO_CLOUD_CREDENTIAL', local_generation=True,
                               cost_basis='User approved same-model local A6000 generation; no external generation API charge.')
    config['local_generator'] = {'model_manifest': str(asset_path.relative_to(root)), 'asset_sha256': assets['asset_sha256'],
                                 'service_unit': 'pm-table3-llama-local-v2.service', 'dtype': 'bfloat16', 'quantization': None,
                                 'gpu': 'NVIDIA RTX A6000',
                                 'decoder_cleanup_spaces': False, 'generation_max_tokens': 100, 'greedy': True}
    config['budget']['previous_run_total_usd'] = 0
    config['amendment'] = {
        'parent_freeze_sha256': parent['freeze_sha256'],
        'reason': 'NVIDIA first generation request returned HTTP 410, model EOL 2026-08-26. User explicitly approved local A6000 recovery.',
        'authorization': 'User: 可以的，推进吧。 (approval of same 8B local recovery proposal)',
        'changes': ['Generation endpoint and numerical serving backend', 'Pinned mirror tokenizer template and per-request token hashes',
                    'Local service attestation before generation and in every generation response'],
        'unchanged': ['Weights identical to official Meta shard hashes', '204 states and six arms', 'PM checkpoint and filter',
                      'All generation messages, temperature, max_tokens and seeds', 'All judge models/prompts/pilot gates',
                      'OpenAI pricing, 9 USD limit and statistical definitions'],
        'limit': 'No claim that local numerical execution or tokenizer wrapper is bit-identical to retired NVIDIA service.'}
    target.mkdir(parents=True)
    write_json(target / 'config.json', config)
    write_json(target / 'pilot_units.json', read_json(parent_dir / 'pilot_units.json'))
    write_json(target / 'parent_start_record.json', read_json(root / 'outputs/table3_under10/run_v1/start_record.json'))
    files = dict(parent['files'])
    extra = [parent_dir / 'manifest.json', target / 'config.json', target / 'pilot_units.json', target / 'parent_start_record.json',
             asset_path, asset_path.parent / 'generation_inputs.jsonl', root / 'scripts/24_local_llama_server.py',
             root / 'scripts/25_run_table3_local.py', root / 'tests/test_table3_local.py', root / 'docs/TABLE3_LOCAL_AMENDMENT_CN.md']
    files.update({str(p.relative_to(root)): sha256_file(p) for p in extra})
    manifest = {'status': 'FROZEN_LOCAL_BEFORE_INFERENCE', 'frozen_at': utc_now(), 'parent_freeze_sha256': parent['freeze_sha256'],
                'files': files, 'packages': parent['packages'], 'asset_sha256': assets['asset_sha256'], 'budget_usd': 9,
                'parent_total_usd': 0, 'planned_states': 204, 'arms': 6, 'authorization': config['amendment']['authorization']}
    manifest['freeze_sha256'] = sha256_text(canonical_json(manifest))
    write_json(target / 'manifest.json', manifest)
    print(canonical_json({'freeze_sha256': manifest['freeze_sha256'], 'files': len(files), 'parent_verified': True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'verify', 'run', 'status', 'export'])
    parser.add_argument('--freeze', type=Path, default=Path('outputs/table3_under10/frozen_local_v2'))
    parser.add_argument('--run-dir', type=Path, default=Path('outputs/table3_under10/run_local_v2'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    frozen, run_dir = root / args.freeze, root / args.run_dir
    if args.command == 'freeze':
        freeze_local(root, frozen)
        return
    manifest = verify_freeze(root, frozen)
    if args.command == 'verify':
        print(canonical_json({'status': 'PASS', 'freeze_sha256': manifest['freeze_sha256']}))
        return
    if args.command == 'status':
        print(canonical_json({'status': read_json(run_dir / 'status.json') if (run_dir / 'status.json').exists() else 'NOT_STARTED',
                              'budget': Ledger(run_dir / 'ledger.sqlite').summary()}))
        return
    runner = LocalRunner(root, frozen, run_dir)
    try:
        if args.command == 'run':
            runner.run()
        else:
            with runner.lock():
                runner.export()
    except (StopRun, httpx.HTTPError, subprocess.CalledProcessError) as exc:
        reason = str(exc) if isinstance(exc, StopRun) else type(exc).__name__
        runner.status('STOPPED', reason=reason)
        raise StopRun(reason) from None
    finally:
        runner.close()


if __name__ == '__main__':
    try:
        main()
    except StopRun as exc:
        print('STOPPED: ' + str(exc), file=sys.stderr)
        sys.exit(2)
