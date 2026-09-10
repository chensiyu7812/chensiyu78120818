"""Complete every 100-token cap hit with the same greedy model, until EOS."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import time

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'outputs/table3_under10/frozen_eos_v3'
RUN = ROOT / 'outputs/table3_under10/run_eos_v3'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def h(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    freeze = json.loads((FROZEN / 'manifest.json').read_text())
    assert h({k: v for k, v in freeze.items() if k != 'freeze_sha256'}) == freeze['freeze_sha256']
    for path, expected in freeze['files'].items():
        assert sha(ROOT / path) == expected, path
    config = json.loads((FROZEN / 'config.json').read_text())
    asset_path = ROOT / config['local_generator']['model_manifest']
    asset = json.loads(asset_path.read_text())
    for package, version in asset['packages'].items():
        assert importlib.metadata.version(package) == version, package
    model_dir = Path(asset['model_directory'])
    for path, expected in asset['model_files_sha256'].items():
        assert sha(model_dir / path) == expected, path
    index = {r['call_id']: r for r in map(json.loads, (asset_path.parent / 'generation_inputs.jsonl').read_text().splitlines())}
    parent_db = ROOT / config['eos_amendment']['parent_ledger']
    with sqlite3.connect(f'file:{parent_db}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = list(db.execute("SELECT * FROM attempts WHERE phase='generation' AND status='done' ORDER BY created_at,id"))
    assert len(rows) == 1224
    selected = []
    for row in rows:
        job, response = json.loads(row['job_json']), json.loads(row['response_json'])
        if response['choices'][0]['finish_reason'] == 'length':
            selected.append((job, response))
    assert len(selected) == config['eos_amendment']['cap_hit_count'] == 28
    destination = RUN / 'extended_replies'
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.get_device_name(0) == 'NVIDIA RTX A6000'
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.bfloat16,
                                                device_map={'': 0}, attn_implementation='sdpa').eval()
    for i, (job, original) in enumerate(selected):
        target = destination / (job['id'] + '.json')
        if target.exists():
            old = json.loads(target.read_text())
            assert old['freeze_sha256'] == freeze['freeze_sha256']
            assert old['original_token_ids'] == original['local_backend']['completion_token_ids']
            continue
        record = index[job['unit_id'] + '::' + job['arm']]
        rendered = tokenizer.apply_chat_template(job['body']['messages'], tokenize=False, add_generation_prompt=True)
        assert hashlib.sha256(rendered.encode()).hexdigest() == record['rendered_sha256']
        inputs = tokenizer(rendered, add_special_tokens=False, return_tensors='pt').to(model.device)
        ids = inputs['input_ids'][0].tolist()
        assert h(ids) == record['input_ids_sha256']
        # No experiment-level length cutoff. The model's finite context is the
        # technical ceiling; hitting it without EOS is an error, never accepted.
        remaining_context = model.config.max_position_embeddings - len(ids)
        generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=remaining_context,
                                      bos_token_id=128000, eos_token_id=asset['eos_token_ids'], pad_token_id=128001,
                                      repetition_penalty=1.0, use_cache=True)
        torch.manual_seed(job['body']['seed'])
        torch.cuda.manual_seed_all(job['body']['seed'])
        started = time.monotonic()
        with torch.inference_mode():
            output = model.generate(**inputs, generation_config=generation)
        new = output[0, len(ids):].tolist()
        old = original['local_backend']['completion_token_ids']
        text = tokenizer.decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        result = {'freeze_sha256': freeze['freeze_sha256'], 'unit_id': job['unit_id'], 'arm': job['arm'],
                  'original_job_id': job['id'], 'original_token_ids': old, 'completion_token_ids': new,
                  'input_ids_sha256': h(ids), 'asset_sha256': asset['asset_sha256'],
                  'seed': job['body']['seed'], 'text': text, 'completion_tokens': len(new),
                  'prefix_matches': new[:len(old)] == old, 'natural_stop': bool(new and new[-1] in asset['eos_token_ids']),
                  'context_ceiling_tokens': model.config.max_position_embeddings,
                  'latency_seconds': time.monotonic() - started, 'api_usd': 0}
        if not result['prefix_matches'] or not result['natural_stop'] or not text:
            (destination / (job['id'] + '.failure.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2))
            raise RuntimeError('Prefix mismatch, empty response, or no EOS at context ceiling; no scoring permitted')
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        temporary.replace(target)
        print(canonical({'stage': 'EOS_GENERATION', 'completed': i + 1, 'total': len(selected),
                         'tokens': len(new), 'prefix_matches': True, 'natural_stop': True}), flush=True)


if __name__ == '__main__':
    main()
