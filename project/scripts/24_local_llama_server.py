"""Loopback-only, frozen BF16 Llama 3.1 8B server for Table III generation."""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.metadata
import json
from pathlib import Path
import time


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def prompt_text(messages):
    """Independent reference for the published mirror's plain-text chat template."""
    if not messages or any(m['role'] not in ('system', 'user', 'assistant') or not isinstance(m['content'], str) for m in messages):
        raise ValueError('Only nonempty plain-text conversations are supported')
    return '<|begin_of_text|>' + ''.join(
        '<|start_header_id|>' + m['role'] + '<|end_header_id|>\n\n' + m['content'].strip() + '<|eot_id|>'
        for m in messages) + '<|start_header_id|>assistant<|end_header_id|>\n\n'


def prepare(model_dir, arms_path, metadata_path, output):
    from transformers import AutoTokenizer
    model_dir, output = Path(model_dir), Path(output)
    if output.exists():
        raise ValueError('Asset package already exists; do not overwrite it')
    sources = json.loads(Path(metadata_path).read_text())
    official, mirror = sources
    files = {p.name: digest(p) for p in model_dir.iterdir() if p.is_file() and p.suffix in ('.json', '.safetensors')}
    weights = [n for n in official['files'] if n.endswith('.safetensors')]
    for name in weights:
        if files.get(name) != official['files'][name] or files[name] != mirror['files'][name]:
            raise ValueError('Downloaded weight mismatch: ' + name)
    for name, expected in mirror['files'].items():
        if not name.endswith('.safetensors'):
            data = (model_dir / name).read_bytes()
            actual = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
            if actual != expected:
                raise ValueError('Downloaded tokenizer/config differs from pinned source: ' + name)
    if json.loads((model_dir / 'generation_config.json').read_text())['eos_token_id'] != [128001, 128008, 128009]:
        raise ValueError('Official generation EOS configuration differs')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    records = []
    for line in Path(arms_path).read_text().splitlines():
        arm = json.loads(line)
        rendered = tokenizer.apply_chat_template(arm['messages'], tokenize=False, add_generation_prompt=True)
        if rendered != prompt_text(arm['messages']):
            raise ValueError('Chat template has unexpected text or framing')
        ids = tokenizer.apply_chat_template(arm['messages'], tokenize=True, add_generation_prompt=True)
        # Transformers 5 may return a BatchEncoding; request IDs explicitly.
        if not isinstance(ids, list):
            ids = ids['input_ids']
        independent = tokenizer.encode(rendered, add_special_tokens=False)
        if ids != independent or ids[0] != 128000 or ids.count(128000) != 1:
            raise ValueError('Tokenization mismatch or duplicate BOS')
        records.append({'call_id': arm['call_id'], 'prompt_messages_sha256': hashlib.sha256(canonical(arm['messages']).encode()).hexdigest(),
                        'rendered_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
                        'input_ids_sha256': hashlib.sha256(canonical(ids).encode()).hexdigest(),
                        'input_tokens': len(ids), 'seed': arm['generation_seed']})
    if len(records) != 1224:
        raise ValueError('Expected 1,224 frozen generation prompts')
    packages = {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'tokenizers', 'safetensors', 'huggingface-hub', 'accelerate', 'jinja2')}
    output.mkdir(parents=True)
    (output / 'generation_inputs.jsonl').write_text(''.join(canonical(r) + '\n' for r in records))
    manifest = {'model_id': 'meta/llama-3.1-8b-instruct', 'model_directory': str(model_dir.resolve()),
                'source_repository': mirror['repo'], 'source_revision': mirror['revision'],
                'official_repository': official['repo'], 'official_revision': official['revision'],
                'weight_hashes_match_official': True, 'model_files_sha256': files,
                'generation_inputs_sha256': digest(output / 'generation_inputs.jsonl'),
                'server_code_sha256': digest(__file__), 'packages': packages,
                'dtype': 'bfloat16', 'quantization': None, 'attention': 'sdpa', 'decoding': 'greedy',
                'max_new_tokens': 100, 'eos_token_ids': [128001, 128008, 128009],
                'template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                'template_scope': 'Pinned mirror plain-text template: trim message edges, one BOS, per-message EOT, trailing assistant header. No date, tools or added system text.',
                'template_validation': {'all_generation_prompts_checked': 1224, 'max_input_tokens': max(r['input_tokens'] for r in records)},
                'provenance_limit': 'Official current tokenizer_config differs and is gated. Published mirror tokenizer/template is pinned; framing cross-checked against public Meta llama3 source. Historical NVIDIA renderer and numeric backend cannot be verified.',
                'source_format_url': 'https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3/chat_format.py'}
    manifest['asset_sha256'] = hashlib.sha256(canonical(manifest).encode()).hexdigest()
    (output / 'model_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(canonical({'asset_sha256': manifest['asset_sha256'], 'weights_verified': len(weights), 'template_validation': manifest['template_validation']}), flush=True)


def serve(asset_dir, port):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    asset_dir = Path(asset_dir)
    manifest = json.loads((asset_dir / 'model_manifest.json').read_text())
    unsigned = {k: v for k, v in manifest.items() if k != 'asset_sha256'}
    if hashlib.sha256(canonical(unsigned).encode()).hexdigest() != manifest['asset_sha256']:
        raise ValueError('Model asset manifest changed')
    if digest(__file__) != manifest['server_code_sha256']:
        raise ValueError('Local server code changed')
    for name, version in manifest['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('Local generation dependency changed: ' + name)
    model_dir = Path(manifest['model_directory'])
    for name, sha in manifest['model_files_sha256'].items():
        if digest(model_dir / name) != sha:
            raise ValueError('Model asset changed: ' + name)
    if digest(asset_dir / 'generation_inputs.jsonl') != manifest['generation_inputs_sha256']:
        raise ValueError('Generation input token manifest changed')
    allowed = {(r['prompt_messages_sha256'], r['seed']): r for r in
               (json.loads(line) for line in (asset_dir / 'generation_inputs.jsonl').read_text().splitlines())}
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.bfloat16,
                                                device_map={'': 0}, attn_implementation='sdpa').eval()
    generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=100,
                                  bos_token_id=128000, eos_token_id=manifest['eos_token_ids'], pad_token_id=128001,
                                  repetition_penalty=1.0, use_cache=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, status, value):
            data = canonical(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/health':
                self.send(200, {'ready': True, 'model': manifest['model_id'], 'asset_sha256': manifest['asset_sha256'],
                                'gpu': torch.cuda.get_device_name(0), 'dtype': str(model.dtype)})
            else:
                self.send(404, {'error': 'Unknown path'})

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                self.send(404, {'error': 'Unknown path'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1_000_000:
                    raise ValueError('Invalid request length')
                body = json.loads(self.rfile.read(length))
                if body['model'] != manifest['model_id'] or body['temperature'] != 0 or body['max_tokens'] != 100:
                    raise ValueError('Request differs from frozen generation settings')
                messages, seed = body['messages'], body['seed']
                message_hash = hashlib.sha256(canonical(messages).encode()).hexdigest()
                record = allowed.get((message_hash, seed))
                if record is None:
                    raise ValueError('Request is not in the frozen generation input list')
                rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if hashlib.sha256(rendered.encode()).hexdigest() != record['rendered_sha256']:
                    raise ValueError('Rendered prompt changed')
                inputs = tokenizer(rendered, add_special_tokens=False, return_tensors='pt').to(model.device)
                ids = inputs['input_ids'][0].tolist()
                if hashlib.sha256(canonical(ids).encode()).hexdigest() != record['input_ids_sha256']:
                    raise ValueError('Prompt token IDs changed')
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                started = time.monotonic()
                with torch.inference_mode():
                    output = model.generate(**inputs, generation_config=generation)
                new = output[0, len(ids):].tolist()
                text = tokenizer.decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                stop = bool(new and new[-1] in manifest['eos_token_ids'])
                self.send(200, {'id': 'local-' + hashlib.sha256((message_hash + str(seed)).encode()).hexdigest()[:24],
                    'object': 'chat.completion', 'created': int(time.time()), 'model': manifest['model_id'],
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop' if stop else 'length'}],
                    'usage': {'prompt_tokens': len(ids), 'completion_tokens': len(new), 'total_tokens': len(ids) + len(new)},
                    'local_backend': {'asset_sha256': manifest['asset_sha256'], 'input_ids_sha256': record['input_ids_sha256'],
                                      'completion_token_ids': new, 'latency_seconds': time.monotonic() - started}})
            except (ValueError, KeyError, TypeError) as exc:
                self.send(400, {'error': {'type': type(exc).__name__, 'message': str(exc)}})
            except Exception as exc:
                self.send(500, {'error': {'type': type(exc).__name__, 'message': 'Local inference failed; see server diagnostics'}})
                print(canonical({'error_type': type(exc).__name__}), flush=True)

    print(canonical({'status': 'READY', 'port': port, 'asset_sha256': manifest['asset_sha256'], 'gpu': torch.cuda.get_device_name(0)}), flush=True)
    HTTPServer(('127.0.0.1', port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'serve'])
    parser.add_argument('--model-dir', default='/home/tokkio/models/table3-llama31-8b-d10aef7')
    parser.add_argument('--arms', default='outputs/table3_rerun_v1/plan/arms.jsonl')
    parser.add_argument('--sources', default='/home/tokkio/audits/pm-v1-table3-20260910/local_recovery_sources.json')
    parser.add_argument('--assets', default='outputs/table3_under10/local_model_v2')
    parser.add_argument('--port', type=int, default=18081)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.model_dir, args.arms, args.sources, args.assets)
    else:
        serve(args.assets, args.port)


if __name__ == '__main__':
    main()
