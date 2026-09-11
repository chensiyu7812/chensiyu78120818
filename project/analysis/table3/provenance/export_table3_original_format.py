"""Export existing v7 scores in the original audit-table layout; no API calls."""
import csv
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path('/home/tokkio/chensiyu78120818-table3/project')
RUN = ROOT / 'outputs/table3_under10/run_identity_v7'
manifest = json.loads((RUN / 'result_manifest.json').read_text())
for name, sha in manifest['hashes'].items():
    assert hashlib.sha256((RUN / name).read_bytes()).hexdigest() == sha, name
jobs = {j['id']: j for j in map(json.loads, (RUN / 'main.jobs.jsonl').open())}
db = sqlite3.connect(f'file:{RUN}/ledger.sqlite?mode=ro', uri=True)
db.row_factory = sqlite3.Row
risks, inputs = defaultdict(dict), defaultdict(dict)
for attempt in db.execute("SELECT * FROM attempts WHERE status='done'"):
    job = jobs.get(attempt['job_id'])
    if job and job['stage'] == 'risk':
        assert job['unit_id'] not in risks[job['arm']]
        risks[job['arm']][job['unit_id']] = json.loads(attempt['parsed_json'])
    if attempt['phase'] == 'generation':
        job = json.loads(attempt['job_json'])
        response = json.loads(attempt['response_json'])
        assert job['unit_id'] not in inputs[job['arm']]
        inputs[job['arm']][job['unit_id']] = response['usage']['prompt_tokens']
db.close()

labels = {'pm': 'Learned PM', 'rule': 'Rule Source Selector',
          'fixed': 'Structured Fixed'}
table = []
for original in csv.DictReader((RUN / 'table3.csv').open()):
    arm = original['arm']
    policy, flag = arm.split('_')
    scores = list(risks[arm].values())
    assert len(scores) == int(original['n']) == 204
    assert set(risks[arm]) == set(inputs[arm])
    severity = mean(r['selected_evidence_misuse'] for r in scores)
    assert abs(severity - float(original['misuse_severity'])) < 1e-10
    row = {
        'arm': arm, 'method': labels[policy], 'filter': flag.upper(), 'n': len(scores),
        'source_set_fit': mean(r['source_set_appropriateness'] for r in scores),
        'context_misuse_severity': severity,
        'sufficiency_selected_context': mean(r['response_support_sufficiency'] for r in scores),
        'major_issues_count': sum(r['verdict'] == 'major_issue' for r in scores),
        'overall': float(original['overall']),
        'emotional_support': float(original['emotional_support']),
        'source_invocations': float(original['source_invocations']),
        'candidate_memory_tokens_est': float(original['candidate_memory_tokens_est']),
        'kept_memory_tokens_est': float(original['memory_tokens_est']),
        'generator_input_tokens': mean(inputs[arm].values()),
        'end_to_end_latency_ms': None,
    }
    assert row['major_issues_count'] == round(float(original['major_issue_rate']) * len(scores))
    table.append(row)

audit_table = [
    '| Method | Filter | n | Source-set fit ↑ | Context misuse ↓ | Sufficiency ↑ | Major issues ↓ |',
    '|---|---|---:|---:|---:|---:|---:|',
]
resource_table = [
    '| Method | Filter | Quality (Overall) ↑ | Retrieval calls ↓ | Memory tokens candidate → kept ↓ | Generator input tokens ↓ |',
    '|---|---|---:|---:|---:|---:|',
]
for r in table:
    audit_table.append(f"| {r['method']} | {r['filter']} | {r['n']} | {r['source_set_fit']:.3f} | {r['context_misuse_severity']:.3f} | {r['sufficiency_selected_context']:.3f} | {r['major_issues_count']} |")
    resource_table.append(f"| {r['method']} | {r['filter']} | {r['overall']:.3f} | {r['source_invocations']:.3f} | {r['candidate_memory_tokens_est']:.1f} → {r['kept_memory_tokens_est']:.1f} | {r['generator_input_tokens']:.1f} |")

text = '\n'.join([
    '# Table III：按原表列重排本次六组结果', '',
    '沿用原 Source-set fit / Context misuse / Sufficiency / Major issues 列，六组使用本次相同的 204 个固定状态。主 judge 为 GPT-4.1-mini-2025-04-14；没有混入历史表格或 GPT-4o pilot 的分数。', '',
    *audit_table, '',
    'Source-set fit 与 Sufficiency 为 selected-context risk 审计的 1–5 分；Context misuse 为 0–3 严重度均值，不是百分比。Major issues 为 verdict=major_issue 的样本数。Sufficiency 沿用原审计列含义，取 risk 阶段 response_support_sufficiency；独立 omission 阶段同名字段是另一上下文下的判断，没有混用。', '',
    '## 质量与资源补充', '',
    *resource_table, '',
    'Quality 为本次预先固定的 overall（1–5），与 emotional_support 是不同字段。Retrieval calls 为实际请求的逻辑来源数（含 Strategy），不是 HTTP 请求数。Memory tokens 是 v1 chars/4 估计；Generator input 为实际生成记录中 Llama tokenizer 的完整输入 token 数，二者口径不同。自然结束补生成沿用完全相同输入，所以不改变输入 token 数。', '',
    '当前可比较的主张是：在相同 Filter 设置下，PM 与 Rule／Fixed 的质量均值接近，同时使用更少的检索来源和生成输入。已有 Overall 配对区间包含 0，支持如实报告差异与不确定性，尚不构成正式等效证明。', '',
    '当前记录没有完整、统一口径的端到端延迟测量。生成计时未包含实际在线 PM、检索与 Filter 全流程，而且补生成与原生成为不同执行批次；因此端到端延迟留空，不能用逻辑来源数或 token 减少比例代替时间减少比例，也不将旧 NVIDIA 延迟并入本次本地重跑。', '',
    '本表保留已授权的 fixed_off 单条 risk 辅助 omission 校验例外；样本和最早原始评分均保留，详见[最终核验](FINAL_AUDIT_CN.md)。', '',
    '[本格式 CSV](table3_original_format.csv) · [完整原始导出](table3.csv) · [配对置信区间](paired_intervals.json)', '',
])
(RUN / 'TABLE3_ORIGINAL_FORMAT_CN.md').write_text(text)
with (RUN / 'table3_original_format.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)
receipt = {
    'freeze_sha256': manifest['freeze_sha256'],
    'source_sha256': {name: hashlib.sha256((RUN / name).read_bytes()).hexdigest()
                      for name in ('main.jobs.jsonl', 'ledger.sqlite', 'table3.csv', 'result_manifest.json')},
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'output_sha256': {name: hashlib.sha256((RUN / name).read_bytes()).hexdigest()
                      for name in ('TABLE3_ORIGINAL_FORMAT_CN.md', 'table3_original_format.csv')},
    'scope': 'Reformat existing scores; no new inference, no sample exclusions, no changes to frozen outputs.',
}
(RUN / 'table3_original_format_manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(text)
indexed = {r['arm']: r for r in table}
for flag in ('off', 'on'):
    pm = indexed['pm_' + flag]
    for baseline in ('rule', 'fixed'):
        other = indexed[baseline + '_' + flag]
        print('Comparison', flag, baseline, 'quality_delta', pm['overall'] - other['overall'],
              'input_saving_percent', 100 * (1 - pm['generator_input_tokens'] / other['generator_input_tokens']),
              'source_saving_percent', 100 * (1 - pm['source_invocations'] / other['source_invocations']))
