"""Validated Chinese Table III analysis for the user-authorized EOS amendment."""
from collections import Counter
import csv
import json
import statistics

from .io import read_json, sha256_file, sha256_text, canonical_json, utc_now, write_json
from .table3_execution import ARMS, StopRun, verify_freeze

LABELS = {'pm_off': 'PM／OFF', 'pm_on': 'PM／ON', 'rule_off': 'Rule／OFF',
          'rule_on': 'Rule／ON', 'fixed_off': 'Fixed／OFF', 'fixed_on': 'Fixed／ON'}


def analyze(runner):
    verify_freeze(runner.root, runner.frozen)
    path = runner.run_dir
    manifest = read_json(path / 'result_manifest.json')
    for name, digest in manifest['hashes'].items():
        if sha256_file(path / name) != digest:
            raise StopRun('Exported result hash changed: ' + name)
    observations = runner.observations('main')
    if len(observations) != 3672:
        raise StopRun('Active main observations are incomplete')
    jobs = runner.jobs('main', runner.config['judge'], pilot=False, batch=True)
    if len(jobs) != 2652 or Counter(j['stage'] for j in jobs) != {'quality': 204, 'risk': 1224, 'omission': 1224}:
        raise StopRun('Active main job counts differ')
    for job in jobs:
        row = runner.ledger.successful(job['id'])
        if not row or row['request_hash'] != sha256_text(canonical_json(job['body'])):
            raise StopRun('Active scoring request missing or changed')
        if json.loads(row['response_json'])['model'] != 'gpt-4.1-mini-2025-04-14':
            raise StopRun('Mixed main judge models')
    with (path / 'table3.csv').open() as stream:
        table = {r['arm']: r for r in csv.DictReader(stream)}
    rows = [json.loads(line) for line in (path / 'observations.jsonl').read_text().splitlines()]
    expected = {(u['unit_id'], a) for u in runner.units for a in ARMS}
    if len(rows) != 1224 or {(r['unit_id'], r['arm']) for r in rows} != expected or set(table) != set(ARMS):
        raise StopRun('Final paired sample differs from all 204 states and six arms')
    for arm in ARMS:
        values = [r for r in rows if r['arm'] == arm]
        if len(values) != int(table[arm]['n']) or len(values) != 204:
            raise StopRun('Final arm sample size differs')
        for metric, value in table[arm].items():
            if metric not in ('arm', 'n') and abs(float(value) - statistics.mean(r[metric] for r in values)) > 1e-10:
                raise StopRun('Table mean differs from exported observations')
    intervals = read_json(path / 'paired_intervals.json')
    if len(intervals) != 14 * (len(table[ARMS[0]]) - 2):
        raise StopRun('Planned interval count differs')
    for item in intervals:
        if item['clusters'] != (18 if item['cluster'] == 'user_id' else 34):
            raise StopRun('Cluster count differs')
        delta = float(table[item['left']][item['metric']]) - float(table[item['right']][item['metric']])
        if abs(delta - item['delta_left_minus_right']) > 1e-10:
            raise StopRun('Paired point estimate differs from table')
    budget = runner.ledger.summary()
    if budget['charged_plus_reserved_usd'] > 9:
        raise StopRun('Cumulative budget exceeded')
    if any(r['status'] in ('unknown', 'reserved', 'in_flight', 'queued') for r in runner.ledger.rows()):
        raise StopRun('Unresolved accounting at final export')
    pilot_prevalence = {}
    for phase in ('pilot_mini', 'pilot_reference'):
        pilot = runner.observations(phase)
        pilot_prevalence[phase] = {
            'clear_misuse': sum(v['selected_evidence_misuse'] >= 2 for k, v in pilot.items() if k[2] == 'risk'),
            'clear_omission': sum(v['omission_severity'] >= 2 for k, v in pilot.items() if k[2] == 'omission')}
    eos = read_json(path / 'eos_generation_summary.json')
    qa = {'status': 'PASS', 'timestamp': utc_now(), 'freeze_sha256': runner.manifest['freeze_sha256'],
          'active_main_calls': 2652, 'active_main_observations': len(observations), 'all_arms_n': 204,
          'generated_to_natural_stop': 1224, 'eos_amendment': eos, 'pilot_prevalence': pilot_prevalence,
          'budget_including_parent': budget, 'parent_snapshot_sha256': sha256_file(runner.root / runner.config['eos_amendment']['parent_ledger']),
          'source_result_manifest_sha256': sha256_file(path / 'result_manifest.json')}
    write_json(path / 'completion_qa.json', qa)
    primary = [r for r in intervals if r['cluster'] == 'user_id' and r['metric'] in ('overall', 'misuse_clear', 'omission_clear')]
    pairs = list(dict.fromkeys((r['left'], r['right']) for r in primary))
    lines = ['# 表 III：自然结束版本的重跑与分析', '',
             f"204 个固定状态 × 6 组已完成，全部 1,224 条回复自然结束。取消原 100-token 限制后补生成 28 条，其中 {eos['text_changed']} 条文本改变；其余 1,196 条复用。28 条的前 100 个 token 均与原回复一致。", '',
             f"累计 API 费用按未缓存单价计算的上界为 **${budget['charged_upper_usd']:.4f}**，包含原 pilot、旧批次及本次更新，实际账单可能因缓存折扣更低。本地生成无 API 费用。", '',
             '| 组别 | Overall | 明确误用率 | 明确遗漏率 | 记忆 tokens 估计 | 来源调用数 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for a in ARMS:
        r = table[a]
        lines.append(f"| {LABELS[a]} | {float(r['overall']):.4f} | {100*float(r['misuse_clear']):.2f}% | {100*float(r['omission_clear']):.2f}% | {float(r['memory_tokens_est']):.2f} | {float(r['source_invocations']):.2f} |")
    lines += ['', '每组 n=204；明确误用和明确遗漏均采用预先冻结的 severity ≥2 阈值。severity ≥1 误用、严重度均值及全部质量分项保存在 CSV。', '',
              '## 配对比较', '',
              '差值为左组减右组；Overall 越高越好，误用／遗漏越低越好。区间按用户聚类，状态等权，18 个用户、10,000 次 bootstrap；场景聚类敏感性分析见 paired_intervals.json。', '',
              '| 左组 − 右组 | Overall 差值 [95% CI] | 明确误用差 [百分点，95% CI] | 明确遗漏差 [百分点，95% CI] |',
              '| --- | ---: | ---: | ---: |']
    for left, right in pairs:
        cells = []
        for metric in ('overall', 'misuse_clear', 'omission_clear'):
            r = next(r for r in primary if (r['left'], r['right'], r['metric']) == (left, right, metric))
            scale = 1 if metric == 'overall' else 100
            cells.append(f"{scale*r['delta_left_minus_right']:.4f} [{scale*r['lower']:.4f}, {scale*r['upper']:.4f}]")
        lines.append('| ' + LABELS[left] + ' − ' + LABELS[right] + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '这些是预先规定的描述性区间，未做多重比较的同时覆盖校正。包含 0 的区间不能据此认定差异方向稳定。', '', '## 数据支持什么', '']
    for metric, label in [('overall', '质量'), ('misuse_clear', '明确误用率'), ('omission_clear', '明确遗漏率')]:
        r = next(r for r in primary if (r['left'], r['right'], r['metric']) == ('pm_on', 'pm_off', metric))
        direction = '区间完全高于 0' if r['lower'] > 0 else ('区间完全低于 0' if r['upper'] < 0 else '区间包含 0')
        lines.append(f"PM 内部打开 Filter 后，{label}的配对差异{direction}；具体幅度见上表。")
    for metric, label in [('misuse_clear', '明确误用'), ('omission_clear', '明确遗漏')]:
        if all(float(r[metric]) == 0 for r in table.values()):
            lines += ['', f'六组均未观察到{label}，这不能证明风险不存在或 Filter 能降低该风险。全零样本的 bootstrap 区间退化为 0，并不表示真实风险被精确估计为零。']
    for policy in ('pm', 'rule', 'fixed'):
        before = float(table[policy + '_off']['memory_tokens_est'])
        after = float(table[policy + '_on']['memory_tokens_est'])
        percentage = f'（减少 {100*(before-after)/before:.2f}%）' if before else ''
        lines += ['', f'{policy.upper()} 的平均记忆输入从 {before:.2f} 降至 {after:.2f} 个估计 tokens{percentage}。这是记忆负载指标；它本身不等于质量提升或风险改善。']
    lines += ['', 'PM 相对 Rule／Fixed 的贡献应结合上表所有预先规定的配对比较判断；不根据排名挑选条件或省略不利比较。', '',
              '## 解释范围', '',
              '主 judge 为 GPT-4.1-mini-2025-04-14；GPT-4o 只参与固定 pilot，未混入正式表格。pilot 门槛沿用原方案，所有受文本变化影响的校验均更新。模型评分不是人类金标准，pilot 通过不代表两种 judge 等价。', '']
    if all(v['clear_misuse'] == v['clear_omission'] == 0 for v in pilot_prevalence.values()):
        lines += ['两种 judge 的更新后 pilot 都没有明确误用／遗漏正例，100% 二元一致率不能验证识别正例的能力。', '']
    lines += ['沿用 v1 PM checkpoint、固定测试状态和六组设计。NVIDIA 退役后使用同权重的本地 A6000；本次按用户要求取消 100-token 截断，保留全部旧结果和费用。这是有明确修订记录的重跑，不是历史 Table III 数值的精确复现。', '',
              '[完整 CSV](table3.csv) · [配对置信区间](paired_intervals.json) · [完成检查](completion_qa.json) · [补生成记录](eos_generation_summary.json)', '',
              f"冻结哈希：`{runner.manifest['freeze_sha256']}`。", '']
    (path / 'TABLE3_CN.md').write_text('\n'.join(lines))
