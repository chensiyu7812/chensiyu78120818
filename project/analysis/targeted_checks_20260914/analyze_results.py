"""Summarize the predeclared diagnostic endpoints, with independent integrity checks."""
from collections import Counter,defaultdict
import csv
import numpy as np
from common import ROOT,CODE,OUT,read,rows,write,csv_write,sha,h,cluster_ci,verify

METRICS=['local_first_text_ms','local_complete_ms','local_first_token_ms','generator_first_token_ms','state_hydration_ms','policy_ms','retrieval_ms','prompt_build_ms','tokenization_ms','generation_setup_ms','generation_ms','final_decode_ms','input_tokens','output_tokens','source_invocations']

def verify_shuffle():
    with (OUT/'shuffle/per_card.csv').open() as f:cards=list(csv.DictReader(f))
    summary=read(OUT/'shuffle/summary.json')
    assert len(cards)==1728 and len({c['card_id'] for c in cards})==1728
    strata=defaultdict(list)
    for c in cards:strata[(int(c['fold']),c['legal_actions'])].append(c)
    # Independent check: map every donor action to each recipient's raw sweep input cost.
    costs={(r['card_id'],r['action_id']):r['cost']['total_input_tokens'] for r in rows(ROOT/'outputs/synthetic_sweep/action_outcomes.jsonl')}
    maximum_cost_error=0.0
    for group in strata.values():
        frequencies=Counter(c['chosen_action'] for c in group)
        for c in group:
            exp=sum(costs[(c['card_id'],a)]*n for a,n in frequencies.items())/len(group)
            maximum_cost_error=max(maximum_cost_error,abs(exp-float(c['shuffle_expectation_input_tokens'])))
    assert maximum_cost_error<1e-9
    for metric in summary['contrasts']:
        m=metric['metric'];p=sum(float(c[f'pm_{m}']) for c in cards)/len(cards)
        q=sum(float(c[f'shuffle_expectation_{m}']) for c in cards)/len(cards)
        assert abs(p-metric['pm_mean'])<1e-9 and abs(q-metric['shuffle_expected_mean'])<1e-9
    # Reconstruct 40 representative card scores independently by augmented least
    # squares, instead of the production helper's normal-equation solver.
    selected={group[0]['card_id']:group[0] for group in strata.values()}
    pairs=defaultdict(list)
    for p in rows(ROOT/'outputs/full_judging_gemini_flash_lite_v3/response_pair_judgments.jsonl'):
        if p['card_id'] in selected and p['training_eligible']:pairs[p['card_id']].append(p)
    maximum_score_error=0.0
    for card_id,c in selected.items():
        actions=c['legal_actions'].split('|');index={a:i for i,a in enumerate(actions)};pp=pairs[card_id]
        design=np.zeros((len(pp),len(actions)));target=[]
        for i,p in enumerate(pp):
            design[i,index[p['action_a']]]=1;design[i,index[p['action_b']]]=-1
            target.append({'A':1,'B':-1,'tie':0}[p['preference']])
        latent=np.linalg.lstsq(np.vstack([design,np.eye(len(actions))*np.sqrt(.10)]),np.r_[target,np.zeros(len(actions))],rcond=None)[0]
        i=index[c['chosen_action']]
        score=sum(1/(1+np.exp(-(latent[i]-latent[j]))) for j in range(len(actions)) if j!=i)/(len(actions)-1)
        maximum_score_error=max(maximum_score_error,abs(score-float(c['pm_response_preference'])))
    assert maximum_score_error<1e-10
    dev={r['card_id']:r for r in rows(ROOT/'outputs/reviewer_supplement_20260914/offline/development_selector_replay.jsonl')}
    folds=[f for f in rows(ROOT/'data/synthetic/folds.jsonl') if f['mode']=='user']
    for f in folds:
        held=set(f['validation_card_ids'])
        assert held=={c['card_id'] for c in cards if int(c['fold'])==f['fold']}
        for c in cards:
            if c['card_id'] in held:assert c['chosen_action']==dev[c['card_id']]['chosen_action']
    return dict(cards=len(cards),strata=len(strata),maximum_independent_cost_error=maximum_cost_error,independently_reconstructed_score_cards=len(selected),maximum_score_error=maximum_score_error,held_out_card_and_action_join='PASS')

def latency():
    config=read(OUT/'config.json');status=read(OUT/'latency/status.json')
    assert status['stage']=='COMPLETE'
    data=rows(OUT/'latency/requests.jsonl');schedule=rows(OUT/'schedule.jsonl')
    assert len(data)==len(schedule)==720
    sched={r['request_id']:r for r in schedule};prompts={(r['unit_id'],r['arm']):r for r in rows(OUT/'prompts.jsonl')}
    assert len({r['request_id'] for r in data})==720
    parts=['state_hydration_ms','policy_ms','retrieval_ms','prompt_build_ms','tokenization_ms','generation_setup_ms','generation_ms','final_decode_ms']
    refs=Counter();groups=defaultdict(list)
    for r in data:
        assert all(r[k]==v for k,v in sched[r['request_id']].items())
        pr=prompts[(r['unit_id'],r['arm'])]
        assert r['messages_sha256']==pr['messages_sha256'] and r['action']==pr['action']
        assert r['natural_eos'] and r['completion_token_ids'][-1] in [128001,128008,128009]
        assert r['output_tokens']==len(r['completion_token_ids']) and r['response_sha256']==h(r['response_text'])
        assert not r['warmup'] and r['api_usd']==0
        assert 0<r['local_first_token_ms']<=r['local_first_text_ms']<=r['local_complete_ms']
        assert abs(sum(r[k] for k in parts)-r['local_complete_ms'])<1e-6
        if pr['reference_response_sha256'] is not None:
            equal=r['response_sha256']==pr['reference_response_sha256']
            assert equal==r['reference_response_matches'];refs['matched' if equal else 'different']+=1
        groups[(r['unit_id'],r['arm'])].append(r)
    assert len(groups)==240 and all(len(v)==3 for v in groups.values())
    unit_rows=[];nonidentical=[]
    for (uid,arm),rr in sorted(groups.items()):
        assert sorted(r['repeat'] for r in rr)==[0,1,2]
        assert len({r['input_ids_sha256'] for r in rr})==1
        if len({r['response_sha256'] for r in rr})!=1:nonidentical.append(dict(unit_id=uid,arm=arm))
        unit_rows.append(dict(unit_id=uid,arm=arm,user_id=rr[0]['user_id'],scenario_id=rr[0]['scenario_id'],seed=rr[0]['seed'],turn_index=rr[0]['turn_index'],**{m:float(np.mean([r[m] for r in rr])) for m in METRICS}))
    means=[]
    for arm in config['arms']:
        rr=[r for r in data if r['arm']==arm]
        means.append(dict(arm=arm,n_states=48,n_requests=len(rr),**{m:float(np.mean([r[m] for r in rr])) for m in METRICS},first_text_p95_ms=float(np.quantile([r['local_first_text_ms'] for r in rr],.95)),complete_p95_ms=float(np.quantile([r['local_complete_ms'] for r in rr],.95))))
    unit_map={(r['unit_id'],r['arm']):r for r in unit_rows};ids=sorted({r['unit_id'] for r in unit_rows})
    contrasts=[]
    for arm in config['arms'][1:]:
        pm=np.array([[unit_map[(u,'pm_off')][m] for m in METRICS] for u in ids])
        other=np.array([[unit_map[(u,arm)][m] for m in METRICS] for u in ids])
        for cluster in ['user_id','scenario_id']:
            labels=[unit_map[(u,'pm_off')][cluster] for u in ids]
            ci=cluster_ci(pm-other,labels,20260918)
            for j,m in enumerate(METRICS):
                contrasts.append(dict(baseline=arm,metric=m,cluster=cluster,n_pairs=48,n_clusters=len(set(labels)),pm_mean=float(pm[:,j].mean()),baseline_mean=float(other[:,j].mean()),delta_pm_minus_baseline=float((pm-other)[:,j].mean()),ci_low=float(ci[0,j]),ci_high=float(ci[1,j]),reduction_percent=float((1-pm[:,j].mean()/other[:,j].mean())*100) if other[:,j].mean()!=0 else None))
    repeat_rows=[]
    for rep in range(3):
        for arm in config['arms']:
            rr=[r for r in data if r['arm']==arm and r['repeat']==rep]
            repeat_rows.append(dict(repeat=rep,arm=arm,n=len(rr),**{m:float(np.mean([r[m] for r in rr])) for m in ['local_first_text_ms','local_complete_ms','input_tokens','output_tokens']}))
    csv_write(OUT/'latency/per_request.csv',data)
    csv_write(OUT/'latency/per_state_means.csv',unit_rows)
    csv_write(OUT/'latency/arm_means.csv',means)
    csv_write(OUT/'latency/paired_intervals.csv',contrasts)
    csv_write(OUT/'latency/by_repeat.csv',repeat_rows)
    result=dict(status='COMPLETE_LOCAL_LATENCY_DIAGNOSTIC',n_requests=720,n_states=48,n_users=18,n_scenarios=config['scenarios'],means=means,contrasts=contrasts,reference_response_comparison=dict(refs),nonidentical_repeat_responses=nonidentical,api_usd=0,limits=['48-state post-hoc diagnostic; not the 204-state primary quality result.','Resident local BF16 model, prepared indices, single request, no external network/database/UI timing.','Same prepared strategy feature optimization for all methods; cannot mix timing with historical unoptimized retrieval logs.','Request repeats are averaged per state before paired cluster intervals.','P95 is descriptive over 144 requests per method.','Context Only/ME+R0 responses have no new quality judgments.'])
    write(OUT/'latency/summary.json',result)
    return result

def main():
    manifest=verify();check=verify_shuffle();lat=latency()
    report=dict(status='PASS',freeze_sha256=manifest['freeze_sha256'],shuffle=check,latency_requests=lat['n_requests'],reference_responses=lat['reference_response_comparison'],nonidentical_repeat_responses=lat['nonidentical_repeat_responses'],api_usd=0)
    write(OUT/'verification.json',report)
    render_report(lat)
    write(OUT/'analysis_manifest.json',dict(analysis_code_sha256=sha(__file__),files={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['analysis_manifest.json','run.lock']}))
    print(report)

def render_report(lat):
    shuffle=read(OUT/'shuffle/summary.json')
    means={r['arm']:r for r in lat['means']}
    labels={'pm_off':'PM OFF','rule_off':'Rule OFF','fixed_off':'Fixed OFF','context_only':'Context Only','me_r0':'ME+R0'}
    lines=['# PM v1 定向补充结果','',
           '日期：2026-09-14。本轮两项补充已完成并通过离线校验，新增 API 费用 **$0**。未修改 PM、阈值、原主表或此前补充。',
           '', '[执行前协议](../../analysis/targeted_checks_20260914/PROTOCOL_CN.md)；执行 freeze：`'+read(OUT/'freeze.json')['freeze_sha256']+'`。',
           '', '## 1. 动作置换：受控开发中存在状态匹配收益','',
           '使用 1,728 cards、16 合成用户、5 个 user-held-out folds。按 fold 和合法动作集合分为 40 层，在层内置换 PM 动作，精确保留来源/RS 组合频率，用已有逐动作 judge 标签及 sweep 输入记录评价。PM 自身预测分未用于效果评分。',
           '', '| 指标 | PM | 置换期望 | PM−置换 | 95% 用户聚类 CI |','|---|---:|---:|---:|---:|']
    for r in shuffle['contrasts']:
        if r['metric']=='source_invocations':ci='总体频率由设计精确保持'
        else:ci=f"[{r['user_ci_low']:+.5f}, {r['user_ci_high']:+.5f}]"
        lines.append(f"| {r['metric']} | {r['pm_mean']:.5f} | {r['shuffle_expected_mean']:.5f} | {r['delta_pm_minus_expected']:+.5f} | {ci} |")
    lines+=['','回复偏好增益 +0.05631，五个 fold 均为正。两者平均输入约 555.30 / 555.98 tokens。该结果支持：在这些合成开发状态上，状态与动作的对应关系提供了额外偏好收益，不能仅由平均资源调用频率解释。它没有显示所有风险同时改善。',
            '', '**范围限制：** OOF 模型应用最终 consensus 阈值，不是新的独立或 nested validation；合成用户留出不等于语义留出。用户 CI 条件于此数据估得的动作频率。开发 response-preference 为 [0,1] 重构量纲，不等于外部 Support，也不推翻既有外部 Context Only / ME+R0 结果。输入预算近似相同，非逐状态精确匹配。',
            '', '[逐 card](shuffle/per_card.csv)、[差值与 CI](shuffle/contrasts.csv)、[固定参照](shuffle/policy_means.csv)、[各 fold](shuffle/by_fold.csv)、[各库存](shuffle/by_inventory.csv)、[独立验证](shuffle/verification.json)。',
            '', '## 2. 本地时延试跑：全部 720 次完成','',
            '48 状态、18 用户、34 scenarios，五组各重复 3 次。NVIDIA RTX A6000，冻结 Llama 3.1 8B BF16 / SDPA / greedy，batch size=1，随机交错顺序。全部回复自然 EOS，无实验输出 token 上限。',
            '', '| 方法 | 首个可见文本 ms | 完整回复 ms | 首文本 P95 ms | 完整回复 P95 ms | 输入 tokens | 输出 tokens |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for a in labels:
        r=means[a];lines.append(f"| {labels[a]} | {r['local_first_text_ms']:.2f} | {r['local_complete_ms']:.2f} | {r['first_text_p95_ms']:.2f} | {r['complete_p95_ms']:.2f} | {r['input_tokens']:.2f} | {r['output_tokens']:.2f} |")
    lines+=['','先在状态内平均三次，再按用户聚类 10,000 次。以下为 PM−baseline；负数代表 PM 较快。所有区间都是探索性逐项区间，没有多重比较联合覆盖保证。',
            '', '| 比较 | 指标 | 差值 ms | 95% 用户 CI | 均值减少 |','|---|---|---:|---:|---:|']
    for r in lat['contrasts']:
        if r['cluster']!='user_id' or r['metric'] not in ['local_first_text_ms','local_complete_ms']:continue
        metric='首个可见文本' if r['metric']=='local_first_text_ms' else '完整回复'
        lines.append(f"| PM−{labels[r['baseline']]} | {metric} | {r['delta_pm_minus_baseline']:+.2f} | [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] | {r['reduction_percent']:+.2f}% |")
    lines+=['','**计时边界：** 常驻模型和预构建 inventory/静态索引下，从 runtime state 反序列化到输出；包含策略选择、查询、实际检索、prompt 构造、tokenization、生成和解码。未包含远程网络、数据库或 UI，所以这是本地处理链路时延，不能直接写成线上用户端实测加速。',
            '', '各方法共享相同 PreparedStrategyRetriever 文档特征预计算；请求间清空查询缓存，只有同请求内可以复用。该共同实现与旧未优化检索日志不同，不跨运行拼接时延。P95 是各方法 144 次请求的描述性统计；三次重复不是新的独立用户样本。',
            '', f"原三组回复对照：{lat['reference_response_comparison']}；同状态同方法跨重复出现不同回复的组数：{len(lat['nonidentical_repeat_responses'])}。Context Only / ME+R0 仅新增时延和回复记录，没有新 judge 质量分。", 
            '', '[全部逐请求](latency/per_request.csv)、[状态内均值](latency/per_state_means.csv)、[均值与分项](latency/arm_means.csv)、[全部配对 CI](latency/paired_intervals.csv)、[各重复](latency/by_repeat.csv)、[完整汇总](latency/summary.json)。',
            '', '## 3. 对论文呈现的含义','',
            '动作置换结果可以作为内部机制证据：在固定边际动作频率、平均输入近似相同的条件下，匹配状态带来更高开发回复偏好。正文须同时标明 synthetic development 的范围。外部资源优势仍单独用原 204 状态结果呈现，不混合两种质量量纲。']
    for arm in ['rule_off','fixed_off']:
        for metric,name in [('local_first_text_ms','首个可见文本'),('local_complete_ms','完整回复')]:
            r=next(r for r in lat['contrasts'] if r['baseline']==arm and r['metric']==metric and r['cluster']=='user_id')
            conclusion='差值区间完全低于零，支持本次本地试跑中 PM 均值较低' if r['ci_high']<0 else '差值区间完全高于零，PM 在该指标较慢' if r['ci_low']>0 else '差值区间跨零，不能确立时延优势'
            lines+=['',f"- 相对 {labels[arm]} 的{name}：{conclusion}。"]
    lines+=['','48 状态计时和动作置换都是已有结果后的补充诊断，不能升级成质量等效或普遍最优的确认性证明。Context Only 和 ME+R0 参照保留，不能仅因 PM 优于较高资源方法就称其全局最划算。',
            '', '## 执行与复现','',
            '首次启动 CUDA 数字编号映射到了 A4500，设备断言在推理前停止。随后按 A6000 UUID 锁定设备；没有计时样本被排除。原日志及 [运行环境](execution_environment.json)保留。',
            '', '在 project 目录使用 sim_eval Python，设置 `PYTHONNOUSERSITE=1 PYTHONPATH=src OPENBLAS_NUM_THREADS=1`。无需 source API 密钥。',
            '', '- 验证执行冻结：`python analysis/targeted_checks_20260914/latency.py verify`。',
            '- 汇总与独立验证：`python analysis/targeted_checks_20260914/analyze_results.py`。',
            '- 原始请求结果按 request_id 落盘；同一冻结运行可续跑，已完成请求不重跑。',
            '', '分析脚本按照冻结指标在运行期编写，哈希单列于 analysis manifest。独立验证包括所有原始成本映射、留出动作对应、40 张代表卡的另一路最小二乘分数重构、720 次请求与执行顺序对应、自然停止、时间分项和完整计时相加一致。',
            '', '[完整验证记录](verification.json)、[分析文件哈希](analysis_manifest.json)。','']
    (OUT/'README_CN.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
