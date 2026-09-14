"""Post-hoc, no-API checks of resource CIs and fixed-input seed consistency."""
import csv
import hashlib
import json
import sqlite3
from collections import Counter
from collections import defaultdict
from pathlib import Path
from statistics import mean
import numpy as np
from metacom_pm.contracts import parse_action_id, StrategyMode

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/reviewer_supplement_20260914/offline/resources'
OUT.mkdir(parents=True,exist_ok=True)
V1=ROOT/'outputs/evoemo_response_v4'
NEW=ROOT/'outputs/table3_under10/run_identity_v7'
paths=[V1/'sample_plan.json',V1/'cost_estimate_full_calls.jsonl',V1/'response_resource_summary.json',NEW/'all_results/per_response.csv']
plan={r['unit_id']:r for r in json.loads(paths[0].read_text())['units']}
calls=[json.loads(s) for s in paths[1].open()]
assert len(plan)==len(calls)==204 and {j['unit_id'] for j in calls}==set(plan)
legacy=[]
for j in calls:
    u=plan[j['unit_id']]
    for c in j['candidate_mapping'].values():
        if c['condition'] not in ['pm','strong_rule','best_fixed']:continue
        sources,strategy=parse_action_id(c['action_id'])
        legacy.append(dict(unit_id=u['state_id'],user_id=u['user_id'],scenario_id=f"{u['user_id']}:{u['topic_index']}",
            seed=u['seed'],turn_index=u['turn_index'],arm=c['condition'],input_tokens=c['input_tokens'],
            source_invocations=len(sources)+int(strategy==StrategyMode.RS)))
old_summary=json.loads(paths[2].read_text())['v4_sampled_turns']['condition_summary']
for arm in ['pm','strong_rule','best_fixed']:
    rs=[r for r in legacy if r['arm']==arm]
    assert len(rs)==204
    assert abs(mean(r['input_tokens'] for r in rs)-old_summary[arm]['total_input_tokens']['mean'])<1e-9
    assert abs(mean(r['source_invocations'] for r in rs)-old_summary[arm]['retrieval_calls']['mean'])<1e-9
new=list(csv.DictReader(paths[3].open(encoding='utf-8-sig')))
for r in new:
    r['seed']=int(r['seed']);r['turn_index']=int(r['turn_index'])
    for field in ['generator_input_tokens','source_invocations','overall','emotional_support']:
        r[field]=float(r[field])
    r['input_tokens']=r['generator_input_tokens']

def paired(rows,left,right):
    a={r['unit_id']:r for r in rows if r['arm']==left}
    b={r['unit_id']:r for r in rows if r['arm']==right}
    assert set(a)==set(b)
    return [(a[u],b[u]) for u in sorted(a)]

def cluster_intervals(pairs,metric,cluster):
    keys=sorted({a[cluster] for a,b in pairs})
    arrays={key:[(a[metric],b[metric]) for a,b in pairs if a[cluster]==key] for key in keys}
    sums=np.array([[sum(p[0] for p in arrays[k]),sum(p[1] for p in arrays[k]),len(arrays[k])] for k in keys])
    rng=np.random.default_rng(20260911)
    draw=rng.integers(0,len(keys),size=(10000,len(keys)))
    totals=sums[draw].sum(axis=1)
    delta=(totals[:,0]-totals[:,1])/totals[:,2]
    savings=100*(1-totals[:,0]/totals[:,1])
    lo,hi=np.quantile(delta,[.025,.975]);rl,rh=np.quantile(savings,[.025,.975])
    allvals=sums.sum(axis=0)
    return dict(metric=metric,cluster=cluster,n_pairs=len(pairs),n_clusters=len(keys),
        pm_mean=allvals[0]/allvals[2],baseline_mean=allvals[1]/allvals[2],
        delta_pm_minus_baseline=(allvals[0]-allvals[1])/allvals[2],lower=float(lo),upper=float(hi),
        reduction_percent=100*(1-allvals[0]/allvals[1]),reduction_lower=float(rl),reduction_upper=float(rh))

cis,seeds,subgroups=[],[],[]
comparisons=[('legacy_204',legacy,'pm','strong_rule'),('legacy_204',legacy,'pm','best_fixed')]
comparisons += [('rerun_204',new,'pm_'+flag,b+'_'+flag) for flag in ['off','on'] for b in ['rule','fixed']]
for dataset,rows,left,right in comparisons:
    pairs=paired(rows,left,right)
    assert len(pairs)==204
    for metric in ['input_tokens','source_invocations']:
        for cluster in ['user_id','scenario_id']:
            cis.append(dict(dataset=dataset,left=left,right=right,**cluster_intervals(pairs,metric,cluster)))
    for seed in [101,202,303]:
        ps=[(a,b) for a,b in pairs if a['seed']==seed]
        assert len(ps)==68
        r=dict(dataset=dataset,left=left,right=right,seed=seed,n_pairs=68,
            input_delta=mean(a['input_tokens']-b['input_tokens'] for a,b in ps),
            input_reduction_percent=100*(1-sum(a['input_tokens'] for a,b in ps)/sum(b['input_tokens'] for a,b in ps)))
        if dataset=='rerun_204':
            for metric in ['overall','emotional_support']:
                r[metric+'_delta']=mean(a[metric]-b[metric] for a,b in ps)
        seeds.append(r)
    saved=[(a,b) for a,b in pairs if a['input_tokens']<b['input_tokens']]
    r=dict(dataset=dataset,left=left,right=right,n=204,input_saved_n=len(saved))
    if dataset=='rerun_204':
        for metric in ['overall','emotional_support']:
            r[metric+'_saved_tie_or_win']=sum(a[metric]>=b[metric] for a,b in saved)
            r[metric+'_saved_lower']=sum(a[metric]<b[metric] for a,b in saved)
    subgroups.append(r)

for name,data in [('resource_cis',cis),('seed_checks',seeds),('descriptive_subgroups',subgroups)]:
    (OUT/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(name)
    for row in data:
        if name!='resource_cis' or (row['dataset']=='legacy_204' and row['cluster']=='user_id'):print(json.dumps(row))
positions=defaultdict(Counter)
for r in new:
    positions[(r['seed'],r['turn_index'],r['arm'])][r['quality_candidate_position']]+=1
position_rows=[dict(seed=seed,turn_index=turn,arm=arm,position_counts=dict(counts))
    for (seed,turn,arm),counts in sorted(positions.items())]
assert all(len(r['position_counts'])==1 and sum(r['position_counts'].values())==34 for r in position_rows)
(OUT/'candidate_order_by_seed_turn.json').write_text(json.dumps(position_rows,indent=2)+'\n')

pilot={}
ledger=NEW/'ledger.sqlite'
paths.append(ledger)
with sqlite3.connect(f'file:{ledger}?mode=ro',uri=True) as connection:
    for filename in ['pilot_mini.jobs.jsonl','pilot_reference.jobs.jsonl']:
        job_path=NEW/filename
        paths.append(job_path)
        scores={}
        for job in map(json.loads,job_path.read_text().splitlines()):
            if job['stage']!='quality':continue
            result=connection.execute("select parsed_json from attempts where job_id=? and status='done' order by number desc limit 1",(job['id'],)).fetchone()
            assert result is not None
            for candidate in json.loads(result[0])['candidates']:
                arm=job['mapping'][candidate['candidate_id']]['condition']
                scores[(job['unit_id'],job['variant'],arm)]=candidate['overall']
        units=sorted({key[0] for key in scores})
        assert len(units)==12 and len(scores)==144
        contrasts=[]
        for flag in ['off','on']:
            for baseline in ['rule','fixed']:
                deltas=[mean(scores[(u,variant,'pm_'+flag)]-scores[(u,variant,baseline+'_'+flag)] for u in units) for variant in [0,1]]
                contrasts.append(dict(comparison=f'pm_{flag} - {baseline}_{flag}',original_order_delta=deltas[0],
                    complementary_order_delta=deltas[1],difference_between_order_deltas=deltas[1]-deltas[0]))
        pilot[filename]=dict(n_units=len(units),contrasts=contrasts)
(OUT/'pilot_order_contrasts.json').write_text(json.dumps(pilot,indent=2)+'\n')

(OUT/'source_manifest.json').write_text(json.dumps({'analysis':'Post-hoc no-API descriptive/sensitivity checks, not a new confirmatory experiment',
    'bootstrap_seed':20260911,'resamples':10000,'estimand':'State-weighted mean; resample users as primary clusters, scenarios as sensitivity',
    'scope':'Legacy resources are reconstructed from saved candidate mappings on 204 scored states; not the 1020-state population. Legacy per-seed quality is unavailable here.',
    'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2)+'\n')
