"""Conditional action permutation using existing held-out development labels."""
from collections import Counter,defaultdict
import numpy as np
from common import ROOT,OUT,rows,write,csv_write,cluster_ci,verify
from metacom_pm.contracts import RuntimeState, parse_action_id, StrategyMode
from metacom_pm.labels import action_response_scores,load_memory_labels,load_strategy_labels
from metacom_pm.policies import FixedPolicy

METRICS=['response_preference','misuse','omission','strategy_risk','input_tokens','source_invocations']

def main():
    freeze=verify()
    if (OUT/'shuffle/summary.json').exists():raise RuntimeError('Completed shuffle output already exists')
    dev=rows(ROOT/'outputs/reviewer_supplement_20260914/offline/development_selector_replay.jsonl')
    states={r['card_id']:RuntimeState.model_validate(r) for r in rows(ROOT/'data/synthetic/runtime_states.jsonl')}
    base=ROOT/'outputs/full_judging_gemini_flash_lite_v3'
    response=action_response_scores(base/'response_pair_judgments.jsonl')
    memory=load_memory_labels(base/'memory_omission_judgments.jsonl',base/'memory_use_judgments.jsonl',ROOT/'outputs/m2b_selected_set_omission_gemini_flash_lite_v3/memory_selected_set_omission_judgments.jsonl')
    strategy=load_strategy_labels(base/'strategy_use_judgments.jsonl',base/'strategy_omission_judgments.jsonl')
    outcome_rows=rows(ROOT/'outputs/synthetic_sweep/action_outcomes.jsonl')
    costs={(r['card_id'],r['action_id']):r['cost']['total_input_tokens'] for r in outcome_rows}
    assert len(costs)==len(outcome_rows)==11136
    assert len(dev)==len(states)==1728 and len({r['card_id'] for r in dev})==1728
    actions=sorted({a for s in states.values() for a in s.allowed_actions});ai={a:i for i,a in enumerate(actions)}
    cube=np.full((len(dev),len(actions),len(METRICS)),np.nan)
    groups=defaultdict(list)
    for i,r in enumerate(dev):
        s=states[r['card_id']];assert set(r['scores'])==set(s.allowed_actions)
        groups[(r['fold'],tuple(s.allowed_actions))].append(i)
        for a in s.allowed_actions:
            k=(s.card_id,a);sources,mode=parse_action_id(a)
            cube[i,ai[a]]=[response[k],memory.misuse_risk[k],memory.omission_risk[k],strategy.risk[k],costs[k],len(sources)+int(mode is StrategyMode.RS)]
    chosen=np.array([ai[r['chosen_action']] for r in dev]);idx=np.arange(len(dev));pm=cube[idx,chosen]
    expected=np.zeros_like(pm);strata=[]
    for (fold,legal),inds in sorted(groups.items()):
        inds=np.array(inds);counts=Counter(chosen[inds]);freq=np.array([counts[ai[a]]/len(inds) for a in legal])
        expected[inds]=(cube[inds][:,[ai[a] for a in legal],:]*freq[None,:,None]).sum(axis=1)
        strata.append(dict(fold=fold,legal_actions=list(legal),n=len(inds),action_counts={actions[a]:c for a,c in counts.items()}))
    assert np.isfinite(pm).all() and np.isfinite(expected).all()
    assert abs(pm[:,-1].sum()-expected[:,-1].sum())<1e-9
    gen=np.random.default_rng(20260915);null=np.zeros((10000,len(METRICS)))
    for b in range(10000):
        perm=chosen.copy()
        for inds in groups.values():perm[inds]=gen.permutation(chosen[inds])
        vals=cube[idx,perm];assert np.isfinite(vals).all()
        null[b]=vals.mean(axis=0)
        assert abs(null[b,-1]-pm[:,-1].mean())<1e-9
    ci=cluster_ci(pm-expected,[r['user_id'] for r in dev],20260916)
    contrasts=[dict(metric=m,pm_mean=float(pm[:,j].mean()),shuffle_expected_mean=float(expected[:,j].mean()),delta_pm_minus_expected=float((pm-expected)[:,j].mean()),user_ci_low=float(ci[0,j]),user_ci_high=float(ci[1,j]),permutation_mean=float(null[:,j].mean()),permutation_p025=float(np.quantile(null[:,j],.025)),permutation_p975=float(np.quantile(null[:,j],.975))) for j,m in enumerate(METRICS)]
    policies={'pm':pm,'shuffle_expectation':expected}
    for name,a in [('context_only','M0+R0'),('me_r0','ME+R0'),('full_structured','MPMSME+RS')]:
        p=FixedPolicy(a);chosen_fixed=np.array([ai[p.choose(states[r['card_id']])] for r in dev]);policies[name]=cube[idx,chosen_fixed]
    descriptive=[dict(policy=name,**{m:float(vals[:,j].mean()) for j,m in enumerate(METRICS)}) for name,vals in policies.items()]
    per_card=[dict(card_id=r['card_id'],state_id=r['state_id'],user_id=r['user_id'],fold=r['fold'],legal_actions='|'.join(states[r['card_id']].allowed_actions),chosen_action=r['chosen_action'],**{f'{name}_{m}':float(vals[i,j]) for name,vals in policies.items() for j,m in enumerate(METRICS)}) for i,r in enumerate(dev)]
    by_fold=[]
    for fold in sorted({r['fold'] for r in dev}):
        inds=[i for i,r in enumerate(dev) if r['fold']==fold]
        by_fold.append(dict(fold=fold,n=len(inds),**{m:float((pm[inds]-expected[inds])[:,j].mean()) for j,m in enumerate(METRICS)}))
    by_inventory=[]
    for legal in sorted({tuple(states[r['card_id']].allowed_actions) for r in dev}):
        inds=[i for i,r in enumerate(dev) if tuple(states[r['card_id']].allowed_actions)==legal]
        by_inventory.append(dict(legal_actions='|'.join(legal),n=len(inds),**{m:float((pm[inds]-expected[inds])[:,j].mean()) for j,m in enumerate(METRICS)}))
    dest=OUT/'shuffle';dest.mkdir(parents=True,exist_ok=True)
    csv_write(dest/'per_card.csv',per_card);csv_write(dest/'contrasts.csv',contrasts);csv_write(dest/'policy_means.csv',descriptive)
    csv_write(dest/'by_fold.csv',by_fold);csv_write(dest/'by_inventory.csv',by_inventory)
    np.savez_compressed(dest/'permutation_means.npz',means=null,metrics=np.array(METRICS))
    write(dest/'summary.json',dict(status='COMPLETE_DEVELOPMENT_DIAGNOSTIC',freeze_sha256=freeze['freeze_sha256'],n_cards=len(dev),n_users=len({r['user_id'] for r in dev}),n_strata=len(groups),n_permutations=len(null),strata=strata,contrasts=contrasts,policy_means=descriptive,api_usd=0,limits=['Existing OOF models with final consensus thresholds; not new independent or nested validation.','Cluster intervals condition on action frequencies estimated from this development set.','Legal action marginals are preserved, but risk feasibility under the recipient state is not reimposed.','Response-preference scale is developmental [0,1], not external 1-5 Support.','Input cost is the stored sweep input-token metric, not newly measured latency.']))
    print({'status':'COMPLETE','cards':len(dev),'contrasts':contrasts},flush=True)

if __name__=='__main__':main()
