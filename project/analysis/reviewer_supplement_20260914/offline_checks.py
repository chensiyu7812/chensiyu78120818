"""Reproducible local selector audit and extraction of existing comparison CIs."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import mean
import subprocess

from metacom_pm.contracts import RuntimeState, StrategyCard, parse_action_id, StrategyMode
from metacom_pm.evoemo import load_evoemo, build_evo_memory, make_evo_runtime_state, _policy_from_selection
from metacom_pm.io import read_json, iter_jsonl, canonical_json, sha256_text, sha256_file, write_json
from metacom_pm.policies import estimated_action_cost, RuleConfig, StrongRulePolicy
from metacom_pm.table3 import PreparedStrategyRetriever
from metacom_pm.training import PMModel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/reviewer_supplement_20260914/offline'
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = []


def source(rel):
    path = ROOT / rel
    SOURCES.append(path)
    return path


def summarize(rows):
    result = dict(n_states=len(rows), n_legal_actions=sum(r['n_legal'] for r in rows),
        chosen_action_counts=dict(Counter(r['chosen_action'] for r in rows)),
        rs_rate=mean(r['chosen_action'].endswith('+RS') for r in rows),
        definition='Replay of frozen decision rule; no response generation and no causal quality attribution')
    for field in ['any_risk_pruned','risk_changes_choice','response_tie','cost_distinguishes_tie','cost_changes_choice',
                  'fallback_used','chosen_rs_below_matched_r0','omission_pruned','strategy_pruned']:
        result[field + '_states'] = sum(r[field] for r in rows)
    for field in ['n_safe','n_candidates','n_legal','pruned_actions','chosen_cost','cost_removed_choice_cost','risk_removed_choice_cost']:
        result[field + '_mean'] = mean(r[field] for r in rows)
    for field in ['misuse_risk','memory_omission_risk','strategy_decision_risk','response_score']:
        values=[v[field] for row in rows for v in row['scores'].values()]
        selected=[row['scores'][row['chosen_action']][field] for row in rows]
        result[field] = dict(all_action_min=min(values), all_action_max=max(values),
                            chosen_min=min(selected), chosen_max=max(selected), chosen_mean=mean(selected))
    matched=[pair for r in rows for pair in r['matched_strategy_predictions']]
    result['matched_strategy_predictions'] = dict(n_pairs=len(matched),
        rs_higher=sum(p['delta_rs_minus_r0']>0 for p in matched),
        ties=sum(p['delta_rs_minus_r0']==0 for p in matched),
        r0_higher=sum(p['delta_rs_minus_r0']<0 for p in matched),
        mean_delta_rs_minus_r0=mean(p['delta_rs_minus_r0'] for p in matched))
    return result


def inspect(state, policy, metadata):
    chosen = policy.choose(state)
    report = policy.last_decision_report
    scores, safe, candidates = report['scores'], report['safe_actions'], report['candidate_actions']
    costs = {a: estimated_action_cost(state,a) for a in scores}
    def tie_key(a, with_cost=True):
        return ((costs[a],) if with_cost else ()) + (-scores[a]['memory_decision_quality'],
             -scores[a]['strategy_decision_quality'], -scores[a]['response_score'], a)
    cost_removed = min(candidates,key=lambda a:tie_key(a,False))
    qmax = max(s['response_score'] for s in scores.values())
    unrestricted = [a for a,s in scores.items() if s['response_score']>=qmax-policy.epsilon]
    risk_removed = min(unrestricted,key=tie_key)
    assert min(candidates,key=tie_key)==chosen
    matched=[]
    for a in scores:
        if a.endswith('+R0') and a[:-2]+'RS' in scores:
            other=a[:-2]+'RS'
            matched.append(dict(memory_action=a.split('+')[0],
                delta_rs_minus_r0=scores[other]['response_score']-scores[a]['response_score']))
    assert matched
    selected_r0=chosen[:-2]+'R0'
    return dict(**metadata,card_id=state.card_id,state_id=state.state_id,user_id=state.user_id,
        chosen_action=chosen,n_legal=len(scores),n_safe=len(safe),n_candidates=len(candidates),
        any_risk_pruned=len(safe)<len(scores),pruned_actions=len(scores)-len(safe),
        risk_changes_choice=risk_removed!=chosen,response_tie=len(candidates)>1,
        cost_distinguishes_tie=len({costs[a] for a in candidates})>1,
        cost_changes_choice=cost_removed!=chosen,cost_removed_choice=cost_removed,risk_removed_choice=risk_removed,
        chosen_cost=costs[chosen],cost_removed_choice_cost=costs[cost_removed],risk_removed_choice_cost=costs[risk_removed],
        fallback_used=report['constraint_fallback_used'],
        omission_pruned=any(s['memory_omission_risk']>policy.tau_omission for s in scores.values()),
        strategy_pruned=any(s['strategy_decision_risk']>policy.tau_strategy for s in scores.values()),
        chosen_rs_below_matched_r0=chosen.endswith('+RS') and scores[chosen]['response_score']<scores[selected_r0]['response_score'],
        safe_actions=safe,candidate_actions=candidates,scores=scores,action_costs=costs,matched_strategy_predictions=matched)


def run():
    selection=read_json(source('outputs/selection_stable.json'))
    old_stats=read_json(source('outputs/evoemo_response_v4/response_statistical_summary.json'))
    extracted=[]
    for field,value in old_stats['paired_pm_deltas']['no_memory_r0'].items():
        if field=='overall':continue  # Legacy alias of emotional_support, not an independent seventh metric.
        extracted.append(dict(metric=field,**value))
    write_json(OUT/'pm_vs_context_only_existing_cis.json',dict(
        status='EXTRACTED_EXISTING_STATISTICS_NOT_RECOMPUTED_FROM_RAW_SCORES',
        scope='204 original V4 scored states; six candidates; original GPT-4o protocol',
        caveat='User cluster estimate in the original file averages users equally; scenario estimate equals state mean because scenarios each contain six states. No simultaneous CI correction.',
        results=extracted))
    with (OUT/'pm_vs_context_only.csv').open('w',newline='') as f:
        fields=['metric','state_mean_delta','scenario_lower','scenario_upper','equal_user_mean_delta','user_lower','user_upper']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in extracted:
            sc,uc=r['scenario_cluster_bootstrap_ci'],r['user_cluster_bootstrap_ci']
            w.writerow(dict(metric=r['metric'],state_mean_delta=r['mean_delta_pm_minus_baseline'],scenario_lower=sc['lower'],scenario_upper=sc['upper'],equal_user_mean_delta=uc['estimate'],user_lower=uc['lower'],user_upper=uc['upper']))
    # The later diagnostic exists on origin/main, outside the earlier v1 tag.
    commit=subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip()
    evidence=OUT/'existing_cost_matched_evidence';evidence.mkdir(exist_ok=True)
    names=['project/docs/METACOM_V33_COST_MATCHED_BASELINE_DIAGNOSTIC_ZH.md',
           'project/scripts/17p_analyze_cost_matched_fixed_baseline.py',
           'project/scripts/17q_run_cost_matched_fixed_baseline.py',
           'project/scripts/17r_eval_cost_matched_fixed_response.py',
           'project/scripts/17s_eval_cost_matched_forced_swap_probe.py']
    restored={}
    missing=[]
    for name in names:
        result=subprocess.run(['git','show',f'{commit}:{name}'],cwd=ROOT,capture_output=True)
        if result.returncode:
            missing.append(name)
            continue
        content=result.stdout
        dest=evidence/Path(name).name;dest.write_bytes(content);restored[name]=sha256_file(dest)
    write_json(evidence/'PROVENANCE.json',dict(commit=commit,files=restored,missing_referenced_scripts=missing,
        raw_results_available=False,scope='Archived implementation and result report; referenced per-state raw outputs absent from both inspected checkouts. Reported old values are not newly recomputed.'))
    print('Extracted old comparisons and diagnostic provenance',flush=True)

    config=read_json(source('outputs/table3_under10/frozen_identity_v7/config.json'))
    units=list(iter_jsonl(source(config['source_plan']+'/units.jsonl')))
    users={u['id']:u for u in load_evoemo(source('data/external/evo_emo.json'))}
    memories={uid:build_evo_memory(u)[0] for uid,u in users.items()}
    final=source('outputs/table3_under10/run_identity_v7/all_results/per_response.csv')
    expected={(r['unit_id'],r['arm']):r['requested_action_id'] for r in csv.DictReader(final.open(encoding='utf-8-sig'))}
    model=PMModel.load(source(config['checkpoint']))
    policy=_policy_from_selection(model,selection)
    strategies=PreparedStrategyRetriever([StrategyCard.model_validate(r) for r in iter_jsonl(source('data/strategy/strategy_cards.jsonl'))],top_k=3)
    rule=StrongRulePolicy(RuleConfig(**selection['strong_rule']['config']),strategies)
    external=[]
    for unit in units:
        user=users[unit['user_id']];topic=next(t for t in user['subsequent_topics'] if t['idx']==unit['topic_index'])
        state=make_evo_runtime_state(user,topic,unit['context_before_turn'],unit['seeker_message'],memories[user['id']],unit['turn_index'],'table3',track_id=unit['track_id'],fixed_open_loop=True)
        assert sha256_text(canonical_json(state.model_dump(mode='json')))==unit['state_sha256']
        row=inspect(state,policy,dict(dataset='external_204',seed=unit['seed'],turn_index=unit['turn_index'],topic_index=unit['topic_index']))
        assert row['chosen_action']==expected[(unit['unit_id'],'pm_off')]
        row['rule_action']=rule.choose(state)
        assert row['rule_action']==expected[(unit['unit_id'],'rule_off')]
        external.append(row)
    (OUT/'external_selector_replay.jsonl').write_text(''.join(canonical_json(r)+'\n' for r in external))
    write_json(OUT/'external_selector_summary.json',summarize(external))
    print('Replayed 204 external states; PM and Rule reproduce every frozen action',flush=True)

    states={r['card_id']:RuntimeState.model_validate(r) for r in iter_jsonl(source('data/synthetic/runtime_states.jsonl'))}
    folds=[r for r in iter_jsonl(source('data/synthetic/folds.jsonl')) if r['mode']=='user']
    development=[];seen=set()
    for fold in folds:
        path=f"outputs/models_cv_m2b_stable/text_metadata_stable_user_fold{fold['fold']}_seed17.joblib"
        p=_policy_from_selection(PMModel.load(source(path)),selection)
        for card in fold['validation_card_ids']:
            assert card not in seen;seen.add(card)
            development.append(inspect(states[card],p,dict(dataset='development_oof_1728',fold=fold['fold'])))
        print('Replayed held-out predictions, fold',fold['fold'],flush=True)
    assert seen==set(states) and len(development)==1728
    (OUT/'development_selector_replay.jsonl').write_text(''.join(canonical_json(r)+'\n' for r in development))
    write_json(OUT/'development_selector_summary.json',dict(**summarize(development),
        threshold_scope='Final frozen consensus thresholds applied to out-of-fold models. Development diagnostics, not an independent test or a newly nested performance estimate.'))
    judgments=list(iter_jsonl(source('outputs/full_judging_gemini_flash_lite_v3/response_pair_judgments.jsonl')))
    matched=[];keys=set()
    for r in judgments:
        if not r['training_eligible']:continue
        a,sa=parse_action_id(r['action_a']);b,sb=parse_action_id(r['action_b'])
        if a!=b or sa==sb:continue
        key=(r['card_id'],tuple(sorted([r['action_a'],r['action_b']])))
        assert key not in keys;keys.add(key)
        winner='tie' if r['preference']=='tie' else (sa if r['preference']=='A' else sb).value
        matched.append(dict(card_id=r['card_id'],memory_action=r['action_a'].split('+')[0],winner=winner,dual_order_agreed=r['dual_order_agreed']))
    write_json(OUT/'development_matched_strategy_labels.json',dict(n_pairs=len(matched),
        winner_counts=dict(Counter(r['winner'] for r in matched)),dual_order_disagreements=sum(not r['dual_order_agreed'] for r in matched),
        by_memory={m:dict(Counter(r['winner'] for r in matched if r['memory_action']==m)) for m in sorted({r['memory_action'] for r in matched})},
        scope='Existing dual-order training-eligible response comparisons with identical memory-source actions and RS/R0 swapped; developmental label evidence, not external causal evidence.'))
    variants=Counter(r['variant'] for r in iter_jsonl(source('data/synthetic/audit_only.jsonl')))
    write_json(OUT/'development_inventory_counts.json',dict(variants=dict(variants),
        legal_action_counts=dict(Counter(len(s.allowed_actions) for s in states.values())),
        users=len({s.user_id for s in states.values()}),states=len({s.state_id for s in states.values()}),
        current_texts=len({s.current_user_text for s in states.values()})))
    source('src/metacom_pm/policies.py');source('src/metacom_pm/training.py');source('src/metacom_pm/labels.py')
    source('src/metacom_pm/evoemo.py');source('src/metacom_pm/preparation.py');source('src/metacom_pm/selection.py')
    write_json(OUT/'source_manifest.json',dict(script_sha256=sha256_file(Path(__file__)),
        sources={str(p.relative_to(ROOT)):sha256_file(p) for p in SOURCES},
        inference='Post-hoc no-API diagnostic and archived-statistic extraction; no tuning, no new quality labels',
        cost_ablation='Remove cost from the existing lexicographic key; other components retained. A choice difference is mechanical, not a causal quality effect.'))
    print('Offline supplements complete',flush=True)


if __name__=='__main__':
    run()
