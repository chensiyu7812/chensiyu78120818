"""Verify transport evidence and analyze only the amended supplemental protocol."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import mean
import numpy as np
from metacom_pm.evoemo import load_evoemo, build_evo_memory
from metacom_pm.io import canonical_json, read_json, sha256_file, sha256_text, write_json
from metacom_pm.table3_execution import Ledger, parse_response, verify_freeze
from metacom_pm.text import normalize_space
import run_checks_v2 as protocol

ROOT, FROZEN, RUN=protocol.ROOT,protocol.FROZEN,protocol.RUN
OUT=ROOT/'outputs/reviewer_supplement_20260914/results'
PARENT=protocol.base.PARENT
METRICS=['overall','emotional_support','personalization','memory_appropriateness','factual_grounding','temporal_consistency','non_intrusiveness']


def write_csv(path,rows):
    assert rows
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def ci(rows,values,key):
    clusters=sorted({r[key] for r in rows})
    grouped={k:[v for r,v in zip(rows,values) if r[key]==k] for k in clusters}
    arrays=np.array([[sum(grouped[k]),len(grouped[k])] for k in clusters])
    rng=np.random.default_rng(20260914)
    totals=arrays[rng.integers(0,len(clusters),size=(10000,len(clusters)))].sum(axis=1)
    lower,upper=np.quantile(totals[:,0]/totals[:,1],[.025,.975])
    return dict(estimate=mean(values),lower=float(lower),upper=float(upper),n=len(rows),clusters=len(clusters))


def main():
    manifest=verify_freeze(ROOT,FROZEN)
    config=read_json(FROZEN/'config.json')
    ledger=Ledger(RUN/'ledger.sqlite',limit=config['budget']['execution_reservation_limit_usd'])
    jobs=protocol.base.jsonl(FROZEN/'jobs.jsonl')
    success={j['id']:ledger.successful(j['id']) for j in jobs}
    assert all(success[j['id']] for j in jobs if j['stage']=='quality'), 'All 96 valid quality jobs required'
    attempts=ledger.rows()
    assert not any(r['status'] in ('reserved','queued','in_flight','unknown') for r in attempts)
    first={r['job_id']:r for r in attempts if r['number']==1}
    # Validate the actual uploaded request and returned response for each selected observation.
    wire_inputs={}
    wire_outputs={}
    for p in RUN.glob('*.batch_input.jsonl'):
        for r in protocol.base.jsonl(p):
            assert r['custom_id'] not in wire_inputs
            wire_inputs[r['custom_id']]=r
    for p in RUN.glob('*.output_file_id.jsonl'):
        for r in protocol.base.jsonl(p):
            assert r['custom_id'] not in wire_outputs
            wire_outputs[r['custom_id']]=r
    units={u['unit_id']:u for u in protocol.base.jsonl(ROOT/config['source_plan']/'units.jsonl')}
    scores={};quality_rows=[];history=[];controls=[]
    for j in jobs:
        # Quality uses validated scoring outputs. History labels are reported from
        # every FIRST response, with citation failures retained and flagged; do not
        # let quote-triggered retries selectively change the reported prevalence.
        record=success[j['id']] if j['stage']=='quality' else first[j['id']]
        aid=record['id']
        assert sha256_text(protocol.base.WIRE.wire_json(wire_inputs[aid]['body']))==j['wire_sha256']
        response=wire_outputs[aid]['response'];assert response['status_code']==200
        assert canonical_json(response['body'])==record['response_json']
        parsed=parse_response(j,response['body'])
        assert canonical_json(parsed)==record['parsed_json']
        if j['stage']=='quality':
            u=units[j['unit_id']]
            for c in parsed['candidates']:
                arm=j['mapping'][c['candidate_id']]['condition']
                row=dict(unit_id=u['unit_id'],user_id=u['user_id'],scenario_id=f"{u['user_id']}:{u['topic_index']}",
                    seed=u['seed'],turn_index=u['turn_index'],order_variant=j['variant'],arm=arm,
                    candidate_position=j['mapping'][c['candidate_id']]['position'],**c)
                key=(u['unit_id'],arm,j['variant']);assert key not in scores
                scores[key]=row;quality_rows.append(row)
        else:
            payload=json.loads(j['body']['messages'][1]['content'])
            quote=parsed['historical_evidence_quote']
            citation_valid=not quote or any(quote in s for s in protocol.base.string_values(payload['historical_context']))
            if j['unit_id'].startswith('control_'):
                expected='necessary' if j['unit_id'].endswith('_recall') else 'not_needed'
                controls.append(dict(**parsed,expected=expected,passed=parsed['history_need']==expected and
                    (expected!='not_needed' or parsed['already_available_in_current_context'])))
            else:
                u=units[j['unit_id']]
                latest=success[j['id']]
                history.append(dict(**parsed,user_id=u['user_id'],scenario_id=f"{u['user_id']}:{u['topic_index']}",seed=u['seed'],turn_index=u['turn_index'],
                    first_attempt_citation_verbatim=citation_valid,first_attempt_status=record['status'],
                    verified_result_available_after_retries=bool(latest),
                    label_in_verified_result=json.loads(latest['parsed_json'])['history_need'] if latest else 'unverified'))
    assert len(quality_rows)==576 and len(history)==204 and len(controls)==12
    OUT.mkdir(exist_ok=True)
    write_csv(OUT/'quality_scores.csv',quality_rows)
    sample=read_json(FROZEN/'sample_units.json')
    sample_meta=[dict(u,scenario_id=f"{u['user_id']}:{u['topic_index']}") for u in sample]
    intervals=[]
    for flag in ['off','on']:
        for baseline in ['rule','fixed']:
            for metric in METRICS:
                values={variant:[scores[(u['unit_id'],'pm_'+flag,variant)][metric]-scores[(u['unit_id'],baseline+'_'+flag,variant)][metric] for u in sample] for variant in [0,1]}
                quantities=dict(original=values[0],reverse=values[1],
                    averaged_orders=[(a+b)/2 for a,b in zip(values[0],values[1])],
                    reverse_minus_original=[b-a for a,b in zip(values[0],values[1])])
                for quantity,vals in quantities.items():
                    for cluster in ['user_id','scenario_id']:
                        intervals.append(dict(comparison=f'pm_{flag}-{baseline}_{flag}',metric=metric,quantity=quantity,cluster=cluster,**ci(sample_meta,vals,cluster)))
    write_csv(OUT/'order_intervals.csv',intervals)
    write_json(OUT/'order_intervals.json',intervals)
    by_arm=[]
    for arm in ['pm_off','rule_off','fixed_off','pm_on','rule_on','fixed_on']:
        for metric in METRICS:
            a=[scores[(u['unit_id'],arm,0)][metric] for u in sample]
            b=[scores[(u['unit_id'],arm,1)][metric] for u in sample]
            by_arm.append(dict(arm=arm,metric=metric,n=48,original_mean=mean(a),reverse_mean=mean(b),
                averaged_mean=mean((x+y)/2 for x,y in zip(a,b)),mean_absolute_order_difference=mean(abs(y-x) for x,y in zip(a,b))))
    write_csv(OUT/'order_score_means.csv',by_arm)
    arm_records={(r['unit_id'],r['arm']):r for r in protocol.base.jsonl(ROOT/config['source_plan']/'arms.jsonl')}
    backend={u['id']:build_evo_memory(u)[0] for u in load_evoemo(ROOT/'data/external/evo_emo.json')}
    for r in history:
        quote=normalize_space(r['historical_evidence_quote'])
        r['has_evidence_quote']=bool(quote)
        r['quote_in_any_deployable_memory']=bool(quote) and any(quote in normalize_space(m.text) for m in backend[r['user_id']])
        for arm in ['pm_off','rule_off','fixed_off']:
            r['quote_in_'+arm+'_memory']=bool(quote) and any(quote in normalize_space(m['text']) for m in arm_records[r['unit_id'],arm]['kept_memory'])
    write_csv(OUT/'history_need_labels.csv',history)
    write_json(OUT/'history_controls.json',controls)
    final=list(csv.DictReader((PARENT/'all_results/per_response.csv').open(encoding='utf-8-sig')))
    labels={r['unit_id']:r['history_need'] for r in history}
    groups=[]
    for label in ['necessary','helpful','not_needed','uncertain']:
        for arm in ['pm_off','rule_off','fixed_off','pm_on','rule_on','fixed_on']:
            rows=[r for r in final if r['arm']==arm and labels[r['unit_id']]==label]
            if not rows:continue
            groups.append(dict(history_need=label,arm=arm,n=len(rows),
                overall_mean=mean(float(r['overall']) for r in rows),
                support_mean=mean(float(r['emotional_support']) for r in rows),
                clear_omission_rate=mean(float(r['omission_severity'])>=2 for r in rows),
                omission_severity_mean=mean(float(r['omission_severity']) for r in rows),
                input_tokens_mean=mean(float(r['generator_input_tokens']) for r in rows)))
    write_csv(OUT/'exploratory_history_subgroups.csv',groups)
    controls_pass=all(c['passed'] for c in controls)
    quote_rows=[r for r in history if r['history_need'] in ['necessary','helpful']]
    verified_quotes=[r for r in quote_rows if r['first_attempt_citation_verbatim']]
    history_summary=dict(n=204,counts=dict(Counter(labels.values())),controls_pass=controls_pass,
        status='EXPLORATORY_LABELS_WITH_UNVERIFIED_EVIDENCE',label_selection='First response for all 204 states, including flagged citation failures; no outcome selection among retries',
        first_attempt_citation_failures=sum(not r['first_attempt_citation_verbatim'] for r in history),
        unresolved_after_frozen_retries=sum(not r['verified_result_available_after_retries'] for r in history),
        verified_result_label_counts=dict(Counter(r['label_in_verified_result'] for r in history)),
        n_controls_passed=sum(c['passed'] for c in controls),n_controls=12,
        interpretation='Response-blind exploratory LLM labels, not human truth. Even passing synthetic controls does not establish annotation accuracy on real states.',
        evidence_quote_scope='Literal quote coverage only; not semantic recall or sufficiency',
        n_necessary_or_helpful=len(quote_rows),
        n_necessary_or_helpful_with_verbatim_quote=len(verified_quotes),
        verified_quote_in_any_deployable_memory=sum(r['quote_in_any_deployable_memory'] for r in verified_quotes),
        verified_quote_in_pm_off=sum(r['quote_in_pm_off_memory'] for r in verified_quotes),
        verified_quote_in_rule_off=sum(r['quote_in_rule_off_memory'] for r in verified_quotes),
        verified_quote_in_fixed_off=sum(r['quote_in_fixed_off_memory'] for r in verified_quotes),
        quoted_evidence_in_any_deployable_memory=sum(r['quote_in_any_deployable_memory'] for r in quote_rows),
        quoted_evidence_in_pm_off=sum(r['quote_in_pm_off_memory'] for r in quote_rows),
        quoted_evidence_in_rule_off=sum(r['quote_in_rule_off_memory'] for r in quote_rows),
        quoted_evidence_in_fixed_off=sum(r['quote_in_fixed_off_memory'] for r in quote_rows),
        by_seed={s:dict(Counter(r['history_need'] for r in history if r['seed']==s)) for s in [101,202,303]},
        by_turn={t:dict(Counter(r['history_need'] for r in history if r['turn_index']==t)) for t in [3,8]})
    write_json(OUT/'history_summary.json',history_summary)
    omissions=[]
    for arm in ['pm_off','rule_off','fixed_off','pm_on','rule_on','fixed_on']:
        rows=[r for r in final if r['arm']==arm]
        omissions.append(dict(arm=arm,n=len(rows),
            severity_counts=dict(Counter(r['omission_severity'] for r in rows)),
            clear_omission_n=sum(float(r['omission_severity'])>=2 for r in rows),
            clear_omission_rate=mean(float(r['omission_severity'])>=2 for r in rows)))
    write_json(OUT/'existing_omission_measure_audit.json',dict(results=omissions,
        definition='Original independent omission judge: missing useful authorized context, not pure historical recall',
        evaluator_information='Includes current_topic plus historical profile/timeline/related sessions; some judge-visible facts are not retrievable memory',
        examples=[{k:r[k] for k in ['unit_id','arm','omission_severity','omission_reason']} for r in final if float(r['omission_severity'])>=2][:6]))
    old=Ledger(protocol.OLD_RUN/'ledger.sqlite',limit=1.50).summary()
    current=ledger.summary();spent=old['charged_plus_reserved_usd']+current['charged_plus_reserved_usd']
    assert spent<=1.50
    receipt=dict(status='QUALITY_COMPLETE_HISTORY_REPORTED_WITH_CITATION_FAILURES',freeze_sha256=manifest['freeze_sha256'],valid_jobs=sum(bool(v) for v in success.values()),
        quality_states=48,quality_orders=2,quality_observations=576,history_states=204,synthetic_controls=12,
        source_transport_identity_verified=312,first_history_labels_reported=204,unresolved_history_citations=history_summary['unresolved_after_frozen_retries'],old_superseded_run=old,final_run=current,
        supplemental_spend_upper_usd=spent,
        total_including_prior_estimated_usd=read_json(FROZEN/'preflight.json')['previous_estimated_usd']+spent,
        combined_supplement_cap_usd=1.50,old_runs_excluded_from_estimates=True,
        inference='Descriptive paired cluster intervals; no familywise adjustment; not proof of quality equivalence')
    write_json(OUT/'receipt.json',receipt)
    print(canonical_json(receipt))
    print(canonical_json(history_summary))
    for r in intervals:
        if r['metric']=='overall' and r['cluster']=='user_id' and r['quantity'] in ['averaged_orders','reverse_minus_original']:
            print(canonical_json(r))
    write_json(OUT/'analysis_manifest.json',dict(script_sha256=sha256_file(Path(__file__)),
        freeze_sha256=manifest['freeze_sha256'],files={str(p.relative_to(ROOT)):sha256_file(p) for p in OUT.iterdir() if p.is_file() and p.name!='analysis_manifest.json'}))


if __name__=='__main__':main()
