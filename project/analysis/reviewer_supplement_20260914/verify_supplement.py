"""Read-only delivery checks for the supplemental evidence package."""
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import mean

import run_checks_v2 as protocol
from metacom_pm.io import canonical_json, read_json, sha256_file
from metacom_pm.table3_execution import verify_freeze

ROOT=protocol.ROOT
OUT=ROOT/'outputs/reviewer_supplement_20260914'


def main():
    verify_freeze(ROOT,protocol.OLD_FROZEN)
    verify_freeze(ROOT,protocol.FROZEN)
    for filename in ['offline/source_manifest.json','results/analysis_manifest.json']:
        manifest=read_json(OUT/filename)
        for rel,expected in manifest.get('sources',manifest.get('files',{})).items():
            assert sha256_file(ROOT/rel)==expected,(filename,rel)
    rows=list(csv.DictReader((OUT/'results/quality_scores.csv').open(encoding='utf-8-sig')))
    keys={(r['unit_id'],r['arm'],r['order_variant']) for r in rows}
    assert len(rows)==len(keys)==576
    assert len({r['unit_id'] for r in rows})==48
    assert set(Counter((r['arm'],r['order_variant'],r['candidate_position']) for r in rows).values())=={8}
    scores={(r['unit_id'],r['arm'],int(r['order_variant'])):r for r in rows}
    intervals=read_json(OUT/'results/order_intervals.json')
    units=sorted({r['unit_id'] for r in rows})
    for interval in intervals:
        pm,baseline=interval['comparison'].split('-')
        field=interval['metric']
        original=[float(scores[u,pm,0][field])-float(scores[u,baseline,0][field]) for u in units]
        reverse=[float(scores[u,pm,1][field])-float(scores[u,baseline,1][field]) for u in units]
        expected={'original':mean(original),'reverse':mean(reverse),
            'averaged_orders':(mean(original)+mean(reverse))/2,
            'reverse_minus_original':mean(reverse)-mean(original)}[interval['quantity']]
        assert abs(interval['estimate']-expected)<1e-12
        assert interval['lower']<=interval['upper']
    source_responses={(r['unit_id'],r['arm']):r for r in csv.DictReader((protocol.base.PARENT/'all_results/per_response.csv').open(encoding='utf-8-sig'))}
    original_jobs={j['unit_id']:j for j in protocol.base.jsonl(protocol.base.PARENT/'main.jobs.jsonl') if j['stage']=='quality'}
    for job in protocol.base.jsonl(protocol.FROZEN/'jobs.jsonl'):
        payload=json.loads(job['body']['messages'][1]['content'])
        if job['stage']=='quality':
            for c in payload['candidates']:
                arm=job['mapping'][c['candidate_id']]['condition']
                assert c['response']==source_responses[job['unit_id'],arm]['response_text']
            old=original_jobs[job['unit_id']]['body']['messages'][1]['content']
            payload['candidates']=json.loads(old)['candidates']
            assert protocol.base.WIRE.wire_json(payload)==old
        else:
            assert len(payload['current_conversation']['context_before_turn'])<=8
            assert 'current_topic' not in payload['historical_context']
    replay=protocol.base.jsonl(OUT/'offline/external_selector_replay.jsonl')
    assert len(replay)==204
    for r in replay:
        assert r['chosen_action']==source_responses[r['state_id'],'pm_off']['requested_action_id']
        assert r['rule_action']==source_responses[r['state_id'],'rule_off']['requested_action_id']
    history=list(csv.DictReader((OUT/'results/history_need_labels.csv').open(encoding='utf-8-sig')))
    assert len(history)==len({r['unit_id'] for r in history})==204
    summary=read_json(OUT/'results/history_summary.json')
    assert dict(Counter(r['history_need'] for r in history))==summary['counts']
    assert sum(r['first_attempt_citation_verbatim']=='False' for r in history)==35
    assert sum(r['verified_result_available_after_retries']=='False' for r in history)==25
    receipt=read_json(OUT/'results/receipt.json')
    assert receipt['supplemental_spend_upper_usd']<=1.50
    assert receipt['total_including_prior_estimated_usd']<10
    assert receipt['unresolved_history_citations']==25
    assert receipt['valid_jobs']==287
    print('PASS: both freezes; source and output hashes; 576 unique scores; balanced positions; exact response reuse; order-only prompt intervention; all CI point estimates; 204 reproduced PM/Rule actions; all 204 history labels with 35/25 citation-failure disclosure; combined budget')


if __name__=='__main__':main()
