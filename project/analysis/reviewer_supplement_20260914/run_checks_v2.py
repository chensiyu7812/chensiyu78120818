"""Explicit amendment: preserve candidate prompt serialization and visible window."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import run_checks as base
from metacom_pm.table3_execution import Ledger

ROOT=base.ROOT
OLD_FROZEN, OLD_RUN=base.FROZEN, base.RUN
FROZEN=base.OUT/'frozen_v2'
RUN=base.OUT/'run_v2'


def prepare():
    if FROZEN.exists():
        print(base.verify_freeze(ROOT,FROZEN)['freeze_sha256'])
        return
    old_ledger=Ledger(OLD_RUN/'ledger.sqlite',limit=1.50)
    old_rows=old_ledger.rows()
    assert not any(r['status'] in ('reserved','queued','in_flight','unknown') for r in old_rows), 'Reconcile old submissions first'
    old_cost=old_ledger.summary()['charged_plus_reserved_usd']
    config=deepcopy(base.read_json(OLD_FROZEN/'config.json'))
    config['protocol'] += '-candidate-field-order-visible-window-v2'
    config['budget']['execution_reservation_limit_usd']=round(1.50-old_cost,9)
    config['amendment']=dict(parent_freeze=base.read_json(OLD_FROZEN/'manifest.json')['freeze_sha256'],
        cause='v1 reverse prompt inadvertently sorted all payload keys, violating the intended order-only intervention; v1 history classification saw more context than the generator',
        correction='Preserve exact original serialization outside candidate order and IDs; use generator-visible last eight prior messages; explicitly prohibit nonhistorical evidence quotations',
        original_results='Retained as superseded technical execution; no v1 quality or history label enters final v2 estimates',
        old_supplement_spent_usd=old_cost, combined_supplement_cap_usd=1.50,
        unchanged='All 48 selected units, all 204 full-sample units, all original responses, model snapshot, scoring rubric, analysis rules and synthetic controls')
    source_units=ROOT/config['source_plan']/'units.jsonl'
    units={u['unit_id']:u for u in base.jsonl(source_units)}
    originals={j['unit_id']:j for j in base.jsonl(base.PARENT/'main.jobs.jsonl') if j['stage']=='quality'}
    jobs=[]
    for prior in base.jsonl(OLD_FROZEN/'jobs.jsonl'):
        body=deepcopy(prior['body']);mapping=deepcopy(prior['mapping'])
        if prior['stage']=='quality':
            original=originals[prior['unit_id']]
            body=deepcopy(original['body']);mapping=deepcopy(original['mapping'])
            original_text=body['messages'][1]['content']
            payload=json.loads(original_text)
            assert base.WIRE.wire_json(payload)==original_text
            if prior['variant']:
                mapping={}
                reverse=list(reversed(payload['candidates']))
                for position,candidate in enumerate(reverse,1):
                    oldid=candidate['candidate_id'];newid=f'C{position}'
                    mapping[newid]=dict(original['mapping'][oldid],position=position)
                    candidate['candidate_id']=newid
                payload['candidates']=reverse
                body['messages'][1]['content']=base.WIRE.wire_json(payload)
            # Exact lexical prefix and suffix around the candidates array stay unchanged.
            marker='"candidates":['
            assert body['messages'][1]['content'].split(marker,1)[0]==original_text.split(marker,1)[0]
            assert json.loads(body['messages'][1]['content'])['output_json_shape']==json.loads(original_text)['output_json_shape']
        else:
            payload=json.loads(body['messages'][1]['content'])
            if prior['unit_id'] in units:
                u=units[prior['unit_id']]
                context=deepcopy(u['context_before_turn'])
                if context and context[-1].get('role')=='seeker' and context[-1].get('content','').strip()==u['seeker_message'].strip():
                    context=context[:-1]
                payload['current_conversation']['context_before_turn']=context[-8:]
            body['messages'][0]['content'] += '\nSTRICT EVIDENCE OUTPUT RULE: For not_needed or uncertain, historical_evidence_quote MUST be the empty string. For necessary or helpful, copy an exact substring from historical_context ONLY, never from current_conversation. Do not paraphrase, concatenate, or change punctuation in the quote. The current conversation shown is the actual generator-visible context window.'
            body['messages'][1]['content']=base.WIRE.wire_json(payload)
        job=base.make_job(body,mapping,prior['unit_id'],prior['variant'],prior['stage'],config['judge'],config['execution'])
        job['identity']=job['identity'].replace('supplement-20260914:','supplement-20260914-v2:')
        job['id']=base.sha256_text(job['identity']+':'+job['wire_sha256'])[:40]
        jobs.append(job)
    jobs.sort(key=lambda j:base.sha256_text('wire-shuffle-20260914:'+j['id']))
    reserve=sum(j['reserve_nano'] for j in jobs)/1e9
    assert reserve<config['budget']['execution_reservation_limit_usd']
    assert Counter(j['stage'] for j in jobs)=={'quality':96,'history_need':216}
    FROZEN.mkdir()
    base.write_json(FROZEN/'config.json',config)
    base.write_json(FROZEN/'pilot_units.json',[])
    base.write_json(FROZEN/'sample_units.json',base.read_json(OLD_FROZEN/'sample_units.json'))
    (FROZEN/'jobs.jsonl').write_text(''.join(base.WIRE.wire_json(j)+'\n' for j in jobs))
    preflight=dict(base.read_json(OLD_FROZEN/'preflight.json'),
        initial_reservation_usd=reserve,old_supplement_spent_usd=old_cost,
        v2_remaining_cap_usd=config['budget']['execution_reservation_limit_usd'],
        context='Actual generator-visible most recent 8 prior messages; no current_topic annotations',
        serialization='Original unsorted JSON field order retained exactly outside candidate array/IDs')
    base.write_json(FROZEN/'preflight.json',preflight)
    files=[Path(__file__),ROOT/'analysis/reviewer_supplement_20260914/run_checks.py',OLD_FROZEN/'manifest.json',
        ROOT/'src/metacom_pm/table3_execution.py',ROOT/'src/metacom_pm/table3.py',ROOT/'src/metacom_pm/evo_response_v4.py',
        ROOT/'scripts/28_run_table3_wire.py',source_units,base.PARENT/'main.jobs.jsonl',base.PARENT/'all_results/per_response.csv']+list(FROZEN.iterdir())
    manifest=dict(status='FROZEN',frozen_at=base.utc_now(),files={str(p.relative_to(ROOT)):base.sha256_file(p) for p in files},
        packages=base.read_json(OLD_FROZEN/'manifest.json')['packages'])
    manifest['freeze_sha256']=base.sha256_text(base.canonical_json(manifest))
    base.write_json(FROZEN/'manifest.json',manifest)
    print(base.canonical_json(preflight))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','verify','run']);args=p.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='verify':print(base.verify_freeze(ROOT,FROZEN)['freeze_sha256'])
    else:
        base.FROZEN,base.RUN=FROZEN,RUN
        runner=base.SupplementalRunner(ROOT,FROZEN,RUN)
        try:runner.run_supplement()
        except Exception as exc:
            runner.status('STOPPED',reason=str(exc) if isinstance(exc,base.StopRun) else type(exc).__name__)
            raise
        finally:runner.close()
