"""Frozen, bounded supplemental judge checks; never changes the v1 experiment."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
from typing import Literal

from pydantic import Field
from metacom_pm.contracts import StrictModel
from metacom_pm.io import canonical_json, read_json, sha256_file, sha256_text, utc_now, write_json
from metacom_pm.table3 import input_token_estimate
from metacom_pm.table3_execution import Runner, SCHEMAS, StopRun, nano_cost, verify_freeze

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/reviewer_supplement_20260914'
FROZEN, RUN = OUT / 'frozen', OUT / 'run'
PARENT = ROOT / 'outputs/table3_under10/run_identity_v7'
SPEC = importlib.util.spec_from_file_location('supplement_wire', ROOT / 'scripts/28_run_table3_wire.py')
WIRE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WIRE)


class HistoryNeed(StrictModel):
    unit_id: str
    history_need: Literal['necessary', 'helpful', 'not_needed', 'uncertain']
    already_available_in_current_context: bool
    historical_evidence_quote: str = Field(max_length=700)
    reason: str = Field(min_length=1, max_length=900)


SCHEMAS['history_need'] = HistoryNeed
HISTORY_SYSTEM = '''Assess the information demand of one emotional-support turn BEFORE seeing any candidate response.
You see only the current conversation and authorized historical context. No method identity, replies, scores or costs are provided.
Classify the incremental value of facts outside the current conversation:
necessary: a competent, non-generic answer to an explicit recall request or unresolved historical reference requires a specific prior fact absent from the current conversation;
helpful: a specific prior fact absent from the current conversation would materially improve relevant emotional support, but an appropriate answer remains possible without it;
not_needed: current conversation supplies the relevant facts, or extra history would be merely decorative, redundant, intrusive or unrelated;
uncertain: the evidence is insufficient to decide.
Do not equate more personalization with better support. Generic empathy is not automatically defective. Do not infer clinical benefit.
already_available_in_current_context is true when the historical fact you considered is already conveyed, including paraphrases, by the current message or earlier current-session turns.
For necessary/helpful, cite one exact verbatim substring from a string value in historical_context, and explain what useful information it adds beyond current_conversation. If the fact is already available, classify not_needed.
For not_needed/uncertain an evidence quote is optional, but any nonempty quote must be exact historical text. Copy unit_id exactly. Return the schema only.'''


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def string_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from string_values(v)
    elif isinstance(value, list):
        for v in value:
            yield from string_values(v)


def sample_units(units):
    """Eight units per seed/turn stratum, greedy user/scenario coverage, no outcomes."""
    chosen, uc, sc = [], Counter(), Counter()
    cells = sorted({(u['seed'], u['turn_index']) for u in units})
    for _ in range(8):
        for cell in cells:
            eligible = [u for u in units if (u['seed'], u['turn_index']) == cell and u not in chosen]
            u = min(eligible, key=lambda u: (uc[u['user_id']], sc[(u['user_id'], u['topic_index'])],
                sha256_text('20260914:' + u['unit_id'])))
            chosen.append(u)
            uc[u['user_id']] += 1
            sc[(u['user_id'], u['topic_index'])] += 1
    assert len(chosen) == 48 and len(uc) == 18
    return chosen


def make_job(body, mapping, unit_id, variant, stage, judge, execution):
    inp = input_token_estimate(body['messages'], SCHEMAS[stage])
    reserved = math.ceil(inp * execution['input_reservation_multiplier']) + execution['input_reservation_extra_tokens']
    identity = f'supplement-20260914:{stage}:{unit_id}:{variant}'
    job = dict(body=body, mapping=mapping, unit_id=unit_id, variant=variant, stage=stage,
        phase='supplement', judge=judge, batch=True, arm=None, identity=identity,
        input_tokens_est=inp, reserve_nano=nano_cost(reserved, body['max_tokens'], judge, 0.5))
    job['body'] = WIRE.ordered_body(job)
    job['wire_sha256'] = sha256_text(WIRE.wire_json(job['body']))
    job['id'] = sha256_text(identity + ':' + job['wire_sha256'])[:40]
    return job


def prepare():
    if FROZEN.exists():
        print('Existing freeze:', verify_freeze(ROOT, FROZEN)['freeze_sha256'])
        return
    parent_config = ROOT / 'outputs/table3_under10/frozen_identity_v7/config.json'
    config = deepcopy(read_json(parent_config))
    config['protocol'] = 'pm-v1-reviewer-supplement-20260914-v1'
    config['budget']['execution_reservation_limit_usd'] = 1.50
    config['execution']['maximum_attempts_per_job'] = 3
    config['execution']['batch_poll_seconds'] = 45
    config['analysis'] = dict(bootstrap_resamples=10000, bootstrap_seed=20260914,
        primary_cluster='user', sensitivity_cluster='scenario', estimand='equal sampled state weights',
        quality_comparisons=['pm_off-rule_off', 'pm_off-fixed_off', 'pm_on-rule_on', 'pm_on-fixed_on'],
        quality_metrics=['overall', 'emotional_support', 'personalization', 'memory_appropriateness',
                         'factual_grounding', 'temporal_consistency', 'non_intrusiveness'],
        primary_order_quantity='Within-state difference of PM-baseline differences: reverse minus original',
        secondary_order_quantity='PM-baseline difference averaged over both new orders',
        inference='Post-hoc diagnostic; no superiority/equivalence pass gate; no outcome-driven sample replacement',
        history_need='Response-blind exploratory judge labels on all 204 states; no claim of human truth',
        control_gate='All six explicit recall controls necessary; all six repetition controls not_needed with already_available=true',
        control_failure='Report classifications as unvalidated diagnostics; do not claim memory-demand prevalence established')
    source_units = ROOT / config['source_plan'] / 'units.jsonl'
    units = jsonl(source_units)
    chosen = sample_units(units)
    originals = {j['unit_id']: j for j in jsonl(PARENT / 'main.jobs.jsonl') if j['stage'] == 'quality'}
    jobs = []
    for unit in chosen:
        original = originals[unit['unit_id']]
        for variant in [0, 1]:
            body = deepcopy(original['body'])
            mapping = deepcopy(original['mapping'])
            if variant:
                payload = json.loads(body['messages'][1]['content'])
                candidates = list(reversed(payload['candidates']))
                mapping = {}
                for pos, candidate in enumerate(candidates, 1):
                    prior_id = candidate['candidate_id']
                    new_id = 'C' + str(pos)
                    mapping[new_id] = dict(original['mapping'][prior_id], position=pos)
                    candidate['candidate_id'] = new_id
                payload['candidates'] = candidates
                body['messages'][1]['content'] = canonical_json(payload)
            jobs.append(make_job(body, mapping, unit['unit_id'], variant, 'quality', config['judge'], config['execution']))

    history_payloads = []
    for u in units:
        history = {k: v for k, v in u['authorized_ground_truth'].items() if k not in ['current_topic', 'evaluator_only']}
        history_payloads.append(dict(unit_id=u['unit_id'], historical_context=history,
            current_conversation=dict(current_seeker_message=u['seeker_message'], context_before_turn=u['context_before_turn'])))
    # Explicitly synthetic annotation controls. Never mixed into the 204-state sample.
    facts = [
        ('I prefer listening without advice when I am overwhelmed.', 'What did I say I prefer when I am overwhelmed?'),
        ('My old dog was named Juniper.', 'What was the name of the dog I told you about last time?'),
        ('I said that Tuesday evenings are when I call my sister.', 'Which evening did I say I call my sister?'),
        ('I wanted to try painting as my relaxing hobby.', 'Which relaxing hobby did I say I wanted to try?'),
        ('My supportive colleague is named Noor.', 'What was the name of the supportive colleague I mentioned?'),
        ('I said I feel calmer after a walk beside the river.', 'Where did I say I walk when I want to feel calmer?'),
    ]
    controls = []
    for i, (fact, question) in enumerate(facts):
        for kind in ['recall', 'repeated']:
            controls.append(dict(unit_id=f'control_{i}_{kind}', historical_context={'earlier_user_statement': fact},
                current_conversation=dict(current_seeker_message=question if kind == 'recall' else fact + ' I have just told you the relevant detail again. Please help me reflect on that.', context_before_turn=[])))
    for payload in history_payloads + controls:
        body = dict(model=config['judge']['model'], temperature=0.0, seed=20260910, max_tokens=600,
            messages=[dict(role='system', content=HISTORY_SYSTEM), dict(role='user', content=canonical_json(payload))],
            response_format={'type': 'json_schema', 'json_schema': {'name': 'HistoryNeed', 'strict': True, 'schema': HistoryNeed.model_json_schema()}})
        jobs.append(make_job(body, None, payload['unit_id'], 0, 'history_need', config['judge'], config['execution']))
    assert len(jobs) == 312
    # Submission order is independent of condition, seed and response outcomes.
    jobs.sort(key=lambda j: sha256_text('wire-shuffle-20260914:' + j['id']))
    with sqlite3.connect(f'file:{PARENT / "ledger.sqlite"}?mode=ro', uri=True) as c:
        prior_cost = c.execute('SELECT SUM(COALESCE(actual_nano,reserve_nano)) FROM attempts').fetchone()[0] / 1e9
    prior_cost += 0.0013  # Previously disclosed off-ledger diagnostic estimate.
    reserve = sum(j['reserve_nano'] for j in jobs) / 1e9
    assert reserve < 1.50 and prior_cost + 1.50 < 10
    FROZEN.mkdir(parents=True)
    write_json(FROZEN / 'config.json', config)
    write_json(FROZEN / 'pilot_units.json', [])
    write_json(FROZEN / 'sample_units.json', chosen)
    (FROZEN / 'jobs.jsonl').write_text(''.join(WIRE.wire_json(j) + '\n' for j in jobs))
    write_json(FROZEN / 'preflight.json', dict(n_quality=96, n_history_need=204, n_synthetic_controls=12,
        selected_users=18, selected_scenarios=len({(u['user_id'], u['topic_index']) for u in chosen}),
        original_plus_supplement_cap_usd=prior_cost+1.50, previous_estimated_usd=prior_cost,
        initial_reservation_usd=reserve, supplemental_cap_usd=1.50,
        pricing_checked_on='2026-09-14', pricing_url='https://developers.openai.com/api/docs/pricing',
        sample_selection='Eight per seed/turn; minimize prior selected user count, then scenario count, then SHA256(20260914:unit_id)',
        controls='Six authored explicit recall and six repeated-information cases; excluded from experiment n',
        unchanged_generation=True, reverse_is_exact=True))
    files = [parent_config, source_units, PARENT / 'main.jobs.jsonl', PARENT / 'all_results/per_response.csv',
             Path(__file__), ROOT / 'src/metacom_pm/table3_execution.py', ROOT / 'src/metacom_pm/table3.py',
             ROOT / 'src/metacom_pm/evo_response_v4.py', ROOT / 'scripts/28_run_table3_wire.py']
    files += list(FROZEN.iterdir())
    manifest = dict(status='FROZEN', frozen_at=utc_now(),
        files={str(p.relative_to(ROOT)): sha256_file(p) for p in files},
        packages=read_json(ROOT / 'outputs/table3_under10/frozen_identity_v7/manifest.json')['packages'])
    manifest['freeze_sha256'] = sha256_text(canonical_json(manifest))
    write_json(FROZEN / 'manifest.json', manifest)
    print(canonical_json(read_json(FROZEN / 'preflight.json')))


class SupplementalRunner(Runner):
    submit_batch = WIRE.WireRunner.submit_batch

    def record_response(self, aid, job, body, http_status=200):
        done = super().record_response(aid, job, body, http_status)
        if done and job['stage'] == 'history_need':
            parsed = json.loads(self.ledger.successful(job['id'])['parsed_json'])
            payload = json.loads(job['body']['messages'][1]['content'])
            quote = parsed['historical_evidence_quote']
            error = None
            if parsed['unit_id'] != job['unit_id']:
                error = 'unit_id mismatch'
            elif quote and not any(quote in value for value in string_values(payload['historical_context'])):
                error = 'Historical evidence quote is not verbatim'
            elif parsed['history_need'] in ('necessary', 'helpful') and (not quote or parsed['already_available_in_current_context']):
                error = 'Incremental history label contradicts required quote/context flag'
            if error:
                self.ledger.update(aid, status='invalid', error=error)
                return False
        return done

    def run_supplement(self):
        jobs = jsonl(FROZEN / 'jobs.jsonl')
        with self.lock():
            for attempt in range(1, 4):
                pending = [j for j in jobs if not self.ledger.successful(j['id'])]
                if not pending:
                    self.status('COMPLETE', valid_jobs=len(jobs))
                    return
                key = f'supplement-20260914-a{attempt}'
                state = self.submit_batch(key, pending)
                try:
                    self.wait_batch(key, state)
                except StopRun:
                    rows = {r['id']: r for r in self.ledger.rows()}
                    if any(rows[a]['status'] != 'done' and rows[a]['status'] != 'invalid' for a in state['attempts']):
                        raise
                    if attempt == 3:
                        raise
            if not all(self.ledger.successful(j['id']) for j in jobs):
                raise StopRun('Unresolved supplemental jobs; no sample removal')
            self.status('COMPLETE', valid_jobs=len(jobs))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'verify', 'run'])
    args = p.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'verify':
        print(verify_freeze(ROOT, FROZEN)['freeze_sha256'])
    else:
        runner = SupplementalRunner(ROOT, FROZEN, RUN)
        try:
            runner.run_supplement()
        except Exception as exc:
            runner.status('STOPPED', reason=str(exc) if isinstance(exc, StopRun) else type(exc).__name__)
            raise
        finally:
            runner.close()
