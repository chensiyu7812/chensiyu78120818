"""PM v1 six-arm diagnostic: deterministic memory filter and no-API plan.

Original generation, state, policy and retrieval implementations are imported
unchanged. Pricing before generation reserves response tokens explicitly; it
never uses fabricated responses as observations. This module makes no API calls.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import json
import math
import time

from pydantic import Field
import tiktoken

from .contracts import MemorySource, StrategyCard, StrategyMode, StrictModel, canonical_action_id, parse_action_id
from .evoemo import (_fixed_context_before_turn, _load_fixed_tracks, _policy_from_selection,
                     _track_key, build_evo_memory, evaluator_context, load_evoemo, make_evo_runtime_state)
from .evo_response_v4 import EvoResponseV4Judgment, _messages_for_unit, balanced_candidate_order
from .evo_sampled_audit import AUDIT_RUBRIC, EvoSampledAuditJudgment
from .io import canonical_json, iter_jsonl, read_json, sha256_file, sha256_text, write_json
from .policies import FixedPolicy, RuleConfig, StrongRulePolicy
from .prompts import SELECTIVE_ESMEM_SYSTEM, generation_messages
from .retrieval import MemoryRetriever, StrategyRetriever, context_query
from .text import estimate_tokens, lexical_score, normalize_for_hash, normalize_space, tokens
from .training import PMModel


ARMS = ('pm_off', 'pm_on', 'rule_off', 'rule_on', 'fixed_off', 'fixed_on')


class PreparedStrategyRetriever(StrategyRetriever):
    """Offline planning optimization; identical v1 lexical scores and tie-breaks.

    Only memoizes fixed document Counters/norms and the latest query scores.
    Never uses evaluation outcomes and does not change candidate retrieval.
    """
    def __init__(self, cards, top_k=3):
        super().__init__(cards, top_k)
        self._documents = [Counter(tokens(c.retrieval_text)) for c in self.cards]
        self._norms = [math.sqrt(sum(v*v for v in d.values())) for d in self._documents]
        self._last_query = None
        self._last_scores = None

    def _scores(self, query):
        if query != self._last_query:
            q = Counter(tokens(query))
            qn = math.sqrt(sum(v*v for v in q.values()))
            self._last_scores = [
                sum(q[t]*d.get(t,0) for t in q)/(qn*dn) if qn and dn else 0.0
                for d,dn in zip(self._documents,self._norms)
            ]
            self._last_query = query
        return self._last_scores

    def retrieve(self, query):
        ranked = sorted(zip(self._scores(query),self.cards),
                        key=lambda x:(x[0],x[1].strategy_id), reverse=True)
        return [c for _,c in ranked[:self.top_k]]

    def confidence(self, query):
        return max(self._scores(query),default=0.0)


class OmissionJudgment(StrictModel):
    omission_severity: int = Field(ge=0, le=3)
    response_support_sufficiency: int = Field(ge=1, le=5)
    reason: str = Field(min_length=1, max_length=900)


def filter_memory(candidates, current, query, config, *, enabled):
    if not config['strategy_passthrough'] or not config['allow_empty_memory']:
        raise ValueError('This diagnostic requires memory-only filtering and permits empty memory')
    if set(config['memory_rules']) != {s.value for s in MemorySource}:
        raise ValueError('Every memory source requires a rule')
    rows = []
    for item in candidates:
        rule = config['memory_rules'][item.source.value]
        cur, ctx = lexical_score(current, item.text), lexical_score(query, item.text)
        tok = estimate_tokens(item.text)
        eligible = cur >= rule['minimum_current_score'] or ctx >= rule['minimum_context_score']
        reason = 'relevance_gate_passed' if eligible else 'below_relevance_threshold'
        if (eligible and rule['long_item_token_threshold'] is not None
                and tok > rule['long_item_token_threshold']
                and cur < rule['long_item_minimum_current_score']):
            eligible, reason = False, 'long_item_requires_high_current_relevance'
        rows.append({'item_id': item.memory_id, 'source': item.source.value,
                     'current_score': cur, 'context_score': ctx, 'selector_score': max(cur, ctx),
                     'estimated_tokens': tok, 'text_key': normalize_for_hash(item.text),
                     'eligible': eligible, 'keep': not enabled,
                     'reason': reason if enabled else 'filter_disabled_keep_candidate'})
    if enabled:
        counts, seen, kept = Counter(), set(), 0
        ranked = sorted((r for r in rows if r['eligible']),
                        key=lambda r: (-r['selector_score'], r['estimated_tokens'], r['item_id']))
        for row in ranked:
            if config['drop_exact_duplicate_text'] and row['text_key'] in seen:
                row['reason'] = 'exact_duplicate_evidence'
            elif counts[row['source']] >= config['memory_rules'][row['source']]['maximum_items']:
                row['reason'] = 'source_item_cap'
            elif kept >= config['maximum_total_memory_items']:
                row['reason'] = 'total_memory_item_cap'
            else:
                row['keep'], row['reason'] = True, 'kept_relevant_candidate'
                counts[row['source']] += 1
                kept += 1
                seen.add(row['text_key'])
    # Preserve original retrieval order, not the sorting order used for caps.
    view = [item for item, row in zip(candidates, rows) if row['keep']]
    return view, [{k: v for k, v in row.items() if k != 'text_key'} for row in rows]


def quality_messages(unit, arm_records, responses, *, variant=0):
    candidates = {a: {'supporter_message': responses[a]} for a in ARMS}
    order = balanced_candidate_order(ARMS, unit_ordinal=unit['ordinal'], order_variant=variant)
    return _messages_for_unit(unit, condition_order=order, candidate_by_condition=candidates,
                              authorized_ground_truth=unit['authorized_ground_truth'])


def risk_messages(unit, arm, response):
    payload = {
        'audit_call_id': sha256_text(arm['call_id'])[:24],
        'rubric': AUDIT_RUBRIC,
        'current_seeker_message': unit['seeker_message'],
        'context_before_turn': unit['context_before_turn'],
        'response': response,
        'selected_memory': arm['kept_memory'],
        'selected_strategy': arm['selected_strategy'],
    }
    system = (
        'Audit one emotional-support response. Score actual response misuse, unnecessary exposure, '
        'stale/conflicting use, unsupported personal claims, source fit, strategy use, and support '
        'sufficiency using the supplied rubric. Irrelevant retrieved evidence that the response '
        'does not use is not itself response misuse. Do not reward using more personal details. '
        'You do not know the method, resource cost, or other responses. Do not infer unseen facts. '
        'Set omission_severity to 0: omission is measured by a separate authorized-context judge. '
        'Return JSON matching the schema, copying audit_call_id exactly.'
    )
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': canonical_json(payload)}]


def omission_messages(unit, response):
    payload = {'current_seeker_message': unit['seeker_message'],
               'context_before_turn': unit['context_before_turn'],
               'authorized_ground_truth': unit['authorized_ground_truth'], 'response': response}
    system = (
        'Independently audit omission of useful authorized personal context in this emotional-support '
        'response. Judge the response against the supplied user history and current turn. '
        'Do not reward mentioning more facts; using no history can be appropriate. Score omission '
        'only when missing authorized information would materially help the current response. '
        'omission_severity: 0 not observed, 1 minor/ambiguous, 2 clear loss of usefulness, 3 severe. '
        'response_support_sufficiency: 1 poor, 2 weak, 3 adequate, 4 good, 5 excellent. '
        'Do not infer the method or resource availability. Return JSON matching the schema with '
        'a brief concrete reason. No other candidate responses are provided.'
    )
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': canonical_json(payload)}]


@lru_cache(maxsize=1)
def encoding():
    return tiktoken.get_encoding('o200k_base')


def input_token_estimate(messages, schema=None):
    # Exact BPE count of visible text, approximate Chat Completions/schema wrapper.
    total = 3 + sum(3 + len(encoding().encode(m['role'])) + len(encoding().encode(m['content'])) for m in messages)
    if schema is not None:
        total += len(encoding().encode(canonical_json(schema.model_json_schema()))) + 32
    return total


def verify_baseline(root, config):
    freeze = read_json(root / 'outputs/study_freeze_stable.json')
    needed = [config['checkpoint'], config['selection'], 'data/external/evo_emo.json',
              'data/strategy/strategy_cards.jsonl', 'outputs/evoemo_fixed_tracks/fixed_seeker_tracks.jsonl',
              'src/metacom_pm/evoemo.py', 'src/metacom_pm/policies.py', 'src/metacom_pm/training.py',
              'src/metacom_pm/features.py', 'src/metacom_pm/retrieval.py', 'src/metacom_pm/prompts.py',
              'src/metacom_pm/text.py', 'src/metacom_pm/contracts.py']
    all_hashes = {}
    for section in ('checkpoint_hashes', 'data_hashes', 'code_hashes', 'prompt_hashes'):
        all_hashes.update(freeze[section])
    result = {}
    for rel in needed:
        actual = sha256_file(root / rel)
        if actual != all_hashes.get(rel):
            raise ValueError(f'Baseline mismatch: {rel}')
        result[rel] = actual
    return result


def build_units_and_arms(root, config):
    users = load_evoemo(root / 'data/external/evo_emo.json')
    tracks = _load_fixed_tracks(root / 'outputs/evoemo_fixed_tracks/fixed_seeker_tracks.jsonl')
    cards = [StrategyCard.model_validate(r) for r in iter_jsonl(root / 'data/strategy/strategy_cards.jsonl')]
    retriever, strategies = MemoryRetriever(), PreparedStrategyRetriever(cards, top_k=3)
    selection = read_json(root / config['selection'])
    pm = _policy_from_selection(PMModel.load(root / config['checkpoint']), selection)
    rule = StrongRulePolicy(RuleConfig(**selection['strong_rule']['config']), strategies)
    fixed = FixedPolicy(selection['best_fixed_action'])
    units, records = [], []
    for user in sorted(users, key=lambda u: u['id']):
        items, _ = build_evo_memory(user)
        for topic in sorted(user['subsequent_topics'], key=lambda t: t['idx']):
            for seed in config['seeds']:
                track = tracks[_track_key(user['id'], topic['idx'], seed, config['simulator_id'])]
                for turn in config['turn_indices']:
                    context = _fixed_context_before_turn(track, turn)
                    current = normalize_space(track['seeker_turns'][turn-1])
                    state = make_evo_runtime_state(user, topic, context, current, items, turn, 'table3',
                                                  track_id=track['track_id'], fixed_open_loop=True)
                    unit = {'unit_id': state.state_id, 'ordinal': len(units), 'user_id': user['id'],
                            'topic_index': topic['idx'], 'seed': seed, 'turn_index': turn,
                            'track_id': track['track_id'], 'seeker_message': current,
                            'context_before_turn': context, 'context_sha256': sha256_text(canonical_json(context)),
                            'state_sha256': sha256_text(canonical_json(state.model_dump(mode='json'))),
                            'authorized_ground_truth': evaluator_context(user, topic)}
                    units.append(unit)
                    query = context_query(current, [h.model_dump(mode='json') for h in state.current_session_history], state.current_session_summary)
                    strategy_view = None
                    for name, policy in [('pm', pm), ('rule', rule), ('fixed', fixed)]:
                        started = time.perf_counter()
                        action = policy.choose(state)
                        policy_ms = (time.perf_counter()-started)*1000
                        sources, strategy = parse_action_id(action)
                        candidates = retriever.retrieve(query, items, sources)
                        if strategy is StrategyMode.RS and strategy_view is None:
                            strategy_view = strategies.retrieve(query)
                        selected_strategy = strategy_view if strategy is StrategyMode.RS else []
                        for enabled in (False, True):
                            started = time.perf_counter()
                            kept, decisions = filter_memory(candidates, current, query, config['filter'], enabled=enabled)
                            filter_ms = (time.perf_counter()-started)*1000
                            arm = f'{name}_{"on" if enabled else "off"}'
                            messages = generation_messages(state, kept, selected_strategy, system_prompt=SELECTIVE_ESMEM_SYSTEM)
                            records.append({
                                'call_id': f'{state.state_id}::{arm}', 'unit_id': state.state_id, 'arm': arm,
                                'requested_action_id': action,
                                'effective_action_id': canonical_action_id(frozenset(x.source for x in kept), strategy),
                                'requested_memory_sources': sorted(s.value for s in sources),
                                'requested_source_invocations': len(sources) + int(strategy is StrategyMode.RS),
                                'candidate_memory': [x.model_dump(mode='json') for x in candidates],
                                'kept_memory': [x.model_dump(mode='json') for x in kept],
                                'selected_strategy': [x.model_dump(mode='json') for x in selected_strategy],
                                'candidate_memory_ids': [x.memory_id for x in candidates],
                                'kept_memory_ids': [x.memory_id for x in kept],
                                'dropped_memory_ids': [x.memory_id for x in candidates if x not in kept],
                                'candidate_memory_tokens_est': sum(estimate_tokens(x.text) for x in candidates),
                                'kept_memory_tokens_est': sum(estimate_tokens(x.text) for x in kept),
                                'offline_policy_compute_ms': policy_ms, 'offline_filter_compute_ms': filter_ms,
                                'filter_decisions': decisions, 'messages': messages,
                                'prompt_sha256': sha256_text(canonical_json(messages)),
                                'generation_seed': seed + turn,
                            })
        print(f'Prepared {len(units)} units / {len(records)} arms', flush=True)
    return units, records


def cost_plan(units, arms, config):
    by_unit = defaultdict(list)
    for arm in arms:
        by_unit[arm['unit_id']].append(arm)
    judge = config['judge']
    stages = defaultdict(list)
    for unit in units:
        rows = by_unit[unit['unit_id']]
        # Empty strings are only prompt-size templates, never generated observations.
        qm, _ = quality_messages(unit, rows, {a: '' for a in ARMS})
        stages['quality'].append(input_token_estimate(qm, EvoResponseV4Judgment) + 6*100)
        for arm in rows:
            stages['generation'].append(input_token_estimate(arm['messages']))
            stages['risk'].append(input_token_estimate(risk_messages(unit, arm, ''), EvoSampledAuditJudgment) + 100)
            stages['omission'].append(input_token_estimate(omission_messages(unit, ''), OmissionJudgment) + 100)
    report = {}
    for stage, counts in stages.items():
        count = len(counts)
        report[stage] = {'calls': count, 'input_tokens_est': sum(counts),
                         'mean_input_tokens_est': sum(counts)/count, 'max_input_tokens_est': max(counts)}
        if stage == 'generation':
            report[stage].update({'output_tokens_cap': count*100, 'api_usd': None,
                                 'note': 'Original NVIDIA free model endpoint deprecated and absent from public catalog; account route unverified. No generator invoice price assumed.'})
            continue
        expected = count*judge[f'{stage}_expected_tokens']
        cap = count*judge[f'{stage}_max_tokens']
        usd = (sum(counts)*judge['input_usd_per_million'] + expected*judge['output_usd_per_million'])/1e6
        upper = (sum(counts)*1.1*judge['input_usd_per_million'] + cap*judge['output_usd_per_million'])/1e6
        report[stage].update({'output_tokens_expected': expected, 'output_tokens_cap': cap,
                              'standard_usd_est': usd, 'standard_usd_planning_upper': upper,
                              'batch_usd_est': usd/2})
    main = sum(r.get('standard_usd_est',0) for r in report.values())
    upper = sum(r.get('standard_usd_planning_upper',0) for r in report.values())
    # 12 units: ordinary pointwise audits plus two quality orders, all extra to full-run.
    fraction = config['pilot_units']/len(units)
    pilot = fraction*(main + report['quality']['standard_usd_est'])
    reserved = (main+pilot)*(1+config['reserve_fraction'])
    return {'stages': dict(report), 'judge_standard_main_usd_est': main,
            'judge_standard_pilot_usd_est': pilot,
            'judge_standard_total_with_reserve_usd_est': reserved,
            'judge_standard_main_planning_upper_usd': upper,
            'judge_batch_main_plus_standard_pilot_with_reserve_usd_est': (main/2+pilot)*(1+config['reserve_fraction']),
            'generator_usd': None, 'total_invoice_usd': None,
            'basis': 'o200k_base BPE of full visible prompts + estimated schema overhead. Generated replies reserve 100 GPT tokens each; Llama tokenizer differs. Expected judge output lengths are assumptions, not measurements. Upper planning figure uses configured output caps and 10% input buffer; not an invoice ceiling. No cache discount assumed.',
            'pricing_source': config['pricing_source'], 'pricing_checked_on': config['pricing_checked_on']}


def write_plan(root, config_path, out):
    config = read_json(config_path)
    baseline = verify_baseline(root, config)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new empty output directory; plans are not overwritten')
    out.mkdir(parents=True, exist_ok=True)
    units, arms = build_units_and_arms(root, config)
    expected = 34*len(config['seeds'])*len(config['turn_indices'])
    assert len(units) == expected and len(arms) == expected*6
    assert len({r['call_id'] for r in arms}) == len(arms)
    for name, rows in [('units.jsonl', units), ('arms.jsonl', arms)]:
        (out/name).write_text(''.join(canonical_json(row)+'\n' for row in rows), encoding='utf-8')
    budget = cost_plan(units, arms, config)
    write_json(out/'cost_estimate.json', budget)
    stats = {}
    for name in ARMS:
        rows = [r for r in arms if r['arm']==name]
        stats[name] = {'n':len(rows), 'actions':dict(Counter(r['requested_action_id'] for r in rows)),
                       'empty_memory_cases':sum(not r['kept_memory_ids'] for r in rows),
                       'mean_source_invocations':sum(r['requested_source_invocations'] for r in rows)/len(rows),
                       'mean_candidate_memory_tokens_est':sum(r['candidate_memory_tokens_est'] for r in rows)/len(rows),
                       'mean_kept_memory_tokens_est':sum(r['kept_memory_tokens_est'] for r in rows)/len(rows)}
    write_json(out/'retrieval_diagnostic.json', stats)
    legacy = root/'outputs/evoemo_fixed_tracks/artifact_attestation.json'
    write_json(out/'baseline_import.json', {
        'baseline_commit': config['baseline_commit'], 'verified_baseline_files': baseline,
        'legacy_track_attestation_sha256': sha256_file(legacy),
        'legacy_attestation_preserved_unmodified': True,
        'not_available': ['original supporter responses/logs', 'original raw seeker calls'],
        'scope': 'Content-verified reuse of existing fixed tracks; no claim of reproducing original API transactions',
    })
    write_json(out/'config.json', config)
    code = {str(p.relative_to(root)):sha256_file(p) for p in (root/'src/metacom_pm').glob('*.py')}
    files = {p.name:sha256_file(p) for p in out.iterdir() if p.is_file()}
    manifest = {'status':'PLANNED_NO_API', 'config':config, 'baseline_hashes':baseline,
                'code_hashes':code, 'artifact_hashes':files, 'units':len(units), 'arms':list(ARMS),
                'generation_endpoint_account_check':'PENDING', 'api_calls_made':0}
    manifest['manifest_sha256'] = sha256_text(canonical_json(manifest))
    write_json(out/'manifest.json', manifest)
    return {'output_dir':str(out), 'units':len(units), 'response_count':len(arms),
            'cost_estimate':budget, 'manifest_sha256':manifest['manifest_sha256']}
