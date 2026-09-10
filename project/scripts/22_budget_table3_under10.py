"""Reprice the frozen six-arm plan without making API calls or changing samples."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from metacom_pm.io import canonical_json, iter_jsonl, read_json, sha256_file, sha256_text, write_json
from metacom_pm.table3 import (
    ARMS, EvoResponseV4Judgment, EvoSampledAuditJudgment, OmissionJudgment,
    input_token_estimate, omission_messages, quality_messages, risk_messages,
)


def fixed_pilot(units, selection_seed):
    """Two units per seed/turn cell, covering twelve different users."""
    cells = defaultdict(list)
    for unit in units:
        cells[(unit['seed'], unit['turn_index'])].append(unit)
    selected, users = [], set()
    for cell in sorted(cells):
        ranked = sorted(cells[cell], key=lambda u: sha256_text(f"{selection_seed}:{u['unit_id']}"))
        added = 0
        for unit in ranked:
            if unit['user_id'] in users:
                continue
            selected.append(unit)
            users.add(unit['user_id'])
            added += 1
            if added == 2:
                break
        if added != 2:
            raise ValueError(f'Cannot select two distinct users in {cell}')
    if len(selected) != 12:
        raise ValueError('This budget requires six cells and twelve pilot units')
    return selected


def count_prompts(units, by_unit, quality_orders):
    counts = defaultdict(list)
    for unit in units:
        rows = by_unit[unit['unit_id']]
        for variant in range(quality_orders):
            messages, _ = quality_messages(unit, rows, {arm: '' for arm in ARMS}, variant=variant)
            counts['quality'].append(input_token_estimate(messages, EvoResponseV4Judgment) + 600)
        for arm in rows:
            counts['risk'].append(input_token_estimate(risk_messages(unit, arm, ''), EvoSampledAuditJudgment) + 100)
            counts['omission'].append(input_token_estimate(omission_messages(unit, ''), OmissionJudgment) + 100)
    return {stage: {'calls': len(values), 'input_tokens_est': sum(values)} for stage, values in counts.items()}


def priced(counts, judge, multiplier):
    stages = {}
    for stage, tokens in counts.items():
        n, inp = tokens['calls'], tokens['input_tokens_est']
        expected, cap = n * judge[f'{stage}_expected_tokens'], n * judge[f'{stage}_max_tokens']
        stages[stage] = {
            **tokens, 'output_tokens_expected': expected, 'output_tokens_cap': cap,
            'usd_expected': multiplier * (inp * judge['input_usd_per_million'] + expected * judge['output_usd_per_million']) / 1e6,
            'usd_output_caps_input_plus_10pct': multiplier * (1.1 * inp * judge['input_usd_per_million'] + cap * judge['output_usd_per_million']) / 1e6,
        }
    return {
        'model': judge['model'], 'billing_multiplier': multiplier, 'stages': stages,
        'usd_expected': sum(s['usd_expected'] for s in stages.values()),
        'usd_output_caps_input_plus_10pct': sum(s['usd_output_caps_input_plus_10pct'] for s in stages.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('outputs/table3_rerun_v1/plan'))
    parser.add_argument('--out', type=Path, default=Path('outputs/table3_under10/plan'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan, out = root / args.plan, root / args.out
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new empty output directory; previous budgets are preserved')
    manifest = read_json(plan / 'manifest.json')
    unsigned = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    if sha256_text(canonical_json(unsigned)) != manifest['manifest_sha256']:
        raise ValueError('Source plan manifest signature differs')
    for category, directory in [('artifact_hashes', plan), ('code_hashes', root), ('baseline_hashes', root)]:
        for name, expected in manifest[category].items():
            if sha256_file(directory / name) != expected:
                raise ValueError(f'Source plan verification failed: {name}')
    original = read_json(plan / 'config.json')
    units, arms = list(iter_jsonl(plan / 'units.jsonl')), list(iter_jsonl(plan / 'arms.jsonl'))
    by_unit = defaultdict(list)
    for arm in arms:
        by_unit[arm['unit_id']].append(arm)
    if len(units) != 204 or len(arms) != 1224 or any(Counter(a['arm'] for a in by_unit[u['unit_id']]) != Counter(ARMS) for u in units):
        raise ValueError('Expected all 204 paired six-arm units')

    config = deepcopy(original)
    config['protocol'] = 'pm-v1-table3-memory-filter-diagnostic-mini-batch-v1'
    config['generator'].update({'api_usd_assumed': 0, 'cost_basis': 'User confirms NVIDIA Llama generation is free; no NVIDIA billing included.'})
    config['judge'].update({'model': 'gpt-4.1-mini-2025-04-14', 'input_usd_per_million': 0.4, 'output_usd_per_million': 1.6, 'main_transport': 'batch', 'batch_billing_multiplier': 0.5})
    config['pilot'] = {
        'units': 12, 'selection_seed': 20260910,
        'selection_rule': 'Seed/turn strata sorted; choose two distinct users per cell in SHA256(seed:unit_id) order, excluding previously chosen users. No outcome or price selection.',
        'quality_orders_per_judge': 2, 'risk_and_omission_per_arm': True,
        'main_judge_transport': 'synchronous', 'reference_judge_transport': 'synchronous',
        'reference_judge': original['judge'],
        'reference_scores_in_main_table': False,
        'purpose': 'Technical and rubric cross-check only; twelve units do not establish equivalence to GPT-4o or human raters.',
    }
    config['budget'] = {
        'user_total_usd_exclusive_limit': 10.0, 'execution_reservation_limit_usd': 9.0,
        'reserve_fraction': 0.25, 'automatic_model_upgrade': False,
        'enforcement_status': 'REQUIRED_IN_FUTURE_RUNNER_NOT_IMPLEMENTED',
        'submission_rule': 'Before every submission or retry: charged usage + unresolved/queued reservations + new request output-cap reservation must be <=9 USD. Recount actual generated text first. Do not assume usage-unknown failures cost zero. Stop before submitting work that exceeds the limit.',
    }
    config['pricing_sources'] = [
        'https://developers.openai.com/api/docs/models/gpt-4.1-mini',
        'https://developers.openai.com/api/docs/models/gpt-4o',
        'https://developers.openai.com/api/docs/guides/batch',
    ]
    config.pop('pricing_source')
    pilot = fixed_pilot(units, config['pilot']['selection_seed'])
    main_counts, pilot_counts = count_prompts(units, by_unit, 1), count_prompts(pilot, by_unit, 2)
    # Stored JSONL sorts nested keys. V4 quality serialization preserves their
    # insertion order, so reloading changes BPE boundaries, not prompt content.
    # Risk/omission builders already canonicalize JSON and must match exactly.
    old_cost = read_json(plan / 'cost_estimate.json')
    token_reconciliation = {}
    for stage, counts in main_counts.items():
        for key in ('calls', 'input_tokens_est'):
            if (key == 'calls' or stage != 'quality') and counts[key] != old_cost['stages'][stage][key]:
                raise ValueError(f'Original prompt count changed: {stage}/{key}')
        token_reconciliation[stage] = {
            'original_in_memory_plan_input_tokens_est': old_cost['stages'][stage]['input_tokens_est'],
            'reloaded_frozen_plan_input_tokens_est': counts['input_tokens_est'],
            'delta': counts['input_tokens_est'] - old_cost['stages'][stage]['input_tokens_est'],
        }
    workloads = {
        'main_mini_batch': priced(main_counts, config['judge'], 0.5),
        'pilot_mini_sync': priced(pilot_counts, config['judge'], 1),
        'pilot_gpt4o_sync': priced(pilot_counts, original['judge'], 1),
    }
    expected = sum(w['usd_expected'] for w in workloads.values())
    conservative = sum(w['usd_output_caps_input_plus_10pct'] for w in workloads.values())
    reserve = expected * config['budget']['reserve_fraction']
    report = {
        'status': 'PLANNED_NO_API', 'api_calls_made': 0, 'units': 204, 'arms_per_unit': 6,
        'generation_main_calls': 1224, 'generation_extra_pilot_calls_budgeted': 72, 'generator_usd_assumed': 0,
        'pilot_units': len(pilot), 'pilot_distinct_users': len({u['user_id'] for u in pilot}),
        'workloads': workloads, 'total_usd_expected': expected, 'reserve_usd': reserve,
        'token_reconciliation': token_reconciliation,
        'serialization_note': 'Quality prompt uses unsorted json.dumps; sorted-key JSONL reload changes nested key ordering and BPE boundaries. Costs here use reloaded frozen inputs. Risk and omission counts match original exactly.',
        'total_usd_with_25pct_reserve': expected + reserve,
        'total_usd_output_caps_input_plus_10pct': conservative,
        'execution_reservation_limit_usd': 9.0,
        'fits_planning_limit': max(expected + reserve, conservative) < 9.0,
        'basis': 'Actual frozen prompts with BPE/schema overhead estimates and 100 GPT reply-token allowance per not-yet-generated response. Expected output lengths are assumptions. Output-cap/10%-input figure is a planning scenario, not a guaranteed invoice cap. No cache discounts. All pilot calls budgeted as extra to full run.',
        'limitation': 'Budget reservation and paid execution are not implemented by this offline planner. An enforced budget can halt an incomplete run; it cannot promise completion regardless of failures.',
        'pricing_checked_on': config['pricing_checked_on'], 'pricing_sources': config['pricing_sources'],
    }
    if not report['fits_planning_limit']:
        raise ValueError(f'Plan exceeds the 9 USD planning limit: {report}')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'config.json', config)
    write_json(out / 'budget.json', report)
    pilot_metadata = [{k: u[k] for k in ('unit_id', 'user_id', 'seed', 'turn_index', 'ordinal')} for u in pilot]
    write_json(out / 'pilot_units.json', pilot_metadata)
    write_json(out / 'manifest.json', {
        'status': 'PLANNED_NO_API', 'api_calls_made': 0,
        'source_plan': str(plan.relative_to(root)), 'source_manifest_sha256': manifest['manifest_sha256'],
        'source_manifest_file_sha256': sha256_file(plan / 'manifest.json'),
        'planner_sha256': sha256_file(Path(__file__)),
        'artifact_hashes': {name: sha256_file(out / name) for name in ('config.json', 'budget.json', 'pilot_units.json')},
        'design_change': 'Full-run judge changed from GPT-4o to GPT-4.1 mini, using Batch. Frozen generation inputs, six arms, sample, filters and three scoring prompts unchanged.',
    })
    print(canonical_json(report))


if __name__ == '__main__':
    main()
