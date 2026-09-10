import copy
import json

from metacom_pm.contracts import MemoryItem, MemorySource
from metacom_pm.table3 import (ARMS, filter_memory, omission_messages, quality_messages,
                              risk_messages, verify_baseline)
from pathlib import Path
import pytest


CONFIG = json.loads((Path(__file__).parents[1]/'configs/table3_rerun.json').read_text())


def memory(identifier, text, source=MemorySource.MP):
    return MemoryItem(memory_id=f'mem_{ord(identifier):020x}', source=source, created_session=1, text=text)


def test_off_preserves_exact_items_order_and_duplicates():
    candidates = [memory('a','different topic'),memory('b','different topic')]
    kept, rows = filter_memory(candidates,'work','work',CONFIG['filter'],enabled=False)
    assert kept == candidates
    assert all(r['keep'] for r in rows)


def test_gate_allows_empty_and_logs_reasons():
    kept, rows = filter_memory([memory('a','family')],'work','work',CONFIG['filter'],enabled=True)
    assert kept == []
    assert rows[0]['reason'] == 'below_relevance_threshold'


def test_dedup_and_cap_keep_subset_in_original_order():
    config = copy.deepcopy(CONFIG['filter'])
    config['maximum_total_memory_items'] = 2
    candidates = [memory('z','work stress'), memory('a','work stress'), memory('b','work')]
    kept, rows = filter_memory(candidates,'work','work',config,enabled=True)
    assert kept == candidates[1:]
    assert rows[0]['reason'] == 'exact_duplicate_evidence'


def test_long_me_cannot_pass_on_weak_current_match():
    text = ('work work family ' * 200)
    kept, rows = filter_memory([memory('a',text,MemorySource.ME)],'work other more','work family',CONFIG['filter'],enabled=True)
    assert kept == []
    assert rows[0]['reason'] == 'long_item_requires_high_current_relevance'


def test_quality_and_omission_do_not_expose_arm_evidence():
    unit = {'unit_id':'anonymous','ordinal':0,'turn_index':3,'context_before_turn':[],
            'seeker_message':'work stress','authorized_ground_truth':{'fact':'authorized fact'}}
    msgs,mapping = quality_messages(unit,[],{a:'support' for a in ARMS})
    body=json.loads(msgs[1]['content'])
    assert len(body['candidates'])==6
    assert set(body['candidates'][0])=={'candidate_id','response'}
    assert 'pm_on' not in json.dumps(msgs)
    assert mapping['C1']['condition']=='pm_off'
    omit=omission_messages(unit,'support')
    assert 'authorized fact' in json.dumps(omit)
    assert 'selected_memory' not in json.dumps(omit)


def test_risk_sees_only_kept_evidence_and_opaque_call_id():
    unit={'seeker_message':'work','context_before_turn':[]}
    arm={'call_id':'state::pm_on','kept_memory':[{'text':'kept'}],
         'candidate_memory':[{'text':'dropped_secret'}],'selected_strategy':[]}
    messages=risk_messages(unit,arm,'support')
    text=json.dumps(messages)
    assert 'kept' in text and 'dropped_secret' not in text and 'pm_on' not in text


def test_source_rule_contract_requires_strategy_passthrough():
    config=copy.deepcopy(CONFIG['filter'])
    config['strategy_passthrough']=False
    with pytest.raises(ValueError):filter_memory([],'','',config,enabled=True)


def test_baseline_verifier_rejects_content_drift(tmp_path):
    root=tmp_path
    (root/'outputs').mkdir()
    (root/'outputs/study_freeze_stable.json').write_text(json.dumps({
        'checkpoint_hashes':{'model':'wrong'},'data_hashes':{},'code_hashes':{},'prompt_hashes':{}}))
    (root/'model').write_text('changed checkpoint')
    with pytest.raises(ValueError,match='Baseline mismatch'):
        verify_baseline(root,{'checkpoint':'model','selection':'selection'})


def test_prepared_strategy_retriever_is_exactly_v1_including_ties_and_empty_query():
    from metacom_pm.retrieval import StrategyRetriever
    from metacom_pm.table3 import PreparedStrategyRetriever
    from types import SimpleNamespace
    cards=[SimpleNamespace(strategy_id=i,retrieval_text=t) for i,t in
           [('a','work family'),('b','work family'),('c','work work'),('d','unrelated')]]
    original,prepared=StrategyRetriever(cards,3),PreparedStrategyRetriever(cards,3)
    for query in ['work','family work work','',"I'm stressed about work",'work']:
        assert prepared.retrieve(query)==original.retrieve(query)
        assert prepared.confidence(query)==original.confidence(query)
