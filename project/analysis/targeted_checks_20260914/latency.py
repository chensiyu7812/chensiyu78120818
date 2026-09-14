"""Frozen, resident-model, single-request local latency diagnostic; no API."""
import csv
import fcntl
import importlib.metadata
import os
import random
import time
from collections import Counter
from common import ROOT,CODE,OUT,read,rows,write,canonical,h,sha,verify
from metacom_pm.contracts import RuntimeState,MemoryItem,StrategyCard,parse_action_id,StrategyMode
from metacom_pm.evoemo import load_evoemo,build_evo_memory,make_evo_runtime_state,_policy_from_selection
from metacom_pm.io import iter_jsonl
from metacom_pm.policies import FixedPolicy,StrongRulePolicy,RuleConfig
from metacom_pm.table3 import PreparedStrategyRetriever
from metacom_pm.retrieval import MemoryRetriever,context_query
from metacom_pm.prompts import generation_messages,SELECTIVE_ESMEM_SYSTEM
from metacom_pm.training import PMModel

ARMS=['pm_off','rule_off','fixed_off','context_only','me_r0']
SOURCE_FILES=[
 'outputs/table3_under10/frozen_identity_v7/config.json',
 'outputs/table3_under10/local_model_v2/model_manifest.json',
 'outputs/table3_under10/local_model_v2/generation_inputs.jsonl',
 'outputs/reviewer_supplement_20260914/frozen_v2/sample_units.json',
 'outputs/reviewer_supplement_20260914/offline/development_selector_replay.jsonl',
 'outputs/table3_under10/run_identity_v7/all_results/per_response.csv',
 'outputs/table3_rerun_v1/plan/units.jsonl','outputs/table3_rerun_v1/plan/arms.jsonl',
 'outputs/final_model_m2b_stable/pm_final.joblib','outputs/selection_stable.json',
 'data/external/evo_emo.json','data/strategy/strategy_cards.jsonl',
 'data/synthetic/runtime_states.jsonl','data/synthetic/folds.jsonl',
 'outputs/synthetic_sweep/action_outcomes.jsonl',
 'outputs/full_judging_gemini_flash_lite_v3/response_pair_judgments.jsonl',
 'outputs/full_judging_gemini_flash_lite_v3/memory_omission_judgments.jsonl',
 'outputs/full_judging_gemini_flash_lite_v3/memory_use_judgments.jsonl',
 'outputs/full_judging_gemini_flash_lite_v3/strategy_omission_judgments.jsonl',
 'outputs/full_judging_gemini_flash_lite_v3/strategy_use_judgments.jsonl',
 'outputs/m2b_selected_set_omission_gemini_flash_lite_v3/memory_selected_set_omission_judgments.jsonl',
]

class Pipeline:
    def __init__(self):
        selection=read(ROOT/'outputs/selection_stable.json')
        cards=[StrategyCard.model_validate(r) for r in rows(ROOT/'data/strategy/strategy_cards.jsonl')]
        self.strategies=PreparedStrategyRetriever(cards,top_k=3)
        self.memory=MemoryRetriever()
        self.policies={'pm_off':_policy_from_selection(PMModel.load(ROOT/'outputs/final_model_m2b_stable/pm_final.joblib'),selection),
                       'rule_off':StrongRulePolicy(RuleConfig(**selection['strong_rule']['config']),self.strategies),
                       'fixed_off':FixedPolicy(selection['best_fixed_action']),
                       'context_only':FixedPolicy('M0+R0'),'me_r0':FixedPolicy('ME+R0')}

    def reset(self):
        self.strategies._last_query=None
        self.strategies._last_scores=None

    def build(self, state_dict, items, arm):
        # Static inventory/catalog/index availability is the serving boundary.
        start=time.perf_counter()
        state=RuntimeState.model_validate(state_dict)
        hydrated=time.perf_counter()
        action=self.policies[arm].choose(state)
        chosen=time.perf_counter()
        query=context_query(state.current_user_text,state.current_session_history,state.current_session_summary)
        sources,mode=parse_action_id(action)
        memories=self.memory.retrieve(query,items,sources)
        strategies=self.strategies.retrieve(query) if mode is StrategyMode.RS else []
        retrieved=time.perf_counter()
        messages=generation_messages(state,memories,strategies,system_prompt=SELECTIVE_ESMEM_SYSTEM)
        prompted=time.perf_counter()
        return messages,action,dict(start=start,prompted=prompted,
            state_hydration_ms=1000*(hydrated-start),policy_ms=1000*(chosen-hydrated),
            retrieval_ms=1000*(retrieved-chosen),prompt_build_ms=1000*(prompted-retrieved),
            source_invocations=len(sources)+int(mode is StrategyMode.RS))

def prepare():
    if (OUT/'freeze.json').exists():raise RuntimeError('Do not overwrite freeze')
    OUT.mkdir(parents=True,exist_ok=True)
    selected=read(ROOT/'outputs/reviewer_supplement_20260914/frozen_v2/sample_units.json')
    selected=sorted(selected,key=lambda r:r['unit_id'])
    assert len(selected)==48 and len({r['user_id'] for r in selected})==18
    assert sorted(Counter((r['seed'],r['turn_index']) for r in selected).values())==[8]*6
    users={u['id']:u for u in load_evoemo(ROOT/'data/external/evo_emo.json')}
    old_arms={(r['unit_id'],r['arm']):r for r in rows(ROOT/'outputs/table3_rerun_v1/plan/arms.jsonl')}
    with (ROOT/'outputs/table3_under10/run_identity_v7/all_results/per_response.csv').open(encoding='utf-8-sig') as f:
        old_responses={(r['unit_id'],r['arm']):r for r in csv.DictReader(f)}
    pipe=Pipeline();prepared=[];prompts=[]
    for unit in selected:
        u=users[unit['user_id']];items,_=build_evo_memory(u)
        topic=next(t for t in u['subsequent_topics'] if t['idx']==unit['topic_index'])
        state=make_evo_runtime_state(u,topic,unit['context_before_turn'],unit['seeker_message'],items,unit['turn_index'],'table3',track_id=unit['track_id'],fixed_open_loop=True)
        sd=state.model_dump(mode='json');assert h(sd)==unit['state_sha256']
        prepared.append(dict(unit_id=unit['unit_id'],user_id=unit['user_id'],scenario_id=old_responses[(unit['unit_id'],'pm_off')]['scenario_id'],seed=unit['seed'],turn_index=unit['turn_index'],state=sd,items=[x.model_dump(mode='json') for x in items]))
        for arm in ARMS:
            pipe.reset();messages,action,times=pipe.build(sd,items,arm)
            old=old_arms.get((unit['unit_id'],arm));response=old_responses.get((unit['unit_id'],arm))
            if old:
                assert old['messages']==messages and old['requested_action_id']==action,(unit['unit_id'],arm)
            prompts.append(dict(unit_id=unit['unit_id'],arm=arm,action=action,messages=messages,messages_sha256=h(messages),generation_seed=unit['seed']+unit['turn_index'],reference_response_sha256=h(response['response_text'].strip()) if response else None,reference_response_text=response['response_text'].strip() if response else None))
    schedule=[];rng=random.Random(20260917)
    for repeat in range(3):
        order=[x['unit_id'] for x in prepared];rng.shuffle(order)
        for unit_id in order:
            methods=ARMS.copy();rng.shuffle(methods)
            for pos,arm in enumerate(methods):
                schedule.append(dict(request_id=f'r{repeat}:{unit_id}:{arm}',repeat=repeat,unit_id=unit_id,arm=arm,position=pos,ordinal=len(schedule)))
    for name,data in [('states.jsonl',prepared),('prompts.jsonl',prompts),('schedule.jsonl',schedule)]:
        (OUT/name).write_text(''.join(canonical(r)+'\n' for r in data))
    write(OUT/'config.json',dict(n_states=48,users=18,scenarios=len({r['scenario_id'] for r in prepared}),arms=ARMS,repeats=3,n_measured_requests=720,api_usd=0,gpu='NVIDIA RTX A6000',warmup_unit=prepared[0]['unit_id'],attention='sdpa',dtype='bfloat16',decoding='greedy_natural_eos',torch_threads=4,query_cache='cleared between requests; reusable inside request',timing_scope='Resident local in-process serving; prebuilt inventory/indices; from runtime state hydration to output. Excludes load, external network/database/UI.',quality_scope='Reference quality remains tied to exact original responses; newly generated Context Only/ME+R0 responses have no new judge scores.'))
    paths=[ROOT/p for p in SOURCE_FILES]+list((ROOT/'src/metacom_pm').glob('*.py'))
    paths+=list(CODE.glob('*.py'))+[CODE/'PROTOCOL_CN.md']
    paths += [OUT/x for x in ['states.jsonl','prompts.jsonl','schedule.jsonl','config.json']]
    manifest=dict(status='FROZEN_BEFORE_TARGETED_MEASUREMENTS',files={str(p.relative_to(ROOT)):sha(p) for p in paths},packages={p:importlib.metadata.version(p) for p in ['numpy','scikit-learn','torch','transformers','tokenizers','safetensors','huggingface-hub','accelerate','jinja2','pydantic','scipy']},api_usd=0)
    manifest['freeze_sha256']=h(manifest);write(OUT/'freeze.json',manifest)
    print(canonical(dict(status='PREPARED',states=len(prepared),prompts=len(prompts),measured_requests=len(schedule),reference_prompt_matches=144,freeze_sha256=manifest['freeze_sha256'])),flush=True)

def run():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
    freeze=verify();config=read(OUT/'config.json')
    for package,version in freeze['packages'].items():assert importlib.metadata.version(package)==version,package
    dest=OUT/'latency';dest.mkdir(exist_ok=True)
    with (dest/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pid=os.getpid()
        def status(stage,**extra):
            write(dest/'status.json',dict(stage=stage,pid=pid,freeze_sha256=freeze['freeze_sha256'],**extra))
        status('VERIFYING_MODEL')
        asset=read(ROOT/'outputs/table3_under10/local_model_v2/model_manifest.json')
        assert h({k:v for k,v in asset.items() if k!='asset_sha256'})==asset['asset_sha256']
        model_dir=ROOT/asset['model_directory']
        for name,expected in asset['model_files_sha256'].items():assert sha(model_dir/name)==expected,name
        assert torch.cuda.get_device_name(0)==config['gpu']
        torch.set_num_threads(config['torch_threads'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa').eval()
        pipe=Pipeline()
        state_rows={r['unit_id']:r for r in rows(OUT/'states.jsonl')}
        items={u:[MemoryItem.model_validate(x) for x in r['items']] for u,r in state_rows.items()}
        prompt_rows={(r['unit_id'],r['arm']):r for r in rows(OUT/'prompts.jsonl')}
        schedule=rows(OUT/'schedule.jsonl')
        output=dest/'requests.jsonl'
        existing=rows(output) if output.exists() else []
        assert len(existing)==len({r['request_id'] for r in existing})
        assert all(r['freeze_sha256']==freeze['freeze_sha256'] and r['natural_eos'] for r in existing)
        done={r['request_id'] for r in existing}

        class ClockStreamer:
            def __init__(self):self.prompt=True;self.first_token=None;self.first_text=None;self.ids=[]
            def put(self,value):
                if self.prompt:self.prompt=False;return
                now=time.perf_counter()
                if self.first_token is None:self.first_token=now
                self.ids.extend(value.reshape(-1).tolist())
                if self.first_text is None:
                    if tokenizer.decode(self.ids,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip():self.first_text=time.perf_counter()
            def end(self):pass

        def measure(job,warmup=False):
            sr=state_rows[job['unit_id']];pr=prompt_rows[(job['unit_id'],job['arm'])]
            torch.manual_seed(pr['generation_seed']);torch.cuda.manual_seed_all(pr['generation_seed'])
            pipe.reset();torch.cuda.synchronize()
            messages,action,t=pipe.build(sr['state'],items[job['unit_id']],job['arm'])
            rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=tokenizer(rendered,add_special_tokens=False,return_tensors='pt').to(model.device)
            torch.cuda.synchronize();tokenized=time.perf_counter()
            n=inputs['input_ids'].shape[1]
            generation=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=model.config.max_position_embeddings-n,bos_token_id=128000,eos_token_id=asset['eos_token_ids'],pad_token_id=128001,repetition_penalty=1.0,use_cache=True)
            stream=ClockStreamer()
            gen_start=time.perf_counter()
            with torch.inference_mode():out=model.generate(**inputs,generation_config=generation,streamer=stream)
            torch.cuda.synchronize();gen_done=time.perf_counter()
            new=out[0,n:].tolist();text=tokenizer.decode(new,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
            ended=time.perf_counter()
            # All integrity checks and disk writes are after the measured region.
            assert action==pr['action'] and h(messages)==pr['messages_sha256']
            assert new==stream.ids and new and new[-1] in asset['eos_token_ids'] and text
            assert stream.first_token is not None and stream.first_text is not None
            assert t['start']<=t['prompted']<=tokenized<=stream.first_token<=stream.first_text<=ended
            result=dict(**job,user_id=sr['user_id'],scenario_id=sr['scenario_id'],seed=sr['seed'],turn_index=sr['turn_index'],action=action,warmup=warmup,
                freeze_sha256=freeze['freeze_sha256'],natural_eos=True,response_text=text,response_sha256=h(text),
                reference_response_matches=(h(text)==pr['reference_response_sha256']) if pr['reference_response_sha256'] else None,
                messages_sha256=h(messages),input_ids_sha256=h(inputs['input_ids'][0].tolist()),input_tokens=n,output_tokens=len(new),completion_token_ids=new,
                local_first_token_ms=1000*(stream.first_token-t['start']),local_first_text_ms=1000*(stream.first_text-t['start']),local_complete_ms=1000*(ended-t['start']),
                tokenization_ms=1000*(tokenized-t['prompted']),generation_setup_ms=1000*(gen_start-tokenized),generation_ms=1000*(gen_done-gen_start),final_decode_ms=1000*(ended-gen_done),
                generator_first_token_ms=1000*(stream.first_token-gen_start),**{k:v for k,v in t.items() if k not in ['start','prompted']},api_usd=0)
            return result

        status('WARMUP',completed=len(done),expected=720)
        warm=[]
        for arm in ARMS:
            warm.append(measure(dict(request_id='warmup:'+arm,repeat=-1,unit_id=config['warmup_unit'],arm=arm,position=ARMS.index(arm),ordinal=-1),True))
        warm_path=dest/f'warmup_{pid}.jsonl';warm_path.write_text(''.join(canonical(r)+'\n' for r in warm))
        print(canonical({'stage':'WARMUP_COMPLETE','pid':pid,'requests':5}),flush=True)
        with output.open('a') as handle:
            for job in schedule:
                if job['request_id'] in done:continue
                status('RUNNING',completed=len(done),expected=720,current_request=job['request_id'])
                try:result=measure(job)
                except Exception as exc:
                    status('FAILED',completed=len(done),expected=720,request=job,type=type(exc).__name__,error=str(exc));raise
                handle.write(canonical(result)+'\n');handle.flush();os.fsync(handle.fileno());done.add(job['request_id'])
                if len(done)%20==0:print(canonical({'stage':'RUNNING','completed':len(done),'expected':720,'last_seconds':result['local_complete_ms']/1000}),flush=True)
        status('COMPLETE',completed=len(done),expected=720,api_usd=0)
        print(canonical({'stage':'COMPLETE','completed':len(done),'api_usd':0}),flush=True)

if __name__=='__main__':
    import sys
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='run':run()
    elif sys.argv[1]=='verify':print(verify()['freeze_sha256'])
    else:raise ValueError('prepare | run | verify')
