"""Run ordinary summary prompts on the endpoint pilot; manually judge outcomes."""
import argparse
import csv
import gc
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from receiver_kv_workflow.cli import add_workflow_arguments, model_config
from receiver_kv_workflow.engine import ReceiverKVRunner
from receiver_kv_workflow.prompts import prompt_parts
from receiver_kv_workflow.source_mapping import source_offsets
from meeting_evaluation.records import load_records
from experiment_1.components.config import sha256
from experiment_1.components.storage import atomic_json
from experiment_2.components.generation import DirectSummaryRunner


def coverage(tokenizer,transcript,selected,key,windows):
    prefix,_,protected=prompt_parts(tokenizer,transcript)
    offsets,origin=source_offsets(tokenizer,transcript,prefix)
    selected=set(selected);protected=set(protected)
    targets={}
    result={}
    for label in ('opening','closing'):
        start,end=key[label]['character_range']
        ids={i for i,(a,b) in enumerate(offsets) if i not in protected and a<origin+end and b>origin+start}
        kept=ids & selected
        targets[label]=ids
        result[label]=dict(source_token_count=len(ids),retained_token_count=len(kept),
            fraction=len(kept)/len(ids) if ids else None,fully_retained=bool(ids) and ids<=selected,
            local_windows=[i for i,w in enumerate(windows) if ids and ids<=set(w['positions'])])
    result['both_in_same_local_window']=any((targets['opening']|targets['closing'])<=set(w['positions']) for w in windows)
    return result


def run(args):
    records=load_records('jsonl',args.dataset/'meetings.jsonl')
    definition=json.loads((args.dataset/'manifest.json').read_text(encoding='utf-8'))
    if definition['meetings_sha256']!=sha256((args.dataset/'meetings.jsonl').read_bytes()):
        raise ValueError('Challenge source changed after preparation; freeze a new version instead')
    if definition['answer_key_sha256']!=sha256((args.dataset/'answer_key.json').read_bytes()):
        raise ValueError('Challenge answer key changed after preparation')
    config=replace(model_config(args),vector_routing=args.condition=='semantic_kv')
    folder=args.output_dir/datetime.now().strftime('%Y%m%d_%H%M%S_%f')/args.condition
    folder.mkdir(parents=True,exist_ok=False)
    atomic_json(folder/'manifest.json',dict(condition=args.condition,config=asdict(config),dataset=definition,
        sample_ids=[r.id for r in records],note='Diagnostic only: no ROUGE against the original unmodified reference. Manual target judgments remain blank.'))
    factory=DirectSummaryRunner if args.condition=='direct_text' else ReceiverKVRunner
    runner=factory(config,progress=lambda s:print(s,flush=True))
    reviews=[];completed=0
    for index,record in enumerate(records):
        case=folder/f'{index:05d}_{record.id}';case.mkdir()
        def live(event):
            stage=event.get('stage','direct_summary')
            (case/f'{stage}.partial.txt').write_text(event['text'],encoding='utf-8')
        try:
            # Only transcript + ordinary shared requirements enter these calls.
            # The answer key is read afterwards for offline coverage/manual review.
            if args.condition=='direct_text':
                def progress(event):
                    count=event.get('tokens',0)
                    if count and (count%256==0 or event.get('finished')):
                        print(f'direct_summary: {count} tokens',flush=True)
                result=runner.run_direct(record,on_generation=live,on_progress=progress)
                answer=result['answers']['direct_text']
            else:
                result=runner.run(record.id,record.context,variants=('chunk_kv',),routing_audit=True,on_generation=live)
                answer=result['answers']['chunk_kv']
            (case/'summary.txt').write_text(answer+'\n',encoding='utf-8')
            key=json.loads((args.dataset/'answer_key.json').read_text(encoding='utf-8'))[record.id]
            if sha256(record.context.encode())!=key['modified_source_sha256']:
                raise ValueError('Transcript does not match the fixed answer-key coordinates')
            if args.condition=='direct_text':
                retained=dict(applicable=False,note='Full original text is visible; no evidence selector runs.')
            else:
                detail=result['chunk_specialists_by_variant']['chunk_kv']
                retained=coverage(runner.tokenizer,record.context,result['selected_source_positions'],key,detail['windows'])
                retained.update(applicable=True,selected_positions=len(result['selected_source_positions']),
                    source_positions=result['source_positions'],retained_fraction=len(result['selected_source_positions'])/result['source_positions'])
                atomic_json(case/'specialists.json',detail)
            atomic_json(case/'coverage.json',retained)
            # Explicit KV-row coverage does not measure every possible influence
            # of earlier source context encoded in the retained vectors.
            atomic_json(case/'result.json',result)
            reviews.append(dict(sample_id=record.id,condition=args.condition,expected=key['expected'],
                incorrect=key['incorrect'],opening_retained=retained.get('opening',{}).get('fully_retained','N/A'),
                closing_retained=retained.get('closing',{}).get('fully_retained','N/A'),
                summary_path=str(case/'summary.txt'),verdict='',notes=''))
            completed+=1
            print(f'{index+1}/{len(records)} {record.id}: completed',flush=True)
        except Exception as exc:
            atomic_json(case/'failure.json',dict(error_type=type(exc).__name__,error=str(exc)))
            print(f'{record.id}: {type(exc).__name__}: {exc}',flush=True)
            gc.collect()
            if runner.device.type=='cuda':runner.torch.cuda.empty_cache()
    with (folder/'manual_review.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['sample_id','condition','expected','incorrect','opening_retained',
            'closing_retained','summary_path','verdict','notes'])
        writer.writeheader();writer.writerows(reviews)
    atomic_json(folder/'status.json',dict(planned=len(records),completed=completed,failed=len(records)-completed))
    print('Results:',folder)
    return folder


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    add_workflow_arguments(parser)
    parser.add_argument('--dataset',type=Path,default=Path('input/ami_endpoint_challenge_3'))
    parser.add_argument('--condition',choices=('direct_text','attention_kv','semantic_kv'),required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('test_result/ami_endpoint_challenge_3'))
    run(parser.parse_args())
