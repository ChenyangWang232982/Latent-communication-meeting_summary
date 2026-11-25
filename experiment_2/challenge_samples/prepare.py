"""Keep each source meeting intact; add exactly one opening and one closing turn."""
import argparse
import json
from pathlib import Path

from experiment_1.components.config import sha256
from experiment_1.components.storage import atomic_json
from meeting_evaluation.records import load_records

ADDITIONS = [
    dict(kind='revised_configuration',
        opening='Marketing: One small point before the main agenda: for the six-unit loan batch, my proposal is to ship disposable cells and no dock. That is only a proposal for that batch, not a decision about the main product.',
        closing='Project Manager: Before we leave, the loan units will go out with the charging dock instead; that supersedes the cell-only arrangement mentioned before we started. The team has settled on this configuration for that batch.',
        expected='For the six-unit loan batch, the agreed configuration includes a charging dock; the earlier disposable-cell/no-dock proposal was superseded. This is not a change to the main product specification.',
        incorrect='The loan batch is finally approved with disposable cells and no dock, remains undecided, or the batch-specific change is asserted as a new main-product decision.',
        scope='six-unit loan batch', anchors=['six-unit loan batch','the loan units']),
    dict(kind='reassigned_owner',
        opening='Marketing: Before the design discussion, I will put together the return kit for the pair of evaluation units. I am taking responsibility for that small kit, separate from the main design deliverables.',
        closing='Industrial Designer: As agreed, I am taking over putting that pair\'s return kit together. Marketing is no longer the owner of this task; I will handle it. This handover does not reassign the main design work.',
        expected='The Industrial Designer is the final owner of preparing the return kit for the pair of evaluation units, replacing Marketing. Main design responsibilities are unchanged.',
        incorrect='Marketing is still the final owner, Marketing and the Industrial Designer are joint owners, or main design responsibilities are reassigned.',
        scope='return kit for the pair of evaluation units', anchors=['the pair of evaluation units',"that pair's return kit"]),
    dict(kind='unmet_authorization_condition',
        opening='User Interface: A minor arrangement before we start: the three-person hallway trial can proceed only after all three volunteers return signed consent forms. No date or authorization is confirmed yet.',
        closing='Project Manager: For those three hallway volunteers, the signed forms still have not arrived. The earlier gate therefore stays closed: we have not authorized the session, and its date remains unconfirmed.',
        expected='The three-person hallway trial is not authorized and has no confirmed date because the required signed consent forms are still missing.',
        incorrect='The trial is approved or scheduled, consent has arrived, or a date is invented.',
        scope='three-person hallway trial', anchors=['three-person hallway trial','those three hallway volunteers']),
]


def create(output_dir, *, input_path=None, source_run=None, additions=None, skip=0,
           version='ami_endpoint_pilot_v1'):
    additions = ADDITIONS if additions is None else additions
    count = len(additions)
    if input_path is not None:
        records=load_records('jsonl',Path(input_path))[skip:skip+count]
        originals=[dict(id=r.id,source=r.context,reference=r.answer) for r in records]
        provenance=dict(input_file=str(input_path),input_sha256=sha256(Path(input_path).read_bytes()))
    else:
        source_run=Path(source_run)
        manifest=json.loads((source_run/'manifest.json').read_text(encoding='utf-8'))
        originals=[]
        for i in range(skip, min(skip+count, len(manifest['sample_ids']))):
            sid=manifest['sample_ids'][i]
            sample=json.loads((source_run/'samples'/f'{i:05d}.json').read_text(encoding='utf-8'))
            if sample['id']!=sid:
                raise ValueError('Source checkpoint ID does not match its manifest')
            originals.append(dict(id=sid,source=sample['source'],reference=sample['reference']))
        provenance=dict(source_run=str(source_run),manifest_sha256=sha256((source_run/'manifest.json').read_bytes()))
    if len(originals)!=count:
        raise ValueError(f'{count} source meetings are required after skipping {skip}')
    folder=Path(output_dir)
    folder.mkdir(parents=True,exist_ok=False)
    rows=[];keys={}
    for index,(original,addition) in enumerate(zip(originals,additions)):
        if any(anchor.lower() in original['source'].lower() for anchor in addition['anchors']):
            raise ValueError('Inserted topic collides with original text; choose another source before freezing the test')
        transcript=addition['opening']+'\n'+original['source']+'\n'+addition['closing']
        sample_id=f'ami_{original["id"]}_endpoint'
        rows.append(dict(id=sample_id,transcript=transcript))
        closing_start=len(addition['opening'])+1+len(original['source'])+1
        keys[sample_id]=dict(source_id=original['id'],category=addition['kind'],scope=addition['scope'],
            original_source_sha256=sha256(original['source'].encode()),modified_source_sha256=sha256(transcript.encode()),
            original_reference=original['reference'],expected=addition['expected'],incorrect=addition['incorrect'],
            opening=dict(text=addition['opening'],character_range=[0,len(addition['opening'])]),
            closing=dict(text=addition['closing'],character_range=[closing_start,len(transcript)]))
        (folder/f'{index:05d}_{sample_id}.txt').write_text(transcript,encoding='utf-8')
    source=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    (folder/'meetings.jsonl').write_text(source,encoding='utf-8')
    atomic_json(folder/'answer_key.json',keys)
    atomic_json(folder/'manifest.json',dict(version=version,records=count,skip=skip,provenance=provenance,
        meetings_sha256=sha256((folder/'meetings.jsonl').read_bytes()),answer_key_sha256=sha256((folder/'answer_key.json').read_bytes()),
        note='Authored endpoint-only additions to real AMI meetings; not authentic new meeting facts. No reference summary or answer key is supplied to inference.'))
    lines=['# AMI endpoint challenge: manual answer key','',
        'Synthetic additions only. Original meeting text is unchanged. Do not pass this file to the model.','']
    for sid,key in keys.items():
        lines.extend([f'## {sid}',f'Category: {key["category"]}',f'Original AMI ID: {key["source_id"]}',
            '',f'Opening: {key["opening"]["text"]}','',f'Closing: {key["closing"]["text"]}',
            '',f'Expected: {key["expected"]}','',f'Incorrect: {key["incorrect"]}',''])
    (folder/'ANSWER_KEY.md').write_text('\n'.join(lines),encoding='utf-8')
    return folder


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input',type=Path)
    source.add_argument('--source-run',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('input/ami_endpoint_challenge_3'))
    args=parser.parse_args()
    print('Prepared:',create(args.output_dir,input_path=args.input,source_run=args.source_run))
