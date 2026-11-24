import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiment_2.challenge_samples.prepare import create
from experiment_2.challenge_samples.run import coverage
from experiment_2.challenge_samples.run import run
from receiver_kv_workflow.engine import ReceiverKVConfig


class EndpointPilotTest(unittest.TestCase):
    def test_all_conditions_use_only_transcripts_and_leave_judgments_blank(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source=root/'source.jsonl'
            source.write_text(''.join(json.dumps(dict(id=str(i),transcript=f'Project Manager: Original {i}.',summary='Old reference'))+'\n' for i in range(3)))
            dataset=create(root/'data',input_path=source)
            seen=[];configs=[]
            class Fake:
                def __init__(self,config,**kwargs):
                    configs.append(config.vector_routing)
                    self.tokenizer=None
                def run_direct(self,record,**kwargs):
                    assert record.answer==''
                    seen.append(record.context)
                    return dict(answers={'direct_text':'Summary without automatic target judgment'})
                def run(self,sid,text,**kwargs):
                    seen.append(text)
                    assert kwargs['variants']==('chunk_kv',)
                    return dict(answers={'chunk_kv':'Summary without automatic target judgment'},
                        selected_source_positions=[0,1],source_positions=10,
                        chunk_specialists_by_variant={'chunk_kv':dict(windows=[])})
            for condition in ('direct_text','attention_kv','semantic_kv'):
                args=SimpleNamespace(dataset=dataset,output_dir=root/'results',condition=condition)
                with patch('experiment_2.challenge_samples.run.model_config',return_value=ReceiverKVConfig(device='cpu')), \
                     patch('experiment_2.challenge_samples.run.DirectSummaryRunner',Fake), \
                     patch('experiment_2.challenge_samples.run.ReceiverKVRunner',Fake), \
                     patch('experiment_2.challenge_samples.run.coverage',return_value={}):
                    output=run(args)
                self.assertEqual(json.loads((output/'status.json').read_text())['completed'],3)
                import csv
                with (output/'manual_review.csv').open(encoding='utf-8-sig') as handle:
                    rows=list(csv.DictReader(handle))
                self.assertEqual(len(rows),3)
                self.assertTrue(all(r['verdict']=='' for r in rows))
            self.assertEqual(configs,[False,False,True])
            self.assertEqual(len(seen),9)

    def test_original_text_preserved_exactly_and_answers_kept_out_of_input(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source.jsonl'
            originals=[dict(id=str(i),transcript=f'Project Manager: Original {i}.\nMarketing: End {i}.',summary='Old summary') for i in range(3)]
            source.write_text(''.join(json.dumps(r)+'\n' for r in originals),encoding='utf-8')
            folder=create(Path(temp)/'pilot',input_path=source)
            rows=[json.loads(line) for line in (folder/'meetings.jsonl').read_text().splitlines()]
            keys=json.loads((folder/'answer_key.json').read_text())
            self.assertEqual(len(rows),3)
            for row,original in zip(rows,originals):
                self.assertEqual(set(row),{'id','transcript'})
                key=keys[row['id']]
                self.assertEqual(row['transcript'],key['opening']['text']+'\n'+original['transcript']+'\n'+key['closing']['text'])
                for label in ('opening','closing'):
                    a,b=key[label]['character_range']
                    self.assertEqual(row['transcript'][a:b],key[label]['text'])
                self.assertNotIn(key['expected'],row['transcript'])
            with self.assertRaises(FileExistsError):
                create(folder,input_path=source)

    def test_coverage_distinguishes_dropped_revision_and_same_window(self):
        text='first\nfiller\nlast'
        key=dict(opening=dict(character_range=[0,5]),closing=dict(character_range=[13,17]))
        offsets=[(i,i+1) for i in range(len(text))]
        with patch('experiment_2.challenge_samples.run.prompt_parts',return_value=([],[],[])), \
             patch('experiment_2.challenge_samples.run.source_offsets',return_value=(offsets,0)):
            result=coverage(None,text,list(range(5)),key,[dict(positions=list(range(5)))])
            self.assertTrue(result['opening']['fully_retained'])
            self.assertFalse(result['closing']['fully_retained'])
            self.assertEqual(result['closing']['fraction'],0.)
            result=coverage(None,text,list(range(17)),key,[dict(positions=list(range(17)))])
            self.assertTrue(result['closing']['fully_retained'])
            self.assertTrue(result['both_in_same_local_window'])
