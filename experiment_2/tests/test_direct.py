import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from meeting_evaluation.records import MeetingRecord
from receiver_kv_workflow.engine import ReceiverKVConfig
from experiment_2.components.config import arguments, DirectConfig, VARIANT
from experiment_2.components.generation import DirectSummaryRunner
from experiment_2.components.runner import run
from experiment_2.rescore import rescore
from experiment_2.compare import compare


class DirectGenerationTest(unittest.TestCase):
    def runner(self, thinking=True):
        import torch
        class Tokenizer:
            def encode(self,text,**kwargs): return [ord(c) for c in text]
            def decode(self,ids,**kwargs): return ''.join(chr(i) for i in ids if i)
            def apply_chat_template(self,messages,**kwargs):
                self.messages=messages
                self.thinking=kwargs['enable_thinking']
                return '\n'.join(m['content'] for m in messages)
        runner=DirectSummaryRunner.__new__(DirectSummaryRunner)
        runner.config=ReceiverKVConfig(summary_thinking=thinking)
        runner.tokenizer=Tokenizer()
        runner.torch=torch
        runner.device=torch.device('cpu')
        runner.context_limit=100000
        runner.model=object()
        return runner

    def test_full_source_fresh_cache_shared_prompt_and_thinking_accounting(self):
        runner=self.runner()
        source='Alice: We rejected the old plan.\nBob: The new plan is approved.'
        record=MeetingRecord('a',source,'summary','The new plan is approved.')
        def generated(model,tokenizer,cache,ids,**kwargs):
            self.assertIsNone(cache)
            self.assertEqual(kwargs['absolute_start'],0)
            self.assertEqual(kwargs['temperature'],.3)
            self.assertEqual(kwargs['top_p'],.9)
            self.assertEqual(kwargs['repetition_penalty'],1.)
            self.assertIn(source,tokenizer.decode(ids[0].tolist()))
            tokens=tokenizer.encode('reason</think>The new plan is approved.')+[0]
            for i,token in enumerate(tokens): kwargs['on_token'](None,token,i)
            return 'unused',len(tokens),False,dict(answer_tokens=len(tokens)-1,terminal_tokens=1)
        with patch('experiment_2.components.generation.generate',generated):
            result=runner.run_direct(record)
        self.assertEqual(result['answers'][VARIANT],'The new plan is approved.')
        self.assertEqual(result['thinking'],'reason')
        counts=result['variants'][VARIANT]['generation_counts']
        self.assertEqual(sum(v for k,v in counts.items() if k!='thinking_complete'),result['variants'][VARIANT]['generated_tokens'])
        from receiver_kv_workflow.prompts import SUMMARY_REQUEST
        self.assertEqual(runner.tokenizer.messages[-1]['content'],SUMMARY_REQUEST)
        accounting=result['variants'][VARIANT]['token_accounting']
        self.assertEqual(accounting['source_prefill_tokens'],0)
        self.assertGreater(accounting['receiver_prompt_tokens'],len(source))

    def test_context_exhaustion_is_failure_not_a_scored_partial(self):
        runner=self.runner(False)
        with patch('experiment_2.components.generation.generate',return_value=('partial',100,True,{})):
            with self.assertRaisesRegex(RuntimeError,'before EOS'):
                runner.run_direct(MeetingRecord('a','Alice: hello','summary','hello'))


class FakeRunner:
    loads=0
    def __init__(self,config,progress=None):
        type(self).loads+=1
        self.device=SimpleNamespace(type='cpu')
        self.model=SimpleNamespace(config=SimpleNamespace(_commit_hash='mock'))
        self.tokenizer=SimpleNamespace(encode=lambda s,**kwargs:s.split())
    def run_direct(self,record,**kwargs):
        if record.id=='bad': raise RuntimeError('mock context failure')
        return dict(id=record.id,answers={VARIANT:'The plan is approved.'},thinking='',
            transcript_tokens=len(record.context.split()),context_limit=40960,
            variants={VARIANT:dict(pipeline_seconds_excluding_model_load=1.,generated_tokens=3,output_cap_hit=False,
                generation_counts=dict(answer_tokens=2,terminal_tokens=1),token_accounting=dict(
                    source_prefill_tokens=0,selection_probe_tokens=0,specialist_prompt_tokens=0,
                    specialist_handoff_tokens=0,receiver_prompt_tokens=5,receiver_output_tokens=3))})


class DirectExperimentTest(unittest.TestCase):
    def test_one_model_checkpoints_failures_rescore_and_matched_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);input_path=root/'meetings.jsonl'
            input_path.write_text(''.join(json.dumps(dict(id=sid,transcript='Alice: The plan is approved.',
                summary='The plan is approved.'))+'\n' for sid in ('a','bad','c')),encoding='utf-8')
            args=arguments(['--input',str(input_path),'--output-dir',str(root/'runs'),'--device','cpu'])
            FakeRunner.loads=0
            folder,failures=run(args,runner_factory=FakeRunner)
            self.assertEqual((FakeRunner.loads,failures),(1,1))
            summary=json.loads((folder/'summary.json').read_text())
            self.assertEqual(summary['completed_samples'],2)
            self.assertEqual(summary['overall']['tokens']['total_tokens']['mean'],8)
            self.assertEqual(summary['length_groups']['0-4000']['planned'],3)
            self.assertTrue((folder/'summaries'/'00000'/'direct_text.md').exists())
            evaluated=rescore(folder)
            self.assertEqual(json.loads((evaluated/'summary.json').read_text())['completed_samples'],2)
            workflow=root/'workflow';(workflow/'samples').mkdir(parents=True)
            config=asdict(ReceiverKVConfig(**asdict(DirectConfig(device='cpu'))))
            (workflow/'manifest.json').write_text(json.dumps(dict(sample_ids=['a','c'],config=config)))
            for new_index,old_index in enumerate((0,2)):
                sample=json.loads((folder/'samples'/f'{old_index:05d}.json').read_text())
                sample['quality']={'chunk_kv':sample['quality'][VARIANT]}
                sample['result']['answers']={'chunk_kv':sample['result']['answers'][VARIANT]}
                sample['result']['variants']={'chunk_kv':sample['result']['variants'][VARIANT]}
                (workflow/'samples'/f'{new_index:05d}.json').write_text(json.dumps(sample))
            output=compare(folder,workflow)
            report=json.loads((output/'comparison.json').read_text())
            self.assertEqual(report['overall']['n'],2)
            self.assertEqual(report['overall']['metrics']['rouge_l_f1']['difference_workflow_minus_direct'],0.)
            sample['source']='Different original source'
            (workflow/'samples'/'00001.json').write_text(json.dumps(sample))
            with self.assertRaisesRegex(ValueError,'Source/reference mismatch'):
                compare(folder,workflow)

    def test_semantic_scorer_supports_one_variant_without_requiring_text_control(self):
        sample=dict(reference='reference',quality={VARIANT:{}},result=dict(answers={VARIANT:'answer'}))
        class Tokenizer:
            model_max_length=128
            def __call__(self,text,**kwargs):return dict(input_ids=[1,2,3])
        class Scorer:
            hash='mock'
            def __init__(self,**kwargs):self._tokenizer=Tokenizer()
            def score(self,predictions,references,**kwargs):
                self.predictions=predictions
                assert predictions==['answer'] and references==['reference']
                return tuple(SimpleNamespace(tolist=lambda:[.8]) for _ in range(3))
        modules={'bert_score':SimpleNamespace(BERTScorer=Scorer), 'transformers':SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer()),
            AutoConfig=SimpleNamespace(from_pretrained=lambda *a,**k:SimpleNamespace(model_type='mock',max_position_embeddings=128)))}
        from experiment_1.components.semantic_quality import add_bertscore
        with patch.dict('sys.modules',modules),patch('experiment_1.components.semantic_quality.importlib.metadata.version',return_value='mock'):
            add_bertscore([sample],'encoder',12,'cpu',variants=(VARIANT,))
        self.assertEqual(sample['quality'][VARIANT]['bertscore_f1'],.8)
