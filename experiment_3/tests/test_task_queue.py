import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiment_3.task_queue import run


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class FakeCommands:
    """Emit the existing CLI checkpoint contracts, without loading any models."""
    def __init__(self, root, *, fail_workflow_once=False, partial_direct=False):
        self.root=root; self.calls=[]; self.fail_workflow_once=fail_workflow_once
        self.partial_direct=partial_direct

    def __call__(self, command, log, cwd):
        log.write_text('fake subprocess log', encoding='utf-8')
        if '-m' not in command:
            self.calls.append('model')
            cache=self.root/'model_cache'; cache.mkdir(exist_ok=True)
            save(Path(command[-1]), dict(path=str(cache)))
            return 0
        module=command[command.index('-m')+1]
        self.calls.append(module)
        output=Path(command[command.index('--output-dir')+1])
        child=output/'saved_run'
        if module=='experiment_3':
            if self.fail_workflow_once:
                self.fail_workflow_once=False
                return 2
            save(child/'status.json',dict(planned=2,completed_records=2,failed=0))
        elif module=='experiment_2':
            save(child/'progress.json',dict(planned=2,completed=1 if self.partial_direct else 2,
                failed=1 if self.partial_direct else 0,finished=True))
            return 1 if self.partial_direct else 0
        elif module=='experiment_3.random_control':
            save(child/'status.json',dict(planned=2,eligible=2,processed=2,completed=2,failed=0))
            (child/'comparison.txt').write_text('finished')
            source=Path(command[command.index('--source-run')+1])
            assert (source/'status.json').exists()
        else:
            assert module=='experiment_3.evaluate'
            random_path=Path(command[command.index('--random-runs')+1])
            assert (random_path/'comparison.txt').exists()
            save(output/'summary.json',dict(planned_workflow_samples=2,paired_samples=1 if self.partial_direct else 2))
            (output/'summary.txt').write_text('fake scores')
        return 0


class TaskQueueTest(unittest.TestCase):
    def arguments(self, root):
        source=root/'ami.jsonl'
        source.write_text(''.join(json.dumps(dict(id=str(i),transcript=f'Alice: Meeting {i}.',summary='Reference.'))+'\n'
                                  for i in range(2)),encoding='utf-8')
        return SimpleNamespace(resume=None,input=source,num_samples=2,random_seeds=[7],
            model='example/model',model_revision='a'*40,device='cpu',embedding_model='example/embedding',
            bertscore_model='example/scorer',bertscore_layers=12,output_dir=root/'queues')

    def test_serial_order_and_completed_queue_resume_do_not_rerun_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); args=self.arguments(root); execute=FakeCommands(root)
            output,failed=run(args,executor=execute)
            self.assertFalse(failed)
            expected=['model','experiment_3','experiment_2','experiment_3.random_control','experiment_3.evaluate']
            self.assertEqual(execute.calls,expected)
            state=json.loads((output/'queue.json').read_text())
            self.assertEqual(state['status'],'completed')
            self.assertEqual((output/'input.jsonl').read_bytes(),args.input.read_bytes())
            args.resume=output
            _,failed=run(args,executor=execute)
            self.assertFalse(failed)
            self.assertEqual(execute.calls,expected)

    def test_crash_stops_dependents_and_resume_reuses_model_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); args=self.arguments(root); execute=FakeCommands(root,fail_workflow_once=True)
            output,failed=run(args,executor=execute)
            self.assertTrue(failed)
            self.assertEqual(execute.calls,['model','experiment_3'])
            args.resume=output
            _,failed=run(args,executor=execute)
            self.assertFalse(failed)
            self.assertEqual(execute.calls.count('model'),1)
            self.assertEqual(execute.calls.count('experiment_3'),2)
            state=json.loads((output/'queue.json').read_text())
            self.assertEqual(state['tasks']['01_workflow']['attempts'],2)

    def test_finished_direct_batch_with_sample_failures_can_advance(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); args=self.arguments(root); execute=FakeCommands(root,partial_direct=True)
            output,failed=run(args,executor=execute)
            self.assertFalse(failed)
            state=json.loads((output/'queue.json').read_text())
            self.assertEqual(state['status'],'completed_with_failures')
            self.assertEqual(state['tasks']['02_direct']['failed'],1)
            self.assertEqual(execute.calls[-1],'experiment_3.evaluate')

    def test_modified_input_is_rejected_before_resuming_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); args=self.arguments(root); execute=FakeCommands(root,fail_workflow_once=True)
            output,_=run(args,executor=execute)
            (output/'input.jsonl').write_text('changed data',encoding='utf-8')
            args.resume=output
            _,failed=run(args,executor=execute)
            self.assertTrue(failed)
            self.assertEqual(execute.calls,['model','experiment_3'])

    def test_too_few_records_fail_before_downloading_a_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); args=self.arguments(root); args.num_samples=100
            execute=FakeCommands(root)
            with self.assertRaisesRegex(ValueError,'only 2'):
                run(args,executor=execute)
            self.assertEqual(execute.calls,[])

    def test_additional_random_seeds_are_serial_steps_before_one_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); args=self.arguments(root); args.random_seeds=[7,19]
            execute=FakeCommands(root)
            output,failed=run(args,executor=execute)
            self.assertFalse(failed)
            self.assertEqual(execute.calls.count('experiment_3.random_control'),2)
            self.assertEqual(execute.calls.count('experiment_3.evaluate'),1)
            state=json.loads((output/'queue.json').read_text())
            self.assertEqual(state['tasks']['03_random_seed19']['status'],'completed')

    def test_missing_completed_artifact_reruns_that_step_and_its_dependents(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); args=self.arguments(root); execute=FakeCommands(root)
            output,_=run(args,executor=execute)
            state=json.loads((output/'queue.json').read_text())
            direct=Path(state['tasks']['02_direct']['artifact'])
            (direct/'progress.json').unlink()
            args.resume=output
            _,failed=run(args,executor=execute)
            self.assertFalse(failed)
            self.assertEqual(execute.calls.count('model'),1)
            self.assertEqual(execute.calls.count('experiment_3'),1)
            self.assertEqual(execute.calls.count('experiment_2'),2)
            self.assertEqual(execute.calls.count('experiment_3.random_control'),2)
            self.assertEqual(execute.calls.count('experiment_3.evaluate'),2)


if __name__=='__main__': unittest.main()
