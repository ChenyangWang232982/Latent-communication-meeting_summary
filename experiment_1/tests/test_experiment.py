import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from experiment_1.components.reporting import paired_interval
from experiment_1.components.runner import run


class FakeRunner:
    loads = 0
    calls = []
    fail = set()

    def __init__(self, config, progress):
        type(self).loads += 1
        self.device = SimpleNamespace(type="cpu")

    def run(self, sample_id, transcript, *, variants, on_variant, on_state=None, on_generation=None, on_token=None):
        assert variants == ("chunk_kv", "chunk_text")
        type(self).calls.append(sample_id)
        if sample_id in type(self).fail:
            raise ValueError("Simulated context failure")
        data = {}
        for name in variants:
            on_variant(name, "start")
            data[name] = {"incoming_positions": 2, "incoming_cache_bytes": 128 if name == "chunk_kv" else 0,
                "logical_payload_bytes": 128, "receiver_seconds": .001,
                "pipeline_seconds_excluding_model_load": .1, "generated_tokens": 3, "output_cap_hit": False,
                "generation_counts": {"answer_tokens": 2, "terminal_tokens": 1},
                "token_accounting": {"source_prefill_tokens": 4, "selection_probe_tokens": 1,
                    "specialist_prompt_tokens": 2 if name == "chunk_text" else 0,
                    "specialist_handoff_tokens": 2 if name == "chunk_text" else 0,
                    "receiver_prompt_tokens": 2, "receiver_output_tokens": 3}}
            if name == "chunk_text":
                data[name].update(specialist_handoff_tokens=2, specialist_handoff_cap_hit=True, handoff_utf8_bytes=5)
            on_variant(name, "end")
        return {"id": sample_id, "answers": {"chunk_kv": "report on Friday", "chunk_text": "report"},
                "handoffs": {"chunk_text": "notes"}, "variants": data, "source_positions": 4,
                "source_cache_bytes": 256, "prefill_seconds": .01, "selection_seconds": .02,
                "selected_source_positions": [0, 2], "vector_routing": None}


class ExperimentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input = self.root / "input.jsonl"
        rows = [{"id": name, "transcript": "Alice: report on Friday", "summary": "report on Friday"} for name in ("one", "two")]
        self.input.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        self.args = Namespace(input=self.input, benchmark="meetingbank", output_dir=self.root / "results", resume=None,
            num_samples=2, model="fake", device="cpu", seed=12, role_budget_ratio=.25,
            speaker_neighbor_blocks=1, span_tokens=32, query_batch_tokens=8, vector_routing=False,
            embedding_model="fake-embedding", vector_weight=.3)
        FakeRunner.loads, FakeRunner.calls, FakeRunner.fail = 0, [], set()

    def tearDown(self):
        self.temp.cleanup()

    def test_pair_outputs_resume_and_sender_cap(self):
        path, failures = run(self.args, FakeRunner)
        self.assertEqual(failures, 0)
        self.assertEqual(FakeRunner.loads, 1)
        self.assertEqual(FakeRunner.calls, ["one", "two"])
        sample = json.loads((path / "samples/00000.json").read_text(encoding="utf-8"))
        self.assertIsNone(sample["result"]["variants"]["chunk_kv"]["cuda_peak_allocated_bytes"])
        self.assertEqual(sample["result"]["variants"]["chunk_text"]["logical_payload_bytes"], 128)
        self.assertEqual((path / "summaries/00000/chunk_text_notes.md").read_text(encoding="utf-8"), "notes\n")
        report = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(report["additional_resources"]["chunk_text"]["handoff_cap_hits"], 2)
        self.assertEqual(report["paired_rouge_l_difference_kv_minus_text"]["n"], 2)
        self.args.resume = path
        run(self.args, FakeRunner)
        self.assertEqual(FakeRunner.loads, 1)
        self.assertEqual(FakeRunner.calls, ["one", "two"])
        with (path / "human_review.csv").open(encoding="utf-8-sig", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 4)

    def test_three_record_smoke_saves_six_final_summaries(self):
        with self.input.open('a', encoding='utf-8') as handle:
            handle.write('\n' + json.dumps({'id': 'three', 'transcript': 'Alice: report on Friday',
                                            'summary': 'report on Friday'}))
        self.args.num_samples = 3
        path, failures = run(self.args, FakeRunner)
        self.assertEqual(failures, 0)
        self.assertEqual(FakeRunner.calls, ['one', 'two', 'three'])
        self.assertEqual(len(list((path / 'summaries').glob('*/chunk_kv.md'))), 3)
        self.assertEqual(len(list((path / 'summaries').glob('*/chunk_text.md'))), 3)
        self.assertEqual(json.loads((path / 'summary.json').read_text(encoding='utf-8'))['completed_samples'], 3)

    def test_requested_limit_above_available_reports_actual_size(self):
        self.args.num_samples = 100
        path, failures = run(self.args, FakeRunner)
        self.assertEqual(failures, 0)
        manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['sampling'], dict(available_records=2, requested_limit=100, selected_records=2))
        report = json.loads((path / 'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(report['planned_samples'], 2)
        self.assertEqual(report['completed_samples'], 2)

    def test_failed_pair_retried_and_annotations_preserved(self):
        FakeRunner.fail = {"two"}
        path, failures = run(self.args, FakeRunner)
        self.assertEqual(failures, 1)
        sheet = path / "human_review.csv"
        annotation = sheet.read_text(encoding="utf-8-sig").replace("one,chunk_kv,", "one,chunk_kv,1")
        sheet.write_text(annotation, encoding="utf-8-sig")
        FakeRunner.fail = set()
        self.args.resume = path
        _, failures = run(self.args, FakeRunner)
        self.assertEqual(failures, 0)
        self.assertEqual(FakeRunner.calls, ["one", "two", "two"])
        self.assertIn("one,chunk_kv,1", sheet.read_text(encoding="utf-8-sig"))
        with sheet.open(encoding="utf-8-sig", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 4)

    def test_resume_refuses_changed_config_or_input(self):
        path, _ = run(self.args, FakeRunner)
        self.args.resume = path
        self.args.role_budget_ratio = .35
        with self.assertRaisesRegex(ValueError, "config"):
            run(self.args, FakeRunner)
        self.args.role_budget_ratio = .25
        self.input.write_text(self.input.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "input_sha256"):
            run(self.args, FakeRunner)

    def test_offline_rescore_reuses_tokens_and_preserves_saved_run(self):
        from experiment_1.rescore import rescore
        path, _ = run(self.args, FakeRunner)
        original = {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()}
        destination = rescore(path, self.root / "rescored")
        self.assertEqual(FakeRunner.loads, 1)
        for relative, data in original.items():
            self.assertEqual((path / relative).read_bytes(), data)
        report = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(report["tokens"]["chunk_kv"]["total_tokens"]["mean"], 10)
        self.assertEqual(report["tokens"]["chunk_text"]["total_tokens"]["sum"], 28)
        self.assertAlmostEqual(report["token_savings_kv_vs_text_percent"], 100 * 4 / 14)
        self.assertEqual(report["metrics"]["quality"]["chunk_kv"]["rouge_2_f1"], 1)
        self.assertEqual(report["metrics"]["quality"]["chunk_text"]["rouge_l_precision"], 1)
        self.assertAlmostEqual(report["metrics"]["quality"]["chunk_text"]["rouge_l_recall"], 1/3)
        with (destination / "per_sample.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["total_tokens"], "10")
        self.assertIn("Logical decoder token events", (destination / "summary.txt").read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "separate"):
            rescore(path, path)

    def test_no_false_confidence_interval_for_one_pair(self):
        sample = {"quality": {"chunk_kv": {"rouge_l_f1": .4}, "chunk_text": {"rouge_l_f1": .3}}}
        self.assertIsNone(paired_interval([sample])["bootstrap_95_percent_interval"])

    def test_chunk_pair_preserves_stage_payload_and_saves_reports(self):
        class ChunkRunner(FakeRunner):
            def run(self, *args, **kwargs):
                result = super().run(*args, **kwargs)
                result['chunk_specialists_by_variant'] = {
                    name: {'reports': [{'window': 0, 'report': 'Local notes'}],
                           'windows': [{'positions': [0, 2]}],
                           'thinking_outputs': {'chunk_summary': 'reasoning'}, 'stages': {}}
                    for name in ('chunk_kv', 'chunk_text')}
                for measurement in result['variants'].values():
                    measurement['logical_payload_bytes'] = 321
                return result
        self.args.summary_thinking = True
        path, failures = run(self.args, ChunkRunner)
        self.assertEqual(failures, 0)
        sample = json.loads((path / 'samples/00000.json').read_text(encoding='utf-8'))
        self.assertEqual(sample['result']['variants']['chunk_text']['logical_payload_bytes'], 321)
        detail = json.loads((path / 'summaries/00000/chunk_kv.specialists.json').read_text(encoding='utf-8'))
        self.assertEqual(detail['windows'][0]['positions'], [0, 2])
        report = json.loads((path / 'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(report['workflow'], 'selected_context_window_specialists')
        self.assertIn('chronological selected-context windows', report['comparison_note'])


if __name__ == "__main__":
    unittest.main()
