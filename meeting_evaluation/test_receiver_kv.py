import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .metrics import score_receiver_kv_results
from .report import write_receiver_kv_reports


def result(record_id="meeting-1"):
    return {
        "id": record_id,
        "gold_answer": "Alice will send the draft Friday.",
        "answers": {
            "routed_kv": "Alice will send the draft Friday.",
            "full_kv": "Alice will send the draft Friday.",
            "selected_text": "Alice will send the draft.",
            "specialist_text": "Alice will send the draft Friday.",
        },
        "prefill_seconds": 1.0, "selection_seconds": 0.5, "source_cache_bytes": 2**20,
        "variants": {
            name: {"incoming_positions": 20, "incoming_cache_bytes": 2**19,
                   "logical_payload_bytes": 200, "receiver_seconds": 0.3,
                   "pipeline_seconds_excluding_model_load": 1.8,
                   "generated_tokens": 8, "output_cap_hit": False}
            for name in ("routed_kv", "full_kv", "selected_text", "specialist_text")
        },
    }


class ReceiverKVEvaluationTests(unittest.TestCase):
    def test_metrics_and_report_hide_answers(self):
        metrics = score_receiver_kv_results([result()], {"meeting-1": "Alice will send the draft Friday."})
        self.assertEqual(metrics["samples"], 1)
        self.assertEqual(metrics["quality"]["routed_kv"]["rouge_l_f1"], 1.0)
        self.assertEqual(metrics["efficiency"]["variants"]["routed_kv"]["mean_incoming_positions"], 20)
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict("os.environ", {"MEETING_EVAL_RUN_ID": "20261021_2100"}, clear=False):
                text_path, json_path = write_receiver_kv_reports(Path(temporary), "meetingbank", metrics, {"workflow": "receiver_kv"})
            text = text_path.read_text(encoding="utf-8")
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertIn("routed_kv", text)
            self.assertNotIn("Alice", text)
            self.assertNotIn("answers", json.dumps(payload))

    def test_meetingbank_entrypoint_uses_receiver_workflow(self):
        from .records import MeetingRecord
        from .test import main
        fake = result()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "meetingbank.jsonl"
            input_path.write_text('{"id":"meeting-1","transcript":"source","summary":"gold"}\n', encoding="utf-8")
            argv = ["test", "--benchmark", "meetingbank", "--workflow", "receiver_kv",
                    "--input", str(input_path), "--output-dir", str(root / "out"), "--device", "cpu"]
            with patch("sys.argv", argv), patch("meeting_evaluation.test.run_receiver_kv", return_value=[fake]) as runner:
                main()
            runner.assert_called_once()
            self.assertTrue(next((root / "out").glob("*/meetingbank_receiver_kv.json")))

    def test_receiver_workflow_rejects_qmsum(self):
        from .test import main
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "records.jsonl"
            source.write_text('{"id":"q","context":"x","question":"why","answer":"a"}\n', encoding="utf-8")
            with patch("sys.argv", ["test", "--benchmark", "qmsum", "--workflow", "receiver_kv", "--input", str(source)]):
                with self.assertRaisesRegex(ValueError, "MeetingBank only"):
                    main()


if __name__ == "__main__":
    unittest.main()
