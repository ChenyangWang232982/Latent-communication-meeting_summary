import json
import tempfile
import unittest
from pathlib import Path
from experiment_1.components.judge_rubric import DIMENSIONS, validate
from experiment_1.components.judge_runner import run_judge


def judgment():
    return {"dimensions": {d: {"coverage": None if d == "unsupported_additions" else 2,
        "factuality": 2, "uncertain": False, "reason": "Supported by source line 1",
        "evidence": [{"source_lines": [1], "summary_quote": "Approved", "explanation": "Explicit approval"}]}
        for d in DIMENSIONS}}


class JudgeTest(unittest.TestCase):
    def test_anonymous_pair_resume_cost_and_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); run = root / "run"; (run / "samples").mkdir(parents=True)
            (run / "manifest.json").write_text(json.dumps({"sample_ids": ["private-id"]}))
            sample = {"id": "private-id", "status": "completed", "source": "Alice: Approved", "result": {
                "answers": {"routed_kv": "Approved", "specialist_text": "Approved"}}}
            checkpoint = run / "samples/00000.json"; checkpoint.write_text(json.dumps(sample))
            original = checkpoint.read_bytes(); calls = []
            def client(messages):
                calls.append(messages)
                text = json.dumps(messages)
                self.assertNotIn("routed_kv", text); self.assertNotIn("specialist_text", text)
                self.assertNotIn("private-id", text)
                return {"text": json.dumps(judgment()), "finish_reason": "stop", "seconds": 1,
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
            settings = {"seed": 12, "audit_pairs": 5}
            out = run_judge(run, root / "judge", settings, client)
            run_judge(run, out, settings, client)
            self.assertEqual(len(calls), 2)
            self.assertEqual(checkpoint.read_bytes(), original)
            report = json.loads((out / "judge_summary.json").read_text())
            self.assertEqual(report["cost"]["total_tokens_reported_sum"], 30)
            self.assertEqual(report["dimensions"]["actions"]["coverage"]["paired_n"], 1)
            self.assertEqual(report["audit_total_pairs"], 1)
            self.assertIn("Automated judgment", next((out / "audit").glob("*.md")).read_text())
            with self.assertRaisesRegex(ValueError, "Cannot resume"):
                run_judge(run, out, {"seed": 7, "audit_pairs": 5}, client)

    def test_bad_references_and_partial_output_never_get_scores(self):
        value = judgment(); value["dimensions"]["actions"]["evidence"][0]["source_lines"] = [99]
        with self.assertRaisesRegex(ValueError, "source line"):
            validate(json.dumps(value), "One line", "Approved")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / "samples").mkdir()
            (root / "manifest.json").write_text(json.dumps({"sample_ids": ["x"]}))
            (root / "samples/00000.json").write_text(json.dumps({"id": "x", "status": "completed", "source": "A", "result": {"answers": {"routed_kv": "A", "specialist_text": "A"}}}))
            def client(messages):
                return {"text": json.dumps(judgment()), "finish_reason": "length", "seconds": 1, "usage": {"total_tokens": 12}}
            out = run_judge(root, root / "judge", {"seed": 12, "audit_pairs": 0}, client)
            report = json.loads((out / "judge_summary.json").read_text())
            self.assertEqual(report["completed_pairs"], 0)
            self.assertEqual(report["cost"]["total_tokens_reported_sum"], 24)
            self.assertEqual(report["audit_total_pairs"], 1)


class JudgeEnvironmentTest(unittest.TestCase):
    def test_dotenv_precedence_and_literal_secrets(self):
        import os
        from unittest.mock import patch
        from experiment_1.components.judge_env import load_judge_env, env_value
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"JUDGE_MODEL": "shell-model"}, clear=True):
            path = Path(folder) / ".env"
            path.write_text('JUDGE_MODEL=file-model\nJUDGE_BASE_URL=https://example.com/v1\nJUDGE_API_KEY="literal-${SECRET}"\n', encoding="utf-8")
            load_judge_env(path)
            self.assertEqual(env_value("JUDGE_MODEL"), "shell-model")
            self.assertEqual(env_value("JUDGE_BASE_URL"), "https://example.com/v1")
            self.assertEqual(os.environ["JUDGE_API_KEY"], "literal-${SECRET}")

    def test_endpoint_is_not_duplicated(self):
        from experiment_1.components.judge_client import JudgeClient
        for base in ("https://example.com/v1", "https://example.com/v1/chat/completions"):
            self.assertEqual(JudgeClient(base, "model", "").url, "https://example.com/v1/chat/completions")
