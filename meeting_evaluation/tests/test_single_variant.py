import tempfile
import unittest
from pathlib import Path

from meeting_evaluation.metrics import score_receiver_kv_results
from meeting_evaluation.report import write_receiver_kv_reports, write_receiver_kv_communication_reports


class SingleVariantEvaluationTest(unittest.TestCase):
    def test_single_variant_scores_and_reports_without_fabricating_missing_variant(self):
        for variant in ('chunk_kv', 'chunk_text'):
            with self.subTest(variant=variant):
                measurement = dict(incoming_positions=10, incoming_cache_bytes=100,
                    logical_payload_bytes=100, receiver_seconds=1,
                    pipeline_seconds_excluding_model_load=2, generated_tokens=3, output_cap_hit=False,
                    token_accounting=dict(source_prefill_tokens=20, selection_probe_tokens=2,
                        specialist_prompt_tokens=0, specialist_handoff_tokens=0,
                        receiver_prompt_tokens=4, receiver_output_tokens=3))
                result = dict(id='one', answers={variant: 'A meeting decision.'},
                    variants={variant: measurement}, gold_answer='A meeting decision.',
                    prefill_seconds=.1, selection_seconds=.2, source_cache_bytes=100)
                with tempfile.TemporaryDirectory() as temp:
                    for quality, writer in ((True, write_receiver_kv_reports),
                                            (False, write_receiver_kv_communication_reports)):
                        metrics = score_receiver_kv_results([result], {'one': 'A meeting decision.'},
                                                           include_quality=quality)
                        self.assertEqual(list(metrics['efficiency']['variants']), [variant])
                        self.assertEqual(metrics['efficiency']['variants'][variant]['mean_end_to_end_total_tokens'], 29)
                        text, _ = writer(Path(temp), 'meetingbank', metrics, {})
                        self.assertIn(variant, text.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
