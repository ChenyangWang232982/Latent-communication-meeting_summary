import time
from dataclasses import asdict

from receiver_kv_workflow.engine import ReceiverKVConfig, ReceiverKVRunner
from receiver_kv_workflow.prompts import SUMMARY_REQUEST
from receiver_kv_workflow.cache import generate
from chunk_specialist_workflow.components.generation import suffix
from chunk_specialist_workflow.components.thinking import split_thinking
from .config import VARIANT


class DirectSummaryRunner(ReceiverKVRunner):
    def __init__(self, config, progress=None):
        # Reuse model loading, numerical settings, tensor/device helpers. Never call
        # ReceiverKVRunner.run(): no source selector, embedding, or specialist runs.
        super().__init__(ReceiverKVConfig(**asdict(config)), progress=progress)

    def run_direct(self, record, *, on_progress=None, on_generation=None):
        self._sync()
        started = time.perf_counter()
        raw_tokens = len(self.tokenizer.encode(record.context, add_special_tokens=False))
        ids = suffix(self.tokenizer, record.context, [], SUMMARY_REQUEST,
                     enable_thinking=self.config.summary_thinking)
        generated_ids = []
        self.torch.manual_seed(self.config.summary_seed)
        answer, count, cap_hit, counts = generate(
            self.model, self.tokenizer, None, self._tensor(ids), absolute_start=0,
            context_limit=self.context_limit, temperature=self.config.summary_temperature,
            top_p=self.config.summary_top_p, top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            on_progress=(lambda event: on_progress(dict(stage='direct_summary', **event))) if on_progress else None,
            on_snapshot=on_generation,
            on_token=(lambda logits, token, step: generated_ids.append(token)) if self.config.summary_thinking else None)
        self._sync()
        if cap_hit:
            raise RuntimeError('direct_summary: model context exhausted before EOS; inspect live output')
        reasoning = ''
        if self.config.summary_thinking:
            answer, reasoning, counts = split_thinking(self.tokenizer, generated_ids, counts['terminal_tokens'])
        elapsed = time.perf_counter()-started
        return dict(id=record.id, answers={VARIANT: answer}, thinking=reasoning,
                    transcript_tokens=raw_tokens, context_limit=self.context_limit,
                    variants={VARIANT: dict(
                        pipeline_seconds_excluding_model_load=elapsed, generated_tokens=count,
                        output_cap_hit=False, generation_counts=counts,
                        token_accounting=dict(source_prefill_tokens=0, selection_probe_tokens=0,
                            specialist_prompt_tokens=0, specialist_handoff_tokens=0,
                            receiver_prompt_tokens=len(ids), receiver_output_tokens=count))})
