# Meeting Evaluation

This package evaluates the receiver-conditioned KV workflow on speaker-labelled
meeting benchmarks. Reports contain metrics only; they do not copy source
transcripts or generated summaries.

```text
MeetingBank JSONL -> full-source KV prefill -> receiver-conditioned routing
                 -> routed-KV / matched-text communication -> metric report
```

Prepare benchmark JSONL files with `data_preprocess/README.md` first.

Run the evaluation:

```bash
python -m meeting_evaluation.test \
  --benchmark meetingbank \
  --input input/meetingbank_speaker_test.jsonl \
  --output-dir test_result \
  --num-samples 25 \
  --model Qwen/Qwen3-4B \
  --receiver-kv-role-budget-ratio 0.25 \
  --receiver-kv-span-tokens 32 \
  --receiver-kv-query-batch-tokens 8 \
  --receiver-kv-max-new-tokens 1024 \
  --device cuda
```

The report compares two communication representations under the same source
selection, prompt, model, and output budget:

| Variant | Receiver input |
|---|---|
| `routed_kv` | Receiver-conditioned selected source KV positions |
| `specialist_text` | The same selected source positions rewritten once as a natural-language handoff |

Reports are written to `test_result/YYYYMMDD_HHMM/meetingbank_receiver_kv.txt`
and `.json`. They include ROUGE-L, a lexical source-support proxy, cache and
runtime measurements, and logical model-token events. KV positions, KV bytes,
and token counts are intentionally reported separately.

Run the available benchmark command:

```bash
python -m meeting_evaluation.run_all --num-samples 25
```
