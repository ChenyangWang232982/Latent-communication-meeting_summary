# Meeting Evaluation

This package evaluates the receiver-conditioned KV meeting workflow on
MeetingBank. It writes metric reports only; source transcripts and generated
summaries are not copied into the evaluation report.

```text
MeetingBank JSONL -> full-source KV prefill -> receiver-conditioned routing
                 -> routed/full/text/specialist controls -> metric report
```

Prepare a normalized MeetingBank test set first:

```bash
python -m data_preprocess.prepare_meetingbank --max-records 8
```

Run the evaluation:

```bash
python -m meeting_evaluation.test \
  --input input/meetingbank_test.jsonl \
  --output-dir test_result \
  --num-samples 8 \
  --model Qwen/Qwen3-4B \
  --receiver-kv-role-budget-ratio 0.15 \
  --receiver-kv-span-tokens 32 \
  --receiver-kv-query-batch-tokens 8 \
  --receiver-kv-max-new-tokens 1024 \
  --receiver-kv-specialist-text-max-new-tokens 384 \
  --device cuda
```

The report compares four controls:

| Variant | Receiver input |
|---|---|
| `routed_kv` | Receiver-conditioned selected source KV positions |
| `full_kv` | Full source KV cache |
| `selected_text` | The same selected source positions re-prefilled as text |
| `specialist_text` | Generated natural-language specialist handoffs |

Reports are written to `test_result/YYYYMMDD_HHMM/meetingbank_receiver_kv.txt`
and `.json`. They include ROUGE-L, a lexical source-support proxy, cache and
runtime measurements, and logical model-token events. KV positions, KV bytes,
and token counts are intentionally reported separately.

Run the available benchmark command:

```bash
python -m meeting_evaluation.run_all --num-samples 8
```
