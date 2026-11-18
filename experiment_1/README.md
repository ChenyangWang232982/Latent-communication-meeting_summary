# Experiment 1: global context + current-part specialists

实验1只比较当前工作流的 `chunk_kv` 和 `chunk_text`，不运行旧版直接摘要或额外 review。
每个样本全文 prefill 和注意力／向量路由只做一次，两条路径共用同一组选中坐标。

| Variant | Local specialist input | Final synthesis input |
|---|---|---|
| `chunk_kv` | Global selected KV + current original part + location/instructions | Global selected KV + ordered reports |
| `chunk_text` | Same global selected evidence as verbatim text + same original part/location/instructions | Global selected text + ordered reports |

Both paths use identical original-transcript partitions, stage seeds and decoding
settings. Each independently generates local reports. Only final synthesis enables
thinking by default; use `--no-summary-thinking` to disable it for both. This is a
workflow comparison: contextualized full-source KV and re-encoded selected text
have different positional histories and available prior context, so it is not a
pure representation-only causal intervention.

## Run / 运行

```bash
conda activate latent-agents
python -m experiment_1 \
  --input input/meetingbank_speaker_4k_8k_test.jsonl \
  --benchmark meetingbank \
  --output-dir test_result/experiment_1_chunk_3_seed12 \
  --num-samples 3 \
  --model Qwen/Qwen3-8B \
  --device cuda \
  --role-budget-ratio 0.25 \
  --speaker-neighbor-blocks 1 \
  --vector-routing \
  --specialist-parts 8 \
  --specialist-overlap 0.25 \
  --summary-thinking \
  --seed 12 \
  --summary-seed 42
```

在已预处理文件中按顺序取前3个样本，每个样本运行两条路径，共3组配对结果。
`--benchmark` 只标识当前输入数据集，不会自动读取其他数据集。
`--num-samples 25` runs the first 25 records; `0` runs all. Input is speaker-labelled
JSONL with unique `id`, `transcript`/`context`, and nonempty human `summary`/`answer`.
The prepared 4k–8k filter uses model-token length, not word count.

Defaults: local temperature 0.3, top_p 0.9, generated-only repetition penalty 1.1;
final temperature 0.3, top_p 0.9, repetition penalty 1.0; top_k disabled. Local seed
is `seed + part_index`; final seed is `summary_seed`. Model precision is the normal
loaded precision unless the explicit optional `--fp32-lm-head` control is supplied.
No output-token or item cap is imposed; context exhaustion is an error.

Parts cover the full original transcript, including turns outside global selected
KV. Approximate 1/8 windows and 1/4 overlap preserve whole speaker blocks, so the
number of parts can exceed eight. Each receives a part number, original block range
and text-position percentage. All specialists share weights and run sequentially.

## Outputs / 输出

Under `test_result/experiment_1_chunk_3_seed12/<timestamp>/`:

```text
manifest.json                         settings, input/code hashes, selected IDs, versions
runtime.json                          hardware and model revision
progress.json                         completed/failed counts
samples/00000.json                    source, reference, both results and measurements
summaries/00000/chunk_kv.md            final KV summary (no thinking text)
summaries/00000/chunk_text.md          final text-control summary
summaries/00000/*.specialists.json     shared KV positions, part coordinates/text, reports, stage costs
summaries/00000/*_notes.md             concatenated local reports for inspection
summaries/00000/*.thinking.md          final reasoning, separate from scored answers
live/00000/*.partial.txt               live generation text
live/00000/*.status.json               per-stage progress and completion
summary.txt / summary.json            aggregated scores, tokens, time, memory and paired intervals
tokens.json / per_sample.csv          sample-level accounting and scores
repetition.json                       exact duplicate bullet diagnostics
human_review.csv                      optional blank manual annotation sheet
```

Local stages are `chunk_specialist_000`, etc.; text stages add `text_`. Final stages
are `chunk_summary` and `text_chunk_summary`. Partial text can include thinking;
only the final answer is quality-scored. Old per-token logits/cache-condition
diagnostics have been retired with the direct-summary workflow.

## Resume / 断点续跑

Repeat the same command with `--resume <existing_timestamp_directory>`. Completed
sample pairs are reused; failed or missing pairs are retried. Inputs, code,
configuration, dependencies and runtime identity must match to combine timings.
If either variant fails, the pair is excluded from paired means and retried on
resume. Outputs/checkpoints are saved after each completed pair. Increase sample
count in a new run. Historical old-workflow results cannot be resumed here.

## Metrics / 指标

- ROUGE-1/2/L F1, ROUGE-L precision/recall and output/reference word ratio are
  reference overlap measures (unstemmed ASCII tokens; not factual accuracy).
- Optional BERTScore P/R/F1 compares the final summaries with human references;
  no LLM judge is used. The lexical source-support proxy is not factual accuracy.
- Logical decoder tokens include source prefill/probes once per standalone path,
  all specialist prompts including original part text, all local outputs, final
  prompt and output including thinking/delimiters/EOS. Reused KV rows do not count
  as newly encoded prompt tokens. Separate embedding encoder tokens are reported.
- Time includes shared prefill/selection plus measured variant wall time, excluding
  model load, scoring and file output. Both variants run in a fixed KV-then-text
  order. Paired-run sums are not actual total shared GPU work.
- CUDA peaks are PyTorch allocated bytes within each variant, including resident
  model, embedding model and full source KV; they exclude allocator reserve and
  non-PyTorch memory. Shared prefill/selection peak is also recorded separately.
- Logical payload sums stage KV tensor bytes or global evidence UTF-8 bytes, part
  UTF-8 bytes and local-report UTF-8 bytes. These are different representations,
  not measured network traffic or billing. Token savings are not FLOP savings.
- Paired KV-minus-text score intervals use 2,000 fixed-seed bootstrap resamples;
  no interval for one sample. Multiple records from one meeting are dependent.
- Duplicate diagnostics count exact repeated bullets, not semantic duplication.

## Offline BERTScore / 离线补评

This reuses saved summaries, token counts and timing; generation is not rerun:

```bash
pip install -r experiment_1/requirements-quality.txt
RUN_DIR=$(find test_result/experiment_1_chunk_3_seed12 -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
python -m experiment_1.rescore \
  --run-dir "$RUN_DIR" \
  --bertscore-model allenai/longformer-base-4096 \
  --bertscore-layers 12 \
  --device cuda
```

The encoder scores summary/reference, not the entire transcript. Overlong inputs
raise rather than silently truncating; use the same encoder for all comparisons.
Do not mix BERTScore values from different encoder settings.

Component responsibilities: `config` (shared CLI + fingerprints), `runner` (paired
execution), `resources` (wall/CUDA measurements), `storage` (atomic checkpoints),
`generation_trace`/`progress` (live status), `scoring`/`reporting` (metrics),
`semantic_quality` (offline BERTScore), `token_accounting` (logical events).

```bash
python -m unittest discover -s experiment_1/tests
```

See [retired experiment notes](HISTORY.md) for the removed direct-summary method.
