# Experiment 1: KV routing vs natural-language handoff

This standalone experiment reuses `ReceiverKVRunner`, the normalized record loader and the existing ROUGE-L/source-overlap metrics. Experiment-specific code lives in `components/`. Both paths use the same source cache and selected source positions from one runner call. **Review is disabled for both paths.** `specialist_text` adds a generated natural-language handoff, so this measures two communication workflows rather than an isolated representation change. There is no full-source baseline in this experiment.

## Run / 运行

From the repository root, with the prepared speaker-labelled JSONL:

```bash
conda activate latent-agents
python -m experiment_1 \
  --input input/meetingbank_speaker_4k_8k_test.jsonl \
  --benchmark meetingbank \
  --num-samples 3 \
  --model Qwen/Qwen3-8B \
  --device cuda \
  --role-budget-ratio 0.25 \
  --speaker-neighbor-blocks 1 \
  --vector-routing \
  --seed 12
```

Use `--num-samples 25` for the next run, or `0` for all records. This takes the first N entries from an already sampled file, rather than drawing a new random sample. The actual sample list is recorded. Unique IDs and nonempty reference summaries are required. Input schema is the same as `meeting_evaluation`: `id`, `transcript`/`context`, `summary`/`answer`. `question` does not change the fixed whole-meeting summary task. Remove `--vector-routing` for attention-only selection; its default is off. Configure the embedding model with `--embedding-model` if it is not in `models/Qwen3-Embedding-0.6B`.

Outputs go to `test_result/experiment_1/<timestamp>/`. Pass a different `--output-dir` to change the parent directory. Only one process should write to a run directory at a time.

## Outputs / 输出

```text
manifest.json                 configuration, sample IDs, input/code hashes and package versions
runtime.json                  GPU identity and model revision when available
progress.json                 checkpoint counts
samples/00000.json            source, reference, both summaries, handoff, sample metrics or failure
summaries/00000/
  routed_kv.md
  specialist_text.md
  specialist_text_handoff.md
  reference.md
summary.json                  aggregated metrics and paired ROUGE-L bootstrap interval
summary.txt                   concise report
per_sample.csv                paired per-record scores and costs
human_review.csv              blank annotation sheet; no automatically claimed factuality scores
```

JSON and Markdown checkpoints are saved after every completed pair. Source transcripts are retained in sample JSON for auditing. Failed samples are recorded and excluded from paired means, with counts and IDs reported explicitly. A failure after one variant completes causes the entire pair to be retried on resume. Model initialization failure stops before inference; any earlier checkpoints remain available.

## Resume / 恢复

Repeat the **same command and settings**, adding:

```bash
--resume test_result/experiment_1/<existing_timestamp>
```

Completed pairs are reused; failed or missing pairs are retried. Resume refuses changed input files, selected IDs, model/routing parameters, source code, dependency versions, or recorded runtime hardware/model revision. Increase the sample count in a new run. Actual source hashes supplement the Git commit, so uncommitted edits are detected. Saved human annotations are retained and new rows are appended on resumed completion. Interrupts can lose the current pair, but earlier atomic checkpoints survive. If every pair is already completed, the reports can be rebuilt without loading the model.

## Metrics and interpretation / 指标口径

- Quality: ROUGE-L and the existing lexical source-support proxy, per record and as macro means. Neither is factual accuracy. Use `human_review.csv` to manually assess decision/action precision and recall, owners, dates and open issues, with evidence notes; agree on annotation rules before filling it in.
- Timing: shared source prefill and selection time are charged once to each hypothetical workflow, not summed as the actual paired invocation duration. Variant wall time includes prompt preparation and, for text, excerpt mapping and handoff generation. Model loading, scoring and file I/O are excluded. Legacy stage totals are retained separately. CUDA profiling introduces small overhead. Variants run in a fixed order (KV then text); control hardware/load when interpreting small differences.
- Memory: variant CUDA peaks are PyTorch allocated bytes and include resident model/embedding weights and the full source KV. `pipeline_peak_allocated_bytes` takes the maximum of the shared prefill/selection peak and that variant's peak. The extra peak is relative to the allocated memory at variant start. These are not whole-process GPU usage or proof of memory savings from dropping the original cache. CPU runs record CUDA values as null.
- Payload: KV tensor bytes and UTF-8 decoded handoff bytes have separate definitions. No network traffic is measured; token counts are separate. Embedding input tokens are reported separately from language-model token events.
- Truncation: report both sender handoff cap hits and receiver cap hits. Cap means the model reached its remaining context limit. Inputs are not silently shortened, and a text control with no selected evidence now fails rather than receiving the full transcript alone.
- Statistical summary: paired KV-minus-text ROUGE-L difference with 2,000 record-level bootstrap resamples (fixed resampling seed 7). The interval is omitted for one sample. Records from the same meeting are not independent; use meeting-level clustering before stronger statistical claims on multiple agenda items from one meeting.

The current adapter accepts only dense full-attention Qwen3 with standard RoPE. No changes to the model generation prompts or decoding are made by this experiment. Freeze configuration and prepared data before the formal run; exploratory old results with different prompts/models are not directly pooled with these results.

## Component layout

| Component | Responsibility |
|---|---|
| `components/config.py` | CLI, settings and fingerprints |
| `components/storage.py` | atomic checkpoints and resume |
| `components/resources.py` | timing and CUDA allocator peaks |
| `components/scoring.py` | reuse quality metrics, normalize cost fields |
| `components/reporting.py` | aggregate report, CSV, confidence interval and review sheet |
| `components/runner.py` | paired execution and error recording |

Verify without loading model weights:

```bash
python -m unittest discover -s experiment_1/tests -v
```


## Live progress and generation records

Interactive terminals show overall sample progress, completed role selectors,
and the current generation stage. Decode counters update every 16 tokens with
elapsed time and token rate. Prompt prefill and embedding have no internal
percentage. Redirected output uses plain stage messages. No output cap or
repetition stopping rule is added.

Each stage saves records below `live/<zero-based sample index>/`:

| Suffix | Contents |
|---|---|
| `.input.json` | Actual prefix/suffix token IDs, selected and protected positions, hashes, config, model revision/dtype, backend and EOS IDs; saved BEFORE generation |
| `.tokens.jsonl` | One row per generated token, zero-based step, actual token ID, top 8 raw logits, top-1/top-2 margin and EOS logits |
| `.partial.txt` | Text after token 16, every 256 tokens thereafter, and normal completion |
| `.status.json` | Snapshot token count, raw tail, repetition flag and termination reason |

Stage names are `routed_kv_receiver`, `specialist_text_sender`, and
`specialist_text_receiver`. The same token observer runs on all three stages.
Token logs buffer 16 rows and flush on snapshots, EOS and normal Python interruption;
a hard process kill may lose the last buffered rows. Exact repeated-tail warnings
are diagnostic only and do not prove semantic repetition. These partial files are
not scored outputs or completion checkpoints.

Top-k collection and disk writes add diagnostic overhead, included in measured
runtime. Do not compare timings across versions with different instrumentation.
Source/selection fingerprints are available even if the first variant loops;
there is no need to wait for a completed pair. Model sampling and logits are not
modified by observers. Full-vocabulary logits are not saved at every step.

## Independent cache controls

```bash
python -m experiment_1.diagnose_generation \
  --input input/meetingbank_speaker_4k_8k_test.jsonl \
  --sample-index 0 \
  --condition refill \
  --model Qwen/Qwen3-8B \
  --device cuda \
  --role-budget-ratio 0.25 \
  --speaker-neighbor-blocks 1 \
  --vector-routing \
  --seed 12
```

Use the same input and sample as the failing formal run. An exact `--sample-id`
can replace the zero-based index. `full` retains all old KV, `routed` slices old
KV, and `refill` recomputes exactly the selected token IDs at their original
position IDs. The summary suffix and original continuation position stay the same.
There is no evidence decoding/re-tokenization or extra natural-language sender
in refill. `both` runs full then routed; `all` runs full, refill, routed. Each
condition still executes the selectors and reconstructs its own full source cache.
These controls are not registered as formal experiment variants or quality-scored.

Outputs are in `test_result/generation_diagnostic/<UTC timestamp>/<condition>/`.
Each condition has `summary.txt`, `status.json`, and the same `live/00000/`
generation records as formal runs. Actual input records replace the earlier
standalone `prompt.json`; there is no duplicate prompt rendering in the wrapper.
If either generation loops, its latest partial output and token records remain
inspectable. Ctrl+C retains the latest diagnostic files.

## First divergence and logits

Old runs do not contain per-step logits; rerun the normal condition and formal
candidate with this version. A normal refill run is useful because it should
have the same selected positions as routed. Set the two directory paths below:

```bash
NORMAL_DIR=test_result/generation_diagnostic/REPLACE_WITH_NORMAL_RUN
FORMAL_DIR=test_result/experiment_1/REPLACE_WITH_FORMAL_RUN

python -m experiment_1.compare_generation \
  --reference "$NORMAL_DIR/refill/live/00000/routed_kv_receiver.tokens.jsonl" \
  --candidate "$FORMAL_DIR/live/00000/routed_kv_receiver.tokens.jsonl"
```

The comparator checks sample ID, source/suffix token hashes, model config/revision,
dtype, attention backend, EOS and temperature before comparing actual generated
token IDs. It reports selected-position hash equality separately, allowing a full
control with different positions to be explicitly identified. The first divergence
contains both actual top-8 logit records, EOS scores and winner margins. At that
step, both runs have the same preceding generated token sequence, so their scores
can be compared. No later scores under different generated prefixes are treated
as evidence of the first cause. A short interrupted trace with matching tokens is
reported only as a matching prefix, not as identical completed output.

`first_divergence.json` is saved beside the candidate trace by default. No model
is loaded by the comparator. The report locates a divergence; it does not by itself
prove why that divergence occurs. Input/code/config/environment manifests remain
in each run folder. Changed source code cannot resume an earlier formal manifest;
keep old results and start a new run after updating.


## FP32 vocabulary projection control

Add `--fp32-lm-head` to either experiment or cache-diagnostic commands to test
rounding in the final vocabulary projection. Default remains the loaded model
precision. This creates a separate FP32 output weight and computes its linear
projection with autocast and TF32 disabled for that projection only. Input
embeddings, transformer layers and KV caches keep their loaded precision; original
position IDs, greedy decoding, selected positions and EOS rules are unchanged.
It does not restore precision lost in the checkpoint weights or transformer states.
Extra FP32 output weights use more GPU memory, and projection time may change.

```bash
python -m experiment_1.diagnose_generation \
  --input input/meetingbank_speaker_4k_8k_test.jsonl \
  --sample-index 0 \
  --condition routed \
  --output-dir test_result/debug_fp32_head \
  --model Qwen/Qwen3-8B \
  --device cuda \
  --role-budget-ratio 0.25 \
  --speaker-neighbor-blocks 1 \
  --vector-routing \
  --seed 12 \
  --fp32-lm-head
```

First check whether the originally repeating routed sample now ends normally.
`*.input.json` records `config.fp32_lm_head`, `lm_head_weight_dtype`, and the last
observed `lm_head_output_dtype`; enabled runs should report `torch.float32` for
both head fields while `model_dtype` remains the loaded transformer dtype.
Repeat with `--condition refill` and the same precision flag to compare conditions
under matched precision. The first-divergence comparator deliberately rejects
FP32-vs-default as mismatched settings. Do not combine precision conditions in
formal means; any apparent recovery is evidence about numerical sensitivity,
not proof that all KV-routing failures are resolved.


## Qwen3 non-thinking sampling control

Qwen's official model card recommends temperature 0.7, top_p 0.8, top_k 20,
min_p 0 for non-thinking mode:
https://huggingface.co/Qwen/Qwen3-8B#best-practices
The explicit greedy/endless-repetition warning in that card is scoped to thinking
mode; it is not proof of the cause of this workflow's non-thinking loops.

Keep existing precision settings and add these flags to the formal experiment or
diagnostic command to isolate decoding as the next experimental change:

```bash
--temperature 0.7 --top-p 0.8 --top-k 20
```

Default temperature stays 0 (greedy), top_p 1 and top_k 0, preserving the previous
control. Thinking remains disabled. Min-p filtering is not applied, equivalent to
min_p 0. The implementation uses Transformers' temperature, top-k and top-p
logit warpers in that order, then samples. The decoder seed is independently reset
to the configured seed before routed generation, natural-language sender generation,
and natural-language receiver generation. A preceding variant's output length must
not determine the next stage's RNG starting state. All three stages use the same
sampling settings. Source selection and embedding ranking are unchanged.

Sampling settings and seed are recorded in both manifests and actual-input logs.
Per-token logits remain RAW logits, before temperature/top-k/top-p transformations;
the sampled token need not be the raw argmax. Seeded sampling supports controlled
repetition of a run within a fixed environment, not guaranteed bitwise reproducibility
across hardware or libraries. Use more than one seed when assessing final robustness;
do not pool greedy and sampling conditions in a single mean. No new generation
length cap, repetition penalty or automatic retry is added.


### Generated-only repetition control

Add `--repetition-penalty 1.1` to the sampling experiment or diagnostic command.
Default 1.0 disables it. The identical setting applies to routed receiver, text
sender and text receiver; each stage starts with empty generated history. Source
and prompt tokens are excluded. Positive logits for previously generated tokens
are divided by the penalty, negative logits multiplied, before sampling filters.
Greedy decoding also supports it. EOS is not blocked. Raw-logit traces stay raw.

No routing/prompt change, output rewriting, automatic retry or new length cap is
introduced. Repetition penalties can affect necessary repeated names and terms;
this is an experimental intervention, not a guarantee of factual correctness.

`summary.txt` now reports sender/receiver sample counts with repeated bullets.
`repetition.json` reports per-sample bullet counts, excess duplicates and maximum
multiplicity. Matching normalizes whitespace and case across sections but does not
merge different owners. It misses paraphrases and prose repetition; zero does not
prove semantic uniqueness. Existing quality, token, cap-hit and live status metrics
remain available. Judge quality and source fidelity as well as termination.

Use a new output directory, e.g. `--output-dir test_result/repetition_control`,
first with 3 samples. To isolate the penalty, run a matched control with 1.0 and
otherwise identical sampling settings, seed, model and input. Do not resume old
runs after changing decoding configuration. Review stays disabled.


### Offline token and quality reports / 离线 token 与质量报告

No summary regeneration or review is required for existing completed runs.
Future runs automatically include the token table and lexical quality metrics.
直接读取已有的 `samples/*.json`，重新计算指标；不会重新生成摘要或调用评估 API。

```bash
RUN_DIR=$(find test_result/experiment_1_25_seed12 -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
python -m experiment_1.rescore --run-dir "$RUN_DIR"
```

Outputs under `RUN_DIR/evaluation/<UTC timestamp>/`:
- `summary.txt` and `summary.json`: mean stage input/output token events, totals,
  aggregate token savings vs specialist_text, ROUGE and length diagnostics.
- `tokens.json`: per-sample and aggregate token counts (missing accounting raises
  an error, never silently treated as zero).
- `per_sample.csv`: token breakdown and quality per paired sample.
- `repetition.json`, `human_review.csv`: duplicate diagnostics and blank manual
  annotation sheet; previous annotations remain in the source run.
- `evaluation_manifest.json`: source checkpoint hashes, evaluator code hashes,
  metric definition, optional semantic encoder settings and missing sample IDs.

Token definitions: source prefill + selection queries + sender prompt + receiver
prompt are decoder input events; sender handoff + receiver answer are output events.
The handoff being generated once and then consumed is deliberately counted twice
in its two roles. Shared prefill/selection is allocated once to EACH standalone
workflow comparison; summing both arms is not the actual paired execution workload.
Embedding encoder input tokens are reported separately in the efficiency note,
not added to Qwen decoder tokens. Neither metric is a provider bill or FLOP count.

Quality defaults: ROUGE-1/2/L precision/recall/F1, ASCII lowercase alphanumeric
word counts and prediction/reference word-length ratio. Historical ROUGE-L stays
unchanged: no stemming, whole-summary LCS (not ROUGE-Lsum). Quality means are macro
averages; overlap precision/recall can help interpret verbose vs short summaries,
but do not establish important-fact coverage. Markdown headings are not stripped,
matching the historical metric definition. No metric rewrites the output.

Optional semantic reference similarity (additional encoder download if uncached):

```bash
pip install -r experiment_1/requirements-quality.txt
python -m experiment_1.rescore \
  --run-dir "$RUN_DIR" \
  --bertscore-model allenai/longformer-base-4096 \
  --bertscore-layers 12 \
  --device cuda
```

BERTScore uses batch size 1, no IDF weighting, no baseline rescaling, and the same
encoder/layer configuration for both workflows. Longformer here accommodates
longer summaries; scores should only be compared within this fixed evaluator
configuration, not to scores from a different BERTScore encoder. Encoder context
length is checked before scoring; overlength prediction OR reference raises an
error instead of silently truncating or excluding one arm. Evaluation compute is
separate from the original workflow time, memory and token costs. Semantic scores
are optional and offline; the normal inference loop does not load this model.

ROUGE measures reference lexical overlap; BERTScore measures reference contextual
similarity. Neither validates whether a decision was reversed, a task owner was
misassigned or a date was invented. Keep source-overlap labeled a lexical proxy;
use the manual sheet for source-supported decisions/actions/owners/dates/omissions.
Sources: https://github.com/google-research/google-research/tree/master/rouge and
https://github.com/Tiiiger/bert_score

### Evaluation scope / 评估范围

质量比较使用 ROUGE-1/2/L 和 BERTScore，分别报告，不合成总分。同时保留 token、耗时、显存和重复诊断。BERTScore 使用固定的编码器计算与参考摘要的相似度，不调用生成式 AI 裁判，不需要 API 密钥。已有实验结果保留，使用上面的 rescore 指令即可补算指标。

Quality comparison uses ROUGE-1/2/L and BERTScore, reported separately without a composite score. Token accounting, runtime, memory and duplicate diagnostics remain available. BERTScore uses a fixed encoder rather than a generative judge. No evaluation API credentials are required. Historical evaluation artifacts are retained.
