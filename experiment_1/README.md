# Experiment 1: KV routing vs natural-language handoff

This standalone experiment reuses `ReceiverKVRunner`, the normalized record loader and the existing ROUGE-L/source-overlap metrics. Experiment-specific code lives in `components/`. Both paths use the same source cache and selected source positions from one runner call. **Review is disabled for both paths.** `specialist_text` adds a generated natural-language handoff, so this measures two communication workflows rather than an isolated representation change. There is no full-source baseline in this experiment.

## Run / 运行

From the repository root, with the prepared speaker-labelled JSONL:

```bash
conda activate latent-agents
python -m experiment_1 \
  --input input/meetingbank_speaker_test.jsonl \
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
