# Experiment 2 — Full-transcript single-agent baseline / 全文单 agent 基线

## Purpose / 目的

Compare the existing meeting workflow against directly asking the **same Qwen3
model** to summarize the full transcript. This evaluates the overall quality–cost
trade-off of the workflow, not a pure communication-representation intervention.

对照流程为：**完整原文 → 一个 summary agent → 最终摘要**。不执行 selector、
embedding、分块 specialist 或独立 review，不加载 embedding 模型。
标准自回归生成仍会使用模型内部 KV cache，这不属于跨 agent 的 latent communication。

The main experiment is the direct baseline. A conventional map–reduce/hierarchical
baseline is not implemented. A separate three-record AMI-derived endpoint pilot
is documented in [challenge_samples/README.md](challenge_samples/README.md).

主实验为直接概括基线；常规分块汇总尚未实现。另有独立的三条 AMI 首尾关联案例，
它不是完整 challenge benchmark，也不混入真实会议自动指标。

## Shared controls / 保持一致的设置

| Setting | Default |
| --- | --- |
| Model | `Qwen/Qwen3-8B` |
| Device | `cuda` |
| Initialization seed | `12` |
| Final-generation seed | `42` |
| Temperature / top_p / top_k | `0.3 / 0.9 / 0` |
| Repetition penalty | `1.0` |
| Thinking | Enabled; use `--no-summary-thinking` to disable |
| Output structure | Key context, decisions, action items, open issues |
| Prompts | Imports shared `SYSTEM` and `SUMMARY_REQUEST` |
| Source handling | Full normalized transcript; no truncation |
| Generation termination | EOS or model context exhaustion; exhaustion is a failure |

The shared final-summary requirements preserve chronology, revisions, uncertainty,
conditions, and ownership. The baseline receives no local reports, so the
workflow-specific report-integration instructions are not appended. Model loading,
numerics, sampling, thinking splitting, quality metrics, checkpoint storage,
progress bars, and resource recording reuse existing components.

统一最终摘要要求及生成参数，但 baseline 不接收局部报告，因此不添加报告整合专用指令。
这是整体工作流比较；不能声称两边输入证据、prompt 或调用次数完全相同。
首次使用模型时按 Hugging Face 的现有加载逻辑下载，已缓存则复用。

## Pilot: the same first three AMI records / 先跑相同前三条

Run in the server project root, after syncing this implementation:

```bash
conda activate latent-agents

python -m experiment_2 \
  --input input/experiment_1_unfiltered_cap100/ami.jsonl \
  --benchmark ami \
  --output-dir test_result/experiment_2_ami_direct_3_seed42 \
  --num-samples 3 \
  --model Qwen/Qwen3-8B \
  --device cuda \
  --seed 12 \
  --summary-seed 42 \
  --summary-temperature 0.3 \
  --summary-top-p 0.9 \
  --summary-thinking
```

Input is the existing JSONL with `id`, `transcript`, and `summary`; normalized
`context`/`answer` aliases are also supported by the shared loader. Original IDs
are preserved. One model is loaded for the entire batch. No extra preprocessing
or separate `.txt` export is needed. Default `--num-samples` is **3**; use **100**
for the formal AMI run or **0** for every available record.

输入仍为原来的 AMI JSONL，保持其原始顺序和 ID。整批只加载一次模型。
前三条应与实验1相同（当前数据为 ID 82、38、101）；代码不硬编码这些 ID。
先检查试跑，再将 `--num-samples` 改为 `100` 并使用新的 output-dir。

## Output / 输出

Each invocation creates a timestamped run directory:

```text
test_result/experiment_2_ami_direct_3_seed42/<timestamp>/
  manifest.json                  # input/code hashes, settings and IDs
  runtime.json                   # model revision and GPU
  progress.json                  # planned/completed/failed
  samples/00000.json             # source, reference, answer, metrics
  summaries/00000/direct_text.md
  summaries/00000/direct_text.thinking.md
  summaries/00000/reference.md
  live/00000/direct_summary.partial.txt
  live/00000/direct_summary.status.json
  summary.txt
  summary.json
  per_sample.csv
```

`summary.*` contains overall and five length-group results. Thinking is stored
separately and excluded from ROUGE/BERTScore. Its tokens still count as generated
output. Live outputs remain inspectable on failure. Failed records are excluded
from mean quality/cost and retained in counts and checkpoints.

`--resume <timestamped-run-dir>` reuses completed records and retries failures
under matching code, inputs, settings, package versions, and runtime. Do not
resume a run after changing its protocol.

## Offline BERTScore / 离线补算

Generation is not rerun. Scoring model and settings should match Experiment 1:

```bash
pip install -r experiment_1/requirements-quality.txt

RUN_DIR=$(find test_result/experiment_2_ami_direct_3_seed42 \
  -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)

python -m experiment_2.rescore \
  --run-dir "$RUN_DIR" \
  --bertscore-model allenai/longformer-base-4096 \
  --bertscore-layers 12 \
  --device cuda
```

Scores are written to `evaluation/<timestamp>/`. The encoder compares final
summaries to references, not entire transcripts; overlong encoder inputs raise
instead of being silently truncated. `idf=False`, `rescale_with_baseline=False`.
Do not use `experiment_1.rescore` for direct-only checkpoints.

## Paired comparison with the existing workflow / 与原工作流配对

```bash
WORKFLOW_RUN="test_result/experiment_1_auto1024_cap100_seed12/ami/20261119_083659_646986"

python -m experiment_2.compare \
  --run-dir "$RUN_DIR" \
  --workflow-run "$WORKFLOW_RUN" \
  --workflow-variant chunk_kv
```

This uses the matching successful subset of the existing workflow, even if that
run has 100 records and the direct pilot has only three. IDs, exact transcript,
reference, model, and final-generation settings are checked. Results appear under
`comparison/<timestamp>/` as `comparison.txt/json` and `paired_samples.csv`.
Differences are **workflow − direct**; paired bootstrap intervals use 2,000
record-level resamples with seed 7. They do not establish factual equivalence.

If both sides have offline BERTScore reports, append:

```bash
  --direct-evaluation "$RUN_DIR/evaluation/<timestamp>" \
  --workflow-evaluation "$WORKFLOW_RUN/evaluation/<timestamp>"
```

Replace the placeholders with actual evaluation directories. The comparison
checks source-run fingerprints and scoring configurations. Without scores for
both sides, no paired BERTScore difference is invented.

目前比较器读取实验1的 `manifest.json` 与 `samples/*.json`；单文件入口生成的
`ami_semantic_off_100` metrics 目录不属于该格式，暂不直接接入。已有语义增强
开启组可以直接用于本次比较，不需要重新生成。

## Accounting and interpretation / 统计口径

- The complete single prompt (source, template, requirements) is counted **once**
  as `receiver_prompt_tokens`. `source_prefill_tokens=0` means no separate prefill
  stage is added to that count, not that the source is unprocessed.
- `total_tokens = receiver_prompt_tokens + receiver_output_tokens`; selector and
  specialist events are zero because those stages do not execute.
- Pipeline time includes prompt preparation, full prompt prefill and decoding,
  but excludes model loading and offline evaluation.
- CUDA measurements cover the direct model/prompt cache. The workflow includes
  its own resident source cache and possibly an embedding model. This difference
  is part of the system comparison; memory measurement is not representation-only.
- The comparator uses the workflow's all-observed-stage peak when available,
  including shared prefill/selection, rather than claiming a full-pipeline peak
  from a variant-only value. Timing on different runtime metadata is flagged.
- ROUGE uses the same unstemmed ASCII-token implementation as Experiment 1;
  source-support is only a lexical proxy, not a factuality score. Exact repeated
  bullets are diagnostic, not semantic redundancy.
- No controlled challenge cases or human scores are synthesized by this module.

## Components and checks / 组件与验证

| Component | Responsibility |
| --- | --- |
| `components/config.py` | CLI, direct configuration, provenance |
| `components/generation.py` | Single full-source prompt and final decoding |
| `components/runner.py` | One model per batch, resources, checkpoints/live output |
| `components/reporting.py` | ROUGE, logical tokens, resources, duplication, length groups |
| `rescore.py` | Optional offline BERTScore |
| `compare.py` | Matched workflow/direct comparison |

```bash
python -B -m unittest discover -s experiment_2/tests -v
```
