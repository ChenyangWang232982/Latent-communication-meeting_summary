# Speaker-labelled benchmark preparation / 数据预处理

当前规则：规范化并验证数据，不按会议长度筛选，每个数据集最多保留100条。
不足100条就全部导出，不复制、不补样本，也不因为不足上限而报错。
超过100条时使用固定 seed7 无放回抽样。`--max-records 0` 可导出全部合格记录。
没有任何有效记录仍会报错，不会复制会议或编造说话人。

| Evaluation name | Source | Sampling unit |
|---|---|---|
| `ami` | QMSum Product / AMI | One distinct meeting, first general-summary reference |
| `icsi` | QMSum Academic / ICSI | One distinct meeting, first general-summary reference |
| `qmsum_committee` | QMSum Committee | One distinct meeting, first general-summary reference |
| `meetingbank` | Official diarized MeetingBank | One agenda record; multiple records may share a meeting |

These QMSum domains are non-overlapping. Do not add the combined QMSum collection
as another independent dataset. Domain `all/` includes multiple official splits;
results are a capped training-free comparison across all lengths, not official test-only
scores. Using query-specific examples or duplicated records to inflate counts
would change the evaluation task and independence assumptions.

## Prepare all four datasets / 准备四个数据集

Run from the project root using Bash. Existing outputs are protected; choose a new
directory or use `--overwrite` only when intentionally rebuilding an export.
The input package paths must already exist. QMSum can be obtained with
`git clone https://github.com/Yale-LILY/QMSum.git external/QMSum`.
MeetingBank requires its official extracted package including `Metadata` and
speaker-labelled `itemInfo.transcripts`; the usual flattened HF text is insufficient.

```bash
conda activate latent-agents

for pair in ami:Product icsi:Academic qmsum_committee:Committee; do
  benchmark=${pair%%:*}
  domain=${pair#*:}
  python -m data_preprocess.prepare_qmsum \
    --input "external/QMSum/data/$domain/all" \
    --output "input/experiment_1_unfiltered_cap100/$benchmark.jsonl" \
    --max-records 100 \
    --seed 7 \
    --tokenizer-model Qwen/Qwen3-8B
done

python -m data_preprocess.prepare_meetingbank \
  --input external/MeetingBank \
  --split test \
  --output input/experiment_1_unfiltered_cap100/meetingbank.jsonl \
  --max-records 100 \
  --seed 7 \
  --require-speakers \
  --tokenizer-model Qwen/Qwen3-8B
```

Defaults are max100, tokenizerQwen3-8B, minimum0 and no maximum transcript length.
Only the tokenizer is loaded to count tokens, not generation weights. Speaker labels and
timestamps count; reference summaries and chat templates do not. No transcript is
truncated. Empty annotation turns can be skipped, but unlabelled nonempty QMSum
turns invalidate that meeting instead of inventing speakers.

Outputs use JSONL with `id`, `transcript`, `summary` and `transcript_tokens` (plus
optional `question`). The sibling `.jsonl.report.json` records requested cap,
actual exported size, eligible pool, exclusions, tokenizer, bounds and sample seed.
QMSum is deduplicated by meeting ID before sampling. Existing exports are not
automatically enlarged or rebuilt when defaults change.

## Experiment 1 / 四个数据集分别运行

Each call evaluates both `chunk_kv` and `chunk_text`. The cap is per dataset, so
total evaluated records are the sum of actual available counts, at most400.
Generation seed12 is separate from preprocessing sample seed7. A `manifest.json`
records available input records, requested limit and actual selected count.

```bash
for benchmark in ami icsi qmsum_committee meetingbank; do
  python -m experiment_1 \
    --input "input/experiment_1_unfiltered_cap100/$benchmark.jsonl" \
    --benchmark "$benchmark" \
    --output-dir "test_result/experiment_1_auto1024_cap100_seed12/$benchmark" \
    --num-samples 100 \
    --model Qwen/Qwen3-8B \
    --device cuda \
    --role-budget-ratio 0.25 \
    --speaker-neighbor-blocks 1 \
    --vector-routing \
    --specialist-overlap 0.25 \
    --summary-thinking \
    --seed 12 \
    --summary-seed 42
done
```

Report datasets separately with actual sample counts; do not claim100 for a domain
with fewer eligible records. See [experiment_1](../experiment_1/README.md) for
decoding defaults, checkpoint/resume, token definitions and offline BERTScore.

```bash
python -m unittest discover -s data_preprocess/tests
```
