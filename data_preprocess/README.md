# Speaker-Labelled Benchmark Preparation

The speaker-block workflow requires a transcript in which every turn begins
with a speaker label. The controlled evaluation suite uses four non-overlapping
domains, with 25 meetings per domain by default:

| Evaluation name | Source | Required export |
|---|---|---|
| `ami` | QMSum Product / AMI | `input/ami_test.jsonl` |
| `icsi` | QMSum Academic / ICSI | `input/icsi_test.jsonl` |
| `qmsum_committee` | QMSum Committee only | `input/qmsum_committee_test.jsonl` |
| `meetingbank` | Official diarized MeetingBank package | `input/meetingbank_speaker_test.jsonl` |

QMSum already includes AMI, ICSI, and committee meetings. Do not additionally
evaluate a combined QMSum set: that would duplicate AMI and ICSI.

## QMSum domains

Clone the official repository, then export exactly 25 general-summary meetings
from each domain. The adapter preserves the official `speaker` field as lines
such as `Alice: utterance`.

```bash
git clone https://github.com/Yale-LILY/QMSum.git external/QMSum

python -m data_preprocess.prepare_qmsum \
  --input external/QMSum/data/Product/all \
  --output input/ami_test.jsonl \
  --max-records 25

python -m data_preprocess.prepare_qmsum \
  --input external/QMSum/data/Academic/all \
  --output input/icsi_test.jsonl \
  --max-records 25

python -m data_preprocess.prepare_qmsum \
  --input external/QMSum/data/Committee/all \
  --output input/qmsum_committee_test.jsonl \
  --max-records 25
```

The domain test splits may contain fewer than 25 distinct meetings (notably
ICSI), so this controlled training-free comparison samples 25 distinct meetings
from each domain's official `all` file with seed 7. It is a balanced comparative
sample, not a claim of an official held-out test score. Use `--overwrite` only
when deliberately rebuilding an existing export.

## MeetingBank

The ordinary Hugging Face MeetingBank split consolidates turns into one text
paragraph, so it cannot validly support speaker-block selection. Download and
extract the official MeetingBank package first. The adapter directly joins its
official split records with the matching diarized `itemInfo.transcripts`; no
hand-built `meetingbank_diarized.jsonl` is needed.

```bash
python -m data_preprocess.prepare_meetingbank \
  --input external/MeetingBank \
  --output input/meetingbank_speaker_test.jsonl \
  --max-records 25 \
  --require-speakers
```

`--require-speakers` intentionally rejects rows lacking turn-level labels rather
than silently converting them into an invalid speaker benchmark.

## Run all four domains

```bash
python -m meeting_evaluation.run_all \
  --num-samples 25 \
  --speaker-neighbor-blocks 1 \
  --receiver-kv-role-budget-ratio 0.25 \
  --model Qwen/Qwen3-4B \
  --device cuda
```

Use `--skip-missing` while only some prepared datasets are available.

## Transcript length baseline / ??????

Both preprocessing commands now default to **4,000?8,000 tokens inclusive**,
counted with `Qwen/Qwen3-8B`'s tokenizer on the complete normalized transcript.
Speaker labels and timestamps are included; reference summaries, chat templates,
and model outputs are excluded. Only tokenizer files are loaded, not model weights.
No transcript is truncated. Invalid records are removed, length filtering runs
on all valid candidates, and deterministic sampling happens **after** filtering.
This is a length-controlled subset, not the complete benchmark or a guaranteed
30-minute meeting subset. MeetingBank records may represent agenda items.

???????????????????????????????????
 tokenizer ???????????????????????????
??? tokenizer??????????????????????????
??? `--max-records`???????????????????????

```bash
python -m data_preprocess.prepare_meetingbank \
  --input external/MeetingBank \
  --output input/meetingbank_speaker_4k_8k_test.jsonl \
  --max-records 25 \
  --require-speakers \
  --tokenizer-model Qwen/Qwen3-8B \
  --min-transcript-tokens 4000 \
  --max-transcript-tokens 8000
```

QMSum/AMI/ICSI/Committee use the same token arguments and defaults.
Use local model directories with `--tokenizer-model /path/to/Qwen3-8B` when offline.
Each exported row includes `transcript_tokens`. The adjacent `.jsonl.report.json`
records bounds, tokenizer, exclusion counts, eligible pool size, seed, and selected
length statistics. Existing exports are not automatically rebuilt. Use a new output
filename to preserve earlier experiments; use the filtered filename as experiment input.

QMSum ?????????????? `transcript_tokens` ??????????
?????????????????????????????????????
????????????? 25 ????? QMSum ???????????????
