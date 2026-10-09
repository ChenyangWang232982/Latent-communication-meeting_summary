# Meeting Evaluation Harness

This folder evaluates the current **Prefix-KV + specialist + StateBridge** workflow without placing transcript excerpts, specialist handoffs, or generated summaries in the final report.

```text
input -> optional ASR -> benchmark adapter -> Prefix-KV + StateBridge workflow
      -> quality / factuality-proxy / communication / KV metrics -> timestamped output
```

`input/` holds local benchmark files. Results are written as
`output/YYYYMMDD/<tool>.txt` and `output/YYYYMMDD/<tool>.json`, for example
`output/20261009/qmsum.txt`. Re-running the same tool on the same day replaces
that tool's report. Both folders are ignored by Git except for their `.gitkeep`
files.

## Supported Inputs

- `--benchmark qmsum`: official QMSum JSON with `meeting_transcripts` and query lists.
- `--benchmark meetingbank`: MeetingBank export with transcript and summary fields, or normalized JSONL.
- `--benchmark ami`: one UTF-8 AMI transcript; add `--reference` for ROUGE-L.
- `--benchmark jsonl`: one normalized JSON/JSONL record per line: `id`, `context`, `question` (optional), `answer` (optional).

With `--enable-asr`, `--input` must instead be an audio file. The optional ASR component uses `faster-whisper`; install it separately with `pip install faster-whisper`.

## Commands

QMSum main evaluation:

```bash
python -m meeting_evaluation.test \
  --benchmark qmsum \
  --input meeting_evaluation/input/qmsum_test.json \
  --output-dir meeting_evaluation/output \
  --num-samples 8 \
  --model Qwen/Qwen3-4B \
  --source-max-tokens 8192 \
  --prefill-chunk-tokens 1024 \
  --prefix-tokens-per-agent 64
```

MeetingBank generalization evaluation:

```bash
python -m meeting_evaluation.test \
  --benchmark meetingbank \
  --input meeting_evaluation/input/meetingbank_test.jsonl \
  --num-samples 8
```

AMI qualitative case with a reference summary:

```bash
python -m meeting_evaluation.test \
  --benchmark ami \
  --input prophetkv_statebridge/input/ami_ES2002a.txt \
  --reference meeting_evaluation/input/ami_ES2002a_reference.txt
```

Audio input:

```bash
python -m meeting_evaluation.test \
  --benchmark ami \
  --enable-asr \
  --input meeting_evaluation/input/meeting.wav \
  --asr-model small
```

## Run All Benchmarks

After placing the standard inputs at `input/qmsum_test.jsonl` and
`input/meetingbank_test.jsonl`, run all three benchmark workflows in sequence:

```bash
python -m meeting_evaluation.run_all --num-samples 8
```

This runs QMSum, MeetingBank, and the bundled AMI transcript. Each run includes
the `statebridge`, `text`, and `no_comm` controls, communication/KV metrics,
and the MESA-style source-overlap proxy. It writes only final metrics to:

```text
meeting_evaluation/output/YYYYMMDD/qmsum.txt
meeting_evaluation/output/YYYYMMDD/meetingbank.txt
meeting_evaluation/output/YYYYMMDD/ami.txt
```

To run the inputs you currently have while skipping unavailable benchmarks:

```bash
python -m meeting_evaluation.run_all --num-samples 8 --skip-missing
```

## Metrics

Each report has only final metrics:

- `ROUGE-L F1`: reference-overlap quality where a gold summary exists.
- `source-support proxy`: transparent lexical claim/source overlap. It is a lightweight MESA-inspired factuality check, **not** a reproduction of MESA's LLM-based evaluator.
- `text handoff tokens`, `latent states`, and their signed difference. Negative means StateBridge transmitted fewer token-equivalent sequence units.
- shared Prefix-KV prefill time and cache size.

For a publication result, use the source-support proxy for debugging and add MESA or human annotation as a separate factuality study.
