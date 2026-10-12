# Meeting Evaluation Harness

This folder evaluates the current **Prefix-KV + specialist + StateBridge** workflow without placing transcript excerpts, specialist handoffs, or generated summaries in the final report.

```text
input -> optional ASR -> benchmark adapter -> Prefix-KV + StateBridge workflow
      -> quality / factuality-proxy / communication / KV metrics -> timestamped output
```

`input/` holds local benchmark files. Results are written as
`output/YYYYMMDD_HHMM/<tool>.txt` and `output/YYYYMMDD_HHMM/<tool>.json`, for
example `output/20261009_1437/qmsum.txt`. Re-running the same tool in the same
minute replaces that tool's report. Both folders are ignored by Git except for
their `.gitkeep` files.

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
meeting_evaluation/output/YYYYMMDD_HHMM/qmsum.txt
meeting_evaluation/output/YYYYMMDD_HHMM/meetingbank.txt
meeting_evaluation/output/YYYYMMDD_HHMM/ami.txt
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

## State Selection

The fixed latent budget may be selected in four ways:

- `tail` (default): use the final K generated handoff states, matching the original baseline.
- `stratified`: uniformly sample a later-middle region and preserve the final
  half as a contiguous span. With `K=64`, this is 32 later-middle and 32 tail
  states; the earliest states are not forced into the prefix.
- `dynamic_compact`: in one longer Sender generation, produce a detailed
  candidate whose items are marked P2/P1/P0. The communication layer admits
  original generated states in priority order (P2, then P1, then P0), preserves
  their chronological order, and sends at most K states. This remains one
  generation per specialist; use `--agent-max-new-tokens` greater than K (128
  generated tokens with K=64 is the default evaluation setting).
- `dynamic_compact_two_pass`: generate a private candidate, then use a second
  Sender generation to rank and rewrite it into at most K tokens. The candidate
  is never reported. This may improve selection but doubles specialist
  generation passes.

Run a fixed-bandwidth ablation with:

```bash
python -m meeting_evaluation.run_all --num-samples 8 \
  --state-selection-strategy stratified
```

For a dedicated, single-report MeetingBank comparison of `tail`, `stratified`,
and `dynamic_compact`, run:

```bash
python -m meeting_evaluation.compare_state_selection \
  --input meeting_evaluation/input/meetingbank_test.jsonl \
  --num-samples 8 \
  --model Qwen/Qwen3-4B \
  --source-max-tokens 8192 \
  --prefill-chunk-tokens 1024 \
  --prefix-tokens-per-agent 64
```

The result is written to
`output/YYYYMMDD_HHMM/meetingbank_state_selection.txt` and includes only final
metrics for `tail`, `stratified`, and the one-pass `dynamic_compact` strategy.
