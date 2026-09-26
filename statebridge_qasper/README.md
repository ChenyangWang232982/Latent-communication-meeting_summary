# StateBridge QASPER Baseline

This directory implements a same-model, training-free QASPER baseline based on
StateBridge: final-layer Sender states are aligned to the Receiver input
embedding space by whitened orthogonal Procrustes, norm calibration, and
vocabulary anchoring. No model weights, adapters, or checkpoints are trained.

The run writes a readable text report and a JSONL report. It includes five
communication conditions:

- `statebridge`: paper method, aligned continuous prefix.
- `text`: Sender message transmitted as ordinary text.
- `raw_hidden`: unaligned final hidden states, testing alignment necessity.
- `random_prefix`: distribution-matched random continuous prefix.
- `no_comm`: Receiver has the question only.

## Smoke test

```bash
python -m statebridge_qasper.run \
  --data data/qasper_latent/validation.jsonl \
  --output statebridge_qasper/output/qasper_smoke.txt \
  --num-samples 8 \
  --model Qwen/Qwen3-4B \
  --source-max-tokens 4096 \
  --prefix-tokens 64
```

`K=64` is the initial StateBridge prefix length. `Qwen/Qwen3-4B` is the
paper's smallest evaluated Qwen setting. Use the same model for Sender and
Receiver. The QASPER JSONL currently contains full paper text rather than
the earlier deleted oracle-evidence derivative; use an oracle-evidence JSONL
with the same fields if you recreate it.

Qwen3 thinking is disabled by default so the fixed generation budget is spent
on the factual handoff and final answer. Pass `--enable-thinking` only for a
longer, paper-style reasoning run; the implementation then excludes all states
up to and including `</think>` from the transmitted prefix.

## Full validation split

```bash
python -m statebridge_qasper.run \
  --data data/qasper_latent/validation.jsonl \
  --output statebridge_qasper/output/qasper_validation_k64.txt \
  --num-samples 0 \
  --model Qwen/Qwen3-4B \
  --source-max-tokens 4096 \
  --sender-max-new-tokens 256 \
  --receiver-max-new-tokens 128 \
  --prefix-tokens 64
```

The decisive comparison is `statebridge` versus `text`, `raw_hidden`,
`random_prefix`, and `no_comm`. StateBridge is meaningful only if it beats the
non-semantic prefix controls and produces answers that carry Sender-specific
information.
