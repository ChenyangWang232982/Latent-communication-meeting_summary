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

## LaTER Sender + StateBridge transfer

`run_later_sender` is a combined *ablation*, not a claim that LaTER was
evaluated on QASPER by its authors.  It uses the official LaTER training-free
latent/explcit switching routine inside the Sender.  The Sender's final
**explicit** handoff is then transferred to the Receiver through StateBridge;
the Receiver never receives that handoff as text in the `statebridge` variant.

First obtain the official implementation and install its dependencies in the
same environment:

```bash
git clone https://github.com/TioeAre/LaTER.git external/LaTER
pip install -r external/LaTER/requirements.txt
```

Then use one sample first.  `text` is the same LaTER-produced handoff sent as
ordinary text, while `no_comm` tests the Receiver without it.

```bash
python -m statebridge_qasper.run_later_sender \
  --data data/qasper_latent/validation.jsonl \
  --output statebridge_qasper/output/later_statebridge_smoke.txt \
  --later-root external/LaTER \
  --model Qwen/Qwen3-4B \
  --num-samples 1 \
  --source-max-tokens 8192 \
  --prefix-tokens 64 \
  --variants statebridge text no_comm
```

The JSONL report records the number of explicit LaTER tokens and the actual
StateBridge prefix length.  Do not compare a failed factual handoff only at the
Receiver: compare the `LaTER explicit handoff` with the gold answer first.  If
that handoff is wrong, StateBridge cannot recover the missing fact.

## One sample, multiple K values

The following command generates the Sender message once for the selected
zero-based JSONL line, then evaluates suffix lengths `K=16,32,64,96,128` on
that exact same message.

```bash
python -m statebridge_qasper.sweep_k \
  --data data/qasper_latent/validation.jsonl \
  --output statebridge_qasper/output/qasper_sample0_k_sweep.txt \
  --sample-index 0 \
  --k-values 16 32 64 96 128 \
  --source-max-tokens 4096
```

For the longer `1908.05441_0_0` example, which previously lost `ARC` at
`K=64`, use its id and an 8192-token source window. This validation JSONL's
longest contexts fit within that window, so Sender reads the full paper text.

```bash
python -m statebridge_qasper.sweep_k \
  --data data/qasper_latent/validation.jsonl \
  --output statebridge_qasper/output/qasper_1908_05441_k_sweep_fulltext.txt \
  --sample-id 1908.05441_0_0 \
  --k-values 16 32 64 96 128 \
  --source-max-tokens 8192
```
