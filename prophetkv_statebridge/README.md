# Query-Aware StateBridge Baseline

This module deploys the practical first stage of a `ProphetKV + StateBridge`
meeting workflow:

```text
long transcript -> query-aware chunk selection -> specialist extraction
                -> StateBridge continuous handoff -> final aggregation
```

It is deliberately labelled **ProphetKV-inspired**, not an exact ProphetKV
implementation. ProphetKV's contribution is query-driven *KV-cache reuse and
selective recomputation* in a modified serving path. The public project does
not provide that cache-fusion backend here, and Hugging Face `past_key_values`
cannot be safely concatenated across independently prefixed chunks. This
module therefore verifies the task-level premise first: whether query-aware
selection preserves the meeting facts each specialist needs.

`statebridge` is still a genuine same-model, training-free StateBridge handoff:
only final hidden states move from specialist to aggregator; no specialist text
is given to that variant. `text` and `no_comm` remain mandatory controls.

## QASPER smoke run

```bash
python -m prophetkv_statebridge.run \
  --data data/qasper_latent/validation.jsonl \
  --output prophetkv_statebridge/output/qasper_smoke.txt \
  --num-samples 1 \
  --model Qwen/Qwen3-4B \
  --chunk-tokens 512 \
  --chunk-overlap-tokens 64 \
  --top-chunks 6 \
  --prefix-tokens-per-agent 16
```

The default four specialists are `facts`, `decisions`, `actions`, and `risks`.
Each uses the question to select its own four excerpts. Their last 16 states
are concatenated, so the default StateBridge prefix has at most 64 states.

For a cheaper QASPER factual-answer check, use only the direct-answer
specialist:

```bash
python -m prophetkv_statebridge.run \
  --data data/qasper_latent/validation.jsonl \
  --output prophetkv_statebridge/output/qasper_facts_only.txt \
  --num-samples 8 \
  --roles facts \
  --top-chunks 6 \
  --prefix-tokens-per-agent 160
```

## What to compare

- `statebridge`: selected specialist information as aligned hidden states.
- `text`: the same specialist information as natural-language text.
- `no_comm`: question only.

The report records selected chunk IDs and scores, so factual misses can be
separated into retrieval failures and StateBridge-transfer failures. Only after
the selected-context pipeline has good evidence coverage is it worthwhile to
add a true ProphetKV-compatible serving backend for KV reuse and selective
recomputation.

By default, each specialist must cite short source phrases with `[chunk N]`
identifiers before reaching a conclusion (`--strict-evidence`). This makes the
handoff more auditable and reduces unsupported claims. Use
`--no-strict-evidence` only for an ablation.

## Exact cross-agent Prefix KV cache

`run_prefix_cache.py` adds a separate, real KV-cache baseline for repeated
analysis of the same source. It pre-fills one complete transcript prefix once,
then each specialist continues from a clone of that immutable `past_key_values`
cache with only its role-specific instruction appended.

```bash
python -m prophetkv_statebridge.run_prefix_cache \
  --data data/qasper_latent/validation.jsonl \
  --output prophetkv_statebridge/output/qasper_prefix_cache_smoke.txt \
  --num-samples 1 \
  --model Qwen/Qwen3-4B \
  --source-max-tokens 4096 \
  --roles facts decisions actions risks \
  --prefix-tokens-per-agent 64
```

The report includes the shared-prefix token count, one-time prefill time, and
KV-cache size. It is an **exact prefix-reuse** baseline: the source prefix is
identical for every specialist. It is not ProphetKV's selective recomputation,
and it does not attempt unsafe concatenation of independently created caches.
