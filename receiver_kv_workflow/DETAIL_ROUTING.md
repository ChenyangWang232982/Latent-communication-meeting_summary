# Boundary-aware routed KV experiment

Use `--selection-strategy sentence` on either `receiver_kv_workflow.run`
or `meeting_evaluation.test`. Default `fixed` preserves the old selector.

The new strategy computes the original role scores and fixed-span union first.
It uses the size of that union as an upper bound. Transcript character offsets
map punctuation/paragraph-delimited units to the exact source KV token indices.
Each role's scores are normalized, then max-pooled across roles. Complete units
are ranked by mean relevance and retained only if they fit. Prompt framing stays
protected. Unused budget is left unused; long units are skipped, never cut to fit.
Original RoPE positions and exact stored KV values remain unchanged.

No extra query, generated text evidence, training or receiver prompt changes are
introduced. Text controls use the same resulting selection. This is a heuristic
sentence boundary method: abbreviations and unpunctuated ASR can produce imperfect
units. Cross-sentence attribution and corrections may still be lost. It is not
a factuality guarantee. Check actual retained positions as well as nominal ratio.

Research basis:
- ChunkKV (2025): https://arxiv.org/abs/2502.00299 advocates retaining chunks to
  avoid losing dependencies among tokens. Our baseline already uses fixed spans;
  this experiment changes their boundaries and disallows partial units.
- SemantiCache (2026): https://arxiv.org/abs/2603.14303 uses delimiters for semantic
  chunks, then clustering/merging and proportional attention. We adopt only the
  boundary idea; no clustering, KV merging or attention correction is reproduced.
- RocketKV (2025): https://proceedings.mlr.press/v267/behnam25a.html combines
  coarse eviction and fine sparse attention. Its decode-time machinery is a larger
  system change, so it is not implemented in this controlled boundary experiment.

Evaluate exact meeting details (owner, date, complete time range, approval status)
against source evidence on held-out meetings; ROUGE alone cannot establish detail
preservation. Compare fixed/sentence at identical model, input, ratio and output
cap. The new strategy can use fewer KV positions. Also compare full_kv to determine
whether a missing fact is caused by selection or the summarizer itself.

Related execution fix: natural_text previously entered the generic KV receiver
loop before its sender/receiver path, generating an unused extra answer. It now
runs only its intended path and includes selection time in pipeline accounting.
