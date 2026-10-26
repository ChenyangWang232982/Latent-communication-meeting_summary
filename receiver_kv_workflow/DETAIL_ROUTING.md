# Boundary-aware routed KV experiment

## Routing diagnosis and controlled comparison

`receiver_kv_workflow.run --routing-audit` writes separate `.audit.txt` and
`.audit.json` beside the content output. It contains exact selected source
excerpts, zero-based half-open source token and transcript character intervals,
per-role fixed selections, differences from fixed, and complete/partial/dropped
sentence coverage. Dropped source is explicitly diagnostic context, never model
input. Audit rendering occurs after generation and is excluded from inference
timings. Metrics files do not contain this source text.

Compare `fixed`, `fixed_pooled`, and `sentence`:
- fixed: original per-role fixed spans, partial last span allowed, then union.
- fixed_pooled: globally pooled role scores, whole fixed spans only.
- sentence: same pooled scoring, budget ceiling and whole-unit policy as
  fixed_pooled, but delimiter-based sentence units.

`fixed_pooled` versus `sentence` isolates the unit boundaries more closely.
`fixed` versus `fixed_pooled` changes both pooling and whole-unit selection, so
it is not a pure pooling ablation. Actual counts can differ below the same ceiling.
No prompts are changed. First inspect whether the desired fact and its conditions
are retained, then compare full_kv to distinguish selection loss from generation.

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

# Spoken repairs and summary detail preservation

## Speaker blocks

Use `--selection-strategy speaker` in the content runner or MeetingBank evaluation.
Speaker blocks are constructed before source prefill and attention scoring.
Consecutive records with the same normalized speaker label form one block;
a return to that speaker after another speaker starts a separate block.
Supported line-start labels include `Alice: text`, `[10:30] Alice: text`, and
`Alice (Team)  1:10:30` followed by text. Unlabelled preamble is a separate block;
inputs without recognized labels fail explicitly. This parser relies on transcript
formatting, not speaker recognition or semantic inference.

The selector first scores whole blocks using pooled role attention and selects
seed blocks within the original per-role fixed-selection union budget. It then
retains the two preceding and two following whole speaker blocks around every
seed (`--speaker-neighbor-blocks`, default `2`). Context expansion can exceed
the seed budget, so the final payload is reported separately from seed positions.
No extra model pass is added. Whole blocks that do not fit the seed budget are
skipped; no speaker block is subdivided. Audit coverage is labelled SPEAKER_BLOCK
(the JSON `sentence_coverage` key is retained for compatibility, with
`coverage_unit`). Selected text and natural text use the same chosen positions.

Sentence units keep adjacent explicit self-corrections (for example, "failed.
Sorry, executed.") together. This lexical heuristic prevents selecting the
retracted phrase without its correction. The combined unit must fit the existing
position budget; it can be skipped as a whole. Speaker-labelled new turns remain
separate. This is not a semantic guarantee for every form of spoken correction.

All summary variants share a request to preserve supported owners, dates, both
endpoints of time ranges and pending conditions, and to resolve explicit
self-corrections and changed statuses. Compare variants under the same prompt
version; older runs are not controlled comparisons with this revision. Actual
summary improvements require model evaluation, not just unit tests.
