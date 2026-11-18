# Retired direct-summary experiment

The earlier experiment compared `routed_kv` (selected source KV sent directly to
one summary agent) with `specialist_text` (one generated natural-language handoff,
then a summary agent). A separate optional evidence/reorganization review and
full-source/selected-refill generation controls were also implemented.

These inference branches, review prompts and dedicated diagnostic entry points
have been removed. Source prefill, attention/vector routing, cache copying,
decoding and metric components are reused by the current chunk specialists.

Current variants are `chunk_kv` and `chunk_text`: both analyze original transcript
parts with global selected context and synthesize ordered local reports. The
old variant names and `--chunk-specialists`/`--review-*` options are no longer
active CLI options; use `--summary-*` for final synthesis.

Historical result directories have not been deleted. Report them separately:
they use different workflows, prompts and token costs. Do not resume an old run
with the current code or relabel old `routed_kv` results as `chunk_kv`. Git history
retains the implementation if an old experiment must be reproduced.
