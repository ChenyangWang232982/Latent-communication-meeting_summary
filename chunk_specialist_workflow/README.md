# Selected-context window specialists / 窗口 KV specialists

已恢复窗口方案：**每个 specialist 只接收当前窗口的 selected KV＋分析要求**。
不额外附加当前块原文，也不向每个 specialist 发送全局 selected KV。
只有最终汇总读取全部 selected KV＋按顺序排列的局部报告。

```text
Transcript → one source KV prefill → attention + optional semantic routing
                                          ↓ selected source positions
                            chronological overlapping speaker-block windows
                                          ↓
                 each specialist: its own selected window KV + instructions
                                          ↓ ordered natural-language reports
                 final synthesis: all selected KV + reports (optional thinking)
```

Target window count is `max(1, floor(n/1024 + 0.5))`, where n is the raw full
transcript token count, including labels/timestamps but excluding chat framing.
This is round-half-up, not Python's ties-to-even `round`. Selected complete speaker
blocks are balanced into that many base windows; fewer available blocks means fewer
windows. Long turns stay intact. `--specialist-overlap 0.25` expands backward at
speaker boundaries without increasing the number of specialists. Discarded rows
are not restored. The old `--specialist-parts` argument has been removed.
Absolute source positions and chronological order are preserved. Specialists
are told their window number and total, and must not assume a local proposal
is the final meeting-wide outcome. Final synthesis resolves supported revisions.

Each stage gets a disposable writable cache copied from the immutable source KV.
Models share weights and execute sequentially. Generated specialist KV is never
concatenated. Local reports are natural language, so this is hybrid communication.
The full original source KV remains resident; no peak-memory saving is guaranteed.

```bash
python -m receiver_kv_workflow.run \
  --transcript input/meeting1.txt \
  --output output/window_kv/meeting1.txt \
  --model Qwen/Qwen3-8B \
  --device cuda \
  --role-budget-ratio 0.25 \
  --speaker-neighbor-blocks 1 \
  --vector-routing \
  --variants chunk_kv \
  --specialist-overlap 0.25 \
  --summary-thinking \
  --seed 12 \
  --summary-seed 42
```

Local defaults: temperature0.3, top_p0.9, repetition penalty1.1; local seed is
`seed + window_index`. Final defaults: temperature0.3, top_p0.9, penalty1.0 and
thinking enabled, controlled by `--summary-*`. Local specialists remain non-thinking.
Final reasoning is saved separately and excluded from summary quality scoring,
but its token cost is counted. No separate review or arbitrary output cap is added;
context exhaustion raises an error.

`chunk_text` supplies the exact same local window positions as verbatim selected
text, and final synthesis receives all selected evidence as text plus its ordered
local reports. Both paths share window definitions, instructions and decoding
settings. Full-source KV retains contextual/positional history different from
re-encoded selected text; this is a workflow comparison, not pure representation
isolation. `experiment_1` runs paired comparisons of these two methods.

`.specialists.json` includes `input_layout=selected_context_windows`,
`transcript_tokens`, `target_specialist_parts`, `actual_specialist_windows`,
`synthesis_selected_positions`, local window `positions`/`block_ids`, reports,
stage token counts and timings. Local KV sizes may differ; only the final stage
receives all selected rows. Per-stage partial/status files support inspection.

Logical decoder totals include prefill/probes once per standalone workflow, all
specialist prompts and outputs, and final prompt/output including thinking. KV
rows are reused, not counted as newly encoded prompt tokens. Embedding tokens
are separate. Logical payload sums stage KV tensors or evidence UTF-8 bytes plus
serialized local-report UTF-8 bytes, not measured network traffic or billing.

```bash
python -m unittest discover -s chunk_specialist_workflow/tests
```
