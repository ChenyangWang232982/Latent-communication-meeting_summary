# FRAME-inspired statement + KV context / 陈述与 KV 上下文

This optional mode adapts the statement–context idea from
[FRAME (Findings EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.1094/).
It is not a full reproduction: no fact scoring, relevance threshold, LLM judge or
iterative topic discovery is added.

```text
anchored original transcript → one full prefill → existing selectors + vector routing
 → routed KV → statement extraction {statement, context_refs}
 → chronological deduplicated source KV referenced by those facts + statements
 → final four-section summary
```

用 `--frame-context` 启用；不启用时继续使用原流程。启用后以“事实提取 → 摘要生成”
替代“摘要 → review”，共两次生成。最后一步接收自然语言陈述与原文 KV，
不使用模型改写的 context 段落，也不传递提取阶段的生成 KV。

`[B0]` 等锚点在 prefill 前插入原始发言块，模型引用同一套编号。它们标识整个发言块，
包括锚点前的发言人标题。锚点会增加输入并影响选择，不能把新旧流程视为严格单变量比较。
`context_refs` 应包含指代、归属、条件、后续修订等相关发言，最终摘要还会收到陈述与锚点映射。
引用是模型提出的证据位置，不是正确性证明。只能引用已选入 routed KV 的块；
若 JSON 或引用无法解析，保留完整原始报告和全部原选中 KV，记录回退原因，避免静默丢信息。
这个回退不保证模型输出正确。输入不会静默截断，也没有新增条数或输出 token 上限。

```bash
python -m receiver_kv_workflow.run \
  --transcript input/meeting1.txt \
  --output output/frame/meeting1.txt \
  --model Qwen/Qwen3-8B \
  --device cuda \
  --role-budget-ratio 0.25 \
  --speaker-neighbor-blocks 1 \
  --vector-routing \
  --variants routed_kv \
  --frame-context \
  --seed 12 \
  --review-seed 42 \
  --review-temperature 0.3 \
  --review-top-p 0.9
```

`--seed` controls fact extraction, while `--review-*` controls final synthesis in
this mode. Thinking stays disabled. The mode currently supports only `routed_kv`;
it is not registered as a paired communication experiment. `--review` does not add
a third pass in FRAME mode.

Outputs: final summary at `--output`; sibling `.facts.json` contains fact records,
exact original-KV positions per block, final context positions, fallback reason,
per-stage tokens, time and KV bytes. `.facts.raw.txt` keeps the original extraction.
The usual metrics files record both stages. Decoder accounting includes source
prefill, selection probes, extraction prompt/output and synthesis prompt/output;
embedding tokens remain separate. Payload bytes sum both stages' KV tensors and
serialized statement/anchor metadata, and are logical accounting rather than network traffic.
The full original cache remains resident, so referenced-KV reduction is not peak-memory reduction.

```bash
python -m unittest discover -s frame_workflow/tests
```
