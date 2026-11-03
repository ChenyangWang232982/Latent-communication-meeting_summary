# Semantic candidate topics for decision evolution / 语义关联决策演变

An optional experiment using the existing Qwen3 embedding model and the single
source prefill. `--grouped-decisions` replaces the one-pass decision-evolution
output with per-topic analysis; it does not change the summary or timeline.
It requires `--vector-routing` and automatically enables decision evolution.

## Flow / 流程

1. The normal attention-based `decisions` query selects source token positions.
   Every speaker block intersecting those positions becomes a candidate seed;
   this is a retrieval signal, not proof that a decision exists.
2. Reuse the vectors computed for **all** speaker blocks in vector routing.
   Decision seed blocks are grouped by cosine similarity to the first seed of
   each group. Every remaining source block can join its nearest representative
   if its similarity meets the threshold, even when that block was omitted from
   the summary receiver's KV.
3. Expand each group by the configured preceding/following speaker blocks,
   deduplicate positions, and keep their original transcript order. Neighbor
   expansion can make topics overlap. Frame tokens and the group's original
   source KV are selected from the one full source cache.
4. Run one receiver generation per candidate topic using that KV and the final
   natural-language summary as an untrusted draft. The prompt asks the model to
   check whether excerpts actually concern the same decision, distinguish
   proposal, change and final status, and report uncertainty. Outputs are
   concatenated in representative source order; there is no separate merge
   model or hard-coded claim filter.

向量只负责生成候选关联，不能证明同一议题、判断否定或最终结论。候选组可能
混合多个议题、重复、遗漏或因代词缺少语境产生错误；最终结果需要核对原文。
整个源 KV 保持驻留，所选 KV 位置是同一份完整前缀的切片，并非独立计算后拼接。
该流程不训练模型、不修改摘要，绝不保证组内讨论完整或全局最终结论正确。

The threshold `--decision-similarity` defaults to 0.75 as an unvalidated
experimental setting; it is not a calibrated probability. More or fewer groups
and inclusion changes can result. `--speaker-neighbor-blocks` is shared with
summary selection. There is no fixed maximum topic count; a long meeting can
require many receiver calls and run significantly longer. Outputs may repeat.

## Linux example

```bash
python -m receiver_kv_workflow.run \
  --transcript input/meeting1.txt \
  --output output/vector/meeting1_grouped.txt \
  --model Qwen/Qwen3-4B \
  --role-budget-ratio 0.25 \
  --speaker-neighbor-blocks 1 \
  --vector-routing \
  --grouped-decisions \
  --decision-similarity 0.75 \
  --variants routed_kv \
  --timeline \
  --routing-audit
```

Outputs include the existing summary, `.timeline.txt`, grouped `.decisions.txt`
and `.audit.json`/`.audit.txt`. The JSON audit and `metrics.json` contain seed,
matched and expanded block IDs, similarity scores, per-topic KV position counts,
generation time and cap hits. The text audit continues to show normal routing;
use the JSON for topic groups. Embedding inputs and time are counted in vector
routing, reused rather than recalculated for groups. Grouping and receiver calls
are additional; prompt tokens are summed across calls, and model context limits
still apply. The intermediate topic vectors, source text and KV are not copied
into the decision prompt as text; only the final summary is a text handoff.

For comparison, replace `--grouped-decisions` with `--decision-evolution` and
keep all other options identical. Assess per-decision coverage, source evidence,
supersession and participant/negation accuracy manually or against labelled
data, together with total retained KV positions and added time/tokens. Comparing
one meeting and lexical overlap alone does not establish an improvement.
