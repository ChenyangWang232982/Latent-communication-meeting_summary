# Semantic-assisted KV block selection / 向量辅助 KV 块选择

This optional, training-free experiment reranks whole speaker blocks. It does
not replace KV communication with retrieved text, and is not a reproduction of
a published routing algorithm. No vector database is needed.

这是可开关的实验模块：用向量辅助选择完整发言块，最终仍传原始 KV。
不会生成新结论，也不判断“赞同/否定/最终决定”。原路由默认保持开启；
只有传入 `--vector-routing` 才使用本模块。

## Model / 模型

Default: `Qwen/Qwen3-Embedding-0.6B`, a multilingual embedding model. Uses
Transformers already required by this project; no sentence-transformers
dependency. CPU is the default to avoid competing with the summarization model
for GPU memory; this model can take more memory and time than E5-small. The
adapter uses Qwen3-specific pooling and is not an E5 adapter.

Model instructions: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

```bash
python -m vector.download
```

Downloads weights/tokenizer via Hugging Face; no model execution or training.
The default local path is `models/Qwen3-Embedding-0.6B`. Alternatively pass
`--embedding-model Qwen/Qwen3-Embedding-0.6B` to download through Transformers
on first use. Local model directories are ignored by Git.

## Algorithm / 方法

1. Existing source prefill and role attention probes remain unchanged.
2. Existing attention selection produces seed speaker blocks.
3. Encode every speaker block symmetrically as text without a retrieval query
   instruction. Pool the last non-padding hidden state from each window, in line
   with the Qwen3 model card. Blocks exceeding the practical 8192-token window
   are split into consecutive windows, combined using content-length weights,
   then L2 normalized. The model supports 32K positions; the smaller window
   bounds per-call resource use and is not a transcript truncation limit. All
   content tokens are processed, but windowing can lose long-range relationships
   within a very long turn. Pooling multiple windows is a workflow heuristic.
4. Each block gets the maximum cosine similarity to an attention seed **other
   than itself**, including distant seeds. No iteration or extra LLM generation.
5. Min-max normalize attention block scores and similarity scores separately.
   `combined = (1-weight)*attention + weight*semantic`. Default weight `0.3` is
   a provisional experiment setting, not a validated optimum. Constant scores
   normalize to zero; deterministic ties use source order.
6. Greedily select complete blocks under the original seed-position budget;
   then apply the existing neighbor expansion, union and chronological ordering.
   Weight zero preserves baseline selection. Both communication variants use
   the same final positions. A block too large for the remaining budget may
   still be skipped; embeddings do not eliminate this constraint.

向量排序发生在邻块扩张之前，只改变块的优先级。注意力种子与语义相关的
远距离块有机会保留；不承诺修复所有后续决策遗漏。先前方案会保留高分早期
块，本方案也可能偏向已有主题，不能把相似度当成决策正确性或时间先后。

## Run / 运行（Linux）

```bash
python -m receiver_kv_workflow.run \
  --transcript input/meeting1.txt \
  --output output/meeting1_vector.txt \
  --model Qwen/Qwen3-4B \
  --role-budget-ratio 0.25 \
  --speaker-neighbor-blocks 1 \
  --vector-routing \
  --embedding-model models/Qwen3-Embedding-0.6B \
  --device cuda \
  --vector-weight 0.3 \
  --variants routed_kv \
  --routing-audit
```

The shared `--device` selects the device for both the language model and the
embedding model. Use `--device cpu` to run both on CPU.

Remove `--vector-routing` and change output filename for the attention-only
control. Keep all other flags identical. The embedding stage itself does not
change prompts.

```bash
python -m meeting_evaluation.test \
  --benchmark meetingbank \
  --input input/meetingbank_speaker_test.jsonl \
  --output-dir test_result/vector \
  --num-samples 25 \
  --receiver-kv-role-budget-ratio 0.25 \
  --speaker-neighbor-blocks 1 \
  --vector-routing \
  --embedding-model models/Qwen3-Embedding-0.6B \
  --variants routed_kv specialist_text \
  --device cuda
```

## Audit and limits / 统计与边界

`metrics.json` and optional audit JSON contain embedding token/window counts,
encoder/ranking seconds, anchor indices, attention/cosine/combined scores,
nearest seed for every block and selected block indices (zero-based).
Evaluation outputs encoder tokens/time separately, never treating them as Qwen
tokens or zero-cost preprocessing. Model loading is separate; actual encoder
execution is included in pipeline time.

同样的种子预算不代表相同的最终 KV 数量：整块填充与邻块扩张都会改变实际
保留量。报告记录 attention-only 和 hybrid 最终位置数；比较时必须同时查看
质量、实际保留率与耗时，不能把多保留上下文带来的收益归因于向量本身。
若需要严格等预算消融，后续应另做最终 KV 总预算约束实验。

This does not check negation, commitment, speaker responsibility or later
supersession. Those still require the receiver to interpret the retained evidence.
