# Receiver-Conditioned KV Meeting Workflow

这是一个 **training-free 实验工作流**。
目标是验证：会议摘要是否可以直接利用按接收需求选出的原文 KV，减少
“专家先写摘要，再压缩专家摘要”的信息瓶颈。

**不是 CacheBack、LatentMAS 或 KVCOMM 的完整复现。**

## 工作流

```text
UTF-8 transcript
  -> 检查完整 prompt + 查询/输出预算是否超出模型上下文
  -> 全文一次 SDPA prefill，得到只读 source KV
  -> 四个角色查询：decisions / actions / open_issues / context
  -> 每个查询独立读取 source KV，计算接收需求相关的注意力分数
  -> 按连续 token span 选取 KV 位置，每个角色单独预算
  -> 所有角色选中位置取并集、去重、恢复时间顺序
  -> 从同一个 source KV 取出这些位置，直接传给最终摘要器
  -> 摘要器生成最终会议摘要
```

四个角色是 **attention-based evidence selectors**，不是会生成推理/结论的
specialist。它们没有生成自然语言 handoff，也没有生成 latent thoughts。
这是一项明确的拓扑改变，而不是把原有四专家流程换个名字。

这里仍然保留一次共享 prefill，因为它本身没有必要浪费掉。新增的是
接收需求驱动的 **选择性 KV 传递**。注意力查询 token 仍是自然语言控制指令，
但传入最终摘要器的证据通道是 KV 张量。

## 运行

在项目根目录，使用已有的 PyTorch / Transformers 环境：

```bash
python -m receiver_kv_workflow.run \
  --transcript input/meeting1.txt \
  --model Qwen/Qwen3-4B \
  --role-budget-ratio 0.15 \
  --max-new-tokens 1024 \
  --variants routed_kv full_kv selected_text specialist_text
```

PowerShell 可以使用以下单行命令，不要复制 Bash 的反斜杠续行：

```powershell
python -m receiver_kv_workflow.run --transcript input/meeting1.txt --model Qwen/Qwen3-4B --role-budget-ratio 0.15 --max-new-tokens 1024 --variants routed_kv full_kv selected_text specialist_text
```

有人工参考摘要时追加 `--reference input/meeting1_reference.txt`。当前复用项目的
英文词级 ROUGE-L，仅表示参考摘要词序重合，不是事实准确率；没有 reference
则显示 N/A。不要拿当前英文分词评分器评中文摘要。

输出位置：

```text
output/receiver_kv/时间戳/routed_kv.txt
output/receiver_kv/时间戳/full_kv.txt
output/receiver_kv/时间戳/selected_text.txt
output/receiver_kv/时间戳/specialist_text.txt
test_result/receiver_kv/时间戳/metrics.txt
test_result/receiver_kv/时间戳/metrics.json
```

内容与指标分开，不把 transcript、review chunks 或摘要放进 metrics。
JSON 保存选中位置索引，便于审计。终端逐角色、逐 receiver 报进度。

## 对照与参数

| 模式 | 最终接收方收到什么 | 用途 |
|---|---|---|
| routed_kv | 四角色选择并集的原文 KV | 新实验方法 |
| full_kv | 完整原文 KV | 无选择损失的直接摘要对照 |
| selected_text | 同一组选中原文 token，重新 prefill | 区分 KV 状态保留与文本重新编码 |
| specialist_text | 四个 specialist 从完整会议生成自然语言 handoff，最终摘要器读取 handoff | 传统多 agent 自然语言通信基线 |

`selected_text` **不是**自然语言专家 handoff 基线；`specialist_text` 才是该基线。

评测报告还会列出每个变体的端到端 logical model-token events：一次源 prefill、选择 query、specialist prompt/handoff、最终 receiver prompt/output。它不等于 attention FLOPs、真实服务商账单或 KV 字节数，但可用于比较同一模型下的 token 处理量。

- `--role-budget-ratio 0.15`：每角色最多约原前缀的 15%。四角色并集最多约 60%，
  实际有重叠时更少；保留 prompt framing 可能使小输入的实际比例更大。
  **不是总共 15%，不是 96 hidden states。**
- `--span-tokens 32`：按相邻位置一起选，降低只保留实体却丢否定词的风险。
  仍可能切断句子，不等于真正语义分句或事实完整性保证。
- `--query-batch-tokens 8`：只控制短查询注意力计算的内存，不切会议摘要、
  不丢原文、不建立多块 summary。
- `--max-new-tokens 1024`：显式最终输出预算。触及上限报告 `output_cap_hit=true`，
  不把被截断的摘要默认为正常完成。可增大，但需给输出留上下文空间。
- `--variants full_kv`：只跑完整 KV 对照，跳过选择器。

建议先测 `0.15 / 0.25 / 1.0`，再按质量调整，别一开始追求 90% 删除。
`1.0` 使 routed_kv 与 full_kv 接收相同缓存，用于一致性检查。

## 实现边界

1. 首版只支持 **同权重、标准 RoPE、dense full-attention Qwen3**。
   不支持异构模型、滑动窗口、动态 RoPE 扩展、多卡自动分片。
2. KV 不重算、不重定位：保留原始 RoPE 位置；新请求从原前缀的绝对长度继续。
   不能因为物理 KV 数减少，就宣称支持超模型上下文。
3. 只能合并同一 prefill 来源的选中位置。没有拼接独立专家 KV，没有 CacheFuse。
4. 第一次 prefill 仍读取全文。超上下文直接 error，不截断，也不偷偷做层级摘要。
5. cache selection 不是无损：丢弃位置上的证据可能无法被准确恢复。
6. 当前为便于对照仍驻留完整 source KV。因此 receiver KV 变小不等于进程峰值显存降低。
7. 四次 probe 有开销。报告包含 prefill、selection、receiver 时间，不承诺更快。
8. 没有规则 verifier / quote 字面匹配层，也没有事实性保证。

## 成本口径

- input positions：最终接收的缓存位置数，不是端到端执行 token 数。
- cache bytes：实际所有层 K/V 张量字节数。
- logical payload bytes：假设跨进程传输时的负载；本地没有真的发送网络请求。
- text token IDs 按 int32 计算；KV 可能比文本字节大很多，即使位置更少。
- 总时长包含首次全文 prefill、查询选择、接收端生成，不含模型加载。
- 多模式共享前置计算，报告总时长为各模式可归属阶段之和，不是独立进程速度基准。

## 测试

```bash
python -m unittest receiver_kv_workflow.test_workflow -v
```

包括小型随机 Qwen3 的缓存一致性、稀疏 KV 与完整缓存 mask 对照、查询分批一致性、
源缓存不可变、上下文越界、输出上限，以及本地三模式 CLI smoke test。
这些测试不验证真实 Qwen3-4B 的会议摘要质量。
