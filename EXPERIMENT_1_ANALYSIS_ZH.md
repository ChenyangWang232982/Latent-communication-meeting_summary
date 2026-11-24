# 实验1数据分析与论文研究问题设计

整理日期：2026年11月21日。分析对象：`D:\Download\experiment_1_auto1024_cap100_seed12`。

BERTScore 更新日期：2026年11月23日。新增评分来自用户提供的服务器终端输出；其原始逐样本评分文件尚未导入本地。

本报告只分析已保存结果及补算汇总，不重新生成摘要，不修改原始数据或工作流。所有质量、效率与分组统计均基于两种方法同时成功的样本；失败记录只用于交代样本流向，不进入配对均值。BERTScore 已补入，未完成的人工评分不作为零分。

## 1. 主要发现

共计划 295 条记录，完成 276 对，完成率 93.56%。按成功样本加权汇总，KV 路径平均逻辑 decoder token 从 43,129.6 降至 20,227.3，减少 **53.10%**；平均流水线耗时从 180.84 秒降至 162.30 秒，减少 **10.25%**。ROUGE-L 均值为 **0.10256 / 0.10288（KV / text）**，观察到的差距很小。

补算后，四个数据集的 BERTScore F1 差值（KV−text）为 −0.0011、+0.0007、−0.0002 和 −0.0023。根据贴出的四位小数按样本数加权，整体 F1 约为 **0.8296 / 0.8307（KV / text）**，差值约 **−0.0011**。这是参考语义相似度接近的描述性证据，不是事实准确率相同或非劣效的证明。

| 证据 | 本次结果支持的描述 | 不能据此声称 |
| --- | --- | --- |
| 四个数据集 | KV 在各数据集均减少约 50%–55% 的逻辑 token | 网络流量、FLOPs 或账单也减少同样比例 |
| ROUGE-L 配对区间 | 四个数据集的区间均包含 0，未观察到一致的优势方向 | 已证明两种方法等效、事实准确率相同 |
| 语义增强 | 两个版本均开启，属于共同设置 | 本实验已证明 embedding 能解决注意力问题 |
| 长度分组 | 较长组的 token 节省比例通常更高 | 长度是唯一原因；可无上限处理长会议 |
| BERTScore / 人工评分 | BERTScore 汇总已补入，F1 差距较小；人工核对尚未实施 | 已证明语义等效或事实正确性相同 |

## 2. 数据来源、协议和核对结果

| 数据集 | 运行目录（相对于数据根目录） | 计划 | 成功 | 排除 | 完成率 |
| --- | --- | --- | --- | --- | --- |
| AMI | `ami/20261119_083659_646986` | 100 | 100 | 0 | 100.00% |
| ICSI | `icsi/20261119_174529_255779` | 59 | 49 | 10 | 83.05% |
| QMSum Committee | `qmsum_committee/20261120_020952_214447` | 36 | 31 | 5 | 86.11% |
| MeetingBank | `meetingbank/20261120_073025_968960` | 100 | 96 | 4 | 96.00% |
| 合计 | — | 295 | 276 | 19 | 93.56% |

每个数据集上限 100 条；ICSI 和 Committee 使用本次预处理提供的全部 59 和 36 条。这里的样本单位是输入记录，不应把所有记录自动称为互相独立的完整会议；MeetingBank 记录可能来自同一会议的不同议程。AMI、ICSI、Committee 也不能因标签不同就称为四套完全独立的来源：前三者是当前 QMSum 预处理下的不同领域。

已读取各运行的 `manifest.json`、`runtime.json`、`summary.json`、`tokens.json`、`per_sample.csv`、`repetition.json` 以及全部 295 个 `samples/*.json`。核对结果：样本 ID 与 manifest 顺序一致；276 对的本地窗口位置、窗口数和最终汇总选中位置一致；每行输入/输出 token 加和正确；逐样本重算的 ROUGE 均值和 token 总量与汇总文件一致。四次运行的配置、代码指纹、Git commit 和 runtime 一致。

初次整理时，数据目录中未包含 `evaluation/` 下的 BERTScore 补算结果；本次已根据用户提供的服务器输出补入四个数据集的 P/R/F1 汇总。`human_review.csv` 的评分栏仍未填写；可控挑战集人工核对目前仅为已确定的后续方案，尚未实施。报告不混入此前 3 样本、旧 routed_kv 或全局 KV + 原文块实验。

## 3. 工作流与可复现设置

共同流程：全文 prefill → 五类 selector 注意力评分 → 语义增强 → 完整发言人块选择与前后邻块合并 → 按时间顺序构造 selected-context 窗口 → 每窗 specialist 生成局部报告 → 最终 summary agent 接收全局选中证据及按序报告并汇总。连续同一发言人的内容合并为一个块。specialists 共享模型权重，当前按顺序执行，不能据此声称并行加速。

| 设置 | 本次值 |
| --- | --- |
| Generation model | Qwen/Qwen3-8B |
| Embedding model | models/Qwen3-Embedding-0.6B |
| vector_routing / vector_weight | true / 0.3 |
| role_budget_ratio / speaker_neighbor_blocks | 0.25 / 1 |
| 窗口目标数 | max(1, floor(n / 1024 + 0.5)) |
| n 的定义 | 原始全文 tokenizer token 数；含说话人/时间标签，不含 chat 模板 |
| specialist_overlap | 0.25 |
| specialist seed / temperature / top_p / repetition_penalty | 12 + window index / 0.3 / 0.9 / 1.1 |
| summary seed / temperature / top_p / repetition_penalty | 42 / 0.3 / 0.9 / 1.0 |
| 思考模式 | 局部 specialist 关闭；最终汇总开启 |
| 解码 | 采样；不是贪心解码；top_k=0 |
| 独立 review | Disabled |
| fp32_lm_head | false |
| GPU | NVIDIA GeForce RTX 4080 SUPER; 31.48 GiB reported |
| Python / PyTorch / Transformers | 3.12.3 / 2.8.0+cu128 / 5.16.1 |
| CUDA / tokenizers / safetensors | 12.8 / 0.23.2 / 0.8.0 |
| Git commit | `3d4d9e7bb57230a578e972c0a1c032d160a834b0` |
| Model revision | `b968826d9c46dd6066d109eabc6255188de91218` |

窗口按完整选中发言块平衡划分，1024 是由全文长度确定 specialist 数量的尺度，不保证每窗恰好 1024 tokens；实际数量不超过可用选中块数。25% overlap 向前一基础窗口扩展完整块，不额外增加 specialist。role_budget_ratio 是选择阶段的预算参数，不是最终证据保留比例。

| 路径 | 局部 specialist 输入 | 最终汇总输入 |
| --- | --- | --- |
| chunk_kv | 当前选中窗口的 source KV + 阶段指令 | 全部选中 source KV + 按序局部自然语言报告 |
| chunk_text | 相同窗口对应的原文证据 + 同类阶段指令 | 全部选中原文证据 + 按序局部自然语言报告 |

局部阶段没有额外整篇 selected KV，也没有额外当前原文块。两个版本均产出局部自然语言报告，因此该系统属于混合通信工作流。text 对照采用原文证据重新 prefill，不是旧版单个生成式自然语言 handoff。全文上下文化 KV 包含原始历史与位置编码，而文本重新编码的上下文历史不同；因此这是受控的工作流比较，不是严格只更换张量/字符串表示的纯因果干预。

## 4. 指标含义、汇总方式与边界

| 指标 | 计算/含义 | 使用边界 |
| --- | --- | --- |
| ROUGE-1 / ROUGE-2 F1 | 与参考摘要的一元/二元词组重合 | 词汇相似，不是事实核查 |
| ROUGE-L P/R/F1 | 全文 token 序列的最长公共子序列；非 ROUGE-Lsum | ASCII 字母数字转小写，无 stemming；不同实现不可直接混比 |
| BERTScore P/R/F1 | 同一 encoder 下，生成摘要与参考摘要的上下文向量匹配；本次汇总已补算 | 语义相似度仍不等于事实准确率；不能混比不同 encoder 或 rescaling 配置 |
| source-support proxy | 每个候选句至少 45% 的独立词汇出现在全文词表中，则判为词汇支持 | 无法可靠区分否定、责任人、日期或决策变化；不是 MESA 或人工事实分 |
| 逻辑 decoder token | 全文 prefill + probes + 局部 prompts/outputs + 汇总 prompt/output | 缓存重用不重复记作新文本输入；不是 attention FLOPs 或账单 |
| 流水线秒数 | 包含 prefill、选择/embedding、specialists、最终生成；不含模型加载 | 同一 GPU 上按固定 KV→text 顺序运行，存在顺序效应可能 |
| CUDA 峰值 | 各 variant 阶段 PyTorch allocated 峰值，包含模型、embedding 与常驻 source KV | 不含 reserved、非 PyTorch 分配与共享 prefill/选择峰值 |
| 完全重复 bullet | 最终摘要/局部报告中出现标准化后重复条目的样本比例 | 不是语义重复率；sender 计数针对合并的局部报告 |
| 实际保留比例 | 唯一选中 source positions / 全部 source positions | source positions 含受保护模板；不是原文字数压缩率 |
| 逻辑 payload | 累计各阶段 KV tensor bytes 或证据文本 UTF-8 bytes，加报告 bytes | 两种表示不是等价字节度量；未测真实跨设备网络传输 |

质量指标先按记录计算，再在成功配对内取算术平均。合并结果按 276 对样本加权；另给出四个数据集等权均值，避免把大数据集的影响隐藏起来。节省率定义为 100 × (1 − KV 总量 / text 总量)，不是逐样本节省百分比的算术平均。原始输出 token 包括最终思考、分隔符和 EOS；展示摘要和质量评分不包含思考正文。embedding encoder token 独立列示，不与 Qwen decoder token 混为同一种计费或计算量。

## 5. 摘要质量结果

| 数据集 | n | Variant | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | L precision | L recall | Support proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AMI | 100 | chunk_kv | 0.1905 | 0.0447 | 0.1058 | 0.0633 | 0.3420 | 0.9171 |
| AMI | 100 | chunk_text | 0.1849 | 0.0465 | 0.1037 | 0.0615 | 0.3514 | 0.9292 |
| ICSI | 49 | chunk_kv | 0.1414 | 0.0243 | 0.0790 | 0.0456 | 0.3116 | 0.9690 |
| ICSI | 49 | chunk_text | 0.1392 | 0.0230 | 0.0785 | 0.0452 | 0.3177 | 0.9684 |
| QMSum Committee | 31 | chunk_kv | 0.1824 | 0.0500 | 0.1084 | 0.0648 | 0.3524 | 0.9957 |
| QMSum Committee | 31 | chunk_text | 0.1792 | 0.0500 | 0.1069 | 0.0638 | 0.3597 | 0.9960 |
| MeetingBank | 96 | chunk_kv | 0.1519 | 0.0683 | 0.1093 | 0.0657 | 0.4810 | 0.9311 |
| MeetingBank | 96 | chunk_text | 0.1595 | 0.0725 | 0.1131 | 0.0691 | 0.4761 | 0.9327 |

| 聚合 | Variant | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Support proxy |
| --- | --- | --- | --- | --- | --- |
| 276 对样本加权 | chunk_kv | 0.1675 | 0.0499 | 0.1026 | 0.9400 |
| 276 对样本加权 | chunk_text | 0.1673 | 0.0518 | 0.1029 | 0.9449 |
| 四数据集等权 | chunk_kv | 0.1666 | 0.0468 | 0.1006 | 0.9532 |
| 四数据集等权 | chunk_text | 0.1657 | 0.0480 | 0.1006 | 0.9566 |

KV 的 ROUGE-L 在 AMI、ICSI、Committee 略高，在 MeetingBank 略低；总体并非某一通信方式全面胜出。各数据集不应只按单一绝对 ROUGE-L 排名，因为参考摘要长度、领域及原始记录粒度不同。

| 数据集 | 参考长度均值 | KV 摘要长度 | text 摘要长度 | KV/参考比例均值 | text/参考比例均值 |
| --- | --- | --- | --- | --- | --- |
| AMI | 112.4 | 602.2 | 634.0 | 5.70 | 5.99 |
| ICSI | 95.9 | 648.1 | 670.0 | 7.17 | 7.47 |
| QMSum Committee | 140.0 | 744.3 | 778.0 | 5.71 | 6.02 |
| MeetingBank | 56.6 | 430.5 | 429.8 | 10.46 | 10.49 |

长度单位沿用 ROUGE 的 ASCII token 统计，不是 Qwen tokenizer tokens。“长度比例均值”是逐样本比例的均值，不等于两列长度均值相除。摘要明显长于参考摘要，会影响 precision 与 F1；这解释了部分低分，但不能把全部差距归因于长度，也不能据此忽略不相关内容或遗漏。

### 5.1 BERTScore 补算结果

评分配置：`allenai/longformer-base-4096`，`num_layers=12`，`idf=False`，`rescale_with_baseline=False`，`batch_size=1`，`use_fast_tokenizer=False`，CUDA。比较对象为最终摘要与参考摘要，不是会议全文。项目对超长评分输入报错，不静默截断；评分开销不计入原工作流 token、耗时和显存指标。

| 数据集 | 成功样本数 | Variant | BERTScore P | BERTScore R | BERTScore F1 |
| --- | --- | --- | --- | --- | --- |
| AMI | 100 | chunk_kv | 0.7747 | 0.8887 | 0.8277 |
| AMI | 100 | chunk_text | 0.7760 | 0.8895 | 0.8288 |
| ICSI | 49 | chunk_kv | 0.7743 | 0.8873 | 0.8269 |
| ICSI | 49 | chunk_text | 0.7729 | 0.8875 | 0.8262 |
| QMSum Committee | 31 | chunk_kv | 0.7924 | 0.8939 | 0.8400 |
| QMSum Committee | 31 | chunk_text | 0.7938 | 0.8927 | 0.8402 |
| MeetingBank | 96 | chunk_kv | 0.7772 | 0.8906 | 0.8297 |
| MeetingBank | 96 | chunk_text | 0.7806 | 0.8915 | 0.8320 |
| 合并估算 | 276 | chunk_kv | ≈0.7775 | ≈0.8897 | ≈0.8296 |
| 合并估算 | 276 | chunk_text | ≈0.7790 | ≈0.8902 | ≈0.8307 |

| 数据集 | ΔP（KV−text） | ΔR（KV−text） | ΔF1（KV−text） |
| --- | --- | --- | --- |
| AMI | −0.0013 | −0.0008 | −0.0011 |
| ICSI | +0.0014 | −0.0002 | +0.0007 |
| QMSum Committee | −0.0014 | +0.0012 | −0.0002 |
| MeetingBank | −0.0034 | −0.0009 | −0.0023 |
| 合并估算 | ≈−0.001563 | ≈−0.000504 | ≈−0.001097 |

合并估算公式：`Σ(n_dataset × dataset_mean) / 276`。表中数据集均值来自终端显示的四位小数，因此合并值与差值均为近似值；不能把这些数值当成原始精度重新导出的结果。合并 F1 是各样本 F1 的加权均值，不能由表中的平均 P/R 再算一次调和平均来代替。按贴出数值核算的合并 F1 在未进一步取整时约为 0.829635 和 0.830732。

解读：KV 在 ICSI 的 F1 略高，在其余三个数据集略低；差值绝对值均不超过 0.0023（基于显示精度）。这与整体 ROUGE-L 接近的现象相符，支持“较少逻辑 token 事件与较低平均延迟，伴随较小的参考语义相似度差异”。P 均低于 R 与生成摘要较长的现象相容，但不能仅凭 P/R 推断原因，也不能将 BERTScore R 当作重要事实覆盖率。

本地尚未取得该次补算的逐样本 BERTScore 或配对区间，所以不新增统计显著性、等效性或非劣效结论，也不从八个数据集均值构造置信区间。后续可读取服务器 `evaluation/<timestamp>/summary.json` 的 `paired_quality_differences.bertscore_f1`；无须重新生成摘要。

| 数据集 | 服务器评分目录（相对于数据根目录） |
| --- | --- |
| AMI | `ami/20261119_083659_646986/evaluation/20261122_222254_692521` |
| ICSI | `icsi/20261119_174529_255779/evaluation/20261122_222329_027442` |
| QMSum Committee | `qmsum_committee/20261120_020952_214447/evaluation/20261122_222355_822880` |
| MeetingBank | `meetingbank/20261120_073025_968960/evaluation/20261122_222426_256504` |

## 6. 配对差异与不确定性

| 数据集 | n | 平均 ΔROUGE-L（KV−text） | 95% bootstrap 区间 | KV 高 / 相同 / 低 |
| --- | --- | --- | --- | --- |
| AMI | 100 | 0.002113 | [-0.000073, 0.004227] | 53 / 0 / 47 |
| ICSI | 49 | 0.000473 | [-0.001920, 0.003107] | 25 / 0 / 24 |
| QMSum Committee | 31 | 0.001454 | [-0.004067, 0.006464] | 20 / 0 / 11 |
| MeetingBank | 96 | -0.003842 | [-0.009045, 0.001363] | 44 / 0 / 52 |
| 合并：按数据集分层 | 276 | -0.000323 | [-0.002547, 0.001767] | 142 / 0 / 134 |

各数据集区间直接读取原始报告：按记录配对 bootstrap，2,000 次，seed=7。合并区间为本次新增分析：按数据集分层、保留各数据集成功样本数，配对重抽样 10,000 次，NumPy default_rng(7)，百分位区间。重抽样单位是记录而不是会议聚类；若多个记录来自同一会议，区间可能偏窄。

| 合并统计量 | 估计 | 新增 95% 分层 bootstrap 区间 |
| --- | --- | --- |
| ΔROUGE-L（KV−text） | -0.000323 | [-0.002547, 0.001767] |
| 逻辑 token 节省 | 53.10% | [52.84%, 53.36%] |
| 流水线耗时减少 | 10.25% | [9.29%, 11.22%] |

所有 ROUGE-L 区间均覆盖 0：可描述为“在本次样本与指标下，观察到的参考重合度相近”，不能写成“证明完全等效”或“事实准确率无损”。等效/非劣效结论需要预先指定容许差值与相应检验。token 和耗时区间只反映记录组成的不确定性，不包含跨随机种子、GPU、模型或 prompt 的变化。

## 7. 逻辑 token 消耗与来源

| 数据集 | Variant | 全文 prefill | Probes | 局部 prompt | 局部 output | 汇总 prompt | 汇总 output | 总 input | 总 output | 合计 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AMI | chunk_kv | 9,344.0 | 124.0 | 1,514.6 | 2,986.9 | 3,870.9 | 1,565.3 | 14,853.5 | 4,552.2 | 19,405.8 |
| AMI | chunk_text | 9,344.0 | 124.0 | 13,250.9 | 3,186.7 | 13,084.2 | 1,727.2 | 35,803.1 | 4,913.9 | 40,717.0 |
| ICSI | chunk_kv | 17,784.5 | 124.0 | 2,894.7 | 5,510.8 | 6,745.9 | 1,575.9 | 27,549.0 | 7,086.7 | 34,635.7 |
| ICSI | chunk_text | 17,784.5 | 124.0 | 26,284.2 | 6,024.4 | 24,760.2 | 1,695.8 | 68,952.9 | 7,720.2 | 76,673.2 |
| QMSum Committee | chunk_kv | 16,642.0 | 124.0 | 2,700.7 | 5,573.5 | 6,744.4 | 1,711.2 | 26,211.1 | 7,284.7 | 33,495.8 |
| QMSum Committee | chunk_text | 16,642.0 | 124.0 | 24,448.7 | 6,623.8 | 23,292.2 | 1,781.7 | 64,507.0 | 8,405.5 | 72,912.5 |
| MeetingBank | chunk_kv | 3,910.6 | 124.0 | 657.6 | 1,411.0 | 2,041.7 | 1,299.3 | 6,734.0 | 2,710.2 | 9,444.2 |
| MeetingBank | chunk_text | 3,910.6 | 124.0 | 6,094.2 | 1,591.5 | 5,876.2 | 1,307.6 | 16,004.9 | 2,899.1 | 18,904.1 |

| 聚合 | Variant | 总 input 均值 | 总 output 均值 | 总 token 均值 | 总 token 累计 |
| --- | --- | --- | --- | --- | --- |
| 276 对成功样本 | chunk_kv | 15,558.9 | 4,668.4 | 20,227.3 | 5,582,738 |
| 276 对成功样本 | chunk_text | 38,026.1 | 5,103.5 | 43,129.6 | 11,903,767 |

| 数据集 | KV 合计/样本 | text 合计/样本 | 节省/样本 | 节省率 |
| --- | --- | --- | --- | --- |
| AMI | 19,405.8 | 40,717.0 | 21,311.3 | 52.34% |
| ICSI | 34,635.7 | 76,673.2 | 42,037.5 | 54.83% |
| QMSum Committee | 33,495.8 | 72,912.5 | 39,416.7 | 54.06% |
| MeetingBank | 9,444.2 | 18,904.1 | 9,459.9 | 50.04% |
| 合并 | 20,227.3 | 43,129.6 | 22,902.3 | 53.10% |

主要差异来自局部及汇总阶段不必把 source evidence 重复作为文本 token 重新输入。局部报告与最终答案仍然生成自然语言，且 KV attention、缓存复制和原始全文 prefill 仍有成本。因此 53.10% 的逻辑 token 节省不应解释为 53.10% 的实际计算或端到端加速。prefill/probes 在每条独立工作流中各记一次；实验配对实现复用共享工作，表格两路径相加不代表实测 GPU 总工作。

| 数据集 | 独立 embedding encoder tokens/样本 | embedding/ranking 秒/样本 |
| --- | --- | --- |
| AMI | 9,826.0 | 2.246 |
| ICSI | 18,444.5 | 2.982 |
| QMSum Committee | 16,768.0 | 1.019 |
| MeetingBank | 3,899.9 | 0.362 |

embedding 时间已包含在流水线时间中，不应再重复加一次；两条路径使用共同的语义路由结果。

## 8. 延迟与显存

| 数据集 | Variant | 均值秒 | 中位数秒 | P95 sec | 局部 specialist 秒 | 最终汇总秒 | CUDA 峰值 MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AMI | chunk_kv | 159.19 | 160.80 | 225.65 | 96.57 | 56.84 | 20,467.7 |
| AMI | chunk_text | 174.96 | 174.08 | 253.86 | 104.49 | 64.39 | 21,260.4 |
| ICSI | chunk_kv | 249.27 | 250.65 | 324.93 | 168.43 | 70.98 | 24,062.7 |
| ICSI | chunk_text | 277.92 | 276.87 | 350.60 | 187.71 | 79.57 | 25,280.0 |
| QMSum Committee | chunk_kv | 248.84 | 240.52 | 344.57 | 167.08 | 74.43 | 23,505.3 |
| QMSum Committee | chunk_text | 291.74 | 273.40 | 439.37 | 202.22 | 81.39 | 24,763.0 |
| MeetingBank | chunk_kv | 93.22 | 72.29 | 238.37 | 46.02 | 45.00 | 18,446.9 |
| MeetingBank | chunk_text | 101.60 | 71.99 | 293.93 | 53.48 | 45.78 | 18,760.9 |

| 数据集 | 耗时减少 | text/KV 加速倍数 | 显存峰值差 MiB | 峰值减少 |
| --- | --- | --- | --- | --- |
| AMI | 9.02% | 1.099× | 792.7 | 3.73% |
| ICSI | 10.31% | 1.115× | 1,217.3 | 4.82% |
| QMSum Committee | 14.70% | 1.172× | 1,257.6 | 5.08% |
| MeetingBank | 8.25% | 1.090× | 314.0 | 1.67% |

固定顺序、同机测量下，四个数据集的 KV 平均延迟均较低。显存收益远小于 token 收益，因为完整 source KV 和模型常驻。MiB=2^20 bytes，GiB=2^30 bytes；这些值不是 nvidia-smi 的整进程峰值。模型加载不计时；更长的局部报告、最终思考及 attention 长度均会影响延迟，因此 token 量和秒数不成固定比例。

## 9. 输入长度分布与分组表现

| 数据集 | 全部记录最短/中位/最长 tokens | 成功记录最短/中位/最长 tokens | 平均目标/实际 specialist 数 |
| --- | --- | --- | --- |
| AMI | 1,537 / 9,488 / 20,479 | 1,537 / 9,488 / 20,479 | 9.08 / 9.08 |
| ICSI | 7,822 / 18,539 / 35,694 | 7,822 / 18,012 / 26,597 | 17.29 / 17.29 |
| QMSum Committee | 10,706 / 16,700 / 31,084 | 10,706 / 16,199 / 26,175 | 16.13 / 16.13 |
| MeetingBank | 244 / 2,046 / 82,061 | 244 / 1,824 / 27,315 | 3.95 / 3.95 |

以下长度桶为左闭右开：[0,4000)、[4000,8000)、[8000,12000)、[12000,16000)、[16000,+∞)。计数按原始全文 tokenizer tokens，不按 source prefill 或生成摘要长度。

| 数据集 | 长度组 | 计划 | 成功 | 排除 | KV R-L | Text R-L | KV sec | Text sec | KV tokens | Text tokens | token 节省 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AMI | 0-4000 | 9 | 9 | 0 | 0.1132 | 0.1093 | 82.4 | 87.0 | 8,092.9 | 14,962.3 | 45.91% |
| AMI | 4000-8000 | 31 | 31 | 0 | 0.1052 | 0.1042 | 122.0 | 135.5 | 14,016.6 | 28,609.7 | 51.01% |
| AMI | 8000-12000 | 38 | 38 | 0 | 0.1061 | 0.1044 | 171.5 | 186.8 | 21,192.7 | 44,857.0 | 52.76% |
| AMI | 12000-16000 | 20 | 20 | 0 | 0.1033 | 0.0990 | 215.5 | 239.2 | 27,556.2 | 58,959.5 | 53.26% |
| AMI | 16000+ | 2 | 2 | 0 | 0.1036 | 0.1056 | 285.1 | 315.4 | 38,390.0 | 83,192.5 | 53.85% |
| ICSI | 0-4000 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| ICSI | 4000-8000 | 1 | 1 | 0 | 0.0707 | 0.0676 | 131.6 | 140.4 | 16,808.0 | 34,873.0 | 51.80% |
| ICSI | 8000-12000 | 5 | 5 | 0 | 0.0748 | 0.0756 | 166.4 | 185.9 | 21,969.2 | 47,998.2 | 54.23% |
| ICSI | 12000-16000 | 9 | 9 | 0 | 0.0760 | 0.0784 | 213.4 | 242.4 | 28,277.3 | 61,850.6 | 54.28% |
| ICSI | 16000+ | 44 | 34 | 10 | 0.0806 | 0.0793 | 274.4 | 304.9 | 38,705.9 | 86,043.1 | 55.02% |
| QMSum Committee | 0-4000 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| QMSum Committee | 4000-8000 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| QMSum Committee | 8000-12000 | 6 | 6 | 0 | 0.1239 | 0.1287 | 187.9 | 208.3 | 23,992.3 | 49,751.5 | 51.78% |
| QMSum Committee | 12000-16000 | 9 | 9 | 0 | 0.1128 | 0.1082 | 213.2 | 246.8 | 28,122.4 | 61,133.9 | 54.00% |
| QMSum Committee | 16000+ | 21 | 16 | 5 | 0.1001 | 0.0980 | 291.8 | 348.3 | 40,082.1 | 88,223.4 | 54.57% |
| MeetingBank | 0-4000 | 69 | 67 | 2 | 0.1283 | 0.1338 | 59.8 | 59.9 | 4,741.4 | 7,867.7 | 39.73% |
| MeetingBank | 4000-8000 | 17 | 17 | 0 | 0.0644 | 0.0628 | 123.7 | 135.5 | 13,472.1 | 28,092.1 | 52.04% |
| MeetingBank | 8000-12000 | 3 | 3 | 0 | 0.0432 | 0.0393 | 172.3 | 219.3 | 20,231.3 | 44,904.7 | 54.95% |
| MeetingBank | 12000-16000 | 5 | 5 | 0 | 0.0889 | 0.0953 | 224.9 | 261.6 | 27,491.2 | 60,760.8 | 54.76% |
| MeetingBank | 16000+ | 6 | 4 | 2 | 0.0571 | 0.0579 | 300.3 | 367.3 | 40,447.8 | 92,893.5 | 56.46% |

| 合并长度组 | 计划 | 成功 | 排除 | KV R-L | Text R-L | token 节省 | 耗时减少 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-4000 | 78 | 76 | 2 | 0.1265 | 0.1309 | 40.99% | 1.09% |
| 4000-8000 | 49 | 49 | 0 | 0.0904 | 0.0891 | 51.38% | 9.49% |
| 8000-12000 | 52 | 52 | 0 | 0.1015 | 0.1007 | 52.91% | 9.49% |
| 12000-16000 | 43 | 43 | 0 | 0.0979 | 0.0962 | 53.81% | 11.63% |
| 16000+ | 73 | 56 | 17 | 0.0853 | 0.0841 | 54.96% | 12.58% |

在成功样本中，合并 token 节省从 <4k 组的约 41% 升至 ≥16k 组的约 55%；耗时收益从约 1% 升至约 13%。这与重复编码较长证据的成本更高相符，但不能只归因于长度：<4k 组以 MeetingBank 为主，而 ≥16k 组以 ICSI/Committee 为主。优先结合每个数据集内部的长度表解读；小样本组（例如 AMI ≥16k 只有 2 对）不适合强结论。没有哪个长度组的 ROUGE 结果能单独证明“注意力问题已解决”。

## 10. 证据保留、重复和逻辑 payload

| 数据集 | 实际保留比例均值 | 最小/最大 | KV sender 重复 | text sender 重复 | KV final 重复 | text final 重复 |
| --- | --- | --- | --- | --- | --- | --- |
| AMI | 94.65% | 72.87% / 99.79% | 7/100 | 4/100 | 9/100 | 11/100 |
| ICSI | 98.11% | 94.52% / 100.00% | 12/49 | 13/49 | 8/49 | 6/49 |
| QMSum Committee | 92.52% | 84.39% / 99.18% | 5/31 | 4/31 | 2/31 | 3/31 |
| MeetingBank | 91.34% | 31.68% / 100.00% | 11/96 | 15/96 | 0/96 | 1/96 |

最终摘要完全重复 bullet：KV 19/276，text 21/276；局部报告合并后的对应计数为 35/276 和 36/276。这是描述性计数，不是语义重复的完整测量，也没有证明两者重复率存在统计差异。成功样本两路径的 handoff/receiver cap-hit 均为 0；不代表排除的失败记录也正常终止。

role budget=0.25 最终却保留约 91%–98% 的 source positions，源于多 selector 合并、完整块和邻块扩张。这是当前协议的实际行为，不能用 25% 标注最终上下文压缩率。本次结果主要支持上下文缓存复用的价值，而不能独立证明激进筛选有效。语义路由消融时需要同时报告最终保留比例，否则很容易把证据量差异误归因于语义分数。

| 数据集 | KV 逻辑 payload GiB/样本 | text 逻辑 payload KiB/样本 |
| --- | --- | --- |
| AMI | 2.811 | 98.2 |
| ICSI | 5.586 | 172.5 |
| QMSum Committee | 5.087 | 199.5 |
| MeetingBank | 1.239 | 46.4 |

payload 累加各阶段同一证据的逻辑交付量，可能重复计算同一 KV 行；它不是唯一常驻内存，也不是实测传输。KV 的张量表示明显比文本大，因此不能从 token 表得出“KV 更省网络带宽”。本实现的主要部署场景是同模型、本地缓存复用；跨机器传输的收益需要另外测量。

## 11. 排除样本的简要记录

按既定分析决定，不重跑、不把失败计入成功配对的质量或效率均值。以下只保留透明的样本流向记录，不把失败样本赋零分，也不依据现有配对 checkpoint 推断两条路径各自的完整成功率。

| 排除原因 | 记录数 |
| --- | --- |
| CUDA 显存不足 | 8 |
| 原文超过模型上下文 | 1 |
| 阶段 prompt 无生成空间 | 6 |
| 生成耗尽上下文仍未 EOS | 4 |
| 合计 | 19 |

ICSI 排除 10 条、Committee 5 条、MeetingBank 4 条、AMI 0 条。长文本失败使成功集合偏向可处理长度，因此“≥16k 组”不是对任意超长会议的无条件泛化结论。MeetingBank 的两条生成耗尽记录仅为 549 与 3,378 tokens，说明并非所有失败都由原文过长造成；本报告不再对其追加修复实验。

## 12. 优化后的三个 Research Questions

建议保留已提交题目：**Design and Evaluation of a Semantic-Assisted Latent Communication Workflow for Long-Form Meeting Summarization**。三个问题分别对应通信机制、语义证据选择、工作流结构；避免把所有改进都归于 latent communication。

### RQ1 — 通信方式

**中文：**在长文本会议概括工作流中，基于 KV cache 的 latent communication 相比传递自然语言证据，如何影响摘要质量、逻辑 token 消耗、执行延迟与显存占用？

**English:** How does KV-cache-based latent communication affect summary quality, logical token consumption, execution latency, and memory usage compared with natural-language evidence transfer in long-form meeting summarization workflows?

本次实验直接提供该问题的主要证据。措辞采用“如何影响”而不是预设“更准确/更高效”；text 需要明确为相同选中证据的自然语言原文传递。由于缓存上下文化历史不同，结论限定在当前工作流比较。

### RQ2 — 语义路由

**中文：**与仅依赖注意力评分的路由相比，语义相似度辅助的证据选择如何影响不同长度会议中的重要信息覆盖与摘要质量，其额外成本是多少？

**English:** How does semantic-similarity-assisted evidence selection, compared with attention-only routing, affect important-information coverage and summary quality across meeting lengths, and what additional cost does it introduce?

不建议表述成“解决注意力问题”：embedding 改变的是路由分数与可见证据，不是直接修复生成模型内部的注意力机制。当前两路径都启用语义增强，因此尚未回答 RQ2。coverage 应由小规模原文对照标注或已定义的重要事实清单衡量，不能直接把 ROUGE recall 当作完整事实覆盖率。

### RQ3 — 工作流结构

**中文：**在相同选中证据和通信方式下，按时间顺序分块的 specialist–synthesis 工作流，相比单 agent 概括，如何影响重要细节与最终决策状态的保留、内容重复及质量–效率权衡？

**English:** Given the same selected evidence and communication mechanism, how does a chronological chunk-specialist–synthesis workflow compare with single-agent summarization in preserving important details and final decision states, reducing redundancy, and balancing quality against efficiency?

这是近期从单步概括演进到分块 specialists 后最自然的第三个问题。它检验“分开分析再汇总”是否有价值，不预设一定优于单 agent。当前没有匹配的单 agent 实验，不能把旧窗口或历史不同 prompt 的结果直接拼成该对照。无需重新引入独立 timeline/decision 模块；只需评价最终摘要是否保留最后支持的决策状态。

| 研究问题 | 建议对照 | 固定条件 | 主要指标 | 状态 |
| --- | --- | --- | --- | --- |
| RQ1 | chunk_kv vs chunk_text | 同一证据、窗口、模型、阶段要求、解码设置 | ROUGE, BERTScore, human quality; tokens, latency, memory | 276 对；BERTScore 汇总已补入，人工诊断待实施 |
| RQ2 | attention-only vs attention + semantic | 固定 chunk_kv、预算、邻块、窗口规则；报告实际保留量 | 重要信息覆盖、ROUGE/BERTScore、embedding 开销、保留比例 | 待做消融 |
| RQ3 | 单 agent vs 分块 specialists + synthesis | 同一 selected KV、模型、最终格式与最终解码；报告新增生成成本 | 事实与最终状态、覆盖、重复、token、时间 | 待做匹配基线 |

RQ2 若希望把效果归因于“选择质量”，除固定 seed budget 外，还应检查最终证据量是否相近；当前近全文保留可能掩盖语义增强的作用。可先报告自然扩张下的消融，再在需要时补一个最终保留量匹配的对照。RQ3 多阶段天然增加调用与输出，不强行宣称计算预算相同，而是把增加的成本纳入结果。无需一开始就运行 2×2×2 的全部组合；先围绕已确定的 KV 工作流分别做单因素对照。

## 13. 论文可使用的结论与尚待完成部分

**可直接改写进 Results/Discussion 的表述：**

> 在四个领域/数据集共 276 对成功完成的会议概括记录中，基于 KV cache 的分块工作流相较于原文文本传递，将平均逻辑 decoder token 事件减少 53.10%，平均流水线耗时减少 10.25%。两种方法的平均 ROUGE-L 分别为 0.10256 和 0.10288，配对区间覆盖零，说明本次实验观察到的参考摘要重合度接近，但尚不能证明语义质量等效或事实准确率无损。收益主要体现为复用上下文化 source cache，且在较长输入组中更加明显；该结论限定于成功样本、当前模型和单机执行协议。

> 补算 BERTScore 后，根据四个数据集显示均值加权得到的整体 F1 约为 0.8296（KV）和 0.8307（text），差值约 −0.0011。该描述性结果进一步支持两路径具有接近的参考语义相似度；尚未依据逐样本 BERTScore 区间建立统计推断，也不将语义相似度解释为事实准确率。

| 可写入论文 | 需要补充或限定 |
| --- | --- |
| 方法、实现与实验协议 | 完整描述两路径、采样、最终 thinking、窗口规则 |
| 实验1的 ROUGE、BERTScore 与效率结果 | BERTScore 汇总已补入；后续导入原始精度评分和配对区间，不需重跑生成 |
| 小规模人工分析方案 | 每数据集随机 5 对，共 20 对/40 摘要；隐藏方法；事实、覆盖、简洁性及重大错误 |
| 单评估者局限 | 可由作者独立完成；如实说明，不能报告评估者间一致性 |
| 语义增强贡献 | 需要 RQ2 对照，不能由实验1推断 |
| 分块协作贡献 | 需要 RQ3 匹配单 agent 对照 |

共同局限：一组生成 seed 设置、单模型/单 GPU、固定 variant 执行顺序、成功样本分析、长度组领域混杂、输出与参考格式/长度不一致、非标准库但明确定义的 ROUGE、缺少当前人工事实评估。三个 RQ 不采用加权综合总分；分别呈现质量、资源成本和局限即可。过去关于 FRAME、topic specialists、global KV 等迭代可作为设计探索叙述，未经同协议对照不能作为正式优越性证据。

## 附录 A. 排除记录索引

| 数据集 | Record ID | 原文 tokens | 类别 |
| --- | --- | --- | --- |
| ICSI | 13 | 30,368 | 阶段 prompt 无生成空间 |
| ICSI | 16 | 35,694 | 阶段 prompt 无生成空间 |
| ICSI | 17 | 26,086 | CUDA 显存不足 |
| ICSI | 18 | 33,769 | 阶段 prompt 无生成空间 |
| ICSI | 27 | 29,229 | 生成耗尽上下文仍未 EOS |
| ICSI | 29 | 27,839 | CUDA 显存不足 |
| ICSI | 30 | 25,522 | CUDA 显存不足 |
| ICSI | 36 | 27,213 | CUDA 显存不足 |
| ICSI | 56 | 25,036 | CUDA 显存不足 |
| ICSI | 58 | 30,010 | 生成耗尽上下文仍未 EOS |
| QMSum Committee | 1 | 31,084 | 阶段 prompt 无生成空间 |
| QMSum Committee | 3 | 30,279 | 阶段 prompt 无生成空间 |
| QMSum Committee | 8 | 26,780 | CUDA 显存不足 |
| QMSum Committee | 9 | 25,991 | CUDA 显存不足 |
| QMSum Committee | 10 | 26,935 | CUDA 显存不足 |
| MeetingBank | LongBeachCC_03192019_19-0260 | 549 | 生成耗尽上下文仍未 EOS |
| MeetingBank | LongBeachCC_04022019_19-0262 | 82,061 | 原文超过模型上下文 |
| MeetingBank | BostonCC_06082022_2022-0726 | 3,378 | 生成耗尽上下文仍未 EOS |
| MeetingBank | LongBeachCC_08102021_21-0765 | 33,092 | 阶段 prompt 无生成空间 |

## 附录 B. 核心来源文件与后续补算

- **AMI**: [summary.json](D:/Download/experiment_1_auto1024_cap100_seed12/ami/20261119_083659_646986/summary.json), [per_sample.csv](D:/Download/experiment_1_auto1024_cap100_seed12/ami/20261119_083659_646986/per_sample.csv), [overview.txt](D:/Download/experiment_1_auto1024_cap100_seed12/ami/20261119_083659_646986/overview.txt), [manifest.json](D:/Download/experiment_1_auto1024_cap100_seed12/ami/20261119_083659_646986/manifest.json).

- **ICSI**: [summary.json](D:/Download/experiment_1_auto1024_cap100_seed12/icsi/20261119_174529_255779/summary.json), [per_sample.csv](D:/Download/experiment_1_auto1024_cap100_seed12/icsi/20261119_174529_255779/per_sample.csv), [overview.txt](D:/Download/experiment_1_auto1024_cap100_seed12/icsi/20261119_174529_255779/overview.txt), [manifest.json](D:/Download/experiment_1_auto1024_cap100_seed12/icsi/20261119_174529_255779/manifest.json).

- **QMSum Committee**: [summary.json](D:/Download/experiment_1_auto1024_cap100_seed12/qmsum_committee/20261120_020952_214447/summary.json), [per_sample.csv](D:/Download/experiment_1_auto1024_cap100_seed12/qmsum_committee/20261120_020952_214447/per_sample.csv), [overview.txt](D:/Download/experiment_1_auto1024_cap100_seed12/qmsum_committee/20261120_020952_214447/overview.txt), [manifest.json](D:/Download/experiment_1_auto1024_cap100_seed12/qmsum_committee/20261120_020952_214447/manifest.json).

- **MeetingBank**: [summary.json](D:/Download/experiment_1_auto1024_cap100_seed12/meetingbank/20261120_073025_968960/summary.json), [per_sample.csv](D:/Download/experiment_1_auto1024_cap100_seed12/meetingbank/20261120_073025_968960/per_sample.csv), [overview.txt](D:/Download/experiment_1_auto1024_cap100_seed12/meetingbank/20261120_073025_968960/overview.txt), [manifest.json](D:/Download/experiment_1_auto1024_cap100_seed12/meetingbank/20261120_073025_968960/manifest.json).

以下保留服务器补算命令以便复现；四个数据集已完成本次补算，无须因更新本报告再次运行。命令直接读取已有摘要，不重新运行工作流：

```bash
for benchmark in ami icsi qmsum_committee meetingbank; do
  RUN_DIR=$(find "test_result/experiment_1_auto1024_cap100_seed12/$benchmark" \
    -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)

  python -m experiment_1.rescore \
    --run-dir "$RUN_DIR" \
    --bertscore-model allenai/longformer-base-4096 \
    --bertscore-layers 12 \
    --device cuda
done
```

新增评分应保留模型、层数和评价配置；若遇到 encoder 长度限制，应明确记录，不能静默截断。后续导入 `evaluation/<timestamp>/` 后，可将本报告基于终端四位小数的 BERTScore 估算替换为原始精度，并加入配对区间；其余原始 token/资源结果无需再生成。原有 ROUGE、资源差异与百分比使用未取整数值计算；新增 BERTScore 表格及合并估算的精度限制见第 5.1 节。
