# 实验 3：20 条首尾关联挑战集结果

## 1. 主要结论

V3 将关键证据共同窗口覆盖率从 **0/20 提高到 19/20（95%）**，达到本项目预先约定的
80% 点估计目标。三组各完成 20 条，没有执行失败或阶段上下文耗尽记录。

这个结果支持的是：**V3 能将已全局选中、但分散在远距离位置的相关发言补充到同一个
局部 specialist 窗口。**它不能直接证明最终摘要更准确，也不能证明模型内部注意力已改善。
逐例核对已发现：部分 V3 摘要在证据同窗后仍出现负责人错误、条件状态错误和别名拆分。

因此，本轮可以作为“证据组织有效”的实验结果记录，**尚不足以单独决定用 V3 替换默认工作流**。
本报告没有生成正式事实正确率，没有调用 AI judge，没有把助手分析当成人工盲评。

## 2. 数据与运行来源

| 项目 | 记录 |
|---|---|
| 本地结果根目录 | `D:/Download/experiment_3_challenge20/20261126_001225_109957` |
| 工作流子目录 | `workflow/20261126_001350_993226` |
| 单模型子目录 | `direct/20261126_031424_662760/direct_text` |
| 原始统计 | 根目录下 `report/coverage.txt`、`report/target_review.csv` |
| 代码提交 | `c819a4e324130f4ba4c24ac77b66f3868206dfbb` |
| 数据来源 | 20 场原 AMI 会议，保留全文，首尾各加入一段人工编写发言 |
| 选样方式 | 在既有 AMI 输入顺序中跳过前 6 条，固定取后续 20 条 |
| 测试性质 | 合成事项诊断；不是原始 AMI 事实或官方挑战测试集 |

目录中的 `20261126` 为原服务器运行标识，原样保留，不将其当成已校准的本地时间。
20 条新增事项未用于此前三个开发案例的调参；但基础 AMI 会议可能已用于其他实验，
不能称为完全未见会议，也不能称为官方 test split。五类场景由同一套人工模板设计，
只测试首尾关联，不覆盖所有会议位置、噪声和多模态情况。

已复核：两组工作流及单模型 manifest 的输入 SHA-256 和有序样本 ID 相同；由已保存的
20 份 `source.json` 重建 `meetings.jsonl` 后，SHA-256 与 manifest 一致。
20 条的 baseline、semantic 最终汇总位置均与各自共同保存的 selected positions 一致。
本轮未修改原始结果和空白人工核对表。

## 3. 配置与比较边界

| 配置 | 值 |
|---|---|
| 生成模型 | Qwen/Qwen3-8B |
| 模型 revision | `b968826d9c46dd6066d109eabc6255188de91218`（工作流 runtime） |
| GPU | NVIDIA GeForce RTX 4080 SUPER，约 32 GiB（工作流 runtime） |
| 运行依赖 | Python 3.12.3；PyTorch 2.8.0+cu128；Transformers 5.16.1 |
| 全局选择 | attention + Qwen3-Embedding-0.6B，vector weight=0.3 |
| 选择 seed budget | role-budget-ratio=0.25；前后各 1 个完整 speaker block |
| 局部窗口 | 自动按全文 token 数分配约 1024-token 份数；overlap=0.25 |
| V3 补充 | relative_diverse_v3；每窗口最多 2 块；排除相邻 1 块 |
| 局部生成 | seed=12+窗口序号；temperature=0.3；top_p=0.9；repetition_penalty=1.1 |
| 最终生成 | seed=42；temperature=0.3；top_p=0.9；thinking=True |

baseline 与 semantic 使用同一全局选择、窗口规划和阶段要求；区别在于 semantic 给局部窗口
补充了远距离已选证据，因此不是等局部证据量对照。两组最终汇总都可以访问全局 selected KV
和按主窗口顺序排列的自然语言报告。baseline 的 0% 局部同窗率**不等于最终模型没看见首尾证据**。

direct_text 一次读取完整文本，不做选择、embedding 或局部报告。与工作流的最终解码参数一致，
但阶段数量、提示词及源证据表示不同，因此是整体工作流比较，不能把质量差异仅归因于 V3。

## 4. 完成率与证据覆盖

共同窗口命中：理解某事项需要的首尾证据 token 位置，完整进入至少一个局部 specialist 窗口。
每事项计一次，不按双向检索或窗口数重复计数。详见 [METRICS.md](experiment_3/METRICS.md)。

| 方法 | 完成数 | 可评估局部覆盖 | 同窗命中 | 覆盖率 | 95% Wilson 区间 |
|---|---:|---:|---:|---:|---|
| baseline | 20/20 | 20 | 0 | 0% | 0%–16.1% |
| semantic V3 | 20/20 | 20 | 19 | **95%** | 76.4%–99.1% |
| direct_text | 20/20 | 不适用 | 不适用 | 不适用 | 不适用 |

Wilson 区间只是这 20 个合成事项的二项比例描述；不是代表性总体推断。
95% 点估计达到 80% 目标，但区间下界约 76.4%，不能声称已用 95% 置信度证明总体覆盖率超过 80%。

| 类别 | 事项数 | baseline 命中 | semantic 命中 | semantic 覆盖率 |
|---|---:|---:|---:|---:|
| revision：明确修订 | 4 | 0/4 | 4/4 | 100% |
| owner_change：负责人交接 | 4 | 0/4 | 4/4 | 100% |
| condition：前置条件 | 4 | 0/4 | 4/4 | 100% |
| scope：适用范围 | 4 | 0/4 | 4/4 | 100% |
| no_change：后续建议不推翻原决定 | 4 | 0/4 | 3/4 | 75% |
| 总计 | 20 | 0/20 | 19/20 | 95% |

两组工作流的开头必要证据均完整保留 19/20，结尾均完整保留 20/20。
唯一未命中的是 **ami_80_endpoint**（序号 17）：开头 23 个目标 token 全部没有进入全局选择，
结尾 33 个目标 token 全部保留。V3 的候选池限于已选 KV，不能找回开头被删掉的位置。
因此，该失败属于**全局选择遗漏**，不能归因于补充检索排名错误。
只在首尾都全局保留的 19 条中，V3 同窗覆盖 19/19；这是补充诊断，主分母仍然保留 20。

## 5. Token、KV 访问与输出

以下为每事项均值；logical tokens 为 decoder 输入/输出事件，含生成的 reasoning、delimiter、EOS，
不是最终摘要词数，也不是注意力 FLOPs、计费或传输流量。embedding 编码 token 单独记录。

| 指标 | baseline | semantic | direct_text |
|---|---:|---:|---:|
| logical input tokens | 24,451.85 | 24,731.50 | 10,444.60 |
| logical output tokens | 4,755.95 | 4,781.60 | 1,471.85 |
| logical total tokens | 29,207.80 | 29,513.10 | 11,916.45 |
| 最终阶段 output token events | 1,443.15 | 1,351.50 | 1,471.85 |
| 局部 specialist 数 | 9.85 | 9.85 | 0 |
| 局部 KV position events | 12,490.30 | 13,704.05 | 不适用 |
| 最终汇总 selected KV positions | 9,614.45 | 9,614.45 | 不适用：直接全文 prefill |
| embedding input token events | 10,645.00 | 10,645.00 | 0（未调用） |

V3 比 baseline 平均增加 **305.30 logical tokens（约 1.05%）**，局部 KV position events
增加 **1,213.75（约 9.72%）**，最终 selected KV 数量不变。所有事项复用了全局选择时的
embedding，补充检索没有额外编码 token，但仍有匹配开销和额外 KV attention 计算。

V3 总 logical tokens 约为 direct 的 **2.48 倍**；本实验不能说 V3 比单模型节省 token。
本轮三组都使用内部 KV，但 direct 没有跨 agent 的 KV 交付。相较自然语言通信的节省应引用实验 1，
不能用这组三组 KV/单模型挑战结果代替。

## 6. 时间干扰：保留原值，不作为正式效率结论

| 时间统计（秒/事项） | baseline | semantic | direct_text |
|---|---:|---:|---:|
| 全部 20 条原始平均 | 296.83 | 245.46 | 51.96 |
| 全部 20 条中位数 | 187.61 | 186.67 | 51.23 |
| 第一条原始记录 | 2,512.42 | 1,382.98 | 52.94 |
| 排除第一条后的探索性平均（n=19） | 180.22 | 185.59 | 51.91 |

执行第一条期间发现未经确认的 GPU 挖矿进程，清理前持续占用约 95%–98% SM。
清理发生在该样本执行过程中；已保存阶段数据没有一个可精确校准的干扰结束事件。
因此，不能把 baseline 均值大于 semantic 解读为 V3 更快，也不能从清理前后测量推导加速比例。

“去掉第一条”仅作为透明的敏感性记录，不宣称其余记录已具备完全受控的计时条件。
两组工作流时间还含共享 prefill/selection 的独立运行估算，不能直接相加作为实际总 GPU 工作量。
正式效率比较应在干净环境重测，或使用独立且确认未受干扰的实验。

## 7. 重复与执行完整性

| 诊断 | baseline | semantic | direct_text |
|---|---:|---:|---:|
| 最终摘要包含完全相同 bullet 的样本 | 0/20 | 0/20 | 1/20 |
| 完全相同 bullet 多余条数合计 | 0 | 0 | 1 |
| 已记录的阶段 cap-hit | 0 | 0 | 0 |
| 失败记录 | 0 | 0 | 0 |

复用项目既有 `repetition_metrics`：逐行识别 bullet，合并空白并忽略大小写后做完全相同匹配。
它不会识别换措辞重复、跨章节重复或同义矛盾，也不能据此证明 workflow 更少语义重复。

## 8. 逐例内容观察：不是正式评分

下表由助手结合固定答案、首尾原文、目标相关摘要段落及部分局部报告整理。
属于**辅助定性核对**，没有独立人工审阅者、盲法或一致性统计；不转换成论文中的人工正确率。
“核心状态正确”也不等于数量、范围等全部细节完整，更不代表整篇纪要正确。
原始 `target_review.csv` 的 60 个 verdict 仍为空。

| 序号 / ID | 事项 | baseline 观察 | semantic 观察 | direct_text 观察 |
|---|---|---|---|---|
| 00 / 93 | 四台访客样机包装 | 写到布袋替代纸套；保留零售包装不变 | 正确修订与零售范围，但目标相关句没有四台数量 | 仍写纸套，漏掉末尾替换结果 |
| 01 / 14 | 两台博物馆借用设备安装 | 写到支架改底座，主产品安装另行区分 | 修订与两台范围明确，但把整场会议描述为围绕博物馆借用，存在主题扩大 | 写到两台设备改底座，主产品安装仍未指定 |
| 02 / 54 | 七包展示样品取件 | 将 Monday collection 改述为 delivery，取件与送达混淆 | 保留周一替换周五，但额外引入物流影响未明的 open issue | 保留周一取件与 launch 不变；又添加 PM 确认已接受时段的任务 |
| 03 / 9 | 六人无障碍排练房间 | 正确换到 Willow，但遗漏六人 | 六人、换房及主研究不变均提及；另增加确认已确认房间的动作 | 换房正确，但把 attendees 无细节写成疑问，遗漏已给的六人数 |
| 04 / 11 | 三台展示设备库存表交接 | 写到 Industrial Designer 接收库存表；数量和原设计职责不充分 | **错误保留 Marketing 为表格负责人** | **Action Items 仍让 Marketing 维护表格** |
| 05 / 55 | 八位访客 badge 记录交接 | Marketing 接任正确；八人数量未保留 | 写到 Marketing 接任及八条记录；tour programme 既有归属未保留 | 接任写对，但背景仍称 PM 正在编制，存在旧状态残留 |
| 06 / 53 | 两台培训设备校准日志交接 | 旧负责人背景与 UI 接手训练日志并列，未充分统一两种称呼 | **把 calibration log 与 training log 拆成两项，并保留两个负责人** | 提及校准日志不等于校准工作，但遗漏最终 UI 接任 |
| 07 / 8 | 五个 reviewer 账户交接 | 既写 UI 将创建又写 PM 接手，跨章节冲突 | PM 接手与 session organisation 独立写清，但未保留五个数量 | 重复旧 UI 任务，并把未分配给 PM 的 session organisation 写给 PM |
| 08 / 30 | 四人录音试验授权 | 保留因 lab approval 缺失而未授权，但未完整记录 releases 已到齐 | **把已收到的 participant releases 写成仍未获得** | **同样把 releases 和 lab approval 都写成未获得** |
| 09 / 126 | 三台借用设备释放 | 已过检查、获准释放且日期未定；数量未保留 | 决策写通过，但 Open Issues 又说检查状态 pending confirmation | 保留三台、检查通过、获准释放、尚未安排发运 |
| 10 / 70 | 九件展品报价条件 | 保留 210>180、九件、未授权采购 | 保留 210>180、九件、未授权采购 | 保留目标金额和未授权；另有其他原会议事实未在此全面核查 |
| 11 / 128 | 两天合作方到访 | 签署已完成、escort 缺失、日期未定；禁止出席未显式保留 | 签署已完成、escort 缺失、日期未定；禁止出席未显式保留 | 明确无 escort 不得到访；两天时长未保留 |
| 12 / 7 | 十台图书馆借用期限 | 保留 14→21 天和零售政策不变 | 决策正确；背景仍用 proposed，表述状态不统一 | 保留十台、14→21 天及零售政策不变 |
| 13 / 105 | 三套学校展示音频 | 芬兰语局部替换与通用接口不变；三套未保留 | **把 school kits 和 classroom kits 拆开，称前者仍英语、后者芬兰语** | 学校改芬兰语，但把未指定语言的通用接口补成英语 |
| 14 / 72 | 两台展览设备表面处理 | 决策局部范围写对，背景却把 matte white 归到生产规格 | 保留只限 showcase pair、生产不变；另补了未明确的生效时间/任务 | 写到两台展示设备改 matte white、生产不变 |
| 15 / 15 | 六人翻译检查问卷 | 写到 extended 替代 short，主评估不变；数量在背景中 | 写到局部替换和主评估不变，背景仍残留 short 现状 | 两处 Decisions 分别写 short 保留与 extended 替换，矛盾 |
| 16 / 28 | 五台档案设备托盘不变 | 保留泡沫托盘有效、纸隔板待研究 | 前面保留原决定，Open Issues 又称托盘决定 pending investigation，状态不一致 | 保留托盘仍有效、纸隔板仅研究 |
| 17 / 80 | 四台 outreach register 不交接 | Marketing 仍负责，但遗漏四台及仅协助细节 | Marketing 仍负责、无 joint ownership；虽未同窗仍答到核心状态 | 四台、Marketing、仅协助且无交接均写明 |
| 18 / 113 | 两房间 sound check 不改期 | 周四 10:00 不变、周五仅建议；两房间未保留 | 两房间及周四 10:00 不变，周五仅建议 | 保留周四 10:00，不提周五建议及两房间 |
| 19 / 74 | 八台借用保险不换供应商 | 保留八台和原供应商有效，未充分描述低价报价未接受 | 保留八台、低价报价未采纳、原安排不变 | 保留八台和原供应商，但遗漏低价建议未采纳 |

上表是与目标有关的观察，不根据遗漏某句就自动判整篇失败，也不将一处正确表述抵消另一处错误。
要产出正式目标正确率，需由研究者按原先固定的 correct/partial/incorrect/omitted 口径填写核对表。

## 9. 已核实的失败阶段与可追溯证据

### 9.1 同窗之后，局部报告仍可能写错负责人

`ami_11_endpoint` 的末尾明确由 Industrial Designer 接收库存表。
V3 的局部窗口 10 报告却写：

> Marketing transferred ownership of the display set inventory sheet to themselves

最终摘要接着写 Marketing 为 owner。错误在局部分析阶段已经出现，不能全部归因于最后汇总。
可查 `workflow/20261126_001350_993226/00004/semantic.specialists.json` 的 `reports[10]`，
以及同目录 `semantic.summary.txt`。这里引用的是局部报告，不是模型内部 reasoning。

### 9.2 同一事项的不同表达被拆分

`ami_53_endpoint` 中 calibration log 与末尾的 training log 指同一项交接；V3 摘要却分别归给
Industrial Designer 和 User Interface。`ami_105_endpoint` 同样把 school demonstration kits 与
classroom kits 拆成两项，导致同时保留 English 与 Finnish 两个“最终决定”。
这支持“指代/实体关联仍有问题”，不支持“检索没把证据送到”：这两例的同窗覆盖均为 True。

### 9.3 条件状态和最终结果仍会跨章节冲突

`ami_30_endpoint` 原文末尾是 releases 全到齐、只缺 lab approval。V3 的窗口 0 已写成
两者都未获得，最终摘要也保留该错误。`ami_126_endpoint` 最终摘要一处写检查通过、允许释放，
另一处又写检查 pending confirmation；其窗口 0 局部报告已经同时出现这两种说法。

### 9.4 没有同窗，不等于不能回答

`ami_80_endpoint` 全局选择丢掉开头，但保留的结尾明确说 Marketing remains responsible，
所以三组都能表达 Marketing 仍负责的核心状态。这表明，按“全部预设首尾证据完整同窗”定义的
覆盖指标是保守的输入组织指标，并不是所有样本正确回答的必要条件，更不是正确性的充分条件。

## 10. 论文可以如何表述

> 在 20 个基于真实 AMI 会议构造的首尾关联事项上，后选取语义补充将关键证据共同窗口覆盖率
> 从 0/20 提升到 19/20，同时平均增加约 1.05% 的逻辑 decoder token 事件和 9.72% 的局部
> KV position 事件。该结果说明语义补充有助于组织已选证据，但不直接证明事实准确性提升。
> 定性检查仍发现负责人归属、条件状态和同一事项别名解析错误。受执行期间异常 GPU 竞争影响，
> 本轮耗时不用于正式效率结论。

不宜写成“95% 摘要正确率”“解决了长文本注意力问题”“语义增强比单模型更准确”或
“V3 在相同证据量下优于 baseline”。本轮也没有证明最终产出可以无需人工审核。

下一步优先完成固定答案的研究者核对，并结合真实会议自动指标评估；如继续调整方法，
这 20 条会转为开发集，后续确认性评价应另留未参与修改的案例。当前不自动更改主工作流。

## 11. 可复核数据与复现

| 文件 | 用途 |
|---|---|
| [summary.json](experiment_3/results/challenge20_20261126_001225_109957/summary.json) | 自动聚合、配置、输入一致性、各方法统计及源文件 SHA-256 |
| [per_case_metrics.csv](experiment_3/results/challenge20_20261126_001225_109957/per_case_metrics.csv) | 60 行逐事项逐方法统计；Excel 可打开；不含助手虚构的准确率 |
| [analyze_challenges.py](experiment_3/analyze_challenges.py) | 仅分析现有结果，不加载生成模型，不调用网络评估服务 |
| 原始 `report/target_review.csv` | 研究者填写的固定目标核对表；当前 60 行 verdict 均为空 |

在项目根目录复算自动统计：

```powershell
conda activate latent-agents
python -B -m experiment_3.analyze_challenges `
  --run-dir D:/Download/experiment_3_challenge20/20261126_001225_109957 `
  --output-dir experiment_3/results/challenge20_20261126_001225_109957
```

脚本会重写生成的聚合 JSON/CSV，但不修改原始摘要、源会议、人工核对表或本 Markdown 的定性观察。
本文件记录的是本次结果，不能拿后续更新版本的输出替换而仍沿用相同结论。
