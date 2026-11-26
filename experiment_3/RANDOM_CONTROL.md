# V3 的随机补充对照

问题：同窗覆盖改善来自语义关联选择，还是单纯增加局部上下文？

复用已保存的 baseline 和 semantic 结果，仅新生成 random。全局 selected positions、原始窗口、
最终全局 KV、模型、阶段提示词和生成 seed 不重新选择。为了获得可读取的源 KV，新进程仍需
加载生成模型并全文 prefill 一次/会议；这不等于重跑原两组，不会重做 selectors 或 embedding。

## 随机匹配规则

- 从全局已选、窗口外、满足相同邻居半径约束的块中取候选。
- 复用非空内容、非主窗口重复内容、正信息权重等基础过滤；不使用余弦、答案或端点位置加权。
- 使用已保存审计文本做过滤；它与线上 tokenizer.decode 在块边界可能略有差异，需披露。
- 每个窗口补充块数与旧 semantic 窗口相同，包括其没有补充块时保持 0。
- 默认 V3 每窗口最多 2 块：枚举相同块数的候选组合，找到补充 token 数最接近的一批，
  用独立固定 seed 从等价组合中均匀随机选取，不按语义或目标命中选取。
  如果旧配置每窗口超过 2 块，则退回最多 512 次随机抽样、保留长度最接近的组合。
- 允许随机选到 V3 同一块，不刻意排除正确证据。候选内容去重，合并后保持源位置顺序。
- 不截断发言块凑 token；逐窗口报告目标量、实际量和绝对偏差。因此称“长度匹配随机对照”，
  不预先声称严格等量。结果确认后才判断匹配是否足够接近。

严格长度匹配可能限制可选组合，甚至使部分窗口随机选到与 V3 完全相同的块。
这不应人为排除，否则是在刻意削弱随机组；`same_supplement_set` 和重合块数会如实记录。
该组是以长度/块数为条件的随机对照，不是从所有远距离块中无条件均匀抽样。

本批 20 条的离线方案核对（不加载生成模型、不查看目标命中）中，197/197 个窗口均匹配到
相同补充 token 数，合计两组各 24,275 个补充 positions；23 个非空窗口选中了与 V3 相同的
补充集合。这是抽样方案检查，不是随机组摘要质量或覆盖率结果。

random_seed=7 只控制证据抽样，生成仍继承旧配置，不混用两种 seed。单个随机 seed 是初步对照；
若用于较强的确认性结论，应预先固定多个随机 seed，避免只挑其中一个结果。

## Linux：先跑前三条

同步代码并进入项目根目录：

```bash
conda activate latent-agents
cd /root/autodl-tmp/Latent-communication-meeting_summary

SOURCE_RUN=test_result/experiment_3_challenge20/20261126_001225_109957/workflow/20261126_001350_993226

python -m experiment_3.random_control \
  --source-run "$SOURCE_RUN" \
  --output-dir test_result/experiment_3_random_control_3_seed7 \
  --num-samples 3 \
  --random-seed 7 \
  --answer-key input/ami_endpoint_challenge_20_v1/answer_key.json \
  --device cuda
```

模型、summary thinking、温度、生成 seed 等继承 source-run 的 manifest，无需重复传入。
若旧 runtime 记录了 Hugging Face commit，脚本在加载前解析并固定到该 commit 的本地 snapshot，
权重和 tokenizer 都从同一快照读取。不会先加载当前 main 再因 revision 不同报错。
内容相同的文件通常可复用 Hub 缓存；旧快照缺少的文件仍需下载。runtime 同时记录固定 commit、
snapshot 路径及配置对象报告的 revision，避免把本地快照加载时的空 `_commit_hash` 当作模型变化。
脚本检查源位置映射和关键生成代码是否与旧对照一致；如提示不一致，应先查明版本差异，
不能跳过检查后仍称只有证据选择改变。

只看补充量偏差、不加载模型：给上述命令加 `--plan-only`；此时不生成摘要或覆盖结论。

## 输出

每次生成一个时间戳目录：

- `comparison.txt`：三个方法在本次可核对案例上的同窗覆盖及 token 匹配偏差。
- `window_matching.csv`：逐窗口补充块数和 token 量偏差。
- `00000/random.summary.txt`：随机组最终摘要。
- `00000/random.specialists.json`：局部报告、源位置与阶段记录。
- `00000/random.windows.json`：生成前冻结的抽样结果。
- `00000/target_coverage.json`：生成后才读取答案坐标，计算输入覆盖。
- `00000/random.metrics.json`：逻辑 token 事件与 replay 阶段成本。
- `status.json`、`failure.json`：完成与失败信息。

本次覆盖的 baseline/semantic 使用相同案例的旧记录，不拿完整 20 条均值对比前三条随机组。
新 random 的耗时不含 selector/embedding，显存中也不加载 embedding，不能与旧全流程耗时/显存
直接比较。shared logical input 事件为旧工作流继承的独立运行记账，不代表本次重新执行过 probes。
最终目标正确性仍需按固定答案核对，覆盖率不是正确率。

三条检查后，将 `--num-samples 3` 改为 `--num-samples 0`，输出目录换成
`test_result/experiment_3_random_control_20_seed7` 即可生成全部 20 条随机组。
