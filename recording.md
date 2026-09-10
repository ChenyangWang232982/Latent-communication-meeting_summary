Question

- 从Encoder里提前最后一层hidden state作为embedding输入到adapter并与prompts embedding 拼接，这属于embedding方法还是hidden state方法 (解决)
- 需要一个云平台来高效训练模型 (解决)
- 在训练过程中出现了过拟合的问题 (测试集损失持续下降，但验证集损失保持在2.7)
  - 基于语义判断的问题，损失函数无法有效做到去计算分数，因此在训练过程中, loss下降只能作为一个参考，对于Adapter是否有效的传输的信息，需要在后续的评估阶段进行判断。
  - 当前训练暂时只使用了交叉熵损失，在后续评估环节中可以依据结果对损失函数进行调整。
- 分块压缩中参数的权衡 (需要做更多的实验来判断)

Progress

- [x] Embedding adapter framework
- [ ] Optimization
- [ ] Evaluation Framework
- [ ] Hidden state adapter framework
- [ ] Optimization
- [ ] Comparision
- [ ] KV adapter framework
- [ ] Optimization
- [ ] Comparision
- [ ] Normal prompts framework
- [ ] Comparision
