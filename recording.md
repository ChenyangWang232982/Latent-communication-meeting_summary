Question

- 从Encoder里提前最后一层hidden state作为embedding输入到adapter并与prompts embedding 拼接，这属于embedding方法还是hidden state方法 (未解决)
- 需要一个云平台来高效训练模型 (半解决)
- 在训练过程中出现了过拟合的问题 (测试集损失持续下降，但验证集损失保持在2.7)
- 如何评估损失分数, 在summary上以算法来进行评估有概率出现误判的情况，可能需要引入AI参与loss打分
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
