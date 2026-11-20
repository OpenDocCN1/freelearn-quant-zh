

## 监控与可观测性

即使基础设施安全且配置充足，金融 AI 系统也不能无人看管。它们在整个部署生命周期中需要对模型性能、数据质量、合规状态和底层系统健康状况进行持续监控和可观测性。

在模型层面，性能监控跟踪不同市场条件和时间跨度下的准确性、偏差、公平性和可靠性等指标。监控不能是被动的；系统必须检测性能下降并启动相应的修复工作流。

以下示例说明了此类监控循环在实践中可能的结构：

```
class FinancialModelMonitor:
    def __init__(self):
        self.metrics = ['accuracy', 'bias', 'fairness', 'reliability']
        self.thresholds = self.load_regulatory_thresholds()
        self.alerting = AlertingSystem()

    def monitor_model_performance(self, model_outputs, ground_truth):
        for metric in self.metrics:
            value = self.compute_metric(metric, model_outputs, ground_truth)

            if value < self.thresholds[metric]:
                self.alerting.trigger_alert(
                    severity='HIGH',
                    message=f'Model {metric} below threshold: {value}',
                    remediation='Initiate model retraining'
                )
```

除了性能指标，数据漂移检测起着核心作用。金融模型特别容易受到不断变化的市场体制和宏观经济周期驱动的输入分布转移的影响。及早识别此类转移有助于防止无声的性能下降。

还必须监控基础设施健康状况。计算资源、网络性能、存储利用率和安全态势需要在分布式环境中实现实时可见性。因此，运营监控补充了模型层面的评估。

最后，合规仪表板为风险经理和合规官提供结构化监督。这些系统将性能、漂移和审计信号整合为支持内部治理和监管审查的视图。

### 2.6.4 完成金融 LLM 管道

从零开始构建金融 LLM 最终需要将预处理、训练和部署集成到一个连贯且可重复的工作流中。前几节分别研究了这些组件。在这里，我们将它们结合在一个具体的端到端示例中，该示例从原始 SEC 10-K 文件开始，最终以能够生成金融洞察的 Transformer 模型结束。

从高层次来看，管道分为三个紧密连接的阶段：

- 1. **预处理与摄入**：获取 10-K、10-Q 或 ESG 报告等文档，跨格式（PDF、HTML、纯文本）进行归一化，并使用领域特定的分词进行处理，以便金融术语、股票代码和数值被一致表示。

- 2. **模型训练与适应**：使用生成的分词语料库，在现实的数据和计算约束下训练或微调基于 Transformer 的语言模型，目标包括掩码语言建模、下一令牌预测或指令调优。

- 3. **服务与监控**：将训练好的模型部署在 API 或代理框架后面，在适当的地方增加检索或工具，并持续监控其质量、延迟和安全性。

预处理和训练循环的最小伪代码草稿可能如下所示。预期输入为原始文件、分词器、模型和优化器；预期输出为更新后的模型状态和标量训练损失：

```
# 1. Load and clean a raw 10-K
raw_text = load_sec_filing(cik = "0000320193", form = "10-K", year = 2024)
clean_text = clean_filing(raw_text) # remove contents pages and boilerplate

# 2. Chunk and tokenize
chunks = chunk_document(clean_text, chunk_size = 2048, overlap = 256)
tokenized = tokenizer(chunks,
    truncation = True,
    max_length = 2048,
    return_tensors = "pt")

# 3. Forward pass through a small financial Transformer
optimizer.zero_grad(set_to_none = True)
outputs = model(**tokenized)
loss = compute_language_modeling_loss(outputs.logits, tokenized["input_ids"])
loss.backward()
optimizer.step()
```

在生产环境中，这个简化的循环会扩展为包含分布式数据加载、梯度累积、混合精度训练和结构化评估。然而，底层逻辑保持不变：混乱的金融文档被转换为令牌序列，模型被优化以预测或有意义地转换这些序列。

还需要考虑配置选择如何随部署规模而变化。从业人员经常询问，概念验证与部门或机构部署相比，合理的默认设置是什么。表 2.3 总结了三种常见场景中的配置模式。

| 设置 | 小型 PoC | 部门级 | 机构级 |
|---|---|---|---|
| 参数数量 | 1–7B | 7–20B | 20B+ |
| 上下文长度 | 4k–8k | 8k–32k | 32k+，配合 RAG |
| 精度 | 8-bit / 4-bit | 8-bit / 4-bit + FP16 | FP8/FP16，带张量并行 |
| 分词器 | 通用 + 轻量金融适配 | 金融特定词汇表 | 金融特定 + 多语言词汇表 |
| 微调 | LoRA/QLoRA | LoRA/QLoRA + 指令调优 | 关键任务上的 PEFT 组合 + 全量微调 |
| 检索 | 简单向量存储 | 混合搜索（BM25 + 向量） | 带治理层的多层 RAG |
| 基础设施 | 单 GPU / CPU | 小型 GPU 集群 | 带 MLOps 栈的专用 GPU Pod |
| 估计基础设施 / 计算成本 | 低：工作站或租用单 GPU | 中：项目级 GPU 预算 | 高：专用集群加上 MLOps、安全和合规开销 |

表 2.3：不同规模金融 LLM 的示例配置模式

这些配置刻意近似。其目的不是规定单一设计，而是将架构决策锚定在具体的资源和风险约束中。小型概念验证可能在单个 GPU 上使用参数高效微调运行，而机构部署则需要关注分片、故障转移、监控以及本章前面讨论的监管控制。

## 2.7 总结

本章从基础概念进展到完整系统实现。我们首先从数学层面考察了 Transformer 架构，拆解注意力机制、多头结构、归一化和位置编码，以理解 LLM 如何处理复杂的金融文本和数值数据。在此基础上，我们探讨了规模化这些模型的工程现实，还追踪了从分词和嵌入到词汇表设计的表征管道，展示了金融语言、标识符和数值结构如何编码到向量空间中。在此基础上，我们研究了预训练目标、语料库设计、迁移学习以及大规模模型开发的计算经济学。

本章后半部分从模型构建转向模型使用和部署。我们分析了结构化提示设计和推理增强技术，例如少样本学习和思维链提示，并引入 RAG 作为面向最新、可审计且权限感知的金融知识的基础层。最后，我们对如何从零开始构建金融 LLM 进行了全面的演练。现在，您了解金融语言模型的内部工作原理，以及如何在真实的机构环境中设计、训练、规模化、治理和部署它们。

在下一章中，我们将探讨如何使用全量微调、LoRA、QLoRA 和 RLHF 将通用预训练模型调整为财报电话会议情感分析、监管问答、数值推理、顾问支持和其他具体金融工作流。

