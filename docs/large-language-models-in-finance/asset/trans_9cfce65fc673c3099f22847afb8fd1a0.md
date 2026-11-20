

### 3.5.6 在全量微调、LoRA 和 QLoRA 之间进行选择

从从业者的角度来看，主要问题不是哪种方法在抽象意义上“最好”，而是在给定约束（数据、计算资源、治理要求）下哪种方法最合适。选择通常归结为几个实际考量：可用 GPU 内存有多少，从单个基础模型中需要服务多少不同任务，以及延迟和治理要求有多严格。表 3.4 总结了典型机构场景的权衡。总体建议是：仅当存在非常大的精选数据集且任务精度至关重要时才使用全量微调（例如，核心信用评分模型）；将 LoRA 作为大多数适配器风格任务的默认选择；并将 QLoRA 保留给 13B 或更大模型必须在单个 GPU 上运行的场景。

| 方面 | 全量微调 | LoRA | QLoRA | 最佳用例 |
| :--- | :--- | :--- | :--- | :--- |
| GPU 内存 | 非常高（所有参数 + 优化器） | 中等（几个百分比） | 低（4 位基础模型 + 适配器） | 银行中 13B+ 模型使用 QLoRA |
| 数据需求 | 较高（更大的拟合能力） | 中等 | 中等 | 对于非常大的精选数据集使用全量微调 |
| 推理变化 | 每个任务新检查点 | 共享基础模型，交换适配器 | 共享 4 位基础模型，交换适配器 | 为许多窄任务使用 LoRA/QLoRA |
| 稳定性 / 风险 | 灾难性遗忘风险更高 | 基础模型冻结时更安全 | 更安全，但存在量化伪迹 | 全量微调用于领域预训练；PEFT 用于任务 |
| 运营方面 | 部署回滚繁重 | 轻量级：适配器路由 | 轻量级：需要量化基础设施 | PEFT 用于快速实验和 A/B 测试 |

表 3.4. 金融应用中全量微调、LoRA 和 QLoRA 的定性比较

方法选择也取决于部署规模。表 3.5 明确了规划假设，包括纯算法比较中常缺失的计算成本行。这些估计是数量级规划指南，而非供应商报价；实际成本取决于云提供商、利用率、安全要求、上下文长度，以及工作负载是训练、批量推理还是交互式推理。

| 维度 | PoC 设置 | 部门设置 | 机构设置 |
| :--- | :--- | :--- | :--- |
| 主要目标 | 在窄数据集上证明任务可行性 | 为一个交易台、业务单元或风险职能提供服务 | 治理跨交易台、地区和控制职能的众多模型 |
| 模式 | 小型编码器或 7B 适配器 | 共享 7B–13B 基础模型并搭配 LoRA/QLoRA 适配器 | 审核过的基础模型目录、适配器注册表和受控提升管道 |
| 数据治理 | 清洗后的样本数据和人工审核 | 经批准的内部数据馈送、访问控制和审计日志 | 正式的血统、保留、隐私、法律和模型风险控制 |
| 预计基础设施 / 计算成本 | 低：租用或本地 GPU 使用数小时至数天；存储有限，无专用服务堆栈 | 中等：一到四台生产 GPU 或托管端点、计划批量作业、监控和回退模型 | 高：多 GPU 训练 / 评估环境、自动扩展推理、可观测性、安全审查、灾难恢复和独立验证 |
| 典型输出 | Notebook、指标表、错误示例 | 版本化适配器、仪表板、业务 KPI 报告 | 模型卡、验证包、部署批准、回滚计划和监控运行手册 |

表 3.5：金融微调项目的运营扩展考量因素

牢记这些权衡，我们现在转向如何为特定金融任务训练微调模型，从监督微调及其在常见 NLP 问题（如实体识别、问答和摘要）中的应用开始。

## 3.6 监督微调

监督微调使用标记数据为特定任务调整模型。有了前一节的参数高效策略——使用 LoRA 或 QLoRA 管理内存、精心策划的金融数据集和稳定的优化设置——接下来的问题是如何为不同金融问题构建训练信号。本节介绍监督微调如何涵盖最重要的任务家族：命名实体识别、问答和摘要。每个任务涉及不同的输入输出格式，并需要不同的模型架构和损失函数，但它们都共享前面描述的相同基础微调工作流。

在转向特定任务之前，有必要探讨如何将任务实际表达给现代生成模型。

### 3.6.1 监督微调的指令格式

像 BERT 这样的基于编码器的模型消耗原始文本序列并产生分类或跨度标签，因此“任务格式”隐含在模型架构和损失函数中。基于解码器的生成模型——Llama、Mistral、Falcon 及其衍生模型——工作方式不同：它们按生成符号生成自由形式文本，并且模型必须在输入中明确被告知要执行什么任务。这是通过结构化的 **指令格式** 完成的，有时称为“聊天模板”或“提示模板”，它用一致的 *角色* 集包装每个训练和推理示例。

三个标准角色是：

- **系统**：描述模型角色、能力和约束的固定前言。在金融应用中，这里存放合规相关指令——例如，“你是一名金融分析师助手。对于前瞻性陈述，始终包含适当的免责声明。”
- **用户**：包含指令或问题的输入轮次。
- **助手**：模型的响应，在微调过程中是唯一受监督的部分。损失仅在助手符号上计算；系统和用户符号被掩盖。

具体的金融指令示例如下：

```
<system>
You are a financial analyst assistant. Classify the sentiment of
the following financial text as Positive, Negative, or Neutral.
Provide only the label and a one-sentence justification.
</system>
<user>
"The company raised its full-year revenue guidance by 8 percent,
citing stronger-than-expected demand in enterprise software."
</user>
<assistant>
Positive. The guidance raise signals management confidence in
sustained demand, a bullish indicator for the stock.
</assistant>
```

特定的分隔符令牌（<system>, <user>, <assistant>）因模型系列而异——Llama 3 使用 <start_header_id>system</end_header_id>，Mistral 使用 user / assistant，等等——但三角色结构是通用的。使用 Hugging Face transformers 库进行微调时，`apply_chat_template` 方法会自动处理分隔符插入：

```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

messages = [
    {"role": "system", "content": "You are a financial analyst assistant..."},
    {"role": "user",
     "content": "Classify sentiment: 'Revenue declined 12%...'"},
    {"role": "assistant", "content": "Negative. A 12% revenue decline ..."},
]

# apply_chat_template inserts correct delimiter tokens for this model family
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)
```

使用 `apply_chat_template` 而非手动编写分隔符对于兼容性很重要：训练时和推理时的模板不匹配会产生系统性降级的输出，且难以诊断。对于多轮金融对话——例如，模型必须回答先前引用的收入数字的后续问题——对话历史记录作为交替的用户 / 助手轮次追加，模型学习根据所有先前上下文条件化其响应。

有了这种结构，我们现在可以转向特定的金融任务，从最基础的任务开始：从文本中提取结构化信息。

### 3.6.2 命名实体识别（NER）

命名实体识别是最基础且最具实际价值的金融 NLP 任务之一。在其原始形式中，NER 回答这样一个问题：给定一段金融文本，哪些符号指代特定的命名实体，实体类型是什么？在金融工作流中，这至关重要——从收益公告、监管文件或新闻文章中自动提取公司名称、货币金额、日期和金融工具，是事件检测、知识图谱构建和风险监控等下游任务的前提。如果模型将公司名称与产品名称混淆，或无法将 EBITDA 等财务指标识别为独立的实体类型，将在所有依赖其输出的系统中传播错误。

金融 NER 比通用 NER 更难，原因有二。首先，实体集更丰富：金融引入了标准 NER 基准中不存在的领域特定类别，如金融工具、指标和监管标签。其次，许多符号本身具有歧义：“Apple”是公司，但不同语境下的“apple”是水果；“margin”可能是财务指标，也可能是普通英语；“10-year”可能描述国债，也可能是 CEO 的任期。金融 NER 模型必须学会区分的实体类型包括：

- **ORG**：公司、机构（Apple Inc.，美联储，高盛）
- **PERSON**：高管、分析师（Tim Cook，Warren Buffett）
- **MONEY**：货币金额（$1.2 billion，€500 million）
- **PERCENT**：百分比、基点（15.3%，25bps）
- **DATE**：日期和时期（Q3 2024，FY2023）
- **INSTRUMENT**：金融产品（AAPL 股票，10 年期国债，S&P 500）
- **METRIC**：财务指标（EBITDA，P/E ratio，EPS）

