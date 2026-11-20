

## 针对金融 NER 的微调

NER 被表述为 token 分类问题：模型为输入序列中的每个 token 分配一个标签。标准的标注方案是 BIO 标记，其中 B- 标记实体跨度的开始，I- 标记同一跨度内的延续 token，O 标记不属于任何实体的 token。此方案使模型能够处理多 token 实体（例如，“Goldman Sachs”跨越两个 token，或“$81.8 billion”跨越三个 token），而无需任何特殊架构更改。

对于金融 NER，使用带大小写的预训练模型而非不带大小写的模型很重要，因为大小写对专有名词携带强信号——公司名称、高管姓名和国家代码在金融文本中比在随意语言中更可靠地大写。下面的代码片段展示了模型设置和单个标记的训练示例：

```python
from transformers import AutoModelForTokenClassification

# Load pre-trained model
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased", # Cased for proper nouns
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)

# Example training data (BIO tagging)
train_example = {
    'tokens': ['Apple', 'reported', 'Q3', 'revenue', 'of', '$',
               '81.8', 'billion'],
    'labels': ['B-ORG', 'O', 'B-DATE', 'O', 'O', 'B-MONEY',
               'I-MONEY', 'I-MONEY']
}

# Fine-tune on financial NER dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=financial_ner_train,
    eval_dataset=financial_ner_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

### 评估指标

NER 评估以 F1 作为主要指标，但可以在两个不同的层级上计算，选择对于金融应用很重要。实体级 F1 要求模型正确识别实体的完整跨度及其类型——部分匹配（例如，在“$81.8”上预测 B-MONEY 但遗漏“billion”）视为失败。Token 级 F1 对部分正确的跨度给予部分得分。对于下游金融管道，其中实体跨度供给计算器或数据库，实体级 F1 是正确的指标：部分提取的金额没有用处，并可能引入错误。

- 实体级 F1：整个实体跨度和类型匹配时正确
- Token 级 F1：每个 token 预测正确
- 每类指标：每个实体类型的单独 F1

在金融 NER 中，单独跟踪每类指标很有价值，因为性能在不同实体类型之间差异很大：ORG 和 MONEY 往往学得较好（出现频繁且具有清晰的文本信号），而较稀有的类型如 INSTRUMENT 或 METRIC 可能需要更大或更专业的训练集才能达到相同的准确度。

虽然 NER 专注于识别和标记实体，但许多金融应用需要更进一步，将提取的信息与推理和计算相结合。这引出了我们关于问答的问题。

## 3.6.3 金融问答

对金融文档的问答比 NER 或情感分类在质上更难，因为模型必须推理和计算，而不仅仅是识别或标记。它必须定位分散在混合文本和表格中的相关信息，确定所需的计算，执行多步算术，并以正确的格式和单位表示结果。一个能够准确解析资产负债表但应用了错误公式的模型——在要求计算毛利率变化时计算同比营收增长——没有价值，并且可能主动误导分析师。

### FinQA：数值推理数据集

FinQA [Chen et al., 2021] 是这项能力的标准基准。它由与年报表格和文本段落配对的问题组成，正确回答需要一个多步推理链，将数值提取与算术集成。下面的示例说明了挑战：问题询问两年间的毛利率变化，这需要计算两个比率并求其差值——这些都没有在源中直接说明：

表：Apple Inc. 损益表（百万美元）

| 年份 | 营收 | 营业成本 | 毛利润 |
|---|---|---|---|
| 2022 | 394,328 | 223,546 | 170,782 |
| 2023 | 383,285 | 214,137 | 169,148 |

问题："从 2022 到 2023 年的毛利率同比变化是多少？"

推理步骤：
- 1. 计算 2022 年毛利率：170,782 / 394,328 = 43.31%
- 2. 计算 2023 年毛利率：169,148 / 383,285 = 44.13%
- 3. 计算变化：44.13% - 43.31% = +0.82 个百分点

请注意，尽管绝对营收和利润下降，毛利率从 2022 年到 2023 年实际上略有*改善*——这种细微差别只有通过执行完整计算而不是仅阅读营收变化方向才能显现。这正是通用模型在没有任务特定微调的情况下经常错过的多步推理类型。

### 针对金融问答的微调

下面四种能力描述了模型必须内化的内容，以可靠地回答 FinQA 风格的问题。每种都代表不同的学习挑战，任何一个阶段的表现不佳都会通过推理链的其余部分传播错误：

- 1. 从表格/文本中提取相关数字
- 2. 确定所需的计算
- 3. 执行多步数值推理
- 4. 生成带正确格式的最终答案

T5（Text-to-Text Transfer Transformer）非常适合这项任务，因为其编码器-解码器架构既处理阅读理解组件（编码器共同关注表格和文本），也处理答案生成组件（解码器生成推理链和最终值）。关键的是，通过训练解码器在最终答案旁生成显式推理程序——编码为分隔符后的字符串——模型学会展示其工作过程。这既提高了准确性（奖励结构正确的推理链），也提高了可解释性，因为分析师可以验证步骤，而不仅仅是接受一个裸数字：

```python
# Fine-tune T5 for Financial QA
from transformers import T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Format as text-to-text
def format_finqa_example(example):
    input_text = f"""
    Answer the financial question based on the table and text.

    Table: {example['table']}
    Text: {example['text']}
    Question: {example['question']}
    """

    target_text = f"{example['answer']} | {example['reasoning']}"
    return {'input': input_text, 'target': target_text}


# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=finqa_train.map(format_finqa_example),
    eval_dataset=finqa_val.map(format_finqa_example),
)
trainer.train()
```

## 3.6.4 金融文本摘要

摘要是最直接有用的金融 NLP 应用之一。分析师和组合经理每周常常面对数百页的财报电话会议记录、10-K 文件和研究报告。一个可以可靠地将这些压缩为简洁、事实准确的摘要而不幻觉数字或遗漏重要披露的模型，可以显著提高投资研究工作流程的效率。

### 财报电话会议摘要

财报电话会议呈现了一种特别苛刻的摘要问题。典型的跨度记录在 10,000–20,000 个单词之间，分布在准备好的管理层演示和延长的分析师问答环节中，两者合计超过了大多数标准 Transformer 模型的上下文窗口。期望的输出是一个 200–300 单词的摘要，保留分析师最关心的四个要素：

- 财务业绩亮点（营收、盈利、利润率与预期相比）
- 关键业务进展（产品发布、收购、重组）
- 前瞻性指引（更新的营收或盈利展望）
- 管理层展望和定性基调

为了使这具体化，我们现在来看摘要模型如何针对此场景进行微调。

### 针对摘要的微调

因为财报电话会议记录经常超过标准 BERT 风格编码器的 512-token 限制，架构必须能够处理长文档。Longformer 编码器-解码器（LED）通过稀疏注意力实现这一点——每个 token 局部关注其邻居，并全局关注一小组指定的 token，而不是以二次方式关注所有 token。这将内存从 O(n²) 降低到大约 O(n)，并使单个 GPU 上可处理 16,000-token 的输入。然后对解码器进行微调以生成结构化摘要：

