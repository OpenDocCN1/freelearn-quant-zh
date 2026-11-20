

在相同的名义衰减率下，这也是 AdamW 存在的全部原因（第 2 章）。同样要注意衰减的作用和非作用：它会将参数向 *零* 收缩，而不是向其预训练值收缩。将参数锚定到预训练权重是一种不同的正则化器（一种显式的 \(\|\theta - \theta_{\text{pre}}\|^2\) 惩罚项，有时称为 L2-SP），只有在确实需要时才必须有意添加。对于金融微调，0.01 到 0.1 的值是典型的。较大的值提供更强的正则化，在训练集非常小或高度同质时更可取；较小的值允许模型有更多自由度进行专门化。一旦这些选择就位，下一个问题是它们在训练期间如何应用，这由微调程序本身体现。

## 训练程序

完整的微调循环遵循标准结构，但有几个细节值得关注。在每个批次中，任务损失会被计算并反向传播；然后由优化器将 AdamW 的解耦权重衰减应用于参数，在损失之外。每个 epoch 结束后，模型会在验证集上进行评估，并保存表现最好的检查点。早期停止会监控耐心窗口内的验证损失，如果性能在若干个 epoch 内没有改善，则停止训练，这是防止该领域典型的小规模金融数据集过拟合的主要保障。算法 5 总结了此过程：

```
Algorithm 5 Full fine-tuning algorithm
Input: Pre-trained model \(\theta_0\), training data \(\mathcal{D}_{train}\), validation data \(\mathcal{D}_{val}\)
Initialize: \(\theta \leftarrow \theta_0\), best\_metric \(\leftarrow -\infty\), patience \(\leftarrow 0\)
for epoch = 1 to max\_epochs do
    for batch \((x, y)\) in \(\mathcal{D}_{train}\) do
        Compute loss: \(\mathcal{L} = \mathcal{L}_{task}(f_\theta(x), y) + \lambda\|\theta - \theta_0\|^2\)
        Compute gradients: \(g \leftarrow \nabla_\theta \mathcal{L}\)
        Update parameters: \(\theta \leftarrow \theta - \eta g\)
    end for
    Evaluate on \(\mathcal{D}_{val}\), compute val\_metric (e.g. F1 for classification, ROUGE for generation)
    if val\_metric > best\_metric then
        best\_metric \(\leftarrow\) val\_metric
        Save checkpoint
        patience \(\leftarrow 0\)
    else
        patience \(\leftarrow\) patience + 1
        if patience \(\ge\) max\_patience then
            break {Early stopping}
        end if
    end if
end for
Return: Best checkpoint
```

这些配置选择和训练步骤共同定义了全量微调在实践中的机制。为了使这更加具体，我们将逐步讲解一个端到端的示例，说明这些元素如何在真实金融任务中结合在一起。

### 3.4.3 案例研究：针对金融情感分析的微调

本案例研究将引导你完成微调 BERT 以进行金融情感分类的完整过程，从数据准备到训练后模型的评估。它被有意地保持为端到端流程，以便每个步骤都能在整个工作流程的背景下理解，而不是孤立地理解。该任务足够简单，可以在单个 GPU 上运行，同时又足够真实，能够说明决定微调成功或失败的实际决策——数据质量、超参数选择、评估设计。

### 任务定义

目标是将金融新闻或评论中的句子分类为三种情感类别之一。这是金融 NLP 的自然起点，因为情感既具有商业价值（它直接驱动新闻驱动的交易信号、风险监控和市场监控系统），又是一个定义得足够清晰，可以拥有干净标记基准的问题。

三个类别反映了金融分析师实际描述市场驱动语言的方式：

- 正面：看涨指标、好消息、积极语调
- 中性：没有明确情感的事实陈述
- 负面：看跌指标、坏消息、悲观语调

关键的是，金融情感与一般情感在多方面不同，这些差异会让通用模型感到困惑。例如，“净销售额从 1.05 亿欧元上升至 1.12 亿欧元”这句话没有任何明显的正面词汇，但分析师会立即将其归类为正面。模型需要学习，暗示增长的数值比较是正面的，指导（guidance）、利润率（margin）和基点（basis points）等术语在语境中具有方向性含义，而避讳语言（“期望将”、“受市场条件影响”）倾向于中性。这种领域特定的映射正是微调旨在教授的内容。任务定义明确后，下一步是查看用于教会模型这种映射的数据。

### 数据集

Financial Phrasebank [Malo et al., 2014] 是一个广泛使用的基准，包含 4,840 个句子，摘自金融新闻文章，由金融专业人士标注。其相对较小的规模使其成为大多数机构内部可构建的标记数据集的现实替代品：足够大以微调预训练编码器，又足够小以至于每个标记决策都至关重要。

标注方案故意严格。标签反映了具备金融素养的读者对句子的解读，而非普通读者。以下示例中体现了这一区别：

| 文本 | 标签 |
|------|-------|
| 该公司表示将削减 200 个工作岗位 | 负面 |
| 净销售额从 1.05 亿欧元上升至 1.12 亿欧元 | 正面 |
| 该公司维持之前的指导 | 中性 |

第一个例子很可能也会被通用情感模型归类为负面——裁员普遍被视为负面消息。第二个例子更具启示性：句子完全没有情感词汇，只有数值比较。没有接触过金融惯例的通用模型通常会将其归类为中性。第三个例子说明，克制且含糊的语言（“维持之前的指导”而非“上调指导”）即使在词汇看似中性时也传递出谨慎信号。

数据集被拆分，以便模型在训练或超参数选择期间永远不会看到测试示例：

拆分：3,872 训练 / 484 验证 / 484 测试

数据准备就绪后，我们现在转向将从中学习的模型。

### 模型架构

BERT（来自 Transformer 的双向编码器表示）是这类句子级分类任务的自然选择。与只生成从左到右文本的仅解码器模型不同，BERT 的编码器双向读取整个句子，使其 [CLS] 标记表示成为整个输入的丰富摘要——这正是做出分类决策所需的。bert-base-uncased 变体（1.1 亿参数）足够大，可以学习微妙的金融语言模式，又足够小，可以在几分钟内在单个 GPU 上微调。

分类所需的架构更改很小：一个线性分类头被附加在 [CLS] 标记输出之上，将 BERT 的 768 维隐藏状态投影为三个输出 logits（每个情感类别一个）。所有这些都由 BertForSequenceClassification 自动处理：

```
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3,  # Positive, Neutral, Negative
    problem_type="single_label_classification"
)
```

此时，模型的 BERT 层带有来自一般英文文本预训练的权重，而分类头是随机初始化的。微调将同时优化两者：BERT 权重转向金融语言模式，而头部学习将结果表示映射到情感标签。

模型定义完成后，下一步是指定如何在实践中训练它。

### 训练配置

训练配置将本章前面讨论的理论选择转化为具体数值。有几个决定值得简要说明。学习率为 2e-5，刻意保守——大约是 BERT 原始预训练期间使用速率的十分之一——以避免覆盖模型已有的通用语言知识。每设备训练批次大小设置为 16，这是上面讨论的 16-32 微批次范围的下限，是单个混合精度 GPU 上使用 512 令牌输入的 bert-base-uncased 的安全默认值。评估批次大小设置为 32，因为评估期间不存储梯度或优化器状态，因此内存压力较小。有 3,872 个训练示例、单个 GPU 且无梯度累积，四个 epoch 意味着大约

4 × [ 3,872 / 16 ] = 968

次优化器更新。这个训练运行足够小，可以避免过度记忆，同时又为分类头和上层 BERT 层提供足够的更新以进行适应。

代码不指定绝对预热长度，而是使用 warmup_ratio=0.1；在此示例中，这对应于大约 97 个预热步骤。这比固定的 500 步预热更安全，因为后者会在如此小的数据集中消耗超过一半的更新。设置 weight_decay=0.01 应用解耦权重衰减，以限制相对较小的训练集上的过拟合。最后，在每个 epoch 结束时进行评估，并按 F1（而非损失）加载最佳检查点，确保保存的模型最大化在不平衡分类设置中最重要的指标。

```
from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir='./financial-sentiment-bert',
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=100,
    fp16=True, # Mixed precision training
)

