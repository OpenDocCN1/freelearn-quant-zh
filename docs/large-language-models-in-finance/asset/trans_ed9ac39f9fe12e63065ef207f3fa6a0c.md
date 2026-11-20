
在实际应用中，金融 BPE 实现通常使用预置词汇表，将常见金融术语视为原子单位，而非允许其被拆分为子组件。示例如下：

```python
financial_seed_vocab = [
    # Financial metrics
    "EBITDA", "P/E", "ROE", "ROA", "ROIC", "EV/EBITDA",
    # Currency codes
    "USD", "EUR", "GBP", "JPY", "CHF", "CAD",
    # Financial instruments
    "NYSE", "NASDAQ", "S&P500", "VIX", "LIBOR",
    # Accounting terms
    "GAAP", "IFRS", "FASB", "SEC", "10-K", "10-Q",
    # Market terminology
    "IPO", "M&A", "LBO", "DCF", "NPV", "IRR"
]
```

数值分词策略需要在保留数值关系与分词效率之间取得平衡。以下多种方法已被证明对金融应用很有用：

- **按数量级分词**：为不同数量级（千、百万、十亿）的数值提供独立标记
- **精度感知分词**：对整数与小数采用不同处理方式
- **金融格式分词**：对百分比、货币金额、比率进行专门处理
- **上下文感知数值分词**：基于数值上下文（价格 vs. 比率 vs. 日期）采用不同分词方式

在涉及极大数值或精确百分比变化的金融场景中，科学计数法的处理尤为重要。分词方案必须在保留尾数与指数之间关系的同时，实现高效的模型处理。

金融文档还将叙述性说明与表格、图表和结构化披露相结合。多模态分词方法通过生成保留跨模态关系的统一表示来应对这种多样性。这可以表示如下：

Token_unified = Concat(Token_text, Token_numerical, Token_structural) (2.32)

最后，金融数据的位置编码必须反映金融文档的结构复杂性。与简单的顺序文本不同，金融报告包含承载分析意义的分层章节、表格依赖关系和交叉引用。因此，位置编码方案必须考虑超出线性标记顺序的内容，以保留结构上下文。

## 构建数据预处理管道

金融数据预处理管道必须在保持工作流程质量和一致性的同时，管理金融内容的复杂性和多样性。管道必须支持多种输入格式，从 PDF 和 HTML 申报文件到结构化数据源，同时生成适合模型训练的标准化输出。

核心目标是保留文档结构。金融报告经常包含承载分析意义的分层章节、表格、脚注和交叉引用。因此，预处理应保留这些结构关系，而非将其展平为无差别的文本。

结构化数据提取同样重要。财务报表和表格披露包含必须准确解析并归一化的量化关系，且不得破坏内部一致性。该过程的简化示例如下。预期输入是从申报文件中提取的已解析 HTML 或 XML 表格；预期输出是可传递给分词器或存储在结构化训练数据集中的表格标记的标准化表示。

```python
def process_financial_table(table_html):
    # Parse HTML table structure
    table_data = parse_html_table(table_html)

    # Identify financial metrics
    metrics = identify_financial_metrics(table_data.headers)

    # Normalize numerical values
    normalized_values = normalize_financial_numbers(table_data.values)

    # Validate relationships
    validate_financial_relationships(metrics, normalized_values)

    # Generate standardized representation
    return generate_table_tokens(metrics, normalized_values)
```

除原始内容外，元数据提取也起着关键作用。捕获发布日期、报告期间、财年、公司标识符、行业分类、文档类型、监管框架和源质量指标等上下文信号，能够增强模型训练和下游检索能力。

最后，交叉引用解决方案处理金融文档中常见的引用模式，其中数值、支持证据和分析结论可能分布在不同章节或关联申报文件中。预处理管道必须检测并保留这些关系，以便模型从连贯的金融叙事中学习，而非碎片化摘录。

## 2.6.2 大规模模型的训练策略

训练大规模金融语言模型需要先进的策略，以应对计算可扩展性、训练稳定性和特定领域的优化目标。本节将考察大规模训练的实际机制：协调分布式更新、管理收敛行为，并融入特定领域约束，使最终模型能够进行金融上连贯的推理。

### 分布式训练架构

训练大规模金融模型必然会超出单台设备的容量。因此，分布式方法成为协调多个 GPU、节点甚至数据中心之间学习的必要手段，同时保持效率和稳定性。因此，分布式训练策略的选择直接影响训练速度、资源利用率，并最终影响生成模型的质量。

最直接的方法是数据并行，它将训练批次分布到多个设备上并聚合梯度。这能够实现批次大小和计算吞吐量的近线性扩展。特别是对于金融模型，由于可用语料的规模以及稳定收敛通常所需的大批次大小，数据并行通常非常有效。在其最简单的形式中，全局梯度计算为设备级梯度的平均值：

Gradient_{global} = \frac{1}{N} \sum_{i=1}^{N} Gradient_{device_i} \qquad (2.33)

其中 N 表示参与设备的数量，梯度在设备之间平均，以保持训练一致性。

然而，随着模型规模增长，单靠数据并行可能不再足够。当参数数量超过单台设备的内存容量时，模型并行成为必要。流水线并行将层分区到不同设备，而张量并行则分布单个操作。在实际操作中，金融模型实现经常采用结合数据和模型并行的混合策略，以适应架构约束和可用硬件资源。

即使在分布式执行中，硬件限制也常常限制可行的批次大小。梯度累积通过在多次前向和后向传递后再执行参数更新来模拟更大的有效批次大小，从而提供了一种实用解决方案。这种方法在金融场景中尤其有价值，因为更大的批次可以提升训练稳定性并实现更平滑的收敛。简化公式如下：

```
Algorithm 4 Gradient accumulation for financial model training
Input: Model M, dataset D, accumulation steps K
Output: Trained model M'

Initialize accumulated gradients G_acc = 0
for each training step do
    for k = 1 to K do
        batch <- SampleBatch(D)
        loss <- ComputeLoss(M, batch)
        gradients <- BackwardPass(loss)
        G_acc <- G_acc + gradients / K
    end for
    Update model parameters using G_acc
    Reset G_acc = 0
end for
```

最后，分布式训练引入了通信开销，特别是在梯度同步期间。包括梯度压缩、分层聚合和异步更新在内的通信优化技术，能够显著降低这种开销。

### 优化与训练策略

除分布式协调外，金融模型训练还依赖对优化过程的精细控制。因此，优化策略必须在收敛速度、训练稳定性和最终模型质量之间取得平衡。学习率调度是这种平衡的核心。在金融场景中，通常倾向于使用保守的调度方案，包含逐步提高学习率的预热阶段，以防止早期不稳定。常见公式如下：

LR(t) = \begin{cases} \text{LR}_{base} \cdot \frac{t}{t_{\text{warmup}}} & \text{if } t \le t_{\text{warmup}} \\ \text{LR}_{base} \cdot \left( \frac{t_{\text{warmup}}}{t} \right)^\alpha & \text{if } t > t_{\text{warmup}} \end{cases} (2.34)

其中 t_warmup 表示预热期，\alpha 控制衰减率。

混合精度训练在降低内存需求并加速训练的同时，保持数值稳定性，这对金融应用至关重要。金融实现通常采用保守的混合精度方法，为关键金融计算保留更高精度 [Micikevicius et al., 2018]。当前的 PyTorch 规范是使用统一的 torch.amp.autocast(device_type=...) 和 torch.amp.GradScaler(device_type) API，而非旧的设备特定 AMP 命名空间 [PyTorch Contributors, 2026]。以下代码片段的预期输入是已放置在目标设备上的分词张量批次；预期输出是一次优化器更新和用于日志记录的标量损失值：
