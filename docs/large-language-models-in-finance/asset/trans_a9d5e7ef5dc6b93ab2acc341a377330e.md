

## 针对金融任务对大语言模型进行微调

本章提供了一份从业者指南，用于将预训练的大语言模型（LLM）适应具体的金融任务，例如情感分析、信息抽取、问答、文档分类和咨询系统。虽然第 2 章侧重于架构和训练机制，但本章回答了一个不同的问题：_给定一个基础模型，在现实的数据、计算和治理约束下，如何将其转化为可靠的金融模型？_

我们分五个步骤进行。首先从微调的理论基础入手，将其视为基于预训练参数初始化的正则化优化问题。接着讨论 _全量_ 微调，随后介绍 _参数高效_ 方法（如 LoRA 和 QLoRA），这些方法使大多数机构都能实现大模型适配。接下来，我们将这些技术与常见金融任务的监督式微调方案以及用于使模型与机构偏好保持一致的基于人类反馈的强化学习（RLHF）相结合。虽然我们会简要涉及对齐和评估，但本章的主要重点是模型本身的适配——它们如何被训练、修改并针对金融任务进行专业化。

在整个章节中，我们强调设计权衡：数据需求与计算资源、灵活性与稳定性、原始性能与治理和运营风险。目标不仅是介绍算法，更是为您提供可在不同金融环境中实例化的模板，从小型量化团队到大型受监管机构。

简而言之，本章将涵盖以下主题：

- 将通用大语言模型适配至金融任务
- 微调基础
- 全量微调方法论
- 参数高效微调（PEFT）
- 监督式微调
- 基于人类反馈的强化学习（RLHF）
- 评估与验证
- 案例研究：金融大语言模型的实际构建方式

### 3.1 生产级微调契约

本章余下部分将介绍许多适配方法，但在进入细节之前，最重要的工程要点可以首先明确：金融微调实验不仅由模型名称和训练集定义。它由决策日期、允许的信息集、任务损失、校准协议和审批边界定义。本节明确阐述该契约，以便后续代码和案例研究有明确的标准。

设每个观测为 $z_i = (x_i, y_i, \tau_i, g_i)$，其中 $x_i$ 是金融文本或结构化提示，$y_i$ 是目标，$\tau_i$ 是观测和标签可用的最早时间，$g_i$ 是分组变量，例如发行人、投资组合、交易对手、产品或交易台。生产可靠的分割选择截止日期 $T_{\text{train}} < T_{\text{valid}} < T_{\text{test}}$ 并设置

- $\mathcal{I}_{\text{train}} = \{i : \tau_i \le T_{\text{train}}\}$
- $\mathcal{I}_{\text{valid}} = \{i : T_{\text{train}} < \tau_i \le T_{\text{valid}}\}$
- $\mathcal{I}_{\text{test}} = \{i : T_{\text{valid}} < \tau_i \le T_{\text{test}}\}$

如果标签的可用时间晚于输入，则必须使用标签的可用时间。例如，财报电话会议记录可能在会议当天可用，但使用稍后价格反应的分析师标签则要等到反应窗口关闭后才可用。

因此，训练目标应被理解为一种尊重时间的加权经验风险：

```
\(\widehat{\theta} \in \arg \min_{\theta \in \Theta} \frac{1}{|\mathcal{I}_{\text{train}}|} \sum_{i \in \mathcal{I}_{\text{train}}} w_i \ell(f_\theta(x_i), y_i) + \lambda \Omega(\theta; \theta_0), \quad (3.1)\)
```

其中 $w_i$ 编码成本不对称或类别不平衡，$\Omega$ 惩罚远离基础模型的破坏性移动。该目标刻意保持谦逊：它不声称微调能发现金融真理。它只表明，适配后的模型应在实际可用的信息下降低特定任务的损失。

**提议 3.1.1** (时间分割消除直接前视)。假设微调程序使用的每个训练工件都是可用时间最多为 $T_{\text{train}}$ 的观测的可测函数。假设时间 $t \in (T_{\text{train}}, T_{\text{valid}}]$ 的每个验证决策仅使用可用时间最多为 $t$ 的工件，且时间 $t \in (T_{\text{valid}}, T_{\text{test}}]$ 的每个测试决策仅使用可用时间最多为 $t$ 的工件。那么，对于任何被评估的决策，没有直接的未来观测会进入模型输入、提示示例、检索索引、标签、超参数选择或评估记录。

证明。每个被评估的对象都由可用时间被相关决策时间上界限制的工件构成。由于构造操作是这些工件的可测函数，它们无法引入超出决策时间或之前可用信息所生成的 σ-代数之外的观测。超参数选择是在验证间隔上进行的，然后在测试间隔之前冻结，因此测试集不用于模型选择。这从结构上证明了没有直接前视。该提议并未排除更微妙的泄露，例如不同时期的重复文本或带有错误时间戳的供应商回填；这些必须通过第 15 章描述的泄露审计来检测。

校准同样重要。对于输出概率 \(p_i\) 的分类器，预期校准误差可以通过将预测分组到置信区间 \(B_m\) 并计算来估算：

```
\(\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} |\text{acc}(B_m) - \text{conf}(B_m)|.\) (3.2)
```

在金融领域，校准通常比原始准确率更重要。一个情感模型在 80% 的时间里是正确的，但无法区分 55% 的置信度和 95% 的置信度，这使得它在仓位大小、升级或面向客户的披露中难以使用。因此，校准应与准确率、F1 分数、对回撤敏感的下游指标以及人工审查结果一起报告 [Guo et al., 2017]。

下面的实现模式故意保持简洁，但它编码了最重要的生产规则：按可用时间分割，验证必填字段，并在分割为空或时间上不一致时失败。

### 列表 3.1：用于金融微调的可审计时间分割

```
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence

@dataclass(frozen=True)
class LabeledExample:
    example_id: str
    text: str
    label: str
    available_at: datetime
    group: str | None = None

@dataclass(frozen=True)
class TemporalSplit:
    train: tuple[LabeledExample, ...]
    valid: tuple[LabeledExample, ...]
    test: tuple[LabeledExample, ...]

def make_temporal_split(
    examples: Iterable[LabeledExample],
    train_end: datetime,
    valid_end: datetime,
) -> TemporalSplit:
    if train_end >= valid_end:
        raise ValueError("train_end must be strictly before valid_end")
    ordered = sorted(examples, key=lambda z: (z.available_at, z.example_id))
    if any(not z.text.strip() for z in ordered):
        raise ValueError("empty text example found")
    train = tuple(z for z in ordered if z.available_at <= train_end)
    valid = tuple(z for z in ordered if train_end < z.available_at <= valid_end)
    test = tuple(z for z in ordered if z.available_at > valid_end)
    if not train or not valid or not test:
        raise ValueError("temporal split produced an empty partition")
    return TemporalSplit(train=train, valid=valid, test=test)

def assert_no_lookahead(split: TemporalSplit) -> None:
    max_train = max(z.available_at for z in split.train)
    min_valid = min(z.available_at for z in split.valid)
    max_valid = max(z.available_at for z in split.valid)
    min_test = min(z.available_at for z in split.test)
    if max_train >= min_valid or max_valid >= min_test:
        raise AssertionError("temporal ordering violated")
```

此代码并非完整的训练脚本；它是每个完整训练脚本都应遵守的边界。一旦此边界稳定，全量微调、LoRA、QLoRA、指令调制或偏好调制之间的选择就成为工程决策，而非隐藏评估偏差的来源。

### 3.2 将通用大语言模型适配至金融任务

如前几章所述，金融语言和分析引入了通用型模型开箱即用无法完全覆盖的若干要求。金融文档依赖专业术语、复杂的数值关系以及跨报告期间的时间比较，所有这些都发生在高度受监管的环境中。此外，尽管 GPT-4、Claude 和 LLaMA 等预训练模型展现出强大的通用语言能力，但它们通常缺乏金融分析、风险评估和监管合规所需的专业知识、精度和领域特定推理 [Wu et al., 2023b]。

微调有助于弥合这一差距。通过在领域特定数据上继续训练，模型参数会逐渐适应金融文本中发现的模式和惯例。这一过程利用了预训练期间编码的广泛语言知识，同时将模型专门化用于金融情感分析、财报电话会议摘要、信用风险评估、欺诈检测和监管文档处理等任务 [Araci, 2019, Wu et al., 2023b, Yang et al., 2023]。

总体而言，尽管现代大语言模型是功能强大的通用语言模型，但在可用于金融分析或决策支持之前，通常还需要进一步适配。微调是执行这种适配的主要机制。

#### 3.2.1 微调范式

微调可以正式理解为一个优化问题。设 \(\theta_0\) 表示预训练语言模型的参数，设 \(\mathcal{D}_{\text{finance}} = \{(x_i, y_i)\}_{i=1}^N\) 为金融文本-标签对的数据集。微调目标旨在找到参数 \(\theta^*\)，以最小化特定任务的损失：

