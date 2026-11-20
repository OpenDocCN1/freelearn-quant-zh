

相关文档，模型越有可能将其作为基础，而不是依赖下方的噪声。

\(\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}\) (4.17)

where rank\(_i\) is the position of the first relevant document for query \(i\).

**归一化折扣累积增益（NDCG）** 通过允许分级相关性来推广 Precision@k 和 MRR：完全回答查询的段落所贡献的分数高于仅相关的段落。这与金融用例高度一致，其中多份披露可能均相关，但程度各异。

\(\text{DCG}@k = \sum_{i=1}^k \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}\) (4.18)

\(\text{NDCG}@k = \frac{\text{DCG}@k}{\text{IDCG}@k}\) (4.19)

where IDCG is the ideal DCG (perfect ranking).

## 4.10.2 生成指标

**忠实性** [Es et al., 2023]: 答案是否与上下文一致？对于 \(m\) 个实质性声明，使用 fail-closed 分段定义

\(\text{Faithfulness} = \left\{ \begin{array}{ll} \frac{\text{Number of supported claims}}{m}, & m \ge 1, \\ \perp, & m = 0. \end{array} \right.\) (4.20)

The \(m=0\) branch is not scored as one: it is undefined and must be rejected unless the task explicitly permits a non-claim response such as a refusal or formatting operation.

**答案相关性**: 答案是否回应了查询?

\(\text{Relevancy} = \text{sim}_\text{cos}(f(q), f(a))\) (4.21)

where \(a\) is the generated answer.

**上下文相关性**: 检索到的上下文是否相关?

\(\text{Context Relevancy} = \frac{1}{k} \sum_{i=1}^k \text{relevance}(q, d_i)\) (4.22)

## 4.10.3 作为正式合同的声明级忠实性

聚合忠实性评分是有用的看板，但金融审查通常需要一个更强的对象：逐声明合同。董事会备忘录、风险解释或监管答案仅因为听起来合理就不足以被接受；每个实质性声明都必须可追溯到可用、已授权且已引用的证据。

**Definition 4.10.1** (Atomic claim support). Let an answer $a$ be decomposed into material atomic claims $A_1, ..., A_m$. For each claim $A_i$, let $E_i \subset \mathcal{D}_{\le t}$ be an evidence bundle retrieved from the authorized corpus available at decision time $t$. A verifier $\mathcal{V}$ assigns $\mathcal{V}(A_i, E_i) = 1$ when the evidence entails or directly supports the claim under the task's review standard, and 0 otherwise. The claim-level score is the piecewise quantity in Definition 4.3.1: for $m \ge 1$ it is $m^{-1} \sum_{i=1}^m \mathcal{V}(A_i, E_i)$; for $m = 0$ it is $\perp$ and fails closed unless the task class explicitly permits a non-claim response.

**Proposition 4.10.1** (Verified-claim grounding). Assume a RAG renderer is constrained to output only claims from a verified set $\mathcal{A}^\star = \{A_i : \mathcal{V}(A_i, E_i) = 1\}$ together with citations to the corresponding evidence bundles $E_i$. Then every material claim in the rendered answer is supported relative to the verifier and the authorized corpus, so Faith$(a, E) = 1$ for the rendered material claims. The proposition is relative to the verifier; it does not imply that the corpus itself is true, complete, current, or legally sufficient.

**Proof.** By construction, each material claim emitted by the renderer belongs to $\mathcal{A}^\star$. Therefore, for every emitted claim $A_i$, the verifier has already returned $\mathcal{V}(A_i, E_i) = 1$. Substituting these ones into the definition of Faith gives a score of one. The final caveat follows because the support relation is evaluated against the selected corpus and verifier, not against all facts in the world.

下面给出了一个确定性的冒烟测试实现。它不如专家审查强，但在持续集成中很有价值，因为它可以防止未引用或未支持的声明悄悄通过金融 RAG 流水线。

```python
from dataclasses import dataclass
from typing import Iterable, Sequence

@dataclass(frozen=True)
class Claim:
    text: str
    material: bool = True

@dataclass(frozen=True)
class Evidence:
    document_id: str
    timestamp: str
    text: str

def token_set(text: str) -> set[str]:
    return {tok.lower() for tok in text.replace("/", " ").split()
            if len(tok) > 2}

def supports_claim(claim: Claim, evidence: Sequence[Evidence],
                   min_overlap: float = 0.35) -> bool:
    claim_tokens = token_set(claim.text)
    if not claim_tokens:
        return False
    evidence_tokens = set().union(*(token_set(e.text) for e in evidence))
    return len(claim_tokens & evidence_tokens) / len(claim_tokens) >= min_overlap

def faithfulness_score(claims: Iterable[Claim],
    evidence: dict[str, Sequence[Evidence]]) -> float:
    material = [c for c in claims if c.material]
    if not material:
        return 1.0
    supported = [supports_claim(c, evidence.get(c.text, [])) for c in material]
    return sum(supported) / len(supported)
```

不应过度宣传冒烟测试。词元重叠会遗漏改写、数值蕴含、否定和法律细微差别。其作用是强制执行接口：答案必须分解为声明，声明必须附带证据包，未支持的实质性声明必须在发布前变得可见。更强的部署使用结合精确引用检查、数值一致性检查、嵌入式检索和专家标记的蕴含测试的混合验证器替换 supports_claim。

## 4.10.4 受监管金融问答的评估协议

对于受监管的用例——例如回答关于资本要求、市场滥用规则或内部政策的问题——RAG 评估必须超越通用的 IR 和 QA 指标。一个最小但稳健的协议通常包括以下步骤:

- 1. **定义评估范围：** 选择一个或多个明确的政策领域（例如，MiFID II 产品治理、市场操纵、适当性），并构建与生产环境中使用的文档（法规、内部政策、常见问题解答、监管指导）相匹配的语料库。

- 2. **创建金标准测试集：** 起草代表真实用户问题（来自合规、前台、运营）的查询，并让主题专家编写：
  - 标准答案，
  - 支持每个答案的文档或段落列表，
  - 指示错误答案影响程度的严重性标签。

- 3. **对整个流水线进行仪器化：** 记录检索结果、选定的上下文、生成的答案以及所有相关元数据（模型版本、索引版本、提示模板），以便每次测试运行都可复现。

- 4. **联合测量检索和生成：** 除了 Recall@k、Precision@k、MRR 和 nDCG 之外，还需计算：
  - **忠实性：** 答案中的声明是否实际由检索到的段落支持。
  - 基础率：包含至少一个引用到相关监管或政策文档的答案的比例。
  - 关键错误率：既不正确又根据专家判断属于高严重性的答案的比例。

- 5. 运行专家审查：让合规或法律审查员对系统答案样本进行审计，重点关注高严重性查询。记录二元判断（可接受 / 不可接受）以及关于缺失警告或上下文的定性反馈。

- 6. 定义准入/否决阈值和监控：设定明确的验收标准（例如，测试集中零关键错误，忠实性高于给定阈值），并定义如何在生产环境中监控这些指标（持续采样，当语料库或模型变化时定期重新评估）。

该协议将离线评估直接与机构的治理和模型风险流程联系起来：测试用例与真实义务保持一致，高严重性故障被明确跟踪，评估结果纳入更广泛的伦理、治理和合规框架。

## 4.10.5 端到端评估框架

在定义了要测量的内容以及赋予这些测量意义的协议之后，我们现在转向如何实际计算这些测量。下面的类将前面介绍的所有指标——Recall@k、Precision@k、MRR、NDCG、忠实性、答案相关性、上下文相关性和延迟——汇集到一个评估工具中，该工具可以针对带标签的测试集运行。

该实现有两部分。EvaluationResult 是一个数据类，它将完整的指标集捆绑到单个对象中，因此每次测试运行都会产生结构化、可比的记录，而不是松散的数字集合。RAGEvaluator 是该工具本身：其 evaluate 方法遍历测试用例，查询 RAG 系统，计算每个查询的检索和生成指标，最后对整个集合进行平均。在实践中，这是生产团队将其接入 CI、发布门控以及上述协议中提到的定期重新评估运行的对象。

## 列表 4.3：RAG 评估实现

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import re

import numpy as np

@dataclass
class EvaluationResult:
    retrieval_precision: float
    retrieval_recall: float
    mrr: float
```

