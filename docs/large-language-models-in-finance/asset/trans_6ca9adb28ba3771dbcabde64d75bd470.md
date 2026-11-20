
必须协调相关的对话历史。下面的可运行核心实现了这些定义。

### 列表 8.6：透明的客户服务升级评分

```
from dataclasses import dataclass
def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

@dataclass(frozen=True)
class EscalationSignals:
    intent_count: int
    missing_information: float
    cross_system_dependency: float
    history_reconciliation: float
    emotion_risk: float
    regulatory_or_transaction_risk: float
    explicit_trigger: float
    loop_or_repetition: float
    hard_trigger: bool = False

def escalation_score(signals: EscalationSignals) -> dict:
    complexity = (
        0.35 * clip01(signals.intent_count / 3.0)
        + 0.25 * clip01(signals.missing_information)
        + 0.20 * clip01(signals.cross_system_dependency)
        + 0.20 * clip01(signals.history_reconciliation)
    )

    score = (
        0.30 * complexity
        + 0.25 * clip01(signals.emotion_risk)
        + 0.20 * clip01(signals.regulatory_or_transaction_risk)
        + 0.15 * clip01(signals.explicit_trigger)
        + 0.10 * clip01(signals.loop_or_repetition)
    )

    urgent = signals.hard_trigger or score >= 0.80
    escalate = urgent or score >= 0.60
    return {
        "score": round(score, 4),
        "complexity": round(complexity, 4),
        "escalate": escalate,
        "urgent": urgent,
    }
```

0.60 和 0.80 的值是示例性的操作阈值，而非通用常数。机构应在标记的历史交互上使用成本敏感目标来估计它们，该目标应对遗漏的投诉、易受伤害的客户、欺诈信号和受监管活动给予适当权重。阈值应按渠道和客户群体进行评估，监控漂移，并在工作流程或政策变化时进行审查。

## 8.3.3 个性化财务规划平台

财务规划系统可以结合目标、现金流摘要、账户余额、情景分析和教育内容，以支持更长期的对话。此类系统可能会提高基础规划支持的可用性，但应区分教育、诊断提示和受监管建议。它们还应避免将推断的生活方式或情感特征视为既定事实。

### 具体的生活通胀审查

用于持续监控的原始长伪代码掩盖了关键的分析问题：系统如何判断支出可能增长快于收入？更有用的实现首先计算一小组确定性的、隐私最小化的月度特征，然后要求 LLM 生成一份结构化审查笔记。模型接收的是汇总数据，而不是原始交易或商家名称。

当非必需支出增长持续超过收入增长、非必需支出占比上升且储蓄率下降时，简单的预筛可以定义可能的生活通胀。LLM 的角色是总结信号、说明注意事项，并为人类规划师或客户提出问题；它不会诊断客户或发出投资建议。

列表 8.7 是一个实际的 LangChain 实现。它使用通过环境变量提供的模型标识符、结构化 Pydantic 输出和显式提示。

```
import os
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class LifestyleInflationReview(BaseModel):
    flag: bool = Field(description="Whether the metrics merit review")
    severity: Literal["none", "low", "medium", "high"]
    evidence: list[str]
    caveats: list[str]
    suggested_questions: list[str]

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a financial-planning review assistant. Use only the "
        "aggregated metrics provided. Do not infer sensitive traits, do not "
        "diagnose behavior, and do not provide investment advice. Identify "
        "whether spending may be rising persistently faster than income. "
        "State counter-explanations and return a structured review note."
    ),
    (
        "human",
        "Income growth, 12 months: {income_growth_12m:.3f}\\n"
        "Discretionary expense growth, 12 months: "
        "{expense_growth_12m:.3f}\\n"
        "Change in discretionary spending share: "
        "{discretionary_share_change:.3f}\\n"
        "Change in savings rate: {savings_rate_change:.3f}\\n"
        "Months pattern persisted: {months_persistent}\\n"
        "Known one-off expense share: {one_off_expense_share:.3f}"
    ),
])

model = ChatOpenAI(
    model=os.environ["LLM_MODEL"],
    temperature=0,
)
review_chain = prompt | model.with_structured_output(
    LifestyleInflationReview
)

ALLOWED_FIELDS = {
    "income_growth_12m",
    "expense_growth_12m",
    "discretionary_share_change",
    "savings_rate_change",
    "months_persistent",
    "one_off_expense_share",
}

def review_lifestyle_inflation(metrics: dict) -> LifestyleInflationReview:
    missing = ALLOWED_FIELDS.difference(metrics)
    if missing:
        raise ValueError(f"Missing required metrics: {sorted(missing)}")

    minimized = {key: metrics[key] for key in ALLOWED_FIELDS}
    return review_chain.invoke(minimized)
```

此示例特意设计得较窄。确定性特征流水线应在 LLM 查看指标之前处理通胀调整、季节性、账户覆盖、一次性购买、家庭变化和数据质量标志。然后，结构化输出应进入审查工作流，而不是在未经验证的情况下发送给客户。

### 作为风险指标的预测性财务洞察

前瞻性信号应作为不确定的风险指标或审查提示来呈现，而不是作为保证的预测或可自动执行的建议。例如，现金流警告可能基于不完整的账户覆盖、不确定的收入时点，或异常但合法的支出。系统应显示其时间范围、假设、数据新鲜度、不确定性以及导致警报的因素。

透明评分比未指定的风险预测占位符更有用。考虑

$R_{cash} = 0.35D + 0.25V + 0.20I + 0.20L,$

其中，\( D \) 是相对于月收入的预计赤字，\( V \) 是相对于验证过的参考水平的支出波动性，\( I \) 是收入不稳定性，\( L \) 是流动性缓冲不足。每个组件都被限制在 [0, 1] 范围内。列表 8.8 实现了该评分并返回其组件以供解释。

### 列表 8.8：透明的现金流审查评分

```
from dataclasses import dataclass

@dataclass(frozen=True)
class CashFlowFeatures:
    projected_net_cash_flow: float
    monthly_income: float
    spending_volatility: float
    volatility_reference: float
    income_instability: float
    liquid_assets: float
    required_liquidity_buffer: float

def cash_flow_risk(
    features: CashFlowFeatures,
    review_threshold: float = 0.30,
) -> dict:
    income_scale = max(abs(features.monthly_income), 1.0)
    deficit = clip01(
        max(0.0, -features.projected_net_cash_flow) / income_scale
    )

    volatility_scale = max(features.volatility_reference, 1.0e-8)
    volatility = clip01(
        features.spending_volatility / volatility_scale
    )
    instability = clip01(features.income_instability)

    buffer_scale = max(features.required_liquidity_buffer, 1.0)
    liquidity_shortfall = clip01(
        max(
            0.0,
            features.required_liquidity_buffer - features.liquid_assets,
        ) / buffer_scale
    )

    score = (
        0.35 * deficit
        + 0.25 * volatility
        + 0.20 * instability
        + 0.20 * liquidity_shortfall
    )

    return {
        "score": round(score, 4),
        "review_required": score >= review_threshold,
        "threshold": review_threshold,
        "components": {
            "deficit": round(deficit, 4),
            "volatility": round(volatility, 4),
            "income_instability": round(instability, 4),
            "liquidity_shortfall": round(liquidity_shortfall, 4),
        },
    }
```

默认阈值 0.30 是一个保守的示例性人工审查触发器，而非财务困境的概率，也不是监管重要性阈值。它应根据机构的损失函数在验证集上选择：错过即将发生的短缺可能比产生额外审查更昂贵。如果将评分描述为概率，则必须首先针对观察到的结果进行校准，并报告校准误差。警报后生成的任何建议仍受上述适宜性、披露和审批控制的约束。

在仔细实施时，这些系统可以支持从纯粹被动服务向更早、可审查的规划提示转变。它们的价值来自于结合确定性测量、明确的不确定性、源数据和人类判断，而不是将模型生成的预测视为事实 [Wang and Wang, 2024]。

## 8.4 总结

本章探讨了 LLM 如何支持金融领域两个文本密集型领域：财务文档智能和面向客户的服务。文档部分涵盖了证据-
