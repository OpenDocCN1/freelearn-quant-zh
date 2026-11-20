
Relevant conversation history must be reconciled. The runnable core below implements these definitions.

### Listing 8.6: Transparent customer-service escalation score

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

The values 0.60 and 0.80 are illustrative operating thresholds, not universal constants. An institution should estimate them on labeled historical interactions using a cost-sensitive objective that gives appropriate weight to missed complaints, vulnerable customers, fraud signals, and regulated activities. Thresholds should be evaluated by channel and customer population, monitored for drift, and reviewed whenever the workflow or policy changes.

## 8.3.3 Personalized financial planning platforms

Financial-planning systems can combine goals, cash-flow summaries, account balances, scenario analysis, and educational content to support longer-term conversations. Such systems may increase the availability of basic planning support, but they should distinguish education, diagnostic prompts, and regulated advice. They should also avoid treating inferred lifestyle or emotional characteristics as established facts.

### A concrete lifestyle-inflation review

The original long pseudo-code for continuous monitoring obscured the key analytical question: how does the system decide that spending may be rising faster than income? A more useful implementation first computes a small set of deterministic, privacy-minimized monthly features and then asks an LLM to produce a structured review note. The model receives aggregates rather than raw transactions or merchant names.

A simple pre-screen can define possible lifestyle inflation when discretionary expense growth persistently exceeds income growth, the discretionary-spending share rises, and the savings rate falls. The LLM's role is to summarize the signals, state caveats, and propose questions for a human planner or customer; it does not diagnose the customer or issue an investment recommendation.

Listing 8.7 is an actual LangChain implementation. It uses a model identifier supplied through an environment variable, structured Pydantic output, and an explicit prompt.

### Listing 8.7: Runnable LangChain review for lifestyle-inflation signals

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

This example is deliberately narrow. The deterministic feature pipeline should handle inflation adjustment, seasonality, account coverage, one-off purchases, household changes, and data-quality flags before the LLM sees the metrics. The structured output should then enter a review workflow rather than be sent to a client without validation.

### Predictive financial insights as risk indicators

Forward-looking signals should be presented as uncertain risk indicators or prompts for review, not as guaranteed forecasts or automatically executable advice. A cash-flow warning, for example, may be based on incomplete account coverage, uncertain income timing, or unusual but legitimate expenditure. The system should display its horizon, assumptions, data freshness, uncertainty, and the factors that contributed to the alert.

A transparent score is more useful than an unspecified risk-prediction placeholder. Consider

$R_{cash} = 0.35D + 0.25V + 0.20I + 0.20L,$

where \( D \) is the projected deficit relative to monthly income, \( V \) is spending volatility relative to a validated reference level, \( I \) is income instability, and \( L \) is a liquidity-buffer shortfall. Each component is clipped to [0, 1]. Listing 8.8 implements the score and returns its components for explanation.

### Listing 8.8: Transparent cash-flow review score

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

The default threshold of 0.30 is a conservative illustrative trigger for human review, not a probability of financial distress and not a regulatory materiality threshold. It should be selected on a validation set according to the institution's loss function: missing an impending shortfall may be more costly than generating an extra review. If the score is described as a probability, it must first be calibrated against observed outcomes and reported with calibration error. Any recommendation generated after the alert remains subject to the suitability, disclosure, and approval controls described above.

When implemented carefully, these systems can support a shift from purely reactive service toward earlier, reviewable planning prompts. Their value comes from combining deterministic measurement, explicit uncertainty, source data, and human judgment rather than from treating model-generated predictions as facts [Wang and Wang, 2024].

## 8.4 Summary

This chapter examined how LLMs can support two text-heavy areas of finance: financial-document intelligence and client-facing service. The document portion covered evidence-
