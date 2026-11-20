vant conversation history must be reconciled. The runnable core below implements these definitions.

# Listing 8.6: Transparent customer-service escalation score

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