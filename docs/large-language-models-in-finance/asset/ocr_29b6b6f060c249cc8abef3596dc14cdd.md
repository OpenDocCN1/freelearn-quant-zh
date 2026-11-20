```
```python
self.tools = tools

def run(self, state):
    snapshot = self.tools.fetch_snapshot(
        tickers=state.tickers,
        horizon=state.horizon
    )
    state.data_metrics = compute_standard_metrics(snapshot)
    state.audit_log.append({
        "agent": "data",
        "snapshot_id": snapshot.id
    })
    return state

class RiskComplianceAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def run(self, state):
        flags = self.tools.check_limits_and_policies(
            memo=state.draft_memo,
            portfolio_id=state.portfolio_id
        )
        state.risk_annotations = self.llm.complete_json(
            role="risk and compliance reviewer",
            task=("Annotate the memo with risk, policy, "
                  "and escalation flags."),
            inputs={
                "memo": state.draft_memo,
                "policy_flags": flags
            },
            schema="RiskAnnotations"
        )
        state.human_review_required = has_material_flags(
            state.risk_annotations
        )
        state.audit_log.append({
            "agent": "risk",
            "flags": flag_ids(flags)
        })
        return state
```
```

The agents are deliberately asymmetric. The Data Analyst agent may not need an LLM for basic calculations, while the Research Analyst and Risk and Compliance agents use an LLM to interpret textual evidence and policy language. This asymmetry is normal: not every component in an agentic system needs to be equally “agentic.”