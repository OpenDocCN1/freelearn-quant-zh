# Step 6: Add validation, guardrails, and review routing

The minimal run above demonstrates orchestration, but it does not yet enforce the controls required for a governed workflow. A practical implementation should validate outputs at every boundary. Validation should be done by deterministic code wherever possible, with LLM-based checking used as a supplement rather than the only control.

```
def validate_workflow_state(state):
    validate_schema(state.research_notes, "ResearchNotes")
    validate_schema(state.data_metrics, "DataMetrics")
    validate_schema(state.draft_memo, "DraftInvestmentMemo")
    validate_schema(state.risk_annotations, "RiskAnnotations")

    if contains_uncited_material_claim(state.draft_memo):
        state.human_review_required = True
        state.audit_log.append({"event": "uncited_claim_detected"})

    if has_stale_data(state.data_metrics):
        state.human_review_required = True
        state.audit_log.append({"event": "stale_data_detected"})

    if has_policy_escalation(state.risk_annotations):
        state.human_review_required = True
        state.audit_log.append({"event": "policy_escalation"})

    return state
```

Validation should also protect against overconfident outputs. For example, if the Research Analyst agent cannot find recent filings or transcripts, the memo should state that the evidence base is incomplete. If the Data Analyst agent cannot compute a metric, the metric should be omitted or marked unavailable, not guessed. If the Risk and Compliance agent receives ambiguous policy results, the workflow should escalate instead of forcing a binary pass/fail decision.

# Step 7: Turn the prototype into a governed workflow

The guided build should be extended in stages. First, run the agents with mocked tools and static fixtures. Next, connect read-only retrieval and data tools. Then add schema validation and regression tests for the most common user requests. Finally, introduce human approval gates and monitoring dashboards before the system is used in a live investment process.

A practical implementation checklist for this build is:

+   1. define JSON schemas for each intermediate output, such as research notes, data metrics, draft memos, and risk annotations;
+   2. create fixture-based tests for common requests and known edge cases;
+   3. log every tool call, retrieved document identifier, model version, prompt version, and human approval decision;
+   4. restrict early deployments to read-only tools and internal draft outputs; and