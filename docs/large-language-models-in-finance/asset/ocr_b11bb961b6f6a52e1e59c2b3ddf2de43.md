```
def check_limits_and_policies(self, memo, portfolio_id):
    return evaluate_against_policy_store(
        memo=memo,
        portfolio_id=portfolio_id,
        checks=[
            "restricted_list",
            "concentration",
            "disclosures"
        ]
    )
```

In production, each tool would also include authentication, authorization, schema validation, rate limits, and logging. The key design point is that the LLM does not access systems directly; it requests tool calls through controlled interfaces.

**Step 3: Build specialist agents around bounded responsibilities** Each specialist agent receives the shared state, performs one bounded task, writes its output back to the state, and appends an audit event. The prompts used by these agents should require structured outputs, citations where applicable, and explicit uncertainty labels.

```
class ResearchAnalystAgent:
    def __init__(self, self, llm, tools):
        self.llm = llm
        self.tools = tools

    def run(self, state):
        docs = self.tools.retrieve_documents(
            query=state.user_query,
            tickers=state.tickers
        )
        state.research_notes = self.llm.complete_json(
            role="senior equity research analyst",
            task=("Summarize evidence, compare sources, "
                  "and cite documents."),
            inputs={
                "query": state.user_query,
                "documents": docs
            },
            schema="ResearchNotes"
        )
        state.audit_log.append({
            "agent": "research",
            "documents": doc_ids(docs)
        })
        return state
```

```
class DataAnalystAgent:
    def __init__(self, self, llm, tools):
        self.llm = llm
```