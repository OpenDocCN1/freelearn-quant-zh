
- Communication review agents that check communications (e-mails, research reports, client presentations) for policy violations.

In these settings, agentic systems must be designed with conservative defaults, strict guardrails, and strong human oversight. Agents should not make final regulatory interpretations or risk decisions; instead, they should support risk officers and compliance professionals by retrieving relevant information, highlighting potential issues, and suggesting questions for further investigation. In the research workstation, this is the function of the Risk and Compliance agent: it annotates the draft memo with concentration risks, restricted-list conflicts, disclosure requirements, and escalation flags, but it does not replace the judgment of the risk officer or compliance reviewer.

## 5.3.3 Operations and client service

Operational and client service workflows often involve repetitive tasks, structured data, and standardized communications, making them amenable to partial automation. LLM agents can assist with:

- processing client inquiries, routing them to the appropriate teams, and providing draft responses;
- reconciling trade data, positions, or cash flows;
- generating summaries of complex cases (e.g., investigations or disputes);
- coordinating between different operational teams or systems.

Here too, agentic systems must be designed with careful escalation rules and monitoring to ensure that errors are detected and corrected promptly, and that clients receive accurate, compliant communications. Although the research workstation is framed around investment research, the same architecture can be adapted to operations: a coordinator routes cases, specialist agents inspect transaction data or client records, and a human operator approves any external communication or irreversible action.

## 5.3.4 From application idea to implementation requirements

Before building any of these systems, teams can translate the application into a small set of implementation requirements. The following questions should be read as proposed design criteria rather than established best practice across all financial institutions. For the research workstation, such criteria are not merely technical; they also define what the system is and is not allowed to do. A useful requirements document can answer the following questions:

- 1. What are the accepted inputs? The system may accept natural-language requests from portfolio managers, but it should also require structured fields such as tickers, portfolio identifiers, analysis horizon, and intended use. A request for an internal pre-investment memo is materially different from a request for client-facing commentary. The same agent output may be acceptable in one context and inappropriate in another.
- 2. What are the required outputs? A useful output is a packet, not only prose. In the workstation example, the packet contains the draft memo, evidence identifiers, computed metrics, risk flags, unresolved questions, and an audit log. Requiring this structure changes agent behavior: agents are less likely to hide uncertainty because the output format gives uncertainty a defined place to appear.
- 3. **What are the permissions?** In a conservative pilot, early versions should be read-only. The workstation may retrieve documents, query market data, and check portfolio constraints, but it should not submit trades, modify official records, or send communications outside the firm. If later versions add write actions, those actions should be isolated behind separate tools, stronger approvals, and more stringent logging.
- 4. **Where are the review gates?** A conservative design attaches human review to the workflow, rather than adding it as an afterthought. The workstation should have at least three review gates: evidence review by an analyst, investment-thesis review by a portfolio manager, and risk or compliance review for flagged outputs. The system should record who reviewed what, what changes were made, and which version of the memo was approved.
- 5. **How will success be measured?** A useful pilot can measure more than user satisfaction. It should track citation coverage, numerical accuracy, stale-data detection, frequency of escalations, review turnaround time, and the number of unsupported claims caught before approval. These metrics convert a subjective demo into an operationally measurable workflow.

These proposed criteria prepare the ground for the guided build below. The code sketch is not intended to be a production implementation; rather, it shows how to translate roles, tools, state, and review gates into an inspectable workflow that can later be implemented with a concrete agent framework.

## 5.3.5 Guided build: Minimal multi-agent research workstation

The following guided build turns the running workstation example into a minimal, end-to-end design. The code remains framework-neutral pseudocode, but it is structured enough that a reader could translate it into Python with an LLM API, a retrieval layer, and a few mocked data tools. A full implementation would be a natural candidate for a companion repository: the chapter can explain the architecture, while the repository can contain runnable notebooks, fixtures, tests, and examples of tool schemas.

The pseudocode intentionally omits several production concerns in order to keep the architecture readable. In a deployed system, the same workflow would need explicit versioning for prompts, models, tools, schemas, retrieval indexes, and policy rules; concurrency controls for parallel tool calls and simultaneous reviewers; error handling for timeouts, partial results, permission failures, and malformed outputs; source hashing for retrieved documents and data snapshots; and evidence linkage from each material memo claim back to the documents, metrics, and policy checks that support it. The code below therefore illustrates the control flow, not a complete production implementation.

## Step 1: Define the workflow state

A useful starting point is to avoid passing unstructured strings between agents. Instead, the orchestrator maintains a workflow state object that records the request, the relevant portfolio context, intermediate outputs, and audit metadata.

```python
class WorkflowState:
    def __init__(self, user_query, tickers, horizon, portfolio_id):
        self.user_query = user_query
        self.tickers = tickers
        self.horizon = horizon
        self.portfolio_id = portfolio_id

        # Intermediate outputs
        self.research_notes = None
        self.data_metrics = None
        self.draft_memo = None
        self.risk_annotations = None
        self.final_memo = None

        # Governance and observability
        self.audit_log = []
        self.human_review_required = False
```

This object enforces a simple but important discipline: every agent reads from and writes to a known part of the workflow state. The Research Analyst agent should not silently modify risk flags; the Risk and Compliance agent should not overwrite retrieved evidence; and the Orchestrator should be able to reconstruct the sequence of events.

Step 2: Wrap external systems as explicit tools. Tools should be deterministic, typed, and narrowly scoped. Early prototypes should favor read-only tools, especially in finance, because they allow teams to validate behavior without allowing the agent to alter portfolios, submit orders, or change official records.

```python
class ResearchTools:
    def retrieve_documents(self, query, tickers):
        return search_internal_corpus(
            query=query,
            tickers=tickers,
            sources=["filings", "transcripts", "research_notes"]
        )

class MarketDataTools:
    def fetch_snapshot(self, tickers, horizon):
        return get_prices_and_fundamentals(
            tickers=tickers,
            horizon=horizon
        )

class PolicyTools:
```
