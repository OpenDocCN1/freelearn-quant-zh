| Stage | Responsible component | Primary input | Inspectable output |
| :--- | :--- | :--- | :--- |
| Request parsing | Orchestrator agent | User request, portfolio identifier, investment horizon | Structured task plan, tickers, data needs, and review gates |
| Evidence retrieval | Research Analyst agent | Task plan, ticker, source policy | Cited notes from filings, transcripts, research notes, and news summaries |
| Metric computation | Data Analyst agent | Ticker, horizon, market-data snapshot | Valuation, growth, profitability, leverage, and sensitivity metrics |
| Memo drafting | Orchestrator agent | Research notes and metrics | Draft thesis with assumptions, evidence, uncertainties, and missing-data warnings |
| Risk and policy review | Risk and Compliance agent | Draft memo, portfolio context, policy rules | Risk annotations, restricted-list checks, disclosure requirements, escalation flags |
| Human review | Analyst, portfolio manager, risk or compliance reviewer | Complete workflow packet | Approved memo, edited memo, rejection reason, or request for further analysis |

Table 5.1: Observable stages in the ACME Corp pre-investment memo workflow

This table is intentionally operational. It shows where the system should create audit records and where a human should be able to interrupt the flow. If the Research Analyst agent cannot find a recent earnings transcript, the workflow should not continue as if the evidence base were complete. It should mark the transcript as missing, identify which claims depend on it, and either ask the user for permission to continue with limitations or route the task back to a human analyst. If the Data Analyst agent finds that the latest financial snapshot is stale, the valuation section of the memo should be labeled as provisional rather than merged into the final thesis without qualification.

The same example also shows why the workstation should expose disagreements rather than hide them. The Research Analyst agent may conclude that management commentary supports an improving margin story, while the Data Analyst agent may show that operating cash conversion has deteriorated over the last four quarters. A weak orchestrator will average those observations into a vague statement. A stronger orchestrator will preserve the tension: management commentary points to margin improvement, but recent cash conversion does not yet confirm the claim. This is the kind of synthesis that makes the system useful to a portfolio manager while preserving analytical discipline.

# 5.2 Failure modes and control requirements

While LLM-based agents offer significant potential, especially in domains such as finance where workflows are knowledge-intensive, they also introduce new challenges and failure