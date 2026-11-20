
As workflows become more complex, this shift from a single agent to multiple agents becomes necessary rather than optional. Single-agent architectures can become brittle or difficult to extend, particularly when they are expected to handle multiple distinct responsibilities. Decomposing the workflow into specialized agents allows each component to remain focused, with its own role, tools, and prompts.

Multi-agent architectures can be organized in several ways:

- **Hierarchical**: A central coordinator agent manages a set of specialist agents, delegating tasks and aggregating results.
- **Peer-to-peer**: Agents communicate directly with each other, possibly following a shared protocol or negotiation mechanism.
- **Hub-and-spoke**: A central hub handles communication and routing, while agents act as spokes that perform specific functions.

Hierarchical architectures are particularly common, reflecting organizational structures. For example, a research assistant may have a coordinator agent that interacts with specialist agents for company analysis, macro analysis, and risk analysis. Each specialist agent may, in turn, interact with additional tools, such as retrieval systems over domain-specific corpora, time-series models, or valuation libraries. This decomposition mirrors how work is structured in practice, making the system easier to reason about, extend, and govern. The research workstation follows this hierarchical pattern: the portfolio manager interacts with one interface, the orchestrator decomposes the request, and the specialist agents return bounded outputs that can be inspected before they are merged into the memo.

![](img/c31705b361fdf81dee8826a2c4ee167f_210_0.png)

Figure 5.1: The agent contract

The model proposes; it never disposes. Everything inside the left group is stochastic and may be wrong; everything inside the right group is deterministic and is where the institution's guarantees actually live. The admissibility projection Π_t is a shield in the sense of Alshiekh et al. [2018]: it is evaluated outside the model, it cannot be argued with by the model, and a proposed action that fails it is rejected or escalated rather than repaired. Note that no arrow bypasses Π_t — that absence is the entire design.

## 5.1.4 Agent roles and specialization

The design of agent roles is a key architectural decision. Inappropriate or overly broad roles can lead to tangled prompts, difficult-to-debug behavior, and unclear accountability. A useful heuristic is to align agent roles with human roles and responsibilities in the institution, such as equity research analyst, macro strategist, risk officer, compliance officer, or operations analyst.

Each agent’s prompt can then be framed as an instruction to a human in that role, using explicit guidelines and constraints. For example, a compliance agent’s prompt should emphasize conservatism, adherence to specific regulations, and the need to flag ambiguous cases for human review. This makes the agent’s behavior easier to reason about, especially when reviewing outputs in a regulated setting.

In the research workstation, role design also determines accountability. The Research Analyst agent is responsible for evidence-backed qualitative synthesis; the Data Analyst agent is responsible for metrics, data freshness, and numerical consistency; the Risk and Compliance agent is responsible for policy checks and escalation flags; and the Orchestrator agent is responsible for routing, reconciliation, and final packaging. Keeping these boundaries explicit prevents the final memo from becoming a black box in which it is unclear whether a claim came from research evidence, market data, or a policy check.

## 5.1.5 Memory and state management

Memory is a central component of LLM agent systems. Agents may need to maintain:

- **Short-term memory**: such as the current conversation context and recent tool outputs;
- **Medium-term memory**: such as the history of a workflow (e.g., previous steps in a research process);
- **Long-term memory**: such as user preferences, past decisions, or accumulated knowledge that spans sessions.

Naively storing all past interactions can lead to context overload and performance degradation. Instead, it is often beneficial to use structured memory mechanisms, including separate memory stores for different types of information (for example, conversation logs versus decisions or documentation), retrieval-based memory that selects relevant past items, and summarized memory, where long histories are periodically compressed into shorter representations. These are not new inventions: Packer et al. [2023] treat the context window as a paged resource and move material between tiers under the model’s own control, and Park et al. [2023] combine a memory stream with retrieval and periodic reflection to keep long histories tractable. What finance adds is not a new mechanism but a new obligation. In a general agent, a stale memory produces an odd answer; in a financial agent, a stale memory produces an _apparently sourced_ answer, because the summary that carried it forward has lost the timestamp that would have condemned it.

A memory policy should specify what is retained, summarized, invalidated, and refreshed. Durable facts—such as approved policy thresholds, security identifiers, portfolio identifiers, and signed-off decisions—should be distinguished from provisional workflow artifacts, such as draft notes, intermediate model outputs, stale data snapshots, and unresolved hypotheses. Summaries should preserve links to the underlying records; state slices should be refreshed when data sources, policy rules, or portfolio exposures change; and outdated or superseded memories should be invalidated rather than silently reused. Without this discipline, long-term memory can turn a previous draft, stale metric, or informal user preference into an apparent fact.

These design choices become more important as systems move from single-agent to multi-agent setups, where different agents may need access to different slices of state. Memory design must also accommodate regulatory requirements for record-keeping and auditability. For instance, records of advice given to clients, or decisions made by agents on behalf of traders, may need to be searchable and immutable.

For the research workstation, a practical memory design separates at least five objects: the user request, retrieved evidence, structured market data, intermediate agent outputs, and the final approved memo. The Research Analyst agent needs access to retrieved evidence and citation metadata; the Data Analyst agent needs tickers, horizons, and data snapshots; the Risk and Compliance agent needs the draft conclusion, portfolio context, and applicable policy rules; and the Orchestrator needs a workflow log that records which agent produced which intermediate result. This separation improves both security and auditability because agents receive the minimum context required for their task, while the system still retains a complete record for later review.

## Illustrating the agent flow

Figure 5.1 depicts the interaction pattern for the research workstation example.

Everything to the left of the boundary is stochastic: the orchestrator and its sub-agents may propose any action at all. Nothing crosses to the right without passing four deterministic checks, none of which is a prompt and none of which the model can argue with. Note the return path: a tool result comes back as data, never as instruction. A system in which a tool response can rewrite the model’s operating instructions has moved the boundary to the far side of the tool, which is exactly where the adversary is

analyst request
orchestrator (plans, decomposes)
retrieval sub-agent
quant sub-agent
drafting sub-agent
deterministic gates
schema · entitlements
limits · recomputation
filings index
market data
portfolio-manager review
immutable evidence log
pricing engine
trust boundary
reject
results return as data
evidence model control human

Figure 5.2: Interaction pattern for the research workstation orchestration agent.

## Domain glossary for the running workflow

- **Concentration constraints** are limits on exposure to a single issuer, sector, country, factor, strategy, or group of highly correlated assets.
- **Restricted lists** identify securities, issuers, or counterparties that cannot be traded, recommended, or discussed in certain contexts because of legal, compliance, conflict-of-interest, or information-barrier restrictions.
- **Risk limits** are pre-approved quantitative boundaries, such as maximum position size, risk factor exposure, tracking error, Value-at-Risk, leverage, drawdown, or stress-loss thresholds.
- **Communication policies** define which outputs may be internal or external, what disclosures are required, who may receive them, and which approvals are needed before distribution.
- **Portfolio identifiers** point the workflow to the relevant holdings, mandate, benchmark, constraints, and approval path for the analysis.
