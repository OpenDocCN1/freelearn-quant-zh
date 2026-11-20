As workflows become more complex, this shift from a single agent to multiple agents becomes necessary rather than optional. Single-agent architectures can become brittle or difficult to extend, particularly when they are expected to handle multiple distinct responsibilities. Decomposing the workflow into specialized agents allows each component to remain focused, with its own role, tools, and prompts.

Multi-agent architectures can be organized in several ways:

+   - **Hierarchical**: A central coordinator agent manages a set of specialist agents, delegating tasks and aggregating results.
+   - **Peer-to-peer**: Agents communicate directly with each other, possibly following a shared protocol or negotiation mechanism.
+   - **Hub-and-spoke**: A central hub handles communication and routing, while agents act as spokes that perform specific functions.

Hierarchical architectures are particularly common, reflecting organizational structures. For example, a research assistant may have a coordinator agent that interacts with specialist agents for company analysis, macro analysis, and risk analysis. Each specialist agent may, in turn, interact with additional tools, such as retrieval systems over domain-specific corpora, time-series models, or valuation libraries. This decomposition mirrors how work is structured in practice, making the system easier to reason about, extend, and govern. The research workstation follows this hierarchical pattern: the portfolio manager interacts with one interface, the orchestrator decomposes the request, and the specialist agents return bounded outputs that can be inspected before they are merged into the memo.

![](bbox=[0.13, 0.53, 0.87, 0.72])

Figure 5.1: The agent contract

The model proposes; it never disposes. Everything inside the left group is stochastic and may be wrong; everything inside the right group is deterministic and is where the institution's guarantees actually live. The admissibility projection Π_t is a shield in the sense of Alshiekh et al. [2018]: it is evaluated outside the model, it cannot be argued with by the model, and a proposed action that fails it is rejected or escalated rather than repaired. Note that no arrow bypasses Π_t — that absence is the entire design.