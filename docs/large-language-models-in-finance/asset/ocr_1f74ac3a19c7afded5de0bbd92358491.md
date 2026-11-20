# Long-term memory and belief state

Autonomous workflows require memory beyond a single prompt. Semantic memory stores retrievable representations of reports, policies, and prior cases; structured memory records positions, limits, approvals, and transaction state; episodic memory preserves previous decisions and outcomes. Retrieval-augmented generation provides one mechanism for grounding the reasoning engine in external evidence [Lewis et al., 2020].

```
Algorithm 19 Auditable memory update protocol
Input: New information i_new, belief state B_t, provenance metadata q_t
Output: Updated belief state B_{t+1} and audit record a_t
Validate source, timestamp, permissions, and document version
Encode i_new as representation v_new
Retrieve candidate memories M_rel <- kNN(v_new, M)
Compute consistency and freshness scores for i_new relative to M_rel
if source validation fails then
    Quarantine i_new and preserve the failure reason
else if material conflict is detected then
    Escalate for reconciliation; do not silently overwrite B_t
else
    B_{t+1} <- integrate(B_t, i_new, q_t)
end if
Append provenance, retrieval links, and update decision to a_t
return B_{t+1}, a_t
```

# Planning, tools, reflection, and governance

The planning module converts a goal into a sequence of verifiable subgoals. The tool layer exposes typed operations—for example, retrieving a filing, running a pricing library, querying a position database, or submitting a paper-trading order. Typed schemas matter because free-form text is not a safe transaction protocol. The reflection module compares expected and realized outcomes, records failures, and proposes improvements. Finally, the governance layer determines whether a proposed action is permitted, requires human approval, or must be rejected.

Figure 14.1 can be read from top to bottom. Market and institutional inputs enter the memory, planning, and reasoning layers with provenance attached. Memory and planning interact iteratively with the reasoning engine to produce a structured proposal. The policy and risk gate then checks mandates, limits, permissions, and approval requirements before any tool can act. Approved tool calls reach only a controlled execution environment. Execution outcomes flow back to evaluation and memory, creating a learning loop without allowing the language model to bypass the control boundary.

# 14.2.2 Formal theory of governed financial agency

The architecture in Figure 14.1 is useful only if its parts are tied to a precise theory of action. A financial agent is not simply a conversational interface with access to tools. It is a delegated