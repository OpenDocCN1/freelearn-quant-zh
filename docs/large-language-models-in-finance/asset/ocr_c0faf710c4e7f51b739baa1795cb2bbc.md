# ## 12.5.1 DeFi, on-chain analytics, and protocol governance

DeFi-focused systems such as DeFiScope integrate LLMs with:

+   - Parsers for smart contracts and protocol documentation;
+   - Indexers for on-chain data and transaction graphs; and
+   - RAG pipelines for governance proposals and forum discussions.

A stylized architecture is the one that can:

+   1. Parse and normalize contract code into an intermediate representation;
+   2. Embed documentation, proposals and forum posts into a vector store;
+   3. Use an LLM to answer questions or summarize proposals, grounded in retrieved evidence and contract semantics.

These systems can assist in summarizing complex proposals, flagging suspicious patterns, and explaining protocol changes to non-expert users. They also operate in adversarial environments, where robustness to prompt injection, data poisoning and adversarial contracts is critical.

# ## 12.5.2 Quantum-enhanced optimization

Quantum-inspired and quantum-assisted optimization methods are being explored for portfolio and risk problems. In such architectures, an LLM may:

+   - Select the optimization formulation based on natural-language descriptions of constraints and objectives;
+   - Generate or refine parameterizations for classical or quantum optimizers; and
+   - Explain the resulting allocations or hedges to human decision makers.

For instance, a quantum approximate optimization algorithm (QAOA) for portfolio selection may seek to minimize a cost function with:

```
C(w) = -\mu^\top w + \frac{\lambda}{2}w^\top\Sigma w,
```

subject to constraints, with \(\mu\) expected returns and \(\Sigma\) covariance. The LLM can map informal risk preferences and constraints into choices of \(\lambda\), feasible sets, or custom penalty terms. At present, hardware and integration constraints limit these approaches to research and small-scale pilots.

# ## 12.6 Comparative evidence matrix

A literature review becomes misleading when architectures, demonstrations, benchmark scores, and economically realistic evidence are placed on one undifferentiated leaderboard. Table 12.6 therefore records not only what each representative system does, but also what