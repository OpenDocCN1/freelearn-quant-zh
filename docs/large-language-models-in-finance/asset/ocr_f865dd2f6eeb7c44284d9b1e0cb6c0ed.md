**4 Audited deployment evidence**
Logged decisions, CIs, failure cases, human overrides, monitoring.

**3 Benchmark experiment**
Dataset, temporal split, model + prompt version, metric suite.

**2 Reproducible smoke test**
Fixed inputs, expected outputs. Catches broken code, not value.

**1 Illustrative example**
Explains a method. **Never** evidence of performance.

*Most published financial-LLM results are level 1 or 2, reported in the language of level 3 or 4.*

Figure 12.3: The evidence ladder.

Each rung licenses a strictly stronger class of claim, and nothing licenses a claim above its rung. The failure this book is written against is not bad arithmetic — it is level-1 evidence described in level-4 language. Every numeric table in this book declares its rung in the caption; if a table you meet elsewhere does not, assume rung one until shown otherwise.

# ## 12.8 Summary

This chapter reviewed how large language models are being specialized, assembled, and evaluated for financial applications. The first lesson is that domain specialization matters. BloombergGPT, FinBERT, FinEAS, FinGPT, and related systems show that financial corpora, financial terminology, and financial task definitions change the behavior of otherwise familiar transformer architectures. The relevant design choice is not only which base model to use, but also how to allocate data, adaptation budget, retrieval infrastructure, and evaluation effort.

The second lesson is that financial LLMs increasingly appear as components of larger systems rather than as isolated chat models. Memory-augmented trading agents, debate-based multi-agent systems, portfolio-level agents, and LLM-assisted risk workflows all reflect the same architectural shift: financial work is sequential, evidence-driven, role-based, and governed. This makes explicit memory, tool access, audit logs, and human review as important as raw language-model accuracy.

The third lesson is that empirical claims must be interpreted carefully. Many reported results are promising, but they are often early, narrow, or sensitive to backtest design. Multi-task benchmarks, numerical reasoning datasets, case-triage workflows, and cost-aware simulations