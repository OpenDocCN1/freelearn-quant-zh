loss, index failover, network degradation, and regional recovery rather than only during healthy steady state.

Cost should be monitored as an operational metric. Useful measures include cost per thousand classified documents, cost per million input tokens, cost per generated token, cost per successful task, cloud egress, idle accelerator hours, reserved-capacity utilization, engineering on-call load, and disaster-recovery overhead. Unit costs should be reported together with quality and latency because an apparently cheaper configuration may create more human-review work.

# Model and retrieval performance metrics

Production model metrics should include task accuracy, precision and recall for material error classes, calibration, abstention rate, robustness, and performance across document types, products, jurisdictions, and market regimes. Numerical extraction deserves separate validation because a model can be linguistically plausible while altering a sign, decimal, currency, or unit.

RAG monitoring should track retrieval recall on judged queries, source freshness, entitlement violations, duplicate evidence, citation coverage, no-answer behavior, and the frequency with which generated statements are unsupported by retrieved material. Drift detection can compare input, embedding, retrieval, confidence, and output distributions, but a statistical alert should trigger investigation rather than automatically imply retraining.

# 10.4.2 Continuous optimization processes

Continuous optimization keeps a deployed system aligned with changing workloads, infrastructure conditions, and business requirements. In financial settings, these processes must be controlled carefully because optimization choices can affect latency, cost, model quality, compliance, and user-facing decisions.

# Controlled experimentation

A/B testing can support deployment decisions, but financial experiments require controls for market conditions, temporal dependence, user heterogeneity, and information leakage. Randomization units should be chosen so that treatment and control do not contaminate one another. Where live experimentation is inappropriate, shadow deployment and replay on timestamped data can compare latency, errors, and decisions without affecting users or trades.

Multi-armed bandits can allocate traffic adaptively, but their changing assignment probabilities complicate inference and governance. For high-impact applications, a pre-specified experiment with clear stopping rules may be preferable. Any strategy-level comparison should include transaction costs, turnover, capacity, and a common market window; infrastructure improvements should not be credited for economic gains that arise from a changed signal or risk policy.