# 10.8 Open questions for production financial LLM infrastructure

Several important questions remain open. First, how should institutions compare infrastructure cost when accelerator utilization, human-review effort, model quality, and regulatory control are jointly endogenous? Cost per token is easy to measure but can be economically misleading when two systems produce different error and escalation rates.

Second, how should tail latency be controlled under correlated bursts, failover, and multi-tenant contention? Queueing policies that maximize average throughput may be inappropriate when a small number of very late responses creates material operational risk. Better workload models are needed for market-open bursts, major announcements, and simultaneous recovery events.

Third, where should the boundary between cloud, sovereign cloud, on-premises, and edge execution be placed? The answer depends not only on data residency but also on audit rights, cryptographic key control, provider concentration, portability, and tested exit time. Hybrid deployment can reduce one risk while multiplying configuration and observability risk.

Fourth, how can auditability be preserved through highly optimized inference stacks? Quantized engines, speculative decoding, caching, dynamic batching, and asynchronous retrieval complicate exact reproduction. Production systems need practical evidence standards that identify which elements must be replayable bit-for-bit and which can be reproduced only within a tolerance.

Fifth, how far can inference optimization proceed without damaging calibration, abstention, numerical fidelity, or rare-event behavior? Aggregate benchmark accuracy is insufficient for financial use; optimization methods need slice-aware acceptance tests and explicit rollback triggers.

Finally, how should governance operate when infrastructure itself changes model behavior? Driver, kernel, compiler, batching, retrieval-index, and hardware changes can alter outputs even when model weights are unchanged. Configuration changes therefore require model-risk controls proportional to their potential behavioral impact.