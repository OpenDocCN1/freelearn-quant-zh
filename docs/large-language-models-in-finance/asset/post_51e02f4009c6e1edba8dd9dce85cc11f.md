
## Custom HNSW retrieval tier

A custom HNSW tier stores 50 million 768-dimensional embeddings and applies entitlement, document-type, jurisdiction, and freshness filters. The custom implementation is justified only because the scenario requires co-location with the inference cluster, a recency-aware routing policy, and deterministic index manifests. A managed vector service would reduce engineering and on-call cost but would introduce network, data-location, and provider dependencies. The custom choice therefore carries an explicit maintenance budget, benchmark suite, backup procedure, and migration plan.

## Cloud role

Cloud capacity is used for development, non-sensitive batch experiments, and emergency slow-path overflow. It is not placed in the fast path because the latency objective and data-control requirements do not justify a synchronous wide-area dependency. Disaster recovery uses a warm, capacity-limited secondary environment rather than assuming that arbitrary cloud capacity will be instantly available during a market-wide disruption.

## 10.5.3 Illustrative benchmark design

The comparison uses a 30-trading-day timestamped replay and a synthetic burst generator. Both systems receive the same deduplicated events and reference data. Fast-path texts are capped at 512 tokens, concurrency reaches 256 requests per region, and the result tables report application-boundary latency after tokenization and enrichment. Strategy metrics are calculated on the same historical window with identical transaction-cost and risk assumptions. These conditions are deliberately stated because results without a common baseline, workload, and measurement boundary are not interpretable.

Table 10.2 makes the baseline visible and explains the measurement context. The cost result is specific to the assumed installed A100 capacity and cloud prices; it does not establish that on-premises infrastructure is generally 41% cheaper. Likewise, the strategy-level difference cannot be attributed to infrastructure alone. Faster event processing, different model errors, and changed signal coverage all contribute, and a statistically valid economic evaluation would require multiple periods, market regimes, and sensitivity analyses.

| Metric | Rule-based baseline | Hybrid design | Interpretation |
|---|---|---|---|
| Fast-path p50 latency | 18.4 ms | 7.3 ms | Same application boundary and regional burst profile |
| Fast-path p99 latency | 41.6 ms | 9.6 ms | Hybrid design meets the illustrative sub-10-ms p99 objective |
| Sustained throughput | 7,500 events/s | 12,000 events/s | After identical upstream filtering and deduplication |
| Macro-F1 | 0.88 | 0.95 | Frozen, time-separated classification set |
| Market-hours availability | 99.92% | 99.97% | Includes planned and unplanned service interruption |
| Annualized infrastructure-cost index | 1.00 | 0.59 | Includes amortized owned hardware and the cost categories in Equation (10.1) |
| Historical-replay Sharpe ratio | 1.04 | 1.28 | A 23% relative increase under identical risk and transaction-cost assumptions; not a production return claim |

Table 10.2: Illustrative before-and-after comparison under the declared replay and synthetic-load setup

## 10.6 Future directions and emerging technologies

The technologies in this section are speculative from the perspective of production financial LLM infrastructure. They illustrate plausible longer-term directions rather than near-term deployment recommendations.

### 10.6.1 Neuromorphic computing

Neuromorphic systems use event-driven, brain-inspired architectures and can be highly efficient for selected spiking workloads. Intel's Loihi 2 remains a research processor accompanied by the Lava software framework [Intel Labs, 2026]. These systems may eventually support always-on anomaly detection or low-power local sensing, but dense Transformer inference is not presently their natural workload. Financial adoption therefore depends on new algorithms, mature tooling, reproducible benchmarks, and integration with conventional control environments.

### 10.6.2 Quantum–classical hybrid systems

Quantum optimizers may eventually accelerate narrowly defined subroutines such as combinatorial portfolio construction or scenario search whose outputs feed a conventional decision-support system. Present hardware remains noisy and resource constrained, and any advantage must include data loading, error mitigation, orchestration, and comparison with strong classical solvers. Quantum components should therefore be treated as experimental services with classical fallback, not as core production dependencies.

### 10.6.3 Edge AI acceleration

Edge accelerators such as NVIDIA Jetson AGX Orin and Qualcomm Cloud AI 100-class devices can execute compact models close to local data sources [NVIDIA, 2026c, Qualcomm Technologies, 2026]. Appropriate financial uses include document triage, private local assistants, speech or image preprocessing, and confidence-based routing. Edge deployment does not remove the need for central governance: model signing, secure update, telemetry, rollback, entitlement, and incident response become more difficult as the number of sites grows.

## 10.7 Summary

This chapter examined the infrastructure and performance considerations involved in deploying financial LLMs. It compared GPUs, TPUs, and specialized accelerators while emphasizing that vendor peak specifications are not substitutes for workload-specific benchmarks. It also corrected common procurement errors: TPU INT8 and bfloat16 rates should not be conflated, cloud elasticity should not be assumed to produce a fixed percentage saving, and power or accelerator price alone does not represent total cost of ownership.

The chapter then covered cloud, on-premises, hybrid, and edge deployment. The central design principle is to place each component according to its latency, data-control, resilience, and governance requirements. Regulatory and data-location obligations call for risk-based controls, auditability, and exit planning rather than a universal infrastructure pattern.

Data architecture was treated as part of the model system. Kafka and Flink can support durable and stateful processing, but end-to-end exactly-once behavior depends on the complete source-processing-sink design. Embedding-runtime and market-volume examples were restated with explicit assumptions, and sub-millisecond retrieval claims were separated from end-to-end RAG latency.

The deployment discussion covered quantization, pruning, distillation, TensorRT-LLM, vLLM, orchestration, and multi-objective performance tuning. Optimization results are model- and workload-specific and should be accepted only after validation of tail latency, task quality, calibration, robustness, cost, and failure behavior.

Finally, the illustrative trading-support case study provided the requested architecture rationale, explicit cost constraint, declared benchmark setup, and before-and-after baseline. Its numerical values are transparent planning assumptions rather than measured production outcomes. Overall, production financial LLM infrastructure depends on secure and replayable data pipelines, reliable serving, controlled optimization, tail-latency management, cost transparency, auditability, and governance that extends to the complete hardware-software configuration.

## 10.8 Open questions for production financial LLM infrastructure

Several important questions remain open. First, how should institutions compare infrastructure cost when accelerator utilization, human-review effort, model quality, and regulatory control are jointly endogenous? Cost per token is easy to measure but can be economically misleading when two systems produce different error and escalation rates.

Second, how should tail latency be controlled under correlated bursts, failover, and multi-tenant contention? Queueing policies that maximize average throughput may be inappropriate when a small number of very late responses creates material operational risk. Better workload models are needed for market-open bursts, major announcements, and simultaneous recovery events.

Third, where should the boundary between cloud, sovereign cloud, on-premises, and edge execution be placed? The answer depends not only on data residency but also on audit rights, cryptographic key control, provider concentration, portability, and tested exit time. Hybrid deployment can reduce one risk while multiplying configuration and observability risk.

Fourth, how can auditability be preserved through highly optimized inference stacks? Quantized engines, speculative decoding, caching, dynamic batching, and asynchronous retrieval complicate exact reproduction. Production systems need practical evidence standards that identify which elements must be replayable bit-for-bit and which can be reproduced only within a tolerance.

Fifth, how far can inference optimization proceed without damaging calibration, abstention, numerical fidelity, or rare-event behavior? Aggregate benchmark accuracy is insufficient for financial use; optimization methods need slice-aware acceptance tests and explicit rollback triggers.

Finally, how should governance operate when infrastructure itself changes model behavior? Driver, kernel, compiler, batching, retrieval-index, and hardware changes can alter outputs even when model weights are unchanged. Configuration changes therefore require model-risk controls proportional to their potential behavioral impact.
