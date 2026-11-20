the same test period, the final result is overfit even if every individual run was temporally clean. This is prompt overfitting, the LLM version of strategy data mining.

The remedy is research accounting. Maintain a development set, a validation set, and a final untouched test set. Log all prompt variants and model-selection decisions. Use nested validation when selecting prompts or adapters. Report the number of trials or use multiple-testing corrections where appropriate. Most importantly, do not let the final test set become part of the creative loop.

# 13.4.10 Hallucinated numerical signals and false precision

LLMs can generate fluent explanations around weak or nonexistent signals. A model may infer a causal story from noise, invent a ratio, misread a table, or report a precise expected return unsupported by the data. In financial forecasting, false precision is dangerous because small numerical differences can drive large position changes.

Production systems should separate language generation from numerical computation. Let the LLM propose hypotheses, retrieve context, or explain model outputs, but let audited numerical code compute returns, volatilities, correlations, drawdowns, and portfolio weights. A forecast report should include confidence intervals, calibration diagnostics, and explicit uncertainty. If the model cannot cite the data and computation behind a number, the number should not enter the investment process.

# 13.4.11 A leakage-aware evaluation protocol

A robust protocol for LLM financial forecasting and backtesting should include the following controls. First, define the information set $\mathcal{F}_t$ for every forecast origin, including data, documents, tools, prompts, and memory. Second, build all features from availability timestamps rather than economic-period timestamps. Third, use rolling-origin splits with embargoes for overlapping labels. Fourth, freeze or time-index all retrieval stores and tool APIs. Fifth, reset memory unless online learning is explicitly modeled. Sixth, evaluate against simple baselines and strong non-LLM baselines. Seventh, report statistical and economic metrics after transaction costs. Eighth, run contamination probes for pretrained and retrieved knowledge. Ninth, reserve a final untouched test period. Tenth, archive prompts, code, data hashes, model versions, and tool outputs so the backtest can be audited.

Algorithm 18 summarizes the protocol.

# Practitioner implementation note

The protocol can be enforced with ordinary engineering components if each component is time-indexed. A practical stack contains five layers. First, a point-in-time feature store, for example a Feast-style or in-house feature store, performs as-of joins and rejects features whose availability timestamp is later than the forecast origin. Second, an append-only lakehouse or immutable data-versioning layer, for example a Delta Lake, Apache Iceberg, lakeFS, or hash-chain design, stores raw vendor drops and corrected versions without overwriting history. Third, a time-indexed retrieval store rebuilds the vector index so that retrieved documents are measurable with respect to $\mathcal{F}_{t_j}$. Fourth, a model and prompt