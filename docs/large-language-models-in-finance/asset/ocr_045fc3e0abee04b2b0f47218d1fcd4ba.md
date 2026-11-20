# ## 15.5 Financial data resources

Financial LLM development depends on reliable datasets for training, evaluation, retrieval, and benchmarking. This section summarizes public financial NLP datasets, market data sources, and simple access patterns for market data and SEC filings.

# ### 15.5.1 Public financial datasets

Financial PhraseBank should be reported together with the selected annotator-agreement split [Malo et al., 2014, Hugging Face, 2026]. FiQA refers to a family of financial opinion mining and question-answering benchmarks, so the exact version and split should always be stated [Maia et al., 2018]. FinEAS is a model and code resource for financial sentiment and sentence embeddings rather than a raw benchmark dataset [Gutiérrez-Fandiño et al., 2021].

# ### 15.5.2 Market data sources

Market data sources differ in coverage, latency, licensing, survivorship-bias controls, corporate-action handling, historical backfill quality, and commercial-use restrictions. Open or unofficial tools are useful for teaching, experimentation, and prototyping, but regulated production systems usually require licensed institutional feeds, service-level agreements, data-quality controls, and legal review.

SEC EDGAR is a filing-access system rather than a single curated NLP dataset [U.S. Securities and Exchange Commission, 2026]; tools such as yfinance are convenient for prototyping but should not be treated as institutional feeds without checking licensing, reliability, and data quality [Aroussi, 2026].

+   * **Yahoo Finance via yfinance:** Convenient access to historical OHLCV data for research and prototyping. It should not be treated as an institutional production feed without checking licensing, reliability, data quality, and Yahoo's terms of use.
+   * **Alpha Vantage:** Time-series and fundamental data through an API-key-based service.
+   * **Quandl/Nasdaq Data Link:** Macro, financial, and alternative datasets, with dataset-specific licensing.
+   * **IEX Cloud or similar providers:** Real-time and historical market data services subject to provider terms and exchange-data rules.
+   * **SEC EDGAR:** Public access to regulatory filings. It is a filing repository, not a ready-made NLP dataset, and requires parsing, cleaning, metadata extraction, and task-specific labeling before modeling.