# 15.4.2 Model monitoring and retraining

Fine-tuning is only the first step. Once a financial model is deployed, its performance must be monitored because market conditions, terminology, document formats, user behavior, and data distributions can change over time.

Production models require ongoing monitoring for data drift, concept drift, performance degradation, latency, operational failures, and changes in the business or regulatory environment. Useful monitoring dimensions include:

+   - **Validation performance:** Track accuracy, macro-F1, class-specific precision and recall, calibration error, and confusion matrices on fresh validation samples.
+   - **Prediction-confidence drift:** Monitor changes in the distribution of predicted probabilities, entropy, margins, and abstention rates.
+   - **Input-distribution drift:** Track changes in token distributions, embedding distributions, document length, language mix, entity frequency, and financial terminology.
+   - **Operational metrics:** Monitor latency, throughput, error rates, timeout rates, failed API calls, and memory or GPU utilization.
+   - **Segmented performance:** Evaluate performance by document type, market regime, geography, asset class, counterparty type, business line, and data vendor.
+   - **Human-review metrics:** Track override rates, escalation rates, reviewer disagreement, and recurring error categories.

For FinEAS-style sentiment and sentence-embedding models, additional monitoring is useful:

+   - **Embedding drift:** Monitor the distance between current sentence embeddings and a reference distribution from the validation period.
+   - **Sentiment-score stability:** Track the mean, variance, and tail behavior of positive, neutral, and negative scores by source and regime.
+   - **Entity-level consistency:** Check whether repeated references to the same issuer, sector, or event type receive stable sentiment labels when the surrounding text is comparable.
+   - **Market-regime breakdown:** Report performance separately in calm, stressed, high-volatility, and crisis periods.

Retraining strategies should be tied to these metrics rather than to calendar time alone:

+   - **Scheduled retraining:** Monthly or quarterly refreshes when the task is stable and data arrives regularly.
+   - **Triggered retraining:** Retraining when F1, calibration, latency, drift statistics, embedding drift, or human-review disagreement cross predefined thresholds.
+   - **Incremental learning:** Controlled updates using newly labeled examples, with validation, approval, rollback control, and audit trails.

In regulated financial environments, retraining should also be documented through versioned datasets, model cards, validation reports, approval records, and post-deployment monitoring logs.