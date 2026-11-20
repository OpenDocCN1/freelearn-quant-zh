```
n_estimators=200,
max_depth=6,
random_state=42
)
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)
# At this point, appropriate backtesting, risk analysis, and validation
# procedures should be applied. This listing focuses on architecture only.
```

In a live trading environment, this pipeline must be embedded in a broader framework:

+   • walk-forward evaluation with realistic transaction costs [Harvey et al., 2016, López de Prado, 2018],
+   • measured latency and reliability service-level objectives [Hasbrouck and Saar, 2013],
+   • risk controls (position limits, stop-loss policies, stress tests) [Jorion, 2007],
+   • integration with portfolio construction and order routing [Grinold and Kahn, 2000, Almgren and Chriss, 2001].

# 7.1.2 Sentiment and narrative signals

Sentiment analysis of financial text has evolved from dictionary-based methods [Loughran and McDonald, 2011] to domain-specific transformers such as FinBERT [Yang et al., 2020, Araci, 2019]. LLMs further generalize this by capturing multi-dimensional sentiment and narrative structure [Hassan et al., 2019, Calomiris and Mamaysky, 2019, Lopez-Lira and Tang, 2023]. For example, a document may be positive about revenue growth but negative about regulatory risk or liquidity conditions. Narrative signals capture recurring themes in text, such as inflation pressure, litigation risk, refinancing pressure, or management confidence.

Instead of a single sentiment label, a production system may estimate a vector:

```
s_t = (s_t^{pos}, s_t^{neg}, s_t^{unc}, s_t^{reg}, s_t^{liq}, ...),
(7.2)
```

where components might track positive tone, negative tone, uncertainty, regulatory risk, liquidity concerns, and so on [Baker et al., 2016, Hassan et al., 2019].

The terms s_t^{reg} and s_t^{liq} deserve separate interpretation. The regulatory-risk component, s_t^{reg}, is intended to capture language suggesting exposure to new rules, supervisory actions, enforcement investigations, capital or reporting constraints, licensing uncertainty, litigation, or other legal and compliance pressures. For example, repeated references to a regulator’s investigation, a pending rule change, or a possible fine should raise the regulatory-risk signal even when the document is otherwise positive about sales or earnings.

The liquidity-concerns component, s_t^{liq}, is intended to capture language indicating difficulty in trading, funding, refinancing, redeeming, or meeting short-term cash obligations without material cost. In market-facing text, liquidity concerns may appear as references to thin order