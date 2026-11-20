```
covenants = []
current = []
for w, p in zip(words, preds):
    label = id2label[p]
    if label.startswith("B-COVenant"):
        if current:
            covenants.append(" ".join(current))
            current = []
        current.append(w)
    elif label.startswith("I-COVenant") and current:
        current.append(w)
    else:
        if current:
            covenants.append(" ".join(current))
            current = []
if current:
    covenants.append(" ".join(current))
return covenants
```

In practice, covenant extraction would be combined with:

+   * core credit models (PD/LGD/EAD) [Basel Committee on Banking Supervision, 2005],
+   * collateral and exposure data,
+   * workflow tools for analysts to review and correct extracted information.

# 7.2.2 Market risk and scenario analysis

Market risk is the risk of losses from changes in prices, rates, spreads, currencies, or volatility. Market risk management uses models such as Value-at-Risk (VaR) and Expected Shortfall (ES) alongside stress testing and scenario analysis. VaR estimates a loss threshold at a given confidence level; Expected Shortfall estimates the average loss beyond that threshold. [Jorion, 2007, McNeil et al., 2015]. LLMs contribute by:

+   * generating narrative descriptions of stress scenarios [Hirtle and Lehnert, 2015],
+   * mapping narratives to shocks on risk factors (rates, spreads, FX, equity levels and volatilities, commodities),
+   * explaining why particular positions or portfolios are vulnerable.

A typical scenario pipeline:

+   1. define a qualitative scenario (e.g. “sharp, unexpected tightening by the central bank due to inflation surprises”)),
+   2. use an LLM to elaborate the scenario into a set of coherent macro events [Schorfheide et al., 2018],
+   3. map those events into shocks on curves and spreads [Diebold and Li, 2006],