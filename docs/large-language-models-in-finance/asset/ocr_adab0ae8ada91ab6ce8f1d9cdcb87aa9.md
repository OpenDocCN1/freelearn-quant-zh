```
```python
for j, col in enumerate(sentiment_cols):
    news_df[col] = scores[:, j]

# Aggregate to daily per-ticker features
daily_sent = (
    news_df
    .groupby(["ticker", "date"])[sentiment_cols]
    .mean()
    .reset_index()
)

# 'daily_sent' can be merged with returns, factors, and risk metrics.
```
```

In practice, this sentiment subsystem would be accompanied by:

+   - calibration and stability checks across time and sectors [Niculescu-Mizil and Caruana, 2005],
+   - analysis of factor interactions (e.g. sentiment vs. value or momentum) [Stambaugh et al., 2012],
+   - careful treatment of lookahead bias and publication timestamps [Katona et al., 2018].

# ## 7.1.3 Event-driven and macro strategies

Unlike continuous forecasting, event-driven strategies focus on specific announcements or shocks. Event-driven and macro strategies revolve around discrete information releases:

+   - quarterly earnings and guidance updates [Beaver, 1968, Ball and Brown, 1968],
+   - merger and acquisition announcements [Mitchell and Pulvino, 2001],
+   - credit rating changes [Hand et al., 1992],
+   - macroeconomic data releases [Andersen et al., 2003, Savor and Wilson, 2014],
+   - central-bank statements and speeches [Nakamura and Steinsson, 2018, Gardner et al., 2022].

LLMs support these strategies by:

+   - reading earnings releases and call transcripts in near real time, extracting surprises, forward guidance, and risk disclosures [Davis et al., 2015, Larcar and Zakolyukina, 2012];
+   - classifying central-bank communications as more or less restrictive than expected [Hansen et al., 2018];
+   - generating structured event summaries that feed into factor models or scenario generators [Ke et al., 2019].