The choice of loss function depends on the financial task. For a single-label classification example, such as document classification, sentiment analysis, or risk categorization, the cross-entropy loss is

\(\mathcal{L}_{\text{CE}} = - \sum_i y_i \log \hat{p}_i,\) (15.10)

where \(\hat{p}_i\) represents predicted class probabilities and \(y_i\) represents the true label.

For continuous targets, such as risk scores, default probabilities, volatility forecasts, or expected returns, regression losses are used. A common example is mean squared error:

\(\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2.\) (15.11)

MSE penalizes larger errors more heavily because the error term is squared.

Finance often requires task-specific losses. In fraud detection or anti-money laundering, false negatives may be much more costly than false positives, so an asymmetric cost-sensitive loss can be used:

\(\mathcal{L}_{\text{cost}} = - \sum_i (c_1 y_i \log \hat{p}_i + c_0(1 - y_i) \log(1 - \hat{p}_i)), \quad c_1 > c_0.\) (15.12)

In credit risk, calibration-aware losses may combine classification accuracy with penalties for miscalibrated probabilities. In return, liquidity, or volatility forecasting, quantile loss is often useful when the objective is to estimate a conditional quantile rather than a conditional mean:

\(\mathcal{L}_{\tau}(y, \hat{q}) = \max \{ \tau(y - \hat{q}), (\tau - 1)(y - \hat{q}) \}, \quad 0 < \tau < 1.\) (15.13)

For example, a lower-tail quantile loss can be used when forecasting downside risk or value-at-risk-style quantities.

For sentence-embedding models, a metric-learning or similarity-learning objective is sometimes used so that two sentences with similar financial sentiment have nearby embeddings, while sentences with different risk implications are farther apart. It is worth being precise that FinEAS does not do this: its published objective is mean squared error on a tanh regression head over a continuous RavenPack sentiment score [Gutiérrez-Fandiño et al., 2021]. The contrastive form below is a general pattern from the sentence-embedding literature, offered as an option and not as a description of FinEAS. A simplified contrastive objective has the form

\(\mathcal{L}_{\text{contrastive}} = - \log \frac{\exp(\text{sim}(u_i, v_i)/\tau)}{\sum_j \exp(\text{sim}(u_i, v_j)/\tau)},\) (15.14)

where \(u_i\) and \(v_i\) are paired sentence embeddings, \(\text{sim}(\cdot, \cdot)\) is a similarity function such as cosine similarity, and \(\tau\) is a temperature parameter. This type of objective is useful when financial sentiment is treated not only as a class label, but also as a geometry over sentence-level representations.