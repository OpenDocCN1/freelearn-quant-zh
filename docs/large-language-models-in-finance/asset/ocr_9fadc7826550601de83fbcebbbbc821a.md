This separation is operationally important. The detector used for real-time screening should remain a validated supervised model with monitored thresholds, calibration, and false-positive rates. The LLM-assisted rationale or triage layer is an auxiliary workflow component: it may improve analyst productivity and explainability, but it should be evaluated for faithfulness and should not be allowed to silently change the underlying alert label.

The overall detector-training objective typically combines a supervised loss for fraud labels (e.g. cross-entropy) with regularization terms that control class imbalance, calibration, or fairness metrics.

# 12.3.2 Narrative-augmented credit risk

Narrative credit risk models combine traditional numerical features with embeddings of borrower narratives. These embeddings may come from a FinBERT- or FinEAS-style encoder, depending on whether the task emphasizes classification or sentence-level similarity. Let \(\mathbf{f}_\text{num} \in \mathbb{R}^{d_n}\) denote numerical features and \(\mathbf{f}_\text{narr} \in \mathbb{R}^{d_t}\) narrative embeddings from a financial sentence encoder. The following formula gives a stylized risk score:

\(\text{RiskScore} = \sigma(\mathbf{w}_\text{num}^\top \mathbf{f}_\text{num} + \mathbf{w}_\text{narr}^\top \mathbf{f}_\text{narr} + b),\)

where \(\sigma\) is a squashing function (e.g. logistic). The model can be trained with:

\(\min_{\mathbf{w}_\text{num}, \mathbf{w}_\text{narr}, b} \mathbb{E}[\ell(y, \text{RiskScore}) + \lambda_\text{fair} \Phi(\mathbf{f}_\text{narr})],\)

where \(\ell\) is a classification loss (e.g. cross-entropy), and \(\Phi\) encodes fairness or regularization penalties (e.g. discouraging correlation between narrative features and protected attributes).

Reported models often show:

+   - Higher discrimination on default vs non-default events;
+   - Particular gains for thin-file or non-traditional borrowers; and
+   - New challenges around fairness, explainability and data governance.

These challenges tie directly into the governance frameworks of Chapter 11.

# 12.4 Evaluation frameworks and benchmarks

Robust, multi-dimensional evaluation is essential to avoid overinterpreting isolated successes. In finance, a model that performs well on sentiment classification may still fail at numerical reasoning; a model that answers questions over filings may still hallucinate unsupported arithmetic; and a model that looks accurate on a static benchmark may still be unsuitable for a workflow with transaction costs, latency limits, audit requirements, and adversarial inputs.

This section therefore treats evaluation as a layered problem. The first layer is task accuracy on standardized datasets. The second layer is reasoning fidelity: whether the model retrieves the right evidence, manipulates tables correctly, and produces numerically valid intermediate