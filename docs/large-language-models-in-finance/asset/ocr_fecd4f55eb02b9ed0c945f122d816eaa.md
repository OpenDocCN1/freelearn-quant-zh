```
Test: "What was revenue growth from Q1 2023 to Q2 2023?"
Model Output: "Revenue grew 8% QoQ"
Validation: Verify model extracted Q2 – Q1, not Q1 – Q2
```

When evaluating **factual grounding**, model outputs must be supported by the underlying source documents. Generative models can produce fluent, plausible-sounding statements that are not supported by the source documents provided. In financial applications this is a serious risk—a hallucinated revenue figure or a misattributed earnings beat can propagate into downstream analysis or client communications. The factual grounding check below verifies that every numerical claim in the model's output can be traced to an explicit calculation from the source data:

```
Claim: "Apple's iPhone revenue grew 12% in Q3 FY2024"
Source Document: "iPhone revenue: Q3 FY2024: $39.30B, Q3 FY2023: $39.67B"
Calculation: ($39.30B - $39.67B) / $39.67B = -0.9%
Validation: FAIL -- the claim asserts growth; the source shows a small decline. The output is not grounded. Block and escalate rather than passing the figure downstream.
```

Even with these domain-specific checks in place, evaluation remains incomplete if it only reflects the conditions captured in the test set.

# 3.8.3 Robustness testing

A model that performs well on the held-out test set may still fail in production when it encounters inputs that differ from the training distribution. Financial text is particularly prone to distribution shift: vocabulary, framing conventions, and the mix of entity types all vary across sectors, geographies, and market regimes. Robustness testing proactively exposes these failure modes before deployment by deliberately constructing challenging inputs that go beyond what the standard test set covers.

A common approach is to use **adversarial examples** to probe whether the model has learned the underlying semantics of financial language or merely memorized surface patterns from training data. A model that correctly labels "revenue declined" as negative but fails on “top-line performance came in below expectations” has learned lexical cues rather than meaning. Systematic paraphrasing, negation testing, and out-of-distribution inputs reveal this brittleness:

+   • Paraphrased queries (test semantic understanding)
+   • Negations and edge cases
+   • Out-of-distribution inputs

A complementary approach is to use **stress testing** to evaluate how the model behaves under more demanding or complex input conditions. Stress tests expose the model to the most demanding inputs it is likely to encounter in production—very long filings, documents with multiple entities sharing similar names (e.g., multiple subsidiaries of the same parent),