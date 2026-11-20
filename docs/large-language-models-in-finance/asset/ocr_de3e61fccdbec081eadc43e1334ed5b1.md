```
claim_tokens = token_set(claim.text)
if not claim_tokens:
    return False
evidence_tokens = set().union(*(token_set(e.text) for e in evidence))
return len(claim_tokens & evidence_tokens) / len(claim_tokens) >= min_overlap
def faithfulness_score(claims: Iterable[Claim],
    evidence: dict[str, Sequence[Evidence]]) -> float:
    material = [c for c in claims if c.material]
    if not material:
        return 1.0
    supported = [supports_claim(c, evidence.get(c.text, [])) for c in material]
    return sum(supported) / len(supported)
```

The smoke test should not be oversold. Token overlap will miss paraphrase, numerical entailment, negation, and legal nuance. Its role is to enforce the interface: answers must be decomposed into claims, claims must have evidence bundles, and unsupported material claims must become visible before release. Stronger deployments replace supports_claim with a hybrid verifier combining exact citation checks, numerical consistency checks, embedding retrieval, and expert-labeled entailment tests.

# ## 4.10.4 Evaluation protocol for regulated financial Q&A

For regulated use cases—such as answering questions about capital requirements, market abuse rules, or internal policies—RAG evaluation must go beyond generic IR and QA metrics. A minimal but robust protocol typically includes the following steps:

+   1. **Define the evaluation scope:** Select one or more well-defined policy domains (e.g., MiFID II product governance, market manipulation, suitability) and assemble a corpus that matches the documents used in production (regulations, internal policies, FAQs, supervisory guidance).

+   2. **Create a gold-standard test set:** Draft queries that represent real user questions (from compliance, front office, operations) and have subject-matter experts write:
+   • canonical answers,
+   • lists of documents or passages that support each answer,
+   • severity labels indicating the impact of an incorrect answer.

+   3. **Instrument the full pipeline:** Log retrieval results, selected contexts, generated answers, and all relevant metadata (model version, index version, prompt template) so that each test run is reproducible.

+   4. **Measure retrieval and generation jointly:** In addition to Recall@k, Precision@k, MRR, and nDCG, compute:
+   • **Faithfulness:** whether claims in the answer are actually supported by retrieved passages.