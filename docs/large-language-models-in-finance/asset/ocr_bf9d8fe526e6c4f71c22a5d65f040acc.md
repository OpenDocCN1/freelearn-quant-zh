+   • **Human-in-the-loop behavior**: Measure how human analysts interact with the model: override rates, escalation patterns, and whether the model reduces or creates manual workload.

We recommend maintaining a living “model evaluation card” for each production model that records datasets, metrics, thresholds, and governance sign-offs, aligned with the institution’s broader model risk management framework.

# 3.8.6 Model risk management and regulatory compliance

In regulated financial institutions, deploying a fine-tuned language model is not purely a machine-learning problem; it is also a scoping, governance, and accountability problem. The governing framework depends on the institution, the use case, the system’s function, and the applicable legal and supervisory perimeter.

**U.S. scope and status.** On 17 April 2026, the Federal Reserve, FDIC, and OCC issued SR 26-2, *Revised Guidance on Model Risk Management* superseding SR 11-7 and SR 21-8. The guidance is non-binding: non-compliance with it alone does not constitute a violation of law or a basis for supervisory criticism. It is expected to be most relevant to banking organizations with more than $30 billion in total assets, while smaller organizations apply it in a manner proportionate to their risk profile and the nature and materiality of their model use. The revised definition also excludes tools that perform only simple arithmetic and software whose outputs are generated deterministically from fixed rules [Board of Governors of the Federal Reserve System et al., 2026].

The scope boundary is critical for this book. Footnote 3 of SR 26-2 states that generative-AI and agentic-AI models are novel and rapidly evolving and are *not within the scope of the guidance*. The same footnote makes two qualifications: an institution’s own governance and risk-management practices should still determine controls for systems outside the guidance, and the guidance continues to apply to traditional statistical and quantitative models and to non-generative, non-agentic AI models. A discriminative FinBERT-style classifier, scorer, or tagger may therefore remain in scope when it satisfies the revised model definition and is used in a relevant banking process; a generative LLM or agent does not become an SR 26-2 model merely because it is fine-tuned for finance. Such a system should not be described as “SR 26-2 compliant.”

Composition creates an important dependency without expanding the formal scope. If a generative LLM supplies features, narratives, extracted values, or decisions to an in-scope quantitative model, validation of the downstream model should examine the quality, stability, controls, and failure modes of that upstream input. The dependency belongs in the system inventory and evidence package even though the generative component remains outside SR 26-2’s stated perimeter.

For models that are within scope, the revised guidance is best summarized through four risk-based practices:

+   • **Determine applicability before borrowing language.** Apply the revised model definition, intended use, materiality, and institutional perimeter before labeling a system or control package.