This formulation unifies training under a text-to-text paradigm, treating both input and output as sequences. Such flexibility provides a general foundation for a wide range of financial tasks, from summarization to structured transformation.

Once the learning objective is established, the next design decision concerns the data itself: what corpus the model will be exposed to during pre-training.

# 2.3.2 Training corpus design

The composition and quality of the pre-training corpus fundamentally determine what a model knows, how it reasons, and where it fails. If the objective defines how the model learns, the corpus defines what it learns. For financial language models, this choice is not incidental. It shapes domain fluency, numerical robustness, regulatory awareness, and ultimately downstream reliability.

# General vs. domain-specific pre-training

One of the first strategic decisions concerns the balance between general and domain-specific data.

Models like GPT-4 and Claude are trained on diverse internet text, including books, websites, academic papers, and code. This broad exposure provides strong general reasoning capabilities. However, general exposure does not guarantee deep familiarity with the structural conventions, regulatory nuances, and numerical patterns that characterize financial discourse.

Domain-specific pre-training addresses this limitation by explicitly incorporating financial text into the corpus. BloombergGPT, for example, was trained on a mixture of approximately 51% financial data and 49% general data. The financial half (FinPile, 363B tokens) comprises news, regulatory filings, research reports, and market commentary; the general half (345B tokens) is drawn from public corpora. This blended approach attempts to preserve general reasoning strength while injecting domain-specific expertise.

The trade-off is structural. Too little financial data results in shallow domain competence. Too much specialization risks overfitting to financial language at the expense of broader reasoning ability. Corpus composition, therefore, becomes a calibration problem rather than a binary choice.

# Components of a financial corpus

A comprehensive financial pre-training corpus must reflect the full ecosystem of financial communication rather than a single document type.

*Regulatory filings* form a foundational layer. SEC forms such as 10-K, 10-Q, 8-K, and S-1, along with proxy statements (DEF 14A) and insider trading reports (Form 4), expose the model to formal disclosure structures, accounting terminology, and legally constrained language.

*Financial news and media* add timeliness and narrative framing. Sources such as Bloomberg, Reuters, and the Wall Street Journal introduce event-driven reporting, market reactions, and