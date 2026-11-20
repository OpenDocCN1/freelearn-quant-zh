To guide index selection in practice, the following heuristics are commonly used:

+   - Flat (Exact): \( N < 10^6 \), maximum accuracy required
+   - IVF-Flat: \( 10^6 < N < 10^7 \), balance of speed and accuracy
+   - IVF-PQ: \( N > 10^7 \), memory constraints
+   - HNSW: High-recall applications with sufficient memory

# 4.5.3 Hybrid search

Purely semantic retrieval can miss exact keyword matches, particularly for structured queries, entity names, or regulatory terms. To address this limitation, hybrid search combines dense (semantic) and sparse (term-based) retrieval methods [Robertson and Zaragoza, 2009].

In this setting, dense retrieval captures semantic similarity between queries and documents, while sparse retrieval (e.g., BM25) emphasizes exact term matching and frequency-based relevance. These signals can be combined into a single scoring function as follows:

```
\text{score}_{\text{dense}}(q, d) = \text{sim}(f(q), f(d)) \tag{4.11}
```

```
\text{score}_{\text{BM25}}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)} \tag{4.12}
```

```
\text{score}(q, d) = \alpha \cdot \text{score}_{\text{dense}}(q, d) + (1 - \alpha) \cdot \text{score}_{\text{BM25}}(q, d) \tag{4.13}
```

where \(\alpha \in [0, 1]\) is a weighting parameter, typically \(\alpha = 0.5\).

# 4.6 Chunking strategies

fixed-size chunking, 400 tokens

Note 7. Debt Covenants
"Consolidated Leverage Ratio" means the ratio of Consolidated Funded Debt to Consolidated EBITDA ...

| Leverage ratio | 3.42x | 3.10x |
| Maximum permitted | 3.50x | 3.50x |

what the retriever can now answer

"What does leverage ratio mean?" definition retrieved. No number.

"Is the company in compliance?" 3.42x retrieved — against what limit? The table has lost its definition.

Fix: chunk on document structure. Never sever a table from its caption, or a term from its definition.

# Figure 4.2: Why token-count chunking fails on financial documents.