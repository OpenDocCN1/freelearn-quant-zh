# ## 4.6.4 Semantic chunking

To better capture meaning, semantic chunking uses embedding models to identify boundaries based on changes in content rather than formatting [Kamradt, 2023].

```
Algorithm 9 Semantic chunking
1: Input: sentences \(\{s_1, \dots, s_n\}\), embedding function \(f\), threshold \(\tau\)
2: Compute embeddings: \(e_i = f(s_i)\) for all \(i\)
3: Compute similarities: \(\text{sim}_i = \text{sim}_{\cos}(e_i, e_{i+1})\) for \(i = 1, \dots, n - 1\)
4: Initialize chunks \(\leftarrow [[s_1]]\)
5: for \(i = 2\) to \(n\) do
6:   if \(\text{sim}_{i-1} < \tau\) then
7:     Start new chunk: chunks.append([\(s_i\)])
8:   else
9:     Add to current chunk: chunks[-1].append(\(s_i\))
10: end if
11: end for
12: return chunks
```

By grouping sentences with high semantic similarity and splitting where similarity drops below a threshold, this approach produces chunks that more closely align with the underlying meaning of the text.

However, this added flexibility comes at the cost of increased computational complexity and reliance on embedding quality.

# ## 4.6.5 Financial document chunking

Financial documents exhibit well-defined structures, and effective chunking strategies should take advantage of these patterns.

For example, in **10-K and 10-Q reports**, chunking is typically aligned with section boundaries (such as Item 1, Item 1A, or Item 7), ensuring that related disclosures remain grouped together and that tables are preserved alongside their explanatory text.

In **research reports**, chunking is often performed at the level of logical sections such as executive summaries, analysis, and recommendations, with care taken to keep charts and quantitative metrics within their surrounding context.

Similarly, in **regulatory filings and legal documents**, it is important to respect paragraph boundaries, preserve definitions alongside their usage, and maintain cross-references between clauses.