# ## 4.7.2 Re-ranking

The retrieval stage is tuned for recall: we want to be reasonably sure that the relevant passage is somewhere in the top-k. Re-ranking is the complementary stage that targets precision. It takes the candidate set returned by the index and applies a more expensive scoring model to it, accepting higher latency in exchange for a much sharper ordering of the truly relevant passages [Nogueira and Cho, 2019].

# ## Cross-encoder re-ranking

Unlike bi-encoders that separately encode query and document, a cross-encoder computes:

score(q, d) = σ(CrossEncoder([q; d])) (4.14)

where [q; d] denotes concatenation and σ is a scoring function.

```
```python
from sentence_transformers import CrossEncoder

def rerank_documents(query: str, documents: list,
                   model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                   top_k: int = 5):
    """Re-rank documents using cross-encoder."""
    model = CrossEncoder(model)

    # Score all query-document pairs
    pairs = [(query, doc) for doc in documents]
    scores = model.predict(pairs)

    # Sort and return top-k
    ranked = sorted(zip(documents, scores),
                    key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
```
```

# ## 4.7.3 Metadata filtering

Embeddings capture semantic similarity but say nothing about whether a passage comes from a 10-K filed last quarter or a research note from five years ago. In financial applications, this distinction is often decisive. Metadata filtering complements semantic search with hard, structured constraints—company, document type, section, filing window—that narrow the candidate set before similarity is even computed.