```
```python
    'chunk_id': i
    }],
    ids=[f"{doc.get('id', 'doc')}_{i}"]
)```
```

```
```python
def retrieve(self, query: str, decision_time: str,
             filters: Optional[Dict] = None) -> List[Dict]:
    """Retrieve documents admissible at `decision_time`.

    decision_time is REQUIRED, not optional. The previous version stored
    an availability stamp on every document and then never filtered on it,
    which is the most expensive kind of near-miss: the provenance is
    recorded, the control is absent, and the retrieval silently returns
    documents that did not exist when the decision was made. A store that
    knows the availability time and does not enforce it is not a
    point-in-time store; it is an ordinary store with a comment.
    """
    if not decision_time:
        raise ValueError(
            "decision_time is required: retrieval without it cannot be "
            "point-in-time admissible")

    query_embedding = self.embedding_model.encode(query)

    # Enforce admissibility IN the query, not after it: filtering after
    # top-k has already discarded the admissible documents that ranked
    # below inadmissible ones.
    where = {"available_at": {"$lte": decision_time}}
    if filters:
        where = {"$and": [where, filters]}

    results = self.collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=self.top_k,
        where=where,
    )

    # Defence in depth: assert the store honoured the filter.
    for md in results.get("metadatas", [[]])[0]:
        if md["available_at"] > decision_time:
            raise AssertionError(
                f"look-ahead: {md['available_at']} > {decision_time}")

    return results```
```

```
```python
def _chunk_text(self, text: str) -> List[str]:
    """Chunk text with overlap"""
    words = text.split()
    chunks = []```
```