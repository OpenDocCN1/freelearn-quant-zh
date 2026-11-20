results are then averaged into a single `EvaluationResult`. The lexical scoring functions in the listing are intentionally conservative smoke tests. They are useful in continuous integration because they are deterministic and cheap, but they should not be mistaken for a complete faithfulness audit. In production, the faithfulness hook should usually be replaced with a claim-level checker that decomposes the answer into atomic claims and verifies each claim against the retrieved passages, while relevancy can be strengthened with embedding similarity or expert-labeled calibration data.

# 4.11 Production considerations

A RAG system that performs well on a benchmark and one that performs well in production at a financial institution are rarely the same artifact. The evaluation protocol of the previous section establishes whether a pipeline is good enough to deploy; this section addresses what it takes to keep it good once it is deployed. Three concerns dominate. First, the system must _scale_: an index that comfortably serves a research prototype may fail at the throughput, latency, and cost profile a production application requires. Second, the system must be _observable_: in regulated settings, every query, retrieval result, and generated answer is potentially subject to audit, and silent quality regressions cannot be tolerated. Third, the system must _improve continuously_: corpora drift, user expectations evolve, and embedding models are routinely upgraded, so a static deployment quickly becomes a stale one. We address each of these in turn.

# 4.11.1 Scaling RAG systems

The first set of pressures comes from sheer volume. As a financial RAG system absorbs more documents, more users, and more concurrent workloads, several scaling challenges surface that the basic pipeline does not address.

+   1. **Index size**: Millions of documents → billions of embeddings
+   2. **Query throughput**: Thousands of queries per second
+   3. **Latency**: Sub-second response times required
+   4. **Cost**: Embedding API costs, vector DB costs
+   5. **Freshness**: Continuous document updates

These pressures rarely act in isolation. Adding more documents drives up index size and embedding cost; serving more queries drives up throughput and latency requirements; and continuous corpus updates strain both. The standard production responses—sharding, caching, and batch processing—each target one or more of these axes simultaneously.

**Sharding** distributes documents across multiple shards to keep per-shard index size manageable and to parallelise queries across machines:

```
shard(d) = hash(id(d)) mod n_shards
```