Caching, via cache embeddings and frequent queries, converts repeat work into O(1) lookups:

+   - Embedding cache: O(1) lookup vs. O(d) computation

+   - Query result cache: Avoid redundant retrieval

+   - TTL-based invalidation for freshness

Batch processing amortizes fixed overheads—API call setup, model loading, network round-trips—across many items per batch:

Cost per query = \frac{\text{Fixed overhead} + n \cdot \text{Variable cost}}{n} (4.24)

As n \to \infty, cost per query \to \text{variable cost}.

# 4.11.2 Monitoring and observability

Scaling addresses the question of whether a system can handle production load; monitoring addresses whether it is actually doing the right thing once it is there. RAG systems fail in subtle, silent ways—a re-indexing job ships a worse embedding model, a new chunking strategy degrades recall, a prompt change increases hallucination—and none of these are visible without explicit instrumentation. The metrics below are the minimum that any production financial RAG system should track continuously.

Key metrics to track:

+   - Latency: P50, P95, P99 retrieval and generation times

+   - Quality: Average relevance scores, user feedback ratings

+   - Coverage: Fraction of queries with relevant results

+   - Cost: API costs, compute costs, storage costs

+   - Freshness: Document age distribution in results

In a regulated context, these metrics are not just operational dashboards but governance artifacts: they feed model risk reviews, support audit trails, and underpin the go/no-go thresholds defined in the evaluation protocol.

For financial institutions, monitoring should also record the controls that surrounded the answer: the corpus version, embedding model version, retrieval filters, user entitlement checks, prompt template version, generated citations, reviewer outcome when human approval is required, and any blocked or escalated requests. These fields make it possible to reconstruct why a system answered as it did and to distinguish a retrieval failure from a generation failure or a permissioning failure.

# 4.11.3 Continuous improvement

Monitoring tells us when something has gone wrong; continuous improvement is the discipline that turns those signals into a better system. Embedding models, chunking parameters, and prompts are not one-time choices but parameters of a living pipeline that evolves with the