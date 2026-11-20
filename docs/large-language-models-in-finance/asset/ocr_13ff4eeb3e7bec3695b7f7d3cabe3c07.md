A 400-token window knows nothing about Note 7 and cuts it in half, and the two halves are individually useless: the definition without the number cannot answer a compliance question, and the number without the definition cannot either — worse, it looks like it can. The chunk boundary is not a tuning parameter here, it is a semantic decision, and the only safe rule in a filing is to let the document’s own structure decide where the cuts fall. Embeddings and indexes operate on chunks, not on full documents, and the way a document is split into chunks determines what evidence the retriever can ever surface. A 10-K is too long to embed as a single unit, but if we naively cut it every N tokens, we are likely to break tables across boundaries, separate definitions from their usages, and merge unrelated risk factors into a single passage. The chunking stage, therefore, plays a much larger role in retrieval quality than its conceptual simplicity suggests, and it is the place where domain knowledge about financial documents pays the highest dividend.

In what follows, we begin with the trade-offs that any chunking strategy must navigate, then progress from the simplest fixed-size approach to recursive and semantic methods, and finally discuss chunking patterns tailored to specific financial document types.

# ## 4.6.1 The chunking problem

LLM context windows are finite and expensive to fill. Even when a model exposes a very long window, documents often still need to be split into smaller units so that retrieval remains auditable, permission-aware, and cost-efficient. These chunks should satisfy several competing requirements:

+   - They must fit within context limits
+   - They should preserve semantic coherence
+   - They must contain sufficient context for understanding
+   - They should optimize retrieval relevance

These requirements introduce trade-offs:

+   - **Small chunks** (128–256 tokens) improve precision but may lack sufficient context
+   - **Large chunks** (512–1024 tokens) provide more context but can dilute relevance