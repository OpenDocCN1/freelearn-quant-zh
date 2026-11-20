during pre-training, the system first retrieves relevant documents, passages, tables, or filings and then conditions generation on that retrieved evidence [Lewis et al., 2020].

A compact abstraction is:

```
\hat{y} = G_\theta(x, R_k(x; \mathcal{K}_t)), \quad (2.31)
```

where x is the user query, K_t is the time-indexed knowledge base available at time t, R_k retrieves the top-k relevant evidence units, and G_\theta is the generator. The time index matters in finance: a response about a company's leverage, capital adequacy, or risk factors should depend on the filings, market data, and policies that were valid at the relevant date.

A practical financial RAG pipeline has five stages:

+   1. **Ingestion**: collect filings, research reports, policies, market commentary, and structured data feeds with metadata such as publication date, fiscal period, source, permission level, and document type.
+   
+   2. **Chunking and indexing**: split documents into coherent passages or table units, embed them, and store them in a vector, keyword, or hybrid search index.
+   
+   3. **Retrieval and reranking**: retrieve candidate evidence using semantic and lexical search, then rerank by relevance, recency, authority, and access permissions.
+   
+   4. **Grounded generation**: pass the selected evidence to the model and require the answer to cite or summarize only the retrieved material.
+   
+   5. **Validation and logging**: check citations, numerical consistency, access rights, latency, and audit records before the answer enters a regulated workflow.

RAG is especially important for financial systems because it reduces reliance on stale parametric memory. For example, if a user asks what changed between two annual reports, the system should retrieve both the current and prior 10-K, preserve their filing dates, compare the relevant sections, and state which version supports each claim. This design also helps with document-level auditability: the answer can be traced back to a specific filing, paragraph, table, or policy record.

RAG, however, is not a complete solution to hallucination or alignment. It improves factual grounding, but the system can still retrieve the wrong document, misread a table, overstate a conclusion, or ignore a compliance constraint. For this reason, alignment and hallucination control enter the architecture at several layers: retrieval filters, permission checks, prompt constraints, citation requirements, numerical validators, human review queues, and post-generation monitoring. In regulated financial deployments, RAG should therefore be viewed as one component of a broader governed generation stack rather than a standalone guarantee of truth.

With prompting, retrieval, and adaptation in place, we can now turn to a more fundamental question—how such a system is built in the first place.