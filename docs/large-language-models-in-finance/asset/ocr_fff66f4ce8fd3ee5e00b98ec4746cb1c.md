# 4.2 RAG system architecture

Having established why RAG is necessary, we now examine how a production RAG system is structured. At a high level, RAG is not a single model call but a multi-stage pipeline that transforms raw documents into grounded, verifiable responses. Each stage plays a distinct role, and design choices at each step directly affect system quality.

![](bbox=[0.14, 0.215, 0.85, 0.425])

Figure 4.1: A financial RAG pipeline.

Two things distinguish it from a generic one, and both are controls rather than models. The availability filter \(\mathcal{D}_{\le t}\) admits only documents that existed at the decision time \(t\), which is what stops a backtest from reading tomorrow's filing. The claim-level verifier checks each material claim \(A_i\) against its retrieved evidence bundle \(E_i\) before the answer is released, which is what stops a fluent answer from being an unsupported one. A pipeline with neither is a demo.

We can view the RAG pipeline as four interacting stages that move from raw data to final answer generation. While these stages are often implemented as separate components, they should be viewed as a tightly unified system in which errors in earlier stages propagate downstream.

It is useful to think of the pipeline as the structural backbone of any RAG system. Data flows from offline document processing on the left to online query processing on the right: documents are ingested, cleaned, chunked, embedded, and stored in an index; user queries are then transformed, embedded, and matched against this index; the resulting passages are assembled into a prompt; and the language model produces a grounded answer with citations. Because each stage consumes the output of the previous one, design decisions taken early on are silently inherited by everything that follows. Aggressive cleaning that drops table headers degrades retrieval; chunking that splits a definition from its usage degrades grounding; weak embeddings degrade ranking, which then forces re-ranking and post-processing to compensate. For this reason, throughout the rest of the chapter we treat chunking quality, embedding quality, and retrieval quality not as isolated tuning knobs, but as the dominant determinants of downstream generation quality.

Stage 1: Document processing and indexing This stage transforms raw, heterogeneous