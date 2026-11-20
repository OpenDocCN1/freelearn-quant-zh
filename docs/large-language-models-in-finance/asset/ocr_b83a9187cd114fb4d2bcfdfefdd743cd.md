# ### 15.3.2 Architectural comparison for financial applications

| Feature | BERT (encoder-only) | GPT (decoder-only) | T5 (encoder-decoder) |
| :--- | :--- | :--- | :--- |
| **Architecture** | Uses only the encoder stack of the Transformer. Processes the entire sequence bidirectionally. | Uses only the decoder stack with causal masking. Processes the sequence autoregressively. | Uses both encoder and decoder stacks, like the original Transformer. |
| **Key feature** | Bidirectional context for understanding relationships between all tokens. | Autoregressive generation optimized for next-token prediction. | Sequence-to-sequence formulation treating every task as text transformation. |
| **Pre-training objective** | Original BERT uses masked language modeling (MLM) and next sentence prediction (NSP). Later BERT-style encoder models may omit, replace, or modify NSP. | Causal language modeling (CLM): predict the next token given previous context. | Text-to-text denoising or span corruption, depending on the encoder-decoder model family. |
| **Financial use cases** | NLU tasks: sentiment analysis, NER, document classification, entity extraction, compliance triage, and sentence-embedding models such as FinEAS. | Generative tasks: report drafting, summarization, scenario generation, code generation, multi-step reasoning. | Complex transformations: document translation, abstractive summarization, structured Q&A, data extraction. |

Table 15.1: Comparison of common Transformer architecture types for finance

The reference to next sentence prediction applies to the original BERT pre-training recipe [Devlin et al., 2019]. Later BERT-style models may remove or replace NSP, so the objective should be understood as architecture-family dependent rather than as a universal property of all encoder-only models.

In finance, model choice depends on the task, available data, integration constraints, latency requirements, and governance needs rather than on abstract architectural superiority. Each architecture offers specific advantages for different financial workflows.