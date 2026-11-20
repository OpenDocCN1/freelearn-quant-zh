# RNN (Sequential)

# Transformer (Parallel)

![](bbox=[0.16, 0.15, 0.84, 0.28])

# Figure 1.2: RNNs versus transformers

dependencies. Unlike RNNs that process text word by word, Transformers can process entire sequences simultaneously, examining all pairwise relationships between words.

RNNs process tokens sequentially, creating a bottleneck for long financial documents. Transformers attend to all positions in parallel — but the claim needs a boundary drawn around it, because it is routinely overstated. Parallel applies to training and to prompt processing: a 200-page filing can be ingested in one forward pass. Generation is a different matter. Autoregressive decoding remains strictly sequential, one token at a time, each conditioned on all those before it. This is why a long prompt and a long answer cost differently: prefill is parallel and decode is not. It is not, however, a licence to treat prompts as free. Prefill is O(n2) in attention and at very long contexts it can dominate both latency and cost outright; what parallelism buys is that the work can be done at once, not that there is less of it. Within that boundary, parallel attention enables efficient processing of full earnings transcripts or regulatory filings.

The self-attention mechanism is computed as:

```
Attention(Q, K, V) = softmax( (QK^T / sqrt(d_k)) V ) (1.2)
```

where Q, K, and V denote the queries, keys, and values, respectively, and dk is the dimension of the keys. This mechanism computes a weighted sum over all positions in the sequence, allowing the model to capture dependencies irrespective of distance. Conceptually, this formula allows the model to determine which words in a financial document (like “debt” and “covenant”) are most relevant to one another, regardless of how far apart they appear in the text.

The advantages of Transformers for financial text processing include:

+   1. Parallel processing: Training and prompt prefill can process positions in parallel; autoregressive decoding remains sequential

+   2. Long-range dependencies: Attention creates direct interaction paths between distant positions, subject to masking and context-window limits