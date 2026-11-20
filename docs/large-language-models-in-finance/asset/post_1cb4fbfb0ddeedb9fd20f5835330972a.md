

The layer normalization operation normalizes activations across the feature dimension:

LayerNorm(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta \quad (2.7)

where $\mu$ and $\sigma^2$ are the mean and variance computed across the feature dimension, $\epsilon > 0$ is a small numerical-stability constant used in standard implementations, and $\gamma$ and $\beta$ are learned scale and shift parameters.

Position encoding is another critical component that enables transformers to understand sequential relationships in financial data. Since transformers process sequences in parallel, positional information must be explicitly encoded. The standard sinusoidal position encoding is defined as:

PE_{(pos,2i)} = sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) \quad (2.8)
PE_{(pos,2i+1)} = cos\left(\frac{pos}{10000^{2i/d_{model}}}\right) \quad (2.9)

What large financial models actually use is more ordinary than this framing suggests, and the difference is worth stating plainly. BloombergGPT, the best-documented finance-specific model of its generation, uses ALiBi (attention with linear biases) — a general-purpose relative-position method that adds a distance-proportional penalty to attention scores [Wu et al., 2023b]. It implements no trading-day adjustment, no temporal-distance measure calibrated to markets, and no market-regime indicator in its positional scheme. Position encodings that condition on session boundaries, exchange calendars, or regime labels are a reasonable research direction for financial time series, and this book flags them as such; they should not be attributed to models that do not implement them. A citation is a claim about what a paper contains, and the difference between “a model could do this” and “this model does this” is the difference the rest of this book is about.

The components above explain how transformers function. The next step is understanding how we make them work at scale in real financial systems.

## 2.1.2 Optimizations for large-scale models

The scaling of transformer models to billions of parameters requires sophisticated optimization techniques to maintain computational feasibility while preserving model performance. These optimizations are particularly critical in financial environments, where real-time inference, regulatory compliance, and cost constraints impose requirements beyond pure model performance.

We begin with attention itself, since it is both the strength and the computational bottleneck of transformers. Sparse attention techniques aim to preserve modeling power while reducing cost.

### Sparse attention mechanisms

Let $n$ denote the sequence length, that is, the number of tokens in the document, prompt, or time-series representation being processed. Traditional self-attention mechanisms exhibit quadratic complexity $O(n^2)$ with respect to this sequence length, which becomes prohibitive for processing long financial documents such as annual reports (often exceeding 100,000 tokens) or extended time series data spanning multiple years of trading history. Sparse attention patterns address this limitation by restricting attention computations to specific, theoretically motivated patterns. Local attention mechanisms restrict each token to attend only to tokens within a fixed window, reducing complexity to $O(n \cdot w)$ where $w$ is the window size. Local attention proves particularly effective for time series analysis where immediate temporal neighbors often contain the most relevant information for prediction tasks:

\(\text{LocalAttention}_{i,j} = \begin{cases} \text{Attention}_{i,j} \quad \text{if } |i - j| \le w/2 \\ 0 \quad \text{otherwise} \end{cases}\) (2.10)

Strided attention patterns allow tokens to attend to every $k$-th position, enabling long-range dependency modeling while maintaining computational efficiency. This approach proves valuable for connecting quarterly earnings reports or annual financial statements across multiple years:

\(\text{StridedAttention}_{i,j} = \begin{cases} \text{Attention}_{i,j} \quad \text{if } (j - i) \mod k = 0 \\ 0 \quad \text{otherwise} \end{cases}\) (2.11)

Random attention patterns introduce stochastic connections that can capture unexpected long-range dependencies while maintaining overall computational efficiency. Combining local, strided, and random attention patterns can preserve useful local and long-range interactions at substantially lower cost than dense attention [Child et al., 2019]. Global attention mechanisms designate certain tokens as "global," allowing them to attend to all positions while other tokens attend only locally. In financial document analysis, summary sections, financial statement headers, or key performance indicator mentions might serve as global tokens:

\(\text{GlobalAttention}_{i,j} = \begin{cases} \text{Attention}_{i,j} \quad \text{if } i \in \mathcal{G} \text{ or } j \in \mathcal{G} \\ \text{LocalAttention}_{i,j} \quad \text{otherwise} \end{cases}\) (2.12)

where $\mathcal{G}$ represents the set of global token positions. The Longformer architecture implements a combination of these attention patterns, achieving linear complexity while maintaining the ability to model long-range dependencies crucial for comprehensive financial document analysis [Beltagy et al., 2020]. Empirical evaluation on financial documents has demonstrated that Longformer-based models can process entire annual reports while maintaining competitive performance on downstream financial reasoning tasks.

Reducing attention cost addresses the sequence length, but scaling model capacity introduces a different challenge. **Mixture of Experts (MoE)** architectures tackle this by increasing parameter count without proportionally increasing computation.

### MoE architectures

MoE architectures represent a paradigm shift in scaling transformer models, enabling massive parameter counts while maintaining computational efficiency through conditional computation. In MoE architectures, the dense feed-forward networks in transformer layers are replaced with multiple expert networks, with a learned gating mechanism determining which experts to activate for each input. The MoE layer can be formulated as:

\(\text{MoE}(x) = \sum_{i=1}^{E} G(x)_i \cdot E_i(x)\) (2.13)

where \(E\) is the number of experts, \(G(x)\) is the gating function that outputs a probability distribution over experts, and \(E_i(x)\) represents the \(i\)-th expert network. The gating function is typically implemented as:

\(G(x) = \text{Softmax}(x \cdot W_g)\) (2.14)

To maintain computational efficiency, only the top-\(k\) experts (typically \(k=1\) or \(k=2\)) are activated for each input token, reducing per-token computation while retaining a large total parameter capacity. The Switch Transformer implements this top-\(k\) routing with additional load balancing mechanisms to ensure expert utilization remains balanced across the dataset [Fedus et al., 2022]. For financial applications, expert specialization proves particularly valuable. Different experts can focus on distinct financial domains: equity analysis, fixed income evaluation, derivatives pricing, risk assessment, or regulatory compliance. This specialization enables the model to maintain high performance across diverse financial tasks while using only a fraction of total parameters for any given input. Advanced MoE implementations for finance incorporate domain-aware routing mechanisms that consider financial context when selecting experts. For instance, the routing function might incorporate market sector information, asset class indicators, or regulatory framework identifiers to improve expert selection accuracy:

\(G(x, c) = \text{Softmax}([x; c] \cdot W_g)\) (2.15)

where \(c\) represents contextual information about the financial domain or task. Load balancing in financial MoE models requires special consideration due to the potentially uneven distribution of financial content across different domains. Auxiliary loss functions encourage balanced expert utilization while maintaining performance:

\(\mathcal{L}_{\text{aux}} = \alpha \cdot \text{CV}(f_1, f_2, ..., f_E)^2\) (2.16)

where CV denotes the coefficient of variation of expert utilization frequencies \(f_i\), and \(\alpha\) is a hyperparameter controlling the strength of load balancing.

Even with architectural innovations such as sparse attention and MoE, large models still require careful engineering. This final optimization focuses on practical memory and hardware strategies that enable the deployment of large-scale financial transformers.

