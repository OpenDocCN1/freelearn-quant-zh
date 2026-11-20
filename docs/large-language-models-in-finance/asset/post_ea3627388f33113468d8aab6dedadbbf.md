
domain. The training distribution D is a mixture distribution

$$\mathcal{D} = \lambda_{\text{fin}}\mathcal{D}_{\text{fin}} + (1-\lambda_{\text{fin}})\mathcal{D}_{\text{gen}}, \quad \lambda_{\text{fin}} \in (0, 1).$$

In practice, \(\lambda_{\text{fin}}\) is chosen so that financial data contribute a substantial fraction of the total training tokens, without eliminating the benefits of broad general-language competence. This can be viewed as a form of _importance sampling_ over domains: financial samples are up-weighted relative to their raw frequency in the global corpus.

Architecturally, BloombergGPT is a GPT-style model with:

- \( L \) transformer blocks, where \( L \) denotes the number of repeated attention–feedforward layers;
- hidden dimension \( d_{\text{model}} \), the width of the internal token representation;
- multi-head self-attention with \( H \) heads per layer;
- ALiBi (attention with linear biases) positional encoding, applied as additive biases inside self-attention rather than as learned or rotary position embeddings; and
- standard autoregressive masking, so the model cannot see future tokens while predicting the next one.

These architectural terms are standard in transformer models. For the present chapter, the important point is qualitative: BloombergGPT allocates substantial model capacity and data budget to financial language, rather than relying only on generic pre-training followed by prompting.

### Reported empirical behavior

The research team evaluated the model on a variety of financial and general NLP tasks. In simplified terms, the reported pattern is as follows:

- On financial tasks, including classification, sentiment analysis, question answering, and entity recognition, BloombergGPT performs strongly relative to generic LLMs of comparable or larger size;
- On general-language benchmarks, it remains competitive with non-financial models; and
- Gains are especially visible on entity-rich, jargon-heavy tasks involving filings, research notes, and market-specific terminology.

These results support the central lesson of domain specialization: allocating a significant fraction of capacity and training budget to financial distributions can yield a better bias–variance trade-off for financial tasks than relying solely on generic pre-training.

### 12.1.2 FinBERT and domain-adaptive encoders

For readers encountering it for the first time, FinBERT is a financial version of the BERT encoder family. Unlike GPT-style models, which generate text one token at a time, BERT-style encoders read a text span bidirectionally and are especially useful for classification, extraction, and embedding tasks. FinBERT therefore exemplifies a complementary strategy: start from a general BERT encoder and continue training it on financial text before task-specific fine-tuning [Araci, 2019].

Let \(\theta_0\) denote the parameters of a pretrained BERT model, and let \(\mathcal{D}_{fin}\) be a collection of financial text. A document or sentence is represented as a token sequence \(x = (x_1, \dots, x_T)\), where \(T\) is the sequence length. Continued pre-training adapts the model to the financial corpus by solving the stylized objective

\[
\theta^* = \arg \min_{\theta} \mathbb{E}_{x \sim \mathcal{D}_{fin}} \left[ \mathcal{L}_{MLM}(x; \theta) + \lambda_{reg} \Omega(\theta - \theta_0) \right], \quad (12.2)
\]

where \(\mathcal{L}_{MLM}\) is the masked language modeling loss, and \(\Omega\) is a regularizer, such as a quadratic penalty, that discourages the new parameters \(\theta\) from drifting too far from the original parameters \(\theta_0\). In words, the model keeps much of BERT's general linguistic competence while learning the terminology, tone, and label structure of financial documents. The regularizer can be interpreted as a simple analogue of Fisher-information constraints used in continual learning.

To capture the intuition behind focusing the model on sentiment-bearing and financially salient tokens, we introduce a stylized sentiment-aware masking distribution:

\[
P(\text{mask}_i) = \frac{\exp(\beta s_i + \gamma f_i)}{\sum_{j=1}^T \exp(\beta s_j + \gamma f_j)}, \quad (12.3)
\]

where \(P(\text{mask}_i)\) is the probability that token \(i\) is selected for a masked-token prediction task, \(s_i\) is a token-level sentiment relevance score, \(f_i\) is a financial salience score, and \(\beta, \gamma\) control how strongly these two scores affect the masking probability. A high value of \(s_i\) may correspond to a word carrying positive or negative tone, while a high value of \(f_i\) may correspond to membership in a lexicon of companies, instruments, risk terms, or accounting concepts. Equation (12.3) should be read as a generic design pattern consistent with many FinBERT-like variants, rather than a literal description of any one implementation.

### Reported empirical behavior

Across multiple datasets, FinBERT-style models typically:

- Improve sentence- and document-level sentiment classification accuracy relative to generic BERT by several percentage points;
- Yield more stable performance across sectors and time periods; and
- Provide embeddings that transfer well to downstream tasks such as topic classification and event extraction.

These gains are consistent with the theory of domain adaptation and the parameter-efficient tuning methods of Chapter 3.

### 12.1.3 FinEAS: Financial sentence embeddings for sentiment

FinEAS, short for *Financial Embedding Analysis of Sentiment*, is a finance-specific sentence-embedding model for financial sentiment analysis [Gutiérrez-Fandiño et al., 2021]. It is useful to place it next to FinBERT, but it should not be described as a general-purpose generative LLM or as a retrieval-augmented generation system. Its distinctive contribution is more specific: it adapts the BERT/Sentence-BERT paradigm so that sentence-level representations become informative for financial sentiment, similarity, and semantic comparison tasks.

This distinction matters because many financial sentence signals are carried at the sentence level rather than isolated tokens. A short sentence such as “margins improved despite weaker guidance” contains a mixed evaluative signal that is not well captured by keyword polarity alone. FinEAS-style models map a financial sentence $s$ into an embedding vector

\[
z_\theta(s) = \frac{e_\theta(s)}{\|e_\theta(s)\|_2} \in \mathbb{R}^d,
\]

where $e_\theta$ is the underlying BERT or Sentence-BERT encoder and $z_\theta(s)$ is the normalized sentence representation. A supervised sentiment head then predicts either a continuous sentiment score or a sentiment class,

\[
\widehat{y} = h_\psi(z_\theta(s)),
\]

and in FinEAS that head is a single linear layer followed by a tanh, because the target is a continuous sentiment score bounded in $[-1, 1]$. The published objective is then plain least squares:

\[
\mathcal{L}_{\text{FinEAS}}(\theta, \psi) = \frac{1}{N} \sum_{i=1}^N \left( y_i - \tanh(w^\top e_\theta(s_i) + b) \right)^2,
\]

with $y_i$ the RavenPack EVENT_SENTIMENT_SCORE of headline $s_i$ [Gutiérrez-Fandiño et al., 2021]. There is no contrastive term, no metric-learning term, and no temperature. The lesson of the paper is that the *encoder* carries the contribution — starting from a sentence-level model rather than a token-level one — and the objective can stay boring.

It is nonetheless natural to ask whether an explicit similarity term would help, and the following is a reasonable extension. **It is not what FinEAS does, and it is not evaluated anywhere in the FinEAS paper; it is offered here as a design option only:**

\[
\mathcal{L}_{\text{sim-reg}}(\theta, \psi) = \underbrace{\mathbb{E}_{(s,y)}\left[\ell(y, h_\psi(z_\theta(s))\right]}_{\text{the FinEAS objective}} + \lambda_{\text{sim}} \mathbb{E}_{i,j} \left[ \left( \langle z_\theta(s_i), z_\theta(s_j) \rangle - \kappa(y_i, y_j) \right)^2 \right],
\]

where $\kappa(y_i, y_j)$ is a target similarity derived from the sentiment scores. The second term encourages sentences with similar sentiment to sit close together in embedding space, which may help downstream retrieval and clustering. Whether it helps *sentiment prediction* is an open question and would need to be established by experiment, not asserted.

In the taxonomy of this chapter, FinEAS therefore represents a fourth path for financial language specialization. BloombergGPT specializes a large decoder by pre-training on
