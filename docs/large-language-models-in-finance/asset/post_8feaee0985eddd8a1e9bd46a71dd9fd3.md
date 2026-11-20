
word "challenging." This contextual understanding marks a substantial advance over early rule-based systems.

As we'll see in subsequent sections, the mathematical principles underlying these models (from probability distributions to attention mechanisms) form the building blocks for the specialized financial applications that are transforming the industry today.

## 1.4 Statistical models vs. deep learning-based models

Understanding the transition from classical statistical language models to modern deep learning architectures is essential for anyone deploying LLMs in finance. Both paradigms aim to model the probability of word sequences, but they differ radically in how they represent context and how far back in a sequence they can "remember".

Classical statistical models (such as n-gram models) rely on counts of short word sequences. They work reasonably well when patterns are local and the vocabulary is limited, but they break down in the presence of long-range dependencies, data sparsity, and nuanced domain terminology. Deep learning models, however, replace explicit counts with learned representations (embeddings and neural networks) that can encode much richer semantic and syntactic structure.

In a financial setting, this distinction is not academic. Determining whether a sentence like "Guidance was revised downward despite strong top-line growth" is positive or negative for a particular issuer requires understanding the interaction between multiple clauses, prior disclosures, and numerical context. As we will see in the following subsections, statistical models struggle with this type of reasoning, while modern neural architectures are explicitly designed to capture it. To understand why, we begin with a closer look at how classical statistical models operate and where they fall short.

### 1.4.1 Statistical language models and their limitations

Statistical language models dominated the field of natural language processing for several decades, and understanding their principles provides essential context for appreciating modern advances. These models use empirical counts and smoothing techniques to estimate the probability of words or sequences appearing in text.

The foundation of statistical language models lies in the chain rule of probability. For a sentence like "The Federal Reserve raised interest rates," a statistical model would decompose this into conditional probabilities: the probability of "The" at the beginning, then "Federal" given "The," then "Reserve" given "The Federal," and so on. This decomposition allows the model to estimate the likelihood of entire sentences by breaking them down into manageable pieces.

However, this approach quickly becomes impractical when dealing with the vast and specialized vocabulary found in financial reports. Consider that financial texts might contain tens of thousands of unique words. To model all possible three-word sequences (trigrams), we would need to track billions of combinations, most of which would never appear in any training corpus. This sparsity problem becomes particularly acute in financial applications where specialized terminology and proper nouns (company names, ticker symbols, executive names) create an ever-expanding vocabulary.

To address these challenges, statistical models employ various smoothing techniques:

- Laplace smoothing: Adds a small count to every possible word combination, ensuring no probability is exactly zero
- Kneser-Ney smoothing: Uses more sophisticated methods to redistribute probability mass from seen to unseen events
- Back-off models: Use lower-order statistics when higher-order ones aren't available
- Interpolation methods: Combine probabilities from different order n-grams with learned weights

While clever, these techniques struggle with fundamental limitations. Statistical models cannot capture long-range dependencies, such as how a mention of "inflation" on page 1 of a Federal Reserve report profoundly influences the interpretation of "growth" on page 10. They also fail to understand semantic similarity; to a statistical model, "acquire" and "purchase" are as different as "acquire" and "banana," despite their semantic relationship being crucial in merger and acquisition contexts.

A closer look at a concrete statistical method highlights these limitations more clearly.

One common example of a statistical approach is the n-gram model. In an n-gram model, the joint probability of a sequence of words $w_1, w_2, ..., w_T$ is approximated using the chain rule with a Markov assumption of order $n-1$:

$$ P(w_1, ..., w_T) = \prod_{t=1}^T P(w_t|w_{t-n+1}, ..., w_{t-1}) $$ (1.1)

This method assumes a Markov property of order $n-1$, significantly reducing computational complexity by ignoring long-range dependencies. The practical implications become clear when considering real financial documents:

- Earnings calls: CEO opening remarks might be elaborated by CFO 20 minutes later—impossible for n-grams to connect
- Annual reports: Risk factors on page 20 relate to mitigation strategies on page 100—beyond n-gram context
- Regulatory filings: Definitions in early sections determine interpretation throughout—requires full document understanding

These foundational constraints are what ultimately limit the usefulness of classical statistical models in finance. This motivated the shift toward deep learning models, which approach language in a fundamentally different way.

Case study — Earnings call assistant: An n-gram model tasked with summarizing an earnings call would struggle immediately. The CEO’s opening remark that “revenue growth was driven by strong enterprise demand” might be elaborated or qualified by the CFO twenty minutes later—a dependency spanning thousands of tokens. A bigram model would never connect the two. As we will see, Transformers were specifically designed to handle exactly this kind of long-range reasoning.

### 1.4.2 Deep learning models and the emergence of transformers

Deep learning introduced recurrent neural networks (RNNs) and long short-term memory units (LSTMs) to capture sequential dependencies. These models represented a paradigm shift from counting co-occurrences to learning distributed representations. Instead of treating words as discrete symbols, neural models embed them in continuous vector spaces where semantic relationships are encoded as geometric relationships.

The fundamental innovation of RNNs was the introduction of recurrence—the ability to maintain a hidden state that gets updated as the model processes each word in sequence. This hidden state theoretically allows the model to maintain information from the beginning of a document all the way to the end. For financial applications, this meant the possibility of understanding documents where early context influences later interpretation.

However, RNNs suffered from several critical limitations:

- **Vanishing gradient problem:** Gradients become exponentially small when backpropagated through many time steps
- **Exploding gradient problem:** Conversely, gradients can grow exponentially, causing training instability
- **Sequential processing bottleneck:** Each word must be processed sequentially, preventing parallelization
- **Limited effective context:** although a recurrent state can in principle carry information indefinitely, in practice the usable range is short and is architecture- and task-dependent; empirical studies of vanilla RNNs and LSTMs typically find effective ranges on the order of tens to low hundreds of tokens, well short of a financial filing

LSTMs addressed some of these issues through sophisticated gating mechanisms:

- **Input gates:** Control what new information to store in the cell state
- **Forget gates:** Determine what information to discard from previous cell state
- **Output gates:** Control what information to pass to the next time step
- **Cell state:** Maintains information across time steps with linear connections

Even with these improvements, early neural models still struggled with long sequences and training efficiency. The advent of Transformers, however, overcame these hurdles through attention mechanisms, offering parallel computation and improved handling of distant
