It is ordinary supervised regression, and the model works because of the encoder it starts from, not the objective it optimizes. Second, modelling sentiment as a *continuous* quantity rather than a three-way class is itself the departure from FinBERT, and it is the more faithful representation of what a trading desk actually consumes: not the label *negative*, but a number that can be aggregated, decayed, and sized.

The reported evaluation uses a large RavenPack US news sample—not FiQA or the Financial PhraseBank—and reports mean squared error rather than accuracy or F1. With the backbone frozen, FinEAS attains MSE 0.0556 against 0.2124 for BERT and 0.2108 for a bidirectional LSTM; with full fine-tuning it edges FinBERT (0.0044 versus 0.0050 on the six-month sample) [Gutiérrez-Fandiño et al., 2021]. We return to this model in Chapter 13, and not to praise it: its evaluation protocol contains exactly the defect this book devotes a chapter to warning against, and the cost of that defect is measurable in the paper's own numbers.

# ## 1.9.2 General-purpose architectures powering financial AI

While domain-specific models like FinBERT, FinEAS, and BloombergGPT are tailored to financial data, most real-world applications in finance still rely on powerful general-purpose LLMs as their foundation. These models are developed by leading research labs and open-source communities and provide flexible architectures on which specialized financial systems are built.

Several major model families form the backbone of this ecosystem:

+   - **OpenAI's GPT series**: Successors to GPT-4 continue to be highly influential, offering strong performance across a wide range of text generation and understanding tasks. They are frequently used as base models for fine-tuning in financial contexts [Brown et al., 2020], [OpenAI, 2022].

+   - **Google's Gemini family**: The Gemini models, known for their native multimodality and reasoning capabilities, are increasingly explored for financial applications that require integrating text with charts, tables, and other data formats [Gemini Team, Google, 2023].

+   - **Meta's Llama series**: Llama 3 is a representative open-weight foundation-model family used as a base for specialized financial tools and research [AI @ Meta, 2024].

+   - **Anthropic's Claude series**: Models like Claude 3 are recognized for their large context windows, strong reasoning abilities, and focus on safety and reliability, making them suitable for analyzing lengthy financial reports and complex queries [Anthropic, 2024].

+   - **Mistral AI models**: Mistral continues to release powerful open models known for their efficiency and performance, often competing closely with proprietary counterparts. These are widely adopted and adapted within the financial sector [Mistral AI, 2023].

+   - **DeepSeek models**: Models from DeepSeek AI, especially those excelling in code generation, are valuable for quantitative finance, algorithmic trading strategy development, and financial data analysis automation.

General-purpose LLMs serve as versatile foundations for numerous financial applications. Their extensive pre-training on diverse internet-scale data allows them to understand language structure, context, and a degree of world knowledge. Fine-tuning these models, even with