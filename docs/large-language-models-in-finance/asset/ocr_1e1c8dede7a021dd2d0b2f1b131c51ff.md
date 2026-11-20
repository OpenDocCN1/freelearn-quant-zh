+   • Supporting quantitative strategy development

The expanding ability of newer models to handle longer contexts, perform more advanced reasoning, and integrate multimodal data (combining text, tables, and charts) further drives their widespread deployment across the financial sector.

With this understanding of how AI has evolved in finance and the current landscape of available models, we can now examine the different categories of LLMs and how to select the most appropriate model type for specific financial applications.

# ## 1.9 Understanding model types

We must now take a closer look at how these models are categorized in practice. Not all LLMs serve the same purpose as some are broad, general-purpose systems built for flexibility, while others are carefully adapted or trained from scratch for finance. Understanding these distinctions helps practitioners decide which type of model to adopt or customize for a given financial use case.

To guide model selection and provide a decision flow, therefore, we categorize LLMs in the following types:

+   1. **Domain-specific models**: Adapted to financial texts to improve handling of industry terminology and task conventions.
+   
+   2. **General-purpose models**: Broad corpora, best for open-domain tasks.
+   
+   3. **Custom fine-tuned models**: Tailored to proprietary datasets, balancing performance and control.

Understanding different model types is crucial for financial practitioners because the choice between general-purpose and domain-specific models significantly impacts both performance and cost. General-purpose models offer broad capabilities and easy deployment but may lack the specialized financial vocabulary and reasoning patterns needed for complex tasks. Domain-specific models can improve performance on matched financial tasks, but they require additional data, validation, and maintenance. This section will help you navigate these trade-offs and make informed decisions about which approach best suits your specific use case, budget, and performance requirements.

# ## 1.9.1 Domain-specific models (BloombergGPT, FinBERT and FinEAS)

Domain-specific models are designed to close the gap between general language understanding and the highly specialized nature of financial discourse. Finance introduces challenges that general-purpose models often struggle with: dense jargon, domain-specific abbreviations, subtle sentiment cues, forward-looking statements, and a heavy reliance on context drawn from regulations, filings, and market conventions.

Models such as BloombergGPT, FinBERT, and FinEAS represent different points along the spectrum of financial specialization. Some are trained at massive scale on proprietary