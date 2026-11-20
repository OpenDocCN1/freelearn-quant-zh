| Source type | Examples and notes |
| :--- | :--- |
| Public filings and disclosures | SEC/ESMA filings (10-K, 10-Q, 8-K, prospectuses), annual reports, earnings call transcripts, regulatory releases. Often large, but require careful temporal splits to avoid lookahead bias. |
| Annotated benchmarks | Financial Phrasebank, FinQA, ConvFinQA, finance-specific NER, and summarization datasets. Small but high-quality; ideal for evaluation and targeted fine-tuning. |
| News and research | Licensed news feeds (Bloomberg, Reuters), analyst reports, house research. Rich signal but subject to strict licensing and retention constraints. |
| Proprietary institutional data | Internal risk reports, credit memos, client communications, support tickets. Highest domain relevance, but with the strongest privacy and compliance constraints. |
| Synthetic and augmented data | LLM-generated variants, paraphrases, back-translations, and templated scenarios. Useful to increase robustness, but must be validated and clearly tagged as synthetic. |

Table 3.2: Typical data sources for financial fine-tuning

Once the data sources have been identified, the next step is to transform this raw, heterogeneous data into a form suitable for training through careful preprocessing and quality control.

# Data preprocessing and quality control

Raw financial text arrives in a wide variety of formats (HTML-stripped filings, OCR'd PDFs, newsfeeds with inconsistent encoding) and requires careful normalization before it can be fed to a tokenizer. The cleaning operations that matter most for financial text differ from those in general NLP. Currency symbols, percentage signs, and decimal separators must be preserved and standardized rather than stripped, because numerical context is often the primary signal the model needs to learn from. Removing a dollar sign before a figure, or collapsing whitespace around a percentage, can cause the tokenizer to merge tokens in ways that obscure meaning.

To make these considerations concrete, the following function illustrates a simple preprocessing routine designed for financial text. In practice, this text cleaning step is less about aggressively cleaning the input but about preserving key numerical and symbolic information while standardizing formatting in a way that is compatible with downstream tokenization:

```
import re
import unicodedata

def clean_financial_text(text):
    # Normalize Unicode and whitespace without changing financial semantics.
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'\s+', ' ', text)
```