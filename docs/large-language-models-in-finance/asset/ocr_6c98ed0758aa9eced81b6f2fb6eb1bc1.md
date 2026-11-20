```
# Standardize currency symbols while preserving signed numbers.
text = re.sub(r'([$€£¥])\s*([+-]?\d)', r'\1 \2', text)

# Handle percentages without destroying decimals or signs: -8.5% -> -8.5 %.
text = re.sub(r'([+-]?\d+(?:\.\d+)?)\s*%', r'\1 %', text)

# Remove only control characters. Do not strip punctuation wholesale: it may
# encode dates (2024-03-31), times (14:30), decimals (3.25), ratios,
# credit ratings (A-), tickers (BRK.B), or signed values (-8 %, +12 bps).
text = re.sub(r'[\x00-\x1f\x7f]', '', text)

return re.sub(r'\s+', ' ', text).strip()
```

This example is deliberately conservative. Production cleaning should be validated with a small regression suite containing dates, times, decimals, ticker symbols, credit ratings, legal references, and signed values. A regex that improves general text cleanliness can still damage financial meaning; for instance, removing colons can corrupt times, removing hyphens can corrupt dates and ratings, and removing plus/minus signs can reverse the interpretation of growth or spread changes.

Once the text is clean, a second pass of **quality filtering** removes examples that would degrade training. In financial NLP, this step is particularly consequential: a mislabeled credit opinion or a duplicated earnings sentence with contradictory annotations teaches the model the wrong mapping and is difficult to recover from later. The four checks below address the most common failure modes:

+   - Remove duplicates and near-duplicates (MinHash, SimHash)
+   - Filter low-quality content (length filters, readability scores)
+   - Validate annotations (inter-annotator agreement, expert review)
+   - Balance class distributions (oversample minorities, undersample majorities)

With clean, filtered data in hand, the final preparation step is constructing the train, validation, and test splits. How these **data splits** are performed is especially consequential in finance: a random split applied to time-stamped financial data can leak future information into the training set, producing inflated evaluation metrics that do not reflect real deployment performance.

In most cases, practitioners allocate approximately 70–80% of the data for training, 10–15% for validation, and 10–15% for testing, with the test set kept strictly separate from model development. For time-series financial data, use temporal splits to prevent lookahead bias.