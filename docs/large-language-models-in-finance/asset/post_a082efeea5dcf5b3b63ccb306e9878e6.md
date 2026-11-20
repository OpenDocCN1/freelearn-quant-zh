
at the same nominal decay rate, which is the entire reason AdamW exists (Chapter 2). Note also what decay does and does not do: it shrinks parameters toward *zero*, not toward their pre-trained values. Anchoring to the pre-trained weights is a different regularizer (an explicit \(\|\theta - \theta_{\text{pre}}\|^2\) penalty, sometimes called L2-SP) and must be added deliberately if that is what is wanted. For financial fine-tuning, values of 0.01 to 0.1 are typical. Larger values provide stronger regularization and are preferable when the training set is very small or highly homogeneous; smaller values allow the model more freedom to specialize. Once these choices are in place, the next question is how they are applied during training, which is captured by the fine-tuning procedure itself.

## Training procedure

The full fine-tuning loop follows a standard structure, but several details merit attention. At each batch the task loss is computed and back-propagated; AdamW's decoupled weight decay is then applied to the parameters by the optimizer, outside the loss. After each epoch the model is evaluated on the validation set, and the best-performing checkpoint is saved. Early stopping monitors validation loss over a patience window and halts training if performance has not improved for several epochs, which is the primary guard against overfitting on the small financial datasets that are typical in this domain. Algorithm 5 summarizes this procedure:

```
Algorithm 5 Full fine-tuning algorithm
Input: Pre-trained model \(\theta_0\), training data \(\mathcal{D}_{train}\), validation data \(\mathcal{D}_{val}\)
Initialize: \(\theta \leftarrow \theta_0\), best\_metric \(\leftarrow -\infty\), patience \(\leftarrow 0\)
for epoch = 1 to max\_epochs do
    for batch \((x, y)\) in \(\mathcal{D}_{train}\) do
        Compute loss: \(\mathcal{L} = \mathcal{L}_{task}(f_\theta(x), y) + \lambda\|\theta - \theta_0\|^2\)
        Compute gradients: \(g \leftarrow \nabla_\theta \mathcal{L}\)
        Update parameters: \(\theta \leftarrow \theta - \eta g\)
    end for
    Evaluate on \(\mathcal{D}_{val}\), compute val\_metric (e.g. F1 for classification, ROUGE for generation)
    if val\_metric > best\_metric then
        best\_metric \(\leftarrow\) val\_metric
        Save checkpoint
        patience \(\leftarrow 0\)
    else
        patience \(\leftarrow\) patience + 1
        if patience \(\ge\) max\_patience then
            break {Early stopping}
        end if
    end if
end for
Return: Best checkpoint
```

Together, these configuration choices and training steps define the mechanics of full fine-tuning in practice. To make this more concrete, we now walk through an end-to-end example that illustrates how these elements come together in a real financial task.

### 3.4.3 Case study: Fine-tuning for financial sentiment analysis

This case study walks through the complete process of fine-tuning BERT for financial sentiment classification, from data preparation through to evaluating the trained model. It is deliberately kept end-to-end so that each step can be understood in the context of the whole workflow, not in isolation. The task is simple enough to be tractable on a single GPU yet realistic enough to illustrate the practical decisions—data quality, hyperparameter choices, evaluation design—that determine whether fine-tuning succeeds or fails.

### Task definition

The goal is to classify a sentence of financial news or commentary into one of three sentiment categories. This is a natural starting point for financial NLP because sentiment is both commercially valuable (it feeds directly into news-driven trading signals, risk monitoring, and market surveillance systems) and a well-defined enough problem to have clean labeled benchmarks.

The three classes reflect the way financial analysts actually characterize market-moving language:

- Positive: Bullish indicators, good news, upbeat tone
- Neutral: Factual statements without clear sentiment
- Negative: Bearish indicators, bad news, pessimistic tone

Crucially, financial sentiment differs from general sentiment in ways that trip up general-purpose models. A sentence like “Net sales rose to EUR 112 million from EUR 105 million” contains no overtly positive words, yet an analyst would immediately classify it as positive. The model needs to learn that numerical comparisons implying growth are positive, that terms like guidance, margin, and basis points carry directional meaning in context, and that hedging language (“expects to,” “subject to market conditions”) tends toward neutral. This domain-specific mapping is precisely what fine-tuning is designed to teach. With the task clearly defined, the next step is to look at the data used to teach the model this mapping.

### Dataset

Financial Phrasebank [Malo et al., 2014] is a widely used benchmark of 4,840 sentences drawn from financial news articles and annotated by financial professionals. Its relatively small size makes it a realistic stand-in for the kind of labeled datasets that most institutions can assemble internally: large enough to fine-tune a pre-trained encoder, small enough that every labeling decision matters.

The annotation scheme is intentionally strict. Labels reflect how a financially literate reader would interpret the sentence, not a general reader. This distinction is apparent in the examples below:

| Text | Label |
|------|-------|
| The company said it would cut 200 jobs | Negative |
| Net sales rose to EUR 112 million from EUR 105 million | Positive |
| The company maintains its previous guidance | Neutral |

The first example would likely be classified as negative by a general sentiment model too—job cuts are universally negative news. The second is more revealing: the sentence contains no sentiment words at all, only a numerical comparison. A general-purpose model with no exposure to financial conventions will often classify this as neutral. The third example illustrates how subdued, hedging language (“maintains previous guidance” rather than “raises guidance”) signals caution even when the words seem innocuous.

The dataset is split so that the model never sees test examples during training or hyperparameter selection:

Split: 3,872 train / 484 validation / 484 test

With the data in place, we now turn to the model that will learn from it.

### Model architecture

BERT (Bidirectional Encoder Representations from Transformers) is a natural choice for sentence-level classification tasks like this one. Unlike decoder-only models that generate text left-to-right, BERT’s encoder reads the entire sentence bidirectionally, making its [CLS] token representation a rich summary of the whole input—exactly what is needed to produce a classification decision. The bert-base-uncased variant (110M parameters) is large enough to learn nuanced financial language patterns but small enough to fine-tune on a single GPU in minutes.

The architecture change required for classification is minimal: a linear classification head is appended on top of the [CLS] token output, projecting from BERT’s 768-dimensional hidden state down to three output logits (one per sentiment class). All of this is handled automatically by BertForSequenceClassification:

```
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3,  # Positive, Neutral, Negative
    problem_type="single_label_classification"
)
```

At this point, the model’s BERT layers carry weights from pre-training on general English text, while the classification head is randomly initialized. Fine-tuning will simultaneously refine both: the BERT weights shift toward financial language patterns, and the head learns to map the resulting representations to sentiment labels.

With the model defined, the next step is to specify how it will be trained in practice.

### Training configuration

The training configuration translates the theoretical choices discussed earlier in this chapter into concrete values. Several decisions deserve brief explanation. The learning rate of 2e-5 is deliberately conservative—roughly ten times smaller than the rates used during BERT's original pre-training—to avoid overwriting the general linguistic knowledge the model already holds. The per-device training batch size is set to 16, which is the lower end of the 16-32 micro-batch range discussed above and is a safe default for bert-base-uncased with 512-token inputs on a single mixed-precision GPU. The evaluation batch size is set to 32 because no gradients or optimizer states are stored during evaluation, so memory pressure is lower. With 3,872 training examples, a single GPU, and no gradient accumulation, four epochs imply approximately

4 × [ 3,872 / 16 ] = 968

optimizer updates. This is a small enough training run to avoid excessive memorization, while still providing enough updates for the classification head and upper BERT layers to adapt.

Instead of specifying an absolute warm-up length, the code uses warmup_ratio=0.1; for this example that corresponds to roughly 97 warm-up steps. This is safer than a fixed 500-step warm-up, which would consume more than half of the updates in such a small dataset. Setting weight_decay=0.01 applies decoupled weight decay to limit overfitting on the relatively small training set. Finally, evaluating at the end of each epoch and loading the best checkpoint by F1 (rather than loss) ensures the saved model maximizes the metric that matters most in imbalanced classification settings.

```
from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir='./financial-sentiment-bert',
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=100,
    fp16=True, # Mixed precision training
)
