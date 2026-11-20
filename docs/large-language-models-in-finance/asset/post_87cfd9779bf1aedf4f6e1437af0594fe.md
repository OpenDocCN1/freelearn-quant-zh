
## Data quality over quantity

For financial fine-tuning, high-quality annotated data can be more valuable than a larger volume of noisy text. A smaller set of carefully reviewed credit memos may outperform a much larger weakly labeled set when the expert labels capture the distinctions required by the deployment task. Invest in expert annotators with domain knowledge, clear labeling guidelines, and regular quality audits. Inter-annotator agreement should be measured and tracked as a first-class metric—systematic disagreement among annotators is a sign that the task definition needs to be tightened before more data is collected.

## Start simple, then scale

The temptation in financial ML projects is to reach for the largest available model and the most sophisticated adaptation method from the outset. Resist this. Begin with the smallest viable model and the simplest adaptation method. Pilot with BERT-base or a 7B-class model and PEFT (e.g., LoRA) before committing to full fine-tuning or very large models. This approach is faster to iterate, easier to debug, and often reveals data quality issues that would be obscured by a more powerful model. Scale up only if business metrics clearly justify the additional complexity and cost.

## Monitor for overfitting

Financial datasets are typically small, and small datasets create strong overfitting pressure—especially when training a model with hundreds of millions of parameters on fewer than ten thousand examples. Guard against overfitting through early stopping on a held-out validation set, regularization (weight decay, dropout), and, where appropriate, data augmentation or ensembling. Track both standard NLP metrics and task-specific business KPIs. It is common for a model to improve on validation F1 while degrading on the business metric it was designed to drive; monitoring both in parallel catches this divergence early.

## Maintain separate test sets

Never evaluate on the test set during development. Keep a strictly held-out test set for final evaluation and, for time-dependent data, enforce temporal splits to avoid lookahead bias. This discipline is particularly important in financial NLP because of the strong temporal structure of financial data: a model trained in 2022 earnings reports and evaluated on a random 20% of the same years' reports will appear more capable than it actually is when deployed in 2024. Document the split logic so that audits are straightforward.

## Version control and experiment tracking

In financial institutions, model reproducibility is not optional—it is a regulatory requirement. Treat datasets, models, and configurations as versioned artifacts. Use experiment tracking tools (e.g., MLflow, Weights & Biases) to log hyperparameters, code versions, metrics, and model checkpoints. This is essential for reproducibility and regulatory review: when a model risk management team asks why the model's output changed between versions, the answer must be retrievable from the experiment log.

For the examples in this chapter, the safest convention is to distinguish three artifact classes: *available* artifacts that are included in the book repository, *referenced* public artifacts that readers must download from the original provider, and *illustrative* artifacts that show the structure of production system without exposing proprietary data or internal infrastructure. Every case-study folder should state which class applies to its datasets, code snippets, and configuration files.

## Plan for model updates

Financial markets and regulations evolve continuously, and a model that performs well today may degrade as the language of filings, market conditions, or regulatory frameworks shifts. Define, in advance, the monitoring signals, retraining cadence, and decommissioning criteria for each model. Make roll-back procedures explicit and test them periodically. A model that cannot be safely rolled back is an operational risk, not just a technical artifact.

## 3.10 Audit-grade fine-tuning: From objective to runnable experiment

A high-quality fine-tuning experiment is not defined by the number of epochs or by the size of the base model. It is defined by whether the experiment answers a decision-time question without leakage, whether the objective matches the business task, and whether the result can be reproduced from logged artifacts. In finance, the most common hidden defect is temporal leakage: a model appears accurate because it indirectly saw documents, labels, embeddings, prompts, or benchmark examples that would not have existed at the time of the decision.

**Definition 3.10.1** (Decision-time admissible artifact). Fix a decision time $t$. An artifact $Z$ is *decision-time admissible* at $t$ if it is measurable with respect to $\mathcal{F}_t$, the information available at or before $t$. Datasets, labels, embeddings, prompt examples, retrieval chunks, hyperparameter choices selected using validation data, and human corrections are all artifacts. A fine-tuning run for decisions at $t$ is admissible only if every artifact used by the training, validation, retrieval, prompting, and evaluation pipeline is decision-time admissible.

**Proposition 3.10.1** (Temporal split validity). Let $(X_i, Y_i, T_i)_{i=1}^n$ be financial examples with timestamps $T_i$, and let a model $f_\theta$ be trained using only artifacts measurable with respect to $\mathcal{F}_{\tau_{\text{train}}}$. Suppose the test set contains only examples with $T_i > \tau_{\text{train}} + \epsilon$, where $\epsilon \ge 0$ is an embargo period, and suppose no prompt, retrieval chunk, label, normalization statistic, or model-selection decision uses information from the test period. Then the reported test loss is an admissible out-of-sample evaluation statistic for the pipeline under the same decision-time information regime. If, in addition, the test losses obey an appropriate law of large numbers for the deployment regime—for example, under stationarity/ergodicity or a declared regime-conditional sampling model—the empirical test loss consistently estimates the corresponding expected decision-time loss. If any artifact from the test period is used in training or model selection, the reported loss evaluates a different task with a larger information set.

_Proof._ Under the stated measurability condition, the trained pipeline is a function of $\mathcal{F}_{\tau_{\text{train}}}$ and is therefore fixed before observing the test-period outcomes. The test loss is then computed on observations whose future labels and documents were not available to the pipeline when it was trained or selected. This proves admissibility of the evaluation design and absence of direct test-period lookahead. Consistency for an expected risk is a separate probabilistic statement: it requires a law of large numbers for the sequence of realized test losses under the target deployment regime. The embargo reduces near-boundary contamination, but it does not by itself remove revised filings, delayed labels, duplicate news, vendor backfills, or incorrect availability timestamps. Those channels require separate revision-chain, label-availability, deduplication, and vendor-timestamp controls. If a test-period artifact enters training or selection, the fitted pipeline becomes a function of a strictly larger sigma-algebra than $\mathcal{F}_{\tau_{\text{train}}}$; the evaluation then conditions on information that would not have been available at deployment time.

The following minimal splitter implements the proposition as code. It is intentionally simple: production systems should extend it with issuer-level grouping, document de-duplication, label-lag rules, and artifact hashing.

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List, Sequence, Tuple

@dataclass(frozen=True)
class TimedExample:
    id: str
    timestamp: datetime
    text: str
    label: int

@dataclass(frozen=True)
class TemporalSplit:
    train: List[TimedExample]
    validation: List[TimedExample]
    test: List[TimedExample]
    cutoff_train: datetime
    cutoff_validation: datetime
    embargo: timedelta

def make_temporal_split(
    examples: Sequence[TimedExample],
    train_end: datetime,
    validation_end: datetime,
    embargo: timedelta = timedelta(days=0),
) -> TemporalSplit:
    """Create a chronological split with an embargo after train/validation.

No item dated after train_end may enter the training set. No item inside
the embargo window enters the next split. The function raises errors for
unsorted or overlapping timestamps because silent leakage is worse than a
