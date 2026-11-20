
## Fine-Tuning LLMs for Finance

This chapter provides a practitioner's guide to adapting pre-trained LLMs to concrete financial tasks such as sentiment analysis, information extraction, question answering, document classification, and advisory systems. While Chapter 2 focused on architectures and training mechanics, here we answer a different question: _given a base model, how do we turn it into a reliable financial model under realistic data, compute, and governance constraints?_

We proceed in five steps. We begin with the theoretical foundations of fine-tuning, viewing it as a regularized optimization problem over a pre-trained parameter initialization. We then discuss _full_ fine-tuning, followed by _parameter-efficient_ methods such as LoRA and QLoRA that make large-model adaptation feasible for most institutions. Next, we connect these techniques to supervised fine-tuning recipes for common financial tasks and to Reinforcement Learning from Human Feedback (RLHF) for aligning models with institutional preferences. While we briefly touch on alignment and evaluation, the primary focus of this chapter is the adaptation of models themselves—how they are trained, modified, and specialized for financial tasks.

Throughout the chapter, we emphasize design trade-offs: data requirements versus compute, flexibility versus stability, and raw performance versus governance and operational risk. The goal is not only to present algorithms, but to give you templates that can be instantiated in different financial environments, from small quantitative teams to large regulated institutions.

In a nutshell, the following topics will be covered in this chapter:

- Adapting general LLMs to financial tasks
- Foundations of fine-tuning
- Full fine-tuning methodology
- Parameter-efficient fine-tuning (PEFT)
- Supervised fine-tuning
- Reinforcement learning from human feedback (RLHF)
- Evaluation and validation
- Case studies: How financial LLMs are actually built

### 3.1 Production-quality fine-tuning contract

The rest of this chapter develops many adaptation methods, but the most important engineering point can be stated before the details: a financial fine-tuning experiment is not defined by a model name and a training set. It is defined by a decision date, a permissible information set, a task loss, a calibration protocol, and an approval boundary. This section states that contract explicitly so that the later code and case studies have a clear standard.

Let each observation be $z_i = (x_i, y_i, \tau_i, g_i)$, where $x_i$ is the financial text or structured prompt, $y_i$ is the target, $\tau_i$ is the earliest time at which the observation and label were available, and $g_i$ is a grouping variable such as issuer, portfolio, counterparty, product, or desk. A production-faithful split chooses cut dates $T_{\text{train}} < T_{\text{valid}} < T_{\text{test}}$ and sets

- $\mathcal{I}_{\text{train}} = \{i : \tau_i \le T_{\text{train}}\}$
- $\mathcal{I}_{\text{valid}} = \{i : T_{\text{train}} < \tau_i \le T_{\text{valid}}\}$
- $\mathcal{I}_{\text{test}} = \{i : T_{\text{valid}} < \tau_i \le T_{\text{test}}\}$

If labels become available later than inputs, the label availability time must be used. For example, an earnings-call transcript may be available on the call date, but an analyst label that uses a later price reaction is not available until after the reaction window closes.

The training objective should therefore be read as a time-respecting, weighted empirical risk:

```
\(\widehat{\theta} \in \arg \min_{\theta \in \Theta} \frac{1}{|\mathcal{I}_{\text{train}}|} \sum_{i \in \mathcal{I}_{\text{train}}} w_i \ell(f_\theta(x_i), y_i) + \lambda \Omega(\theta; \theta_0), \quad (3.1)\)
```

where $w_i$ encodes cost asymmetry or class imbalance and $\Omega$ penalizes destructive movement away from the base model. The objective is deliberately modest: it does not claim that fine-tuning discovers financial truth. It only states that the adapted model should reduce a task-specific loss under the information that would actually have been available at the time.

**Proposition 3.1.1** (Temporal split eliminates direct lookahead). Suppose every training artifact used by a fine-tuning procedure is a measurable function of observations with availability time at most $T_{\text{train}}$. Suppose every validation decision at time $t \in (T_{\text{train}}, T_{\text{valid}}]$ uses only artifacts with availability time at most $t$, and every test decision at time $t \in (T_{\text{valid}}, T_{\text{test}}]$ uses only artifacts with availability time at most $t$. Then no direct future observation enters the model input, prompt examples, retrieval index, label, hyperparameter choice, or evaluation record for any evaluated decision.

Proof. Each evaluated object is constructed from artifacts whose availability time is bounded above by the relevant decision time. Because the construction operations are measurable functions of those artifacts, they cannot introduce observations outside the sigma-algebra generated by information available at or before the decision time. Hyperparameter choices are made on the validation interval and then frozen before the test interval, so the test set is not used in model selection. This proves absence of direct lookahead by construction. The proposition does not rule out more subtle leakage, such as duplicated text across periods or vendor backfills with incorrect timestamps; those must be detected by the leakage audits described in Chapter 15.

Calibration is equally important. For a classifier that outputs probabilities \(p_i\), expected calibration error can be estimated by binning predictions into confidence bins \(B_m\) and computing

```
\(\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} |\text{acc}(B_m) - \text{conf}(B_m)|.\) (3.2)
```

In finance, calibration is often more important than raw accuracy. A sentiment model that is correct 80% of the time but cannot distinguish 55% confidence from 95% confidence is hard to use in position sizing, escalation, or client-facing disclosure. Calibration should therefore be reported alongside accuracy, F1, drawdown-sensitive downstream metrics, and human-review outcomes [Guo et al., 2017].

The implementation pattern below is intentionally small, but it encodes the production rule that matters most: split by availability time, validate required fields, and fail when the split is empty or temporally inconsistent.

### Listing 3.1: Auditable temporal split for financial fine-tuning

```
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence

@dataclass(frozen=True)
class LabeledExample:
    example_id: str
    text: str
    label: str
    available_at: datetime
    group: str | None = None

@dataclass(frozen=True)
class TemporalSplit:
    train: tuple[LabeledExample, ...]
    valid: tuple[LabeledExample, ...]
    test: tuple[LabeledExample, ...]

def make_temporal_split(
    examples: Iterable[LabeledExample],
    train_end: datetime,
    valid_end: datetime,
) -> TemporalSplit:
    if train_end >= valid_end:
        raise ValueError("train_end must be strictly before valid_end")
    ordered = sorted(examples, key=lambda z: (z.available_at, z.example_id))
    if any(not z.text.strip() for z in ordered):
        raise ValueError("empty text example found")
    train = tuple(z for z in ordered if z.available_at <= train_end)
    valid = tuple(z for z in ordered if train_end < z.available_at <= valid_end)
    test = tuple(z for z in ordered if z.available_at > valid_end)
    if not train or not valid or not test:
        raise ValueError("temporal split produced an empty partition")
    return TemporalSplit(train=train, valid=valid, test=test)

def assert_no_lookahead(split: TemporalSplit) -> None:
    max_train = max(z.available_at for z in split.train)
    min_valid = min(z.available_at for z in split.valid)
    max_valid = max(z.available_at for z in split.valid)
    min_test = min(z.available_at for z in split.test)
    if max_train >= min_valid or max_valid >= min_test:
        raise AssertionError("temporal ordering violated")
```

This code is not a complete training script; it is the boundary that every complete training script should respect. Once this boundary is stable, the choice between full fine-tuning, LoRA, QLoRA, instruction tuning, or preference tuning becomes an engineering decision rather than a source of hidden evaluation bias.

### 3.2 Adapting general LLMs to financial tasks

As discussed in earlier chapters, financial language and analysis introduce several requirements that general-purpose models do not fully capture out of the box. Financial documents rely on specialized terminology, complex numerical relationships, and temporal comparisons across reporting periods, all within a highly regulated environment. Moreover, while pre-trained models such as GPT-4, Claude, and LLaMA exhibit strong general linguistic capabilities, they often lack the specialized knowledge, precision, and domain-specific reasoning required for financial analysis, risk assessment, and regulatory compliance [Wu et al., 2023b].

Fine-tuning helps bridge this gap. By continuing to train on domain-specific data, the model's parameters gradually adjust to the patterns and conventions found in financial texts. This process leverages the extensive linguistic knowledge encoded during pre-training while specializing the model for tasks such as financial sentiment analysis, earnings call summarization, credit risk assessment, fraud detection, and regulatory document processing [Araci, 2019, Wu et al., 2023b, Yang et al., 2023].

Overall, even though modern LLMs are powerful general language models, they typically need further adaptation before they can be trusted for financial analysis or decision support. Fine-tuning provides the primary mechanism for performing that adaptation.

#### 3.2.1 The fine-tuning paradigm

Fine-tuning can be formally understood as an optimization problem. Let \(\theta_0\) represent the parameters of a pre-trained language model, and let \(\mathcal{D}_{\text{finance}} = \{(x_i, y_i)\}_{i=1}^N\) be a dataset of financial text-label pairs. The fine-tuning objective seeks to find parameters \(\theta^*\) that minimize a task-specific loss:
