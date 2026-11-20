
The fp16=True flag enables mixed-precision training. In practice, mixed precision reduces the memory used by many activations and matrix operations and usually speeds up training on modern GPUs, although it does not literally halve every memory component because optimizer states, model buffers, and some numerically sensitive operations may remain in higher precision.

With these choices fixed, training can now be executed using a standard fine-tuning loop.

## Training loop

With configuration in place, training is launched through Hugging Face's Trainer API, which handles the batch loop, gradient computation, optimizer steps, and checkpoint management internally. The key custom piece is compute_metrics, a function that runs at the end of each evaluation step and computes the metrics used to select the best checkpoint. Weighted F1 is reported alongside accuracy because Financial Phrasebank has some class imbalance: weighted F1 gives proportionally more credit for correct predictions on the more frequent classes, providing a fairer summary of overall performance than macro F1 would.

```python
from transformers import Trainer
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted')
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate on test set
test_results = trainer.evaluate(test_dataset)
print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
print(f"Test F1: {test_results['eval_f1']:.4f}")
```

Once trainer.train() completes, the final call to trainer.evaluate(test_dataset) runs the model on the held-out test split for the first and only time. This separation is essential.

Under the declared sampling and temporal assumptions, it provides an out-of-sample estimate for the specified test population; it does not by itself guarantee performance after deployment or under regime change.

We can now examine how well the model performs after fine-tuning.

## Results

Table 3.3 is a *hypothetical reporting profile*. It shows the form of a before-and-after evaluation, but it is not the output of a companion run and should not be cited as empirical performance. A reproducible experiment must record the dataset and license, split rule, preprocessing, seed, checkpoint, hardware, software versions, and uncertainty around every reported metric.

| Metric | Illustrative baseline | Illustrative adapted model |
| :--- | :--- | :--- |
| Accuracy | 0.52 | 0.88 |
| F1 Score (Weighted) | 0.49 | 0.87 |
| F1 (Positive) | 0.45 | 0.89 |
| F1 (Neutral) | 0.51 | 0.84 |
| F1 (Negative) | 0.52 | 0.88 |

Table 3.3: Hypothetical reporting profile for a financial sentiment experiment

The baseline must also be defined precisely. A raw pre-trained BERT encoder coupled to a randomly initialized three-class head is not a meaningful performance baseline, because the head has not learned the label mapping. Defensible comparators include the majority-class rule, a trained linear probe on frozen representations, a documented general-domain sentiment classifier, and a fully trained but non-domain-adapted model. If a frozen probe is intended, the training protocol for that probe belongs in the experiment record.

The practical lesson is not the numerical gap in this synthetic profile. It is the reporting discipline: expose held-out aggregate and class-level metrics, calibration, confidence intervals or repeated-seed variation, class prevalence, and the economic cost of each error type. The same split and evidence standard should be used to compare full fine-tuning with PEFT; the performance difference is task- and configuration-dependent and must be measured rather than assumed.

![](img/c31705b361fdf81dee8826a2c4ee167f_118_0.png)

Figure 3.1: The adaptation ladder.

The single most expensive mistake in financial LLM projects is climbing this ladder before exhausting the rung below. Note the boundary: everything to the left of it changes what the model sees and is reversible in an afternoon; everything to the right changes what the model is, creates a new artifact with its own validation, monitoring, and inventory obligations, and — if the result is a non-generative classifier — makes SR 26-2 potentially applicable to it. Whether it is in scope is not settled by the architecture alone: it turns on the revised model definition, the institutional perimeter, the use to which the output is put, and materiality.

## 3.5 Parameter-efficient fine-tuning

While full fine-tuning can achieve strong performance, it requires updating all model parameters and can be prohibitively expensive for large models. Parameter-efficient fine-tuning (PEFT) methods update only a small fraction of parameters and can approach full fine-tuning performance on many tasks.

The conceptual shift underlying PEFT is important to appreciate. Full fine-tuning treats all parameters as equally open to change, as if the model had learned nothing of lasting value during pre-training. PEFT methods take the opposite view: a large pre-trained model already encodes rich, reusable representations of language and reasoning. Adapting it to a financial task should require only small, targeted modifications—not a wholesale rewrite. This insight, combined with the practical observation that weight updates during fine-tuning tend to lie in a low-dimensional subspace, motivates the family of methods described in this section. The result is not merely a memory-saving trick; it is a principled approach to adaptation that can reduce optimization cost and, in some low-data settings, may also reduce overfitting. Both claims require task-specific validation.

In this section, we examine how this idea translates into practical methods. We begin with the motivation for PEFT in real-world financial settings, then introduce LoRA as the primary technique for efficient adaptation, followed by QLoRA for handling large models under tight memory constraints. We also briefly consider alternative approaches and conclude with a case study and practical comparison to help guide method selection.

### 3.5.1 Motivation for PEFT

Before introducing specific PEFT methods, it is worth examining why full fine-tuning quickly becomes impractical in real-world financial settings.

Consider fine-tuning GPT-3 (175B parameters):

- Full fine-tuning requires storing: 175B parameters × 4 bytes = 700GB
- Plus gradients: 700GB
- Plus optimizer states (Adam): 1.4TB

This brings the total memory requirement to approximately 2.8TB GPU memory.

This is infeasible for most institutions. PEFT methods reduce memory requirements by 10-100×, making large model adaptation accessible.

Among the several PEFT strategies that have been proposed, LoRA has emerged as the dominant choice in both research and production. Its appeal rests on a combination of strong empirical performance, minimal inference overhead (the adapter weights can be merged back into the base model at deployment time, adding zero latency), and a clean theoretical justification. The remaining methods covered in this section—QLoRA, adapter layers, and prefix tuning—each address specific constraints or use cases, but LoRA is the natural starting point for any practitioner considering parameter-efficient adaptation.

### 3.5.2 Low-Rank Adaptation (LoRA)

LoRA [Hu et al., 2022] is the most popular PEFT method, adding trainable low-rank matrices to attention layers while keeping the original model frozen.

### Mathematical formulation

The central insight of LoRA is that the change in weights needed for task adaptation has a much lower intrinsic dimensionality than the full parameter space. Rather than adding a dense update matrix \(\Delta W\) of the same shape as the original weight \(W_0\), LoRA decomposes \(\Delta W\) into the product of two small matrices. This keeps the number of new trainable parameters proportional to the rank \(r\), which is typically set to 4-32, rather than to the full dimensions of the layer—a reduction of two to three orders of magnitude for typical transformer weight matrices.

For a pre-trained weight matrix \(W_0 \in \mathbb{R}^{d \times k}\), LoRA represents the update as:

\[
W = W_0 + \Delta W = W_0 + BA \quad (3.9)
\]

where \(B \in \mathbb{R}^{d \times r}\), \(A \in \mathbb{R}^{r \times k}\), and rank \(r \ll \min(d, k)\).

During fine-tuning, \(W_0\) remains frozen; only \(B\) and \(A\) are trained. The forward pass computes:

\[
h = W_0 x + \Delta W x = W_0 x + BAx \quad (3.10)
\]

The practical effect of this formulation is a sharp reduction in the number of parameters that must be trained:

- Original parameters: \(d \times k\)
- LoRA parameters: \(d \times r + r \times k = r(d + k)\).

For a typical attention layer with \(d = k = 4096\) and \(r = 8\):

- Original: 16,777,216 parameters
- LoRA: 65,536 parameters (0.4% of original)

Reduction: 256×

The practical implication is that a 7B-parameter model can be adapted to a financial task by training fewer than 5 million parameters—small enough to fit the adapters in memory alongside the frozen base model on a single consumer-grade GPU. At deployment time, the adapter matrices \(B\) and \(A\) can be absorbed back into \(W_0\) via \(W = W_0 + BA\), leaving the inference graph completely unchanged.
