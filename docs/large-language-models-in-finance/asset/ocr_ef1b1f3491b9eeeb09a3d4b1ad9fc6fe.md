micro-batch size and the sequence length. A useful way to read the commonly used range of 16–32 examples per GPU is therefore not as a theoretical constant, but as a practical starting point for encoder-style fine-tuning with moderate sequence lengths, such as BERT-class models with inputs up to 512 tokens. The range comes from a simple engineering compromise: choose the largest per-device micro-batch that fits in memory without out-of-memory errors; then adjust gradient accumulation so that the optimizer sees a sufficiently stable effective batch.

Formally, if \( B_{dev} \) is the per-device micro-batch, \( G \) is the number of GPUs, and \( A \) is the number of gradient-accumulation steps, then the optimizer behaves as if it had received

\( B_{eff} = B_{dev} G A \quad (3.7) \)

examples per update. Thus, a single-GPU run with \( B_{dev} = 16 \) and \( A = 4 \) has an effective batch of 64, while a four-GPU run with \( B_{dev} = 16 \) and \( A = 2 \) has an effective batch of 128. This is the reason the text recommends 16–32 per GPU as an initial micro-batch range rather than a final effective-batch prescription.

```
effective_batch_size = batch_size_per_gpu * num_gpus * grad_accum_steps
```

The same rule should be scaled down for larger decoder-only models, longer context windows, or QLoRA settings. In those cases, the 4-bit base model reduces parameter memory, but activation memory can still dominate, so the feasible per-device micro-batch may be only 1–4 examples. Gradient accumulation is then used to recover an effective batch in the 32–128 range without exceeding GPU memory.

The **number of epochs** is another important consideration. Financial datasets are often small, which means each training epoch cycles through the data quickly, and overfitting can set in fast. The usual 3–5 epoch range for encoder-based classification is best understood through the number of optimizer updates:

\( U = E \left\lceil \frac{N_{train}}{B_{eff}} \right\rceil, \quad (3.8) \)

where \( E \) is the number of epochs and \( N_{train} \) is the number of training examples. For example, with \( N_{train} \approx 4,000 \) and \( B_{eff} = 32 \), three to five epochs produce roughly 375–625 optimizer updates, which is usually enough to adapt the classification head and upper transformer layers without repeatedly memorizing a small dataset. Generative SFT may need fewer epochs (1–3) because each example contributes many token-level loss terms. In practice, the epoch count should be treated as an early-stopping upper bound rather than a fixed target: monitor validation performance after each epoch and halt as soon as improvement stalls rather than training to the preset maximum.

The final parameter to consider is **weight decay**, which controls how strongly the model is regularized during training. In AdamW the decay is *decoupled*: it is applied directly to the parameters as a multiplicative shrinkage \( \theta \leftarrow (1 - \alpha\lambda)\theta \) after the adaptive step, and is *not* an L2 penalty added to the loss. The distinction is not pedantic — an L2 term added to the gradient is rescaled by Adam’s adaptive denominator, so the two methods behave differently