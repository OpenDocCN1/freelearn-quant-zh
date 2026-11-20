model sees and is reversible in an afternoon; everything to the right changes what the model is, creates a new artifact with its own validation, monitoring, and inventory obligations, and — if the result is a non-generative classifier — makes SR 26-2 potentially applicable to it. Whether it is in scope is not settled by the architecture alone: it turns on the revised model definition, the institutional perimeter, the use to which the output is put, and materiality.

# 3.5 Parameter-efficient fine-tuning

While full fine-tuning can achieve strong performance, it requires updating all model parameters and can be prohibitively expensive for large models. Parameter-efficient fine-tuning (PEFT) methods update only a small fraction of parameters and can approach full fine-tuning performance on many tasks.

The conceptual shift underlying PEFT is important to appreciate. Full fine-tuning treats all parameters as equally open to change, as if the model had learned nothing of lasting value during pre-training. PEFT methods take the opposite view: a large pre-trained model already encodes rich, reusable representations of language and reasoning. Adapting it to a financial task should require only small, targeted modifications—not a wholesale rewrite. This insight, combined with the practical observation that weight updates during fine-tuning tend to lie in a low-dimensional subspace, motivates the family of methods described in this section. The result is not merely a memory-saving trick; it is a principled approach to adaptation that can reduce optimization cost and, in some low-data settings, may also reduce overfitting. Both claims require task-specific validation.

In this section, we examine how this idea translates into practical methods. We begin with the motivation for PEFT in real-world financial settings, then introduce LoRA as the primary technique for efficient adaptation, followed by QLoRA for handling large models under tight memory constraints. We also briefly consider alternative approaches and conclude with a case study and practical comparison to help guide method selection.

# 3.5.1 Motivation for PEFT

Before introducing specific PEFT methods, it is worth examining why full fine-tuning quickly becomes impractical in real-world financial settings.

Consider fine-tuning GPT-3 (175B parameters):

+   - Full fine-tuning requires storing: 175B parameters × 4 bytes = 700GB
+   - Plus gradients: 700GB
+   - Plus optimizer states (Adam): 1.4TB

This brings the total memory requirement to approximately 2.8TB GPU memory.

This is infeasible for most institutions. PEFT methods reduce memory requirements by 10-100×, making large model adaptation accessible.

Among the several PEFT strategies that have been proposed, LoRA has emerged as the dominant choice in both research and production. Its appeal rests on a combination of strong