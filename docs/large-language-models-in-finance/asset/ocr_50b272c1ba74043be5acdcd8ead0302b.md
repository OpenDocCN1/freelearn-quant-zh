Training duration reflects the cumulative weight of these constraints. The following orders of magnitude should be interpreted as planning estimates, not universal constants. They assume modern data-center GPUs, high-bandwidth interconnects, mixed precision, distributed data loading, and non-trivial but imperfect MFU:

+   - **Small continued-pretraining or fine-tuning runs (7B–13B)**: days to weeks on a small multi-GPU setup, depending on dataset size and context length.
+   - **Domain-scale pre-training runs (30B–70B)**: weeks to months on dedicated GPU clusters with tensor, pipeline, and/or ZeRO-style sharding.
+   - **BloombergGPT-scale reference point**: the published run used 512 A100 40GB GPUs, ZeRO stage 3, model parallelism, activation checkpointing, and a 1.3M GPU-hour compute budget, so the reported timing cannot be interpreted without those infrastructure assumptions [Wu et al., 2023b].

Ambition, therefore, has to be grounded in infrastructure and budget. Choices about parameter count, corpus size, and architectural complexity are not just technical design decisions, as they carry real financial and strategic consequences. The cost of pre-training influences not only how a model is built, but whether building it from scratch is a sensible path in the first place.

# ## 2.4 Understanding prompt engineering in finance

Up to this point, we have focused on how financial language models are built: architectures, tokenization strategies, training objectives, and infrastructure requirements. Yet even the most powerful model does not operate in isolation. Its practical value depends on how it is *instructed*. This section examines how prompt engineering translates financial expertise and regulatory constraints into structured instructions that LLMs can reliably execute. We will explore how prompts can be designed to produce compliant and professionally accurate outputs in financial contexts.

Prompt engineering, in this sense, is not cosmetic phrasing. It is the operational interface between human judgment and model capability. Financial analysis demands precision and contextual awareness. Effective prompts must therefore encode not only the task itself, but also the institutional environment in which the output will be used.

We begin with the foundations of prompt design, focusing on how to structure instructions so they are clear and repeatable. We then move into reasoning-enhancement techniques that strengthen analytical depth and reliability. Finally, we examine financial domain-specific strategies, where prompting must adapt to regulatory frameworks, risk discipline, quantitative validation, and evolving market realities.

# ### 2.4.1 Prompt design

Effective prompt engineering requires a systematic approach that considers the unique characteristics of financial data expected in financial services. Hence, let’s understand the design patterns that make prompts reliable in such regulated, high-stakes environments.