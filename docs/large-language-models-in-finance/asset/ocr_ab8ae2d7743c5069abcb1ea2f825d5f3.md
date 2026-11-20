Learning rates assume AdamW with linear warm-up and cosine decay; actual values depend on batch size, optimizer, and hardware.

This high-level cookbook will be revisited throughout the chapter as we introduce theoretical foundations and practical considerations for each family of methods.

# 3.3 Foundations of fine-tuning

Understanding the theoretical underpinnings of fine-tuning provides insights into when and why it works, and how to optimize the process for financial applications.

# 3.3.1 Transfer learning theory

Fine-tuning can be understood as a form of transfer learning, in which knowledge acquired in one domain (general text) is adapted to a new domain (finance). The effectiveness of this transfer depends largely on the relationship between the source and target domains.

Transfer learning works best when the two domains have a similar underlying structure. Financial text, for example, retains the same basic linguistic foundations as general language—grammar, syntax, and common expressions—while introducing specialized terminology and domain-specific semantics. Pre-training on large general corpora, therefore, provides a strong linguistic base that fine-tuning can refine for financial contexts.

The similarity of tasks also plays an important role. Models trained on next-token prediction during pre-training learn broad language modeling capabilities that transfer naturally to downstream tasks such as financial text generation, summarization, and question answering. Because these tasks rely on similar underlying abilities (understanding context, predicting plausible continuations, and organizing information) the knowledge acquired during pre-training remains highly useful.

At a representational level, pre-training produces hierarchical internal representations within the model. Lower layers tend to capture general linguistic patterns, such as syntax and frequently occurring words, while higher layers encode more task-specific relationships. During fine-tuning, adaptation often occurs more strongly in the upper layers, allowing the model to incorporate financial concepts while preserving the broader linguistic knowledge learned during pre-training [Yosinski et al., 2014].

# 3.3.2 Catastrophic forgetting and regularization

A key challenge in fine-tuning is *catastrophic forgetting*: the tendency of a model to lose general knowledge when it is aggressively adapted to a narrow task. If fine-tuning proceeds too quickly or relies on a small financial dataset, the model may overfit to the training examples and degrade its broader reasoning capabilities.

One common mitigation strategy is **weight regularization**, which discourages large deviations from the original pre-trained parameters. In practice, this is implemented by penalizing parameter changes during training: