

## 15.3.2 金融应用的架构比较

| Feature | BERT (encoder-only) | GPT (decoder-only) | T5 (encoder-decoder) |
| :--- | :--- | :--- | :--- |
| **Architecture** | 仅使用 Transformer 的编码器堆栈。双向处理整个序列。 | 仅使用带因果掩码的解码器堆栈。自回归地处理序列。 | 同时使用编码器和解码器堆栈，类似于原始 Transformer。 |
| **Key feature** | 用于理解所有标记之间关系的双向上下文。 | 针对下一个标记预测优化的自回归生成。 | 将每项任务视为文本转换的序列到序列公式。 |
| **Pre-training objective** | 原始 BERT 使用掩码语言建模（MLM）和下一句预测（NSP）。后续 BERT 风格的编码器模型可能会省略、替换或修改 NSP。 | 因果语言建模（CLM）：基于之前上下文预测下一个标记。 | 文本到文本的去噪或跨度损坏，取决于编码器-解码器模型系列。 |
| **Financial use cases** | NLU 任务：情感分析、命名实体识别（NER）、文档分类、实体提取、合规性分流，以及像 FinEAS 这样的句子嵌入模型。 | 生成式任务：报告起草、摘要生成、场景生成、代码生成、多步推理。 | 复杂转换：文档翻译、抽象摘要、结构化问答、数据提取。 |

表 15.1：金融常见 Transformer 架构类型比较

对下一句预测的引用适用于原始 BERT 预训练配方 [Devlin et al., 2019]。后续 BERT 风格的模型可能会删除或替换 NSP，因此该目标应被理解为取决于架构系列，而不是所有仅编码器模型的通用属性。

在金融领域，模型选择取决于任务、可用数据、集成约束、延迟要求和治理需求，而非抽象的架构优越性。每种架构都为不同的金融工作流程提供特定优势。

## 15.3.3 作为句子嵌入专精的 FinEAS

FinEAS（Financial Embedding Analysis of Sentiment 的缩写）是一种基于监督微调句子嵌入的金融情感模型 [Gutiérrez-Fandiño et al., 2021]。它应被视为句子级情感和语义表示的编码器风格专精，而不是通用对话型 LLM。它与本章的相关性在于，它展示了与大型仅解码器系统不同的设计点：它主要不针对开放式生成进行优化，而是优化金融文本嵌入的几何结构，以用于情感敏感的比较、分类和下游分析。

FinEAS 风格的组件在更大的金融 LLM 系统中至少可在四个方面发挥作用。首先，它可以为新闻、文件、财报电话会议摘录或分析师评论提供领域特定的情感评分。其次，它可以为携带情感的金融句子生成用于语义搜索的嵌入。第三，它可以支持围绕企业、行业、事件或市场制度所用语言的聚类和监控。第四，它可以在代理工作流中充当专用信号，在此类工作流中，更广泛的 LLM 代理将狭义情感编码器作为一个经过审计的工具来使用，而不仅仅依赖于自由形式的生成。

主要限制同样具有启发性。FinEAS 风格的模型需要仔细控制训练分布、标签定义、时间分割和来源覆盖。当金融语言发生变化、新闻来源发生变化，或模型从研究数据移至生产数据时，其输出应重新校准。通过充足性–集中度–结构视角来看，目标不仅仅是分配情感标签，而是学习一种稳定的表示，该表示保留与情感相关的信息，集中相似的金融含义，并在嵌入空间中分隔经济上不同的陈述。因此，FinEAS 最好被理解为一个紧凑的金融情感和句子嵌入模块，它补充而非替代检索系统、生成模型和治理层。

## 15.4 实现模式

本节提供常见金融 LLM 工作流的示例性实现模式。示例为了清晰起见进行了简化，旨在展示主要步骤，而非生产就绪实现。生产部署必须添加符合机构要求的身份验证、错误处理、日志记录、可观测性、数据验证、访问控制和安全加固。

### 15.4.1 微调 FinBERT 和 FinEAS 风格的情感模型

微调将预训练模型调整为特定的金融任务，例如情感分析、文档分类、实体提取或风险分类。下面的示例使用 FinBERT 风格的模型进行金融情感分类。相同的工作流还说明了 FinEAS 风格的句子嵌入模型的位置：FinBERT 风格的模型通常用作直接分类器，而 FinEAS 风格的模型则强调句子级金融嵌入和情感几何。

此代码仅供示例。它展示了受监督微调工作流的主要步骤：加载分词器和模型、预处理数据集、配置训练参数、初始化训练器以及保存微调后的模型。生产部署应添加身份验证、日志记录、验证、可观测性、访问控制、模型卡和回滚程序。

清单 15.1：金融情感分析的受监督微调工作流

```python
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

class FinancialSentimentTrainer:
    """
    Illustrative fine-tuning pipeline for financial sentiment analysis.
    The example is simplified for exposition and should be hardened before
    production use.
    """
    def __init__(
        self,
        model_name="ProsusAI/finbert",
        dataset_name="financial_phrasebank",
        dataset_config="sentences_75agree",
        output_dir="./finbert-financial-sentiment",
        max_length=128,
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.output_dir = output_dir
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
        )

    def tokenize_batch(self, batch):
        return self.tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, average="macro"),
        }

    def load_and_prepare_data(self):
        raw = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split="train",
        )
        # NOTE ON THE SPLIT -- read this before reporting any number.
        # Financial PhraseBank ships without timestamps, so a chronological
        # split is not available for this dataset. The random split below is
        # therefore forced, and it is precisely the protocol this book
        # criticises elsewhere (Chapters 1 and 13). It is kept because it
        # demonstrates the training loop, not because it produces a
        # trustworthy result. Any score obtained this way is a smoke test of
        # the pipeline, NOT evidence of deployable performance.
        # For a model that will inform a decision, use a timestamped corpus
        # and the chronological split in temporal.py.
        # THREE-way split. The previous version made ONE split and then
        # passed dataset["test"] to the Trainer as eval_dataset with
        # load_best_model_at_end=True -- i.e. it selected the checkpoint on
        # the test set. That is selection on the test partition, and this
        # book spends a chapter explaining why it invalidates the number.
        # Validation selects; test is touched once, after selection.
        tmp = raw.train_test_split(test_size=0.2, seed=42)
        held = tmp["test"].train_test_split(test_size=0.5, seed=42)
        split = DatasetDict({
            "train": tmp["train"],
            "validation": held["train"],
        })
```

