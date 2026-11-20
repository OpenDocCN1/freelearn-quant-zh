

| Task | BloombergGPT | GPT-NeoX | OPT-66B | BLOOM-176B |
|---|---|---|---|---|
| Financial PhraseBank (FPB) | 51.07 | 44.64 | 48.67 | 50.25 |
| FiQA sentiment (FiQA SA) | 75.07 | 50.59 | 51.60 | 53.12 |
| Headline | 82.20 | 73.22 | 79.41 | 76.51 |
| Financial NER | 60.82 | 60.98 | 57.49 | 55.56 |
| ConvFinQA | 43.41 | 30.06 | 27.88 | 36.31 |
| Five-task average | 62.51 | 51.90 | 53.01 | 54.35 |

Table 3.5. Public financial benchmark scores reported in BloombergGPT Table 8; higher is better.

extraction. The central lesson is therefore conditional. Domain-aware corpus design can materially improve matched financial tasks, while the size and location of the gain depend on the task, comparator, and evaluation protocol. For most institutions, a from-scratch run at this scale is neither necessary nor economical; continued pre-training, retrieval, and parameter-efficient adaptation are more accessible routes.

## 3.9.2 FinBERT and the BERT fine-tuning lineage

The BERT fine-tuning case study in the _Case study: Fine-tuning for financial sentiment analysis_ section walked through the complete technical workflow for a single run. This case study broadens the lens to examine _what happened in the research community_ when the same technique was applied systematically and publicly, producing models that have become standard infrastructure in financial NLP.

FinBERT [Araci, 2019] was among the early studies to show that domain-adaptive training and financial sentiment fine-tuning could improve a general-purpose BERT model on matched tasks, while also releasing a reusable public checkpoint. The model became an influential baseline for subsequent financial NLP work.

### What FinBERT added beyond a standard BERT fine-tune

The key contribution relative to a vanilla fine-tune was twofold: first, a domain-adaptive pre-training step on financial corpora _before_ the supervised fine-tuning stage; and second, systematic ablation experiments that isolated the contribution of each step. This allowed subsequent researchers to understand which part of the performance gain came from domain-adaptive pre-training and which came from the supervised fine-tuning signal—a distinction that matters for practitioners deciding where to allocate labeling budget.

### The BERT fine-tuning lineage

FinBERT's release triggered a productive lineage of financial encoder models, each targeting a specific gap:

- FinBERT for sentiment analysis adapts BERT to financial sentiment and demonstrates the value of domain-specific language-model adaptation before supervised classification [Araci, 2019].

- FinBERT for financial communications develops a separate finance-pretrained encoder and evaluates it across communication-focused tasks, reinforcing the importance of matching pre-training data to the target document domain [Yang et al., 2020].

- FinEAS shifts the emphasis from token classification to sentence embeddings: a Sentence-BERT-derived representation feeding a bounded regression head trained with mean-squared error against continuous sentiment scores. It is a supervised scalar objective, not a contrastive one, and it illustrates a complementary route to retrieval, clustering, and similarity-based financial NLP [Gutiérrez-Fandiño et al., 2021].

### Production adoption and practical lessons

The broad use of FinBERT and related models in research and practitioner prototypes offers several practical lessons. First, domain-adaptive pre-training on unlabeled financial corpora can improve downstream performance on matched tasks, although the magnitude depends on the corpus, labeling protocol, model family, and evaluation split. Second, a public checkpoint may be further adapted on carefully governed internal data that better matches an institution's document types. Third, the incremental value of domain adaptation for larger foundation models remains task- and data-dependent and should be measured rather than inferred from model scale alone.

The FinBERT lineage collectively argues that encoder-based financial models, despite their smaller scale, remain highly competitive for classification, NER, and sentiment tasks where the input fits within 512 tokens. The computational efficiency, smaller operational footprint, and deployment simplicity of encoder models can make them preferable to large generative models for bounded classification, NER, and sentiment workflows, while instruction-following LLMs remain useful for more open-ended tasks.

If FinBERT highlights gains in classification and representation learning, FinQA shifts the focus to a different challenge altogether: numerical reasoning over financial data.

## 3.9.3 FinQA: Numerical reasoning over financial data

The FinQA benchmark [Chen et al., 2021] tests a capability that neither BloombergGPT nor FinBERT was primarily designed for: multi-step numerical reasoning over structured financial data. Answering a FinQA question correctly requires the model to locate the right values in a table or passage, select the required operations, execute them in the correct order, and express the result with the expected scale and units.

This task combines three distinct sources of error:

- evidence retrieval and number extraction from tables and narrative text;
- program construction, including operation and argument selection; and
- deterministic execution, scaling, and answer formatting.

FinQA therefore reports both *execution accuracy*, which tests the final numerical answer, and *program accuracy*, which tests the generated operation sequence against the reference program. Table 3.6 reports the corrected FinQA test-set results for FinQANet and the original human baselines, together with the later APOLLO ensemble result [Chen et al., 2021, Sun et al., 2024].

| System | Execution accuracy (%) | Program accuracy (%) |
|---|---|---|
| General crowd | 50.68 | 48.17 |
| FinQANet (RoBERTa-large) | 61.24 | 58.86 |
| APOLLO (ensemble) | 71.07 | 68.94 |
| Human expert | 91.16 | 87.49 |

Table 3.6: Corrected FinQA numerical-reasoning results and a later APOLLO result

For FinQANet, the execution–program gap is 61.24 – 58.86 = 2.38 percentage points: about one benchmark item in 42 overall, or approximately 2.38/61.24 ≈ 3.9% of items with a correct final execution. This gap is a diagnostic, not a literal probability of invalid reasoning, because program accuracy uses exact reference matching and can penalize an execution-equivalent alternative program. It nevertheless identifies a material set of apparently correct answers whose trace requires inspection before the result can support an audited workflow.

The benchmark's own maintenance history reinforces that lesson. On 4 May 2022, the official FinQA repository documented a label-leak defect in the `table_row_to_text` preprocessing routine and published revised FinQANet scores [Chen et al., 2022a]. This does not invalidate the benchmark; it demonstrates why a financial evaluation package must version preprocessing code, data, labels, and result tables rather than treating a benchmark name as a sufficient specification.

The production lesson is to expose the generated program or calculation trace, execute it in a deterministic engine, reconcile extracted values with source evidence, and record units and scaling. A numerically correct answer with an unsupported or non-reproducible trace should be escalated rather than accepted on the final number alone.

## 3.9.4 What these case studies actually teach us

The three case studies above, along with the methods developed throughout this chapter, converge on a set of practical principles that apply across financial fine-tuning projects regardless of scale. These are not rigid rules but heuristics that have proven robust across a wide range of institutional settings, from small quant teams working with a single labeled dataset to large banks managing dozens of fine-tuned models across multiple desks.

