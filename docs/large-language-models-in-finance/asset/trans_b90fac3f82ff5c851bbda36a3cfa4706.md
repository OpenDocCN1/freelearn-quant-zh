

## 4.4.2 用于嵌入的对比学习

现代嵌入模型采用对比学习目标 [Gao et al., 2021, Chen et al., 2020]。给定一组训练样本 $(q_i, d_i^+, \{d_i^{-,j}\}_{j=1}^m)$，其中 $q_i$ 是查询，$d_i^+$ 是相关文档，$\{d_i^{-,j}\}$ 是负面（无关）文档：

InfoNCE（噪声对比估计）损失定义为：

```
$$\mathcal{L} = - \log \frac{\exp(\text{sim}(f(q_i), f(d_i^+))/\tau)}{\exp(\text{sim}(f(q_i), f(d_i^+))/\tau) + \sum_{j=1}^m \exp(\text{sim}(f(q_i), f(d_i^{-,j}))/\tau)}$$
```

其中 $\tau$ 是控制分布尖锐程度的温度参数。

温度参数 $\tau$ 起着至关重要的作用：较小的 $\tau$ 会产生更尖锐的分布，而较大的 $\tau$ 会平滑分布。典型值范围从 $\tau = 0.05$ 到 $\tau = 0.1$。

## 4.4.3 金融嵌入模型

嵌入模型的选择直接影响检索准确性，尤其是在处理领域特定的金融术语和结构化披露时。因此，我们在此区分通用模型和领域特定模型。

在实践中，从业者通常从通用模型开始，例如：

- OpenAI text-embedding-3-large (3072-dim)
- Cohere embed-english-v3.0 (1024-dim)
- sentence-transformers/all-mpnet-base-v2 (768-dim)

然而，随着检索需求变得更加专业，尤其是在处理金融申报文件、研究报告或内部文档时，领域特定模型例如：

- 经过检索训练的句子编码器（例如 E5、GTE 和 BGE 家族，或 sentence-transformers 模型），进一步在从目标文档类型中抽取的对比对金融语料库上进行微调
- 在专有申报文件、备忘录或转录文本上微调的自定义编码器，按检索指标而非分类准确性进行评估

由于该领域对此十分粗心，有必要做两点澄清。**分类器不是嵌入模型。** ProsusAI/finbert 是一个金融*情感分类器*；其隐藏状态可以池化为向量，但它从未使用检索目标进行过训练，在最近邻搜索上将不如专门训练的编码器。**BloombergGPT 不是嵌入服务。** 它是专有生成模型 [Wu et al., 2023b]；没有公开发布的检查点，也没有可部署的嵌入端点，因此它无法出现在读者可以实际构建的检索栈中。命名一个无法获取的模型并非设计建议。在实践中，许多系统从通用检索编码器开始，并随着需求变得更加专业而逐步加入领域特定的微调。

## 4.4.4 Matryoshka 嵌入

从系统设计角度来看，能够使用同一嵌入的低维版本，使得在检索质量与计算和存储成本之间取得平衡变得更容易，尤其是在大规模金融系统中。最近的进展使得 Matryoshka 表征学习（MRL）[Kusupati et al., 2022] 成为可能，其中单个模型产生可变维度的嵌入：

$f(x) = [f_1(x), f_2(x), ..., f_d(x)]$ (4.9)

我们可以截断到任何 \(d' < d\)，同时保持语义属性。这允许：

- 用于精度的高维嵌入（例如，1024 维）
- 用于效率的低维嵌入（例如，256 维）
- 基于计算预算的动态调整

在实践中，可以通过一次生成完整嵌入并选择性地将其截断为低维表示来实现，如下所示。

```
import numpy as np

# Full 1024-dimensional embedding
full_embedding = embed_model.encode(text) # shape: (1024,)

# Truncate to 256 dimensions for faster search
truncated = full_embedding[:256] # shape: (256,)

# Both preserve semantic properties with appropriate scaling
```

## 4.5 向量数据库和索引

计算嵌入后，我们需要能够大规模存储它们并在毫秒级检索查询嵌入最近邻的基础设施。这就是向量数据库的作用。在金融 RAG 系统中，索引通常存储来自申报文件、转录文本、研究报告和内部文档的数百万个分块嵌入，并且在考虑重排序、多查询检索和元数据过滤后，每个用户请求会被查询几十到几百次。因此，索引算法的选择直接影响延迟、内存占用、召回率，并最终影响系统能够产生的答案质量。

在本节中，我们讨论现代向量索引背后的核心算法思想、在生产中实现大部分这些索引的 FAISS 库，以及结合语义和词汇信号的混合搜索。

## 4.5.1 近似最近邻搜索

高维中的精确最近邻搜索具有 O(Nd) 复杂度，其中 N 是语料库大小，d 是嵌入维度。因此，对于大规模语料库（N > 10^6），精确搜索在计算上变得昂贵，我们改而依赖于以少量准确性换取显著速度提升的近似方法。

### 倒文件索引（IVF）

一种常用方法是 **倒文件索引（IVF）** 算法，它将嵌入空间 [Jégou et al., 2011] 分区为簇，并在查询时将搜索限制在这些簇的子集中。

算法 6：IVF 索引构建和搜索

| 步骤 | 描述 |
|------|-------------|
| 1 | 训练阶段： |
| 2 | 使用 k-means 将嵌入聚类为 m 个质心：{c1, ..., cm} |
| 3 | 将每个嵌入 ei 分配到最近的质心簇 |
| 4 | 查询阶段： |
| 5 | 给定查询嵌入 q |
| 6 | 找到 q 最近的 n_probe 个质心 |
| 7 | 仅在这些 n_probe 个簇中搜索嵌入 |
| 8 | 返回 top-k 结果 |

在实践中，这种方法通过仅关注嵌入空间中最相关的区域，减少了查询时所需的比较次数。

在计算复杂度方面，IVF 方法具有以下特征：

- 训练：O(Nmd · iterations)
- 查询：O(n_probe · (N/m) · d)
- 当 n_probe << m 时，我们实现了显著的加速

### 分层可导航小世界（HNSW）

另一种广泛使用的方法是 **分层可导航小世界（HNSW）** [Malkov and Yashunin, 2020]，它构建具有跳表结构的基于图的索引，其特性如下：

- 索引组织为多层图，高层逐渐变得更稀疏
- 搜索从顶层开始，向下导航至低层
- 这导致大约 O(log N) 的搜索复杂度，且召回率高
- 与 IVF 相比，权衡是更高的内存使用量

### 产品量化

**产品量化（PQ）** [Jégou et al., 2011] 通过关注内存效率而非仅搜索速度来解决另一个挑战。它通过将 $d$ 维空间分解为 $s$ 个维度为 $d/s$ 的子空间来压缩嵌入：

$\mathbb{R}^d = \mathbb{R}^{d/s} \times \mathbb{R}^{d/s} \times \cdots \times \mathbb{R}^{d/s}$

然后，每个子空间使用 $k$ 个质心独立进行量化。这使得嵌入可以以压缩形式表示，仅需 $s \cdot \log_2(k)$ 位，而标准 float32 需要 $32d$ 位。

虽然这些索引方法定义了相似性搜索的执行方式，但我们依赖于专业库来在大规模上高效实现它们。

## 4.5.2 Facebook AI 相似性搜索（FAISS）

**Facebook AI 相似性搜索（FAISS）** [Johnson et al., 2019, Douze et al., 2024] 提供了广泛向量索引和相似性搜索方法的优化实现，使其成为构建生产级检索系统的常见选择。构建和查询索引的典型工作流如下所示：

```
import faiss
import numpy as np

# Index parameters
d = 768 # embedding dimension
nlist = 100 # number of clusters for IVF
m = 8 # number of subquantizers for PQ
nbits = 8 # bits per subquantizer

# Create IVF-PQ index
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

# Train on sample embeddings
embeddings = np.random.randn(10000, d).astype('float32')
index.train(embeddings)
index.add(embeddings)

# Search
query = np.random.randn(1, d).astype('float32')
k = 5
distances, indices = index.search(query, k)
```

