

同样的测试期间，即使每次单独运行在时间上都是干净的，最终结果仍然是过拟合。这就是提示词过拟合，即LLM版本的策略数据挖掘。

补救措施是研究核算。保留开发集、验证集和最终未触碰的测试集。记录所有提示词变体和模型选择决策。在选择提示词或适配器时使用嵌套验证。报告试验次数或在适当的地方使用多重检验校正。最重要的是，不要让最终测试集成为创意循环的一部分。

### 13.4.10 幻觉数值信号与虚假精度

LLM可以围绕微弱或不存在的信号生成流畅的解释。模型可能会从噪声中推断出因果故事，发明一个比率，读错表格，或报告数据不支持的精确预期回报。在金融预测中，虚假精度是危险的，因为微小的数值差异可能驱动巨大的仓位变化。

生产系统应将语言生成与数值计算分开。让LLM提出假设、检索上下文或解释模型输出，但让经过审计的数值代码计算收益率、波动率、相关性、回撤和投资组合权重。预测报告应包含置信区间、校准诊断和明确的不确定性。如果模型无法引用数字背后的数据和计算，该数字不应进入投资流程。

### 13.4.11 泄漏感知的评估协议

一个健壮的LLM金融预测和回测协议应包括以下控制措施：

- 首先，为每个预测起点定义信息集 $\mathcal{F}_t$，包括数据、文档、工具、提示词和记忆。
- 其次，根据可用性时间戳而非经济周期时间戳构建所有特征。
- 第三，对重叠标签使用带禁忌期的滚动起点划分。
- 第四，冻结或为所有检索存储和工具API建立时间索引。
- 第五，除非明确建模在线学习，否则重置记忆。
- 第六，针对简单基准和强非LLM基准进行评估。
- 第七，报告交易成本后的统计指标和经济指标。
- 第八，针对预训练和检索知识运行污染探测。
- 第九，预留一个最终未触碰的测试周期。
- 第十，归档提示词、代码、数据哈希、模型版本和工具输出，以便回测可以被审计。

算法18总结了该协议。

```
Algorithm 18 Leakage-aware LLM backtest
1: Input: forecast origins {tj }, horizon H, point-in-time data store, time-indexed retrieval store, model M
2: for each forecast origin tj do
3: Freeze numerical data, documents, tools, prompts, and memory to Ftj
4: Construct features and context using availability timestamps only
5: Reset memory or load only state that would have existed at tj
6: Generate forecast distribution p̃j(ytj +1:tj +H)
7: Convert forecast to positions using pre-specified risk and cost model
8: Store prompt, retrieved documents, data hashes, model version, and forecast
9: end for
10: Reveal outcomes only after all forecasts are locked
11: Compute statistical metrics, economic metrics after costs, and paired significance tests
12: Run contamination probes and document any suspected overlap
13: Output: audited performance report and leakage diagnostics
```

### 从业者实现说明

如果每个组件都建立时间索引，该协议可以通过普通工程组件来强制执行。实用堆栈包含五个层：

- 首先，一个时点特征存储（例如Feast风格或内部特征存储）执行时点连接，并拒绝可用性时间戳晚于预测起点的特征。
- 其次，一个仅追加湖仓或不可变数据版本化层（例如Delta Lake、Apache Iceberg、lakeFS或哈希链设计）存储原始供应商数据包和修正版本，而不覆盖历史记录。
- 第三，一个带时间索引的检索存储重建向量索引，以便检索到的文档相对于 $\mathcal{F}_{t_j}$ 是可度量的。
- 第四，模型和提示词注册表记录每次运行所使用的基础模型、适配器、温度、系统提示词、工具权限和后处理规则。
- 第五，审计账本存储预测起点、数据哈希、检索文档ID、模型版本、提示词版本、记忆策略、预测和锁定时间戳。精确的供应商选择不如不变量重要：LLM消耗的每个对象都必须能够按历史决策时的状态重建。

### 最小实现模式

下面的Python风格草图说明了算法18背后的实际控制流程。它故意保持中立，不依赖特定库；重要的特性是每个预测起点处的时间回溯边界。

```
for origin in forecast_origins:
    X = feature_store.asof(entity_ids, features, as_of=origin)
    docs = retrieval_index.snapshot(as_of=origin).search(query(origin))

    if online_learning:
        memory = memory_store.load_state(as_of=origin)
    else:
        memory = None

    forecast = model.predict(
        features=X,
        documents=docs,
        memory=memory,
        prompt_version="locked_v3",
        tools="historical_only",
    )

    audit_log.write(
        forecast_origin=origin,
        data_hash=X.hash(),
        document_ids=[d.id for d in docs],
        model_version=model.version,
        prompt_version="locked_v3",
        memory_policy="online" if online_learning else "reset",
        forecast=forecast,
        locked_before_outcome=True,
    )

# Outcomes are joined only after every forecast has been locked.
performance = evaluate_locked_forecasts(audit_log, realized_outcomes)
```

主要教训在于方法论。时间序列LLM可能成为金融预测系统的有价值组件，但其评估必须比普通机器学习评估更严格，而非更宽松。模型越强大，隐藏泄漏的表面积就越大。鲁棒性来自于将整个LLM系统——基础模型、提示词、工具、检索、记忆、数据和研究工作流——视为回测的一部分。

![](img/c31705b361fdf81dee8826a2c4ee167f_438_0.png)

### 图13.8：LLM回测中的泄漏通道

经典文献恰好只监管其中一个——特征矩阵——而一个LLM系统至少有五个。每条虚线箭头都是预测起点之后的信息到达其之前决策的路径。最后一个在经典回测中没有类比，也没有技术修复：针对测试分数进行迭代的研究人员是一个有薪水的泄漏通道。

## 13.5 时点文本信号：审计完整评分工件

文本记录上的时间戳并不代表读取它的模型的时间戳。2008年的标题是历史性的，但用于对其评分的工具可能已使用2012年或2026年的信息进行训练、校准、提示或检索增强。表面上的历史信号因此混合了两个信息集：文本到达时可用的内容，以及后来嵌入测量装置中的内容。回测可以按时间顺序排列，但仍然无法交易。前一部分定性地列举了这些泄漏通道，并给出了关闭它们的操作协议。本节将该协议细化为衡量标准：完整评分工件的正式可采性条件、明确说明干净流水线保证的命题、关于每个违规程度有多大的控制证据，以及审计员可以逐个对象应用的工件级检查表。

### 13.5.1 完整评分工件及其版本

文本信号是一个组合，

```
X_j \longrightarrow \mathcal{A}_{v(j)} \longrightarrow s_j \longrightarrow S_{a,d} \longrightarrow z_{a,d} \longrightarrow \pi_d, \quad (13.1)
```

其中 $X_j$ 是可用文章，$\mathcal{A}_{v(j)}$ 是在版本 v(j) 部署的评分工件，$s_j$ 是文章得分，$S_{a,d}$ 是资产-日汇总，$z_{a,d}$ 是变换信号，$\pi_d$ 是在收盘时执行的决策。任何依赖数据的箭头都可能引入未来信息。对于稀疏词典，脆弱对象显然是一个单词权重表。对于现代系统，它要大得多。在构建版本 t 时，定义

```
\mathcal{A}_t = (\text{tokenizer}_t, \text{vocabulary}_t, \text{base}_t, \text{head}_t, \text{calibrator}_t, \text{prompt}_t, \text{retrieval}_t, \text{code}_t, \text{config}_t, \xi_t), \quad (13.2)
```

其中 $\xi_t$ 记录重放所需的所有随机状态，未使用的组件为空。对于词典，词汇表和权重承载实质内容；对于检索增强的LLM，每个条目都可能很重要。

可采性不是训练数据截止点。设 $\rho(\mathcal{A}_t)$ 为组装工件构建、发布或授权、部署并可供操作使用的最早时间。在实践中，

```
\rho(\mathcal{A}_t) = \max\{c_t^{\text{data}}, c_t^{\text{tune}}, c_t^{\text{build}}, c_t^{\text{release}}, c_t^{\text{deploy}}, c_t^{\text{avail}}\}, \quad (13.3)
```

这六个时间戳分别是源数据截止、微调截止、构建完成、发布或授权、部署到评分路径，以及对将要使用该得分的流程的操作可用性。最后两个最常被忽略，也是最容易出问题的：一个模型在三月发布但直到六月才接入研究流水线，那么在四月它就不是可采用的工件，无论它的权重知道什么。一个模型在2010年的数据上训练，但直到2012年才首次发布，它不是一个可采用的2011工件。

