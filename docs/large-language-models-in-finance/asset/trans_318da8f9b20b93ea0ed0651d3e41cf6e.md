

## 7.3.2 欺诈检测与案件调查

欺诈检测系统通常使用：

- 规则（例如，大额或异常交易）[Bolton and Hand, 2002]，
- 交易图上的异常检测模型 [Akoglu et al., 2015]，
- 设备、会话和行为信号，受隐私和相称性控制约束。

这些系统生成的警报必须由人工分析师进行调查。LLM 可显著提高 *调查效率*，方法如下：

- 总结交易历史及相关元数据，
- 用自然语言解释某模式为何可能可疑或无害 [Hilal et al., 2022]，
- 建议检查的其他数据点。

### 实施草案：欺诈警报总结

代码清单 7.5：使用 LLM 总结欺诈警报

```
# Illustrative template. Not production code.

def summarize_alert(alert, llm):
    """
    alert: dict with keys such as:
    - "transactions": list of dicts with date, amount, currency, merchant, country, etc.
    - "customer_profile": dict with basic KYC info
    - "rules_triggered": list of rule identifiers
    llm: wrapper with llm.generate(prompt: str) -> str
    """
    tx_lines = []
    for tx in alert["transactions"]:
        tx_lines.append(
            f"{tx['date']} {tx['amount']} {tx['currency']} "
            f"at {tx['merchant']} ({tx['country']})"
        )
    tx_block = "\n".join(tx_lines)

    prompt = f"""
You are a fraud analyst at a bank.

Customer profile:
{alert['customer_profile']}

Rules triggered:
{alert['rules_triggered']}

Transactions:
{tx_block}

Task:
1. Summarize the transaction pattern in 3-4 bullet points.
2. Explain why this pattern might be suspicious or benign.
3. Suggest what additional information an analyst should check.
"""
    return llm.generate(prompt)

# Example usage (conceptual):
# summary = summarize_alert(alert_json, llm_client)
```

此类功能成为分析师工作站的一部分，其中：

- 每个警报都配有结构化摘要和理由，
- 分析师可以修改或更正摘要，
- 反馈会被记录，并可用于优化提示词或模型 [Lebichot et al., 2020]。

## 7.3.3 KYC 和 AML 工作流

了解你的客户（KYC）和反洗钱（AML）流程依赖于：

- 身份证明文件，
- 公司注册信息和实益所有权数据，
- 地址验证，
- 交易监控系统，以及
- 不良媒体和制裁筛查 [Weber et al., 2018]。

LLM 通过以下方式支持这些流程：

- 将文档解析并规范化为结构化表单，
- 综合客户档案（包括复杂的公司架构），
- 以标准化方式总结不良媒体命中项，以及
- 使用清晰、一致的语言起草可疑活动报告（SAR）。

重要的是，LLM 在以下严格约束下运行：

- 它们不决定开户或去风险结果；它们支持分析师；
- 它们不替代制裁筛查引擎或基于规则的 AML 系统；它们解释并给警报排序；
- 所有输出均需接受人工审查和监管要求 [European Union, 2015]。

## 7.3.4 运营工作流和文档

除信贷和 AML 之外，后台运营会产生和处理大量非结构化 artifact，例如：

- 确认单、对账单和对账文件，
- 内部政策文件和程序，
- 异常报告和事件日志，以及
- 运营、前台和交易对手之间的电子邮件线程。

LLM 可以：

- 读取异常队列并按语义相似性聚类问题，
- 基于历史事件推断可能的根本原因，以及
- 生成带有相关上下文的草拟回复或内部工单。

该架构与其他用例相似：基于内部文档的 RAG [Guu et al., 2020]、用于提出草稿的 LLM，以及人工审查和执行。

## 7.4 四个经过审计的实施案例

前面的章节勾勒了应用场景。本节通过四个端到端案例，使设计标准具体化。每个案例均围绕相同的七个问题组织：支持什么决策、哪些信息可被采纳、模型可以做什么、确定性系统必须做什么、哪个人类负责结果、系统如何评估、以及什么导致系统停止。

### 7.4.1 案例 1：财报电话会议即时研究

**支持的决策。** 股票分析师需要一份当日研究报告，将管理层的新指引与上一季度、共识预期以及最新提交的财务报表进行对比。该系统支持研究优级；在未经分析师批准的情况下，它不会下单交易或发布推荐。

**可采纳的输入。** 每个输入都包含可用性时间戳，而不仅仅是期间日期。演讲稿、网络直播幻灯片、提交文件、先前指引、共识快照、价格数据和内部研究限制均必须满足 \(a_j \le t\)，其中 \(t\) 为分析师的决策时间。稍后收到的修订版演讲稿不得悄然进入历史评估。

#### 流程。

1. 导入演讲稿和幻灯片，保留发言轮次、表格、页码引用和源哈希。
2. 使用发行人、财政期间和可用性时间过滤器，检索之前的电话会议、当前和先前的提交文件以及授权的共识快照。
3. 将报告数值和指引范围提取到类型化模式中。
4. 在确定性代码中重新计算增长率、利润率变化、指引中点和共识差值。
5. 生成声明级摘要，其中每个实质性陈述均链接到演讲稿段落、提交文件表格或计算对象。
6. 在报告显示或共享之前，应用限制名单、重大非公开信息和分发政策检查。

紧凑的输出协议如下：

$o = (\text{summary, guidance changes, reconciled metrics, citations, uncertainties, policy flags})$

语言模型可以解释管理层听起来更谨慎，但应由单独的分类器或审查者分配任何情感分数，且在没有样本外实验的情况下，不应将该分数视为因果交易信号。

**评估。** 该研究系统在按时间顺序留出的电话会议上进行测试。指标包括提取精确率/召回率、数值对账率、声明级忠实度、无证据声明率、首个可用草稿所需时间、分析师更正率和决策影响。有用的生产力比较记录分析师活跃分钟数，而不仅仅是系统延迟。

**停止条件。** 当演讲稿不完整、共识快照缺少时间点、数字未通过对账、实质性声明缺乏证据或分发限制未解决时，工作流会拒绝或升级。

### 7.4.2 案例 2：KYC/AML 调查协助

**支持的决策。** 调查员必须决定警报是否可以关闭、是否需要更多信息，或是否应升级为可疑活动审查。LLM 准备一个与证据相关联的案件叙述；它不作出提交决定。

**可采纳的输入。** 案件可能包括客户和实益所有人记录、交易历史、警报规则、制裁和政治敏感人物结果、不良媒体文件、先前案件和分析师备注。访问受目的限制且因敏感管辖权而异。搜索结果必须保留来源、检索时间、语言、身份匹配置信度和重复关系。

#### 流程。

1. 使用确定性标识符和经审查的匹配阈值解析实体；不要让 LLM 仅根据名称相似性合并客户或实益所有人。
2. 在模型之外构建交易时间线并计算金额、速度、地理分散、同行偏差和资金流动摘要。
3. 仅检索授权的不良媒体和政策证据。
4. 要求模型将案件组织为观察、合理解释、矛盾、缺失信息和建议的调查步骤。

