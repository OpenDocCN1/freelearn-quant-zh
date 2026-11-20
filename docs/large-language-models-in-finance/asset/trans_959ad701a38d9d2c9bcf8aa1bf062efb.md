

对于此文档类，提取架构应明确表示：

- 分层优先级、余额、票息和法律最终到期日；
- 每个瀑布步骤及其适用条件；
- 覆盖测试定义、分子、分母、阈值和补救机制；
- 再投资期规则和资产准入标准；
- 集中度限制、抵押品质量测试和管理人费用；以及
- 提取条款与任何消耗它们的现金流或压力测试模型之间的链接。

生产工作流应在数据进入风险引擎之前，针对已知示例测试生成的结构化表示，并对提取的余额、百分比和瀑布优先级进行对账。验证后，结构化条款可用于契约监控、异常管理以及合同风险敞口的投资组合层面分析 [Chen et al., 2024b]。

## 8.1.2 SEC 和 FINRA 材料的自动化合规检查

上市公司和受监管中介机构在详细的申报、披露、沟通、审批和记录保存要求下运营。LLM 可以帮助识别可能缺失的披露、不一致的叙述、未经证实的主张或风险语言，但不应做出最终的法律或合规判定。适当的输出是供合格人工审查的文档化发现。

### 分层验证设计

当任务分配给最适合它们的组件时，合规审查效果最好：

- **结构性检查** 使用确定性规则验证必需章节、签名、日期、附件和文档版本；
- **定量检查** 使用普通代码对账表格、总计、期间、百分比和 XBRL 事实；
- **语义检查** 使用分类器或 LLM 标记披露缺口、不一致陈述、样板风险因素和未经证实的主张；
- **证据检查** 要求每个语义发现引用相关的源段落以及检查器使用的政策或规则；以及
- **人工处置** 记录审查人员是接受、拒绝还是补救每个发现。

下面的紧凑工作流用运营上重要的控制逻辑取代了冗长的占位符类。

### 清单 8.2：带有证据和人工处置的分层申报审查

```python
def review_filing(document, filing_type, policy_engine, semantic_model):
    structural = check_required_sections(document, filing_type)
    numerical = reconcile_tables_and_xbrl(document)

    semantic_findings = semantic_model.flag_disclosures(
        document=document,
        filing_type=filing_type,
    )

    grounded_findings = []
    for finding in semantic_findings:
        evidence = locate_exact_evidence(document, finding)
        rule = policy_engine.lookup(finding.policy_id)
        if evidence is not None and rule is not None:
            grounded_findings.append({
                "finding": finding,
                "evidence": evidence,
                "policy_version": rule.version,
                "status": "awaiting_human_review",
            })

    return {
        "structural": structural,
        "numerical": numerical,
        "semantic": grounded_findings,
        "final_determination": None,
    }
```

显式设置 `final_determination = None` 是有意为之：自动化系统汇总证据并确定工作优先级，而经过授权的审查人员负责合规决策。

### FINRA 沟通审查

FINRA 规则 2210 为经纪商-交易商与公众的沟通设定了标准，包括内容、监督、审批和记录保存要求。沟通审查系统可筛查夸大或承诺性语言、单方面陈述、未经证实的绩效主张、缺失的风险披露或过期的审批。其他规则集可能适用于研究报告、建议、产品、司法管辖区和沟通渠道，因此活动策略库必须进行版本控制，而不能永久嵌入提示中 [Financial Industry Regulatory Authority, 2026]。

有用的发现记录应包含涉嫌问题、精确文本跨度、活动规则或内部政策、严重性、理由以及推荐的审查人员行动。严重性应基于显式因素得出，而非不透明的模型生成标签。例如，机构可以将主张的重大性、客户覆盖范围、证据质量、披露缺口和规则关键性结合为可配置的分级分数。该分数用于确定审查优先级，而不宣判沟通合规或不合规。

多标签分类可以支持此过程，因为一段文本可能同时引发多个不同的问题。但是，概率输出必须在具有代表性的、带时间戳的验证数据上进行校准，并随着规则、产品和沟通风格的变化进行监控 [Guo et al., 2017]。自动编辑应保持为拟议的修订线，直到获得授权的合规专业人员批准。

## 8.2 高级文档智能应用

单文档提取的自然延伸是对一组相关文档进行推理。财务分析通常需要跨时间比较年度和季度申报、收益发布、脚注、政策、合同和市场评论。因此，本节在章节转向面向客户的系统之前，完成了文档智能的讨论。

### 8.2.1 多文档推理与综合

多文档系统必须解决四个独立问题：选择相关文档、对齐其实体和报告期间、检测不一致或变化，以及综合带有可追溯引用的答案。当关系本身很重要时，文档图很有用——例如，当修订协议取代较早条款、脚注解释某行项目或风险因素随报告期间变化时。

### 长上下文、RAG 和文档图

*检索增强生成（RAG）* 在请求模型生成答案之前检索与查询特定的源段落 [Levis et al., 2020]。百万 token 级别上下文窗口（包括一百万到两百万 token 范围的配置）的出现改变了但并未消除架构选择。大上下文窗口使得可以在一个请求中提供许多完整的申报，但标称容量不等于可靠地使用每个 token。成本、延迟、访问控制、位置效应、重复证据和引用纪律仍然重要 [Liu et al., 2024a]。

表 8.1 总结了实际的权衡。

| 架构 | 最适用场景 | 主要优势 | 主要控制和局限性 |
| :--- | :--- | :--- | :--- |
| 长上下文提示 | 在有效上下文预算内舒适容纳的有限已知文档集 | 保留完整文档顺序并避免检索遗漏 | 更高的 token 成本和延迟；无关文本可能稀释证据；提示构建前必须执行权限控制 |
| RAG | 大型或频繁变更的存储库以及聚焦的事实性问题 | 更低的上下文成本；元数据过滤；源级访问控制；更容易更新语料库 | 检索可能遗漏关键段落；需要评估召回率、重新排序、分块和引用忠实性 |
| 文档图 | 交叉引用、修订、实体关系、时间依赖和矛盾分析 | 使文档关系显式化并支持结构化一致性检查 | 更多的摄入和维护工作；图构建错误可能传播到综合中 |
| 混合 | 大型语料库上的高风险分析 | 使用检索或图选择证据，然后向长上下文模型提供完整的相关章节或文档 | 最稳健但操作复杂；需要端到端测试、来源追踪和成本控制 |

Table 8.1: 何时使用长上下文、RAG、文档图或混合设计

当证据集小、稳定、权限兼容且远低于有效上下文限制时，长上下文提示很合适。当语料库大、动态或受访问控制，或者查询狭窄且重复时，RAG 更可取。当答案取决于修订链、实体消解、期间对齐或跨文档一致性，而非简单段落检索时，文档图变得有价值。对于许多金融应用来说，最可辩护的设计是混合：检索并重新排序候选材料，扩展到完整章节，保留显式关系，然后在受控长上下文请求中进行综合。

### 清单 8.3：混合多文档编排（架构草图）

```python
def answer_across_filings(question, corpus, model, token_budget):
    permitted = corpus.filter_by_user_entitlements()
    candidates = hybrid_retrieve(question, permitted, top_k=40)
    ranked = rerank(question, candidates)[:12]
    expanded = expand_to_complete_sections(ranked)

    if estimated_tokens(expanded) <= token_budget:
        evidence = expanded
        mode = "retrieval_plus_long_context"
```

