

```
# 3. Summarizer agent
summary = llm.summarize_company(
    ticker=ticker,
    docs=context_docs,
    pricing=pricing
)

# 4. Risk agent
risk_report = llm.analyze_risks(
    ticker=ticker,
    docs=context_docs,
    risk_data=risk_info
)

# 5. Thesis agent
thesis = llm.write_investment_thesis(
    ticker=ticker,
    question=question,
    summary=summary,
    risk_report=risk_report
)

return {
    "summary": summary,
    "risk_report": risk_report,
    "thesis": thesis
}
```

在生产环境中，对 LLM 的每次函数调用都会：

- 包含清晰的系统提示词和用户提示词 [Reynolds and McDonnell, 2021]，
- 受工具模式和治理规则的约束，
- 记录所有中间产物以供日后审查。

> **最终阶段更新：受控代理交易**
> 对于交易工作流，多代理工作站不应被理解为直接发出买入或卖出指令的 LLM。一种最小受控设计将五个平面分离：(i) 一个 **证据平面**，用于时点数据、文档、时间戳和来源；(ii) 一个 **认知平面**，用于专家代理、预测、场景和研究记忆；(iii) 一个 **控制平面**，用于授权、权力预算、约束、审批和杀止状态；(iv) 一个 **执行平面**，用于意图订单编译、交易场所适配器、交易前闸门、成交和对账；以及 (v) 一个 **保障平面**，用于验证、影子评估、监控、事件响应和不可变审计。这是对本章前台架构的最小补充：LLM 最自然地放置在慢认知和解释中，而市场准入则保留在确定性控制和执行服务中 [Noguer i Alonso and Pacheco Azmar, 2026]。

![](img/c31705b361fdf81dee8826a2c4ee167f_274_0.png)

图 7.1：同一堆栈，三个办公室。

用例在术语和签署人方面有所不同；底层机制却并非如此。如果一家机构为每个办公室构建单独的证据库、工具注册表和审计跟踪，那么它并非构建了三个系统——而是构建了三套相同的未经验证的依赖，并在首次检查时会发现这一点。

## 7.2 中台：风险管理与监控

虽然前台应用程序关注市场机会和交易决策，但中台应用程序专注于衡量、监控和控制风险。中台功能将交易和银行活动转化为风险概况、限额和监管报告。LLM 通过结构化自由文本信息、生成场景以及优先排序人工关注来提供辅助。

### 7.2.1 信用风险与预警信号

信用风险分析依赖于：

- 财务报表及附注 [Altman, 1968, Ohlson, 1980]，
- 管理层评论与指引 [Mayew et al., 2015]，
- 评级机构报告 [Becker and Milbourn, 2011]，
- 贷款协议和信托契约中的契约条款 [Chava and Roberts, 2008]，
- 板块与宏观经济分析 [Duffie et al., 2007]。

LLM 的帮助体现在：

- 从信用备忘录中提取关键比率和文本信号 [Mai et al., 2019, Netzer et al., 2019, Berg et al., 2020]，
- 将叙述性风险描述映射为标准化标签，
- 监控新闻和申报文件中的预警指标（例如，法律纠纷、监管调查、降级）[Campbell et al., 2008]。

系统化设置可能会为每个债务人维护：

- 1. 一个结构化概况，总结业务模式、杠杆、流动性和抵押品；
- 2. 按风险类型分类的叙述性事件流；
- 3. 从结构化因素和文本因素中得出的预警评分 [Bharath and Shumway, 2008]。

### 实现草图：契约提取

贷款和债券文件通常包含对信用风险至关重要的契约：杠杆上限、覆盖比率、资产出售限制、交叉违约条款等。持续提取这些契约是文档 NLP 的一项自然任务，但生产系统需要条款级证据、明确的计算规则和法律审查。CUAD 和 ContractNLI 等公共合同基准为提取和文档级推理提供了有用的起点 [Hendrycks et al., 2021, Koreeda and Manning, 2021]。

### 清单 7.4：从贷款协议中简单提取契约

```
# Illustrative template. Not production code.
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
model_name = "approved-legal-ner-model" # replace with the reviewed model identifier
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
id2label = ner_model.config.id2label
def extract_covenants(text):
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )
    with torch.no_grad():
        out = ner_model(**tokens)
    logits = out.logits[0]
    preds = torch.argmax(logits, dim=-1).cpu().numpy()
    words = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    covenants = []
    current = []
    for w, p in zip(words, preds):
        label = id2label[p]
        if label.startswith("B-Covenant"):
            if current:
                covenants.append(" ".join(current))
                current = []
            current.append(w)
        elif label.startswith("I-Covenant") and current:
            current.append(w)
        else:
            if current:
                covenants.append(" ".join(current))
                current = []
    if current:
        covenants.append(" ".join(current))
    return covenants
```

在实践中，契约提取将与以下内容结合：

- 核心信贷模型 (PD/LGD/EAD) [巴塞尔银行监管委员会, 2005]，
- 抵押品和敞口数据，
- 供分析师审查和更正提取信息的工作流工具。

### 7.2.2 市场风险与场景分析

市场风险是指由于价格、利率、利差、货币或波动率变化而导致损失的风险。市场风险管理使用诸如风险价值 (VaR) 和预期短缺 (ES) 等模型，并结合压力测试和场景分析。VaR 估计特定置信水平下的损失阈值；预期短缺估计超过该阈值的平均损失。[Jorion, 2007, McNeil et al., 2015]。LLM 通过以下方式做出贡献：

- 生成压力场景的叙述性描述 [Hirtle and Lehnert, 2015]，
- 将叙述映射到风险因素冲击（利率、利差、外汇、股票水平和波动率、大宗商品），
- 解释为何特定头寸或投资组合容易受损。

典型的场景流水线：

- 1. 定义定性场景（例如，“由于通胀意外，央行出人意料地大幅收紧”）
- 2. 使用 LLM 将场景阐述为一组连贯的宏观事件 [Schorfheide et al., 2018]，
- 3. 将这些事件映射到曲线和利差冲击 [Diebold and Li, 2006]，
- 4. 将冲击通过定价/风险引擎运行，
- 5. 使用 LLM 按交易台、产品和风险因素总结影响。

### 7.2.3 实时市场监控

市场监控系统监控交易活动，以发现潜在的市场滥用行为（操纵、虚假下单、分层交易和内幕交易）。它们基于统计触发器和基于规则的过滤器生成大量警报 [Cumming et al., 2020, Aitken et al., 2015]。

LLM 通过以下方式提供辅助：

- 分析交易者通信（电子邮件、聊天记录）以识别与可疑行为一致的叙述，
- 为人工调查人员总结并提供警报上下文，
- 聚类相似警报并突出跨交易台、工具或时间的模式 [Golbeck et al., 2018]。

在这里，LLM 主要充当 *分诊引擎*，而非决策者。人工调查人员仍对最终决策和监管申报负责。

## 7.3 后台：银行业务、欺诈、反洗钱与运营

后台应用程序较少关注实时交易决策，而更侧重于准确处理、文档记录、调查和合规支持。后台功能包括信贷承销、交易监控、KYC/AML 流程以及各种运营工作流。这些过程产生和消耗大量结构化数据、非结构化文本和文档 [Arner et al., 2017]。

### 7.3.1 信贷承销与审查

在零售和中小企业 (SME) 银行业务中，信贷承销通常结合：

- 结构化数据（收入、余额、交易历史、征信评分）[Thomas et al., 2017]，
- 基于文档的信息（工资单、纳税申报表、财务报表），
- 叙述性信息（商业计划书、资金用途说明）。

LLM 可以：

- 从文档中提取结构化特征（收入、支出、抵押品）[Huang et al., 2022, Kim et al., 2022]，
- 将申请人概况总结为标准化信贷备忘录，
- 标记不一致或缺失的信息 [Netzer et al., 2019]。

目标是增强，而非取代，现有的信用评分模型。LLM 派生特征纳入现有的 PD/LGD 框架和人工信贷委员会 [Berg et al., 2020]。

