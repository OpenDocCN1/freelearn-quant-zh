

```
n_estimators=200,
max_depth=6,
random_state=42
)
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)
# At this point, appropriate backtesting, risk analysis, and validation
# procedures should be applied. This listing focuses on architecture only.
```

在实时交易环境中，此管道必须嵌入更广泛的框架：

- 带有现实交易成本的滚动式前向评估 [Harvey et al., 2016, López de Prado, 2018],
- 测量的延迟和可靠性服务级别目标 [Hasbrouck and Saar, 2013],
- 风险控制（仓位限制、止损政策、压力测试）[Jorion, 2007],
- 与投资组合构建和订单路由集成 [Grinold and Kahn, 2000, Almgren and Chriss, 2001].

## 7.1.2 情感与叙事信号

金融文本的情感分析已从基于词典的方法 [Loughran and McDonald, 2011] 发展为领域特定的 Transformer 模型，如 FinBERT [Yang et al., 2020, Araci, 2019]。LLM 进一步将其泛化，通过捕捉多维情感和叙事结构 [Hassan et al., 2019, Calomiris and Mamaysky, 2019, Lopez-Lira and Tang, 2023]。例如，一份文档可能对收入增长持积极态度，但对监管风险或流动性状况持消极态度。叙事信号捕捉文本中的反复出现的主题，如通胀压力、诉讼风险、再融资压力或管理层信心。

与单一情感标签不同，生产系统可能会估计一个向量：

```
s_t = (s_t^{pos}, s_t^{neg}, s_t^{unc}, s_t^{reg}, s_t^{liq}, ...),
(7.2)
```

其中各分量可能跟踪积极基调、消极基调、不确定性、监管风险、流动性担忧等等 [Baker et al., 2016, Hassan et al., 2019]。

s_t^{reg} 和 s_t^{liq} 这两个术语值得单独解释。监管风险分量 s_t^{reg} 旨在捕捉暗示面临新规则、监管行动、执法调查、资本或报告约束、许可不确定性、诉讼或其他法律合规压力的语言。例如，反复提及监管机构的调查、待定的规则变更或可能的罚款，即使文档在销售或收益方面持积极态度，也应提高监管风险信号。

流动性担忧分量 s_t^{liq} 旨在捕捉暗示在没有显著成本的情况下难以进行交易、融资、再融资、赎回或满足短期现金义务的语言。在面向市场的文本中，流动性担忧可能表现为提及薄型订单簿、更宽的买卖价差、市场深度有限、被迫出售、赎回压力或对回购和融资市场的受限访问。在公司或信贷文件中，它们可能表现为关于现金烧耗、契约压力、债务到期、抵押品追缴或依赖应急信贷额度的警告。分离这些维度很有用，因为一份文本可能具有有利的业务叙事，但仍包含具有经济实质性的监管或流动性警告。

这些信号通常用于：

- 横截面预期收益或波动率模型中的特征 [Kelly et al., 2019, Freyberger et al., 2020],
- 风险模型中的条件变量（例如，用于预测波动率激增）[Manela and Moreira, 2017, Baker et al., 2019],
- 事后交易分析中的解释变量 [Hendershott and Menkveld, 2014].

## 实现草稿：标题级情感

以下代码片段说明了如何使用金融情感模型对标题进行评分。确切标签取决于所选的情感模型，使用前应检查。

## 列表 7.2：标题级金融情感评分

```python
# Illustrative template. Not production code.

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

model_name = "ProsusAI/finbert" # or an internal finance-specific model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

id2label = model.config.id2label

def score_headlines(headlines):
    inputs = tokenizer(
        headlines,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs

# Load headlines tagged with ticker and date
news_df = pd.read_csv("news_ticker.csv") # columns: date, ticker, headline
scores = score_headlines(news_df["headline"].tolist())

sentiment_cols = [f"sent_{id2label[i].lower()}" for i in range(scores.shape[1])]

for j, col in enumerate(sentiment_cols):
    news_df[col] = scores[:, j]

# Aggregate to daily per-ticker features
daily_sent = (
    news_df
    .groupby(["ticker", "date"])[sentiment_cols]
    .mean()
    .reset_index()
)

# 'daily_sent' can be merged with returns, factors, and risk metrics.
```

在实践中，此情感子系统通常还会伴随：

- 时间和跨行业的校准与稳定性检查 [Niculescu-Mizil and Caruana, 2005],
- 因子交互分析（例如，情感 vs. 价值或动量）[Stambaugh et al., 2012],
- 对前视偏差和发布时间戳的谨慎处理 [Katona et al., 2018].

## 7.1.3 事件驱动与宏观策略

与连续预测不同，事件驱动策略关注特定公告或冲击。事件驱动和宏观策略围绕离散信息发布展开：

- 季度收益和指引更新 [Beaver, 1968, Ball and Brown, 1968],
- 并购公告 [Mitchell and Pulvino, 2001],
- 信用评级变动 [Hand et al., 1992],
- 宏观经济数据发布 [Andersen et al., 2003, Savor and Wilson, 2014],
- 中央银行声明和讲话 [Nakamura and Steinsson, 2018, Gardner et al., 2022].

LLM 通过以下方式支持这些策略：

- 准实时阅读收益发布和电话会议记录，提取意外情况、前瞻性指引和风险披露 [Davis et al., 2015, Larcker and Zakolyukina, 2012],
- 将中央银行沟通分类为比预期更紧缩或更宽松 [Hansen et al., 2018],
- 生成结构化事件摘要，用于输入因子模型或场景生成器 [Ke et al., 2019].

宏观事件的典型管道包括：

1. 检索历史事件文档及相关市场反应，
2. 训练模型将文本特征映射到实现的价格/曲线变动 [Bholat et al., 2015],
3. 使用 LLM 实时将新事件处理到相同的特征空间中，
4. 将文本信号与订单簿和流量信息结合 [Menkveld, 2013].

## 7.1.4 多代理研究工作站

资产管理公司和对冲基金的研究涉及：

- 基于基本面、估值和主题筛选工具 [Asness et al., 2013],
- 阅读和综合大量文本，
- 构建和维护现金流量及风险模型，
- 撰写投资备忘录和建议。

如第 5 章所述，LLM 代理可在人类监督下编排此工作流 [Wang et al., 2024, Xi et al., 2025]。典型的研究副驾驶可能会协调：

- 检索代理（对文件、记录和研究进行 RAG）[Lewis et al., 2020],
- 摘要代理（公司快照、关键指标、近期事件）[Zhang et al., 2020],
- 风险代理（治理、法律、宏观风险、ESG）[Berg et al., 2022],
- 论点代理（草拟带有利弊的投资论点）。

## 伪代码：研究循环

列表 7.3：多代理研究循环（简化）

```python
# Illustrative template. Not production code.

def research_loop(ticker, question, tools, llm):
    """
    tools: dict with components such as:
    - "retriever": document retrieval / RAG interface
    - "pricing_api": market data client
    - "risk_db": internal risk and exposure database
    llm: orchestrator LLM with suitable prompts and policies
    """

    # 1. Retrieve relevant documents
    context_docs = tools["retriever"].get_company_docs(ticker)

    # 2. Pull pricing history and risk profile
    pricing = tools["pricing_api"].get_history(ticker)
    risk_info = tools["risk_db"].get_exposures(ticker)
```

