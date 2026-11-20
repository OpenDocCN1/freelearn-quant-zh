

## 金融领域大语言模型应用

本章探讨大语言模型（LLMs）在金融机构前台、中台和后台的具体应用。在此，前台是指产生收入的职能，如交易；中台涵盖风险与控制职能；后台涵盖运营与处理。我们不将模型视为孤立的组件，而是将其视为完整决策流程（从数据摄入到模型处理、人工审核、行动及监控的完整路径）中的构建模块，应用于交易、风险管理、银行运营、欺诈检测和监管合规 [Cao, 2022, Yang et al., 2020]。

本章的重点是架构设计。对于每个用例，我们描述：

- **输入**（数据来源及其结构），
- **处理流水线**（模型、RAG 索引、代理、工具），
- **输出**（信号、评分、报告、警报），
- 和 **控制层**（限额、审批、日志、监控）。

我们避免做出性能声明或编造指标；任何实际部署都必须在严格的模型风险框架下，使用专有数据进行验证 [Danielsson et al., 2016]。

在本章中，我们沿用此前介绍的 *金融 LLM 栈*。该栈是指构建、连接、部署和管理基于 LLM 的金融系统所需的各层。在下方的栈中，*工具-上下文集成*被用作通用架构术语：当机构采用 MCP 时，它包括模型上下文协议（MCP）风格的接口，但也包括专有 API 网关、内部服务总线、工作流引擎，以及与风险、订单管理、合规和文档系统的直接集成。

*数据 → 预训练 → 微调 → RAG → 代理 → 工具-上下文集成 → RL / 对齐 → 基础设施 → 治理 → 前沿。*

每个部分说明了该栈中不同组件如何在生产级工作流中结合。我们从前台开始，在此，LLM 应用最接近投资决策、市场解读和交易工作流。

## 7.1 前台：市场分析与交易

前台活动包括市场分析、信号生成、酌情交易和系统性交易以及投资组合构建。这些应用尤其敏感，因为错误可能直接影响交易决策、投资组合风险和客户结果。LLM 可以通过使非结构化信息（如文件、新闻、研究、转录记录甚至交易员聊天记录）大规模可供机器使用，来补充传统定量方法 [Wu et al., 2023b, Lopez-Lira and Tang, 2023]。

### 7.1.1 混合时间序列与文本预测

经典时间序列模型（ARIMA、GARCH、状态空间模型、LSTM、Transformer）作用于价格、成交量和宏观经济指标等数值输入 [Tsay, 2010, Bollerslev, 1986, Hochreiter and Schmidhuber, 1997, Vaswani et al., 2017]。例如，盈利预警或央行讲话可能会迅速影响资产价格。然而，在实践中，价格形成受文本信息的强烈影响，例如：

- 财报发布和电话会议 [Tetlock, 2007, Jegadeesh and Wu, 2013]，
- 宏观数据发布和央行讲话 [Hansen et al., 2018, Ehrmann and Wabitsch, 2022]，
- 评级机构公告 [Cornaggia et al., 2018]，
- 重大新闻和企业行为 [Tetlock, 2007, Calomiris and Mamaysky, 2019]。

LLM 使我们能够将此类文本转换为可整合到预测模型中的数值表示。所产生的 *混合* 架构包含：

- 1. **文本处理层**，将文档编码为嵌入向量或结构化信号；
- 2. **特征层**，将文本衍生信号与传统定量特征聚合；以及
- 3. **预测模型**，消费联合特征空间。

### 架构

设 $r_{t+1}$ 表示 $t$ 与 $t+1$ 之间的对数收益率，$\mathbf{x}_t$ 表示时刻 $t$ 的数值特征（收益率、波动率、宏观指标），$\mathbf{z}_t$ 表示总结在 $(t-1, t]$ 期间内观察到的所有与所研究工具或指数相关的文档的文本衍生特征。简单来说，该模型结合市场数据和文档衍生信号来预测下一期收益率。混合预测器可以写成

$$\widehat{r}_{t+1} = f_\theta(\mathbf{x}_t, \mathbf{z}_t),$$ (7.1)

此处，目标是下一期收益率，而输入结合了数值市场特征和文本衍生特征。函数 $f_\theta$ 在历史数据上学习得到 [Gu et al., 2020, Chen et al., 2024a]。

文本流水线通常如下进行：

- 1. 收集原始文档（新闻、文档、转录记录），
- 2. 清洗、去重并将记录映射到时点发行人和证券标识符，
- 3. 将每个文档编码为向量或结构化摘要 [Reimers and Gurevych, 2019, Gao et al., 2021]，
- 4. 按工具和时间桶（如一天、一小时或事件窗口）聚合为 z_t。

聚合可以简单到对嵌入向量求平均，也可以复杂到基于时效性或来源可靠性加权的注意力池化 [Yang et al., 2016, Lin et al., 2017]。

### 每日预测的简单混合流水线

下方清单展示了股票指数的每日混合流水线最小示例。该示例仅使用公开风格的数据来源来展示工作流结构。它是一个说明性模板，而非生产代码。在实时交易环境中，此类流水线在使用前需要回测、风险分析、验证和监控。

### 清单 7.1：带价格和新闻的最小混合预测流水线

```
# Illustrative template. Not production code.
# Evaluation, risk limits, and validation omitted for brevity.

import pandas as pd
import numpy as np
import yfinance as yf
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestRegressor
import torch

# 1. Download price data
ticker = "SPY"
data = yf.download(ticker, period="10y", interval="1d")

# Robust choice of price column (Adj Close if available, else Close)
price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
data["ret_1d"] = data[price_col].pct_change()
data["target"] = data["ret_1d"].shift(-1)

# 2. Load daily news headlines (pre-collected)
# Assumed format: columns: date (YYYY-MM-DD), headline (str)
news_df = pd.read_csv("daily_news.csv", parse_dates=["date"])

# 3. Aggregate news by day
daily_headlines = (
    news_df
    .groupby("date")["headline"]
    .apply(lambda x: " ".join(x))
    .sort_index()
)
```

### 4. 加载文本编码器（句子嵌入）

```
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoder = AutoModel.from_pretrained(model_name)

def embed_text(texts, batch_size=16):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            out = encoder(**inputs).last_hidden_state[:, 0, :]
        all_vecs.append(out.cpu().numpy())
    return np.vstack(all_vecs)
```

```
news_embeddings = embed_text(daily_headlines.tolist())
news_features = pd.DataFrame(
    news_embeddings,
    index=daily_headlines.index,
    columns=[f"news_{i}" for i in range(news_embeddings.shape[1])]
)
```

### 5. 对齐数值特征与文本特征

```
daily = data.copy()
daily["date"] = daily.index.normalize()
X_num = daily[["ret_1d"]].shift(1)  # simple example: lagged return

X = (
    X_num
    .set_index(daily["date"])
    .join(news_features, how="inner")
)
y = daily["target"].loc[X.index]

mask = ~y.isna()
X, y = X[mask], y[mask]
```

### 6. 训练简单模型（训练/测试划分）

```
n_train = int(0.7 * len(X))
X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
y_train, y_test = y.iloc[:n_train], y.iloc[:n_train:]

model_rf = RandomForestRegressor(
```

