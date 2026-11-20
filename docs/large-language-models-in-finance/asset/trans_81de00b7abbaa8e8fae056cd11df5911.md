

```
precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
return {"precision": precision, "recall": recall, "f1": f1,
       "tp": tp, "fp": fp, "fn": fn}
```

## 15.7 库生态系统

金融 LLM 工作流通常结合通用机器学习库、检索工具、市场数据包、回测框架和部署基础设施。此处列出的工具仅为示例，而非固定推荐，其适用性取决于许可、维护状态、集成要求和机构审批。

### 15.7.1 金融 LLM 开发的核心库

表 15.7.1 汇总了用于原型设计和构建金融 LLM 工作流的常见 Python 库。

| 库 | 类别 | 金融领域的关键特性 |
|---|---|---|
| transformers | 模型框架 | 预训练模型和检查点，包括 FinBERT 风格的分类器和 FinEAS 风格的情感组件 |
| langchain | LLM 应用 | 用于多步金融分析和工具集成的代理框架 |
| sentence-transformers | 嵌入 | 文档相似度、语义搜索和 FinEAS 风格的句子嵌入工作流 |
| chromadb | 向量数据库 | 用于 RAG 系统的高效检索，元数据过滤 |
| yfinance | 市场数据 | 用于历史价格数据和企业行为的研究/原型访问；生产环境使用需要许可和质量检查 |
| pandas-ta | 技术分析 | 用于量化策略的 130 多个指标 |
| quantlib | 衍生品 | 期权定价模型、收益率曲线、风险指标 |
| backtrader | 回测 | 带有真实市场条件的策略测试 |
| sec-edgar | 监管数据 | 直接访问 SEC 文件和 XBRL 数据 |
| newsapi | 新闻数据 | 用于情感分析的实时新闻 |

表 15.3: 金融 LLM 工作流的 Python 库示例

在生产环境中，这些开发库通常与服务、编排、监控、存储、消息传递和 API 管理工具结合使用。

### 15.7.2 生产部署栈

以下栈组件是常见基础设施类别的示例。实际选择应遵循机构的安全、合规、延迟、可靠性和运营要求。

- **模型服务:** Triton 推理服务器, TorchServe, TensorFlow Serving
- **编排:** Kubernetes, Airflow, Prefect
- **监控:** Prometheus, Grafana, MLflow
- **数据存储:** PostgreSQL, MongoDB, S3 兼容对象存储
- **消息队列:** Kafka, RabbitMQ, Redis
- **API 网关:** Kong, Traefik, AWS API 网关

## 15.8 经验金融 LLM 研究的可重复性协议

金融 LLM 研究应在数据来源、时间划分、模型版本、提示词版本、检索语料版本和评估脚本层面具有可重复性。以下协议是本书经验示例所假设的最低标准。

- 1. **数据卡。** 说明数据来源、许可、时间戳、幸存者偏差政策、文档宇宙、预处理规则以及任何排除标准。
- 2. **时间分割。** 使用走势前行或严格按时间顺序的训练/验证/测试划分。任何日期晚于决策时间的文档、标签、嵌入、提示词示例或检索分块均不得进入模型上下文。
- 3. **模型卡。** 记录基础模型、分词器、微调方法、量化、上下文窗口、推理参数和安全过滤器。
- 4. **提示词和检索版本控制。** 对提示词模板、工具模式、向量索引、分块策略、重排序器和检索证据包进行哈希处理。
- 5. **指标。** 报告任务指标和治理指标：准确率或 F1 分数、校准、引用忠实度、数值一致性、拒绝准确率、升级精确度、延迟、成本和失败案例。
- 6. **不确定性。** 在抽样变化重要时提供置信区间或自助法区间，并将示例性示例与经审计的经验结果分开。
- 7. **审计日志。** 对每次重要运行，存储模型版本、提示词版本、检索的文档标识符、工具调用参数、人工批准和最终输出哈希。

配套实现可以遵循以下目录结构：

```
experiments/
 rag_sec_filings/
  data_card.md
  build_index.py
  evaluate_faithfulness.py
 financial_sentiment/
  temporal_split.py
  train_peft.py
  evaluate_calibration.py
 governance_eval/
  policy_tests.yaml
  refusal_and_escalation_eval.py
 leakage_tests/
  contamination_checks.py
  timestamp_audit.py
model_cards/
prompt_registry/
audit_logs/
```

核心规则很简单：除非读者能够识别决策时可用的信息，并从原始输入复现到报告指标的评估路径，否则不应将任何结果作为经验证据呈现。

## 15.9 参考实现标准

官方配套仓库是 Packt 的官方 _金融中的 LLM_ 仓库。它不打算成为一个完整的生产平台。其目标更狭隘且更重要：提供本书中使用的不变量的小型、可审计的参考实现。对于此版本，可重复性标识符是提交 `c80ddc04d29fc2ec245bd7bda03c6833e2b2392e` 上的标签 `B32413-v1.0.0`。提交包含该标签的源代码存档和完整不可变的 Git 束。官方 Packt 仓库应镜像精确的标签和提交；默认分支不是可重复性标识符。

仅当参考实现满足以下检查清单时，才被认为是可接受的：

- 1. 所有对外可见对象都具有类型化架构；
- 2. 时间戳检查是显式的且失败封闭；
- 3. 检索返回证据标识符，而非匿名文本块；
- 4. 重要声明可以链接到证据标识符；
- 5. 工具调用在执行前通过确定性授权；
- 6. 评估脚本同时计算任务指标和治理指标；
- 7. 测试在没有私有凭据或专有数据的情况下运行。

在标记版本中，目录 `companion/src/finllm` 使用轻量级依赖的 Python 实现了此检查清单，以便逻辑保持可检查。其组件包括类型化的文档、声明、工具调用、版本和审计对象；确定性检索冒烟测试；时间戳泄漏检查；声明支持评分；校准指标；以及受保护的工具执行。这些并非企业系统的替代品；它们是手稿中使用的设计原则的可执行规范。目录 companion/experiments/rag_sec_filings 包含一个无凭据的冒烟基准，包含虚构文件、问题、可运行脚本、结果表和审计日志。

目录 companion/experiments/pit 包含第 13 章生成器、可机读的图书结果、种子-11 运行以及声明的 60 种子漂移分解。这些基准测试可重复性和治理；它们不是实时市场预测性能的证据。

参考实施强制执行的最重要的数学不变量是

$a_j \le t$ 对于在决策时间 $t$ 使用的每个证据项 $d_j$。

如果此不等式对任何检索项失败，管道应拒绝该运行。这个单一不等式是第 3、4 和 14 章中无前瞻命题的操作形式。

### 清单 15.6: 配套包的最小用法

```
from datetime import datetime, timezone
from finllm.contracts import Claim, Document, EvidenceBundle
from finllm.rag import faithfulness_score, retrieve_by_bm25_like
from finllm.temporal import assert_documents_available

now = datetime(2026, 7, 1, tzinfo=timezone.utc)
corpus = (
    Document(
        document_id="filing-001",
        text="Revenue increased by 8 percent in fiscal 2025.",
        source="10-K",
        timestamp=datetime(2026, 2, 15, tzinfo=timezone.utc),
    ),
)
assert_documents_available(corpus, decision_time=now)
evidence = retrieve_by_bm25_like("How did revenue change?", corpus, k=3)
claim = Claim(text="Revenue increased by 8 percent.")
print(faithfulness_score([EvidenceBundle(claim, evidence)]))
```

示例故意保持简小，并使用标记配套版本实现的接口。它捕捉了基本的生产习惯：系统存储证据标识符，检查可用时间，从有效语料库检索，并评估声明是否被引用。更复杂的实现可能使用密集嵌入、混合搜索、交叉编码器重排序、声明级自然语言推理和不可变审计存储；不变量保持不变。

