

归档版本通过以下方式检查：

## 清单 15.7：运行包含的参考基准和测试

```
cd companion
python experiments/rag_sec_filings/run_experiment.py
bash run_checks.sh
```

第一个命令重新生成 results.csv 和 audit_log.jsonl；第二个命令运行单元测试，重新生成确定性基准输出，验证第 13 章 seed-11 表格，并检查声明的 60-seed 输出是否存在漂移。这些命令是发布合同的一部分，而非实时投资表现的证据。

## 仓库发布合同

此版本包含：

- 1. 语义化版本标签 B32413-v1.0.0 和不可变提交 c80ddc04d29fc2ec245bd7bda03c6833e2b2392e；
- 2. pyproject.toml 以及针对受支持环境的哈希锁定的依赖锁文件和预置 wheel；
- 3. 无需私密凭据即可运行的持续集成和本地测试；
- 4. 数据卡、模型卡和提示词注册表模板，以及许可证信息；
- 5. 可从代码重新生成每本书结果的命令；
- 6. 机器可读输出和依赖项校验和清单；
- 7. 随出版商提交提供的源代码存档和完整的 Git 包。

## 表 15.4：参考包的书籍到仓库对照表

| 书籍对象 | 仓库位置 |
|---|---|
| 类型化合同和审计对象 | src/finllm/contracts.py |
| 时间点检查 | src/finllm/temporal.py 和 experiments/leakage_tests/ |
| 确定性检索模式 | src/finllm/rag.py |
| 策略门控的工具执行 | src/finllm/governance.py |
| 校准和忠实度指标 | src/finllm/metrics.py |
| 免凭据的申报基准 | experiments/rag_sec_filings/ |
| 第 13 章时间点实验 | experiments/pit/ 和 experiments/OUTPUT_SHA256SUMS.txt |
| 回归测试 | tests/test_core.py, run_checks.sh, 和持续集成工作流 |

因此，手稿与仓库通过稳定合同相联系，而不是将每个长脚本复制到印刷页中。

### 最终阶段更新：代理交易部署的最小检查清单

在交易或投资组合代理超越只读协助之前，部署记录应包括：时间点证据；密封或走向前评估；费用、点差、借贷、冲击和容量；与简单基线和强学习基线的比较；订单级模拟；压力测试和逆压力测试；对抗性文档和提示词扰动；治理消融；确定性重放；以及从证据到订单再到成交的审计血缘。此检查清单故意不创建新章节或新框架。它是本书中添加的代理材料的紧凑型最终阶段控制面。

## 15.10 总结

本章汇集了支持金融 LLM 开发所需的技术基础和实用资源。此处介绍的数学概念、架构比较和实用示例为设计、评估和调整金融 LLM 工作流提供了参考点。

主要要点包括：

- 理解注意力、优化和损失函数有助于读者将 Transformer 行为与实现选择联系起来
- 仅编码器、仅解码器和编码器-解码器架构支持不同的金融 NLP 和生成任务；FinEAS 风格的句子嵌入说明了紧凑型领域特定情感模块的价值
- 微调、监控、RAG 和评估示例应被视为说明性模式，在投入生产使用前需要进行调整
- 必须对金融数据集、市场数据源、SEC 申报文件和库生态系统进行覆盖范围、许可、时效性和适用性检查
- 强健的数据管道、评估方法、FinEAS/FinBERT 监控和治理控制对于可靠的金融 LLM 工作流是必要的

所介绍的资源和模式应根据机构要求、监管约束和特定用例进行调整。由于金融 AI 生态系统变化迅速，团队必须保持资源、库、数据集和评估实践的最新状态，同时维持稳定性、合规性和治理。

## 获取本书 PDF 版本及更多内容

扫描二维码（或访问 packtpub.com/unlock）。按名称搜索此书，确认版本，然后按照页面上的步骤操作。

![](img/c31705b361fdf81dee8826a2c4ee167f_512_0.png)

立即解锁

*注意：请准备好发票。直接从 Packt 购买不需要发票。*

## 您可能还喜欢的其他图书

如果您喜欢本书，可能会对 Packt 的这些其他图书感兴趣：

![](img/c31705b361fdf81dee8826a2c4ee167f_513_0.png)

面向金融的 AI 代理构建
Hanane Dupoux, Fayssal El Mofatiche
ISBN: 978-1-83702-229-8

- 了解金融中的 AI 代理和代理系统
- 掌握核心 AI 代理设计模式并将其应用于金融
- 探索代理工作流中使用的推理范式
- 使用各种架构风格设计多代理编排
- 通过动手 Python 实验构建金融用例
- 有效地评估和测试 AI 代理行为
- 实施护栏、追踪和可观测性
- 将 AI 代理应用于基本面分析、交易、研究和合规

![](img/c31705b361fdf81dee8826a2c4ee167f_514_0.png)

## 每位 AI 工程师必须构建的 30 个代理

Imran Ahmad
ISBN: 978-1-80610-901-2

- 部署可安全可靠扩展的生产就绪代理系统
- 使用 LangChain 和 LangGraph 构建具有模块化架构的自主代理
- 实现具有复杂内存、规划和推理能力的代理
- 将工具、API 和外部数据无缝集成到代理工作流中
- 建立强健的评估框架以衡量和优化代理性能
- 实施护栏和可解释性功能，以确保合乎道德且安全的部署
- 为复杂的协作任务编排构建多代理系统
- 将特定代理架构应用于医疗保健、金融和法律领域

## 分享您的想法

读完《金融中的大语言模型》后，我们非常希望听到您的想法！扫描下方的二维码，直接前往本书的 Amazon 评论页面并分享您的反馈。

![](img/c31705b361fdf81dee8826a2c4ee167f_515_0.png)

```
https://packt.link/r/1837024537
```

您的评论对我们和技术社区非常重要，并将帮助我们确保提供优质内容。

## 参考文献

Daron Acemoglu, Asuman Ozdaglar, and Alireza Tahbaz-Salehi. 金融网络中的系统性风险与稳定性. _美国经济评论_, 105(2):564–608, 2015. doi: 10.1257/aer.20130456.

AI @ Meta. Llama 3. `https://ai.meta.com/blog/meta-llama-3/`, 2024.

Michael Aitken, Douglas Cumming, and Feng Zhan. 市场操纵与市场质量. _金融市场期刊_, 26:162–185, 2015.

Leman Akoglu, Hanghang Tong, and Danai Koutra. 基于图的异常检测与描述：一项综述. _数据挖掘与知识发现_, 29(3):626–688, 2015.

Robert Almgren and Neil Chriss. 投资组合交易的最优执行. _风险期刊_, 3:5–40, 2001.

Mohammed Alsheikh, Roderick Bloem, Ruediger Ehlers, Bettina Koenighofer, Scott Niekum, and Ufuk Topcu. 基于屏蔽的安全强化学习. 在 _第三十二届 AAAI 人工智能会议论文集_, 页 2669–2678, 2018. URL `https://ojs.aaai.org/index.php/AAAI/article/view/11797`.

Edward I Altman. 财务比率、判别分析与企业破产预测. _金融期刊_, 23(4):589–609, 1968.

Kartik Anand, Sophia Kazinnik, Agnese Leonello, and Ettore Panetti. Ex machina：人工智能时代的金融稳定. 工作文件系列 3225, 欧洲中央银行, 2026 年 5 月. URL `https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp3225~52d8adae99.en.pdf`.

Torben G Andersen, Tim Bollerslev, Francis X Diebold, and Clara Vega. 宏观公告的微观影响：外汇中的实时价格发现. _美国经济评论_, 93(1):38–62, 2003.

Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, and Yuyang Wang. Chronos：学习时间序列的语言, 2024. URL `https://arxiv.org/abs/2403.07815`.

Abdul Fatir Ansari, Oleksandr Shchur, Jaris Küken, Andreas Auer, Boran Han, Pedro Mercado, Syama Sundar Rangapuram, Huibin Shen, Lorenzo Stella, Xiyuan Zhang,

