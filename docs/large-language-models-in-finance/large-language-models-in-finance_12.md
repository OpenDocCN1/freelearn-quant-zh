# 11 伦理、治理与合规：面向金融大语言模型的综合框架

金融大语言模型可以拓宽分析准入、加速研究和合规工作、改善客户服务，并支持更一致的决策流程。只有当系统可靠、可质疑、安全，并且治理与其使用后果成比例时，这些收益才具有经济意义。在金融领域，一个看似合理但错误的答案可能变成一项建议、一笔订单、一个授信决定、一份披露或一次控制失效；因此，治理决定了技术能力是转化为持久价值，还是成为未受管理的风险。

本章为金融大语言模型和代理系统的伦理、治理与合规构建了一套综合风险框架。它结合了范围受限的美国模型风险指南、跨领域人工智能风险管理、金融稳定问题，以及来自受控市场实验的新兴证据[美联储理事会等，2026年，美国国家标准与技术研究院，2023年、2024年，金融稳定委员会，2024年、2026年]。该框架围绕三个实质性支柱和一个四层实施架构构建。已记载的监管事件与受控实验、假设性压力测试和综合实施场景被清晰区分，以防示例材料被误认为历史事实。

本章将涵盖以下主题：

- 为何全面治理对有益的金融人工智能至关重要
- 理论基础与三支柱治理模型
- 受控市场实验确立了什么——以及没有确立什么
- 从基础设施到问责的四层实施架构
- 已记载、假设性和综合治理案例
- 与决策关联的指标与持续改进
- 未来挑战、公开问题与实用参考资源

## 11.1 全面治理的必要性

治理的核心目的不是阻止使用大语言模型。而是使有价值的应用可重复，同时将故障控制在可容忍范围内。金融机构已在安全与稳健、市场公正、消费者保护、隐私、运营韧性和高级管理层问责方面承担严苛义务。大语言模型并未取代这些义务；它们引入了义务可能被违反的新途径，包括缺乏依据的生成、提示诱导行为、无法追溯的数据使用、工具滥用、同模式响应以及对第三方模型的过度依赖。

大语言模型还与许多传统金融模型不同。其输出是从学习到的统计表示生成的，而非来自一组精简的显式方程，其行为在很大程度上可能取决于提示、检索上下文、工具、内存、解码设置和模型版本。因此，生产环境中的“模型”更好地被理解为一个社会技术系统，包括基础模型、数据和检索管道、提示、工具、接口、用户、控制以及组织激励。仅对基础模型进行验证，并不等于对部署系统的验证。

因此，治理挑战比准确性更广泛。一个系统平均准确仍可能不可接受，因为其罕见错误可能严重，其解释可能具有误导性，其行为在更新后发生变化，其数据血统不完整，或其行为在压力期间与其他系统相关联。反之，大语言模型不必完美才能有用，当它被限制在适当角色中、周围有验证措施，并且在未经额外控制的情况下被阻止采取不可逆行动时。

现有框架提供了重要基础，但必须结合仔细的范围分析。在美国，SR 26-2 取代了 SR 11-7 和 SR 21-8。它是非约束性的监管指导，预计对总资产超过 300 亿美元的银行组织最为相关，并通过排除简单算术工具和确定性规则软件来缩小模型定义。就本章而言最重要的是，脚注 3 明确将生成式人工智能和代理式人工智能模型排除在其范围之外，同时保留其对传统统计和定量模型以及非生成式、非代理式人工智能模型的适用性[美联储理事会等，2026年]。因此，本章仅对落入其正式边界的系统直接引用 SR 26-2，并在明确陈述类比时，才为大语言模型治理选用部分原则进行类比。机构自身的治理和风险管理实践，连同适用的行业法律和监管要求，仍然是超出该指导范围的生成式和代理式系统的控制基础。

混合架构仍可能产生验证依赖：当范围外的大语言模型为范围内的定量模型提供数据、特征、叙述或决策时，下游审查应审查该接口并将其记录在系统清单中。这是架构影响，而非 SR 26-2 正式范围的扩展。美国国家标准与技术研究院人工智能风险管理框架及其生成式人工智能分层对有效性、可靠性、安全性、透明度、隐私和有害内容风险进行了生命周期导向的处理[美国国家标准与技术研究院，2023年、2024年]。金融稳定委员会将第三方集中度、市场相关性、网络风险以及模型风险、数据质量和治理中的弱点识别为金融脆弱性的潜在渠道[金融稳定委员会，2024年、2026年]。FAIR 框架为整合这些问题提供了特定于金融的结构[Noguer i Alonso 和 Mendell，2025a,b]。

由此产生的目标是相称治理：控制应随自主性、规模、速度、不可逆性、客户影响和系统相关性而加强。起草备忘录的研究助理不需要与能够提交订单、变更投资组合、批准授信或直接与客户沟通的代理相同的控制。因此，下面的框架从重要性和预期用途开始，而不仅仅是模型标签。

## 11.2 理论基础：构建多维框架

连贯的治理框架必须将技术属性与金融后果相连接，再连接到可问责的决策。本节首先解释了为何需要多个专业学科，然后将其贡献组织成三个相互依赖的支柱。其目的不是创建另一份抽象清单，而是提供一种从模型行为到运营控制、风险限额和法律责任的转译机制。

### 11.2.1 学科的融合

有效的金融大语言模型治理需要结构化的跨学科协作。计算机科学家和机器学习工程师贡献架构、训练动态、评估、安全和技术限制方面的知识。定量研究员和金融工程师贡献市场微观结构、定价、执行、投资组合、流动性和系统性风险方面的专业知识。金融分析师和投资组合经理就信息质量和经济相关性提供领域判断。行为经济学家研究决策偏差和交互效应。法律、合规、隐私和伦理专家解释有关公平性、披露、问责、客户结果和补救的义务。

挑战不仅仅是将这些专家放在同一个委员会中。机构需要一种共享语言，使技术观察能够成为风险决策。例如，对提示模板的敏感性必须转译为可衡量的敞口；敞口必须指定负责人和阈值；阈值必须触发定义的行动，例如路由至人工审查、降低限额、回滚模型版本或暂停工具。

我们将这种转译过程称为认知桥接。它由学科之间的文档化接口组成：模型卡片和系统清单用于技术到风险的转译；风险分类法和场景库用于风险到业务的转译；控制映射、问责矩阵和证据包用于业务到监管的转译。认知桥接的价值在于实用性：它防止漏洞始终只是从未触及有权更改生产环境行为的人员的技术观察。





### 11.2.2 The three-pillar model

The framework rests on three pillars. None is sufficient in isolation. Technical robustness without behavioral oversight can produce a stable system that acts in undesirable ways; behavioral monitoring without accountability can detect problems that no one owns; and ethical or regulatory principles without engineering controls remain aspirational.

### Pillar I: Technical robustness and operational excellence

This pillar combines model engineering, data governance, cybersecurity, and operational resilience. It covers data provenance, retrieval quality, versioning, access control, testing, observability, fallback modes, incident response, and recovery. It also requires that the complete deployed system, not only the base model, be reproducible and auditable. The relevant unit of control includes prompts, retrieval indexes, tools, permissions, memory, and post-processing rules.

A central design principle is defensive AI development: systems should be constructed with explicit assumptions about where they may fail. Defensive development includes adversarial and out-of-distribution testing, uncertainty and abstention mechanisms, least-privilege tool access, deterministic validation of critical calculations, and hard limits that cannot be modified by the model itself. The objective is graceful degradation rather than the unrealistic promise of error-free generation [National Institute of Standards and Technology, 2024, Vassilev et al., 2025].

### Pillar II: Behavioral governance and market dynamics

This pillar addresses how LLM-based agents behave over time and in interaction with other participants. Controlled market studies show that behavior depends materially on model family, role instructions, prompts, information, and experimental design. Some studies find more fundamental-value-oriented behavior than in human experimental markets, while others document extrapolative beliefs, disposition effects, bubbles, or prompt-induced common behavior [Henning et al., 2025, Lopez-Lira, 2025, Ouyang and Sui, 2026, Anand et al., 2026]. These findings are not contradictory; they show why behavior must be measured for the particular system and environment rather than inferred from the label “LLM.”

Behavioral governance monitors decision distributions, calibration, abstention, sensitivity to prompt and context changes, adaptation after distribution shifts, tool-use patterns, and interactions among agents. The Financial Stability Board treats market correlation as a potential AI-related vulnerability, and broader algorithmic research shows that independently learning agents can converge to strategically similar behavior in controlled environments [Financial Stability Board, 2024, Calvano et al., 2020]. This does not establish that LLMs have caused a real market-wide coordination event. It establishes a credible common-mode risk that merits testing, diversity analysis, and escalation rules.

Diversity should therefore be measured behaviorally rather than cosmetically. Two systems built by different vendors may still be strategically redundant if they rely on similar base model data, prompts, objectives, and risk signals. Conversely, two instances of the same base model may behave differently when their information sets and constraints differ. Institutions should evaluate cross-play, response correlation, shared dependencies, and concentration in providers or data sources before treating model variety as risk diversification.

### Pillar III: Ethical alignment and regulatory compliance

The third pillar ensures that technical capability and market behavior remain compatible with law, institutional values, customer interests, and public trust. It includes fairness, privacy, transparency, explainability, contestability, recordkeeping, market integrity, and clear assignment of responsibility. The same model may require different controls across use cases: a trading-support tool raises market-integrity and operational-risk questions, while a credit or insurance system may also raise discrimination, adverse-action, and individual-rights concerns.

Regulatory regimes differ in form and scope. The EU AI Act does not classify every financial AI application as high-risk. It identifies specific uses, including certain creditworthiness assessments and risk assessment or pricing in life and health insurance, while other financial uses must be assessed under their actual function and applicable sectoral law [European Parliament and Council of the European Union, 2024]. The UK Financial Conduct Authority continues to use an outcomes-focused, principles-based and technology-neutral approach, applying existing rules to AI rather than creating a separate rulebook for every technique [Financial Conduct Authority, 2025]. In the United States, model-risk, securities, consumer-protection, privacy, and other obligations remain distributed across sectoral authorities [Board of Governors of the Federal Reserve System et al., 2026, U.S. Securities and Exchange Commission, 2024].

This pillar also covers _AI washing_: materially overstating, mischaracterizing, or failing to substantiate an organization’s use of AI or the capabilities of an AI-enabled product. AI washing is a governance failure because marketing, investor communications, regulatory filings, and product descriptions are part of the control environment. In March 2024, the U.S. Securities and Exchange Commission settled charges against Delphia (USA) Inc. and Global Predictions Inc. concerning false or misleading AI-related claims; the firms agreed to civil penalties totaling $400,000 [U.S. Securities and Exchange Commission, 2024]. Capability claims should therefore be supported by technical evidence, reviewed by compliance, and kept consistent across public communications.

## 11.3 Empirical evidence: Learning from market experiments

Empirical evidence is necessary to turn governance principles into testable controls, but the evidence base must be interpreted carefully. Most studies of LLM agents in endogenous financial markets are recent, simulation-based, and often preprint-stage. They are valuable for identifying mechanisms, constructing stress tests, and revealing sensitivity to design choices. They do not by themselves estimate the frequency or magnitude of failures in live financial markets.

This section therefore separates findings that are directly supported by controlled experiments from governance inferences that require additional validation. Numerical claims are reported only when tied to a specific cited design; no simulated result is presented as a documented historical market event.

### 11.3.1 What controlled market studies establish

Controlled market studies are useful because they test LLM agents in simplified but observable trading environments. They do not prove how agents will behave in real markets, but they help identify behavioral patterns, prompt sensitivity, coordination risks, and governance questions that may be difficult to observe in live deployments. The studies below show why financial institutions should treat LLM-agent behavior as something to validate empirically rather than assume from human-market analogies.

**LLM traders do not simply replicate human traders.** Henning et al. compare LLM agents with human participants in a classic experimental-asset-market design. Their LLM markets generally remain closer to fundamental value and produce weaker bubbles than human markets, with lower variance and more fundamentals-oriented reasoning [Henning et al., 2025]. The governance implication is not that LLMs are universally more rational. It is that human behavioral assumptions cannot be transferred mechanically to LLM agents, and synthetic markets populated by LLMs require validation against the phenomenon they are intended to model.

**Prompts and roles can create persistent strategy patterns.** Lopez-Lira develops a simulated limit-order market in which heterogeneous LLM agents can act as value investors, momentum traders, market makers, or other strategy types. The study shows that natural-language role instructions can produce consistent strategy adherence and can generate correlated behavior that affects market outcomes [Lopez-Lira, 2025]. For governance, prompts and role definitions must therefore be version-controlled and treated as model components rather than informal configuration text.

**Behavioral tendencies can aggregate into market dynamics.** Ouyang and Sui document disposition effects and extrapolative beliefs in an experimental market populated by LLM agents. They also show that targeted prompt interventions can amplify or suppress particular mechanisms and alter bubble magnitude [Ouyang and Sui, 2026]. This result makes prompt sensitivity a causal control question: an institution should know whether small changes in framing, memory, or system instructions materially alter orders, risk taking, or liquidity provision.

**System design matters for financial-stability outcomes.** In a simulated mutual-fund redemption game, Anand et al. find that different AI technologies generate different coordination and redemption profiles [Anand et al., 2026]. The broader lesson is that “AI exposure” is not a single risk factor. Architecture, training objective, information, memory, action constraints, and interaction structure all affect system-level outcomes.





### 11.3.2 治理影响与证据边界

上述受控研究对于识别可能风险很有用，但不应将其视为实时市场行为的直接证据。其主要价值在于展示需要治理控制、验证测试和证据边界的地方。以下影响将研究发现转化为金融机构在部署或依赖 LLM 代理系统前应审查的实际领域。

- **制度与分布偏移风险：** 受控冲击处理显示，当基本面或信息发生变化时，行为可能会发生改变，但不存在 LLM 适应所需的通用周期数量。机构应通过带版本的变点检测、压力测试和分布外测试，为每个用例估计适应与恢复情况。仅仅因为系统仍在持续输出流畅结果，就不应让其在经过验证的领域之外运行。

- **共同模式与协调风险：** 共享提示词、目标、数据、供应商和市场信号可能导致独立部署的系统做出类似反应。适当的控制不是无依据的通用相关系数阈值。它是一个基线与行动框架：在正常和压力条件下测量策略和订单流相关性；识别共同依赖；定义机构特定的限制；并将违规与具体行动挂钩，例如限额削减、策略多样化、额外人工审查或暂时暂停。

- **语义与输入通道风险：** 没有可辩护的依据，对 LLM 代理解读模糊的中央银行措辞的方式应用通用百分比。支持的结论更为广泛：生成式系统可能对措辞、检索到的上下文、提示词注入和冲突指令敏感。因此，面向市场的系统需要语义扰动测试、来源身份验证、来源追踪检查，以及在证据模糊时采用弃权路径 [美国国家标准与技术研究院, 2024, Vassilev et al., 2025]。

- **外部效度边界：** 模拟证明在特定条件下可以产生一种机制；但它并不确立该机制在实时市场中已发生，或其效应大小将保持不变地迁移。因此，实验报告应披露模型版本、提示词、工具、信息集、解码参数、复现次数、无效输出、市场规则和敏感性分析。治理委员会在使用证据制定政策前，应将其分类为文档化运营证据、受控实验证据、假设性压力测试或专家判断。

### 11.3.3 将研究转化为实践

受控研究与生产环境之间的差距可通过分阶段部署、受限适应和协作测试来缩小。这些机制将新兴证据转化为控制措施，而不主张比证据所支持的更高确定性。以下实践为从研究发现到受控部署提供了实用桥梁：

- **分级部署协议：** 新系统应从离线评估推进到历史重演、沙箱测试、模拟交易或影子模式、受严格限制的生产环境，然后才能进行更广泛的部署。阶段间的提升应取决于预先指定的证据，而非日历时间。每个阶段不仅应测试任务性能，还应测试校准、提示词敏感性、安全性、运营韧性、限制合规性以及压力下的行为 [美联储理事会等, 2026, 美国国家标准与技术研究院, 2023]。

- **受控变更与持续学习：** 任何会更改提示词、检索数据、工具、内存、参数或决策策略的机制都可能改变系统行为。因此，变更应带版本、经过测试、获批且可逆。在线或持续学习应受明确风险限制约束，并针对重大漂移、新能力或预期用途变更设置重新验证触发器。未经独立授权流程，不应允许系统重写其自身的风险偏好、权限或控制逻辑。

- **跨机构研究协作：** 机构在实施上竞争，但在稳定市场和可信治理方面有共同利益。竞前协作可支持共同事件分类法、压力场景、语义挑战集、第三方依赖映射以及总体行为的隐私保护式测量。金融稳定委员会最近的工作强调信息共享、全组织治理、生命周期控制以及对市场相关性和第三方集中度的监控 [金融稳定委员会, 2024, 2026]。

## 11.4 实施架构：从理论到实践

三大支柱定义了必须治理的内容；实施架构定义了控制措施在哪里运作。该架构有四层，如图 11.1 所示。证据和保证从基础设施和行为向上流向风险和问责，而政策、限制、权限和升级规则则向下流动。各层是相互依赖的，而非按顺序排列的成熟度等级。

实用架构还应将生成式推理与不可逆执行分离。在许多用例中，LLM 最适合用于研究、解释、监控或推荐功能，而确定性服务则验证计算并强制执行权限、持仓限制、订单类型和客户影响规则。LLM 只有在完整路径满足可衡量的、特定于用例的延迟和可靠性预算后，才应进入硬实时执行路径。

## 最终阶段更新：作为可撤销预算的权限

一种最小的治理改进是将权限视为阶梯而非二元开关。系统可以是分析助手、决策支持工具、受监督的提议者、受限分配器、受限执行器或自适应机构；这些状态之间的提升应取决于证据，当监控指标离开其批准范围时，降级应自动发生。对于承载资本的工作流程，紧凑的权限预算可以写成

```
\frac{B_t^{\text{auth}}}{B_{\max}^{\text{auth}}} = K_t \quad \prod_{j \in \{\text{cal,reg,data,ops,gov}\}} \quad \rho_{j,t}, \quad K_t \in \{0, 1\}, \quad 0 \le \rho_{j,t} \le 1.
```

各因子分别代表校准、制度熟悉度、数据完整性、运营健康和治理状态。硬闸门 \( K_t \) 是杀伤开关：当它为零时，权限会坍缩为弃权或人工审批。该公式不是普遍的监管要求；它是一个使有限自主性可衡量且可撤销的实施模式 [Noguer i Alonso 和 Pacheco Azmar, 2026]。

| 层级 | 描述 |
|------|------|
| 4 责任、审计与治理 | 指定负责人、升级权限、杀伤开关 |
| 3 风险决策、限制与升级 | 系统在无需人工干预下可执行的操作 |
| 2 行为监控与控制 | 漂移、忠实性、覆盖率 |
| 1 基础设施 | 数据血缘、版本控制、权限、不可更改日志 |

一层只能和它下面的一层一样好：对无法版本化的数据进行行为监控是空中楼阁，指向无人拥有的日志触发的升级规则是不存在的规则。

图 11.1：四层治理架构。

证据向上流动，政策向下流动，这种顺序并非装饰——它是依赖关系。机构往往先搭建第 3 层，因为限制和升级规则是能够让委员会放心的部分，但在检查时才发现它们无法提供任何一项证据，因为第 1 层从未搭建过。从底部开始。这是最不引人注目的幻灯片，也是唯一站得住脚的幻灯片。

### 11.4.1 第 1 层：基础设施

基础层提供了更高级别治理可信所需的技术条件。它涵盖计算与模型服务、数据和检索架构、身份与访问管理、安全开发、可观测性、版本控制、业务连续性和恢复。

以下基础设施组件构成了可靠且可治理的金融 AI 系统的基线：

- **计算、服务与韧性：** 容量、延迟、可用性和恢复目标应根据用例确定，而非声明为通用数字。客户服务助手、夜间监控流程和做市组件具有不同的服务级别要求。关键路径需要冗余、健康检查、确定性回退行为以及在不禁用基本金融控制的情况下禁用模型依赖功能的能力。

- **数据架构与治理：** 每一条重要输出都应可追溯至其模型版本、系统和用户指令、检索到的来源、工具调用、相关数据、后处理和审批状态。数据质量检查应涵盖新鲜度、完整性、泄露、冲突记录、许可和代表性。检索增强系统需要源级控制，因为行为良好的模型仍可能从过时、恶意或无关的检索内容中生成错误决策。保护敏感数据可能需要差分隐私、联邦学习、安全聚合或安全多方计算。这些技术强大但并非免费。差分隐私引入了隐私-实用性权衡；联邦学习增加了通信轮次和非独立同分布数据挑战；加密聚合或多方计算增加了通信、内存和计算开销 [Dwork 和 Roth, 2014, Kairouz et al., 2021, Bonawitz et al., 2017, Evans et al., 2018]。它们通常非常适合异步训练、跨机构分析和批量模型开发。它们在超低延迟交易路径中的使用应由经过衡量的端到端性能来证明，而非基于架构偏好。

- **模型开发与部署管道：** 部署单元应包括代码、模型、提示词模板、检索配置、工具、权限、数据集、评估套件和依赖版本。质量门应测试功能正确性、安全性、数据泄露、行为稳定性、性能和回滚。当血缘不完整或关键测试失败时，应阻止生产提升，无论标题准确性如何。





### 11.4.2 第 2 层：行为监控与控制

第二层观察已部署系统的实际行为。监控应将生产行为与验证范围进行比较，并应围绕操作员可以采取的决策进行设计，而不仅仅围绕仪表板。

以下监控和控制实践有助于机构在部署后检测行为漂移、协调风险和不安全的工具使用：

- **决策与交互分析：**监控应涵盖任务成功、错误严重性、校准、弃权、提示和上下文敏感性、工具使用模式、策略违规和分布漂移。对于代理，事件日志应重建从观察、推理、工具选择、动作、外部响应到后续适应的序列。当隐私或保密性限制日志记录时，机构应记录哪些证据不可用以及如何降低该剩余风险。

- **跨代理和共同依赖监控：**机构应评估内部代理之间以及（在可观察时）与更广泛市场行为的相关性，涵盖建议、订单、风险承担和流动性撤出。依赖图谱应识别共享模型提供商、云服务、数据集、检索馈送和优化目标。相关性警报应触发调查和相称行动；不应将其视为串通或意图的证据。

- **语义鲁棒性和工具使用验证：**定期挑战测试应包括改写、歧义、矛盾来源、过时信息、提示注入、角色混淆以及嵌入在检索文档中的指令。来自不可信来源的输入应隔离并进行身份验证，工具调用应根据明确的策略引擎进行检查。高影响操作应要求结构化参数和确定性验证，而不仅仅是自由形式的模型文本 [National Institute of Standards and Technology, 2024, Vassilev et al., 2025].

### 11.4.3 第 3 层：风险管理集成

该层将 AI 特定证据转化为熟悉的财务风险流程。它确保模型行为影响限额、压力测试、资本和流动性规划、供应商管理、事件响应和业务决策。

以下实践将 AI 特定风险与既定的财务风险管理流程联系起来：

- **统一风险分类：**传统的市场、信用、流动性、运营、行为、合规、网络和声誉风险仍然相关。AI 特定机制应映射到这些后果，而不是单独管理在孤岛中。例如，未经支持的生成可能造成合规行为、法律或市场风险；提示注入可能造成网络和运营风险；同模式行为可能造成市场和流动性风险；而提供商集中可能造成第三方和系统性风险 [Financial Stability Board, 2024, 2026].

- 增强压力测试：压力测试应将金融冲击与 AI 失效机制相结合。场景可能包括检索损坏、模型提供商丧失、多个代理同时回退、突然的制度变化、看起来官方的对抗性通信、人工审批者不可用或模型生成订单的快速增长。测试应评估单个系统和机构的综合响应，包括保障措施本身是否会造成流动性撤出或运营瓶颈。

- 风险偏好、限额和事件管理：每个重要用例都应有文档化的风险接受、负责的业务所有者、独立挑战以及定量或定性限额。限额应与诸如降低自主权、缩小允许任务、降低财务敞口、回滚版本或暂停系统等行动相关联。事件手册应定义通知、证据保存、客户补救、监管报告和事后再验证 [Board of Governors of the Federal Reserve System et al., 2026].

### 11.4.4 第 4 层：伦理与法规对齐

第四层确保系统的目的、结果和披露对客户、市场、董事会、法院和监管机构保持可接受。它将法律和伦理义务转化为可测试的控制措施，并为剩余风险分配责任。

以下实践帮助机构将伦理和监管期望转化为运营控制措施：

- 可解释性与透明度：解释应匹配受众和决策。开发人员需要技术诊断；验证者需要关于局限性和稳定性的证据；客户需要可理解的理由和对重大结果提出质疑的渠道；监管机构或法院可能要求可复现的审计追踪。解释不能仅仅听起来合理：它应忠实于系统使用的实际证据和流程。

- 公平性、客户结果和补救：高影响金融用例需要在相关群体、产品、渠道和市场条件下进行测试。由于公平性标准可能相互冲突，机构应记录所选指标的法律和政策依据、受影响人群、可接受的权衡以及补救路径。只有当审查者拥有足够的信息、权限、时间和培训来改变结果时，人工审查才具有意义。

- 责任与治理结构：责任应遵循决策权。开发人员负责实施质量；独立验证者挑战假设和证据；风险与合规职能定义或监督限额；业务所有者接受用例风险；高级管理层和董事会监督重大总敞口。第三方采购不会转移责任：合同应提供文档访问权限、变更通知、事件合作、绩效证据和退出安排 [Board of Governors of the Federal Reserve System et al., 2026; Financial Stability Board, 2026].

- 监管映射与真实披露：控制映射应将每个用例与适用的财务、数据、消费者、市场和 AI 特定义务联系起来。关于 AI 能力、性能、自主性或监管状态的公开声明应由保留的证据支持，并通过与其他重要披露相同的控制流程进行审查。此要求直接解决 AI-washing 风险 [U.S. Securities and Exchange Commission, 2024].

## 11.5 案例研究与治理短篇：从实施中学习

只有在证据地位明确时，示例才有用。以下第一个示例是假设性压力测试，第二个是复合跨境实施短篇，第三个是有记录的执法行动。这种标记防止说明性场景被误认为历史事件。

### 11.5.1 假设性压力测试：语义共识级联

以下压力测试说明了语义歧义、相似模型配置和同步响应如何结合形成市场层面的反馈循环。该场景是假设性的，应视为治理测试用例，而不是有记录市场事件的重建。

- 场景状态：以下事件是假设性压力测试，而不是有记录市场事件的重建。并未声称 2024 年 3 月 15 日发生过 3% 的 AI 驱动闪崩。中央银行发布一份包含在政策语境下真正存在歧义的语言的声明。多家机构使用相似的模型提供商、检索馈送、情绪提示和风险信号。它们的代理将该声明朝同一方向分类，同时降低风险，并将由此产生的价格波动视为对初始解读的确认。流动性提供者也因波动性控制触发而收窄或撤回报价。

- 控制响应：治理良好的机构不会依赖单一情绪标签。来源身份验证将确认官方声明；语义扰动测试将比较跨改写和模型的解读；置信度和分歧度量将决定是否允许自主行动；确定性市场风险控制将限制头寸变化和流动性消耗。如果出现同模式行为或异常市场影响，系统将降低自主权，扩大人工审查范围，并保留证据以供事件分析。

- 教训：该场景测试了多个单独合理组件之间的交互作用。即使没有代理是恶意的，语义歧义、同步风险降低和基于波动性的撤出也可能形成反馈循环。因此，治理目标不仅仅是改进语言分类，而是防止不确定的解读演变为无界、不可逆的市场行动。





### 11.5.2 合成情景：跨境部署

The following vignette shows how a shared LLM platform might be governed across multiple jurisdictions. It is a composite example intended to illustrate control design, not a report of a named institution or verified deployment outcome.

下述情景展示了共享 LLM 平台如何在多个司法管辖区进行治理。这是一个合成示例，旨在说明控制设计，而非对特定机构或验证过的部署结果的报告。

- **Scenario status:** This is a composite implementation vignette assembled from common regulatory and operational requirements. It does not describe a named institution and does not report verified performance improvements.

- **场景状态：** 这是一个从常见监管和运营要求中组装而来的合成实施情景。它不描述特定机构，也不报告经验证的性能改进。

A financial group proposes a shared LLM platform for portfolio research, customer communications, and selected decision-support functions across several jurisdictions. The group first separates the use cases because their legal and operational risk differs. It then creates a common control baseline covering inventory, data lineage, model validation, security, human oversight, incident management, and third-party risk. Jurisdiction-specific modules add requirements for particular products, customer rights, documentation, disclosures, and regulatory reporting.

一个金融集团提议建立一个共享 LLM 平台，用于跨多个司法管辖区的投资组合研究、客户沟通以及精选的决策支持功能。该集团首先区分用例，因为其法律和运营风险不同。随后，它建立一个覆盖清单、数据血缘、模型验证、安全性、人工监督、事件管理和第三方风险的共同控制基线。特定于司法管辖区的模块针对特定产品、客户权利、文档、披露和监管报告增加了要求。

In the EU, the group assesses whether each use falls within an Annex III high-risk category and separately maps applicable financial-services and data-protection obligations [European Parliament and Council of the European Union, 2024].

在欧盟，该集团评估每项用途是否属于附录 III 高风险类别，并分别映射适用的金融服务和数据保护义务[欧洲议会和欧洲联盟理事会，2024]。

In the UK, it maps the system to existing FCA outcomes and accountability requirements under the regulator’s technology-neutral approach [Financial Conduct Authority, 2025].

在英国，它根据监管者的技术中立方法，将系统映射到现有的 FCA 结果和问责要求[金融行为监管局，2025]。

In the United States, it first determines whether any component is formally within SR 26-2; it does not describe a generative or agentic component as compliant with guidance that expressly excludes it. The group then applies its own proportionate controls and maps securities, consumer, privacy, marketing, and other applicable obligations [Board of Governors of the Federal Reserve System et al., 2026, U.S. Securities and Exchange Commission, 2024].

在美国，它首先确定任何组件是否正式属于 SR 26-2 的范围；它不会将生成式或代理组件描述为符合明确排除该组件的指南。随后，该集团应用其自身的相称控制措施，并映射证券、消费者、隐私、营销及其他适用义务[美联储理事会等，2026，美国证券交易委员会，2024]。

- **Implementation lesson:** The scalable solution is not a collection of unrelated local systems. It is a common evidence model with jurisdiction-specific control overlays. The organization should be able to show what the system does, which version is operating, what data and tools it uses, who approved it, what limitations remain, and which controls change by location and use case.

- **实施教训：** 可扩展的解决方案不是一系列无关的本地系统。它是带有特定于司法管辖区的控制叠加层的共同证据模型。组织应能够展示系统的功能、运行中的版本、使用的数据和工具、批准人、剩余的局限性，以及按地点和用例变化的控制措施。

### 11.5.3 有据执行：SEC AI 洗白案件

On March 18, 2024, the U.S. Securities and Exchange Commission announced settled charges against Delphia (USA) Inc. and Global Predictions Inc. for false or misleading statements concerning their purported use of AI. The SEC reported aggregate civil penalties of $400,000, consisting of $225,000 for Delphia and $175,000 for Global Predictions [U.S. Securities and Exchange Commission, 2024].

2024 年 3 月 18 日，美国证券交易委员会宣布，就 Delphia (USA) Inc. 和 Global Predictions Inc. 就其所谓的 AI 使用所作的虚假或误导性陈述，对其达成和解指控。SEC 报告的累计民事罚款为 400,000 美元，其中 Delphia 225,000 美元，Global Predictions 175,000 美元[美国证券交易委员会，2024]。

The governance lesson is broader than marketing review. Claims about training data, forecasting capability, personalization, regulatory status, or “AI-driven” performance must be traceable to evidence. Product, engineering, legal, compliance, investor-relations, and marketing teams should use one approved capability record so that technical reality and public representation cannot diverge. Unsupported capability claims should be treated as a control breach even when the underlying product otherwise functions.

治理教训不仅限于营销审查。关于训练数据、预测能力、个性化、监管状态或“AI 驱动”性能的声明必须可追溯到证据。产品、工程、法律、合规、投资者关系和营销团队应使用一个经批准的能力记录，以确保技术现实与公众表述不会背离。即使底层产品运作正常，未经证实的能力声明也应被视为控制违规。

## 11.6 衡量成功：KPI 与持续改进

Metrics are useful only when they change decisions. A governance dashboard should therefore begin with a risk hypothesis and specify the measurement, threshold or review criterion, accountable owner, required response, and retained evidence. A number without an owner or action is an observation, not a control.

指标只有在能改变决策时才有用。因此，治理仪表板应从风险假设开始，并明确规定测量指标、阈值或审查标准、负责人、所需响应和保留证据。没有负责人或行动的数字只是观察，而非控制措施。

The framework below replaces universal targets with use-case-specific baselines. Appropriate thresholds depend on materiality, market conditions, customer impact, autonomy, and the cost of false alarms. They should be approved before final evaluation, monitored after deployment, and revised only through a controlled process [Board of Governors of the Federal Reserve System et al., 2026, National Institute of Standards and Technology, 2023].

下面的框架用特定于用例的基线取代了通用目标。适当的阈值取决于重要性、市场条件、客户影响、自主性以及误报成本。它们应在最终评估前获得批准，在部署后进行监控，并仅通过受控过程进行修订[美联储理事会等，2026，美国国家标准与技术研究院，2023]。

### 11.6.1 决策关联型测量框架

A useful measurement framework should connect each metric to a possible governance or operational response. The goal is not only to monitor model behavior, but to define when the institution should intervene, restrict use, escalate review, or change system configuration. The following categories link measurement areas to the decisions they should support.

一个有用的测量框架应将每个指标与可能的治理或运营响应相关联。其目标不仅是监控模型行为，更是界定机构何时应进行干预、限制使用、升级审查或更改系统配置。以下类别将测量领域与其应支持的决策相关联。

### Behavioral performance: from model behavior to intervention

### 行为性能：从模型行为到干预

Behavioral metrics should capture calibration, abstention, sensitivity to prompt and retrieval changes, adaptation after distribution shifts, strategy stability, and cross-agent correlation. Each metric should have a defined response. For example, excessive prompt sensitivity may trigger a prompt rollback or mandatory human review; abnormal cross-agent correlation may trigger lower limits, diversification analysis, or a stress test; and deteriorating calibration may narrow the system’s permitted task.

行为指标应捕获校准、弃权、对提示和检索变化的敏感性、分布转移后的适应性、策略稳定性和跨代理相关性。每个指标都应有明确的响应。例如，过度的提示敏感性可能触发提示回滚或强制人工审查；异常的跨代理相关性可能触发降低限额、多样化分析或压力测试；而校准恶化可能会收紧系统允许的任务范围。

### Risk and financial impact: from exposure to risk appetite

### 风险与财务影响：从风险敞口到风险偏好

Risk measures should connect AI behavior to market, credit, liquidity, operational, conduct, and reputational consequences. Relevant indicators include limit breaches, loss severity, erroneous or unsupported actions, market impact, liquidity consumption, concentration in providers or data sources, customer remediation, and incident recurrence. Risk-adjusted business value should be reported alongside gross performance so that apparent gains are not achieved by accepting hidden tail exposure.

风险措施应将 AI 行为与市场、信贷、流动性、运营、合规和声誉后果相关联。相关指标包括限额违规、损失严重性、错误或未经支持的行为、市场影响、流动性消耗、提供商或数据源集中度、客户补救和事件复发。风险调整后的业务价值应与毛绩效一起报告，以免表面收益是通过接受隐藏的尾部风险敞口而实现的。

### Operational resilience: from service levels to fallback behavior

### 运营韧性：从服务水平到降级行为

Availability, latency, throughput, recovery time, failure detection, and rollback success should be measured against the system’s service-level objective. The critical question is not whether every component remains available, but whether the institution continues to perform essential functions safely when the model, retrieval layer, tool, cloud service, or human approval channel fails. Tests should include degraded and disconnected modes.

可用性、延迟、吞吐量、恢复时间、故障检测和回滚成功率应针对系统的服务级别目标进行衡量。关键问题不在于每个组件是否保持可用，而在于模型、检索层、工具、云服务或人工审批渠道失效时，机构是否仍能安全地执行基本功能。测试应包括降级模式和断开连接模式。

### Compliance and ethics: From policy to auditable evidence

### 合规与伦理：从政策到可审计的证据

Metrics should cover inventory completeness, data-lineage coverage, validation status, unresolved findings, explanation and notice delivery, fairness assessments, human-review outcomes, complaints, appeals, policy exceptions, public-claim substantiation, and timeliness of regulatory or internal reporting. A high score in one dimension should not offset a severe breach in another; material violations require direct escalation rather than averaging.

指标应涵盖清单完整性、数据血缘覆盖率、验证状态、未解决的发现、说明和通知交付、公平性评估、人工审查结果、投诉、申诉、政策例外、公开声明的证实以及监管或内部报告的及时性。一个维度的高分不应抵消另一个维度的严重违规；重大违规需要直接升级，而非取平均值。

### 11.6.2 定性评估维度

Quantitative measures should be complemented by structured qualitative review. Boards, risk committees, validators, frontline users, customers, auditors, and regulators observe different parts of the system and can identify weaknesses that aggregate metrics conceal.

定量措施应辅以结构化的定性审查。董事会、风险委员会、验证者、一线用户、客户、审计师和监管机构观察系统的不同部分，并能发现汇总指标掩盖的弱点。

The following qualitative dimensions help institutions assess whether governance is working in practice, not just whether metrics are being reported:

以下定性维度有助于机构评估治理在实践中是否有效，而不仅仅是指标是否在报告：

- **Stakeholder confidence and effective challenge:** Surveys and interviews should test whether stakeholders understand the system’s purpose, limitations, and escalation routes. Confidence should not be interpreted as popularity alone. A healthy governance culture allows users and reviewers to challenge the system, document disagreement, and stop deployment without organizational penalty.

- **利益相关者信心与有效挑战：** 调查和访问应测试利益相关者是否了解系统的目的、局限性和升级路径。信心不应仅被解释为受欢迎程度。健康的治理文化允许用户和审查者挑战系统、记录分歧，并在不受组织惩罚的情况下停止部署。

- **Learning and adaptation:** Institutions should evaluate how quickly incidents and near misses lead to improved tests, controls, documentation, and training. Knowledge-transfer measures should assess whether expertise is distributed across the organization or concentrated in a few individuals or vendors. Repeated incidents with the same underlying mechanism indicate that monitoring exists without organizational learning.

- **学习与适应：** 机构应评估事件和未遂事件多快能导致改进的测试、控制措施、文档和培训。知识转移措施应评估专业知识是在整个组织中分布，还是集中在少数个人或供应商手中。具有相同底层机制的重复事件表明存在监控，但缺乏组织学习。

- **Regulatory and ethical alignment:** Qualitative review should examine whether system use remains consistent with institutional purpose, customer expectations, and evolving law. It should also evaluate whether explanations are meaningful, whether human oversight can genuinely alter decisions, whether affected individuals can seek redress, and whether public claims accurately represent the system.

- **监管与伦理一致性：** 定性审查应检查系统使用是否仍与机构目的、客户期望和不断发展的法律保持一致。它还应评估说明是否有意义、人工监督是否能真正改变决策、受影响的个人是否能寻求补救，以及公开声明是否准确地代表了系统。

### 11.6.3 持续改进循环

Measurement should feed a closed loop: observe behavior, diagnose causes, decide an intervention, implement through controlled change, retest, and document residual risk. Periodic review should also ask whether the use case remains necessary and proportionate. Retirement is a valid governance outcome when benefits no longer justify complexity or when a simpler method performs the task more reliably.

测量应形成闭环：观察行为、诊断原因、决定干预措施、通过受控变更实施、重新测试并记录剩余风险。定期审查还应询问用例是否仍然必要且相称。当收益不再证明复杂性的合理性，或当更简单的方法能更可靠地完成任务时，退役是一种有效的治理结果。

## 11.7 未来方向：演进框架

Financial AI is advancing from isolated assistants toward multimodal, tool-using, persistent, and increasingly autonomous systems. The governance challenge will therefore shift from reviewing individual outputs to supervising long-lived systems that learn from interaction,

金融 AI 正在从孤立的助手向多模态、使用工具、持续存在且日益自主的系统发展。因此，治理挑战将从审查个别输出转变为监督从交互中学习的长寿命系统，



依赖外部提供商，并相互影响。本节强调了控制和研究必须发展的主要方向。

### 11.7.1 地平线上的新兴挑战

以下发展可能会同时增加机遇和治理复杂性。

机构应将这些发展作为前瞻性风险评估和战略规划的一部分进行监测：

- **智能体与多模态能力：** 结合文本、图像、音频、代码、数据库、记忆和外部工具的智能体可以完成更丰富的工作流，但每增加一种模态和工具，就会产生新的输入、权限和故障通道。因此，评估必须针对特定任务和系统进行，并明确测试长时域行为、记忆污染、工具滥用以及部分失败后的恢复 [美国国家标准与技术研究院, 2024, 金融稳定委员会, 2026]。

- **第三方与基础设施集中：** 许多机构可能依赖于少数模型、云、芯片、数据和网络安全提供商。即使机构审慎管理自身应用程序，集中化也可能导致共同漏洞和关联性中断。退出规划、可移植性、合同审计权、多提供商测试和行业级监测将变得越来越重要 [金融稳定委员会, 2024, 2026]。

- **市场结构转型：** 随着更多决策被自动化，市场质量可能取决于异质智能体、确定性算法、人类交易员和交易所控制之间的相互作用。相关问题涉及价格发现、流动性提供、反馈交易、共同数据依赖，以及在同时压力下的保障措施行为。仿真证据应为这些问题提供信息，但实时市场结论需要仔细的因果和监管分析。

- **监管与社会演变：** 监管方法将继续在司法管辖区和用例之间存在差异。机构将需要动态控制图，而非静态合规摘要。同时，即使正式规则保持技术中立，客户对披露、隐私、可争议性和人工访问的期望也可能变得更严格。

### 11.7.2 研究与治理创新优先事项

优先领域包括面向使用工具的智能体的忠实解释、提示和记忆干预的因果评估、分布转移下的稳健监测、保护隐私的跨机构风险测量、标准化事件分类法，以及在不要求企业披露专有策略的情况下测量共同模式行为的方法。监管沙盒和共享测试环境可以提供帮助，但前提是其证据有据可查，且其范围限制明确。

治理创新还应关注机器可读控制。权限、限制、经批准的数据来源、披露要求和升级规则，越来越多地可以表示为在模型之外强制执行的策略。这在生成灵活性与机构权威之间创造了一种分离：模型可以提出建议，但由独立治理的控制平面决定可以执行的内容。

## 11.8 总结

金融LLM可以通过更快的分析、改善的信息获取、更具可扩展性的控制以及新的决策支持形式来创造真正的价值。治理的目的是在不确定性、市场变化、第三方依赖、人为错误和对抗行为等现实条件下保留这些收益。

本章开发的框架将三大支柱——技术稳健性、行为治理、以及伦理与监管一致性——与四层实施架构相结合。它将已部署系统，而非单纯的基础模型，视为控制单元。它还将证据与决策相关联：提示和工具进行版本管理；行为针对验证过的范围进行监测；风险指标触发预定义操作；责任随权力而定。

一个核心的治理教训是，需要诚实地标记证据。受控实验可以揭示机制，但它们并非历史事件。假设性压力测试可以提高准备程度，但不得被叙述为事实。复合性小故事可以说明架构，但不应声称未经验证的结果。有据可查的执法行动，例如美国证券交易委员会（SEC）的AI洗白案件，应与上述三者区分开来。这种证据纪律本身也是负责任的人工智能治理的一部分。

没有任何框架可以消除不确定性，治理也不应被简化为合规文书工作。决定性的检验在于：机构是否能够解释其系统获准做什么，展示其如何运行，在假设失效时及时发现，在危害扩大前进行干预，保留证据，并为结果分配责任。能够做到这一点的机构将更有信心地进行创新；那些做不到的机构可能会同时扩大能力与脆弱性。

在下一章中，我们将探讨如何将金融领域的LLM与数学方法和经验证据相结合，以解决复杂的金融问题。

## 11.9 未解决的问题与研究议程

若干基础问题仍未解决。它们应被视为研究议程，而非推迟负责任部署的理由。

### 实验证据应如何迁移到实时市场？

研究者需要原则性方法，以识别哪些机制能够经受住市场设计、模型版本、提示、信息结构、延迟和人类参与变化的考验。外部效度应被明确评估，而非从看似逼真的仿真中假定。

### 如何在不暴露专有系统的前提下测量共同模式风险？

市场稳定可能需要关于模型提供商、依赖关系、策略相关性和同时回退行为的汇总信息。悬而未决的问题是，如何在保护知识产权、保密性、隐私和竞争的同时，获得足够有信息量的度量。

### 机器速度下什么才算有意义的人类监督？

当审核者缺乏时间、信息或权限时，名义上的人类审批步骤可能几乎无法提供保护。研究应确定哪些决策可以安全地自动化，哪些需要事先约束，以及在何种情况下人类干预仅在组合、政策或系统层面有效，而非在交易层面。

### 自适应智能体应如何重新验证？

持续学习、记忆、工具获取和自我修改的工作流模糊了新观察与新模型之间的界限。机构需要实质性测试，以确定行为何时发生了足以要求重新批准、独立验证或回归有限部署的变化。

### 保护隐私的协作能否满足金融性能约束？

差分隐私、联邦学习、安全聚合和多方计算可以支持跨机构学习，但其效用、通信和延迟成本仍取决于具体用例。需要针对提供可信隐私保证而不掩盖验证或破坏时间敏感型操作的架构进行研究。

### AI供应链中如何分配问责？

已部署的服务可能结合基础模型提供商、云平台、数据供应商、集成商、金融机构和最终用户。合同可以分配职责，但监管机构和法院可能仍需要证据，以确定哪一方控制了设计、信息、权限、监测和公开陈述。

### 如何在执法之前检测AI洗白？

组织需要保证方法，将公开声明与技术清单、测试结果、训练和数据记录、实际模型使用以及有据可查的局限性进行比较。独立的保证标准可以缩小营销语言与运营现实之间的差距。

### 哪些事件应在企业和监管机构之间共享？

有用的事件分类法必须将无害的生成错误与对客户、市场、安全、隐私或系统产生后果的事件区分开来。未解决的设计问题是，如何共享足够的细节以实现集体学习，而不产生新的保密或安全风险。





## 11.10 精选标准、监管来源与研究资源

以下注释参考图涵盖了本章使用的主要资源，并为从业人员提供实施的结构化起点。它补充了本书的完整参考文献。

- **美国金融模型风险管理（范围限定）。** SR 26-2, 《模型风险管理修订指南》，取代 SR 11-7 和 SR 21-8。它不具约束力，预计在总资产超过 300 亿美元时最为相关，并明确排除生成式 AI 和代理式 AI 模型，同时继续适用于符合其定义的传统定量型及非生成式、非代理式 AI 模型。因此，它不是生成式 AI 合规标准；仅在完成正式范围分析后才可直接使用 [Board of Governors of the Federal Reserve System et al., 2026]。

- **跨行业 AI 风险管理。** NIST 的 AI RMF 围绕 Govern、Map、Measure 和 Manage 开展工作，而生成式 AI 简介则增加了针对生成式系统特有的风险和措施 [National Institute of Standards and Technology, 2023, 2024]。

- **金融稳定视角。** FSB 的 2024 年报告分析了 AI 相关脆弱性，其 2026 年咨询报告为金融机构的全组织治理和 AI 生命周期提出了健全做法 [Financial Stability Board, 2024, 2026]。

- **欧盟。** 欧盟法规 (EU) 2024/1689 提供了 AI 法案的基于风险的义务，包括可能被归类为高风险的特定附录 III 金融和保险用途 [European Parliament and Council of the European Union, 2024]。

- **英国金融服务。** FCA 的当前方法仍侧重于成果、基于原则且技术中立，将现有监管框架应用于 AI 赋能的金融服务 [Financial Conduct Authority, 2025]。

- **AI 洗白与真实声明。** SEC 的 2024 年 Delphia 和 Global Predictions 行动，提供了涉及虚假或误导性 AI 相关陈述的金融服务执法案例的文档化示例 [U.S. Securities and Exchange Commission, 2024]。

- **带 LLM 代理的实验金融市场。** Henning 等人、Lopez-Lira、Ouyang 和 Sui 以及 Anand 等人提供了关于行为分歧、提示条件策略、市场泡沫和系统设计效应的互补控制证据 [Henning et al., 2025, Lopez-Lira, 2025, Ouyang and Sui, 2026, Anand et al., 2026]。

- **算法交互与协调。** Calvano 等人提供了一项关于自主定价算法之间默示串通行为的基础控制性研究；该结果并非 LLM 特有，但与共模风险和交互风险相关 [Calvano et al., 2020]。

- **隐私保护学习。** 差分隐私、联邦学习、安全聚合和安全多方计算由 Dwork 和 Roth、Kairouz 等人、Bonawitz 等人以及 Evans 等人涵盖 [Dwork and Roth, 2014, Kairouz et al., 2021, Bonawitz et al., 2017, Evans et al., 2018]。

- **对抗机器学习。** NIST 的对抗机器学习分类法为逃避、投毒、隐私和滥用风险提供了术语，可纳入安全测试 [Vassilev et al., 2025]。

## 获取本书 PDF 版本及更多内容

扫描二维码（或访问 packtpub.com/unlock）。按书名搜索此书，确认版本，然后按照页面上的步骤操作。

![](img/c31705b361fdf81dee8826a2c4ee167f_385_0.png)

立即解锁

*Note: 请随身保管好发票。直接从 Packt 购买无需发票。*
