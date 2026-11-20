

这不是 CQL：Q 是 (s, a) 的函数，而保守项是在批次中*每个状态*上的动作上的 log-sum-exp，减去数据集自身动作在该状态下达到的价值。在扔掉状态的情况下对一袋采样动作平均 Q 值，并没有特别惩罚任何东西。

下面是离散动作集。对于连续动作，用 Kumar 等人的重要性采样估计值替换精确的 log-sum-exp。

```
def __init__(self, q_network, dataset, alpha=5.0, n_actions=None):
    self.q_network = q_network        # Q(s, .) -> [B, n_actions]
    self.dataset = dataset
    self.alpha = alpha
    self.n_actions = n_actions
def cql_loss(self, states, actions, rewards, next_states, dones):
    # 1. The ordinary TD / Bellman term.
    td_loss = self.standard_q_loss(
        states, actions, rewards, next_states, dones)
    # 2. The conservative term, evaluated AT EACH STATE.
    q_all = self.q_network(states)        # [B, n_actions]
    logsumexp_q = torch.logsumexp(q_all, dim=1)    # [B]
    q_data = q_all.gather(
        1, actions.unsqueeze(1)).squeeze(1)    # [B]
    # Push down the soft-max over ALL actions at s; hold up the
    # value of the action actually taken in the data at s.
    conservative = (logsumexp_q - q_data).mean()
    return td_loss + self.alpha * conservative
```

## 针对金融安全的宪法 AI

宪法 AI 使用一套书面原则来批评、修订和指导模型输出 [Bai et al., 2022]。在金融场景中，这些原则可以反映风险披露、隐私保护、适合性以及避免误导性或禁止性建议等要求。宪法本身并非法律控制：其原则必须进行版本管理，针对具体政策进行测试，并与确定性验证和人工升级相配合。

以下示例说明了模型响应如何生成、针对金融安全原则进行检查、修订，然后用作训练的一部分。

### 列表 9.13：金融 AI 响应的宪法训练循环

```python
class ConstitutionalFinancialAI:
    def __init__(self):
        self.principles = [
            "Never provide advice that could lead to illegal activities",
            "Always disclose risks associated with investments",
            "Do not make guarantees about future returns",
            "Respect client confidentiality and data privacy",
            "Provide advice suitable for client's risk tolerance"
        ]

    def constitutional_training(
        self, model, dataset, num_epochs=1, learning_rate=1e-6
    ):
        """
        Train on critique-and-revision targets.
        """
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate)
        total_loss = 0.0
        num_updates = 0

        for _ in range(num_epochs):
            for batch in dataset:
                # Generate an initial response without retaining a graph.
                with torch.no_grad():
                    response = model.generate(batch['prompt'])
                    critique = model.critique(
                        response, self.principles)
                    revised_response, critique = model.revise(
                        response, critique)

                # Train on the revised target.
                optimizer.zero_grad()
                loss = model.compute_loss(
                    batch['prompt'], revised_response)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_updates += 1

        return total_loss / max(num_updates, 1)
```

## 9.7 总结

本章介绍了强化学习（RL）作为一种框架，用于根据反馈、偏好和更长期的目标对齐 LLM 行为。它解释了 RL 如何通过奖励信号优化模型行为，而不仅仅是从固定的输入-输出示例中学习，从而与监督学习不同。

本章涵盖了语言模型的基本 RL 公式，包括状态、动作、策略、奖励和轨迹。然后讨论了奖励建模和人类反馈收集，展示了如何使用偏好数据来训练模型，以产生更有用、准确、合规和风险意识的输出。

我们还研究了 PPO、GRPO 和 DPO 等现代对齐方法。PPO 使用剪裁的策略更新来提高稳定性，GRPO 比较一组响应以估计相对优势，而 DPO 直接从偏好和拒绝的响应中进行优化，无需单独的奖励模型。

金融示例展示了 RL 概念如何应用于交易研究、风险管理、对冲和投资组合优化。这些示例应被理解为概念性决策支持架构，而非生产就绪系统。在实际金融环境中，此类系统需要大量验证、回测、压力测试、风险控制、合规审查和人工监督。

最后，本章讨论了高级技术和未来方向，包括层次化 RL、多智能体 RL、离线 RL 和宪法 AI。这些方法可能有助于使金融 LLM 系统更具适应性并得到更好的治理，但它们也引入了围绕奖励设计、分布漂移、非平稳性、可解释性、安全性和监管问责的挑战 [Daníelsson et al., 2022]。实际结论是，对齐目标和代码级保障必须共同评估：既复杂的损失函数，也流畅的模型，都无法替代独立验证、确定性控制和可问责的人工监督。

## 获取本书的 PDF 版本和更多内容

扫描二维码（或访问 packtpub.com/unlock）。按书名搜索此书，确认版本，然后按照页面上的步骤操作。

![](img/c31705b361fdf81dee8826a2c4ee167f_344_0.png)

立即解锁

注意：请妥善保管发票。直接从 Packt 购买无需发票。

## 10 基础设施与性能

在金融服务中部署大型语言模型（LLM）不仅仅是获取加速器容量。生产设计必须平衡延迟、吞吐量、模型质量、安全性、合规性、可审计性、可靠性、可恢复性和成本。这些目标经常相互冲突：更大的批量可以提高加速器利用率，但可能会恶化尾延迟；更强的持久性保证可能会增加处理延迟；而将工作负载靠近用户可以减少网络延迟，同时增加运营和治理复杂性。

金融工作负载也彼此存在实质性差异。交易支持分类器可能需要可预测的毫秒级响应，而监管助手可能容忍较慢的生成，但要求完整的血缘、可复现性和人工审查。面向客户的系统增加了隐私、可用性和行为风险约束。因此，不存在单一的“金融 LLM 栈”。基础设施应相反地从明确的服务级别规范中选择，该规范至少应涵盖中位数和尾延迟百分位数、持续和突发吞吐量、可用性、恢复点和恢复时间目标、任务质量以及总拥有成本。

因此，基础设施要求远不止计算。机构需要来自市场和企业源的安全数据摄入、确定性数据转换、受控检索、版本化提示和模型、可追溯输出，以及证据表明已部署配置继续满足其服务级别目标。从传统分析系统向 LLM 支持工作流的过渡，会一起改变数据、服务、监控和治理层，而不仅仅是替换一个模型端点。

本章探讨了生产金融 LLM 所需的主要基础设施层。它涵盖了硬件和部署选择、数据管道、用于检索增强生成（RAG）的向量数据库、模型优化、推理服务、编排、监控和性能调优。它以一个明确的示例性案例研究结束，针对延迟敏感的交易支持工作流；该场景是容量规划示例，而非经过审计的生产部署报告。

