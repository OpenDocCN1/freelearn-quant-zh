

```python
'preferred': responses[i],
'rejected': responses[j],
'confidence': self.compute_confidence(
    rankings[i], rankings[j])
})

# Quality control
filtered_data = self.quality_control.filter_preferences(
    preference_data)

return filtered_data

def compute_confidence(self, rank1, rank2):
"""
Compute confidence in preference based on rank difference.
"""
return 1.0 - np.exp(-abs(rank1 - rank2))
```

## 9.3 面向大语言模型的现代强化学习与偏好优化算法

策略梯度为基于强化学习的对齐奠定了基础，但实际的大语言模型训练需要使更新更稳定、更易于控制的方法。PPO、GRPO 和 DPO 以不同方式解决此问题：PPO 约束在线策略更新，GRPO 从一组生成结果中估计相对优势，DPO 直接从离线偏好对进行优化。表 9.1 汇总了主要区别。

| 方法 | 训练信号 | 其他学习组件 | 运营权衡 |
| :--- | :--- | :--- | :--- |
| PPO | 对新采样响应的标量奖励 | 通常包括奖励模型、参考策略和学习的值函数评估者 | 稳定的裁剪更新，但相对复杂且内存密集的在线训练循环 |
| GRPO | 针对同一提示的一组响应内部归一化的奖励 | 奖励源和参考策略，但没有单独的值函数评估者 | 与评估者相关的内存使用较低；性能取决于组采样以及奖励的质量和离散程度 |
| DPO | 偏好/拒绝响应对 | 固定参考策略；DPO 优化期间没有显式奖励模型或评估者 | 简单的离线目标，但学习受偏好对数据集的覆盖范围和质量限制 |

表 9.1: 大语言模型后训练中 PPO、GRPO 和 DPO 的比较

### 9.3.1 大语言模型的近端策略优化 (PPO)

近端策略优化 (PPO) 广泛用于大语言模型对齐，因为它在使用奖励反馈改进模型的同时，限制了更新后的策略与早期策略的偏离程度。这一约束有助于防止可能导致模型不稳定或降低先前学习行为的重大策略更新。

简单来说，PPO 比较旧策略和新策略下动作的概率。如果更新幅度过大，裁剪目标会降低模型在一个训练步骤中过度移动的动力。这种稳定性在金融应用中尤为重要，因为模型行为的微小变化可能会影响合规性、风险解释或决策支持输出 [Schulman et al., 2017, Ouyang et al., 2022]。

第二种控制是相对于固定参考策略的 Kullback–Leibler (KL) 散度。对于提示 _x_，它是

\( D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} [\log \pi_\theta(y | x) - \log \pi_{\text{ref}}(y | x)] . \qquad (9.10) \)

对采样响应的当前减去参考对数概率的标记级平均值是该量的蒙特卡洛估计。惩罚较大的 KL 散度会阻止对齐策略突然偏离基础模型，尽管系数必须调节，因为过度惩罚可能会抑制有用的学习。

#### 面向金融大语言模型的 PPO 实现

对于实际实验，通常应优先使用维护的后训练库，而非临时训练器。例如，Hugging Face TRL 提供配置好的 PPOTrainer 和 DPOTrainer 接口。下面的代码清单旨在暴露奖励、KL、裁剪和优化组件；它是一个概念性训练循环示例，而非推荐的生产环境重新实现。

下面的说明性示例展示了面向金融响应的 PPO 风格对齐循环的主要组件：奖励模型、用于 KL 惩罚的参考模型、领域特定的金融惩罚以及裁剪的策略更新。

代码清单 9.4: 面向金融响应的 PPO 风格对齐循环

```python
class FinancialPPO:
    def __init__(self, model, ref_model, reward_model, config):
        self.model = model
        self.ref_model = ref_model # Reference model for KL constraint
        self.reward_model = reward_model
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate)

    def compute_rewards(self, prompts, responses):
        """
        Compute rewards with KL penalty from reference model.
        """
        rewards = []

        for prompt, response in zip(prompts, responses):
            # Base reward from reward model
            base_reward = self.reward_model(prompt, response)

            # KL penalty to prevent model from deviating too far
            kl_penalty = self.compute_kl_penalty(
                prompt, response)

            # Additional financial-specific rewards/penalties
            financial_penalties = self.compute_financial_penalties(response)

            total_reward = (base_reward - self.config.kl_coef * kl_penalty -
                            financial_penalties)
            rewards.append(total_reward)

        return torch.stack(rewards)

    def compute_kl_penalty(self, prompt, response):
        """
        Compute KL divergence between current and reference model.
        """
        with torch.no_grad():
            ref_logprobs = self.ref_model.get_logprobs(
                prompt, response)
            current_logprobs = self.model.get_logprobs(
                prompt, response)

        # Sample-based estimate of KL(current || reference)
        kl = (current_logprobs - ref_logprobs).mean()
        return kl

    def compute_financial_penalties(self, response):
        """
        Apply domain-specific penalties for financial responses.
        """
        penalties = 0.0

        # Penalize overconfident predictions
        if self.detect_overconfidence(response):
            penalties += 0.5

        # Penalize lack of risk disclosure
        if not self.has_risk_disclosure(response):
            penalties += 0.3

        # Penalize non-compliant advice
        compliance_score = self.check_regulatory_compliance(
            response)
        if compliance_score < 0.8:
            penalties += 0.7 * (1 - compliance_score)

        return penalties

    def update_policy(self, trajectories):
        """
        PPO policy update with clipping.
        """
        states, actions, old_log_probs, rewards, advantages = self.process_trajectories(trajectories)

        for _ in range(self.config.ppo_epochs):
            # Compute current log probabilities
            current_log_probs = self.model.get_log_probs(
                states, actions)

            # Compute probability ratio
            ratio = torch.exp(current_log_probs - old_log_probs)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_param, 1 + self.config.clip_param) * advantages

            # Policy loss
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss (if using actor-critic)
            if self.config.use_value_function:
                value_loss = self.compute_value_loss(
                    states, rewards)

            else:
                value_loss = 0

            # Entropy bonus for exploration
            entropy = self.model.get_entropy(states)
            entropy_loss = -self.config.entropy_coef * entropy.mean()

            # Total loss
            total_loss = policy_loss + self.config.value_coef * value_loss + entropy_loss

            # Optimization step
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm)
            self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
            if self.config.use_value_function else 0.0,
            'entropy': entropy.mean().item()
        }
```

在 actor–critic PPO 中，值函数网络 \(V_\psi(s)\) 是一个学习的评估者，用于预测策略从状态 \(s\) 开始的预期未来回报。从观察回报中减去该预测可产生优势估计，但评估者会引入其自身的参数、优化器状态、前向激活、目标和验证要求。较新的方法试图在保留有用基线的同时去除该额外模型。

### 9.3.2 组相对策略优化 (GRPO)

组相对策略优化 (GRPO) 是 DeepSeekMath 论文 [Shao et al., 2024] 引入的面向大语言模型的一种强化学习方法。它用从一组采样响应中计算的每个提示内基线取代了单独的值函数评估者。这减少了与评估者相关的内存和实现复杂性，但并未消除对可靠奖励或充分响应多样性的需求。

#### GRPO 数学框架

GRPO 的关键创新是使用基于组的相对优势估计。对于每个提示 \(q\)，GRPO 从当前策略 \(\pi_{\theta_{old}}\) 采样一组 \(G\) 个输出 \(\{o_1, o_2, ..., o_G\}\)。输出 \(i\) 的组相对优势计算如下：

\(\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r}\) (9.11)

其中 \(\mu_r = \frac{1}{G} \sum_{j=1}^G r_j\) 是平均奖励，\(\sigma_r = \sqrt{\frac{1}{G} \sum_{j=1}^G (r_j - \mu_r)^2}\) 是标准差。

这意味着，响应不仅因为具有较高的绝对得分而获得奖励，还因为其得分优于为同一提示生成的其他响应。

#### 面向金融应用的 GRPO 实现

下面的说明性示例展示了 GRPO 如何为同一金融提示生成多个响应、对其评分、在组内归一化奖励，并使用得分较高的响应更新模型。数值检查器故意假设一个受限的算术求值器。此类求值器应将表达式解析为抽象语法树，仅允许数字字面量、括号和一小段算术运算符白名单，拒绝名称、属性访问、索引、导入和函数调用，并强制执行

