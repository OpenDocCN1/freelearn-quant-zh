

```python
policy_loss_total = policy_loss_total + (
    -(log_probs * advantages.detach()).mean())
# This is what actually trains the baseline.
value_loss_total = value_loss_total + F.mse_loss(values, returns)

n = len(trajectories)
loss = (policy_loss_total + self.value_coef * value_loss_total) / n
if not torch.isfinite(loss):
    raise FloatingPointError("non-finite loss; check advantages")
loss.backward()
torch.nn.utils.clip_grad_norm_(
    list(self.model.parameters())
    + list(self.baseline_network.parameters())),
    self.max_grad_norm,
)
self.optimizer.step()
return {"policy_loss": float(policy_loss_total) / n,
        "value_loss": float(value_loss_total) / n}
```

策略优化说明了模型如何更新，但它仍然依赖于有用的奖励信号。因此，下一步是理解奖励来自哪里。

## 9.2.3 奖励建模方法

奖励模型 \(R_\phi(x, y)\) 通常是另一个神经网络，经过训练用于为给定提示 \(x\) 的模型响应 \(y\) 打分。在金融应用中，此分数可能需要同时反映多个标准，包括事实准确性、数值正确性、监管合规性、清晰度和风险意识。

在成对偏好数据中，审核人员比较同一提示的两个响应，并确定哪个更好地满足评估标准。然后，奖励模型被训练为首选响应分配高于被拒绝响应的分数。在 Bradley–Terry 模型下，首选响应 \(y_w\) 被选中而非被拒绝响应 \(y_l\) 的概率为

\(\text{Pr}_\phi(y_w \succ y_l \mid x) = \sigma(R_\phi(x, y_w) - R_\phi(x, y_l)),\)

其中 \(\sigma(z) = (1 + e^{-z})^{-1}\)。最大化此概率等同于最小化成对逻辑损失

\(\mathcal{L}_{\text{BT}}(\phi) = -\mathbb{E}[\log \sigma(R_\phi(x, y_w) - R_\phi(x, y_l))].\)

因此，损失取决于两个分数之间的差异，而非其绝对水平；需要正则化或校准以防止奖励尺度漂移 [Bradley and Terry, 1952, Christiano et al., 2017, Ouyang et al., 2022]。

以下说明性示例展示了一种可能的奖励模型结构。它将提示-响应对的学习分数与针对数值准确性、合规性和风险的金融特定检查相结合，但权重和辅助组件应被视为概念性而非生产就绪的。

## 清单 9.2：说明性金融奖励模型结构

```python
class FinancialRewardModel(nn.Module):
    """Three things the previous version of this listing got wrong, all of
    which stop it from running at all:

    1. base_model_name="finbert-base" is not a Hub identifier.
    2. nn.Linear(768, ...) hard-codes the encoder width. Change the base
       model and the head silently mismatches -- or does not, which is worse,
       because the bug then depends on which model you happened to pick.
    3. self.encoder(prompt, response) passes RAW STRINGS to a transformer.
       A transformer consumes input_ids and an attention_mask. There was no
       tokenizer anywhere in the class.

    Below: a real tokenizer, real attention masks, the width read from the
    encoder's own config, and masked mean pooling (pooler_output does not
    exist on every architecture, and where it does it is trained for NSP,
    not for scoring).
    """
    def __init__(self, base_model_name="ProsusAI/finbert", max_length=512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.max_length = max_length

        h = self.encoder.config.hidden_size    # NOT a literal 768
        self.reward_head = nn.Sequential(
            nn.Linear(h, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    # Deterministic, non-learned components. These are checks, not scores:
    # they belong outside the network so they cannot be optimized away.
    self.accuracy_scorer = NumericalAccuracyScorer()
    self.compliance_checker = ComplianceChecker()
    self.risk_assessor = RiskAssessor()

    def _encode(self, prompts, responses):
        """Tokenize as a PAIR, on the model's own device."""
        device = next(self.encoder.parameters()).device
        batch = self.tokenizer(
            prompts, responses,
            padding=True, truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)
        out = self.encoder(**batch).last_hidden_state  # [B, T, H]
        mask = batch["attention_mask"].unsqueeze(-1).float()
        # Masked mean pooling: padding must not dilute the representation.
        return (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, prompts, responses):
        """prompts/responses are LISTS of strings. Returns [B] rewards."""
        pooled = self._encode(prompts, responses)
        base = self.reward_head(pooled).squeeze(-1)  # [B]

        acc = self.accuracy_scorer(responses)  # each [B]
        comp = self.compliance_checker(responses)
        risk = self.risk_assessor(responses)

        return (0.4 * base + 0.3 * acc + 0.2 * comp - 0.1 * risk)

    def train_on_preferences(self, preference_data):
        """Train reward model on pairwise preference data.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-5)

        for batch in preference_data:
            preferred_responses = batch['preferred']
            rejected_responses = batch['rejected']
            prompts = batch['prompts']

            # Compute rewards
            preferred_rewards = self(
                prompts, preferred_responses)
            rejected_rewards = self(prompts, rejected_responses)

            # Bradley-Terry model loss
            loss = -torch.log(
                torch.sigmoid(
                    preferred_rewards - rejected_rewards)
            ).mean()

            # Add regularization to prevent reward hacking
            regularization = 0.001 * (preferred_rewards.pow(2).mean() +
                                      rejected_rewards.pow(2).mean())

            total_loss = loss + regularization

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

因为奖励模型依赖于首选和被拒绝响应的示例，人类反馈数据的质量直接影响对齐模型的质量。

## 人类反馈收集策略

对于金融应用，收集高质量的人类反馈需要领域专业知识。以下说明性示例展示了如何将专家排名转换为奖励模型训练的成对偏好数据。

## 清单 9.3：来自专家排名的偏好数据收集

```python
class FinancialPreferenceCollector:
    def __init__(self):
        self.expert_pool = ExpertPool()  # Pool of financial experts
        self.quality_control = QualityControl()

    def collect_preferences(self, prompts, model_responses):
        """
        Collect preference data with quality control.
        """
        preference_data = []

        for prompt in prompts:
            # Generate multiple responses
            responses = [model_responses[prompt][i] for i in range(4)]

            # Get expert rankings
            rankings = self.expert_pool.rank_responses(
                prompt,
                responses,
                criteria={
                    'accuracy': 0.3,
                    'completeness': 0.2,
                    'clarity': 0.2,
                    'compliance': 0.2,
                    'practical_value': 0.1
                }
            )

            # Create pairwise preferences from rankings
            for i in range(len(rankings)):
                for j in range(i + 1, len(rankings)):
                    if rankings[i] < rankings[j]: # i is preferred over j
                        preference_data.append({
                            'prompt': prompt,
```
