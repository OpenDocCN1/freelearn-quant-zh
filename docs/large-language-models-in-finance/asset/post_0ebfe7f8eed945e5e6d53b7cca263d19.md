

limits on expression depth and numeric magnitude. LLM-generated text must never be passed directly to Python's eval.

## Listing 9.5: GRPO-style alignment loop with restricted arithmetic evaluation

```python
class FinancialGRPO:
    def __init__(self, model, reward_model, config):
        self.model = model
        self.reward_model = reward_model
        self.config = config
        self.ref_model = copy.deepcopy(model)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate)
        self.math_evaluator = RestrictedArithmeticEvaluator()

    def generate_group_samples(self, prompt, group_size=8):
        """
        Generate multiple responses for group-based advantage estimation.
        """
        samples = []
        with torch.no_grad():
            for _ in range(group_size):
                response = self.model.generate(
                    prompt,
                    max_length=self.config.max_response_length,
                    temperature=self.config.sampling_temperature,
                    do_sample=True
                )
                samples.append(response)
        return samples

    def compute_group_advantages(self, prompt, responses):
        """
        Compute relative advantages within a group of responses.
        """
        # Compute rewards for all responses
        rewards = []
        for response in responses:
            reward = self.reward_model(prompt, response)

            # Add financial-specific reward components
            if self.config.use_accuracy_reward:
                accuracy_reward = self.compute_numerical_accuracy(response)
                reward += self.config.accuracy_weight * accuracy_reward

            if self.config.use_compliance_reward:
                compliance_reward = self.compute_compliance_score(response)
                reward += self.config.compliance_weight * compliance_reward

            rewards.append(reward)

        # Build on the model's device with an explicit dtype. torch.tensor()
        # on a list of tensors copies AND detaches, and silently lands on CPU.
        if torch.is_tensor(rewards[0]):
            rewards = torch.stack(rewards).to(self.device).float()
        else:
            rewards = torch.tensor(
                rewards, dtype=torch.float32, device=self.device)

        # A group of one has no dispersion to normalize against, and the
        # UNBIASED std of a single sample is NaN -- which then propagates
        # silently through the whole update.
        G = rewards.numel()
        if G < 2:
            raise ValueError(
                f"GRPO needs group_size >= 2 to form a relative baseline; got {G}")
        mean_reward = rewards.mean()
        std_reward = rewards.std(unbiased=False)

        if std_reward < 1e-6:
            # Every completion scored the same. There is no relative signal.
            # Returning zeros is honest; dividing by 1e-8 manufactures one.
            advantages = torch.zeros_like(rewards)
        else:
            advantages = (rewards - mean_reward) / (std_reward + 1e-8)

        return advantages, rewards
```

Three things here are easy to get wrong, and all three change the algorithm rather than merely its efficiency.

- (1) The importance ratio must be taken against the BEHAVIOUR policy that actually generated the samples -- not against the reference model. The reference model exists to anchor the KL term. If the ratio and the KL share a denominator, the objective is no longer the one being described.

- (2) The behaviour log-probs must be snapshotted BEFORE any update. Stepping the optimizer inside the group loop means later members of the group are scored against a policy that has already moved, so the "fixed batch" is not fixed.

- (3) Filtering to positive advantages biases the estimator: it discards exactly the half of the group that carries the downward gradient.

```python
def update_policy(self, training_prompts):
    """
    GRPO policy update for financial reasoning.
    It is a heuristic, and it is off by default.
    """
    # ---- PHASE 1: freeze the behaviour policy for the WHOLE batch ----
    # A snapshot taken inside the prompt loop is a snapshot per GROUP, not
    # per batch: once the optimizer steps for prompt 1, prompt 2's samples
    # come from a policy that no longer equals the one whose log-probs sit
    # in the denominator of the ratio. The importance weight is then against
    # a policy that never generated anything, and the clip range no longer
    # means what it says. Snapshot once, generate everything, THEN update.
    pi_old = copy.deepcopy(self.model).eval()
    for p in pi_old.parameters():
        p.requires_grad_(False)

    batch = []
    with torch.no_grad():
        for prompt in training_prompts:
            responses = self.generate_group_samples(
                prompt, self.config.group_size, policy=pi_old)
            advantages, _ = self.compute_group_advantages(prompt, responses)
            batch.append({
                "prompt": prompt,
                "responses": responses,
                "advantages": advantages,
                "old_lp": [pi_old.get_log_probs(prompt, r) for r in responses],
                "ref_lp": [self.ref_model.get_log_probs(prompt, r)
                           for r in responses],
            })

    # ---- PHASE 2: optimize the current policy on that frozen batch ----
    total_loss, num_updates = 0.0, 0
    for _ in range(self.config.ppo_epochs):
        self.optimizer.zero_grad()
        batch_loss = 0.0
        n_terms = 0

        for g in batch:
            idxs = range(len(g["responses"]))
            if self.config.use_advantage_filtering:
                # Off by default: discarding negative advantages throws away
                # exactly the half of the group carrying the down-gradient.
                idxs = [i for i in idxs if g["advantages"][i] > 0]

            for i in idxs:
                lp = self.model.get_log_probs(g["prompt"], g["responses"][i])
                adv = g["advantages"][i]

                # Ratio against the FROZEN behaviour policy.
                ratio = torch.exp(lp - g["old_lp"][i])
                surr1 = ratio * adv
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon,
                ) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # KL to the REFERENCE, k3 estimator: unbiased AND
                # non-negative. A plain mean of (log pi - log ref) is a
                # log-ratio, not a KL: it can go negative, at which point
                # the "penalty" pays the model to drift.
                # log_r = g["ref_lp"][i] - lp
                kl = (torch.exp(log_r) - log_r - 1.0).mean()

                batch_loss = batch_loss + policy_loss + self.config.kl_coef * kl
                n_terms += 1

            if n_terms == 0:
                continue
            batch_loss = batch_loss / n_terms
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            total_loss += float(batch_loss.item())
            num_updates += 1

    # The reference policy is the FIXED SFT anchor. It is never refreshed:
    # a moving anchor is not a KL constraint, it is a slow-motion reward
    # for whatever the model has already become.

    return {
        'average_loss':
        (total_loss / num_updates
        if num_updates > 0 else 0),
        'num_updates': num_updates
    }

    def compute_numerical_accuracy(self, response):
        """
        Evaluate numerical accuracy in financial calculations.
        """
        extracted_calculations = self.extract_calculations(
            response)
        accuracy_score = 0.0

        for calc in extracted_calculations:
            try:
                # Parse and evaluate only whitelisted arithmetic.
                result = self.math_evaluator.evaluate(
                    calc['expression'])
                if abs(result - calc['stated_result']) < 0.01:
                    accuracy_score += 1.0
                else:
                    accuracy_score -= 0.5
            except (ValueError, TypeError, ArithmeticError):
                # Reject malformed or non-whitelisted expressions.
                accuracy_score -= 1.0

        return accuracy_score / max(
            len(extracted_calculations), 1)
```

PPO and GRPO still rely on reward-style optimization during training. DPO takes a different route by using preference pairs directly, making the alignment process simpler in settings where preferred and rejected responses are already available.

## 9.3.3 Direct preference optimization (DPO)

Direct preference optimization (DPO) is a preference-optimization method that avoids training a separate reward model. Instead, it directly updates the language model so that preferred responses become more likely than rejected responses [Rafailov et al., 2023].

### DPO mathematical formulation

DPO derives a closed-form solution for the optimal policy given a reward model, reformulating the RL problem as a simple binary classification loss:

$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right] \quad (9.12)$$

where \(y_w\) is the preferred response, \(y_l\) is the rejected response, and \(\beta\) is a temperature parameter.

In practical terms, the loss encourages the model to assign higher probability to the preferred response while keeping the update anchored to a reference model.

### DPO implementation for financial applications

The following illustrative example shows how DPO can be applied to financial preference data by comparing preferred and rejected responses, then optionally weighting the preference signal using financial criteria such as numerical accuracy and compliance. Here, “financial regularization” means an explicitly documented auxiliary penalty, for example

$$\mathcal{R}_{\text{fin}} = \lambda_{ \text{comp}} P_{ \text{comp}} + \lambda_{ \text{num}} P_{ \text{num}} + \lambda_{ \text{cal}} P_{ \text{cal}}, \quad (9.13)$$
