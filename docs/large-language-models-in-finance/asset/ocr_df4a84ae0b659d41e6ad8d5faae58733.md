```
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
```