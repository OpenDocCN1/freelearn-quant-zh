```
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
        surr2 = torch.clamp(ratio, 1 - self.config.clip_param, 1 + self.config.clip_param)
    clip_param) * advantages

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
        total_loss = policy_loss + self.config.value_coef * value_loss +
entropy_loss

        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
```