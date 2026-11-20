```
```python
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
```

```
```python
def update_policy(self, training_prompts):
    """
GRPO policy update for financial reasoning.
```
```

Three things here are easy to get wrong, and all three change
the algorithm rather than merely its efficiency.

+   (1) The importance ratio must be taken against the BEHAVIOUR policy
+   that actually generated the samples -- not against the reference
+   model. The reference model exists to anchor the KL term. If the
+   ratio and the KL share a denominator, the objective is no longer
+   the one being described.

+   (2) The behaviour log-probs must be snapshotted BEFORE any update.
+   Stepping the optimizer inside the group loop means later members
+   of the group are scored against a policy that has already moved,
+   so the "fixed batch" is not fixed.

+   (3) Filtering to positive advantages biases the estimator: it discards
+   exactly the half of the group that carries the downward gradient.