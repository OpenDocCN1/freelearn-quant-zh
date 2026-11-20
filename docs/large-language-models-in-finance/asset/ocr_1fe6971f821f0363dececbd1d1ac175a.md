```
self.config.max_grad_norm)
self.optimizer.step()

return {
    'policy_loss': policy_loss.item(),
    'value_loss': value_loss.item()
    if self.config.use_value_function else 0.0,
    'entropy': entropy.mean().item()
}
```

In actor–critic PPO, the value-function network \(V_\psi(s)\) is a learned critic that predicts the policy's expected future return from state \(s\). Subtracting this prediction from the observed return produces an advantage estimate, but the critic introduces its own parameters, optimizer state, forward activations, targets, and validation requirements. Newer methods try to remove that additional model while retaining a useful baseline.

# 9.3.2 Group relative policy optimization (GRPO)

Group Relative Policy Optimization (GRPO) is an RL method for LLMs introduced in the DeepSeekMath paper [Shao et al., 2024]. It replaces the separate value-function critic with a within-prompt baseline computed from a group of sampled responses. This reduces critic-related memory and implementation complexity, although it does not eliminate the need for reliable rewards or adequate response diversity.

# GRPO mathematical framework

The key innovation of GRPO is using group-based relative advantage estimation. For each prompt \(q\), GRPO samples a group of \(G\) outputs \(\{o_1, o_2, ..., o_G\}\) from the current policy \(\pi_{\theta_{old}}\). The group-relative advantage for output \(i\) is computed as:

\(\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r}\) (9.11)

where \(\mu_r = \frac{1}{G} \sum_{j=1}^G r_j\) is the mean reward and \(\sigma_r = \sqrt{\frac{1}{G} \sum_{j=1}^G (r_j - \mu_r)^2}\) is the standard deviation.

This means that a response is rewarded not only for having a high absolute score, but for scoring better than other responses generated for the same prompt.

# GRPO implementation for financial applications

The following illustrative example shows how GRPO can generate several responses to the same financial prompt, score them, normalize rewards within the group, and update the model using the higher-scoring responses. The numerical checker deliberately assumes a restricted arithmetic evaluator. Such an evaluator should parse an expression into an abstract syntax tree, admit only numeric literals, parentheses, and a small whitelist of arithmetic operators, reject names, attribute access, indexing, imports, and function calls, and enforce