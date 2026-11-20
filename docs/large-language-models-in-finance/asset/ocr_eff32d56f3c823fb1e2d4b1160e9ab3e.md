```
```python
loss = 0.0
for episode_data, adv_seq in zip(trajectories, advantages):
    for (_, memory, action, log_prob, _), A_t in zip(episode_data, adv_seq):
        loss -= log_prob * A_t # gradient ascent on J
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_llm.parameters(), max_grad_norm)
    optimizer.step()
```
```

This sketch is illustrative. In practice, memory updates and prompts would be carefully designed to limit context length, invalidate stale information, separate durable facts from provisional workflow artifacts, and ensure stable behavior under changing market regimes.

# ## Reported empirical behavior

In backtests on a small set of U.S. equities over a limited historical period, Yu et al. [2023] reported that FinMem:

+   - Outperforms deep RL baselines and buy-and-hold on several assets, particularly those with strong trends and rich news flow;
+   - Achieves these results with shorter training windows than DRL baselines; and
+   - Produces behavior that qualitatively resembles regime awareness and risk-profile conditioning.

While the performance is dependent on asset and period, the experiments are limited in scope. Nonetheless, they provide evidence that explicit memory structures can stabilize LLM-based trading agents.

# ## 12.2.2 TradingAgents: Debate-based multi-agent trading

TradingAgents [Xiao et al., 2025] introduce a debate-based framework in which several LLM agents examine the same market situation from different professional perspectives. The design mirrors a human investment committee: one participant gathers evidence, another proposes a trade, and another challenges the proposal from a risk-control perspective. This role separation is useful because it makes the reasoning process more auditable than a single monolithic prompt.

The framework uses distinct LLM agents playing roles such as:

+   - Analysts, who interpret news and fundamentals;
+   - Traders, who propose position changes; and
+   - Risk managers, who critique and constrain proposals.

# ## Markov game formulation

A Markov game generalizes an MDP from one decision maker to several interacting decision makers. At each time step, the environment has a state, each agent receives an observation, each agent proposes an action or recommendation, and the system combines these proposals