```
if done:
    break

# Update both meta-controller and sub-policy
meta_reward = self.evaluate_goal_completion(
    goal, sub_trajectory)
self.meta_controller.update(state, goal, meta_reward)
sub_policy.update(sub_trajectory)

return sub_trajectory
```

Hierarchical RL focuses on decisions within a single agent or system. Multi-agent RL instead models settings where multiple agents interact, compete, or adapt to one another.

# 9.5.2 Multi-agent reinforcement learning

Financial markets involve many interacting participants, including investors, market makers, arbitrageurs, and liquidity providers. Multi-agent RL can be used in research simulations to study how different strategies interact under simplified market conditions.

The following illustrative example shows a simplified market simulation in which several agent types submit actions, the market clears orders, and each agent updates its policy based on the resulting trades.

# Listing 9.11: Multi-agent market simulation with order clearing and policy updates

```
class MultiAgentMarketSimulation:
    def __init__(self, num_agents=10):
        self.agents = []
        self.create_agent(agent_type=random.choice(
            ['momentum', 'value', 'arbitrage'])
        )
        for _ in range(num_agents)
    ]
    self.market = MarketEnvironment()

def simulate_trading_day(self):
    """
    Simulate interactions between multiple RL agents.
    """
    order_book = []

    for agent in self.agents:
        # Each agent observes market state
        market_state = self.market.get_state()

        # Agent decides on action
        action = agent.decide_action(
```