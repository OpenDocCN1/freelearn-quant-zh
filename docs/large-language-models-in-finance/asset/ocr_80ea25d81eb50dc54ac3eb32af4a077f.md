```
```python
news_embedding = self.llm.encode_news(news_data)
sentiment_scores = self.llm.analyze_sentiment(news_data)

# Market microstructure
microstructure = self.analyze_microstructure(market_data)

# Combine into state vector
state = torch.cat([
    technical_features,
    news_embedding,
    sentiment_scores,
    microstructure
])

return state

def define_reward_function(self, action, market_response):
    """
    Multi-objective reward function for trading.
    """
    # Base PnL reward
    pnl = market_response['pnl']

    # Risk-adjusted returns (Sharpe ratio component)
    returns = market_response['returns']
    volatility = market_response['volatility']
    sharpe_component = returns / (volatility + 1e-8)

    # Positive penalty for drawdowns above 10%.
    drawdown = market_response['max_drawdown']
    drawdown_penalty = 0.5 * max(0.0, drawdown - 0.10)

    # Transaction cost penalty
    transaction_cost = market_response['transaction_cost']

    # Combine rewards; penalties enter with negative signs.
    reward = (
        0.4 * pnl +
        0.3 * sharpe_component -
        0.2 * drawdown_penalty -
        0.1 * transaction_cost
    )

    # Non-compliant actions receive a hard negative adjustment.
    if not self.check_regulatory_compliance(action):
        reward -= 1.0

    return reward
```
```