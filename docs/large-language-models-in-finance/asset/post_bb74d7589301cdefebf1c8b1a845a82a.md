
where $P_{comp}$ penalizes non-compliant or unsuitable responses, $P_{num}$ penalizes inconsistent calculations, and $P_{cal}$ penalizes unsupported certainty or poor probability calibration. These penalties should be computed by independently tested rules or evaluators, reported separately, and audited so that they do not silently override the preference objective.

### Listing 9.6: DPO training with financial preference weighting

```python
class FinancialDPO:
    def __init__(self, model, ref_model, config, beta=0.1):
        self.model = model
        self.ref_model = ref_model
        self.config = config
        self.beta = beta
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate)

    def compute_dpo_loss(self, batch):
        """
        Compute DPO loss for financial preference data.
        """
        prompts = batch['prompts']
        preferred = batch['preferred_responses']
        rejected = batch['rejected_responses']

        # Get log probabilities from both models
        pi_logprobs_preferred = self.model.get_log_probs(
            prompts, preferred)
        pi_logprobs_rejected = self.model.get_log_probs(
            prompts, rejected)

        ref_logprobs_preferred = self.ref_model.get_log_probs(
            prompts, preferred).detach()
        ref_logprobs_rejected = self.ref_model.get_log_probs(
            prompts, rejected).detach()

        # Compute log ratios
        pi_logratios = (
            pi_logprobs_preferred - pi_logprobs_rejected)
        ref_logratios = (
            ref_logprobs_preferred - ref_logprobs_rejected)

        # DPO loss
        losses = -F.logsigmoid(
            self.beta * (pi_logratios - ref_logratios))

        # Add preference strength weighting for financial domain
        if 'preference_strength' in batch:
            losses = losses * batch['preference_strength']

        return losses.mean()

    def train_on_financial_preferences(self, preference_dataset):
        """
        Train model using DPO on financial preference data.
        """
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0

            for batch in preference_dataset:
                # Augment with financial-specific preferences
                batch = self.augment_financial_preferences(batch)

                # Compute loss
                loss = self.compute_dpo_loss(batch)

                # Add regularization for financial constraints
                reg_loss = self.compute_financial_regularization(batch)
                total_loss = loss + self.config.reg_weight * reg_loss

                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0
                )
                self.optimizer.step()

            epoch_loss += total_loss.item()

            # Validation on financial benchmarks
            val_metrics = self.validate_on_financial_tasks()

            print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, "
                  f"Financial Accuracy = {val_metrics['accuracy']:.2%}")

    def augment_financial_preferences(self, batch):
        """
        Augment preference data with financial-specific signals.
        """
        augmented_batch = batch.copy()

        # Add preference strength based on financial criteria
        preference_strengths = []

        for preferred, rejected in zip(
            batch['preferred_responses'],
            batch['rejected_responses']
        ):
            strength = 1.0

            # Stronger preference if numerical accuracy differs
            if self.has_calculations(preferred) or self.has_calculations(rejected):
                acc_diff = abs(
                    self.get_numerical_accuracy(preferred) -
                    self.get_numerical_accuracy(rejected))
                strength *= (1 + acc_diff)

            # Stronger preference if compliance differs
            if self.has_compliance_content(preferred) or self.has_compliance_content(rejected):
                comp_diff = abs(
                    self.get_compliance_score(preferred) -
                    self.get_compliance_score(rejected))
                strength *= (1 + 0.5 * comp_diff)

            preference_strengths.append(strength)

        augmented_batch['preference_strength'] = torch.tensor(
            preference_strengths)

        return augmented_batch
```

![](img/c31705b361fdf81dee8826a2c4ee167f_327_0.png)

Figure 9.1: Three routes from preferences to an aligned policy.

They differ in what machinery they need — PPO wants a reward model and a value network, DPO discards both, GRPO keeps the RL loop but replaces the value network with a group-relative baseline — and they are usually compared on that axis alone. In finance the axis that matters is the one on the right: none of the three has any notion of a risk limit or a suitability rule, so the constraint set is not a property of the algorithm and must be imposed outside it.

## 9.4 Applications of RL-trained LLMs in finance

RL can support financial decision-making systems that adapt to feedback, market conditions, and multi-objective constraints. In practice, these systems must be treated as decision-support tools and require careful validation, risk controls, compliance review, and human oversight before any real-world use.

The following examples show how RL concepts can be mapped to common financial workflows. They are best read as conceptual architectures rather than production-ready systems.

### 9.4.1 Trading strategy learning

RL-trained models can be explored for trading-related research tasks, such as combining market data with textual signals from news, filings, or analyst commentary.

**Safeguard before market use.** An RL-based trading system must not move directly from a training notebook to live order routing. At minimum, it requires leakage-resistant out-of-sample and walk-forward backtesting; realistic commissions, bid-ask spreads, slippage, latency, borrow costs, and market-impact modeling; stress tests across regimes and liquidity shocks; hard position, leverage, concentration, loss, and turnover limits; a tested kill switch and fallback policy; independent model-risk and compliance review; paper trading; and explicit human authorization and supervision before capital is exposed. Reported performance should include uncertainty and sensitivity to execution assumptions, not only a headline Sharpe ratio.

### Multi-modal trading agent

The following illustrative example shows how a trading agent might combine technical indicators, news sentiment, market microstructure features, reward design, and risk checks. It is a conceptual example and does not represent a deployable trading system.

Listing 9.7: Conceptual RL trading agent with risk checks and human review

```python
class RLTradingAgent:
    def __init__(self, llm_model, market_env):
        self.llm = llm_model
        self.env = market_env
        self.action_space = ['BUY', 'SELL', 'HOLD']
        self.position_sizes = [0.25, 0.5, 0.75, 1.0] # Fraction of capital

    def create_state_representation(self, market_data, news_data):
        """
        Create comprehensive state representation for trading decisions.
        """
        # Technical indicators
        technical_features = self.compute_technical_indicators(market_data)

        # Sentiment from news
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
