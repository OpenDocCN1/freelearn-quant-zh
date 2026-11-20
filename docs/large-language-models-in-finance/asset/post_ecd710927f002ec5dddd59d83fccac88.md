
grows, which is the right direction: searching over more actions is a form of selection, and selection must be paid for. The constant should be recalibrated if the deployment uses a different CVaR estimator, dependence model, or scenario-generation scheme.

The second theoretical layer is strategic. Once agents interact with clients, counterparties, exchanges, validators, and other agents, the deployment is a game rather than a stand-alone prediction problem. Let

\[
\Gamma = (N, \{\Theta_i\}_{i \in N}, p, \{\mathcal{A}_i\}_{i \in N}, \{\mathcal{I}_i\}_{i \in N}, \{u_i\}_{i \in N}, \mathcal{M}) \tag{14.11}
\]

be the deployment game, where \(\Theta_i\) is a type space, \(p\) is a type distribution, \(\mathcal{A}_i\) is an action space, \(\mathcal{I}_i\) is an information structure, \(u_i\) is a payoff, and \(\mathcal{M}\) is a communication or protocol mechanism. A policy profile \(\pi = (\pi_1, \dots, \pi_n)\) induces both outcomes and incentives. In this setting, prompts are only one force among many. Payoffs, information, repeated interaction, and communication protocols may dominate the model's stated persona.

Empirical work on algorithmic pricing shows why this matters. Reinforcement-learning agents can learn supracompetitive pricing and punishment strategies without direct communication, and recent language-model agent studies report collusive or market-division behavior under oligopoly, Cournot, and auction protocols [Calvano et al., 2020]. The precise magnitude and generality of these findings remain a matter for ongoing research, but the lesson for deployment is already clear: a financial institution cannot certify multi-agent safety by reading the prompt. It must ask which equilibria the mechanism permits.

A useful bounded-rational baseline is entropy-regularized or quantal response. In a finite game, player \(i\)'s regularized best response is

\[
\text{BR}_i^\tau(\pi_{-i}) = \arg \max_{q_i \in \Delta_i} \{ U_i(q_i, \pi_{-i}) + \tau H(q_i) \}, \quad H(q_i) = - \sum_a q_i(a) \log q_i(a), \tag{14.12}
\]

where \(\tau > 0\) is a temperature parameter. If strategic interaction is bounded, high enough entropy regularization makes the best-response map a contraction. More concretely, define the interaction constant

\[
\kappa_{ij} = \max_{a_i} \max_{a_{-i}, a'_{-i} \text{ agree off } j} |u_i(a_i, a_{-i}) - u_i(a_i, a'_{-i})|, \quad K = \max_j \sum_{i \neq j} \kappa_{ij}. \tag{14.13}
\]

Under the product \(\ell_1\) norm on mixed strategies, the regularized best-response map satisfies a sufficient bound of the form

\[
\| \text{BR}^\tau(q) - \text{BR}^\tau(q') \|_1 \le \frac{K}{2\tau} \| q - q' \|_1. \tag{14.14}
\]

Thus \(\tau > K/2\) is a sufficient contraction condition for uniqueness and convergence of the regularized equilibrium under the stated Lipschitz bound — and it is sufficient, not necessary, and conditional on a softmax Lipschitz constant that is norm-specific and must be established rather than assumed. It is not a theorem about LLM societies. It is a proof obligation: a paper claiming stable strategic convergence should identify the metric, the invariant set, and a contraction or monotonicity condition. In aggressive pricing or execution regimes \(K\) is large, the sufficient condition fails, and it fails exactly where strategic sensitivity is most dangerous.

The third theoretical layer is population composition. If $n$ agents share a foundation model, a vendor, a retrieval corpus, a prompt template, or a risk model, their actions may be highly correlated even when deployed in separate containers. Let \(\theta_i(t)\) be a state or sufficient statistic of agent \(i\) and let

\[
\mu_t^n = \frac{1}{n} \sum_{i=1}^n \delta_{\theta_i(t)} \tag{14.15}
\]

be the empirical population measure. A mean-field approximation may describe the limit as $n$ grows, for example through dynamics of the schematic form

\[
d\theta_i(t) = b(\theta_i(t), \mu_t^n)dt + \sigma(\theta_i(t), \mu_t^n)dW_i(t). \tag{14.16}
\]

Under Lipschitz conditions, propagation-of-chaos results can justify approximating finite populations by a McKean–Vlasov limit; however, the approximation is useful only if the chosen state has low effective dimension and captures the variables that close the dynamics. In financial terms, the relevant state is rarely an uncompressed prompt. It is more likely to be exposure buckets, inventory, leverage, liquidity demand, signal direction, model family, tool-use mode, or approval status.

Common priors make this layer more important. Multiple instances of the same model family are not independent economic agents in the classical sense. They may share default interpretations, failure modes, refusal behavior, and salient heuristics. From the perspective of market stability, this can act like a correlation device. Model diversity is therefore not a procurement detail; it is a systemic-risk control. Diversity should be measured behaviorally across actions, not merely declared because two applications run in different cloud accounts.

The fourth layer is geometry and invariance. Robustness claims are incomplete unless they specify the transformations under which behavior should be stable and the metric in which stability is measured. Let $G$ be a group or family of meaning-preserving transformations of a financial problem: sentence reordering, table layout changes, currency formatting, boilerplate insertion, document paraphrase, or coordinate changes that preserve economic content. For an observable loss or score $M$, define the orbit mean

\[
M_G(P) = \mathbb{E}_{g \sim \nu_G}[M(gP)] \tag{14.17}
\]

and the orbit dispersion

\[
V_G(P) = \mathbb{E}_{g \sim \nu_G}\left[\left(M(gP) - M_G(P)\right)^2\right]. \tag{14.18}
\]

A trade-ticket extractor should not change quantity because a ticket is converted into a table. A credit assistant should not change approval probability because irrelevant sections are reordered. A counterparty agent should not change contractual economics because boilerplate is paraphrased. Nonzero orbit dispersion is therefore a deployment risk, not merely a benchmark inconvenience. The same principle connects to adversarial robustness and distribution shift: the perturbations of interest in finance are often action-preserving reformulations of contracts, tickets, reports, and mandates rather than generic token noise.

Geometry also determines what it means for two policies to be close. KL divergence is natural for likelihood ratios and policy updates; total variation controls worst-case event probability; Fisher-Rao geometry describes local statistical curvature; Wasserstein distance is useful when outcome space has an economically meaningful cost metric; and Bregman divergences appear in mirror descent and entropy-regularized games. A compliance policy may be close in KL while still far in total variation on rare prohibited events. An execution policy may be close in parameter space while far in Wasserstein distance over fills. The metric must be chosen for the financial consequence under review.

Finally, an acting system requires causal rather than merely observational adequacy. A model that predicts $P(Y \mid X)$ need not correctly predict $P(Y \mid \text{do}(X))$. A credit recommendation, hedge adjustment, order routing decision, or liquidity intervention changes the environment it seeks to evaluate. Testing therefore has to include intervention, transport, and timing. Evidence from one desk, client segment, market regime, or document corpus does not automatically transport to another. Selection diagrams and transportability theory formalize when source-domain experiments can be combined with target-domain covariates, but absent those assumptions, deployment evidence remains local [Pearl, 2009].

These theoretical layers yield a compact rule for the rest of the chapter: language can enter at the semantic edge, but authority should pass only through contracts. The model may search, summarize, propose, and explain. Executable commitments should be typed, shielded, permissioned, monitored, and replayable. Model cards, datasheets, internal audits, and AI risk-management frameworks are therefore necessary but not sufficient; in financial agents they must be bound to action classes, shield predicates, audit packets, fallbacks, and named owners. This is the bridge from an impressive demo to an institutionally accountable financial agent.

## 14.2.3 The cognitive leap: From instruction-following to goal-seeking

A predictive LLM is trained to minimize a loss that measures disagreement with a target:

\[
\mathcal{L}_{\text{predictive}} = \mathbb{E}_{(x,y)\sim\mathcal{D}} [-\log p_\theta(y \mid x)]. \tag{14.19}
\]

A sequential decision system instead evaluates a policy over trajectories:

\[
J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \gamma^t r(s_t, a_t) \right], \quad \mathcal{L}_{\text{agent}} = -J(\pi_\theta), \tag{14.20}
\]

where \(\tau\) is a trajectory, \(\pi_\theta\) is the policy, \(\gamma\) is the discount factor, and \(r(s_t, a_t)\) is the reward.

Remark 14.2.1 (What PPO Does—and Does Not—Guarantee). Proximal policy optimization (PPO) stabilizes empirical policy updates by optimizing a clipped surrogate objective [Schulman et al., 2017]:

\[
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left\{ \rho_t(\theta) \widehat{A}_t, \text{clip}(\rho_t(\theta), 1-\varepsilon, 1+\varepsilon) \widehat{A}_t \right\} \right]. \tag{14.21}
