# 11.4.2 Layer 2: Behavioral monitoring and control

The second layer observes how the deployed system actually behaves. Monitoring should compare production behavior with validated ranges and should be designed around decisions that operators can take, not around dashboards alone.

The following monitoring and control practices help institutions detect behavioral drift, coordination risk, and unsafe tool use after deployment:

+   - **Decision and interaction analytics:** Monitoring should cover task success, error severity, calibration, abstention, prompt and context sensitivity, tool-use patterns, policy violations, and distribution drift. For agents, the event log should reconstruct the sequence from observation through reasoning, tool selection, action, external response, and subsequent adaptation. Where privacy or confidentiality limits logging, the institution should document what evidence is unavailable and how that residual risk is mitigated.

+   - **Cross-agent and common-dependency monitoring:** Institutions should assess correlation in recommendations, orders, risk-taking, and liquidity withdrawal across internal agents and, where observable, against broader market behavior. Dependency maps should identify shared model providers, cloud services, datasets, retrieval feeds, and optimization objectives. A correlation alert should trigger investigation and proportionate action; it should not be treated as proof of collusion or intent.

+   - **Semantic robustness and tool-use validation:** Regular challenge tests should include paraphrases, ambiguity, contradictory sources, stale information, prompt injection, role confusion, and instructions embedded in retrieved documents. Inputs from untrusted sources should be isolated and authenticated, and tool calls should be checked against an explicit policy engine. High-impact actions should require structured arguments and deterministic validation rather than free-form model text alone [National Institute of Standards and Technology, 2024, Vassilev et al., 2025].

# 11.4.3 Layer 3: Risk management integration

This layer translates AI-specific evidence into familiar financial-risk processes. It ensures that model behavior affects limits, stress tests, capital and liquidity planning, vendor management, incident response, and business decisions.

The following practices connect AI-specific risks to established financial-risk management processes:

+   - **Unified risk taxonomy:** Traditional market, credit, liquidity, operational, conduct, compliance, cyber, and reputational risks remain relevant. AI-specific mechanisms should be mapped into those consequences rather than managed in a separate silo. For example, unsupported generation may create conduct, legal, or market risk; prompt injection may create cyber and operational risk; common-mode behavior may create market and liquidity risk; and provider concentration may create third-party and systemic risk [Financial Stability Board, 2024, 2026].