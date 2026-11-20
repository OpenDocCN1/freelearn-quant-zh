# 6.11 Implementation-quality checklist

Before a financial workflow is allowed to use MCP in production, the institution should be able to answer “yes” to the following checks. These checks do not replace formal model-risk, security, legal, or compliance review, but they define the minimum implementation evidence expected for a material MCP-assisted workflow.

+   1. **Protocol and capability pinning.** The exact MCP revision, transport, declared capabilities, approved servers, and schema hashes are pinned, versioned, and logged.
+   2. **Server ownership and trust classification.** Every server has an accountable owner, trust classification, data-flow review, egress policy, and approved operational purpose.
+   3. **Secret isolation.** The model-visible context contains no access tokens, private keys, passwords, or unnecessary sensitive data.
+   4. **Narrow tool design.** Tools are narrow, typed, allow-listed, and separated by business effect: read, calculate, draft, approve, and execute.
+   5. **Deterministic authorization.** Business authorization is enforced outside the model and does not depend on prompt compliance or model self-assessment.
+   6. **Evidence validity.** All evidence has a recorded source, version, hash, permission status, and availability time.
+   7. **Numerical reconciliation.** Material numbers are recomputed, reconciled, and validated outside the LLM using deterministic code or approved calculation engines.
+   8. **Approval binding.** Approvals bind to the exact canonical request and expire after any material change to parameters, evidence, policy version, destination, or output.
+   9. **Idempotent execution.** Write actions use idempotency keys, reconciliation, execution ledgers, and recoverable state transitions.
+   10. **Adversarial testing.** Prompt injection, malicious tool metadata, confused-deputy behavior, over-broad tool use, and data exfiltration are tested.
+   11. **Operational monitoring.** Latency, error, timeout, retry, queue-depth, and cancellation distributions are monitored against an explicit service-level objective (SLO).
+   12. **Audit reconstruction.** Audit records support reconstruction of the material decision without storing secrets or relying on hidden chain-of-thought.
+   13. **Change-triggered revalidation.** Model, prompt, policy, schema, server, dependency, and infrastructure changes trigger risk-based revalidation before promotion.

If any item fails, the workflow should remain in pilot, read-only, shadow, or human-reviewed mode until the gap is remediated and retested.