# 6.12 Summary

MCP is valuable because it standardizes how LLM applications connect to context and tools. It provides interoperable messages, lifecycle handling, capabilities, resources, prompts, and tools. Its value should not, however, be inflated into a claim that the protocol alone secures a financial institution.

This chapter showed that secure financial integration requires a separate control envelope around MCP-based tool use. That envelope provides identity, least privilege, business authorization, typed tool contracts, point-in-time evidence checks, deterministic numerical validation, approval binding, idempotent execution, and auditability. The model may propose, explain, summarize, or draft, but deterministic systems must decide what may execute.

The chapter also developed a finance-specific threat model, covering prompt injection, confused-deputy behavior, token passthrough, compromised servers, over-broad tools, stale data, duplicate execution, data exfiltration, and approved mismatch. These risks cannot be addressed by prompt wording alone. They require enforceable controls at the host, server, policy, validation, execution, and audit layers.

The central design boundary is therefore:

+   - MCP provides interoperable messages, lifecycle, capabilities, resources, prompts, and tools;
+   - the host provides consent, orchestration, and clear user interaction;
+   - the financial control envelope provides identity, least privilege, business authorization, temporal validity, numerical integrity, approval, idempotency, and audit;
+   - the model proposes and explains, while deterministic systems decide what may execute.

The practical goal is to turn tool use from an impressive demonstration into a governable financial workflow. When MCP is combined with typed tools, point-in-time evidence, deterministic checks, approval gates, audit records, validation tests, and operational monitoring, it can support financial systems that are not only more capable, but also more controlled and accountable.

The next chapter builds on this foundation by examining how LLM-based systems can be applied across financial workflows while preserving the same principles of validation, supervision, and governance.