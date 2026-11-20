and the canonical request hash. Approval is bound to the exact artifact that will be published and must expire automatically if any material input changes.

+   6. Publish and reconcile. A separate publication tool distributes the approved report to the authorized recipient list. It records the artifact hash, delivery status, and execution identifier. A reconciliation process then confirms that exactly one approved report was published for the relevant portfolio and date, preventing duplication or drift between the approved and distributed versions.

Figure 6.2 summarizes the same flow at a system level.

![](bbox=[0.13, 0.27, 0.87, 0.49])

# Figure 6.2: Governed MCP-assisted risk reporting.

The model drafts the narrative, while deterministic systems remain responsible for data validity, calculations, policy checks, approval, publication, and audit. This example illustrates the broader design principle of the chapter: the language model contributes explanation and synthesis, but the financial control envelope owns admissibility, numerical correctness, authorization, execution, and traceability.

# 6.8 Validation and adversarial testing

A deployment should not be approved merely because a demonstration succeeds. A material MCP-assisted workflow should pass a versioned test suite that covers protocol behavior, financial correctness, security, policy enforcement, and recovery. The goal is to test not only whether the workflow works under normal conditions, but also whether it fails safely under malformed inputs, stale data, adversarial content, unavailable services, and partial execution.

The test suite should include the following categories:

+   • Protocol conformance tests: Test lifecycle negotiation, capability changes, malformed JSON-RPC, unsupported methods, pagination, cancellation, timeouts, session expiry, transport reconnects, and version mismatch. For Streamable HTTP, also test origin rejection, authentication failures, secure session handling, and replay behavior.