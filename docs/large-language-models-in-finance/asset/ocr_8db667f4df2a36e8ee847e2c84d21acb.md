# 6.5 Execution semantics: idempotency, retries, and audit

Distributed systems routinely retry failed or timed-out requests. In financial systems, however, a retry must not duplicate the business effect. A repeated network message should not create two trades, two payments, two reports, or two review cases. The execution layer must therefore separate transport-level retry behavior from business-level execution.

For this reason, the execution layer should distinguish:

+   * a transport request identifier, which may change across network attempts;
+   * a stable business idempotency key, which identifies the intended business action;
+   * an execution identifier, which is created only when the underlying system accepts the action.

Before execution, the adapter checks whether the idempotency key has already been completed, rejected, cancelled, or left in an uncertain state. Completed and rejected actions should not be executed again. Uncertain outcomes require reconciliation against the execution ledger or target system, not blind retry.

# 6.5.1 Canonical request hashing

Approvals should attach to the exact request that will execute, not to a loose description of the request. Let canon(r) serialize the normalized request with stable field ordering, units, decimal precision, evidence identifiers, and version identifiers. The approval hash is:

h_r = H(canon(r)), (6.9)

where H is the approved hash function. The execution adapter verifies the same hash immediately before action. Any change to parameters, evidence, policy version, destination, amount, or execution target invalidates the approval and sends the request back for review.

# 6.5.2 The audit record

A material MCP-assisted workflow should produce an audit record that allows the institution to reconstruct what happened, which evidence was used, which controls were applied, who approved the action, and what was ultimately executed. A minimum audit record contains: