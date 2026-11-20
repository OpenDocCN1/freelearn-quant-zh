| MCP can standardize | The institution must still provide |
|---|---|
| Message formats and lifecycle | Identity assurance, entitlement mapping, and segregation of duties |
| Capability discovery Input and output schemas | Approved-tool inventory and server trust policy Business-rule validation and numerical reconciliation |
| Resources, prompts, and tools | Data classification, purpose limitation, and point-in-time admissibility |
| Transport-level authorization | Product, client, desk, jurisdiction, and action-level authorization |
| Structured errors and logs | Complete, immutable, privacy-aware audit evidence |
| Cancellation and progress messages | Idempotency, duplicate-execution prevention, recovery, and rollback |

Table 6.2: Protocol guarantees versus institutional obligations

# ## 6.2 The financial control envelope

A safe financial integration places every MCP exchange inside a control envelope that remains outside the language model. Figure 6.1 shows the reference pattern.

![](bbox=[0.15, 0.485, 0.85, 0.675])

Figure 6.1: A hardened MCP control envelope.

Blue denotes stochastic model behavior, amber denotes deterministic controls, teal denotes evidence or services, and gray denotes accountable human or audit functions. The envelope has nine separable functions:

+   1. authenticate the user, service, device, and workload;
+   2. bind the request to a declared purpose and decision time;
+   3. expose only approved servers and tools;
+   4. validate the proposed call against an exact schema;
+   5. evaluate authorization and business policy outside the model;