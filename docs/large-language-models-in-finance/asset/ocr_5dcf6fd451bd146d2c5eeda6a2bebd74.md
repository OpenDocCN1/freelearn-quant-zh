modes. These go beyond issues encountered in single-shot LLM applications, because agentic systems involve multi-step reasoning, tool use, and interaction with external systems.

In this section, we analyze the main categories of challenges and failure modes, focusing on practical implications for financial institutions. The controls described below should be read as layered risk reductions rather than guarantees. They can lower the probability and severity of failures, but they do not eliminate residual model risk, data risk, operational risk, or the risk of inattentive human review.

# 5.2.1 Prompt injection and tool abuse

Prompt injection refers to adversarial or malicious inputs that cause the model to behave contrary to the designer’s intent. In agentic systems, prompt injection can be particularly dangerous when agents are allowed to call tools that affect external systems (e.g., trading systems, databases) or that access sensitive data. In the research workstation, the most realistic injection may not come from the portfolio manager directly; it may be hidden inside a retrieved document, an e-mail attachment, a scraped webpage, or an analyst note that is later passed to the Research Analyst agent as context.

This is the *indirect* case, and it is the one that matters here. Greshake et al. [2023] showed that once an application retrieves untrusted content into its context, an adversary who can influence that content can influence the model without ever holding an account on the system — the retrieved document becomes, in effect, code. The structural observation is that retrieval-augmented and tool-using architectures blur the line between data and instructions, and every control below exists to redraw it. A financial institution should assume that any corpus it retrieves from — vendor research, scraped filings, counterparty correspondence, a shared drive — is a channel an adversary may already be able to write to.

Common variants of prompt injection include:

+   - **Direct instruction override:** Inputs that instruct the model to ignore previous instructions or system prompts.
+   - **Tool misuse:** Inputs that attempt to coerce the model into calling tools with inappropriate parameters (e.g., executing arbitrary code, exfiltrating data).
+   - **Indirect prompt injection:** Malicious content hidden in retrieved documents or external data, which is then fed to the model as context.

To mitigate these risks, financial institutions should:

+   - implement strict allow-lists and schema validation for tool calls;
+   - separate untrusted input (e.g., user prompts, external documents) from trusted system instructions in prompt construction;
+   - treat tool outputs as untrusted data that require validation and sanitization; and
+   - implement monitoring and anomaly detection for unusual tool call patterns.

Applied to the research workstation, this means that retrieved documents should be quoted or summarized as evidence, not allowed to alter system instructions. A filing or transcript