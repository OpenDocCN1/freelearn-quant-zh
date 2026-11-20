+   • Communication review agents that check communications (e-mails, research reports, client presentations) for policy violations.

In these settings, agentic systems must be designed with conservative defaults, strict guardrails, and strong human oversight. Agents should not make final regulatory interpretations or risk decisions; instead, they should support risk officers and compliance professionals by retrieving relevant information, highlighting potential issues, and suggesting questions for further investigation. In the research workstation, this is the function of the Risk and Compliance agent: it annotates the draft memo with concentration risks, restricted-list conflicts, disclosure requirements, and escalation flags, but it does not replace the judgment of the risk officer or compliance reviewer.

# 5.3.3 Operations and client service

Operational and client service workflows often involve repetitive tasks, structured data, and standardized communications, making them amenable to partial automation. LLM agents can assist with:

+   • processing client inquiries, routing them to the appropriate teams, and providing draft responses;
+   • reconciling trade data, positions, or cash flows;
+   • generating summaries of complex cases (e.g., investigations or disputes);
+   • coordinating between different operational teams or systems.

Here too, agentic systems must be designed with careful escalation rules and monitoring to ensure that errors are detected and corrected promptly, and that clients receive accurate, compliant communications. Although the research workstation is framed around investment research, the same architecture can be adapted to operations: a coordinator routes cases, specialist agents inspect transaction data or client records, and a human operator approves any external communication or irreversible action.

# 5.3.4 From application idea to implementation requirements

Before building any of these systems, teams can translate the application into a small set of implementation requirements. The following questions should be read as proposed design criteria rather than established best practice across all financial institutions. For the research workstation, such criteria are not merely technical; they also define what the system is and is not allowed to do. A useful requirements document can answer the following questions:

+   1. What are the accepted inputs? The system may accept natural-language requests from portfolio managers, but it should also require structured fields such as tickers, portfolio identifiers, analysis horizon, and intended use. A request for an internal pre-investment memo is materially different from a request for client-facing commentary. The same agent output may be acceptable in one context and inappropriate in another.
+   2. What are the required outputs? A useful output is a packet, not only prose. In the workstation example, the packet contains the draft memo, evidence identifiers, computed