4. run the shocks through a pricing/risk engine,
5. use the LLM to summarize the impact by desk, product, and risk factor.

# 7.2.3 Real-time market surveillance

Market-surveillance systems monitor trading activity for potential market abuse (manipulation, spoofing, layering, and insider trading). They generate large volumes of alerts based on statistical triggers and rule-based filters [Cumming et al., 2020, Aitken et al., 2015].

+   LLMs assist by:
+   • analyzing trader communications (emails, chats) to identify narratives consistent with suspicious behavior,
+   • summarizing and contextualizing alerts for human investigators,
+   • clustering similar alerts and highlighting patterns across desks, instruments, or time [Golbeck et al., 2018].

Here, LLMs primarily act as *triage engines*, not as decision-makers. Human investigators remain responsible for final decisions and regulatory filings.

# 7.3 Back office: Banking, fraud, AML, and operations

Back-office applications are less focused on real-time trading decisions and more focused on accurate processing, documentation, investigation, and compliance support. Back-office functions include credit underwriting, transaction monitoring, KYC/AML processes, and a wide variety of operational workflows. These processes generate and consume large volumes of structured data, unstructured text, and documents [Arner et al., 2017].

# 7.3.1 Credit underwriting and review

In retail and small and medium-sized enterprise (SME) banking, credit underwriting typically combines:

+   • structured data (income, balance, transaction history, bureau scores) [Thomas et al., 2017],
+   • document-based information (pay slips, tax returns, financial statements),
+   • narrative information (business plans, explanations of purpose of funds).

LLMs can:

+   • extract structured features from documents (income, expenses, collateral) [Huang et al., 2022, Kim et al., 2022],
+   • summarize applicant profiles into standardized credit memos,
+   • flag inconsistencies or missing information [Netzer et al., 2019].

The goal is to augment, not replace, established credit-scoring models. LLM-derived features feed into existing PD/LGD frameworks and manual credit committees [Berg et al., 2020].