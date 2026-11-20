One boundary must be fixed before any of this is useful, because the rest of the book depends on it. A prompt can *express* a regulatory requirement; it cannot *enforce* one. An instruction not to give investment advice, not to exceed a position limit, or not to cite an unavailable document is a request to a stochastic system, and a stochastic system will occasionally decline it — under distribution shift, under adversarial input, or for no visible reason at all. Enforcement is the job of deterministic machinery outside the model: permission checks, schema validation, policy gates, and human review. Treat every constraint in a prompt as a hint that improves the odds, and place the control that actually binds somewhere the model cannot reach. With that understood: financial prompts must encode context, role, constraints, and output expectations in ways that reduce ambiguity and align model behavior with professional standards. We'll begin with structured prompt design frameworks that formalize how tasks are specified. We then examine advanced prompting techniques, including conditional logic and meta-prompting. Together, these elements transform prompts from informal instructions into repeatable specifications.

# Structured prompt design

Structured prompt design gives you a repeatable way to specify financial tasks. Rather than writing free-form instructions, the most effective prompts follow a clear structure: they set the context, define the task, state the constraints, and specify the required output format:

```
CONTEXT: [Market conditions, time period, regulatory framework]
TASK: [Specific financial analysis or computation required]
CONSTRAINTS: [Regulatory requirements, risk parameters, assumptions]
FORMAT: [Output structure, precision requirements, compliance notes]
EXAMPLES: [Representative input-output pairs for guidance]
```

This structure reduces ambiguity. It ensures the model understands not just what to do, but the conditions under which it should do it. For example, if you request a credit risk analysis, the prompt should clearly state the regulatory framework (such as Basel III or IFRS 9), define risk appetite parameters, and specify required output metrics and documentation standards. Without this information, the model may generate something plausible but misaligned with institutional requirements.

Establishing context matters especially in finance. The same analytical method can produce very different conclusions depending on market conditions or regulation. Effective prompts therefore include temporal markers, market indicators, regulatory frameworks, and any relevant institutional constraints:

```
Context: Analyzing Q3 2024 earnings for large-cap technology companies
under current Federal Reserve monetary policy (5.25% federal funds
rate) and updated SEC disclosure requirements for AI-related
investments. Market volatility (VIX) currently at 18.5, with technology
sector trading at 23.2x forward P/E ratio.
```

Role-based prompting adds another layer of precision. By specifying the professional perspective, you guide both depth and tone of your model's responses. In financial settings, this typically involves defining technical seniority and institutional context: