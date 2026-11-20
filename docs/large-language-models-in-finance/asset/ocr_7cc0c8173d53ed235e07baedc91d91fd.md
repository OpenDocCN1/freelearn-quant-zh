# # 5.1 Understanding the core concepts of agentic systems

The term “agent” has a long history in artificial intelligence, multi-agent systems, and software engineering. In recent discussions around LLMs, however, it is often used loosely—sometimes to describe a simple prompt-and-response setup, and at other times to refer to complex, multi-step systems. This ambiguity can make it difficult to reason clearly about how these systems are actually designed and deployed.

In this section, we start by clarifying what we mean by an LLM agent and a multi-agent system. We then build on these definitions to examine how single-agent and multi-agent setups differ in practice, how responsibilities are distributed across agents, and how memory and state are managed. While we establish the terminology, we also keep the research workstation in view, using it as a working mental model for how agentic systems are designed in real finance workflows.

**Separation from traditional software components**

It is important to distinguish LLM agents from traditional microservices or APIs. A classical microservice implements a well-specified mapping from requests to responses, typically using deterministic logic and a fixed schema. An LLM agent, by contrast, relies on a learned model that exhibits stochastic behavior, operates in natural language, and generalizes across a wide range of inputs rather than adhering to a fixed contract.

This flexibility is both a strength and a liability. On the one hand, LLM agents can handle tasks that would be prohibitively expensive to encode procedurally (for example, reading and synthesizing free-form research reports). On the other hand, the lack of strict guarantees on behavior introduces challenges for reliability, interpretability, and compliance. For financial systems in particular, this trade-off is not optional but central to system design. Agentic architectures must therefore combine the expressive power of LLMs with strong monitoring and human oversight.

We adopt the following concise terminology throughout this chapter.

# ## 5.1.1 LLM agent

An LLM agent is an LLM-centric software component that is assigned a role, provided with tools and context, and tasked with achieving specific objectives through a sequence of actions. More concretely, it is a software component that:

+   - exposes an interface for receiving inputs—typically a combination of natural language instructions, structured context, and state;
+   - invokes an LLM, possibly multiple times, to reason about the input and plan actions;
+   - interacts with tools, data sources, or other agents to perform tasks; and
+   - produces outputs that may include natural language, structured data, or new actions.

Formally, an LLM agent can be viewed as a policy

```
$\pi_\theta : \mathcal{S} \rightarrow \mathcal{A}, \qquad (5.1)$
```