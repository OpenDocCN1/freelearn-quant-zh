as controlled workflow components rather than as unconstrained autonomy: tool schemas, permissions, evidence capture, and escalation rules are part of the agent definition.

We begin by introducing core definitions and terminology, focusing on the difference between *LLM-centric agents* and traditional software components. We then present architectural patterns for single-agent and multi-agent systems, including orchestration strategies, memory design, and tool integration. We conclude with case studies and design templates for multi-agent systems in research, risk, and operations, as well as a discussion of open problems and research directions.

In a nutshell, the following topics will be covered in this chapter:

+   - Running example: Multi-agent research workstation
+   - Understanding the core concepts of agentic systems
+   - Failure modes and control requirements
+   - Applications of LLM agent systems in finance
+   - Design patterns for governed agentic workflows
+   - Implementation roadmap
+   - Open problems and research directions

# # Running example: Multi-agent research workstation

Throughout this chapter, we will refer to a running example: a multi-agent research workstation for a buy-side investment team. The goal of the workstation is to support a portfolio manager and a small group of analysts in producing high-quality, documented investment theses under time constraints.

At a high level, the workstation coordinates several specialized agents:

+   - a **Research Analyst agent** that retrieves, summarizes, and compares company and macro research;
+   - a **Data Analyst agent** that fetches and analyzes structured data (prices, fundamentals, alternative data);
+   - a **Risk and Compliance agent** that checks proposed conclusions against risk limits, internal policies, and regulatory constraints;
+   - a **Coordinator or Orchestrator agent** that decomposes complex user requests into sub-tasks, routes them to the appropriate agents, and reconciles their outputs into a coherent report.

User queries such as “Prepare a pre-investment memo on ACME Corp, focusing on earnings quality and downside risks” are handled by the orchestrator, which invokes the analyst, data, and risk agents as specialized modular components. This example will serve as a concrete anchor as we discuss agent roles, communication patterns, memory, tooling, and orchestration strategies in the sections that follow. We will deliberately return to the same workstation at each major design step so that the chapter reads less like a collection of isolated concepts and more like a progressive build of one finance-specific agentic system.