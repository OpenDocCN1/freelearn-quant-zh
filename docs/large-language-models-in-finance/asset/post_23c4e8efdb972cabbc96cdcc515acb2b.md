
Some examples are simplified for exposition. Production deployments require additional security, observability, testing, access control, documentation, validation, and rollback procedures. The examples should therefore be treated as reference patterns rather than direct production implementations.

## Companion code and versioning

The official companion repository is Packt's official *LLMs in Finance* repository. The edition-specific companion supplied with this submission is release B32413-v1.0.0 at commit c80ddc04d29fc2ec245bd7bda03c6833e2b2392e. It is archived both as `LLM_in_Finance_COMPANION_B32413-v1.0.0.zip` and as the immutable Git bundle `LLM_in_Finance_COMPANION_B32413-v1.0.0.gitbundle`. Those artifacts, not a moving default branch, are the canonical reference for this edition; the official Packt repository should mirror the same tag and commit. A reproducible run records the repository commit, Python environment, data snapshot, model and prompt versions, generated artifact hashes, and the command used to execute the experiment. The printed code emphasizes interfaces and invariants; the archived release contains complete runnable examples, tests, and generated results.

## Technical quality contract

This book uses a simple discipline throughout: every important financial LLM idea should be readable at four levels at once. First, it should be motivated by a real financial workflow. Second, it should be represented by a precise mathematical object. Third, it should map to an implementable algorithm or software pattern. Fourth, it should have an evaluation and governance protocol that would detect failure before the system is trusted in production.

This contract is included explicitly because financial AI projects often fail at the interfaces between these levels. A technically elegant model can fail because the data split leaks future information. A well-written prompt can fail because tool permissions are not enforced outside the model. A strong retrieval system can fail because answers are not checked claim by claim against retrieved evidence. Conversely, a modest model can be valuable if the workflow is clear, the data are clean, the permissions are deterministic, and the evaluation protocol is honest.

## The four-layer standard

For each architecture developed in the book, the intended standard is:

- 1. **Motivation.** State the financial decision problem before the model. The reader should know whether the system supports research, risk, trading, compliance, client service, operations, or governance.
- 2. **Mathematics.** Define the inputs, outputs, loss function, information set, timestamps, admissible actions, and uncertainty model before stating guarantees.
- 3. **Code.** Provide code that is typed, auditable, deterministic where possible, and explicit about what is a runnable reference implementation versus a production-dependent integration.
- 4. **Validation.** Measure task quality, calibration, leakage, faithfulness, latency, cost, and control failures under a protocol that mirrors the deployment setting.

The practical consequence is that the book avoids treating an LLM response as a primitive object. A response is acceptable only when it is linked to evidence, generated under an allowed policy, checked against the relevant numerical and textual constraints, and recorded in a reproducible audit trail.

## Mathematical discipline

Whenever a result concerns financial deployment, the default object is not a static dataset but a time-indexed experiment. A data point is written as:

$$z_i = (x_i, y_i, \tau_i, g_i), \quad (1)$$

where $x_i$ is the input, $y_i$ is the target or reviewer judgment, $\tau_i$ is the availability time, and $g_i$ is an optional grouping variable such as issuer, portfolio, desk, jurisdiction, or product family. A valid evaluation at decision time $t$ may use only artifacts whose availability time is at most $t$.

The canonical supervised objective is:

$$\widehat{\theta} \in \arg \min_{\theta \in \Theta} \frac{1}{|\mathcal{I}_{train}|} \sum_{i \in \mathcal{I}_{train}} w_i \ell(f_\theta(x_i), y_i) + \lambda \Omega(\theta; \theta_0), \quad (2)$$

where the weights $w_i$ account for class imbalance, desk priority, or cost asymmetry, and $\Omega(\theta; \theta_0)$ controls deviation from the base model. The evaluation set is not a random afterthought: it is a formal object determined by time, group, and availability constraints.

For generative systems, quality is vector-valued. A financial LLM is not judged only by a semantic metric such as ROUGE, BLEU, cosine similarity, or pairwise preference. It is judged by a tuple:

$$Q = (Q_{\text{task}}, Q_{\text{faith}}, Q_{\text{num}}, Q_{\text{cal}}, Q_{\text{policy}}, Q_{\text{lat}}, Q_{\text{cost}}), \quad (3)$$

covering task usefulness, evidence faithfulness, numerical consistency, calibration, policy compliance, latency, and cost. This is the reason the later chapters repeatedly combine model metrics with governance metrics.

### Code discipline

The companion code follows five rules:

- 1. **Typed boundaries.** Inputs, outputs, evidence items, tool calls, and claims are represented by explicit data structures rather than free-form dictionaries whenever possible.
- 2. **Fail closed.** If a timestamp, permission, citation, schema, or numerical check cannot be verified, the reference pattern rejects or escalates rather than silently continuing.
- 3. **No hidden future information.** Index construction, prompt examples, labels, and evaluation artifacts must pass an availability-time audit.
- 4. **Deterministic tests.** Minimal examples are small enough to run in continuous integration and do not depend on private infrastructure.
- 5. **Separation of concerns.** Retrieval, generation, tool execution, policy checks, and audit logging are separate functions; prompts are not used as substitutes for access control.

### Explanatory discipline

A high-quality explanation in financial AI must tell the reader why the method is needed, what assumption makes it valid, how it can fail, and how that failure would be detected. Therefore, the chapters use a repeated pattern: start from a financial pain point, introduce the mathematical abstraction, show the implementable pattern, then state the validation or governance hook. This is also the recommended way to present internal financial AI projects to model risk, compliance, legal, investment, and technology stakeholders.

### Notation and abbreviations

This section fixes the notation used repeatedly in the book. The notation is intentionally operational: symbols are chosen so that they can be mapped directly to code artifacts such as datasets, prompts, retrieval indexes, tools, policies, and audit logs.

| Symbol or term | Meaning |
| :--- | :--- |
| LLM | Large Language Model. |
| RAG | Retrieval-Augmented Generation: a generation pipeline grounded in retrieved evidence. |
| MCP | Model Context Protocol: a standard host-client-server protocol for connecting LLM applications to resources, prompts, and tools. Financial authorization and governance remain implementation obligations. |
| D | Dataset or document corpus. In empirical sections, $D_{\le t}$ denotes only artifacts available at decision time $t$. |
| X, Y | Input and output spaces. For classification, $x \in X$ is a text or feature object and $y \in Y$ is its label. |

| Symbol | Description |
| :--- | :--- |
| $\ell(\hat{y}, y)$ | Task loss. Examples include cross-entropy for classification, squared error for regression, and claim-level support loss for grounded generation. |
| $R(\theta)$ and $\hat{R}(\theta)$ | Population and empirical risks of model parameters $\theta$. |
| $\mathcal{F}_t$ | Filtration: information available up to time $t$. This is the formal object behind “no future information.” |
| $\mathcal{F}_t^{MCP}$ | Information available to a workflow after MCP latency, authorization, redaction, and bandwidth constraints. |
| $A_i$ | Atomic claim extracted from a generated answer. |
| $E_i$ | Evidence bundle retrieved for claim $A_i$, including document identifiers, timestamps, and passages. |
| $E_i \vDash A_i$ | Evidence bundle $E_i$ supports claim $A_i$ under a specified verifier and admissible corpus. |
| Faith($a, E$) | Claim-level faithfulness score: fraction of material claims in answer $a$ supported by evidence $E$. |
| Gate | Policy gate that maps a proposed tool call and state to allow, reject, or escalate. |
| $\pi_\theta$ | Model or agent policy with parameters $\theta$. In trading contexts, it maps information states to actions; in LLM contexts, it maps prompts and context to token/action distributions. |
| $V(t, x)$ | Value function in dynamic-programming formulations. |
| $\Delta f$ | Sensitivity of a function in differential privacy. |
| $\epsilon, \delta$ | Differential-privacy parameters. |
| $\lambda$ | Kyle-style price-impact coefficient; model-specific, not a universal entropy transform. |
| Human oversight | A control process with named responsibility, evidence review, escalation authority, and override/kill-switch capability. |

**Implementation convention.** The companion code uses the same conceptual objects. A `Document` corresponds to an element of $\mathcal{D}$; a `Claim` corresponds to $A_i$; a `ToolCall` is a candidate action evaluated by `Gate`; and an audit record stores the model, prompt, corpus, tool, and policy versions needed to reproduce a material output.

### Conventions used

In this book, you will find a number of text styles that distinguish between different kinds of information. Here are some examples of these styles and an explanation of their meaning.

Code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, and user input are shown as follows: “You can confirm Python is running by importing the `antigravity` module at the `>>>` prompt.”
