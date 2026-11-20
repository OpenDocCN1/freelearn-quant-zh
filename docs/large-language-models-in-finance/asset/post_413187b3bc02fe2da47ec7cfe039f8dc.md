
The archived release is checked with:

## Listing 15.7: Running the included reference benchmarks and tests

```
cd companion
python experiments/rag_sec_filings/run_experiment.py
bash run_checks.sh
```

The first command regenerates results.csv and audit_log.jsonl; the second runs the unit tests, regenerates the deterministic benchmark outputs, verifies the Chapter 13 seed-11 table, and checks the declared 60-seed output for drift. These commands are part of the release contract, not evidence of live investment performance.

## Repository release contract

The edition release contains:

- 1. the semantic version tag B32413-v1.0.0 and immutable commit c80ddc04d29fc2ec245bd7bda03c6833e2b2392e;
- 2. pyproject.toml plus a hash-pinned dependency lock and vendored wheel for the supported environment;
- 3. continuous-integration and local tests that run without private credentials;
- 4. data cards, model-card and prompt-registry templates, and license information;
- 5. commands that regenerate every book result generated from code;
- 6. machine-readable output and dependency checksum manifests;
- 7. a source archive and a complete Git bundle supplied with the publisher submission.

## Table 15.4: Book-to-repository crosswalk for the reference package

| Book object | Repository location |
|---|---|
| Typed contracts and audit objects | src/finllm/contracts.py |
| Point-in-time checks | src/finllm/temporal.py and experiments/leakage_tests/ |
| Deterministic retrieval pattern | src/finllm/rag.py |
| Policy-gated tool execution | src/finllm/governance.py |
| Calibration and faithfulness metrics | src/finllm/metrics.py |
| Credential-free filing benchmark | experiments/rag_sec_filings/ |
| Chapter 13 point-in-time experiment | experiments/pit/ and experiments/OUTPUT_SHA256SUMS.txt |
| Regression tests | tests/test_core.py, run_checks.sh, and the continuous-integration workflow |

The manuscript and repository are therefore linked by stable contracts rather than by copying every long script into the printed page.

### Final-phase update: minimal checklist for agentic trading deployment

Before a trading or portfolio agent moves beyond read-only assistance, the deployment record should include: point-in-time evidence; sealed or walk-forward evaluation; fees, spreads, borrow, impact, and capacity; comparisons with simple baselines and strong learned baselines; order-level simulation; stress and reverse-stress tests; adversarial documents and prompt perturbations; governance ablations; deterministic replay; and an evidence-to-order-to-fill audit lineage. This checklist intentionally does not create a new chapter or a new framework. It is a compact final-stage control surface for the agentic material added throughout the book.

## 15.10 Summary

This chapter brought together the technical foundations and practical resources needed to support financial LLM development. The mathematical concepts, architectural comparisons, and practical examples presented here provide a reference point for designing, evaluating, and adapting financial LLM workflows.

Key takeaways include:

- Understanding attention, optimization, and loss functions helps readers connect transformer behavior to implementation choices
- Encoder-only, decoder-only, and encoder-decoder architectures support different financial NLP and generation tasks; FinEAS-style sentence embeddings illustrate the value of compact domain-specific sentiment modules
- Fine-tuning, monitoring, RAG, and evaluation examples should be treated as illustrative patterns that require adaptation before production use
- Financial datasets, market data sources, SEC filings, and library ecosystems must be checked for coverage, licensing, currency, and suitability
- Robust data pipelines, evaluation methods, FinEAS/FinBERT monitoring, and governance controls are necessary for reliable financial LLM workflows

The resources and patterns presented should be adapted to institutional requirements, regulatory constraints, and specific use cases. Because the financial AI ecosystem changes quickly, teams must keep resources, libraries, datasets, and evaluation practices up to date while maintaining stability, compliance, and governance.

## Get this book's PDF version and more

Scan the QR code (or go to packtpub.com/unlock). Search for this book by name, confirm the edition, and then follow the steps on the page.

![](img/c31705b361fdf81dee8826a2c4ee167f_512_0.png)

UNLOCK NOW

*Note: Keep your invoice handy. Purchases made directly from Packt do not require an invoice.*

## Other Books You May Enjoy

If you enjoyed this book, you may be interested in these other books by Packt:

![](img/c31705b361fdf81dee8826a2c4ee167f_513_0.png)

Building AI Agents for Finance
Hanane Dupoux, Fayssal El Mofatiche
ISBN: 978-1-83702-229-8

- Understand AI agents and agentic systems in finance
- Master core AI agent design patterns and apply them to finance
- Explore reasoning paradigms used in agentic workflows
- Design multi-agent orchestration using various architectural styles
- Build financial use cases with hands-on Python labs
- Evaluate and test AI agent behavior effectively
- Implement guardrails, tracing, and observability
- Apply AI agents across fundamental analysis, trading, research, and compliance

![](img/c31705b361fdf81dee8826a2c4ee167f_514_0.png)

## 30 Agents Every AI Engineer Must Build

Imran Ahmad
ISBN: 978-1-80610-901-2

- Deploy production-ready agent systems that scale securely and reliably
- Use LangChain and LangGraph to build autonomous agents with modular architectures
- Implement agents with sophisticated memory, planning, and reasoning capabilities
- Seamlessly integrate tools, APIs, and external data into agent workflows
- Establish robust evaluation frameworks to measure and optimize agent performance
- Implement guardrails and explainability features to ensure ethical and safe deployment
- Build multi-agent systems for complex, collaborative task orchestration
- Apply specific agent architectures across healthcare, finance, and legal domains

## Share Your Thoughts

Once you've read Large Language Models in Finance, we'd love to hear your thoughts! Scan the QR code below to go straight to the Amazon review page for this book and share your feedback.

![](img/c31705b361fdf81dee8826a2c4ee167f_515_0.png)

```
https://packt.link/r/1837024537
```

Your review is important to us and the tech community and will help us make sure we're delivering excellent quality content.

## Bibliography

Daron Acemoglu, Asuman Ozdaglar, and Alireza Tahbaz-Salehi. Systemic risk and stability in financial networks. _American Economic Review_, 105(2):564–608, 2015. doi: 10.1257/aer.20130456.

AI @ Meta. Llama 3. `https://ai.meta.com/blog/meta-llama-3/`, 2024.

Michael Aitken, Douglas Cumming, and Feng Zhan. Market manipulation and market quality. _Journal of Financial Markets_, 26:162–185, 2015.

Leman Akoglu, Hanghang Tong, and Danai Koutra. Graph based anomaly detection and description: a survey. _Data mining and knowledge discovery_, 29(3):626–688, 2015.

Robert Almgren and Neil Chriss. Optimal execution of portfolio transactions. _Journal of Risk_, 3:5–40, 2001.

Mohammed Alsheikh, Roderick Bloem, Ruediger Ehlers, Bettina Koenighofer, Scott Niekum, and Ufuk Topcu. Safe reinforcement learning via shielding. In _Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence_, pages 2669–2678, 2018. URL `https://ojs.aaai.org/index.php/AAAI/article/view/11797`.

Edward I Altman. Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. _The journal of finance_, 23(4):589–609, 1968.

Kartik Anand, Sophia Kazinnik, Agnese Leonello, and Ettore Panetti. Ex machina: Financial stability in the age of artificial intelligence. Working Paper Series 3225, European Central Bank, May 2026. URL `https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp3225~52d8adae99.en.pdf`.

Torben G Andersen, Tim Bollerslev, Francis X Diebold, and Clara Vega. Micro effects of macro announcements: Real-time price discovery in foreign exchange. _American economic review_, 93(1):38–62, 2003.

Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, and Yuyang Wang. Chronos: Learning the language of time series, 2024. URL `https://arxiv.org/abs/2403.07815`.

Abdul Fatir Ansari, Oleksandr Shchur, Jaris Küken, Andreas Auer, Boran Han, Pedro Mercado, Syama Sundar Rangapuram, Huibin Shen, Lorenzo Stella, Xiyuan Zhang,
