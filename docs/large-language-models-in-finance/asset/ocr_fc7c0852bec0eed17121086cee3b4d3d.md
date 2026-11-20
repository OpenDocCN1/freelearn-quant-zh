TPUs can be attractive when the workload maps efficiently to XLA and the institution is comfortable operating in Google Cloud. Their regular architecture and high-speed inter-chip interconnect can provide strong throughput for large, stable workloads. Nevertheless, performance-per-cost and tail latency must be measured on the institution’s own models.

Compilation overhead, input-pipeline behavior, host-to-device transfer, batching, and topology all affect realized performance. TPU access is cloud-based, which can be incompatible with deployment policies requiring customer-controlled physical infrastructure or a specific sovereign environment.

# Specialized and emerging AI accelerators

Specialized accelerators offer alternatives to conventional GPU and TPU execution, but they introduce migration and ecosystem risk. Cerebras’ current CS-3 system is based on the WSE-3 and is specified with 900,000 AI-optimized cores, 44 GB of on-chip SRAM, and 21 PB/s of on-chip memory bandwidth [Cerebras Systems, 2026].

The design can simplify the mapping of very large models by presenting a wafer-scale compute fabric, although its economics and software fit must be assessed at system rather than chip level.

SambaNova systems built around the SN40L use reconfigurable dataflow units. A representative 16-RDU configuration is specified with 1 TB of aggregate HBM and 10.2 PFLOPS of bfloat16 processing capability [SambaNova Systems, 2025]. The dataflow compiler can reduce manual kernel engineering for supported models, but model coverage, compiler behavior, observability, and integration with existing serving systems require validation.

Graphcore’s documented GC200 IPU provides 1,472 processor cores, 900 MB of in-processor memory, and 250 TFLOPS at supported FP16 formats [Graphcore, 2026]. Its fine-grained parallelism can suit irregular or sparse workloads, but the Poplar ecosystem is materially smaller than CUDA’s. Procurement availability, roadmap continuity, and cloud access should therefore be confirmed before an institution makes a long-lived platform commitment.

The comparison in Table 10.1 is intentionally qualitative. A defensible procurement process should benchmark the exact model and serving policy with the same prompt-length distribution, output-length distribution, concurrency, batching limits, quantization, and latency objective expected in production. Comparing peak throughput while allowing one platform a much larger batch can produce a misleading conclusion for latency-sensitive applications.