# Learning Dossier 06: Production vLLM Engine Configuration & SLA Benchmarking

> **Folder Path**: `docs/dossier/06_vllm_engine_configuration_and_benchmarking.md`  
> **Session Topic**: vLLM Engine Memory Tuning, PagedAttention, Continuous Batching, and Standard SLA Load Testing  
> **Date**: July 24, 2026  

---

## 1. Production vLLM Engine Tuning Parameters

In production LLM infrastructure, running vLLM with default settings often leaves GPU performance underutilized or vulnerable to latency spikes. We tune the following key parameters:

```
                                [ vLLM MEMORY LAYOUT ]
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TOTAL GPU VRAM (100%)                            │
├──────────────────────────────┬───────────────────────────────┬──────────────┤
│  PyTorch & CUDA Workspace    │   Model Weights (FP16/BF16)   │  KV-Cache    │
│  (10% Headroom / Allocation) │   (e.g., ~3 GB for 1.5B)      │  Block Pool  │
└──────────────────────────────┴───────────────────────────────┴──────────────┘
 ▲                              ▲                              ▲
 └── gpu_memory_utilization     └── Model size                 └── PagedAttention
     (0.90 = 90% reserved)                                         Blocks Pool
```

### Parameter Breakdown

1. **`--gpu-memory-utilization 0.90`**
   - **Why 0.90?** Reserves 90% of GPU VRAM for model weights and the PagedAttention KV-cache block pool. Leaving a 10% cushion prevents PyTorch host-side dynamic allocations (e.g., intermediate activation tensors during prefill) from causing CUDA OOM crashes.
2. **`--block-size 16`**
   - **PagedAttention Granularity:** Dictates the token capacity per physical KV block. A smaller block size (16 tokens) reduces internal memory fragmentation; larger block sizes (32 tokens) improve cache alignment on modern Tensor Cores.
3. **`--enable-chunked-prefill True` & `--max-num-batched-tokens 2048`**
   - **Prevents TPOT Spikes:** Large prompt prefills can block decode iterations for 100+ ms. Chunking splits huge prompts into 2048-token chunks across iterations, preserving low **Time Per Output Token (TPOT)** for active generation streams.
4. **`--enable-prefix-caching`**
   - **Radix Tree KV Cache Sharing:** Caches KV blocks for shared prompt prefixes (e.g., common system prompts or multi-shot RAG context), drastically reducing prefill latency for recurrent requests.

---

## 2. Standard Industry SLAs for LLM Serving

When evaluating production LLM serving platforms, performance is measured across 4 core SLAs:

| SLA Metric | Abbreviation | Definition & Target Standard |
| :--- | :--- | :--- |
| **Time-To-First-Token** | **TTFT** | Time from request start to first streaming token. Target: $< 100\text{ ms}$ ($P_{90}$) for chat models. |
| **Time-Per-Output-Token** | **TPOT** | Latency per generated output token during decode. Target: $< 30\text{ ms/token}$ ($P_{99}$) for real-time streaming. |
| **Request Throughput** | **RPS** | Completed requests per second across the engine cluster. |
| **Token Throughput** | **TPS** | Total generated output tokens per second across all active requests. |

---

## 3. `vllm-engine` Tools Added

* [config.yaml](../../vllm-engine/vllm_engine/config.yaml): Production engine settings file.
* [config_loader.py](../../vllm-engine/vllm_engine/config_loader.py): Pydantic configuration parser that outputs exact vLLM entrypoint CLI flags.
* [benchmark_serving.py](../../vllm-engine/vllm_engine/benchmarks/benchmark_serving.py): Standard load test & SLA percentiles evaluator ($P_{50}, P_{90}, P_{95}, P_{99}$).
* [cli.py](../../vllm-engine/vllm_engine/cli.py): CLI tool providing `vllm-bench show-config` and `vllm-bench benchmark`.
