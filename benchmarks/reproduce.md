# Benchmark Reproduction & Metric Provenance Specification

This document provides the exact methodology, hardware environments, reproduction commands, and raw telemetry provenance for all metrics and claims published across the **AetherControl** repository.

---

## 1. Metric Classification Taxonomy

To ensure absolute systems engineering rigor, all metrics in this repository are categorized into one of four explicit classes:

| Class | Definition | Caveat / Boundary |
| :--- | :--- | :--- |
| **`[HARDWARE-MEASURED]`** | Executed directly on physical GPU silicon under active CUDA / driver execution. | Measures real latency and throughput on specific hardware; subject to driver and batch parameters. |
| **`[ANALYTICAL MODEL]`** | Derived from closed-form mathematical equations and queueing models. | Proves mathematical ceiling / asymptotic bounds; does **not** assert empirical driver traces. |
| **`[MICRO-BENCHMARK]`** | In-memory evaluation measuring isolated software function timing. | Times Python/RAM operations in isolation; does not represent distributed end-to-end network latency. |
| **`[SYNTHETIC MODEL]`** | Algorithmic evaluation executed over synthetic sequence sets. | Proves algorithmic reduction properties; not tied to an end-to-end training throughput run. |

---

## 2. Complete Metric Provenance Registry

| Claimed Metric | Repository Class | Methodology & Formula | Source Code Path |
| :--- | :--- | :--- | :--- |
| **1,141.41 tokens/s** | `[HARDWARE-MEASURED]` | Stock vLLM 0.7.2 benchmarked on RTX 4090 (50 requests, concurrency 8, max 128 tokens). Control plane was not inline. | [`benchmarks/reports/live_silicon_execution.md`](benchmarks/reports/live_silicon_execution.md) |
| **22.1 ms P50 TTFT** | `[HARDWARE-MEASURED]` | FlashAttention-2 prefill latency on RTX 4090 (vLLM 0.7.2, chunked prefill 2048). P99 (110ms) is based on 50 samples. | [`benchmarks/reports/live_silicon_execution.md`](benchmarks/reports/live_silicon_execution.md) |
| **6.5 ms P50 TPOT** | `[HARDWARE-MEASURED]` | Autoregressive token decode latency bounded by RTX 4090 HBM bandwidth (886.1 GB/s). | [`benchmarks/reports/live_silicon_execution.md`](benchmarks/reports/live_silicon_execution.md) |
| **GRPO Reward (0.325 ──► 0.850)** | `[TRAINING-BATCH]` | 10-step HuggingFace TRL `GRPOTrainer` smoke test with PEFT LoRA ($r=16, \alpha=32$). Measures batch reward across 4 prompts/step ($n=40$ completions total). **Not a held-out test evaluation.** | [`rlhf-pipeline/rlhf_pipeline/grpo_trl_trainer.py`](rlhf-pipeline/rlhf_pipeline/grpo_trl_trainer.py) |
| **−98.6% Driver Mutex Contention** | `[ANALYTICAL MODEL]` | $\text{Reduction} = 1.0 - \frac{1,120\text{ queries/s}}{80,000\text{ queries/s}} = 98.6\%$. Models 800 theoretical GPUs comparing 500ms flat scraping vs tiered 5s/15s polling. No physical driver ioctl measured. | [`k8s-infra/k8s_infra/telemetry_governor.py`](k8s-infra/k8s_infra/telemetry_governor.py#L46-L82) |
| **45.0 ms P99 TTFT Lock** | `[ANALYTICAL MODEL]` | Formulaic queue scaling function estimating latency ceiling when driver lock contention is eliminated. | [`k8s-infra/k8s_infra/telemetry_governor.py`](k8s-infra/k8s_infra/telemetry_governor.py#L83-L97) |
| **0.005 ms XID Trap Isolation** | `[MICRO-BENCHMARK]` | Python `re.search` execution time on a mock `/dev/kmsg` string. End-to-end K8s API patch + taint takes ~200ms–2s in production. | [`k8s-infra/k8s_infra/telemetry_governor.py`](k8s-infra/k8s_infra/telemetry_governor.py#L140-L170) |
| **0.0007 ms /dev/shm Scrape Read** | `[MICRO-BENCHMARK]` | Python in-memory dictionary lookup time reading pre-parsed metrics from RAM. | [`k8s-infra/k8s_infra/telemetry_governor.py`](k8s-infra/k8s_infra/telemetry_governor.py#L105-L135) |
| **40.2pt Padding Token Reduction** | `[SYNTHETIC MODEL]` | Heuristic padding reduction on synthetic set `[120, 240, 800, 1500, 300, 950, 450, 1800, 210, 650]`. Waste drops from 50.4% to 10.2% vs random order. | [`trainsight/trainsight/remediators/bin_packer.py`](trainsight/trainsight/remediators/bin_packer.py#L50-L74) |
| **Single-Layer Meta-Probe (7.34 ms)**| `[MICRO-BENCHMARK]` | Single-layer tensor instantiation on `torch.device('meta')` executed on host CPU (4.04ms on Apple Silicon MPS/CPU). Zero GPU VRAM allocated or CUDA kernels executed; latency measures Python runtime and transformers config traversal. Bound by `META_PROBE_SLA_MS = 15.0`. | [`trainsight/trainsight/profilers/meta_probe.py`](trainsight/trainsight/profilers/meta_probe.py) |

---

## 3. Step-by-Step Hardware Reproduction Harness

### Setup A: Bare-Metal Silicon (NVIDIA GeForce RTX 4090)
* **Instance ID**: Vast.ai `#50399758`
* **GPU**: 1x NVIDIA GeForce RTX 4090 (24 GB VRAM, 886.1 GB/s bandwidth)
* **Host CPU**: AMD EPYC 7C13 64-Core Processor, PCIe 4.0
* **Driver / CUDA**: NVIDIA Driver `570.211.01` | CUDA `12.8` | Ubuntu 24.04 LTS

#### 1. Live Stock vLLM Inference SLA Benchmark
```bash
# 1. Launch stock vLLM API server
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 4096 \
  --enable-chunked-prefill \
  --enable-prefix-caching > vllm.log 2>&1 &

# 2. Wait for startup completion:
grep "Application startup complete" vllm.log

# 3. Execute vllm-bench SLA benchmark harness:
python3 -m vllm_engine.cli benchmark \
  --host http://localhost:8000 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --num-requests 50 \
  --concurrency 8 \
  --max-tokens 128
```
* **Observed Output**:
  ```text
  Success / Total Requests: 50 / 50 (100%)
  Output Token Throughput: 1141.41 tokens/s
  TTFT (Prefill Latency) P50: 22.1 ms | P99: 110.3 ms
  TPOT (Decode Latency) P50: 6.5 ms/token | P99: 6.8 ms/token
  ```
  *(Statistical Caveat: $N=50$ requests total; P99 represents a single request tail sample ($1/50 = 2\%$). True production SLA verification requires $\ge 10,000$ requests under Poisson load).*

#### 2. Live HuggingFace TRL GRPO Training Loop Smoke Test
```bash
# Run 10-step distributed GRPO fine-tuning loop on GSM8K with LoRA:
python3 rlhf-pipeline/rlhf_pipeline/grpo_trl_trainer.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --steps 10
```
* **Observed Batch Telemetry**:
  ```text
  Step 5:  {'rewards/reward_function_accuracy': 0.20, 'reward': 0.325}
  Step 10: {'rewards/reward_function_accuracy': 0.70, 'reward': 0.850}
  Runtime: 80.09s (0.125 steps/s)
  ```
  *(Note: Evaluated across 4 training batch prompts per step; not an evaluation on the 1,319 GSM8K test split).*

---

### Setup B: Cloud Kubernetes Cluster (Google Cloud GKE)
* **Cluster**: GKE Standard, 1x `g2-standard-4` node (NVIDIA L4 24GB VRAM)
* **Region**: `us-central1-a`
* **Raw Logs**: [`docs/GCP_Cloud_Proof_Evidence.md`](docs/GCP_Cloud_Proof_Evidence.md)

```bash
# Provision cluster and run GKE SLA load test:
make cloud-up
vllm-bench benchmark --host http://<GKE_INGRESS>:8000 --num-requests 50 --concurrency 8
```
* **Observed Output**:
  ```text
  Output Token Throughput: 48.89 tokens/s
  TTFT P50: 855.4 ms
  TPOT P99: 43.0 ms/token
  ```

---

### Setup C: Zero-GPU Analytical Invariant Suite (Local CPU)
No GPU required. Runs purely in Python in $<15\text{ seconds}$:
```bash
# Run all 64 unit tests:
make test

# Run 4-stage analytical invariant model:
make benchmark
```
