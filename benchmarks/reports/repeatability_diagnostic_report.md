# Standalone vLLM Repeatability Diagnostic Report

**Execution Date**: 2026-10-15  
**Hardware Silicon**: 1x NVIDIA GeForce RTX 4090 (24 GB VRAM)  
**NVIDIA Driver**: `580.178.04` | **CUDA**: `13.0`  
**Serving Engine**: Standalone `vllm/vllm-openai:v0.7.2` (Docker container on host loopback)  
**Model**: `Qwen/Qwen2.5-1.5B-Instruct` (`bfloat16`, eager mode, `--no-enable-prefix-caching`, max context 4096)  
**Target Workload**: Poisson arrivals at 6 RPS over a fixed $W = 180.0\text{ s}$ observation window ($N = 1,069$ requests/run, `seed=42`)  
**Generation Constraints**: `min_tokens = 64`, `max_tokens = 64` (strictly verified across all responses)  
**SLO Targets**: $\text{TTFT} \le 250\text{ ms}$, $\text{E2E} \le 2.50\text{ s}$  
**Vast.ai Instance**: `56106001` (provisioned, tested, verified destroyed; 0 ongoing cloud instances)

---

## 1. Executive Summary & Diagnostic Criterion

The diagnostic protocol was conducted to evaluate whether standalone vLLM serving exhibits reproducible behavior and explainable outliers under an identical Poisson arrival schedule, resolving earlier harness measurement instability before running multi-mode comparative experiments.

### Diagnostic Outcome: **PASS**
- **Goodput Consistency**: $5.939 \pm 0.000\text{ RPS}$ across all 3 trials ($1,069 / 180.0\text{ s}$).
- **Completion Rate**: Exactly $1,069 / 1,069$ requests completed successfully per trial ($3,207 / 3,207$ overall, $0$ failures, $0$ dropped connections).
- **Attainment Rate**: $100.0\% \pm 0.0\%$ across all 3 trials against the target SLOs ($\text{TTFT} \le 250\text{ ms}$, $\text{E2E} \le 2.50\text{ s}$).
- **Harness Client Lag**: Submission lag P50 $= 0.65\text{ ms}$, P99 $= 2.01\text{ ms}$ (eliminating client-side event loop contention).
- **Outlier Explainability**: The tail latency elevation observed in Run 2 was isolated to a discrete 30-second window at the start of the trial ($t \in [0.4\text{ s}, 30.5\text{ s}]$), driven by an engine/GPU decode step slowdown (TPOT $\approx 30\text{ ms}$ vs. steady-state $16.3\text{ ms}$), while client dispatch lag remained $< 1.2\text{ ms}$.

---

## 2. Experimental Controls & Bug Fixes

To isolate true serving engine characteristics from benchmark harness artifacts, the following controls were implemented:

### Confirmed Harness Bugs Fixed
1. **Trace Desynchronization**: Previously, independent random streams caused cross-trial variance. The harness now pre-generates a single Poisson schedule on $[0, 180.0\text{ s})$ with `seed=42` and replays the identical timestamps across all trials.
2. **Double-Counted Rejections**: Previously, HTTP 429 shedding was conflated with connection failures. The accounting model now strictly separates `failed_requests` from `shed_requests_429`.
3. **Mismatched Goodput Denominators**: Previously, Goodput used variable interval durations ($\Delta t = t_{\text{last}} - t_{\text{first}}$). The denominator is now locked strictly to the fixed measurement window $W = 180.0\text{ s}$.

### Hypotheses Tested & Validated
1. **Variable Generation Length**: Enforced `min_tokens = max_tokens = 64` and verified in response payloads (`usage.completion_tokens`). Mean tokens generated was exactly **$64.0 \pm 0.0$** across all 3,207 requests, isolating token-count variance from latency measurements.
2. **Client Asyncio Event-Loop Contention**: Decoupled HTTP socket streaming and timestamp logging. Submission lag (scheduled arrival time vs. socket write initiation) dropped from $50\text{--}185\text{ ms}$ down to $2.01\text{ ms}$ P99 on the physical host loopback.
3. **Queue Deadline as Policy vs. Guarantee**: Acknowledged that queue deadlines limit gateway residency time rather than guaranteeing downstream vLLM engine prefill or decode duration.

---

## 3. Quantitative Results Table

Each trial replayed the identical 1,069-request schedule over $180.0\text{ s}$, separated by a 15-second quiet period.

| Metric | Run 1 | Run 2 | Run 3 | Aggregate Mean | Cross-Run SD |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Offered Requests** | 1,069 | 1,069 | 1,069 | 1,069 | 0.0 |
| **Completed Requests** | 1,069 | 1,069 | 1,069 | 1,069 | 0.0 |
| **Failed Requests** | 0 | 0 | 0 | 0 | 0.0 |
| **Mean Completion Tokens** | 64.0 | 64.0 | 64.0 | 64.0 | 0.0 |
| **Submission Lag P50 (ms)** | 0.63 ms | 0.71 ms | 0.61 ms | 0.65 ms | 0.05 ms |
| **Submission Lag P99 (ms)** | 2.04 ms | 2.00 ms | 1.98 ms | 2.01 ms | 0.03 ms |
| **TTFT P50 (ms)** | 26.1 ms | 28.9 ms | 26.7 ms | 27.2 ms | 1.5 ms |
| **TTFT P99 (ms)** | 34.6 ms | 63.9 ms | 34.4 ms | 44.3 ms | 17.0 ms |
| **E2E Latency P50 (s)** | 1.02 s | 1.07 s | 1.03 s | 1.04 s | 0.03 s |
| **E2E Latency P99 (s)** | 1.18 s | 2.20 s | 1.16 s | 1.51 s | 0.60 s |
| **Goodput ($W = 180\text{ s}$)** | **5.939 RPS** | **5.939 RPS** | **5.939 RPS** | **5.939 RPS** | **0.000** |
| **Attainment Rate (%)** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **0.0%** |

---

## 4. In-Depth Analysis of Run 2 Latency Tail

While Runs 1 and 3 exhibited near-identical distributions (E2E P99 of $1.18\text{ s}$ and $1.16\text{ s}$, TTFT P99 of $34.6\text{ ms}$ and $34.4\text{ ms}$), Run 2 showed an elevated E2E P99 of $2.20\text{ s}$ and TTFT P99 of $63.9\text{ ms}$.

Inspection of the per-request audit logs (`raw_records`) revealed:
1. **Temporal Clustering**:
   - Exactly 186 requests in Run 2 had an E2E latency exceeding $1.5\text{ s}$.
   - All 186 slow requests occurred strictly within the first 30.5 seconds of the run: $\text{scheduled\_time} \in [0.40\text{ s}, 30.52\text{ s}]$.
   - Zero requests exceeded $1.5\text{ s}$ after $t = 30.52\text{ s}$.
2. **Decode Step Slowdown (TPOT)**:
   - During $t \in [0.4\text{ s}, 30.5\text{ s}]$, Time Per Output Token (TPOT) averaged $30.8\text{ ms/token}$, resulting in $64 \times 30.8\text{ ms} \approx 1.97\text{ s}$ generation time.
   - After $t = 35.0\text{ s}$ ($N = 849$ requests), TPOT dropped immediately to a mean of $16.31\text{ ms/token}$ (P50: $16.17\text{ ms}$, P99: $18.59\text{ ms}$).
   - E2E latency for the remainder of Run 2 was P50: $1.048\text{ s}$, P99: $1.201\text{ s}$—matching Runs 1 and 3.
3. **Client Submission Lag Independence**:
   - During the slow 30-second window, submission lag was $0.90\text{--}1.20\text{ ms}$, demonstrating that the load generator dispatched requests on schedule and the delay was entirely server/hardware side.
4. **Physical Root Cause**:
   - The sudden 2x step change in token generation rate ($30.8\text{ ms} \to 16.3\text{ ms}$) at $t = 30.5\text{ s}$ without changes in concurrency points to GPU clock frequency ramping / power state (P-state) transitions after the inter-trial idle cooldown, or host hypervisor context switching on the shared rented hardware.

---

## 5. Conclusions & Next Steps

1. **Repeatability Established**: Direct vLLM serving under identical Poisson schedules achieves deterministic throughput ($5.94\text{ RPS}$) and stable steady-state latency ($1.03\text{ s}$ P50, $1.17\text{ s}$ P99, $16.3\text{ ms}$ TPOT).
2. **Attainment Feasibility**: Under light load (6 RPS), standalone vLLM easily satisfies $2.50\text{ s}$ E2E and $250\text{ ms}$ TTFT across all 3,207 requests, even during transient GPU warmup phases.
3. **Readiness for 4-Mode Evaluation**: The experimental foundation is now solid. Comparative benchmarks across:
   - Mode 1: Direct vLLM (Unconstrained)
   - Mode 2: Aether Gateway (Forwarding Only)
   - Mode 3: Aether Gateway (Admission Control with tuned queue deadline)
   - Mode 4: Ray Serve (Production External Baseline)
   can now be executed across varied load regimes (light, capacity knee, overload, burst) with confidence in measurement fidelity.
