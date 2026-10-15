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

> **Headline Finding**: The corrected harness completed three repeatability trials at 6 RPS with all 3,207 requests meeting the declared latency targets; an early-run latency slowdown remains unexplained.

The diagnostic protocol was conducted to evaluate whether standalone vLLM serving exhibits reproducible behavior and explainable outliers under an identical Poisson arrival schedule, resolving earlier harness measurement instability before running multi-mode comparative experiments.

### Diagnostic Outcome: **PASS**
- **Goodput Consistency**: $5.939 \pm 0.000\text{ RPS}$ across all 3 trials ($1,069 / 180.0\text{ s}$). *(Note: Zero goodput variance across the three runs is an expected mathematical consequence of identical offered request counts, the identical fixed 180.0-second denominator, and 100% attainment. Latency distributions still varied across runs).*
- **Completion Rate**: Exactly $1,069 / 1,069$ requests completed successfully per trial ($3,207 / 3,207$ overall, $0$ failures, $0$ dropped connections).
- **Attainment Rate**: $100.0\% \pm 0.0\%$ across all 3 trials against the target SLOs ($\text{TTFT} \le 250\text{ ms}$, $\text{E2E} \le 2.50\text{ s}$).
- **Harness Client Lag**: Submission lag P50 $= 0.65\text{ ms}$, P99 $= 2.01\text{ ms}$ (eliminating client-side event loop contention).
- **Outlier Breakdown**: An early-run latency elevation in Run 2 was isolated to a discrete window at the start of the trial ($t \le 30.5\text{ s}$), driven by an engine/GPU decode step slowdown (TPOT $\approx 30.8\text{ ms}$ vs. steady-state $16.3\text{ ms}$), while client dispatch lag remained $< 1.2\text{ ms}$.

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

## 4. In-Depth Analysis & Request Reconciliation of Run 2 Latency Tail

While Runs 1 and 3 exhibited near-identical distributions (E2E P99 of $1.18\text{ s}$ and $1.16\text{ s}$, TTFT P99 of $34.6\text{ ms}$ and $34.4\text{ ms}$), Run 2 showed an elevated E2E P99 of $2.20\text{ s}$ and TTFT P99 of $63.9\text{ ms}$.

Inspection and exact reconciliation of the per-request audit logs (`raw_records`) across the 1,069 requests in Run 2 establishes the following discrete phases:

### Request Reconciliation ($N = 1,069$)
1. **Early Slowdown Interval ($t \le 30.50\text{ s}$)**: **189 requests**
   - 186 requests had $\text{E2E} > 1.5\text{ s}$ (ranging $1.51\text{--}2.20\text{ s}$).
   - 3 requests finished just under the threshold ($1.35\text{--}1.48\text{ s}$).
   - Time Per Output Token (TPOT) averaged $30.8\text{ ms/token}$, resulting in $64 \times 30.8\text{ ms} \approx 1.97\text{ s}$ generation time.
2. **Transition Recovery Interval ($30.50\text{ s} < t \le 35.00\text{ s}$)**: **31 requests**
   - Generation speed progressively accelerated: TPOT ranged from $22.5\text{ ms}$ down to $17.1\text{ ms/token}$ (mean $18.7\text{ ms}$).
   - End-to-end latency ranged from $1.11\text{ s}$ to $1.47\text{ s}$ (mean $1.21\text{ s}$).
3. **Steady-State Interval ($t > 35.00\text{ s}$)**: **849 requests**
   - TPOT averaged $16.31\text{ ms/token}$ (P50: $16.17\text{ ms}$, P99: $18.59\text{ ms}$).
   - E2E latency returned to steady-state baseline: P50 $= 1.048\text{ s}$, P99 $= 1.201\text{ s}$—matching Runs 1 and 3.
- **Sum**: $189 + 31 + 849 = \mathbf{1,069\text{ requests}}$ (100% reconciled).

### Causal Assessment: Early-Run Backend Slowdown of Unresolved Cause
- **Client Dispatch Ruled Out**: Submission lag during the entire early interval was strictly $0.90\text{--}1.20\text{ ms}$, ruling out client-side load generator delay.
- **Unconfirmed Backend Mechanism**: Without synchronous hardware telemetry (sampling GPU core clocks, power states via NVML) or engine internals profiling during the trial, attributing the slowdown specifically to GPU clock ramp-up (P-state transition) or hypervisor scheduling remains a working hypothesis. It is formally classified as an **early-run backend slowdown of unresolved cause**.

---

## 5. Conclusions & Next Steps

1. **Harness Repeatability Verified**: Direct vLLM serving under identical Poisson schedules achieves deterministic throughput ($5.94\text{ RPS}$) and stable steady-state latency ($1.03\text{ s}$ P50, $1.17\text{ s}$ P99, $16.3\text{ ms}$ TPOT).
2. **100% Target Attainment**: Under 6 RPS, standalone vLLM easily satisfies $2.50\text{ s}$ E2E and $250\text{ ms}$ TTFT across all 3,207 requests, despite the transient early slowdown in Run 2.
3. **Protocol for Next Phase**:
   - **Active Warm-up**: Enforce a standardized active warm-up request burst before every measured trial (and after any idle cooldown) outside the evaluation window.
   - **Load Recalibration**: Treat previously proposed load labels (e.g. 10 RPS as "near capacity", 14–16 RPS as "overload") as uncalibrated candidates. Execute a corrected calibration sweep at candidate rates (e.g. 6, 10, 14, 18 RPS) with fixed windows to locate the true empirical saturation boundary.
   - **Continuous Telemetry**: Continuously log GPU clocks, power, and utilization during runs to correlate any future backend latency anomalies with physical silicon behavior.

