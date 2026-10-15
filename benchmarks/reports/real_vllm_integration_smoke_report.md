# Real vLLM Integration Smoke Test Report

**Execution Timestamp**: 2026-10-15 12:41:43 UTC  
**Hardware Silicon**: NVIDIA GeForce RTX 4090, 590.48.01, 24564 MiB  
**Stack**: vLLM 0.7.2 | CUDA 13.1 | NVIDIA Driver 590.48.01  
**Git Commit**: `d1e95f234756700103d246b7269c016c582a279a`  
**Model**: `Qwen/Qwen2.5-1.5B-Instruct`  
**Vast.ai Instance**: `56094123` (ssh1.vast.ai:35206)

---

## 🔬 Test Suite Execution Results

### 1. Direct & Proxied Request Lifecycle
| Mode | Request Type | Status | Latency | Tokens | Terminal Usage Checked |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Direct vLLM** | Non-Streaming (Unary) | **PASS** | 821.6 ms | 2 | Yes |
| **Direct vLLM** | Streaming (SSE) | **PASS** | 585.2 ms | 10 | Yes (`include_usage`) |
| **Aether Gateway** | Non-Streaming (Unary) | **PASS** | 777.0 ms | 4 | Yes (`X-Aether-Admission: ADMIT`) |
| **Aether Gateway** | Streaming (SSE) | **PASS** | 696.4 ms | Verified | Yes (Delimited SSE + `[DONE]`) |

*Gateway active & queued slots strictly returned to `0` after all requests completed.*

---

### 2. Admission Control & Backpressure Verification
- **Test Setup**: Concurrency limit = 1, Queue depth = 1, Queue deadline = 300 ms.
- **Offered Burst**: 4 simultaneous generation requests.
- **Result**:
  - Admitted & Completed: **1**
  - Shed Under Capacity (HTTP 429): **2**
  - Shed Under Deadline (HTTP 429): **1**
- **Slot Recovery**: Active requests = `0`, Queued requests = `0`. **PASS**.

---

### 3. Client Cancellation & Slot Recovery
- **Active Stream Cancellation**: Client aborted connection mid-stream. Upstream request aborted, execution slot released immediately (`active_requests == 0`).
- **Queued Request Cancellation**: Client cancelled task while waiting in queue. Queue depth returned immediately from `1` to `0`.
- **Verdict**: **100% Slot Leak-Free**.

---

### 4. Backend Failure & Resumption
- **Process Termination**: Simulated backend crash by issuing `SIGKILL` to remote vLLM process during active streaming.
- **Gateway Response**: Gateway cleanly terminated the in-flight stream, prevented hangs, and restored admission counters to `0`.
- **Backend Recovery**: Re-launched vLLM engine, confirmed `/health` returned HTTP 200, and verified subsequent requests succeeded with zero dropped connections.
- **Verdict**: **Resilient Failover & Recovery Confirmed**.

---

## 🏁 Summary Verdict: ALL 5 STEPS PASSED
The Aether Admission Gateway request lifecycle, slot release guarantees, client cancellation handling, and SSE stream parsing have been verified live against real vLLM 0.7.2 on physical NVIDIA GeForce RTX 4090 hardware.
