# Real vLLM Integration Smoke Test Report

**Execution Timestamp**: 2026-10-15 12:41:43 UTC  
**Hardware Silicon**: NVIDIA GeForce RTX 4090, 590.48.01, 24564 MiB  
**Stack**: vLLM 0.7.2 | CUDA 13.1 | NVIDIA Driver 590.48.01  
**Git Commit**: `d1e95f234756700103d246b7269c016c582a279a`  
**Model**: `Qwen/Qwen2.5-1.5B-Instruct`  
**Vast.ai Instance**: `56094123` (ssh1.vast.ai:35206)

---

> [!NOTE]
> Latency figures reported below are smoke-test functional observations to confirm end-to-end completion, not a gateway overhead benchmark. Output token lengths varied across prompts, and the SSH tunnel between the local test client and the rented GPU cloud instance introduced network latency.

---

## 🔬 Test Suite Execution Results

### 1. Direct & Proxied Request Lifecycle
| Mode | Request Type | Status | Latency | Tokens | Terminal Usage & Completion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Direct vLLM** | Non-Streaming (Unary) | **PASS** | 821.6 ms | 2 | `usage` payload asserted (`prompt_tokens`, `completion_tokens`, `total_tokens`) |
| **Direct vLLM** | Streaming (SSE) | **PASS** | 585.2 ms | 10 | Terminal usage chunk verified via `stream_options.include_usage=True` |
| **Aether Gateway** | Non-Streaming (Unary) | **PASS** | 777.0 ms | 4 | `X-Aether-Admission: ADMIT` & `usage` fields asserted |
| **Aether Gateway** | Streaming (SSE) | **PASS** | 696.4 ms | Verified | Reassembled multi-chunk SSE deltas & terminal `data: [DONE]` verified |

*Gateway active and queued slots strictly returned to `0` after all requests completed.*

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
- **Active Stream Cancellation**: Client disconnected during active generation. Gateway aborted upstream HTTP request handling and released the admission slot immediately (`active_requests == 0`). *(Note: Gateway slot recovery establishes gateway cleanup; backend engine-level token abortion was not separately verified via backend logs/metrics in this smoke test).*
- **Queued Request Cancellation**: Client cancelled task while waiting in admission queue. Queue depth returned immediately from `1` to `0`.
- **Verdict**: **No slot leaks observed in the tested scenarios.**

---

### 4. Backend Failure & Resumption
- **Process Termination**: Simulated backend crash by issuing `SIGKILL` to remote vLLM process during active streaming.
- **Gateway Response**: Gateway cleanly terminated the in-flight stream without deadlocking and restored admission counters to `0`.
- **Backend Recovery**: Re-launched vLLM engine, confirmed `/health` returned HTTP 200, and verified subsequent requests succeeded cleanly (`'BACKEND ALIVE'`).
- **Verdict**: **Backend interruption and recovery after restart verified.**

---

### 5. Evidence & Environmental Persistence
- **Environment Provenance**: Verified exact Git commit SHA (`d1e95f2`), vLLM version (0.7.2), model repository (`Qwen/Qwen2.5-1.5B-Instruct`), driver (`590.48.01`), CUDA version (13.1), and physical GPU silicon (RTX 4090 24GB).
- **Teardown**: Vast.ai instance `56094123` verified destroyed post-test, preventing cloud resource/credit leaks.
- **Verdict**: **Evidence captured and environment cleanly decommissioned.**

---

## 🏁 Summary Verdict: ALL 5 STEPS PASSED
The Aether Admission Gateway request lifecycle, slot release guarantees, client cancellation handling, and SSE stream parsing have been verified live against real vLLM 0.7.2 on physical NVIDIA GeForce RTX 4090 hardware across all five integration checks.
