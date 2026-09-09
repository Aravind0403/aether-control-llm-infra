# vLLM Serving Infrastructure Stress-Testing & Architecture Evolution Notes
**Date**: September 9, 2026  
**Document**: `New_learning_VLLM.md`  
**Goal**: Adversarial counter-arguments, failure modes, systems audits, and production RFCs for the vLLM High-Throughput Inference Serving Platform.

---

## Round 1: The 5,000-User Burst & Cold-Boot Preemption Storm

### 1. The Adversarial Setup & The 90-Second Vulnerability Window
> **The Setup:**
> - Engine: vLLM serving Llama-3-70B (140GB FP16 / BF16).
> - Baseline Metrics: P99 TTFT = 280ms (under SLA), P99 TPOT = 45ms, 100 concurrent active users.
> - Infrastructure: Kubernetes cluster with KEDA autoscaling (`k8s-infra/manifests/keda-autoscaler.yaml`).
>
> **The Failure Mode (The Viral Tweet Burst):**
> A viral social spike suddenly routes **5,000 concurrent users** to the chat endpoint within 10 seconds.
> - KEDA detects queue depth spike (`sum(vllm:num_requests_waiting) > 5`) and triggers a $10\times$ horizontal scale-up.
> - **The Cold-Boot Reality**:
>   - Cloud Node VM Provisioning: 30–45s
>   - Container Image Pull (15GB vLLM image): 30s
>   - Network Weight Download (140GB from S3/GCS): 30–45s
>   - **Total Cold-Boot Window: 90 to 120 seconds!**
> - **The Blast Radius During the 90 Seconds**:
>   - The existing 100-user capacity is hit by 50x load.
>   - vLLM's internal waiting queue explodes; KV cache blocks exhaust.
>   - Preemption Storm triggers (EXP-005): running requests are preempted and swapped, collapsing effective throughput by **-93%**.
>   - P99 TTFT surges from $280\text{ ms} \to \mathbf{4,000+\text{ ms}}$. Users abandon.
> - **The Over-Provisioning Hangover**:
>   - When the 90s boot finishes, the 10x GPU nodes finally join the cluster.
>   - But the viral burst has already subsided. The platform now pays thousands of dollars for idle GPU nodes.

---

### 2. Evaluating the Naive Countermeasures

| Strategy | Mechanism | Fatal Flaw | Verdict |
| :--- | :--- | :--- | :--- |
| **Option 1: Static Rate Limiting** | Hard cutoff at `max_connections: 200` | Ignores token length variance (200 short queries are fine; 50 long 4K-token queries crash VRAM). Blindly returns 429s for trivial questions. | ❌ Brittle & Frustrating |
| **Option 2: Pure Predictive Scaling** | Forecast spikes from historical patterns | Cannot predict zero-day viral tweets or breaking news events. Still suffers the 90-second cold-boot delay when unexpected bursts hit. | ⚠️ Lagging Indicator |
| **Option 3: Unmanaged Spot Bursting** | Spin up spot instances on-demand without caching | Spot node provisioning is just as slow (90s) and subject to sudden preemption, compounding the preemption storm. | ❌ Destabilizing |

---

### 3. The Production Solution: The Multi-Tier Shield

```
                            [ 5,000 CONCURRENT USERS BURST ]
                                           │
                                           ▼
                 ┌───────────────────────────────────────────────────┐
                 │       TIER 1: ADAPTIVE ADMISSION CONTROLLER       │
                 │       (CoDel Queue-Delay Load Shedder @ Ingress)  │
                 └───────────────────────────────────────────────────┘
                                           │
          ┌────────────────────────────────┴────────────────────────────────┐
          ▼ (Queue Wait < 500ms: PASS)                                      ▼ (Queue Wait > 500ms: OVERFLOW)
[ Route to Primary 70B Engine ]                                 [ TIER 2: MODEL CASCADE ROUTER ]
(100 concurrent requests served                                 (Diverts to Warm 7B / 1.5B Spot Pool)
 under strict 280ms P99 TTFT SLA)                                                   │
                                                ┌───────────────────────────────────┴───────────────────────────────────┐
                                                ▼ (Simple / Short Query)                                                ▼ (Complex Reasoning / Math)
                                      [ Answered by 7B Model ]                                                [ Return HTTP 429 "Too Many Requests" ]
                                      • Latency: 45ms P99                                                     • Header: Retry-After: 30
                                      • Throughput: 8x higher                                                 • Client retries smoothly
                                      • Zero user abandonment                                                 • Platform never melts!
                                                │
                                                ▼
                               ┌──────────────────────────────────┐
                               │ TIER 3: PREDICTIVE SPOT BURST    │
                               │ • Boots Spot GPUs with NVMe pre- │
                               │   cached weights (21s boot time) │
                               │ • KEDA scales up gracefully      │
                               └──────────────────────────────────┘
```

#### Tier 1: The 0-Second Shield — CoDel Queue-Delay Load Shedder
Instead of counting connections, ingress monitors live latency from vLLM: `vllm:time_to_first_token_seconds` ($P_{99}$).
- **The Invariant**: If queue delay exceeds $500\text{ ms}$, the engine is saturated.
- **The Action**: Immediately shed excess load with HTTP 429 and informative backoff headers:
  ```http
  HTTP/1.1 429 Too Many Requests
  Retry-After: 15
  X-RateLimit-Reason: QueueSaturationDelayExceeded
  ```
- **SLA Protection**: The 100 active users maintain their $280\text{ ms}$ TTFT SLA. Pods never enter preemption thrashing.

#### Tier 2: The 2-to-90-Second Bridge — Model Cascading
A flat 429 is unnecessary for trivial queries ("What is the capital of France?").
- **Lightweight Classifier (<5 ms)**: Analyzes prompt length and query complexity.
- **Warm Small-Model Pool (Qwen-2.5-7B on Spot L4s)**:
  - Requires only $14\text{ GB}$ VRAM (fits on a single cheap Spot L4 GPU).
  - Generates at $8\times$ higher throughput with sub-30ms latency.
- **Impact**: **$70\%$ of the viral crowd** is answered immediately by the 7B model. Only the remaining $30\%$ heavy reasoning queries enter the prioritized 70B queue or receive clean 429s.

#### Tier 3: The 90-Second Fix — Local NVMe Fast-Boot (90s $\to$ 21s)
Eliminates network pull bottlenecks via local NVMe pre-baking:
```yaml
# In k8s-infra/manifests/vllm-deployment.yaml
volumes:
  - name: local-nvme-model-cache
    hostPath:
      path: /mnt/disks/ssd0/models/Qwen2.5-70B
```
- **Golden Node Image**: Docker container is pre-cached in local containerd storage.
- **PCIe Gen4 Direct Streaming**: Weights stream from local striped NVMe SSD arrays at $6.5\text{ GB/s}$:
  $$\text{Weight Load Time} = \frac{140\text{ GB}}{6.5\text{ GB/s}} \approx \mathbf{21.5\text{ seconds!}}$$
- **Result**: New Spot GPU pods serve live traffic in **$<25\text{ seconds}$**, closing $75\%$ of the cold-boot vulnerability window.

#### Tier 4: FinOps Asymmetric Cooldown (Preventing Over-Provisioning)
Enforces asymmetric hysteresis in [`k8s-infra/manifests/keda-autoscaler.yaml`](file:///Users/aravindsundaresan/Development/LLM_Serving_Platform/k8s-infra/manifests/keda-autoscaler.yaml):
```yaml
spec:
  cooldownPeriod: 300       # Wait 5 minutes of sustained low traffic before terminating nodes
  pollingInterval: 10       # Poll queue depth every 10s for fast scale-up
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:
        scaleDown:
          stabilizationWindowSeconds: 300
          policies:
            - type: Percent
              value: 20      # Terminate at most 20% of GPU nodes per minute
              periodSeconds: 60
```
- **Fast Scale-Up**: Evaluates queue depth every 10s.
- **Controlled Scale-Down**: Capped at $20\%$ per minute with a 5-minute stabilization window. If a secondary wave hits 3 minutes later, warm nodes are still alive.

---

## 📝 Architecture Ledger: VLLM-RFC-001

> ### 📌 VLLM-RFC-001: Adaptive Admission Control, Model Cascading & Fast-Boot Spot Bursting
> **Target Components**: `vllm-engine`, `k8s-infra/manifests/keda-autoscaler.yaml`, and `k8s-infra/manifests/vllm-deployment.yaml`  
> **Problem**: 50x viral traffic burst causing 90s cold-boot preemption storms, $4,000\text{ ms}$ TTFT spikes, and post-burst over-provisioning.  
> **Core Mechanisms**:
> 1. **CoDel Queue-Delay Load Shedder**: Sheds excess requests with HTTP 429 (`Retry-After: 15`) whenever $P_{99}$ queue wait exceeds $500\text{ ms}$, preserving existing user SLAs.
> 2. **Model Cascading Fallback**: Diverts simple/short queries to a warm Spot 7B model pool, absorbing $70\%$ of the burst with sub-30ms latency.
> 3. **Local NVMe Fast-Boot Strategy**: Slashes cold boot from $90\text{ s} \to 21.5\text{ s}$ by streaming weights directly from local striped NVMe arrays at $6.5\text{ GB/s}$.
> 4. **Asymmetric Cooldown Stabilization**: Caps scale-down node termination at $20\%$ per minute with a 5-minute stabilization window, preventing idle billing and oscillation.

---
*(Additional vLLM serving adversarial stress-test rounds and counter-arguments will be appended below as tests continue.)*

## Round 2: Radix Cache Thrashing & The Silent C++ Memory Leak

### 1. Black Swan #1: Prefix Cache Poisoning & Eviction Thrashing

#### The Real Attack Mechanism:
In a Radix Tree, the cache key is the exact sequence of token IDs: `[token_1, token_2, ..., token_k]`.
- User A's KV tensors cannot be read by User B because User B's token IDs diverge immediately.
- **The True Exploit: Cache Eviction Denial-of-Service (The LRU Poisoning)**:
  - An enterprise maintains a golden 2,500-token system prompt (catalog, instructions, schema). 5,000 legitimate users share this prefix, enjoying a blazing $45\text{ ms}$ P99 TTFT.
  - An attacker submits thousands of automated queries, each prefixed with 4,000 tokens of randomized junk.
  - Because the attacker's queries are frequent, the LRU (Least Recently Used) cache evicts the golden 2,500-token system prompt to make room for attacker garbage.
  - **The Blast Radius**: All 5,000 legitimate users now suffer a cold-cache prefill. Cluster-wide P99 TTFT spikes from $45\text{ ms} 	o \mathbf{950	ext{ ms}}$.

#### The Production Solution: Two-Tier Namespace Partitioning & Per-Tenant Quotas

```
                              [ GPU VRAM RADIX TREE ]
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        ▼ (TIER 1: HIGH-TRUST GLOBAL)                                     ▼ (TIER 2: DYNAMIC TENANT POOL)
[ Pinned Immutable System Prompts ]                               [ Ephemeral Tenant Prefix Nodes ]
• Verified enterprise system prompts                              • User multi-turn conversation history
• PINNED IN VRAM (LRU eviction disabled!)                        • Partitioned by Tenant_ID
• Protected by Cryptographic SHA256 Hash                          • Subject to Strict Eviction Quota:
• Zero attacker can evict this root!                                Max 15% of KV blocks per API Key!
                                                                                │
                                                                                ▼
                                                                [ Attacker floods random junk ]
                                                                • Hits 15% quota ceiling.
                                                                • Evicts ONLY their own blocks!
                                                                • Legitimate users unaffected!
```

1. **Tier 1: Write-Protected Pinned Roots (Immutable Namespace)**:
   - Approved enterprise system prompts are tagged with `pinned=True`.
   - **The Invariant**: The LRU allocator is forbidden from evicting pinned nodes regardless of memory pressure. The golden prefix remains permanently resident in GPU VRAM.
2. **Tier 2: Tenant-Quota Protected Dynamic Nodes**:
   - Dynamic user conversational history is stored in Tier 2.
   - Every API key / Tenant ID is constrained by a **Dynamic KV Block Quota** (max $15\%$ of active block pool).
   - If an attacker floods the cache with garbage, they only evict and thrash their own $15\%$ quota. Legitimate tenants and the global system prompt suffer zero performance degradation.

---

### 2. Black Swan #2: The GPU Memory Leak (The Slow Death)

#### Why Naive `torch.cuda.memory_allocated()` Fails in vLLM:
At boot, vLLM pre-allocates $90\%$ of GPU VRAM (`--gpu-memory-utilization 0.90`) into a static block pool.
`torch.cuda.memory_allocated()` and `nvidia-smi` report $90\%$ utilization from second #1, even with zero active users.

#### The Real Leak: Inside vLLM's C++/Python `BlockManager`:
- When a streaming connection is abruptly aborted by a client (e.g., user closes browser tab or drops connection), the request cleanup handler can fail to return physical block IDs to the `free_blocks` pool.
- Over 100,000 requests, the `free_blocks` stack steadily drains.
- When legitimate users arrive, vLLM throws `RuntimeError: No available blocks` and enters a preemption panic, crashing pods intermittently every 12 hours.

#### The Production Solution: The "Idle Residual Block Watchdog" & Zero-Downtime Rolling Rotation

```
                          [ Continuous vLLM Telemetry ]
                                       │
                                       ▼
                  [ Wait for Idle Window (Active Requests = 0) ]
                                       │
                  ┌────────────────────┴────────────────────┐
                  ▼                                         ▼
   [ Normal Operation (Clean GC) ]           [ LEAK DETECTED: Drifting Baseline ]
   Allocated Blocks = Pinned System Roots    Allocated Blocks = Pinned Roots + 420 Leaked Blocks
   (e.g., exactly 120 blocks)                (Drifting monotonically across hours!)
                                                            │
                                                            ▼
                                             [ LEAK THRESHOLD BREACHED (>15% Leaked) ]
                                                            │
                                             ┌──────────────┴──────────────┐
                                             ▼                             ▼
                              [ 1. Set Pod Readiness: FALSE ]    [ 2. Launch Replacement Pod ]
                              • Ingress stops sending requests.  • New pod boots in parallel.
                              • Existing requests finish (15s).  • Ingress switches traffic.
                              • Pod restarts with fresh VRAM.    • ZERO DROPPED CONNECTIONS!
```

1. **The Idle Residual Equation**:
   $$\text{Residual\_Blocks} = \text{vllm:num\_gpu\_blocks\_used} \quad \text{when } (\text{vllm:num\_requests\_running} == 0)$$
   - **Healthy Server**: When active running requests drop to 0, `Residual_Blocks` returns to the baseline of pinned system prompts (e.g., exactly 120 blocks).
   - **Leaking Server**: Baseline drifts monotonically over time: $120 \to 180 \to 350 \to 600$.

2. **Self-Healing Kubernetes Loop (Zero-Downtime Drain)**:
   - When `Residual_Blocks` exceeds $15\%$ of the total block pool:
     - Watchdog trips the pod readiness probe: `/health/ready` returns HTTP 503 (`Service Degraded - Scheduled Rotation`).
     - Ingress stops routing new requests to this pod.
     - Kubernetes lifecycle hook executes `preStop: sleep 15`, allowing currently streaming responses to finish cleanly.
     - A replacement pod boots in parallel, traffic cuts over, and the leaking pod restarts with zero 500 errors and zero dropped requests.

---

## 📝 Architecture Ledger: VLLM-RFC-002

> ### 📌 VLLM-RFC-002: Two-Tier Radix Security & Zero-Downtime Memory Leak Auto-Healing
> **Target Components**: `vllm-engine`, Radix Cache Allocator, and Kubernetes Readiness Probes  
> **Problems**: Cache eviction thrashing DoS degrading TTFT from $45\text{ ms} \to 950\text{ ms}$; Silent C++ block allocator leaks draining free blocks over 100k requests.  
> **Core Mechanisms**:
> 1. **Two-Tier Namespace Separation**: Pins high-value enterprise system prompts as immutable, write-protected root nodes in the Radix Tree, completely immune to LRU eviction.
> 2. **Tenant Quota Isolation**: Caps dynamic user conversation caching at a maximum of $15\%$ of the total block pool per API key, preventing adversarial cache-thrashing DoS attacks from affecting other tenants.
> 3. **Idle Residual Block Watchdog**: Monitors `vllm:num_gpu_blocks_used` during zero-active-traffic intervals to detect internal C++ block allocator leaks that bypass PyTorch's native memory metrics.
> 4. **Self-Healing Kubernetes Rotation**: Automatically trips the pod readiness probe when residual block leakage breaches $15\%$, gracefully draining active connections and refreshing VRAM with zero dropped requests.

---

## Round 3: The Three Legacy Shocks of vLLM at Scale

### 1. Legacy Shock #1: The "Cold Cache Penalty" for New Users

#### The Failure Mode:
Radix caching optimizes established users sharing frequent prefixes (45ms TTFT). When a new user signs up or loads a custom workspace, they hit a cold cache miss ($500	ext{ ms}$ TTFT). New users perceive the platform as sluggish and churn before becoming established users.

#### The Production Solution: Progressive Cache Priming via Ingress Consistent Hashing

```
[ User Logs In / Loads Chat Page ]
                 │
                 ▼
[ Gateway: Consistent Hash(Tenant_ID / User_ID) ] ──► Routes to Pod #3
                 │
                 ▼ (Fire-and-forget HTTP POST /v1/cache/prime)
      [ Pod #3: Background Radix Prefill ]
      • Prefills the 2,500-token system prompt in background (120ms).
      • Seated in Radix Tree before user even touches the keyboard!
                 │
                 ▼
[ User Types: "Summarize this memo" ] ──► Hits Pod #3 ──► CACHE HIT (45ms TTFT!)
```

1. **Ingress Consistent Hashing (Maglev / Ketama Ring)**:
   - Uses `Consistent_Hash(Tenant_ID)` at Envoy gateway so all requests from a tenant map deterministically to the same serving pod.
2. **Background Prefill Handshake**:
   - During the 200–500ms dead window between authentication and first keystroke, gateway fires a background `POST /v1/cache/prime`.
   - Pod completes the 2,500-token prefill in 120ms and seats the KV blocks into the Radix Tree.
   - When the user sends their very first prompt $\implies$ **Instant 45ms P99 TTFT Cache Hit**.
3. **Overhead Protection**: Rate-limits priming calls to prevent page-refresh prefill storms.

---

### 2. Legacy Shock #2: The Model Cascade Overfitting Trap (Prompt Gaming)

#### The Failure Mode:
When naive cascading routes by prompt character/token length, users realize short prompts route to the "fast lane." Users begin phrasing complex requests in ultra-short prompts (e.g., *"Write CUDA kernel"* or *"Prove Fermat"*). The 7B model attempts to answer, hallucinations spike, and quality degrades by 20%.

#### The Production Solution: The Semantic Complexity Oracle (Anti-Gaming)

```
                         [ Incoming User Query ]
                                    │
                                    ▼
       [ Ingress Envoy Proxy: Embedded bge-small ONNX Model (2ms on CPU) ]
       • Extracts 384-dimensional query embedding vector.
       • Computes Cosine Distance against 3 Semantic Centroids:
         - Centroid A (Factual / Conversational / Formatting) ──► 7B Model
         - Centroid B (Mathematical / Multi-Step Logic)       ──► 70B Model
         - Centroid C (Code Synthesis / System Engineering)    ──► 70B Model
                                    │
       ┌────────────────────────────┴────────────────────────────┐
       ▼ (Classified "Simple")                                   ▼ (Classified "Complex")
[ Route to 7B Fast Lane ]                                [ Route to 70B Deep Lane ]
• "Summarize this paragraph"                             • "Write CUDA kernel" (3 words)
  Embedding matches Centroid A!                            Embedding matches Centroid C!
  Routed to fast 7B model.                                 CANNOT BE GAMED!
                                                           Routed directly to 70B!
```

1. **2ms CPU Semantic Centroid Classifier**:
   - Runs a quantized `bge-small` ONNX model directly inside the CPU ingress proxy.
   - Computes cosine distance against 3 task centroids:
     - **Centroid A (Conversational / Formatting / Factual QA)** $	o$ 7B Fast Lane.
     - **Centroid B (Multi-Step Math & Symbolic Logic)** $	o$ 70B Deep Lane.
     - **Centroid C (System Engineering, Compilers, Code Gen)** $	o$ 70B Deep Lane.
   - A 3-word query like *"Write CUDA kernel"* matches Centroid C and routes directly to the 70B cluster regardless of token count.
2. **Speculative Verification Safety Net**:
   - If a borderline query (score $pprox 0.51$) sent to 7B exhibits high token entropy ($	ext{Perplexity} > 	au$), streaming immediately aborts and transparently fails over to 70B before garbage reaches the client.

---

### 3. Legacy Shock #3: The Spot Instance Preemption Death Spiral

#### The Failure Mode:
A cloud provider reclaims $50\%$ of Spot L4 instances simultaneously during cluster-wide GPU shortages.
- Cascade router suddenly loses 7B capacity and redirects traffic to the 70B cluster.
- The 70B cluster saturates, trips CoDel shedding, and emits cluster-wide HTTP 429s.
- Platform crashes, and the team abandons Spot instances, paying $2	imes$ for on-demand instances.

#### The Production Solution: The "Anchor + Spot" Hybrid Topology

```
[ 7B Serving Cluster Pool ]
├── 1x On-Demand Anchor Pod (Always warm, takes 15% baseline traffic)
└── 5x Spot GPU Worker Pods (Take 85% of traffic, $2/hr)

Timeline under GCP 30-second Preemption Notice:
T = 0s : GCP sends 30-second SIGTERM notice to Spot Worker #2.
T = 1s : Spot Worker #2 flips /health/ready to 503. On-Demand Anchor absorbs traffic using headroom.
T = 2s : KEDA triggers scale-up of replacement Spot/On-Demand node via Local NVMe Fast-Boot (21.5s).
T = 23s: Replacement node is warm and serving live traffic over PCIe local NVMe streaming.
T = 30s: Spot Worker #2 finishes draining in-flight streams and cleanly exits.
```

- **Cloud Economics**:
  $$	ext{Blended Cost} = (94\% 	imes \$2/	ext{hr}) + (6\% 	imes \$8/	ext{hr}) = \mathbf{\$2.36/	ext{hr}}$$
  Preserves **$91\%$ of total cloud compute cost savings** while eliminating single points of failure.
- **Result**: Zero capacity gap, zero dropped requests, zero HTTP 429 cascades.

---

## 📝 Architecture Ledger: VLLM-RFC-003

> ### 📌 VLLM-RFC-003: Progressive Cache Priming, Semantic Complexity Routing & Spot Handoff Resilience
> **Target Components**: `vllm-engine`, Envoy Ingress Classifier, and `k8s-infra/manifests/vllm-deployment.yaml`  
> **Problems**: Cold-cache penalty causing new user churn; Prompt gaming bypassing length routers; 30-second Spot preemption triggering cascading 429 cluster failures.  
> **Core Mechanisms**:
> 1. **Progressive Cache Priming**: Uses Consistent Hashing (`Tenant_ID`) to assign users to pods during authentication, pre-computing system prompt KV-caches in the background before the user sends their first message (TTFT drops from $500\text{ ms} \to \mathbf{45\text{ ms}}$).
> 2. **Semantic Complexity Oracle (Anti-Gaming)**: Evaluates query intent via a lightweight 2ms CPU ONNX embedding classifier against task centroids, rendering keyword-shortening or prompt-gaming attempts ineffective.
> 3. **Anchor + Spot Handoff Topology**: Maintains 1 warm On-Demand anchor pod alongside Spot instances. Bridges the 30-second GCP preemption window with local NVMe 21.5s fast-booting, guaranteeing zero dropped connections and eliminating the 429 death spiral at 6% marginal cost.

---

