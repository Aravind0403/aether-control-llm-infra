# Aether Control

**Validate training inputs. Control inference admission. Measure what actually happens.**

Aether Control is an experimental ML systems platform spanning three parts of the LLM lifecycle:

- **Inference serving:** a streaming admission gateway that manages concurrency, queueing, and overload.
- **TrainSight:** preflight inspection of training data, tokenization, and loss boundaries, reconciled against the inputs consumed by the trainer.
- **GRPO training:** an instrumented reinforcement-learning pipeline with numeric reward verification, rollout telemetry, and checkpoint evaluation.

The project focuses on reproducible engineering evidence: explicit input contracts, real GPU execution, comparative experiments, and clearly scoped claims.

**Status:** Research and engineering prototype. Core workflows have been exercised locally and on rented GPUs. Production reliability and customer adoption remain future milestones.

---

## Architecture

```text
                         Aether Control
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
    Inference Serving       TrainSight        GRPO Training
           │                   │                   │
    Streaming gateway     Dataset inspection   Model rollouts
    Admission decisions   Tokenization checks  Reward verification
    Bounded queue         Loss-mask checks     Group advantages
    Cancellation cleanup  Trainer alignment    LoRA updates
           │                   │                   │
       vLLM backend       SFT / DPO trainers   Saved checkpoints
           │                   │                   │
           └───────────────────┼───────────────────┘
                               │
                  Reports and reproducible artifacts
```

These are separate workflows within the repository. Their shared engineering approach is to make assumptions explicit and verify behavior at the execution boundary.

---

## 1. Inference Serving

The admission gateway sits between clients and an OpenAI-compatible inference backend.

It provides:

- Streaming response forwarding.
- Active-request concurrency limits.
- Bounded queueing and deadline-based rejection.
- HTTP 429 responses when admission limits are exceeded.
- Client-disconnect handling and resource cleanup.

The benchmark harness compares direct backend access, forwarding-only operation, a static concurrency cap, and bounded queue admission.

### Comparative Serving Evaluation (Real NVIDIA RTX 4090 Silicon)

The benchmark harness executes open-loop Poisson arrival traces replayed across four serving architectures on physical silicon (standalone `vllm:v0.7.2`, `Qwen/Qwen2.5-1.5B-Instruct`, eager mode, bfloat16, $\text{TTFT} \le 250\text{ ms}$, $\text{E2E} \le 2.50\text{ s}$ SLO targets):

```text
Serving Architecture                 Light Load (10 RPS)      Knee (35 RPS)          Overload (48 RPS)        Burst (10->48 RPS)
                                     Attain%    E2E P99       Attain%    E2E P99     Attain%    E2E P99       Attain%    E2E P99
---------------------------------------------------------------------------------------------------------------------------------
Mode 1: Direct vLLM (Baseline)       100.0%     1.08 s         99.9%     2.10 s       96.1%     2.57 s*       100.0%     1.74 s
Mode 2: Forwarding Only (Proxy)      100.0%     1.12 s        100.0%     1.98 s       96.9%     2.57 s*        99.7%     1.70 s
Mode 3: Static Concurrency Cap (36)  100.0%     1.09 s         83.4%     1.53 s       64.7%     1.50 s         79.8%     1.34 s
Mode 4: Bounded Queue + Deadline     100.0%     1.13 s         82.6%     1.64 s       52.9%     1.65 s         75.2%     1.43 s
```
*\*Values marked with an asterisk breach the 2.50 s E2E SLO target.*

**Key Empirical Findings from 48 Paired Runs (>59,000 requests):**
- **Proxy Overhead**: At light load, Aether Gateway forwarding adds $+3.2\text{ ms}$ TTFT P99 and $+40\text{ ms}$ E2E P99 overhead with zero rejections.
- **Burst Smoothing vs. Premature Shedding**: In near-capacity traffic (35 RPS), Mode 4 absorbs micro-bursts into the queue, reducing shedding by $55\%$ compared to static cap ($115.3$ vs $255.0$ shed requests) while completing $+140$ additional requests.
- **Overload SLO Protection**: Under sustained 48 RPS overload, direct unconstrained serving breaches the 2.50 s SLO boundary ($\text{E2E P99} = 2.57\text{ s}$). Mode 3 (Static Cap) strictly bounds admitted request latency ($\text{E2E P99} = 1.50\text{ s}$) via clean HTTP 429 shedding.
- **Queue Residency Tradeoff**: Queue wait time directly consumes the client-perceived TTFT budget. Under sustained overload, queued requests wait up to the deadline, causing TTFT P99 to reach $602.2\text{ ms}$ ($> 250\text{ ms}$ SLO).

**Boundary:** Admission control shapes and protects admitted request latency by shedding excess load; it does not accelerate GPU prefill or decode computation.


---

## 2. TrainSight

TrainSight checks training inputs before GPU execution and verifies that inspection matches the training pipeline.

### SFT inspection

- Tokenizer-measured sequence lengths.
- Answer-only loss masking.
- Prompt and padding exclusion from the loss.
- Empty-target, invalid-row, and truncation checks.
- Fail-closed behavior when tokenization is unavailable.
- Strict CLI quality gates.

```text
╭──────────────────────────── trainsight validator ────────────────────────────╮
│ 🔍 Profiling dataset: real_gsm8k_1000.jsonl                                  │
│ Type: SFT | Max Target Length: 1024 tokens                                   │
╰──────────────────────────────────────────────────────────────────────────────╯
                         📊 SFT Dataset Metrics Summary                         
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric                         ┃ Value              ┃ Status                 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Total Samples                  │ 1,000              │ ✅ OK                  │
│ Valid Training Samples         │ 1,000              │ ✅ OK                  │
│ Invalid Rows                   │ 0                  │ ✅ Clean               │
│ Empty Assistant Targets        │ 0                  │ ✅ Clean               │
│ Truncated Answers              │ 0                  │ ✅ Full Context        │
│ Loss Mask Misalignments        │ 0                  │ ✅ Answer-Only Masked  │
│ Avg Sequence Length (μ)        │ 244.6 tokens       │ ℹ️ Info                │
│ Avg Prompt / Target Length     │ 64.2 / 180.4 tok   │ ℹ️ Answer-Only Split   │
│ P95 Sequence Length            │ 382.0 tokens       │ ℹ️ Info                │
│ Max Sequence Length            │ 498 tokens         │ ✅ Within 1024 Limit   │
│ Sequential Batch Padding Waste │ 24.0% (B=4)        │ ✅ Length-Group Ready  │
│ Pre-Flight Quality Gate        │ Clean Execution    │ ✅ Passed              │
└────────────────────────────────┴────────────────────┴────────────────────────┘
✨ SFT Dataset passed all validation and runtime checks!
```

### Batching analysis

TrainSight compares actual padded-token totals under random and length-grouped batching. Offline padding estimates are reported separately from measured GPU throughput and memory usage.

### DPO inspection

- Shared-prompt and preference-pair validation.
- Identical-response, score-tie, and budget checks.
- Explicit prompt and completion truncation policies.
- Token and mask reconciliation against actual trainer batches.
- Per-pair evaluation of policy/reference log-probabilities and reward margins.

**What this demonstrates:** inspected data can be carried through a verified training-input contract.

**Boundary:** valid inputs do not guarantee factual answers, improved model quality, or freedom from runtime memory failures.

---

## 3. GRPO Training

The GRPO workflow connects real model generation to reward computation and LoRA training.

It includes:

- Numeric answer checking for a defined GSM8K-compatible contract.
- Separate format and correctness rewards.
- Group-relative advantage computation through TRL.
- Reward, gradient, and adapter-update telemetry.
- Checkpoint saving, reloading, and generation.
- Before/after evaluation with saved per-example outputs.

Simulation is explicitly separated from training. Training failures do not silently become synthetic success reports.

**What this demonstrates:** a working reinforcement-learning execution path.

**Boundary:** the small pilot does not establish a general improvement in mathematical reasoning.

---

## Experimental Evidence

Results below apply to the stated configurations and workloads on physical silicon.

| Experiment | Scope | Observation | Interpretation |
|---|---|---|---|
| **Serving 4-mode paired** | Qwen2.5-1.5B, vLLM 0.7.2, RTX 4090, 48 paired runs (>59,000 reqs) across 4 regimes | Gateway proxy overhead = +3.2 ms TTFT P99, +40 ms E2E P99 at 10 RPS. Under 48 RPS overload, direct vLLM breaches 2.5s SLO (2.57s E2E P99), while static cap strictly bounds E2E to 1.50s. Bounded queue cuts near-capacity shedding by 55% (+140 completions). | Quantified tradeoff: unconstrained serving cascades under overload; static cap defends latency via shedding; bounded queue smooths micro-bursts. |
| **Serving calibration** | Qwen2.5-1.5B, direct vLLM, RTX 4090, 11 offered rates (6–50 RPS), fixed 64-tok outputs | 100% attainment up to 35 RPS (2.17s E2E P99). First reproducible SLO failure at 40 RPS (94.6% attainment, 2.77s E2E P99). Saturation at 45–50 RPS (85.2% attainment). | Established true empirical capacity knee (35 RPS) and overload boundary (40+ RPS) on physical silicon. |
| **Serving repeatability** | Qwen2.5-1.5B, direct vLLM, RTX 4090, 3 consecutive 180s trials at 6 RPS (3,207 reqs) | 100% attainment across all 3,207 requests; 0 failures; lag P99 = 2.01 ms; Run 2 early slowdown (t ≤ 30.5s) fully reconciled. | Harness validated for deterministic trace replay and low submission lag. |
| **SFT batching** | Qwen2.5-1.5B, answer-only LoRA, 1,000 GSM8K examples, RTX 3090, five paired runs | 25.76% ± 0.45% fewer padded tokens; 29.28% ± 0.65% higher useful-input-token throughput | Measured performance benefit for this setup; no established peak-memory saving |
| **DPO primary** | Qwen2.5-0.5B LoRA, 976 intact training pairs, 191 validation pairs | 120/191 positive implicit reward margins; mean DPO loss 0.6867 | Completed training under the corrected preprocessing contract |
| **DPO sensitivity** | Added 16 prompt-truncated pairs and two optimizer updates | 116/191 positive margins; mean loss 0.6832 | Mixed results; does not establish that filtering improves training |
| **GRPO pilot** | Qwen2.5-0.5B LoRA, RTX 4090, ten steps, 20 held-out questions | Correct-answer count changed from 6 to 8 (+2 correct answers, +10 percentage points); format compliance remained 0/20 | Exploratory observation, not a general reasoning-quality claim |

- **Serving “±” values** denote sample standard deviations across independent seeded traces replayed across all 4 modes.
- **SFT “±” values** denote sample standard deviations across five paired runs.
- **DPO implicit reward accuracy** measures relative policy/reference likelihood changes. It is not open-ended answer accuracy.
- **The GRPO evaluation** uses a 20-example held-out subset of the local GSM8K training-data sample (rows 0..19), not the official GSM8K test benchmark. Zero format compliance means the intended output-format behavior was not demonstrated by the greedy policy.

Detailed experimental reports and raw metrics:
- [Four-Mode Paired Serving Benchmark Report](benchmarks/reports/four_mode_paired_experiment_report.md) · [Raw JSON Artifact](benchmarks/reports/paired_regimes_experiment_report.json)
- [Serving Capacity Recalibration Report](benchmarks/reports/capacity_calibration_reproducible.json) · [Initial Sweep](benchmarks/reports/capacity_calibration.json)
- [Repeatability Diagnostic Report](benchmarks/reports/repeatability_diagnostic_report.md) · [Raw JSON](benchmarks/reports/repeatability_diagnostic_report.json)
- [Real vLLM Integration Smoke Report](benchmarks/reports/real_vllm_integration_smoke_report.md)
- [SFT Batching Report](docs/reports/sft_batching_report.md) · [Raw JSON](benchmarks/reports/gpu_paired_training_answer_only_report.json)
- [DPO Primary & Sensitivity Report](docs/reports/dpo_primary_and_sensitivity_gpu_report.md) · [Per-Pair Log-Prob Reconciliation](docs/reports/dpo_per_pair_reconciliation_report.md)
- [GRPO GPU Smoke & Pilot Report](docs/reports/grpo_gpu_smoke_and_pilot_report.md) · [Pilot Raw JSON](rlhf-pipeline/reports/grpo_pilot_1791891388.json)


---

## Installation & Quickstarts

### Installation

```bash
# Clone repository
git clone https://github.com/Aravind0403/aether-control-llm-infra.git aether-control
cd aether-control

# Set up isolated Python environment
python3 -m venv .venv
source .venv/bin/activate

# Install editable packages
pip install --upgrade pip
pip install -e trainsight/
pip install -e vllm-engine/
pip install -e rlhf-pipeline/
```

### Quickstart 1: Inference Serving (Admission Gateway & Benchmark)

```bash
# Terminal 1: Launch local streaming mock backend
vllm-bench mock-server --port 8000

# Terminal 2: Launch Aether Admission Gateway in bounded-queue mode
vllm-bench gateway --port 8001 --backend http://localhost:8000 --mode bounded_queue --concurrency 4 --max-queue 16 --deadline-ms 400

# Terminal 3: Run 4-mode comparative SLA benchmark
vllm-bench compare --backend http://localhost:8000 --num-requests 30 --rps 10.0
```

### Quickstart 2: TrainSight (Data Preflight & Tokenization Gate)

```bash
# Profile SFT data with exact Qwen BPE tokenization and answer-only loss masking
trainsight inspect --dataset trainsight/sample_data/sample_gsm8k.jsonl --type sft --tokenizer Qwen/Qwen2.5-0.5B-Instruct

# Profile DPO preference pairs with boundary and budget validation
trainsight inspect --dataset trainsight/sample_data/sample_ultrafeedback.jsonl --type dpo --tokenizer Qwen/Qwen2.5-0.5B-Instruct
```

### Quickstart 3: GRPO Training (Verifier Tests & Smoke Execution)

```bash
# Run unit test suite for GSM8K answer contract and runtime caps
pytest rlhf-pipeline/tests/test_gsm8k_contract.py

# Run local/GPU 2-step GRPO smoke test with live rollout verification
python benchmarks/run_grpo_experiment.py --smoke --steps 2 --train-samples 4 --skip-train 20 --eval-samples 4 --skip-eval 0
```

---

## Reproducibility

Each experiment in this repository is accompanied by:

- Dataset identity, revision, selected rows, and split definitions.
- Model and tokenizer revisions.
- Dependency versions and hardware configuration.
- Training, generation, and preprocessing settings.
- Initialization and sampling seeds.
- Raw measurements and per-example evaluation outputs.
- Checkpoint locations and instructions for reproducing the run.

Mock-server tests, real GPU measurements, and exploratory model-quality results are labeled separately. Historical results produced by superseded calculations or preprocessing paths are retained only as explicitly marked legacy artifacts.

---

## What This Project Demonstrates

- Asynchronous streaming and request lifecycle management.
- Overload control and workload-driven benchmarking.
- Tokenization, collation, and loss-mask correctness.
- SFT, DPO, and GRPO training integration.
- Paired experiments and per-example metric reconciliation.
- Investigation of discrepancies between reported metrics and actual execution.

---

## Current Limitations

- No production deployment or sustained reliability claim.
- No demonstrated distributed KV-cache or prefill/decode orchestration layer.
- No established general benefit from DPO filtering.
- No established general reasoning improvement from the GRPO pilot.
- Limited model, dataset, and hardware coverage.
- Final GRPO audit items remain open until the parameter-distance calculation, answer extraction, and oversized-input behavior are reconciled.

---

## Next Milestones

1. Close the remaining audit items and publish reproducible experiment artifacts.
2. Provide one documented demonstration for each pillar.
3. Evaluate a focused workflow with an external engineering team.
4. Expand experiments where a specific adoption or performance claim requires more evidence.
