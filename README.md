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

### Comparative Serving Evaluation (Open-Loop Poisson Overload)

Under an open-loop Poisson arrival workload exceeding the backend's concurrency ceiling, the gateway shapes traffic by shedding excess queue delay before it cascades into timeouts:

```text
Mode                             Throughput    TTFT P50    TTFT P99    TPOT P99    E2E P99    429 Shed %    SLO (2s)
--------------------------------------------------------------------------------------------------------------------
Mode 1: Direct vLLM (Baseline)   48.2 tok/s     42.1 ms    812.4 ms     14.8 ms     3.85 s         0.0%       62.5%
Mode 2: Forwarding Only          47.9 tok/s     43.2 ms    824.1 ms     15.0 ms     3.91 s         0.0%       61.0%
Mode 3: Static Concurrency Cap   38.4 tok/s     38.5 ms    112.0 ms     12.1 ms     1.42 s        35.0%       98.0%
Mode 4: Bounded Queue + Deadline 44.1 tok/s     39.8 ms    284.5 ms     13.2 ms     1.86 s        15.0%       95.0%
```

**What this demonstrates:** how admission policies trade accepted work against latency under a specified workload.

**Boundary:** admission control does not itself guarantee an end-to-end latency SLA or improve the backend’s token-generation speed.

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

Results below apply to the stated configurations and workloads.

| Experiment | Scope | Observation | Interpretation |
|---|---|---|---|
| **SFT batching** | Qwen2.5-1.5B, answer-only LoRA, 1,000 GSM8K examples, RTX 3090, five paired runs | 25.76% ± 0.45% fewer padded tokens; 29.28% ± 0.65% higher useful-input-token throughput | Measured performance benefit for this setup; no established peak-memory saving |
| **DPO primary** | Qwen2.5-0.5B LoRA, 976 intact training pairs, 191 validation pairs | 120/191 positive implicit reward margins; mean DPO loss 0.6867 | Completed training under the corrected preprocessing contract |
| **DPO sensitivity** | Added 16 prompt-truncated pairs and two optimizer updates | 116/191 positive margins; mean loss 0.6832 | Mixed results; does not establish that filtering improves training |
| **GRPO pilot** | Qwen2.5-0.5B LoRA, RTX 4090, ten steps, 20 held-out questions | Correct-answer count changed from 6 to 8 (+2 correct answers, +10 percentage points); format compliance remained 0/20 | Exploratory observation, not a general reasoning-quality claim |

- **SFT “±” values** denote sample standard deviations across five paired runs.
- **DPO implicit reward accuracy** measures relative policy/reference likelihood changes. It is not open-ended answer accuracy.
- **The GRPO evaluation** uses a 20-example held-out subset of the local GSM8K training-data sample (rows 0..19), not the official GSM8K test benchmark. Zero format compliance means the intended output-format behavior was not demonstrated by the greedy policy.

Detailed experimental reports and raw metrics:
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
