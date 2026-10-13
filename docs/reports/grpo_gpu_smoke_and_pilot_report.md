# GRPO GPU Smoke Test & 10-Step Exploratory Pilot Report

**Environment**: Remote RTX 4090 (24 GB VRAM, Vast.ai Instance `55728940`, destroyed post-run)  
**Model**: `Qwen/Qwen2.5-0.5B-Instruct` (Bfloat16 base model, LoRA adapter $r=16, \alpha=32$)  
**Dataset**: GSM8K Mathematical Reasoning (`trainsight/sample_data/real_gsm8k_1000.jsonl` — 1,000-sample training slice)  
**Evaluation Cohort**: **20-example held-out subset of your GSM8K training-data sample** (rows 0..19), *not the official GSM8K test benchmark*.  
**Trainer**: HuggingFace TRL 0.24.0 `GRPOTrainer`  

---

## 1. Technical Corrections & Configuration Deviations

### 1.1 Deviations from Initial Proposal
The final executed configuration differs from the initial feasibility proposal in two specific structural parameters:
1. **Four-Module LoRA Configuration**:
   - *Initial proposal*: Two attention projection modules (`q_proj`, `v_proj`).
   - *Final executed setup*: Four attention projection modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`) with $r=16, \alpha=32$ across all 24 transformer layers (192 parameter tensors).
   - *Actual Trainable Parameters*: **2,162,688** out of 496,195,456 total base model parameters (**0.4359%**). (The earlier 2.16M parameter figure covered all four modules, not two).
2. **Completion Token Limit**:
   - *Initial proposal*: 256 tokens (`max_completion_length=256`).
   - *Final executed setup*: Raised to **512 tokens** (`max_completion_length=512`) after empirical validation revealed that multi-step mathematical rollouts frequently clipped before the final numerical answer at 256 tokens, suppressing reward variance.

### 1.2 `<think>` Prefill Boundary Reconciliation
- **Boundary Condition**: When the prompt includes `<|im_start|>assistant\n<think>\n`, generated completions contain only the continuation of the reasoning trace. Evaluating raw completions directly against strict XML schemas would cause valid reasoning traces lacking the opening `<think>` tag to fail format checks.
- **Contract Solution**: Both training reward functions (`grpo_format_reward`, `grpo_accuracy_reward`) and the evaluation engine invoke `reconstruct_assistant_completion(text)` in [`gsm8k_contract.py`](file:///Users/aravindsundaresan/Development/LLM_Serving_Platform/rlhf-pipeline/rlhf_pipeline/rewards/gsm8k_contract.py). If `<think>` is missing from the prefix, the full assistant turn is reconstructed before regex extraction:
  $$\text{Completion} \gets \text{"<think>\\n"} + \text{text}$$
- Evaluated consistently during both training rollouts and held-out evaluation.

### 1.3 Zero-Advantage Theoretical Clarification
- **Correction**: In GRPO, when all $G$ completions for a given prompt receive identical rewards ($r_1 = \dots = r_G$), the sample reward standard deviation $\sigma_G = 0$ (or with eps $\epsilon$, $A_i = \frac{r_i - \bar{r}}{\sigma_G} = 0$). This produces **zero reward-driven policy-gradient contribution for that specific group**.
- It does **not** imply zero parameter updates for the optimizer step if:
  1. Multiple groups are accumulated across the batch where other groups exhibit reward variance.
  2. Optimizer momentum / state carries historical updates.
  3. The KL divergence penalty is active ($\beta > 0$), which penalizes drift from the reference policy independently of reward signal.
- **Explicit GRPO Configuration Recorded**:
  - `loss_type`: `"grpo"`
  - `beta` (KL coefficient): `0.04`
  - `scale_rewards`: `"group"`
  - `mask_truncated_completions`: `True`

### 1.4 Verifier Runtime Bounds
To prevent Regex Denial of Service (ReDoS) and arbitrary-precision integer overflow, [`gsm8k_contract.py`](file:///Users/aravindsundaresan/Development/LLM_Serving_Platform/rlhf-pipeline/rlhf_pipeline/rewards/gsm8k_contract.py) enforces deterministic bounds:
- Completion text length: $\le 4,096$ characters (truncated before regex).
- Extracted answer candidate string: $\le 64$ characters.
- Numeric digit count: $\le 20$ digits.
- Number magnitude: $|x| \le 10^{15}$.

---

## 2. GPU Smoke Test Evidence (2 Steps)

The smoke test verified the full execution chain on real hardware: rollout generation, reward callbacks, group advantage normalization, gradient backpropagation, adapter weight updates, and checkpoint reloading.

| Metric | Measurement | Verification Status |
| :--- | :--- | :--- |
| **Hardware** | NVIDIA RTX 4090 (24 GB) | Verified |
| **Peak VRAM** | 2.05 GB | Verified |
| **Step 1 Telemetry** | Loss = 0.0, GradNorm = 0.0, RewardMean = 0.0, RewardStd = 0.0 | Identical 0 rewards $\implies A_i=0$ |
| **Step 2 Telemetry** | Loss = -1.49e-08, GradNorm = **0.7948**, RewardMean = 0.25, RewardStd = **0.5000** | Non-zero variance $\implies$ Finite gradient |
| **Initial Adapter Norm** | 221.669727 (across 192 tensors) | Verified |
| **Post Adapter Norm** | 221.684437 | Verified |
| **Total Adapter $\Delta W$** | **0.014710** (Max tensor delta: $0.00022271$ on `lora_B`) | $\Delta W > 0$ **Verified Updated** |
| **Checkpoint Reload** | Reloaded `PeftModel` from `/workspace/model_output/grpo_smoke` and generated text | **Passed** |

---

## 3. 10-Step Exploratory Pilot on 20-Example Held-Out Training Subset

> [!NOTE]
> This run is treated strictly as an **exploratory pilot** ($N=20$ evaluation questions; each question represents $5.0$ percentage points). It serves to validate multi-step training dynamics and transition reporting, not to assert a general reasoning claim.
> 
> The evaluation set is a **20-example held-out subset of your GSM8K training-data sample (rows 0..19), not the official GSM8K test benchmark**.

### 3.1 Setup & Cohort Isolation
- **Initialization**: Fresh base initialization (`Qwen/Qwen2.5-0.5B-Instruct` + fresh LoRA adapter; not continuing the smoke checkpoint).
- **Frozen Evaluation Cohort**: Exactly 20 examples (rows `0..19` of `real_gsm8k_1000.jsonl`), evaluated greedy at `max_new_tokens=512`.
- **Training Cohort**: Strictly isolated to rows `20..59` (zero overlap with evaluation).
- **Generation Parameters**: $G=4$ rollouts per prompt, temperature = 0.9, `max_completion_length=512`.

### 3.2 Evaluation Metrics (Before vs. After)

| Metric | Pre-Training Baseline (Step 0) | Post-Training (Step 10) | Delta |
| :--- | :---: | :---: | :---: |
| **Correct Answers** | **6 / 20** (30.0%) | **8 / 20** (40.0%) | **+2 correct answers, +10 percentage points** |
| **Format Compliance** | 0 / 20 (0.0%) | 0 / 20 (0.0%) | 0.0 percentage points |
| **Completion Limit Hits** | 0 / 20 (0.0%) | 0 / 20 (0.0%) | 0 |
| **Mean Total Reward** | 0.3000 | 0.4000 | +0.1000 |

> [!IMPORTANT]
> **Zero format compliance means the intended output-format behavior was not demonstrated.**  
> While the greedy evaluation policy produced numerically correct answers for 8 out of 20 questions via fallback regex extraction, it generated zero completions strictly adhering to the `<think>...</think><answer>...</answer>` XML structure. Although stochastic rollouts during exploration began sampling format tags (e.g. `rewards/grpo_format_reward/mean = 0.0625` at step 10), 10 training steps were insufficient to establish format compliance in the deterministic greedy decoding policy.

### 3.3 Training Step Telemetry
During training, within-group reward variance was observed in all logged steps:

| Step | Loss | Gradient Norm | Mean Reward | Reward Std ($\sigma_G$) | KL Divergence |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2** | $1.32 \times 10^{-5}$ | 0.00617 | 0.2500 | 0.2887 | 0.000330 |
| **4** | $2.32 \times 10^{-5}$ | **1.00568** | 0.1250 | 0.2500 | 0.000580 |
| **6** | $3.73 \times 10^{-5}$ | **0.77017** | 0.5000 | **0.5774** | 0.000932 |
| **8** | $2.70 \times 10^{-5}$ | 0.00615 | 0.3750 | 0.2500 | 0.000676 |
| **10** | $2.80 \times 10^{-5}$ | **0.73662** | 0.1875 | 0.3750 | 0.000700 |

- **Total Adapter Parameter Update**: $\Delta W = \mathbf{0.167291}$ across all 192 tensors (Max single tensor delta: $0.001660$).
- **Peak VRAM**: 2.17 GB.
- **Runtime**: 88.2 seconds (8.82 s/step, 0.113 steps/s).

---

## 4. Per-Example Output & Transition Breakdown

Across all 20 frozen evaluation questions:

| Transition Category | Count | Percentage |
| :--- | :---: | :---: |
| **Correct $\to$ Correct** (Retained) | 6 | 30.0% |
| **Incorrect $\to$ Correct** (Gained) | 2 | 10.0% |
| **Correct $\to$ Incorrect** (Lost) | 0 | 0.0% |
| **Incorrect $\to$ Incorrect** (Unchanged) | 12 | 60.0% |

### Transitions Details:
1. **Row 0** (Target: `72`): Retained Correct
2. **Row 1** (Target: `10`): Retained Correct
3. **Row 3** (Target: `42`): Retained Correct
4. **Row 4** (Target: `624`): **Incorrect $\to$ Correct**
   - *Pre-training*: Stopped incomplete before computing annual total.
   - *Post-training*: Multiplied $6 \text{ pages/week} \times 2 \text{ friends} = 12 \text{ pages/week}$, then $12 \times 52 = 624$ pages. Final answer: `624`.
5. **Row 6** (Target: `48`): Retained Correct
6. **Row 12** (Target: `85`): Retained Correct
7. **Row 13** (Target: `35`): **Incorrect $\to$ Correct**
   - *Pre-training*: Correct calculation ($10 + 5 + 20 = 35$) but halted abruptly without clean final answer synthesis.
   - *Post-training*: Produced full reasoning breakdown and concluded: "Therefore, Jasper spends $35 on the ingredients." Final answer extracted: `35`.
8. **Row 18** (Target: `16`): Retained Correct

---

## 5. Summary & Status

1. **Protocol Integrity**:
   - Evaluator and reward checker use the same `<think>` boundary reconstruction.
   - Evaluation cohort rows 0..19 strictly isolated from training rows 20..59.
   - Zero-advantage semantics, adapter parameter count (2,162,688 params on 4 modules), and completion limit (512 tokens) recorded accurately.
2. **Cloud Resource Teardown**:
   - Remote Vast.ai instance `55728940` was destroyed immediately after artifact download to conserve user balance (`0 instances active`).
3. **Artifacts Saved Locally**:
   - Smoke report: [`rlhf-pipeline/reports/grpo_smoke_1791891064.json`](file:///Users/aravindsundaresan/Development/LLM_Serving_Platform/rlhf-pipeline/reports/grpo_smoke_1791891064.json)
   - Pilot report: [`rlhf-pipeline/reports/grpo_pilot_1791891388.json`](file:///Users/aravindsundaresan/Development/LLM_Serving_Platform/rlhf-pipeline/reports/grpo_pilot_1791891388.json)
