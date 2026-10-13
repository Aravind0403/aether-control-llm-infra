# TrainSight DPO GPU Primary vs. Sensitivity Run Report

**Date**: October 13, 2026  
**Hardware**: NVIDIA GeForce RTX 4090 (24 GB VRAM)  
**Model**: `Qwen/Qwen2.5-0.5B-Instruct` (LoRA $r=16, \alpha=32$ on $q, k, v, o$, 2,162,688 trainable parameters)  
**Precision**: `torch.bfloat16`  
**Hyperparameters**: $\beta = 0.1$, $\text{lr} = 5 \times 10^{-5}$, per-device batch size 2, gradient accumulation 4 (effective batch size 8)  
**Held-Out Evaluation Cohort**: Exactly 191 frozen intact pairs (`eval_ultrafeedback_test_191.jsonl`)

---

## 1. Executive Summary & Core Results

| Run Identifier | Training Cohort | Updates (Steps) | Peak VRAM | Population Wins ($W / 191$) | Pair-Weighted Accuracy | Mean Reward Margin | Mean DPO Loss | TRL Batch-Mean Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fresh Init Baseline** | — | 0 | — | 0 / 191 | 0.00%* | 0.0000 | 0.6931 | 0.0000 |
| **Primary Run (976)** | 976 intact pairs | 122 | 13.44 GB | **120 / 191** | **62.83%** | **+0.1400** | **0.6867** | 62.50% (60/96) |
| **Sensitivity Run (992)** | 976 intact + 16 prompt-clipped | 124 | 13.43 GB | **116 / 191** | **60.73%** | **+0.1512** | **0.6832** | 60.42% (58/96) |
| **Delta ($\Delta_{992 - 976}$)** | +16 prompt-truncated pairs | +2 steps | -0.01 GB | **-4 wins** | **-2.10%** | **+0.0112** | **-0.0035** | -2.08% |

*\* Note: Fresh initialization sets policy equal to reference ($\pi = \pi_{\text{ref}}$), where strict win inequality $\Delta r > 0$ yields 0 wins and DPO loss $- \log \sigma(0) = \ln(2) \approx 0.693147$.*

---

## 2. Review of the Results (GPU Primary Run: 976 Pairs)

### Training Execution & Convergence
- **Update Steps**: 122 optimization steps over 1 epoch ($976 / 8 = 122$).
- **Runtime**: 163.7 seconds (~0.75 steps/s on RTX 4090).
- **Final Training Loss**: 0.6606.
- **Pre-flight Assertion**: Token IDs, attention masks, and completion masks from `get_eval_dataloader()` bitwise matched the approved tokens from `DPOPreprocessor` (0 mismatches across all 191 pairs).

### Held-Out Evaluation Metrics
- **Exact Pair-Weighted Accuracy**: **120 / 191 (62.83%)**.
  - Positive margins ($\Delta r > 0$): 120 pairs.
  - Zero margins ($\Delta r = 0$): 0 pairs.
  - Negative margins ($\Delta r < 0$): 71 pairs.
- **Mean Reward Margin**: **+0.1400** (median +0.1792, min -1.8816, max +2.8538, stdev 0.7310).
- **Mean Held-Out DPO Loss**: **0.6867** (decreased from the initial reference baseline of 0.6931).

### Resolution of TRL Batch Averaging vs. Exact Population Metric
- In TRL 0.24.0, batch evaluation with batch size 2 over 191 pairs produces 96 batches (95 pairs of size 2, and 1 terminal singleton of size 1).
- **Batch 95 (Row 190)**:
  - Policy chosen logp: `-380.1848`, rejected: `-640.4183`
  - Reference chosen logp: `-367.4794`, rejected: `-632.1458`
  - Reward chosen: `-1.2705`, rejected: `-0.8273`
  - Margin: **-0.4433** (classified as a **Loss**, accuracy = 0.0).
- **Full Batches (0 to 94)**: 60 batches had positive wins (sum of batch accuracies = 60.0).
- **TRL Batch Mean**: $60.0 / 96 = \mathbf{62.50\%}$.
- **Population Accuracy**: $120 / 191 = \mathbf{62.83\%}$.

---

## 3. GPU Sensitivity Run (992 Pairs)

### Experimental Protocol
- **Initial Model**: Identical fresh initialization from `Qwen/Qwen2.5-0.5B-Instruct` with the identical random seed (42).
- **Training Cohort**: `clean_ultrafeedback_992.jsonl`, consisting of the 976 intact pairs plus the 16 pairs whose prompts exceeded the 1,024-token budget (row 828, whose completion was truncated, remained strictly excluded).
- **Updates**: 124 optimization steps over 1 epoch ($992 / 8 = 124$).
- **Runtime**: 172.9 seconds.
- **Final Training Loss**: 0.6637.

### Held-Out Evaluation Metrics
- **Exact Pair-Weighted Accuracy**: **116 / 191 (60.73%)**.
  - Positive margins: 116 pairs.
  - Zero margins: 0 pairs.
  - Negative margins: 75 pairs.
- **Mean Reward Margin**: **+0.1512** (median +0.1939, min -2.0668, max +2.2933, stdev 0.7381).
- **Mean Held-Out DPO Loss**: **0.6832**.
- **TRL Batch Mean**: $58.0 / 96 = \mathbf{60.42\%}$ (row 190 remained a loss with margin -0.3764).

---

## 4. Detailed Comparison & Pair Transition Analysis

### Pair-by-Pair Classification Matrix (191 Held-Out Pairs)

| Category | Pair Count | Proportion | Interpretation |
| :--- | :---: | :---: | :--- |
| **Both Runs Win** | 109 | 57.07% | Stable positive preference alignment |
| **Both Runs Lose** | 64 | 33.51% | Stable negative or inherently difficult pairs |
| **976 Win $\to$ 992 Lose** | 11 | 5.76% | Flipped from win to loss when 16 examples added |
| **976 Lose $\to$ 992 Win** | 7 | 3.66% | Flipped from loss to win when 16 examples added |
| **Total Unchanged** | **173** | **90.58%** | Identical classification across both runs |
| **Total Flipped** | **18** | **9.42%** | Sensitivity boundary shifts |
| **Net Classification Shift** | **-4 wins** | **-2.10%** | $120 \to 116$ wins on this validation cohort |

### Breakdown of the 18 Transitioned Pairs

| Pair Index | 976 Margin | 992 Margin | $\Delta \text{Margin}$ | Transition Classification |
| :---: | :---: | :---: | :---: | :--- |
| 10 | -0.0513 | +0.0369 | +0.0882 | 976 Lose $\to$ 992 Win |
| 29 | -0.2475 | +0.0316 | +0.2791 | 976 Lose $\to$ 992 Win |
| 35 | +0.0902 | -0.1262 | -0.2164 | 976 Win $\to$ 992 Lose |
| 45 | +0.0258 | -0.1161 | -0.1419 | 976 Win $\to$ 992 Lose |
| 51 | -0.0178 | +0.1039 | +0.1217 | 976 Lose $\to$ 992 Win |
| 59 | +0.0682 | -0.0555 | -0.1237 | 976 Win $\to$ 992 Lose |
| 76 | +0.0498 | -0.0307 | -0.0805 | 976 Win $\to$ 992 Lose |
| 80 | +0.1158 | -0.2147 | -0.3305 | 976 Win $\to$ 992 Lose |
| 116 | +0.2174 | -0.0194 | -0.2368 | 976 Win $\to$ 992 Lose |
| 118 | +0.3504 | -0.1934 | -0.5438 | 976 Win $\to$ 992 Lose |
| 129 | -0.0879 | +0.3622 | +0.4501 | 976 Lose $\to$ 992 Win |
| 148 | +0.2559 | -0.0781 | -0.3340 | 976 Win $\to$ 992 Lose |
| 150 | -0.0926 | +0.0092 | +0.1018 | 976 Lose $\to$ 992 Win |
| 151 | +0.0152 | -0.0132 | -0.0284 | 976 Win $\to$ 992 Lose |
| 168 | -0.0517 | +0.1180 | +0.1697 | 976 Lose $\to$ 992 Win |
| 170 | +0.0005 | -0.0120 | -0.0125 | 976 Win $\to$ 992 Lose |
| 175 | +0.1521 | -0.0183 | -0.1704 | 976 Win $\to$ 992 Lose |
| 183 | -0.3336 | +0.0355 | +0.3691 | 976 Lose $\to$ 992 Win |

---

## 5. Methodological Interpretation & Defensible Claims

1. **What Changed When the 16 Prompt-Truncated Examples Were Included?**
   - **Step Count**: 1 epoch increased from 122 to 124 optimization steps (+1.6% gradient updates).
   - **Pair Classification**: 90.58% (173/191) of validation pairs retained identical win/loss classifications. 18 pairs (9.42%) crossed the zero-margin threshold, with 11 pairs flipping from win to loss and 7 flipping from loss to win (net -4 wins, -2.10 percentage points).
   - **Margin and Loss**: Mean margin shifted slightly higher (+0.1400 $\to$ +0.1512), and mean DPO loss decreased slightly (0.6867 $\to$ 0.6832). This demonstrates that while average reward separation expanded marginally, individual decision boundaries shifted across 18 pairs near the decision frontier.

2. **Claim Boundaries**:
   - The comparison reflects the **joint effect** of adding the 16 specific prompt-clipped training pairs along with the accompanying 2 additional optimization steps under a 1-epoch schedule.
   - On this 191-pair cohort, including the 16 prompt-truncated examples yielded a net loss of 4 pair-weighted wins alongside a +0.0112 increase in mean margin.
   - To make a statistically generalized claim of superiority or detriment, this protocol must be repeated across multiple random seeds and evaluated against a larger untouched benchmark cohort.
   - The GPU environment has been fully de-provisioned (0 active Vast.ai instances).
