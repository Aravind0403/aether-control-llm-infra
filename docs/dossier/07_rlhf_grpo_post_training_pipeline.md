# Learning Dossier 07: GRPO (Group Relative Policy Optimization) Pipeline Architecture

> **Folder Path**: `docs/dossier/07_rlhf_grpo_post_training_pipeline.md`  
> **Session Topic**: GRPO Mathematics, No-Critic Advantage Estimation, and Rule-Based Reasoning Reward Functions  
> **Date**: July 24, 2026  

---

## 1. Why GRPO Replaces PPO in Modern Reasoning Models

In traditional **PPO (Proximal Policy Optimization)** RLHF, training requires 4 active model copies in VRAM:
1. Policy (Actor) Model $\pi_\theta$
2. Reference Model $\pi_{\text{ref}}$
3. Reward Model $R_\psi$
4. Value (Critic) Model $V_\phi$

**The VRAM Bottleneck:** The Value (Critic) model must be roughly the same parameter size as the Policy model, consuming **2x the GPU memory footprint**.

### The GRPO Innovation (DeepSeek Math / DeepSeek R1)
**Group Relative Policy Optimization (GRPO)** completely **eliminates the Critic (Value) network**. Instead of training a Critic to estimate expected state value $V(s)$, GRPO samples a group of $G$ completions $\{o_1, o_2, \dots, o_G\}$ for each prompt $q$ and normalizes the rewards relative to the group mean and standard deviation:

$$A_i = \frac{r_i - \mu_{\text{group}}}{\sigma_{\text{group}} + \epsilon}$$

Where:
* $\mu_{\text{group}} = \frac{1}{G} \sum_{g=1}^G r_g$
* $\sigma_{\text{group}} = \sqrt{\frac{1}{G} \sum_{g=1}^G (r_g - \mu_{\text{group}})^2}$

```
                                [ PROMPT q ]
                                     │
                   ┌─────────────────┼─────────────────┐
                   ▼                 ▼                 ▼
             [ Sample o1 ]     [ Sample o2 ]     [ Sample o3 ]
                   │                 │                 │
                   ▼                 ▼                 ▼
              Reward r1         Reward r2         Reward r3
                   │                 │                 │
                   └─────────────────┼─────────────────┘
                                     ▼
                   [ Compute Group Mean μ & Std σ ]
                                     │
                                     ▼
                       [ Compute Advantage Ai ]
                       Ai = (ri - μ) / (σ + ε)
```

---

## 2. Dual-Reward Strategy: Format + Accuracy

For reasoning models trained on Math (GSM8K/MATH) or Code (HumanEval/MBPP), GRPO uses a composite rule-based reward function:

### 1. Format Reward ($R_{\text{format}}$)
* **Goal:** Forces the model to structure its output into distinct reasoning steps and explicit final answers.
* **Format:** `<think> ... </think> <answer> ... </answer>`
* **Score:** $+0.5$ if tags exist in proper order; $0.0$ otherwise.

### 2. Accuracy / Verifier Reward ($R_{\text{accuracy}}$)
* **Goal:** Verifies if the extracted answer matches the ground-truth solution.
* **Score:** $+1.0$ for exact string or numeric equivalence; $0.0$ otherwise.

### 3. Total Reward
$$r_i = R_{\text{accuracy}} + R_{\text{format}}$$

---

## 3. GRPO Module Verification

* [math_reward.py](../../rlhf-pipeline/rlhf_pipeline/rewards/math_reward.py): Evaluates format and accuracy rewards.
* [grpo_trainer.py](../../rlhf-pipeline/rlhf_pipeline/grpo_trainer.py): Calculates group mean, standard deviation, and relative advantages $A_i$.
* [cli.py](../../rlhf-pipeline/rlhf_pipeline/cli.py): CLI tool providing `rlhf-train simulate-step` and `rlhf-train info`.
