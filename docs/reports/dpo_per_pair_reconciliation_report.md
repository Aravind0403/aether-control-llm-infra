# Exact Per-Pair Validation Reconciliation Audit (DPO Checkpoint)

**Model:** `Qwen/Qwen2.5-0.5B-Instruct` + LoRA Checkpoint (`model_output/dpo_qwen_0_5b`)  
**Training Cohort:** 976 intact preference pairs (`clean_ultrafeedback_976.jsonl`)  
**Validation Cohort:** 191 held-out intact preference pairs (`eval_ultrafeedback_test_191.jsonl`)  
**Evaluation Mode:** Exact per-pair log-likelihood computation (FP32)

---

## 1. Dissecting Raw Likelihood Ranking vs. DPO Reward Margin

To avoid mixing incompatible metrics, we distinguish between:
1. **Raw Conditional-Likelihood Ranking:** Does the model assign higher log-probability to the chosen completion than to the rejected completion ($\log \pi(y_w \mid x) > \log \pi(y_l \mid x)$)?
2. **DPO Implicit Reward Ranking:** Did training increase the chosen likelihood relative to the reference model more than the rejected likelihood ($r_w > r_l \iff \Delta r > 0$)?

| Metric | Base Model (Pre-Train) | Policy Checkpoint (Post-DPO) | Delta / Shift |
| :--- | :---: | :---: | :---: |
| **Raw Conditional-Likelihood Wins** | **75 / 191 (39.27%)** | **76 / 191 (39.79%)** | **+1 win (+0.52 pp)** |
| **DPO Implicit Reward Wins ($\Delta r > 0$)** | — (Identically 0% at init) | **105 / 191 (54.97%)** | Separate metric |

> [!IMPORTANT]
> **Why Raw Ranking and DPO Margin Diverge:**  
> A preference pair can have a positive DPO reward margin ($\Delta r > 0$) while the policy still assigns higher raw probability to the rejected response ($\log \pi_\theta(y_l \mid x) > \log \pi_\theta(y_w \mid x)$). This occurs whenever the chosen completion has a lower raw token probability (e.g. longer response length), but training shifted the relative probability ratio in favor of the chosen completion.  
> Conflating raw base ranking (39.27%) with post-DPO reward ranking (54.97%) is mathematically invalid. The comparable before/after raw ranking moved from **39.27% to 39.79%**.

---

## 2. Per-Pair Margin Distribution & Reconciliation (FP32 Evaluator)

From the saved per-pair log-probabilities across all 191 validation pairs:

| Metric | Population Value | Formula / Details |
| :--- | :---: | :--- |
| **Total Validation Pairs ($N$)** | **191** | Full validation cohort |
| **Positive Reward Margins ($\Delta r > 0$)** | **105** | $54.9738\%$ of population |
| **Zero Reward Margins ($\Delta r == 0$)** | **0** | $0.0000\%$ |
| **Negative Reward Margins ($\Delta r < 0$)** | **86** | $45.0262\%$ |
| **Pair-Weighted Reward Accuracy** | **54.97%** | $105 / 191$ |
| **Batch-Averaged Accuracy ($B=2$, 96 batches)** | **54.69%** | $\frac{105 + s}{192} = \frac{105 + 0}{192} = 54.6875\%$ ($s=0$ on singleton row 190) |
| **Mean Reward Margin ($\mu_{\Delta r}$)** | **+0.0527** | Median: $+0.0522$, Std: $0.5115$ |
| **Mean Chosen Implicit Reward ($r_w$)** | **-0.2700** | Had a less negative mean reward than rejected |
| **Mean Rejected Implicit Reward ($r_l$)** | **-0.3227** | |
| **Mean DPO Loss ($-\log \sigma(\Delta r)$)** | **0.6984** | Evaluated in FP32 across all 191 pairs |

---

## 3. Dissecting the 54.97% vs. 58.85% Evaluator Discrepancy

During the GPU training run, TRL logged:
$$\text{eval\_rewards/accuracies} = 0.5885416865348816 \approx 58.854\%$$

Applying the exact batch aggregation algebra for $N=191$ evaluated with batch size $B=2$ (95 pairs of 2 + 1 singleton):
$$\text{Batch Mean} = \frac{1}{96} \left[ \sum_{b=1}^{95} \frac{w_{2b-1} + w_{2b}}{2} + s \right] = \frac{\text{Total Wins} + s}{192}$$

Where $s \in \{0, 1\}$ is the classification outcome of the 191st singleton pair.

Since $\frac{113}{192} \approx 0.58854167$, the aggregate alone permits two possibilities:
1. **112 wins with a winning singleton ($s = 1$):** $\frac{112 + 1}{192} = \frac{113}{192} \approx 58.854\%$
2. **113 wins with a losing singleton ($s = 0$):** $\frac{113 + 0}{192} = \frac{113}{192} \approx 58.854\%$

> [!IMPORTANT]
> **Aggregate Indeterminacy & Pairwise Flips:**  
> - We cannot infer exactly 112 GPU wins from the aggregate alone without confirming the 191st singleton outcome from saved per-pair records.  
> - A net difference of 7 (or 8) total wins between evaluators (e.g., 105 vs. 112/113) also does **not** establish that exactly 7 individual pairs changed classification. Pairs can flip classifications in both directions between evaluators.

### Separate Unresolved Factors
The divergence between the original GPU evaluation and the CPU FP32 evaluation stems from at least two separate, unmeasured factors:
1. **Formatting Factor (Raw Text vs. ChatML Template):** In the original run, `DPOTrainer` was passed raw prompt strings without message formatting, tokenizing directly without ChatML delimiters (`<|im_start|>user...<|im_start|>assistant\n`), whereas the custom evaluator used TrainSight's standard ChatML formatting.
2. **Precision & Runtime Factor (BF16 on GPU vs. FP32 on CPU):** The GPU run evaluated in `bfloat16` with hardware attention kernels, while the local audit evaluated in full `float32`.

Their individual contributions have not been isolated or measured independently; they must be treated as separate unresolved factors until controlled ablations are run.

---

## 4. Defensible Status & Next Steps

The defensible claim is:
> **"TrainSight’s shared DPO preprocessing is integrated into the trainer, with evaluation-token alignment verified. Full training under the corrected contract is pending."**

Notice that **"with lower loss" is explicitly omitted**:
- The reference baseline loss at initialization is $-\log \sigma(0) = \ln 2 \approx 0.6931$.
- The post-training evaluation reports a mean loss of **0.6984** on the 191 pairs.
- Because the DPO loss $L(\Delta r) = \log(1 + e^{-\Delta r})$ is convex, outliers with negative margins incur large loss penalties (e.g., $\Delta r = -1.0 \implies L = 1.313$) that outweigh modest positive gains ($\Delta r = +1.0 \implies L = 0.313$). Consequently, a **positive mean margin can coexist with a higher mean loss**.

The next concrete steps are:
1. Make the experiment runner consume the same preprocessing and collator contract as the TrainSight inspector.
2. Verify approved tensors against batches taken directly from the trainer's dataloader, including completion masks and EOS handling.
3. Reconcile the custom evaluator and TRL using identical tokens, checkpoint, reference, precision, and evaluation settings.
4. Rerun the 976-pair primary experiment from the original initialization, and run the 992-pair sensitivity experiment afterward.

---

## 5. Completed Integration & Non-Identical Evaluator Reconciliation (Checks 1 & 2)

### Check 1: Full Dataloader & Collator Audit
Audited all **976 training pairs** (`clean_ultrafeedback_976.jsonl`) and all **191 evaluation pairs** (`eval_ultrafeedback_test_191.jsonl`) processed through `PreprocessedDPOTrainer.data_collator`:
- **Token Mismatches:** 0 / 976 (Train), 0 / 191 (Eval) — 100% bitwise identity with `DPOPreprocessor.process_sample()`.
- **Terminal EOS Handling:** 0 / 976 (Train), 0 / 191 (Eval) errors — all chosen and rejected completions terminate strictly with Qwen `<|im_end|>` (`151645`).
- **Unexpected Truncation / Over-budget:** 0 occurrences across all 1,167 pairs (prompt $\le 1024$, completion $\le 1024$, total $\le 2048$).

### Check 2: Non-Identical Evaluator Reconciliation on Trained Checkpoint
Scored all **191 validation pairs** using the saved LoRA checkpoint against frozen base reference (`Qwen/Qwen2.5-0.5B-Instruct`) comparing TRL batched forward passes (`concatenated_forward`, batch size 2) against the custom unpadded evaluator in identical FP32 precision:

| Metric | Mean Absolute Difference | Max Absolute Difference | Notes |
| :--- | :---: | :---: | :--- |
| **Policy Chosen Log-Probability** | $0.009156$ | $0.042358$ | Sub-0.01 logp difference |
| **Policy Rejected Log-Probability** | $0.007818$ | $0.053467$ | |
| **Reference Chosen Log-Probability** | $0.009189$ | $0.042603$ | |
| **Reference Rejected Log-Probability** | $0.007846$ | $0.053711$ | |
| **Reward Margin ($\Delta r$)** | $\mathbf{0.000005}$ ($4.6 \times 10^{-6}$) | $\mathbf{0.000043}$ ($4.3 \times 10^{-5}$) | Negligible numerical drift |
| **DPO Loss ($-\log \sigma(\Delta r)$)** | $\mathbf{0.000002}$ ($2.3 \times 10^{-6}$) | $\mathbf{0.000022}$ ($2.2 \times 10^{-5}$) | Virtually identical loss curves |
| **Margin Sign Disagreements** | **0 / 191 (0.0%)** | — | **Zero classification flips** |
| **Population Positive Margins (Wins)** | **98 / 191 (51.31%)** | **98 / 191 (51.31%)** | Identical on both evaluators |

> [!NOTE]
> **Why Margin and Loss Differences Are Orders of Magnitude Smaller than Log-Prob Differences:**  
> The minor log-probability variance ($<0.05$) between batched and unpadded forward passes affects the policy model and the reference model symmetrically. In the DPO reward margin calculation:
> $$\Delta r = \beta [ (\log \pi(y_w) - \log \pi_{\text{ref}}(y_w)) - (\log \pi(y_l) - \log \pi_{\text{ref}}(y_l)) ]$$
> the shared batch-padding drift cancels out, yielding mean margin differences on the order of $10^{-6}$ and identical sign classifications.


