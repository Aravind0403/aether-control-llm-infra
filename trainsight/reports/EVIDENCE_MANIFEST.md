# TrainSight Milestone 1 Evidence Package Manifest

This manifest freezes and indexes all artifacts, configurations, model checkpoints, and per-pair telemetry produced for the **TrainSight DPO Data Inspection, Preprocessing, and GPU Training Milestone**.

---

## 1. Primary & Sensitivity Run Evidence

| Asset | Path | Description | Key Findings |
| :--- | :--- | :--- | :--- |
| **Primary Run Report (976 Pairs)** | [`corrected_milestone/dpo_run_clean_ultrafeedback_976.json`](file:///Users/aravindsundaresan/Development/LLM_Serving_Platform/trainsight/reports/corrected_milestone/dpo_run_clean_ultrafeedback_976.json) | 1-epoch GPU DPO training on 976 intact pairs; per-pair logps across 191 frozen eval pairs. | 120/191 wins (62.83%), mean margin +0.1400, mean loss 0.6867, peak VRAM 13.44 GB. |
| **Sensitivity Run Report (992 Pairs)** | [`corrected_milestone/dpo_run_clean_ultrafeedback_992.json`](file:///Users/aravindsundaresan/Development/LLM_Serving_Platform/trainsight/reports/corrected_milestone/dpo_run_clean_ultrafeedback_992.json) | 1-epoch GPU DPO training on 992 pairs (976 intact + 16 prompt-truncated); per-pair logps across same 191 pairs. | 116/191 wins (60.73%), mean margin +0.1512, mean loss 0.6832, peak VRAM 13.43 GB. |
| **Evaluator Reconciliation** | [`corrected_milestone/evaluator_reconciliation_diagnostic.json`](file:///Users/aravindsundaresan/Development/LLM_Serving_Platform/trainsight/reports/corrected_milestone/evaluator_reconciliation_diagnostic.json) | Bitwise check between TRL batched forward pass and custom unpadded forward pass on trained LoRA adapter. | Absolute margin delta $< 5 \times 10^{-6}$, 0 sign flips across all 191 evaluation pairs. |
| **Data Profile (1000 Raw Pairs)** | [`corrected_milestone/dpo_dataset_profile_ultrafeedback_1000.json`](file:///Users/aravindsundaresan/Development/LLM_Serving_Platform/trainsight/reports/corrected_milestone/dpo_dataset_profile_ultrafeedback_1000.json) | Inspection output on raw UltraFeedback slice. | 16 prompt budget exceedances (>1024), 1 completion exceedance (missing EOS), 124 score ties, 7 exact duplicates. |
| **2-Step Smoke Run** | [`corrected_milestone/dpo_smoke_clean_ultrafeedback_976.json`](file:///Users/aravindsundaresan/Development/LLM_Serving_Platform/trainsight/reports/corrected_milestone/dpo_smoke_clean_ultrafeedback_976.json) | Pre-flight execution telemetry on GPU prior to primary run. | Validated token collation, backward pass, loss logging, and save logic. |

---

## 2. Checkpoints & Adapters

| Checkpoint Name | Path | Hardware / Weights | Status |
| :--- | :--- | :--- | :--- |
| **Trained LoRA Adapter** | [`model_output/dpo_qwen_0_5b/`](file:///Users/aravindsundaresan/Development/LLM_Serving_Platform/model_output/dpo_qwen_0_5b/) | Qwen2.5-0.5B-Instruct (`checkpoint-122`, `adapter_model.safetensors`, 8.68 MB) | Preserved locally; evaluated in reconciliation diagnostic. |

---

## 3. Dataset Partitions & Provenance

| Split File | Row Count | Description |
| :--- | :---: | :--- |
| [`real_ultrafeedback_1000.jsonl`](file:///Users/aravindsundaresan/Development/LLM_Serving_Platform/trainsight/sample_data/real_ultrafeedback_1000.jsonl) | 1,000 | Raw UltraFeedback preference pairs from HuggingFace (`openbmb/UltraFeedback`). |
| [`clean_ultrafeedback_976.jsonl`](file:///Users/aravindsundaresan/Development/LLM_Serving_Platform/trainsight/sample_data/clean_ultrafeedback_976.jsonl) | 976 | Approved intact training slice: 0 prompt truncations, 0 completion truncations, valid terminal EOS (`151645`). |
| [`clean_ultrafeedback_992.jsonl`](file:///Users/aravindsundaresan/Development/LLM_Serving_Platform/trainsight/sample_data/clean_ultrafeedback_992.jsonl) | 992 | 976 intact + 16 prompt-truncated pairs (row 828 excluded due to completion truncation). |
| [`clipped_prompts_16.jsonl`](file:///Users/aravindsundaresan/Development/LLM_Serving_Platform/trainsight/sample_data/clipped_prompts_16.jsonl) | 16 | The isolated 16 pairs whose prompts exceed the 1,024-token budget. |
| [`clipped_completions_1.jsonl`](file:///Users/aravindsundaresan/Development/LLM_Serving_Platform/trainsight/sample_data/clipped_completions_1.jsonl) | 1 | Row 828, whose chosen completion exceeds 1,024 tokens and truncates terminal EOS. |
| [`eval_ultrafeedback_test_191.jsonl`](file:///Users/aravindsundaresan/Development/LLM_Serving_Platform/trainsight/sample_data/eval_ultrafeedback_test_191.jsonl) | 191 | Frozen held-out test cohort: 0 truncations, fully preserved prompts and completions. |

---

## 4. Revisions & Defensible Claims

1. **Defensible Pipeline Claim**:
   > *"TrainSight validates real DPO preference pairs against the intended tokenization and training budgets, identifies ineffective or truncated pairs prior to training, and enforces bitwise token-level parity between inspector gates and trainer dataloaders."*
2. **Empirical Sensitivity Finding**:
   > *"On the frozen 191-pair held-out cohort, training on 976 intact pairs yielded 120/191 wins (62.83%) with a mean margin of +0.1400. Including the 16 prompt-truncated pairs (992 pairs, 2 additional update steps) yielded 116/191 wins (60.73%) with a mean margin of +0.1512. 90.58% (173/191) of validation pairs exhibited identical classifications across both models, with 18 pairs transitioning near the zero-margin decision boundary."*
3. **Legacy Separation**:
   > *All pre-reconciliation records containing raw concatenation and conflated metric definitions have been archived to [`legacy_pre_reconciliation/`](file:///Users/aravindsundaresan/Development/LLM_Serving_Platform/trainsight/reports/legacy_pre_reconciliation/).*
