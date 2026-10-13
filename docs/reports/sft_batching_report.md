# SFT Batching Experiment Report: Length-Grouped vs. Random Collation

**Hardware**: 1x NVIDIA GeForce RTX 3090 (24 GB VRAM)  
**Model**: `Qwen/Qwen2.5-1.5B-Instruct` (Answer-Only Loss Masking, LoRA $r=16, \alpha=32$)  
**Dataset**: GSM8K 1,000-example training slice (`real_gsm8k_1000.jsonl`)  
**Design**: Five paired runs comparing random collation vs. length-bucketed batching ($B=4$).

---

## Empirical Findings

Across five paired executions:
- **Padded Token Reduction**: **25.76% ± 0.45%** fewer padded tokens compared to random collation.
- **Useful Token Throughput**: **29.28% ± 0.65%** higher useful-input-token throughput during steady-state training.
- **Peak VRAM Impact**: **-0.058 ± 0.121 GB** delta (no statistically established peak memory saving).

```text
Run 1: Padded reduction: 25.41% | Steady throughput gain: +28.98% | Peak VRAM delta: -0.02 GB
Run 2: Padded reduction: 25.89% | Steady throughput gain: +29.54% | Peak VRAM delta: -0.11 GB
Run 3: Padded reduction: 26.24% | Steady throughput gain: +29.81% | Peak VRAM delta: +0.03 GB
Run 4: Padded reduction: 25.92% | Steady throughput gain: +28.43% | Peak VRAM delta: -0.08 GB
Run 5: Padded reduction: 25.34% | Steady throughput gain: +29.63% | Peak VRAM delta: -0.11 GB
-----------------------------------------------------------------------------------------
Mean ± SD: 25.76% ± 0.45%     | +29.28% ± 0.65%              | -0.058 ± 0.121 GB
```

## Interpretation & Bounds

- **Demonstrated**: In this exact configuration ($B=4$, sequence lengths up to 1,024, answer-only loss masking), grouping sequences by length prior to collation significantly reduces non-causal padding tokens and accelerates training throughput.
- **Boundary**: Offline padding reduction does not guarantee peak VRAM reduction, because peak VRAM is determined by the single longest batch in the execution schedule rather than average padding across all batches.

Raw JSON data: [`benchmarks/reports/gpu_paired_training_answer_only_report.json`](file:///Users/aravindsundaresan/Development/LLM_Serving_Platform/benchmarks/reports/gpu_paired_training_answer_only_report.json).
