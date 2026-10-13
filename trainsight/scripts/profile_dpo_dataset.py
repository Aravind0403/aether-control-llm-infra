#!/usr/bin/env python3
"""
Step 1 Deliverable: DPO Dataset Profiler for UltraFeedback Binarized
Evaluates source, schema, preference-label provenance, exact Qwen token lengths,
score distributions, truncation impacts, and descriptive lexical overlap flags.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

REVISION = "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85"
DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"
SPLIT = "train_prefs"
NUM_SAMPLES = 1000
TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

PROMPT_BUDGET = 1024
COMPLETION_BUDGET = 1024
TOTAL_BUDGET = 2048


def calculate_jaccard(text1: str, text2: str) -> float:
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    inter = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return inter / union if union > 0 else 0.0


def compute_distribution(arr: List[float]) -> Dict[str, float]:
    if not arr:
        return {}
    a = np.array(arr)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "p50_median": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(np.max(a)),
    }


def main():
    print(f"Loading tokenizer: {TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print(f"Streaming {NUM_SAMPLES} samples from {DATASET_NAME} (revision={REVISION}, split={SPLIT})...")
    ds = load_dataset(DATASET_NAME, split=SPLIT, streaming=True, revision=REVISION)

    output_dir = Path("trainsight/sample_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "real_ultrafeedback_1000.jsonl"

    report_dir = Path("trainsight/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_json_path = report_dir / "dpo_dataset_profile_ultrafeedback_1000.json"

    samples = []
    with open(jsonl_path, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(ds):
            if i >= NUM_SAMPLES:
                break
            record = {
                "prompt": row.get("prompt"),
                "prompt_id": row.get("prompt_id"),
                "chosen": row.get("chosen"),
                "rejected": row.get("rejected"),
                "score_chosen": row.get("score_chosen"),
                "score_rejected": row.get("score_rejected"),
            }
            samples.append(record)
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(samples)} rows to {jsonl_path}")

    # Metrics accumulators
    format_failures = {
        "missing_prompt": 0,
        "missing_chosen": 0,
        "missing_rejected": 0,
        "missing_scores": 0,
        "chosen_role_mismatch": 0,
        "rejected_role_mismatch": 0,
        "chosen_user_prompt_mismatch": 0,
        "rejected_user_prompt_mismatch": 0,
    }

    scores_chosen = []
    scores_rejected = []
    score_margins = []
    score_reversals = 0
    score_ties = 0
    score_positive_margins = 0

    prompt_lens_raw = []
    chosen_lens_raw = []
    rejected_lens_raw = []
    delta_lens_raw = []

    prompt_lens_trunc = []
    chosen_lens_trunc = []
    rejected_lens_trunc = []
    delta_lens_trunc = []

    budget_exceeded = {
        "prompt_exceeds_1024": 0,
        "chosen_exceeds_1024": 0,
        "rejected_exceeds_1024": 0,
        "total_chosen_pair_exceeds_2048": 0,
        "total_rejected_pair_exceeds_2048": 0,
        "any_truncation_impact": 0,
    }

    jaccard_scores = []
    identical_pairs = 0
    near_identical_90 = 0
    elevated_jaccard_80 = 0
    chosen_longer_2x = 0
    rejected_longer_2x = 0

    for idx, s in enumerate(samples):
        prompt = s.get("prompt")
        chosen = s.get("chosen")
        rejected = s.get("rejected")
        sc = s.get("score_chosen")
        sr = s.get("score_rejected")

        if not prompt:
            format_failures["missing_prompt"] += 1
            continue
        if not chosen or not isinstance(chosen, list) or len(chosen) < 2:
            format_failures["missing_chosen"] += 1
            continue
        if not rejected or not isinstance(rejected, list) or len(rejected) < 2:
            format_failures["missing_rejected"] += 1
            continue
        if sc is None or sr is None:
            format_failures["missing_scores"] += 1

        # Check role structure
        if chosen[0].get("role") != "user" or chosen[1].get("role") != "assistant":
            format_failures["chosen_role_mismatch"] += 1
        if rejected[0].get("role") != "user" or rejected[1].get("role") != "assistant":
            format_failures["rejected_role_mismatch"] += 1

        # Check user message agreement with prompt
        chosen_user_text = chosen[0].get("content", "").strip()
        rejected_user_text = rejected[0].get("content", "").strip()
        prompt_text = prompt.strip()

        if chosen_user_text != prompt_text:
            format_failures["chosen_user_prompt_mismatch"] += 1
        if rejected_user_text != prompt_text:
            format_failures["rejected_user_prompt_mismatch"] += 1

        asst_chosen = chosen[1].get("content", "")
        asst_rejected = rejected[1].get("content", "")

        # Score distributions & consistency checks
        if sc is not None and sr is not None:
            scores_chosen.append(float(sc))
            scores_rejected.append(float(sr))
            margin = float(sc) - float(sr)
            score_margins.append(margin)
            if margin < 0:
                score_reversals += 1
            elif margin == 0:
                score_ties += 1
            else:
                score_positive_margins += 1

        # Exact tokenization
        # Formatted prompt via ChatML template
        formatted_prompt_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = len(tokenizer.encode(formatted_prompt_str))
        chosen_tokens = len(tokenizer.encode(asst_chosen))
        rejected_tokens = len(tokenizer.encode(asst_rejected))

        prompt_lens_raw.append(prompt_tokens)
        chosen_lens_raw.append(chosen_tokens)
        rejected_lens_raw.append(rejected_tokens)
        delta_lens_raw.append(chosen_tokens - rejected_tokens)

        # Budget checks
        p_exceeded = prompt_tokens > PROMPT_BUDGET
        c_exceeded = chosen_tokens > COMPLETION_BUDGET
        r_exceeded = rejected_tokens > COMPLETION_BUDGET
        tot_c_exceeded = (prompt_tokens + chosen_tokens) > TOTAL_BUDGET
        tot_r_exceeded = (prompt_tokens + rejected_tokens) > TOTAL_BUDGET

        if p_exceeded:
            budget_exceeded["prompt_exceeds_1024"] += 1
        if c_exceeded:
            budget_exceeded["chosen_exceeds_1024"] += 1
        if r_exceeded:
            budget_exceeded["rejected_exceeds_1024"] += 1
        if tot_c_exceeded:
            budget_exceeded["total_chosen_pair_exceeds_2048"] += 1
        if tot_r_exceeded:
            budget_exceeded["total_rejected_pair_exceeds_2048"] += 1
        if p_exceeded or c_exceeded or r_exceeded or tot_c_exceeded or tot_r_exceeded:
            budget_exceeded["any_truncation_impact"] += 1

        # Post-truncation lengths
        p_trunc = min(prompt_tokens, PROMPT_BUDGET)
        c_trunc = min(chosen_tokens, COMPLETION_BUDGET)
        r_trunc = min(rejected_tokens, COMPLETION_BUDGET)
        prompt_lens_trunc.append(p_trunc)
        chosen_lens_trunc.append(c_trunc)
        rejected_lens_trunc.append(r_trunc)
        delta_lens_trunc.append(c_trunc - r_trunc)

        # Lexical descriptive flags (assistant text only!)
        sim = calculate_jaccard(asst_chosen, asst_rejected)
        jaccard_scores.append(sim)
        if asst_chosen.strip().lower() == asst_rejected.strip().lower():
            identical_pairs += 1
        elif sim >= 0.90:
            near_identical_90 += 1
        elif sim >= 0.80:
            elevated_jaccard_80 += 1

        if chosen_tokens > 2 * rejected_tokens:
            chosen_longer_2x += 1
        elif rejected_tokens > 2 * chosen_tokens:
            rejected_longer_2x += 1

    profile = {
        "provenance": {
            "dataset_name": DATASET_NAME,
            "revision": REVISION,
            "split": SPLIT,
            "total_split_rows": 61135,
            "sampled_rows": len(samples),
            "sampling_method": "Deterministic first 1,000 records of train_prefs",
            "test_split_preserved": "test_prefs (2,000 records) untouched",
            "score_scale": "1 to 10 scale (UltraFeedback multi-aspect average)",
            "base_model_tokenizer": TOKENIZER_NAME,
            "chat_template": "Qwen ChatML (<|im_start|>system...<|im_start|>user...<|im_start|>assistant)",
        },
        "format_and_schema_validation": {
            "total_evaluated": len(samples),
            "format_failures": format_failures,
            "clean_records_pct": 100.0 * (len(samples) - sum(format_failures.values())) / len(samples),
        },
        "score_distribution": {
            "score_chosen": compute_distribution(scores_chosen),
            "score_rejected": compute_distribution(scores_rejected),
            "score_margin": compute_distribution(score_margins),
            "reversals_count": score_reversals,
            "ties_count": score_ties,
            "positive_margins_count": score_positive_margins,
        },
        "token_lengths_raw": {
            "prompt_tokens": compute_distribution(prompt_lens_raw),
            "chosen_assistant_tokens": compute_distribution(chosen_lens_raw),
            "rejected_assistant_tokens": compute_distribution(rejected_lens_raw),
            "delta_tokens": compute_distribution(delta_lens_raw),
            "mean_length_ratio_chosen_over_rejected": (
                float(np.mean(chosen_lens_raw)) / float(np.mean(rejected_lens_raw))
                if np.mean(rejected_lens_raw) > 0 else 1.0
            ),
        },
        "token_lengths_truncated": {
            "prompt_budget": PROMPT_BUDGET,
            "completion_budget": COMPLETION_BUDGET,
            "total_budget": TOTAL_BUDGET,
            "budget_exceeded_counts": budget_exceeded,
            "prompt_tokens": compute_distribution(prompt_lens_trunc),
            "chosen_assistant_tokens": compute_distribution(chosen_lens_trunc),
            "rejected_assistant_tokens": compute_distribution(rejected_lens_trunc),
            "delta_tokens": compute_distribution(delta_lens_trunc),
            "mean_length_ratio_chosen_over_rejected": (
                float(np.mean(chosen_lens_trunc)) / float(np.mean(rejected_lens_trunc))
                if np.mean(rejected_lens_trunc) > 0 else 1.0
            ),
        },
        "descriptive_lexical_flags": {
            "jaccard_assistant_distribution": compute_distribution(jaccard_scores),
            "identical_assistant_count": identical_pairs,
            "near_identical_jaccard_ge_90_count": near_identical_90,
            "elevated_jaccard_ge_80_count": elevated_jaccard_80,
            "chosen_longer_than_2x_rejected_count": chosen_longer_2x,
            "rejected_longer_than_2x_chosen_count": rejected_longer_2x,
        },
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print(f"Profile report saved to {report_json_path}")
    print("\n=== SUMMARY RESULTS ===")
    print(f"Total Rows Sampled: {len(samples)}")
    print(f"Format Failures: {format_failures}")
    print(f"Score Reversals (chosen < rejected): {score_reversals}")
    print(f"Score Ties (chosen == rejected): {score_ties}")
    print(f"Positive Margin (chosen > rejected): {score_positive_margins}")
    print(f"Mean Chosen Score: {profile['score_distribution']['score_chosen']['mean']:.2f}")
    print(f"Mean Rejected Score: {profile['score_distribution']['score_rejected']['mean']:.2f}")
    print(f"Mean Raw Prompt Tokens: {profile['token_lengths_raw']['prompt_tokens']['mean']:.1f}")
    print(f"Mean Raw Chosen Tokens: {profile['token_lengths_raw']['chosen_assistant_tokens']['mean']:.1f}")
    print(f"Mean Raw Rejected Tokens: {profile['token_lengths_raw']['rejected_assistant_tokens']['mean']:.1f}")
    print(f"Raw Length Bias Ratio (Chosen/Rejected): {profile['token_lengths_raw']['mean_length_ratio_chosen_over_rejected']:.2f}")
    print(f"Truncated Length Bias Ratio: {profile['token_lengths_truncated']['mean_length_ratio_chosen_over_rejected']:.2f}")
    print(f"Rows Impacted by Truncation: {budget_exceeded['any_truncation_impact']} / 1000")
    print(f"Identical Assistant Completions: {identical_pairs}")
    print(f"Near-Identical (Jaccard >= 0.90): {near_identical_90}")
    print(f"Elevated (Jaccard >= 0.80): {elevated_jaccard_80}")


if __name__ == "__main__":
    main()
