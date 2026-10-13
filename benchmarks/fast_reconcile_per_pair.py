#!/usr/bin/env python3
"""
Fast Exact Per-Pair Evaluation Reconciler for DPO Checkpoint.
Evaluates all 191 validation pairs using memory-efficient selective lm_head projection
and adapter toggling on a single PeftModel instance.
"""

import sys
import gc
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "trainsight"))

from trainsight.preprocessing.dpo_preprocessor import DPOPreprocessor

device = "cpu"
print(f"Running exact per-pair reconciliation on: {device} (FP32)")

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

prep = DPOPreprocessor(tokenizer_name=model_id, max_prompt_length=1024, max_completion_length=1024, max_length=2048)

eval_path = BASE_DIR / "trainsight/sample_data/eval_ultrafeedback_test_191.jsonl"
with open(eval_path, "r", encoding="utf-8") as f:
    eval_rows = [json.loads(line) for line in f]

examples = [prep.process_sample(r) for r in eval_rows]
print(f"Total validation pairs loaded: {len(examples)}")

print("Loading base model and LoRA adapter...")
base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device)
policy_model = PeftModel.from_pretrained(base_model, str(BASE_DIR / "model_output/dpo_qwen_0_5b")).to(device)
policy_model.eval()

def compute_completion_logp(seq_ids, len_p, len_c):
    """Computes log-probability of completion tokens conditioned on prefix."""
    inp = torch.tensor([seq_ids], dtype=torch.long, device=device)
    # Backbone forward pass
    h = policy_model.base_model.model.model(inp)[0]
    # Predict completion tokens from positions len_p - 1 to len_p + len_c - 1
    sel_h = h[:, len_p - 1 : len_p + len_c - 1, :]
    sel_logits = policy_model.base_model.model.lm_head(sel_h)
    sel_labels = inp[:, len_p : len_p + len_c]
    
    loss = F.cross_entropy(
        sel_logits.reshape(-1, sel_logits.shape[-1]),
        sel_labels.reshape(-1),
        reduction="none"
    )
    return -loss.sum().item()

beta = 0.1
ref_chosen_lps = []
ref_rejected_lps = []
pol_chosen_lps = []
pol_rejected_lps = []
margins = []
chosen_rewards = []
rejected_rewards = []
base_wins = 0

print("Evaluating 191 pairs...")
for idx, ex in enumerate(examples):
    p_ids = ex.prompt_input_ids
    c_ids = ex.chosen_input_ids
    r_ids = ex.rejected_input_ids
    
    len_p = len(p_ids)
    len_c = len(c_ids)
    len_r = len(r_ids)
    
    seq_c = p_ids + c_ids
    seq_r = p_ids + r_ids
    
    with torch.no_grad():
        # 1. Reference model (adapter disabled)
        with policy_model.disable_adapter():
            ref_c_lp = compute_completion_logp(seq_c, len_p, len_c)
            ref_r_lp = compute_completion_logp(seq_r, len_p, len_r)
        
        # 2. Policy model (adapter active)
        pol_c_lp = compute_completion_logp(seq_c, len_p, len_c)
        pol_r_lp = compute_completion_logp(seq_r, len_p, len_r)
    
    ref_chosen_lps.append(ref_c_lp)
    ref_rejected_lps.append(ref_r_lp)
    pol_chosen_lps.append(pol_c_lp)
    pol_rejected_lps.append(pol_r_lp)
    
    if ref_c_lp > ref_r_lp:
        base_wins += 1
        
    r_c = beta * (pol_c_lp - ref_c_lp)
    r_r = beta * (pol_r_lp - ref_r_lp)
    m = r_c - r_r
    
    chosen_rewards.append(r_c)
    rejected_rewards.append(r_r)
    margins.append(m)
    
    if (idx + 1) % 10 == 0 or (idx + 1) == len(examples):
        print(f"  Processed {idx + 1}/{len(examples)} pairs (current margin mean: {torch.tensor(margins).mean().item():.4f})", flush=True)

margins_t = torch.tensor(margins)
pos_count = (margins_t > 0).sum().item()
zero_count = (margins_t == 0).sum().item()
neg_count = (margins_t < 0).sum().item()
total = len(margins_t)

exact_acc = pos_count / total
batch_acc_approx = 56.5 / 96  # 96 batches (95x2 + 1x1)

print("\n" + "="*60)
print("EXACT PER-PAIR POPULATION RECONCILIATION AUDIT")
print("="*60)
print(f"Total Validation Pairs:                     {total}")
print(f"Base Model Raw Win Count (logp_c > logp_r): {base_wins} ({base_wins/total:.4%})")
print(f"Policy Positive Margins (r_c > r_r):        {pos_count} ({pos_count/total:.4%})")
print(f"Policy Zero Margins (r_c == r_r):           {zero_count} ({zero_count/total:.4%})")
print(f"Policy Negative Margins (r_c < r_r):        {neg_count} ({neg_count/total:.4%})")
print(f"Exact Population Accuracy:                  {pos_count}/{total} = {exact_acc:.4%}")
print(f"TRL Unweighted Batch Mean Accuracy (96 b):  {batch_acc_approx:.4%}")
print(f"Discrepancy (Batch Mean vs Exact):          {(batch_acc_approx - exact_acc)*100:+.3f} percentage points")
print(f"Mean Reward Margin:                         {margins_t.mean().item():.4f}")
print(f"Median Reward Margin:                       {margins_t.median().item():.4f}")
print(f"Std Reward Margin:                          {margins_t.std().item():.4f}")
print(f"Mean Chosen Implicit Reward:                {torch.tensor(chosen_rewards).mean().item():.4f}")
print(f"Mean Rejected Implicit Reward:              {torch.tensor(rejected_rewards).mean().item():.4f}")

out_path = BASE_DIR / "trainsight/reports/dpo_exact_per_pair_val_reconciliation.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "total_pairs": total,
        "base_model_raw_wins": base_wins,
        "base_model_raw_winrate": base_wins / total,
        "policy_positive_margins": pos_count,
        "policy_zero_margins": zero_count,
        "policy_negative_margins": neg_count,
        "exact_population_accuracy": exact_acc,
        "trl_unweighted_batch_mean_accuracy": batch_acc_approx,
        "mean_margin": margins_t.mean().item(),
        "median_margin": margins_t.median().item(),
        "std_margin": margins_t.std().item(),
        "mean_chosen_reward": torch.tensor(chosen_rewards).mean().item(),
        "mean_rejected_reward": torch.tensor(rejected_rewards).mean().item(),
        "margins": margins,
        "ref_chosen_lps": ref_chosen_lps,
        "ref_rejected_lps": ref_rejected_lps,
        "pol_chosen_lps": pol_chosen_lps,
        "pol_rejected_lps": pol_rejected_lps
    }, f, indent=2)

print(f"\nSaved exact per-pair audit to: {out_path.name}")
