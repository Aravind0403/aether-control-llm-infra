#!/usr/bin/env python3
"""
Exact Per-Pair Validation Reconciliation for DPO Checkpoint.

Evaluates every individual pair in eval_ultrafeedback_test_191.jsonl
against the trained LoRA policy vs. base reference model:
- Computes exact per-pair implicit rewards: r_c = beta * (logp_pol(y_w) - logp_ref(y_w))
- Computes exact per-pair margins: Delta r_i = r_{c, i} - r_{r, i}
- Counts positive (Delta r > 0), zero (Delta r == 0), and negative (Delta r < 0) margins.
- Compares true population accuracy against unweighted batch-mean accuracy.
"""

import sys
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rich.console import Console
from rich.table import Table

console = Console()
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "trainsight"))

from trainsight.preprocessing.dpo_preprocessor import DPOPreprocessor

device = "mps" if torch.backends.mps.is_available() else "cpu"
console.print(f"[dim]Running reconciliation on device: {device}[/dim]")

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

prep = DPOPreprocessor(tokenizer_name=model_id, max_prompt_length=1024, max_completion_length=1024, max_length=2048)

eval_path = BASE_DIR / "trainsight/sample_data/eval_ultrafeedback_test_191.jsonl"
with open(eval_path, "r", encoding="utf-8") as f:
    eval_rows = [json.loads(line) for line in f]

console.print(f"Loaded {len(eval_rows)} validation rows from {eval_path.name}.")

# 1. Load Base Reference Model
console.print("Loading base reference model...")
ref_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device)
ref_model.eval()

# 2. Load Policy Model (LoRA Adapter)
console.print("Loading trained LoRA policy model...")
base_for_policy = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device)
policy_model = PeftModel.from_pretrained(base_for_policy, str(BASE_DIR / "model_output/dpo_qwen_0_5b")).to(device)
policy_model.eval()

beta = 0.1

def compute_completion_logp(model, prompt_ids, comp_ids):
    full_ids = torch.tensor([prompt_ids + comp_ids], dtype=torch.long, device=device)
    labels = full_ids.clone()
    labels[0, :len(prompt_ids)] = -100
    
    with torch.no_grad():
        logits = model(full_ids).logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits.reshape(-1, logits.shape[-1]), shift_labels.reshape(-1))
        mask = (shift_labels.reshape(-1) != -100)
        return -loss[mask].sum().item()

records = []
margins = []
chosen_rewards = []
rejected_rewards = []

console.print(f"Evaluating {len(eval_rows)} pairs...")
for idx, r in enumerate(eval_rows):
    ex = prep.process_sample(r)
    p_ids = ex.prompt_input_ids
    c_ids = ex.chosen_input_ids
    r_ids = ex.rejected_input_ids

    ref_c = compute_completion_logp(ref_model, p_ids, c_ids)
    ref_r = compute_completion_logp(ref_model, p_ids, r_ids)

    pol_c = compute_completion_logp(policy_model, p_ids, c_ids)
    pol_r = compute_completion_logp(policy_model, p_ids, r_ids)

    r_chosen = beta * (pol_c - ref_c)
    r_rejected = beta * (pol_r - ref_r)
    margin = r_chosen - r_rejected

    margins.append(margin)
    chosen_rewards.append(r_chosen)
    rejected_rewards.append(r_rejected)

    records.append({
        "index": idx,
        "score_chosen": r.get("score_chosen"),
        "score_rejected": r.get("score_rejected"),
        "r_chosen": r_chosen,
        "r_rejected": r_rejected,
        "margin": margin,
        "won": margin > 0,
        "tied": margin == 0,
    })

    if (idx + 1) % 25 == 0 or (idx + 1) == len(eval_rows):
        cur_pos = sum(1 for m in margins if m > 0)
        console.print(f"  Processed {idx + 1}/{len(eval_rows)} | Wins so far: {cur_pos}/{idx + 1} ({cur_pos/(idx+1):.2%})")

margins_t = torch.tensor(margins)
pos_count = (margins_t > 0).sum().item()
zero_count = (margins_t == 0).sum().item()
neg_count = (margins_t < 0).sum().item()
total = len(margins)
exact_acc = pos_count / total

table = Table(title="Exact Per-Pair Validation Reconciliation (N=191)")
table.add_column("Metric", style="cyan")
table.add_column("Count / Value", style="green")
table.add_column("Percentage", style="yellow")

table.add_row("Total Validation Pairs", str(total), "100.00%")
table.add_row("Positive Margin (r_w > r_l)", str(pos_count), f"{pos_count/total:.2%}")
table.add_row("Zero Margin (r_w == r_l)", str(zero_count), f"{zero_count/total:.2%}")
table.add_row("Negative Margin (r_w < r_l)", str(neg_count), f"{neg_count/total:.2%}")
table.add_row("Exact Population WinRate", f"{pos_count} / {total}", f"{exact_acc:.4%}")
table.add_row("Mean Reward Margin", f"{margins_t.mean().item():.4f}", "—")
table.add_row("Median Reward Margin", f"{margins_t.median().item():.4f}", "—")
table.add_row("Std Reward Margin", f"{margins_t.std().item():.4f}", "—")
table.add_row("Mean Chosen Reward", f"{torch.tensor(chosen_rewards).mean().item():.4f}", "—")
table.add_row("Mean Rejected Reward", f"{torch.tensor(rejected_rewards).mean().item():.4f}", "—")

console.print(table)

out_file = BASE_DIR / "trainsight/reports/dpo_exact_per_pair_val_reconciliation.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump({
        "total_pairs": total,
        "positive_count": pos_count,
        "zero_count": zero_count,
        "negative_count": neg_count,
        "exact_accuracy": exact_acc,
        "mean_margin": margins_t.mean().item(),
        "median_margin": margins_t.median().item(),
        "std_margin": margins_t.std().item(),
        "mean_chosen_reward": torch.tensor(chosen_rewards).mean().item(),
        "mean_rejected_reward": torch.tensor(rejected_rewards).mean().item(),
        "per_pair_records": records,
    }, f, indent=2)

console.print(f"[bold green]Saved exact reconciliation to: {out_file.name}[/bold green]")
