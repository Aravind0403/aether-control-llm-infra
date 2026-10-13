#!/usr/bin/env python3
"""
Integration Audit & Non-Identical Evaluator Reconciliation.

Part 1: Dataloader Contract Audit
- Validates all 976 training pairs and all 191 evaluation pairs through
  PreprocessedDPOTrainer and its collator.
- Checks bitwise token IDs, attention masks, completion boundaries,
  terminal EOS tokens, and absence of unexpected truncation.

Part 2: Evaluator Reconciliation on Trained Checkpoint (Non-Identical Policy & Frozen Reference)
- Compares TRL concatenated_forward scoring vs Custom unpadded scoring
  across all 191 held-out pairs using the trained LoRA adapter.
- Records all 4 logps (pol_chosen, pol_rejected, ref_chosen, ref_rejected),
  reward margins, and DPO losses per pair.
- Reports max, mean, and distribution of differences, and checks for margin sign flips.
- Tests the batch-padding hypothesis with a controlled single-pair forward pass.
"""

import sys
import gc
import json
from pathlib import Path
from typing import Dict, List, Any
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer
from datasets import Dataset

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "trainsight"))

from trainsight.preprocessing.dpo_preprocessor import DPOPreprocessor
from benchmarks.run_dpo_experiment import PreprocessedDPOTrainer, load_preprocessed_dataset

def run_audit():
    device = "cpu"
    torch_dtype = torch.float32
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    prep = DPOPreprocessor(
        tokenizer_name=model_id,
        max_prompt_length=1024,
        max_completion_length=1024,
        max_length=2048,
        add_eos_to_completion=True,
    )

    train_data_path = BASE_DIR / "trainsight/sample_data/clean_ultrafeedback_976.jsonl"
    eval_data_path = BASE_DIR / "trainsight/sample_data/eval_ultrafeedback_test_191.jsonl"

    train_ds = load_preprocessed_dataset(train_data_path, prep)
    eval_ds = load_preprocessed_dataset(eval_data_path, prep)

    with open(train_data_path, "r", encoding="utf-8") as f:
        train_raw = [json.loads(line) for line in f]
    with open(eval_data_path, "r", encoding="utf-8") as f:
        eval_raw = [json.loads(line) for line in f]

    print("=" * 80)
    print("CHECK 1: DATALOADER & COLLATOR CONTRACT AUDIT (976 TRAIN + 191 EVAL PAIRS)")
    print("=" * 80)

    # Initialize trainer to access its data collator
    base_dummy = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
    base_dummy.warnings_issued = {}
    lora_dummy = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    dummy_model = get_peft_model(base_dummy, lora_dummy)
    dummy_model.warnings_issued = {}

    training_args = DPOConfig(
        output_dir="model_output/tmp_audit",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        beta=0.1,
        max_prompt_length=1024,
        max_completion_length=1024,
        max_length=2048,
        use_cpu=True,
        bf16=False,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = PreprocessedDPOTrainer(
        model=dummy_model,
        args=training_args,
        processing_class=tok,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    collator = trainer.data_collator

    def audit_dataset_split(dataset, raw_rows, split_name: str):
        print(f"\nAuditing {split_name} Split ({len(dataset)} pairs)...")
        mismatches = 0
        eos_errors = 0
        truncation_errors = 0

        # Step through in batches of 2 via the collator
        batch_size = 2
        for start_idx in range(0, len(dataset), batch_size):
            end_idx = min(start_idx + batch_size, len(dataset))
            batch_features = [dataset[i] for i in range(start_idx, end_idx)]
            batch_tensors = collator(batch_features)

            for i, global_idx in enumerate(range(start_idx, end_idx)):
                approved = prep.process_sample(raw_rows[global_idx])

                p_mask = batch_tensors['prompt_attention_mask'][i] == 1
                c_mask = batch_tensors['chosen_attention_mask'][i] == 1
                r_mask = batch_tensors['rejected_attention_mask'][i] == 1

                p_ids = batch_tensors['prompt_input_ids'][i][p_mask].tolist()
                c_ids = batch_tensors['chosen_input_ids'][i][c_mask].tolist()
                r_ids = batch_tensors['rejected_input_ids'][i][r_mask].tolist()

                if p_ids != approved.prompt_input_ids or c_ids != approved.chosen_input_ids or r_ids != approved.rejected_input_ids:
                    mismatches += 1

                # EOS audit: terminal token must be 151645
                if c_ids[-1] != 151645 or r_ids[-1] != 151645:
                    eos_errors += 1

                # Truncation check
                if len(p_ids) > 1024 or len(c_ids) > 1024 or len(r_ids) > 1024:
                    truncation_errors += 1

        print(f"[{split_name}] Completed audit of {len(dataset)} pairs.")
        print(f"[{split_name}] Token Mismatches: {mismatches}")
        print(f"[{split_name}] EOS Errors: {eos_errors}")
        print(f"[{split_name}] Unexpected Truncation / Over-budget: {truncation_errors}")
        assert mismatches == 0, f"{split_name} has token mismatches!"
        assert eos_errors == 0, f"{split_name} has EOS errors!"
        assert truncation_errors == 0, f"{split_name} has truncation errors!"

    audit_dataset_split(train_ds, train_raw, "TRAINING")
    audit_dataset_split(eval_ds, eval_raw, "EVALUATION")

    print("\n✓ CHECK 1 PASSED: 100% Bitwise fidelity across all 976 training and 191 evaluation pairs!")

    # Free dummy model
    del dummy_model, base_dummy
    gc.collect()

    print("\n" + "=" * 80)
    print("CHECK 2: NON-IDENTICAL EVALUATOR RECONCILIATION ON SAVED CHECKPOINT (191 PAIRS)")
    print("=" * 80)

    checkpoint_dir = BASE_DIR / "model_output/dpo_qwen_0_5b"
    print(f"Loading diagnostic LoRA checkpoint from: {checkpoint_dir}")
    base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
    base_model.warnings_issued = {}
    policy_model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
    policy_model.warnings_issued = {}
    if hasattr(policy_model, "base_model"):
        policy_model.base_model.warnings_issued = {}
        if hasattr(policy_model.base_model, "model"):
            policy_model.base_model.model.warnings_issued = {}
    policy_model.eval()

    # Create trainer for policy evaluation with batch_size=2
    trainer_policy = PreprocessedDPOTrainer(
        model=policy_model,
        args=training_args,
        processing_class=tok,
        eval_dataset=eval_ds,
    )

    eval_dataloader = trainer_policy.get_eval_dataloader()

    beta = 0.1

    # Storage for per-pair metrics
    trl_results = []
    custom_results = []

    def compute_custom_logp(model_to_use, seq_ids, len_p, len_c):
        dev = next(model_to_use.parameters()).device
        inp = torch.tensor([seq_ids], dtype=torch.long, device=dev)
        logits = model_to_use(inp).logits[:, :-1, :]
        labels = inp[:, 1:]
        mask = torch.zeros_like(labels, dtype=torch.bool)
        mask[:, len_p - 1 : len_p + len_c - 1] = True
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none")
        return -loss[mask.reshape(-1)].sum().item()

    print(f"Scoring all {len(eval_ds)} validation pairs through both evaluators...")

    current_pair_idx = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            batch_size = batch['prompt_input_ids'].shape[0]

            # 1. TRL scoring on this batch
            # We can run concatenated_forward for policy and reference:
            with policy_model.disable_adapter():
                ref_forward = trainer_policy.concatenated_forward(policy_model, batch, is_ref_model=True)
            pol_forward = trainer_policy.concatenated_forward(policy_model, batch, is_ref_model=False)

            num_ex = batch_size
            trl_pol_c_all = pol_forward['chosen_logps'].tolist()
            trl_pol_r_all = pol_forward['rejected_logps'].tolist()
            trl_ref_c_all = ref_forward['chosen_logps'].tolist()
            trl_ref_r_all = ref_forward['rejected_logps'].tolist()

            for i in range(batch_size):
                global_idx = current_pair_idx + i
                approved = prep.process_sample(eval_raw[global_idx])
                p_ids = approved.prompt_input_ids
                c_ids = approved.chosen_input_ids
                r_ids = approved.rejected_input_ids

                # TRL values for this pair
                t_pol_c = trl_pol_c_all[i]
                t_pol_r = trl_pol_r_all[i]
                t_ref_c = trl_ref_c_all[i]
                t_ref_r = trl_ref_r_all[i]

                t_rew_c = beta * (t_pol_c - t_ref_c)
                t_rew_r = beta * (t_pol_r - t_ref_r)
                t_margin = t_rew_c - t_rew_r
                t_loss = -F.logsigmoid(torch.tensor(t_margin)).item()

                trl_results.append({
                    "idx": global_idx,
                    "pol_c": t_pol_c, "pol_r": t_pol_r,
                    "ref_c": t_ref_c, "ref_r": t_ref_r,
                    "margin": t_margin, "loss": t_loss
                })

                # 2. Custom unpadded scoring for this pair
                with policy_model.disable_adapter():
                    c_ref_c = compute_custom_logp(policy_model, p_ids + c_ids, len(p_ids), len(c_ids))
                    c_ref_r = compute_custom_logp(policy_model, p_ids + r_ids, len(p_ids), len(r_ids))

                c_pol_c = compute_custom_logp(policy_model, p_ids + c_ids, len(p_ids), len(c_ids))
                c_pol_r = compute_custom_logp(policy_model, p_ids + r_ids, len(p_ids), len(r_ids))

                c_rew_c = beta * (c_pol_c - c_ref_c)
                c_rew_r = beta * (c_pol_r - c_ref_r)
                c_margin = c_rew_c - c_rew_r
                c_loss = -F.logsigmoid(torch.tensor(c_margin)).item()

                custom_results.append({
                    "idx": global_idx,
                    "pol_c": c_pol_c, "pol_r": c_pol_r,
                    "ref_c": c_ref_c, "ref_r": c_ref_r,
                    "margin": c_margin, "loss": c_loss
                })

            current_pair_idx += batch_size
            if (batch_idx + 1) % 20 == 0 or current_pair_idx == len(eval_ds):
                print(f"Evaluated {current_pair_idx}/{len(eval_ds)} pairs...")

    # Reconciliation analysis across all 191 pairs
    diff_pol_c = [abs(t["pol_c"] - c["pol_c"]) for t, c in zip(trl_results, custom_results)]
    diff_pol_r = [abs(t["pol_r"] - c["pol_r"]) for t, c in zip(trl_results, custom_results)]
    diff_ref_c = [abs(t["ref_c"] - c["ref_c"]) for t, c in zip(trl_results, custom_results)]
    diff_ref_r = [abs(t["ref_r"] - c["ref_r"]) for t, c in zip(trl_results, custom_results)]
    diff_margin = [abs(t["margin"] - c["margin"]) for t, c in zip(trl_results, custom_results)]
    diff_loss = [abs(t["loss"] - c["loss"]) for t, c in zip(trl_results, custom_results)]

    sign_disagreements = []
    for t, c in zip(trl_results, custom_results):
        t_sign = 1 if t["margin"] > 0 else (-1 if t["margin"] < 0 else 0)
        c_sign = 1 if c["margin"] > 0 else (-1 if c["margin"] < 0 else 0)
        if t_sign != c_sign:
            sign_disagreements.append((t["idx"], t["margin"], c["margin"]))

    print("\n" + "=" * 80)
    print("AUDIT RESULTS: TRL BATCHED VS CUSTOM UNPADDED (191 PAIRS)")
    print("=" * 80)
    print(f"Policy Chosen Logp Diff:    mean={sum(diff_pol_c)/len(diff_pol_c):.6f}, max={max(diff_pol_c):.6f}")
    print(f"Policy Rejected Logp Diff:  mean={sum(diff_pol_r)/len(diff_pol_r):.6f}, max={max(diff_pol_r):.6f}")
    print(f"Reference Chosen Logp Diff:  mean={sum(diff_ref_c)/len(diff_ref_c):.6f}, max={max(diff_ref_c):.6f}")
    print(f"Reference Rejected Logp Diff: mean={sum(diff_ref_r)/len(diff_ref_r):.6f}, max={max(diff_ref_r):.6f}")
    print(f"Reward Margin Diff:         mean={sum(diff_margin)/len(diff_margin):.6f}, max={max(diff_margin):.6f}")
    print(f"DPO Loss Diff:              mean={sum(diff_loss)/len(diff_loss):.6f}, max={max(diff_loss):.6f}")
    print(f"Margin Sign Disagreements:  {len(sign_disagreements)} / {len(eval_ds)}")
    if sign_disagreements:
        print("Disagreements detail (idx, trl_margin, custom_margin):", sign_disagreements)

    trl_wins = sum(1 for t in trl_results if t["margin"] > 0)
    custom_wins = sum(1 for c in custom_results if c["margin"] > 0)
    print(f"\nPopulation Positive Margins (Wins):")
    print(f"  TRL Evaluator:    {trl_wins} / 191 ({trl_wins / 191 * 100:.2f}%)")
    print(f"  Custom Evaluator: {custom_wins} / 191 ({custom_wins / 191 * 100:.2f}%)")

    audit_summary = {
        "train_pairs_audited": len(train_ds),
        "eval_pairs_audited": len(eval_ds),
        "check1_status": "PASSED_100_PERCENT_BITWISE",
        "evaluator_reconciliation": {
            "num_pairs": len(eval_ds),
            "pol_c_diff": {"mean": sum(diff_pol_c)/len(diff_pol_c), "max": max(diff_pol_c)},
            "pol_r_diff": {"mean": sum(diff_pol_r)/len(diff_pol_r), "max": max(diff_pol_r)},
            "ref_c_diff": {"mean": sum(diff_ref_c)/len(diff_ref_c), "max": max(diff_ref_c)},
            "ref_r_diff": {"mean": sum(diff_ref_r)/len(diff_ref_r), "max": max(diff_ref_r)},
            "margin_diff": {"mean": sum(diff_margin)/len(diff_margin), "max": max(diff_margin)},
            "loss_diff": {"mean": sum(diff_loss)/len(diff_loss), "max": max(diff_loss)},
            "sign_disagreements_count": len(sign_disagreements),
            "sign_disagreements": sign_disagreements,
            "trl_positive_margin_count": trl_wins,
            "custom_positive_margin_count": custom_wins,
        }
    }

    report_out = BASE_DIR / "trainsight/reports/evaluator_reconciliation_diagnostic.json"
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"\nAudit summary saved to: {report_out}")

if __name__ == "__main__":
    run_audit()
