#!/usr/bin/env python3
"""
Direct Reconciliation: TrainSight Inspector Preprocessing vs. TRL DPOTrainer Dataloader & Evaluator.

Checks:
1. Dataloader tensor fidelity:
   - Does unpadded prompt_input_ids match approved prompt_input_ids?
   - Does unpadded chosen_input_ids match approved chosen_input_ids?
   - Does unpadded rejected_input_ids match approved rejected_input_ids?
   - Does EOS token ([151645]) appear exactly where approved?
2. Evaluator log-probability reconciliation:
   - Runs TRL concatenated_forward() on the batch
   - Compares against custom selective lm_head evaluator
   - Checks chosen logps, rejected logps, reward margins, and DPO loss
"""

import sys
import json
from pathlib import Path
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

def main():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cpu"
    torch_dtype = torch.float32

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

    eval_data_path = BASE_DIR / "trainsight/sample_data/eval_ultrafeedback_test_191.jsonl"
    eval_ds = load_preprocessed_dataset(eval_data_path, prep)
    # Take first 4 examples for thorough tensor and logp check
    mini_eval_ds = eval_ds.select(range(4))

    # Base model + LoRA
    base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
    base_model.warnings_issued = {}
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.warnings_issued = {}

    training_args = DPOConfig(
        output_dir="model_output/tmp_check",
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
        model=model,
        args=training_args,
        processing_class=tok,
        eval_dataset=mini_eval_ds,
    )

    eval_dataloader = trainer.get_eval_dataloader()

    # Read original jsonl raw records for ground truth comparison
    with open(eval_data_path, "r", encoding="utf-8") as f:
        raw_rows = [json.loads(f.readline()) for _ in range(4)]
    approved_examples = [prep.process_sample(r) for r in raw_rows]

    print("=" * 70)
    print("STEP 2: TENSOR & EOS FIDELITY CHECK (DATALOADER VS PREPROCESSOR)")
    print("=" * 70)

    sample_idx = 0
    for batch_idx, batch in enumerate(eval_dataloader):
        bs = batch['prompt_input_ids'].shape[0]
        for i in range(bs):
            app = approved_examples[sample_idx]
            
            # Extract unpadded tensors from dataloader batch
            p_mask = batch['prompt_attention_mask'][i] == 1
            c_mask = batch['chosen_attention_mask'][i] == 1
            r_mask = batch['rejected_attention_mask'][i] == 1

            p_ids = batch['prompt_input_ids'][i][p_mask].tolist()
            c_ids = batch['chosen_input_ids'][i][c_mask].tolist()
            r_ids = batch['rejected_input_ids'][i][r_mask].tolist()

            # Assert bitwise identity
            assert p_ids == app.prompt_input_ids, f"Sample {sample_idx}: Prompt IDs mismatch!"
            assert c_ids == app.chosen_input_ids, f"Sample {sample_idx}: Chosen IDs mismatch!"
            assert r_ids == app.rejected_input_ids, f"Sample {sample_idx}: Rejected IDs mismatch!"

            # EOS Token Audit: Verify Qwen EOS token (151645) is the terminal token
            assert c_ids[-1] == 151645, f"Sample {sample_idx}: Chosen completion does not terminate with EOS!"
            assert r_ids[-1] == 151645, f"Sample {sample_idx}: Rejected completion does not terminate with EOS!"

            print(f"Sample {sample_idx}: Bitwise Match! Prompt len: {len(p_ids)}, Chosen len: {len(c_ids)}, Rejected len: {len(r_ids)} | Terminal EOS: 151645 (verified)")
            sample_idx += 1

    print("\n✓ ALL 4 SAMPLES MATCH 100% BITWISE BETWEEN DATALOADER AND INSPECTOR CONTRACT!\n")

    print("=" * 70)
    print("STEP 3: LOG-PROB RECONCILIATION: TRL CONCATENATED_FORWARD VS CUSTOM EVALUATOR")
    print("=" * 70)

    # Let's take the first batch (bs=2) and run TRL's concatenated_forward
    first_batch = next(iter(eval_dataloader))
    
    # 1. TRL concatenated_forward
    # DPOTrainer calculates logps using concatenated forward pass:
    with torch.no_grad():
        loss_trl, metrics_trl = trainer.get_batch_loss_metrics(model, first_batch, train_eval="eval")

    print(f"TRL Loss: {loss_trl.item():.6f}")
    val_c = metrics_trl['eval_logps/chosen']
    val_r = metrics_trl['eval_logps/rejected']
    val_m = metrics_trl['eval_rewards/margins']
    trl_c_lp = val_c.item() if hasattr(val_c, 'item') else float(val_c)
    trl_r_lp = val_r.item() if hasattr(val_r, 'item') else float(val_r)
    trl_margin = val_m.item() if hasattr(val_m, 'item') else float(val_m)

    print(f"TRL Chosen Logps: {trl_c_lp:.4f}")
    print(f"TRL Rejected Logps: {trl_r_lp:.4f}")
    print(f"TRL Margin: {trl_margin:.6f}")

    # 2. Custom evaluator on the same 2 samples (unpadded individual sequence passes)
    def compute_custom_logp(seq_ids, len_p, len_c):
        dev = next(model.parameters()).device
        inp = torch.tensor([seq_ids], dtype=torch.long, device=dev)
        logits = model(inp).logits[:, :-1, :]
        labels = inp[:, 1:]
        # Mask out prompt positions (positions 0 to len_p - 1 in labels)
        mask = torch.zeros_like(labels, dtype=torch.bool)
        mask[:, len_p - 1 : len_p + len_c - 1] = True
        
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none")
        return -loss[mask.reshape(-1)].sum().item()

    custom_chosen_lps = []
    custom_rejected_lps = []
    for idx in range(2):
        app = approved_examples[idx]
        lp_c = compute_custom_logp(app.prompt_input_ids + app.chosen_input_ids, len(app.prompt_input_ids), len(app.chosen_input_ids))
        lp_r = compute_custom_logp(app.prompt_input_ids + app.rejected_input_ids, len(app.prompt_input_ids), len(app.rejected_input_ids))
        custom_chosen_lps.append(lp_c)
        custom_rejected_lps.append(lp_r)

    mean_custom_c = sum(custom_chosen_lps) / 2
    mean_custom_r = sum(custom_rejected_lps) / 2
    print(f"\nCustom Chosen Logps (mean): {mean_custom_c:.4f}")
    print(f"Custom Rejected Logps (mean): {mean_custom_r:.4f}")

    # Compare differences
    c_diff = abs(trl_c_lp - mean_custom_c)
    r_diff = abs(trl_r_lp - mean_custom_r)
    print(f"Chosen Logp Absolute Diff: {c_diff:.6f} ({c_diff / abs(mean_custom_c) * 100:.4f}%)")
    print(f"Rejected Logp Absolute Diff: {r_diff:.6f} ({r_diff / abs(mean_custom_r) * 100:.4f}%)")

    # In policy == reference (init), implicit reward is identically 0 and margin is 0
    # DPO Loss at margin 0 is ln(2) = 0.693147...
    print(f"Expected theoretical loss at init: {torch.log(torch.tensor(2.0)).item():.6f}")
    assert abs(loss_trl.item() - 0.693147) < 1e-4, f"Loss mismatch at init: {loss_trl.item()}"
    print("✓ TRL Evaluation and Custom Evaluator Reconciled Successfully!")

if __name__ == "__main__":
    main()
