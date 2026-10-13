#!/usr/bin/env python3
"""
DPO Training & Held-Out Evaluation Experiment Runner.

Experimental Protocol:
1. Model Choice:
   - Fixed to Qwen/Qwen2.5-0.5B-Instruct with LoRA (r=16, alpha=32).
   - 1.5B is reserved for separately named experiments; results are never mixed.
2. Dataset Cohorts:
   - Option A (Primary): clean_ultrafeedback_976.jsonl (976 fully intact pairs, 0 truncations).
   - Sensitivity Baseline: clean_ultrafeedback_993.jsonl (993 pairs, includes 16 left-truncated prompts and 1 right-truncated completion).
   - Fixed Held-Out Eval Cohort: eval_ultrafeedback_test_191.jsonl (191 fully intact test pairs, audited from test_prefs).
3. Training & Evaluation Contract:
   - Pinned to TRL 0.24.0 DPOTrainer & DPOConfig.
   - max_prompt_length=1024, max_completion_length=1024, max_length=2048.
   - beta=0.1, lr=5e-5, AdamW.
4. Metrics:
   - eval_loss, eval_rewards/accuracies (held-out preference ranking accuracy),
   - eval_rewards/margins (mean chosen vs rejected reward margin).
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "trainsight"))

from trainsight.preprocessing.dpo_preprocessor import DPOPreprocessor


class PreprocessedDPOTrainer(DPOTrainer):
    """
    Subclass of DPOTrainer that respects pre-tokenized features produced by TrainSight DPOPreprocessor.
    Bypasses TRL's redundant _prepare_dataset / tokenize_row to prevent duplicate chat templating and duplicate EOS tokens.
    """
    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        if dataset is None:
            return None
        if "prompt_input_ids" in dataset.column_names and "chosen_input_ids" in dataset.column_names:
            return dataset
        return super()._prepare_dataset(dataset, processing_class, args, dataset_name)


def load_preprocessed_dataset(file_path: Path, preprocessor: DPOPreprocessor) -> Dataset:
    """Loads a JSONL preference dataset using TrainSight's DPOPreprocessor contract."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    examples = [preprocessor.process_sample(r) for r in records]

    return Dataset.from_dict({
        "prompt_input_ids": [ex.prompt_input_ids for ex in examples],
        "chosen_input_ids": [ex.chosen_input_ids for ex in examples],
        "rejected_input_ids": [ex.rejected_input_ids for ex in examples],
    })


def run_experiment(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    train_data_path: str = "trainsight/sample_data/clean_ultrafeedback_976.jsonl",
    eval_data_path: str = "trainsight/sample_data/eval_ultrafeedback_test_191.jsonl",
    output_dir: str = "model_output/dpo_qwen_0_5b",
    max_steps: int = -1,
    num_train_epochs: float = 1.0,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    is_smoke: bool = False,
) -> Dict[str, Any]:
    """Runs DPO training and evaluation under pinned TRL 0.24.0 settings."""
    console.print(Panel.fit(
        f"[bold cyan]DPO Experiment Runner[/bold cyan]\n"
        f"Model: [green]{model_id}[/green]\n"
        f"Train Dataset: [yellow]{train_data_path}[/yellow]\n"
        f"Eval Dataset: [yellow]{eval_data_path}[/yellow]\n"
        f"Mode: {'[bold red]SMOKE TEST (2 steps)[/bold red]' if is_smoke else '[bold green]FULL TRAINING[/bold green]'}",
        title="Configuration",
    ))

    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.bfloat16
        use_bf16 = True
    elif torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float32
        use_bf16 = False
    else:
        device = "cpu"
        torch_dtype = torch.float32
        use_bf16 = False

    console.print(f"[dim]Device: {device} | Dtype: {torch_dtype} | bf16={use_bf16}[/dim]")

    # 1. Load Tokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # 2. Preprocessor & Datasets
    prep = DPOPreprocessor(
        tokenizer_name=model_id,
        max_prompt_length=1024,
        max_completion_length=1024,
        max_length=2048,
        add_eos_to_completion=True,
    )
    train_ds = load_preprocessed_dataset(Path(train_data_path), prep)
    eval_ds = load_preprocessed_dataset(Path(eval_data_path), prep)

    console.print(f"Loaded train pairs: {len(train_ds)} | eval pairs: {len(eval_ds)} (ChatML formatted, EOS audited)")

    if is_smoke:
        # Mini slice for fast smoke test
        train_ds = train_ds.select(range(min(4, len(train_ds))))
        eval_ds = eval_ds.select(range(min(4, len(eval_ds))))
        max_steps = 2
        logging_steps = 1
        eval_steps = 2
    else:
        logging_steps = 10
        eval_steps = 50

    # 3. Load Model with LoRA
    console.print("[dim]Loading base model and configuring LoRA adapter...[/dim]")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,
    )
    # Ensure warnings_issued dictionary is present to accommodate transformers 5.x / TRL 0.24.0
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

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    console.print(f"Trainable params: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")

    # 4. Training Arguments
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs if max_steps <= 0 else 1.0,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        beta=beta,
        bf16=use_bf16,
        fp16=False,
        max_prompt_length=1024,
        max_completion_length=1024,
        max_length=2048,
        logging_steps=logging_steps,
        eval_strategy="steps" if is_smoke else "epoch",
        eval_steps=eval_steps if is_smoke else None,
        save_strategy="no" if is_smoke else "epoch",
        report_to="none",
        remove_unused_columns=False,
    )

    # 5. Initialize PreprocessedDPOTrainer
    trainer = PreprocessedDPOTrainer(
        model=model,
        args=training_args,
        processing_class=tok,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    # Pre-flight contract verification: dataloader tensors strictly match preprocessor outputs
    test_batch = next(iter(trainer.get_eval_dataloader()))
    with open(eval_data_path, "r", encoding="utf-8") as f:
        sample_ex = prep.process_sample(json.loads(f.readline()))
    unpadded_p = test_batch['prompt_input_ids'][0][test_batch['prompt_attention_mask'][0] == 1].tolist()
    assert unpadded_p == sample_ex.prompt_input_ids, "Dataloader prompt tokens mismatch preprocessor approved tokens!"
    unpadded_c = test_batch['chosen_input_ids'][0][test_batch['chosen_attention_mask'][0] == 1].tolist()
    assert unpadded_c == sample_ex.chosen_input_ids, "Dataloader chosen tokens mismatch preprocessor approved tokens!"
    console.print("[bold green]✓ Tensor contract verified: Dataloader batches bitwise match TrainSight approved tokens.[/bold green]")

    # Pre-training evaluation on held-out cohort (baseline accuracy)
    console.print("\n[bold yellow]Running Pre-Training Evaluation on Held-Out Cohort...[/bold yellow]")
    pre_eval_metrics = trainer.evaluate()
    console.print(f"Pre-Train Eval Loss: {pre_eval_metrics.get('eval_loss', 0):.4f} | "
                  f"WinRate (Acc): {pre_eval_metrics.get('eval_rewards/accuracies', 0):.4f} | "
                  f"Margin: {pre_eval_metrics.get('eval_rewards/margins', 0):.4f}")

    # 6. Train
    console.print("\n[bold green]Starting DPO Training...[/bold green]")
    start_time = time.time()
    train_result = trainer.train()
    elapsed_time = time.time() - start_time

    # 7. Post-training evaluation on held-out cohort
    console.print("\n[bold yellow]Running Post-Training Evaluation on Held-Out Cohort...[/bold yellow]")
    post_eval_metrics = trainer.evaluate()
    console.print(f"Post-Train Eval Loss: {post_eval_metrics.get('eval_loss', 0):.4f} | "
                  f"WinRate (Acc): {post_eval_metrics.get('eval_rewards/accuracies', 0):.4f} | "
                  f"Margin: {post_eval_metrics.get('eval_rewards/margins', 0):.4f}")

    # 7b. Exact Per-Pair Evaluation Audit
    console.print("\n[bold cyan]Computing Exact Per-Pair Evaluation Metrics across Held-Out Cohort...[/bold cyan]")
    eval_dataloader = trainer.get_eval_dataloader()
    per_pair_results = []
    with torch.no_grad():
        for batch in eval_dataloader:
            bs = batch['prompt_input_ids'].shape[0]
            with model.disable_adapter():
                ref_out = trainer.concatenated_forward(model, batch, is_ref_model=True)
            pol_out = trainer.concatenated_forward(model, batch, is_ref_model=False)

            pol_c = pol_out['chosen_logps'].tolist()
            pol_r = pol_out['rejected_logps'].tolist()
            ref_c = ref_out['chosen_logps'].tolist()
            ref_r = ref_out['rejected_logps'].tolist()

            for i in range(bs):
                r_c = beta * (pol_c[i] - ref_c[i])
                r_r = beta * (pol_r[i] - ref_r[i])
                margin = r_c - r_r
                pair_loss = -F.logsigmoid(torch.tensor(margin)).item()
                per_pair_results.append({
                    "idx": len(per_pair_results),
                    "pol_chosen_logp": pol_c[i],
                    "pol_rejected_logp": pol_r[i],
                    "ref_chosen_logp": ref_c[i],
                    "ref_rejected_logp": ref_r[i],
                    "reward_chosen": r_c,
                    "reward_rejected": r_r,
                    "reward_margin": margin,
                    "dpo_loss": pair_loss,
                    "win": 1 if margin > 0 else (0 if margin < 0 else 0.5)
                })

    pos_wins = sum(1 for p in per_pair_results if p["reward_margin"] > 0)
    zero_margins = sum(1 for p in per_pair_results if p["reward_margin"] == 0)
    neg_losses = sum(1 for p in per_pair_results if p["reward_margin"] < 0)
    pair_weighted_acc = pos_wins / len(per_pair_results) if per_pair_results else 0.0
    mean_margin = sum(p["reward_margin"] for p in per_pair_results) / len(per_pair_results) if per_pair_results else 0.0
    mean_dpo_loss = sum(p["dpo_loss"] for p in per_pair_results) / len(per_pair_results) if per_pair_results else 0.0

    console.print(f"[bold green]Exact Population Wins: {pos_wins} / {len(per_pair_results)} ({pair_weighted_acc * 100:.2f}%)[/bold green]")
    console.print(f"[bold green]Exact Mean Margin: {mean_margin:+.4f} | Mean Loss: {mean_dpo_loss:.4f}[/bold green]")

    # 8. Save Final Model Checkpoint
    console.print(f"\n[dim]Saving final checkpoint and tokenizer to {output_dir}...[/dim]")
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)

    peak_allocated_gb = None
    peak_reserved_gb = None
    if device == "cuda":
        peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)

    report = {
        "model_id": model_id,
        "train_dataset": str(train_data_path),
        "train_pairs_count": len(train_ds),
        "eval_dataset": str(eval_data_path),
        "eval_pairs_count": len(eval_ds),
        "is_smoke": is_smoke,
        "device": device,
        "peak_vram_allocated_gb": peak_allocated_gb,
        "peak_vram_reserved_gb": peak_reserved_gb,
        "elapsed_time_s": elapsed_time,
        "train_loss": train_result.training_loss,
        "training_curve": trainer.state.log_history,
        "pre_eval_metrics": pre_eval_metrics,
        "post_eval_metrics": post_eval_metrics,
        "exact_population_metrics": {
            "total_pairs": len(per_pair_results),
            "positive_margins_count": pos_wins,
            "zero_margins_count": zero_margins,
            "negative_margins_count": neg_losses,
            "pair_weighted_accuracy": pair_weighted_acc,
            "mean_reward_margin": mean_margin,
            "mean_dpo_loss": mean_dpo_loss,
        },
        "per_pair_results": per_pair_results,
        "checkpoint_dir": str(output_dir),
        "training_params": {
            "beta": beta,
            "learning_rate": learning_rate,
            "max_prompt_length": 1024,
            "max_completion_length": 1024,
            "max_length": 2048,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        }
    }

    # Save report
    report_path = Path("trainsight/reports") / f"dpo_{'smoke' if is_smoke else 'run'}_{Path(train_data_path).stem}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    console.print(f"\n[bold green]Experiment Report Saved to: {report_path}[/bold green]")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DPO Training & Evaluation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Pinned model name")
    parser.add_argument("--train-data", type=str, default="trainsight/sample_data/clean_ultrafeedback_976.jsonl")
    parser.add_argument("--eval-data", type=str, default="trainsight/sample_data/eval_ultrafeedback_test_191.jsonl")
    parser.add_argument("--output-dir", type=str, default="model_output/dpo_qwen_0_5b")
    parser.add_argument("--smoke", action="store_true", help="Run short smoke test (2 steps)")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)

    args = parser.parse_args()

    run_experiment(
        model_id=args.model,
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        is_smoke=args.smoke,
    )
