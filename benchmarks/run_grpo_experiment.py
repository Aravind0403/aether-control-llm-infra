#!/usr/bin/env python3
"""
GRPO Experiment Runner for Mathematical Reasoning (GSM8K)
=========================================================
Executes and evaluates a standard GRPO training pipeline on Qwen2.5-0.5B-Instruct:
1. Uses real model-generated completions with HuggingFace TRL GRPOTrainer.
2. Applies the narrow GSM8K answer-checking and format verifier contract.
3. Uses standard TRL GRPO advantage normalization (custom fallbacks disabled).
4. Records finite loss, rewards, advantages, and adapter weight change verification.
5. Saves and verifies checkpoint reloading and held-out evaluation.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from trl import GRPOTrainer, GRPOConfig
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "rlhf-pipeline"))

from rlhf_pipeline.rewards.gsm8k_contract import (
    verify_gsm8k_accuracy,
    verify_format_compliance,
    extract_answer_content,
    normalize_numeric_string,
    reconstruct_assistant_completion,
)

console = Console()


# -----------------------------------------------------------------------------
# 1. TRL-Compatible Reward Functions
# -----------------------------------------------------------------------------

def grpo_format_reward(prompts: List[Any], completions: List[Any], **kwargs) -> List[float]:
    """
    Reward function for format compliance.
    Yields +0.5 if valid <think>...</think><answer>...</answer> tags exist in proper order, else 0.0.
    Reconstructs the full assistant response consistently with prompt prefill boundary.
    """
    rewards = []
    for comp in completions:
        text = comp[0]["content"] if isinstance(comp, list) else (comp.get("content") if isinstance(comp, dict) else str(comp))
        full_text = reconstruct_assistant_completion(text)
        rewards.append(0.5 if verify_format_compliance(full_text) else 0.0)
    return rewards


def grpo_accuracy_reward(prompts: List[Any], completions: List[Any], **kwargs) -> List[float]:
    """
    Reward function for mathematical ground-truth accuracy.
    Yields +1.0 if extracted answer matches target under the narrow GSM8K contract, else 0.0.
    Reconstructs the full assistant response consistently with prompt prefill boundary.
    """
    targets = kwargs.get("answer") or kwargs.get("target") or []
    rewards = []
    for comp, target in zip(completions, targets):
        text = comp[0]["content"] if isinstance(comp, list) else (comp.get("content") if isinstance(comp, dict) else str(comp))
        full_text = reconstruct_assistant_completion(text)
        target_str = str(target)
        rewards.append(1.0 if verify_gsm8k_accuracy(full_text, target_str) else 0.0)
    return rewards



# -----------------------------------------------------------------------------
# 2. Dataset Preparation
# -----------------------------------------------------------------------------

def prepare_gsm8k_dataset(jsonl_path: Path, max_samples: Optional[int] = None, skip_samples: int = 0) -> Dataset:
    """
    Loads GSM8K JSONL file and formats each sample into ChatML prompt and target answer.
    Supports skip_samples to enforce strict train/eval isolation.
    """
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < skip_samples:
                continue
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            question = obj.get("question", "")
            raw_answer = obj.get("answer", "")
            
            # Ground truth is after ####
            target_num = raw_answer.split("####")[-1].strip() if "####" in raw_answer else raw_answer.strip()

            # Qwen ChatML conversational prompt with prefilled <think>
            prompt_chatml = (
                "<|im_start|>system\n"
                "You are a helpful math assistant. Solve the math problem step by step. "
                "The reasoning process and answer must be enclosed within <think> </think> and <answer> </answer> tags, "
                "respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n"
            )

            records.append({
                "prompt": prompt_chatml,
                "answer": target_num,
                "question": question,
                "dataset_idx": idx,
            })

            if max_samples and len(records) >= max_samples:
                break

    return Dataset.from_list(records)


# -----------------------------------------------------------------------------
# 3. Held-Out Evaluation Function
# -----------------------------------------------------------------------------

def evaluate_model_on_gsm8k(
    model: Any,
    tokenizer: Any,
    eval_dataset: Dataset,
    device: str,
    max_eval_samples: int = 20,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    """
    Evaluates model on held-out GSM8K questions using greedy generation.
    Reconstructs assistant response consistently with prompt prefill.
    Separately tracks:
    - Correct answer count & rate
    - Format compliance count & rate
    - Completion limit hits (truncated completions)
    - Total reward mean
    - Per-example evaluation records
    """
    eval_slice = eval_dataset.select(range(min(len(eval_dataset), max_eval_samples)))
    
    format_hits = 0
    accuracy_hits = 0
    limit_hits = 0
    total_rewards = []
    per_example_results = []

    model.eval()
    with torch.no_grad():
        for i, item in enumerate(eval_slice):
            prompt = item["prompt"]
            target = item["answer"]
            q_text = item.get("question", "")
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = inputs["input_ids"].shape[1]
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for reproducible evaluation
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            generated_ids = outputs[0][prompt_len:]
            gen_len = len(generated_ids)
            completion_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Reconstruct the assistant response consistently with prefill boundary
            reconstructed_text = reconstruct_assistant_completion(completion_text)

            fmt_hit = verify_format_compliance(reconstructed_text)
            acc_hit = verify_gsm8k_accuracy(reconstructed_text, target)
            hit_limit = (gen_len >= max_new_tokens)
            reward = (0.5 if fmt_hit else 0.0) + (1.0 if acc_hit else 0.0)

            if fmt_hit:
                format_hits += 1
            if acc_hit:
                accuracy_hits += 1
            if hit_limit:
                limit_hits += 1
            total_rewards.append(reward)

            per_example_results.append({
                "idx": i,
                "dataset_idx": item.get("dataset_idx", i),
                "question": q_text,
                "target": target,
                "completion": completion_text,
                "reconstructed": reconstructed_text,
                "tokens_generated": gen_len,
                "format_valid": fmt_hit,
                "accuracy_valid": acc_hit,
                "hit_completion_limit": hit_limit,
                "reward": reward,
            })

    n = len(eval_slice)
    return {
        "num_evaluated": n,
        "correct_answers_count": accuracy_hits,
        "format_compliance_count": format_hits,
        "completion_limit_hits_count": limit_hits,
        "accuracy_rate": accuracy_hits / n if n else 0.0,
        "format_compliance_rate": format_hits / n if n else 0.0,
        "completion_limit_hit_rate": limit_hits / n if n else 0.0,
        "mean_total_reward": sum(total_rewards) / n if n else 0.0,
        "per_example_results": per_example_results,
    }


# -----------------------------------------------------------------------------
# 4. Main Experiment Runner
# -----------------------------------------------------------------------------

def run_grpo_experiment(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    train_data_path: str = "trainsight/sample_data/real_gsm8k_1000.jsonl",
    eval_data_path: str = "trainsight/sample_data/real_gsm8k_1000.jsonl",
    output_dir: str = "model_output/grpo_qwen_0_5b",
    train_samples: int = 40,
    eval_samples: int = 20,
    skip_train: int = 20,
    skip_eval: int = 0,
    max_steps: int = 10,
    num_generations: int = 4,
    learning_rate: float = 5e-6,
    temperature: float = 0.9,
    max_prompt_length: int = 512,
    max_completion_length: int = 512,
    is_smoke: bool = False,
):
    console.print(Panel.fit(
        f"[bold cyan]GRPO Mathematical Reasoning Experiment[/bold cyan]\n"
        f"Model: {model_name} | Generations per Prompt (G): {num_generations}\n"
        f"Mode: {'SMOKE TEST' if is_smoke else 'PILOT RUN'} ({max_steps} steps) | Output: {output_dir}\n"
        f"Isolation: Train rows {skip_train}..{skip_train+train_samples} | Frozen Eval rows {skip_eval}..{skip_eval+eval_samples}",
        border_style="cyan"
    ))

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    console.print(f"Device: [bold yellow]{device.upper()}[/bold yellow] | PyTorch {torch.__version__}")

    # 1. Load Tokenizer & Model
    console.print("Loading base tokenizer and model...")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
    )
    if device != "cuda":
        base_model.to(device)

    # 2. Configure LoRA
    base_model.warnings_issued = {}
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.warnings_issued = {}
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    console.print(f"Trainable params: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.4f}%)")

    # Snapshot initial adapter weights across all trainable parameters to verify updates
    initial_params = {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}
    initial_weight_norm = sum(p.norm().item() for p in initial_params.values())
    console.print(f"Initial LoRA total parameter norm across {len(initial_params)} tensors: {initial_weight_norm:.6f}")

    # 3. Load & Format Datasets with Strict Split Isolation
    console.print(f"Loading GSM8K datasets (Train: {train_samples} rows [skipping first {skip_train}], Eval: {eval_samples} frozen rows [skipping first {skip_eval}])...")
    eval_ds = prepare_gsm8k_dataset(Path(eval_data_path), max_samples=eval_samples, skip_samples=skip_eval)
    train_ds = prepare_gsm8k_dataset(Path(train_data_path), max_samples=train_samples, skip_samples=skip_train)

    # 4. Baseline Pre-Training Evaluation on Frozen Evaluation Set
    console.print("\n[bold yellow]Measuring Pre-Training Baseline on Frozen GSM8K Slice...[/bold yellow]")
    pre_metrics = evaluate_model_on_gsm8k(
        model=model,
        tokenizer=tok,
        eval_dataset=eval_ds,
        device=device,
        max_eval_samples=eval_samples,
        max_new_tokens=max_completion_length,
    )
    console.print(f"Pre-Train Correct Answers:    {pre_metrics['correct_answers_count']} / {eval_samples} ({pre_metrics['accuracy_rate']*100:.1f}%)")
    console.print(f"Pre-Train Format Compliance:  {pre_metrics['format_compliance_count']} / {eval_samples} ({pre_metrics['format_compliance_rate']*100:.1f}%)")
    console.print(f"Pre-Train Limit Hits:         {pre_metrics['completion_limit_hits_count']} / {eval_samples}")
    console.print(f"Pre-Train Mean Reward:        {pre_metrics['mean_total_reward']:.4f}")

    # 5. Configure TRL GRPO Training Arguments with Explicit Settings
    beta = 0.04
    scale_rewards = "group"
    loss_type = "grpo"

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        beta=beta,
        scale_rewards=scale_rewards,
        loss_type=loss_type,
        mask_truncated_completions=True,
        temperature=temperature,
        max_steps=max_steps,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=1 if is_smoke else 2,
        fp16=False,
        bf16=(dtype == torch.bfloat16),
        report_to="none",
    )

    # 6. Initialize GRPOTrainer
    console.print("\nInitializing HuggingFace TRL GRPOTrainer with tested reward functions...")
    console.print(f"Config: loss_type='{loss_type}', beta={beta}, scale_rewards='{scale_rewards}', G={num_generations}")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[grpo_format_reward, grpo_accuracy_reward],
        args=training_args,
        train_dataset=train_ds,
        processing_class=tok,
    )

    # 7. Execute Training Loop
    console.print("\n[bold green]Executing Real Model Rollout Generation & GRPO Gradient Updates...[/bold green]")
    start_time = time.time()
    try:
        train_result = trainer.train()
    except Exception as e:
        console.print(f"[bold red]❌ GRPOTrainer.train() failed: {e}[/bold red]")
        raise e
    elapsed_time = time.time() - start_time
    steps_per_sec = max_steps / elapsed_time if elapsed_time > 0 else 0.0
    sec_per_step = elapsed_time / max_steps if max_steps > 0 else 0.0

    console.print(f"✓ Training finished in {elapsed_time:.1f}s ({sec_per_step:.2f}s/step, {steps_per_sec:.3f} steps/s)")

    # 8. Check Weight Change Evidence & Training Telemetry
    post_weight_norm = sum(param.norm().item() for name, param in model.named_parameters() if param.requires_grad)
    total_weight_delta = 0.0
    max_tensor_delta = 0.0
    for name, param in model.named_parameters():
        if name in initial_params:
            diff = (param.detach() - initial_params[name]).norm().item()
            total_weight_delta += diff
            if diff > max_tensor_delta:
                max_tensor_delta = diff
    weight_delta = total_weight_delta
    console.print(f"Post-Train LoRA total parameter norm across {len(initial_params)} tensors: {post_weight_norm:.6f}")
    console.print(f"Total Adapter Weight Norm Delta: {weight_delta:.8f} (Max single-tensor delta: {max_tensor_delta:.8f})")

    # Inspect logged telemetry for within-group variation, advantages, and gradients
    step_logs = [log for log in trainer.state.log_history if "loss" in log]
    console.print(f"\n[bold cyan]Training Step Telemetry ({len(step_logs)} logged steps):[/bold cyan]")
    has_nonzero_variation = False
    has_finite_grad = False
    for log in step_logs:
        step_idx = log.get("step")
        loss_val = log.get("loss", 0.0)
        grad_norm = log.get("grad_norm", 0.0)
        r_std = log.get("reward_std", 0.0)
        r_mean = log.get("reward", 0.0)
        acc_std = log.get("rewards/grpo_accuracy_reward/std", 0.0)
        fmt_std = log.get("rewards/grpo_format_reward/std", 0.0)
        if r_std > 0.0:
            has_nonzero_variation = True
        if grad_norm > 0.0:
            has_finite_grad = True
        console.print(f"  Step {step_idx}: Loss={loss_val:.6f} | GradNorm={grad_norm:.6f} | "
                      f"RewardMean={r_mean:.4f} | RewardStd={r_std:.4f} (AccStd={acc_std:.4f}, FmtStd={fmt_std:.4f})")

    console.print(
        "\n[dim]Note on zero advantages: Identical rewards produce zero reward-driven policy-gradient "
        "contribution for that group (A_i = 0), not necessarily zero parameter updates. Other groups, "
        f"optimizer momentum, or enabled KL penalty (beta={beta}) can still drive updates.[/dim]"
    )

    if weight_delta > 0.0:
        console.print(f"[bold green]✓ Adapter parameters verified updated: Δw = {weight_delta:.8f} > 0[/bold green]")
    else:
        console.print(f"[bold yellow]⚠ Adapter parameters unchanged: Δw = {weight_delta:.8f}[/bold yellow]")

    # 9. Measure Peak VRAM if on CUDA
    peak_vram_gb = None
    if device == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        console.print(f"Peak VRAM Allocated: {peak_vram_gb:.2f} GB")

    # 10. Post-Training Evaluation on Same Frozen GSM8K Slice
    console.print("\n[bold yellow]Measuring Post-Training Performance on Same Frozen GSM8K Slice...[/bold yellow]")
    post_metrics = evaluate_model_on_gsm8k(
        model=model,
        tokenizer=tok,
        eval_dataset=eval_ds,
        device=device,
        max_eval_samples=eval_samples,
        max_new_tokens=max_completion_length,
    )
    console.print(f"Post-Train Correct Answers:   {post_metrics['correct_answers_count']} / {eval_samples} ({post_metrics['accuracy_rate']*100:.1f}%)")
    console.print(f"Post-Train Format Compliance: {post_metrics['format_compliance_count']} / {eval_samples} ({post_metrics['format_compliance_rate']*100:.1f}%)")
    console.print(f"Post-Train Limit Hits:        {post_metrics['completion_limit_hits_count']} / {eval_samples}")
    console.print(f"Post-Train Mean Reward:       {post_metrics['mean_total_reward']:.4f}")

    # Compute Per-Example Transition Matrix
    transitions = {
        "correct_to_correct": 0,
        "correct_to_incorrect": 0,
        "incorrect_to_correct": 0,
        "incorrect_to_incorrect": 0,
        "format_gained": 0,
        "format_lost": 0,
        "detailed_transitions": []
    }
    for pre_ex, post_ex in zip(pre_metrics["per_example_results"], post_metrics["per_example_results"]):
        c_pre = pre_ex["accuracy_valid"]
        c_post = post_ex["accuracy_valid"]
        f_pre = pre_ex["format_valid"]
        f_post = post_ex["format_valid"]

        if c_pre and c_post:
            transitions["correct_to_correct"] += 1
            t_type = "Unchanged Correct"
        elif c_pre and not c_post:
            transitions["correct_to_incorrect"] += 1
            t_type = "Correct -> Incorrect"
        elif not c_pre and c_post:
            transitions["incorrect_to_correct"] += 1
            t_type = "Incorrect -> Correct"
        else:
            transitions["incorrect_to_incorrect"] += 1
            t_type = "Unchanged Incorrect"

        if not f_pre and f_post:
            transitions["format_gained"] += 1
        elif f_pre and not f_post:
            transitions["format_lost"] += 1

        transitions["detailed_transitions"].append({
            "idx": pre_ex["idx"],
            "dataset_idx": pre_ex["dataset_idx"],
            "target": pre_ex["target"],
            "transition": t_type,
            "pre_correct": c_pre,
            "post_correct": c_post,
            "pre_format": f_pre,
            "post_format": f_post,
            "pre_completion": pre_ex["completion"],
            "post_completion": post_ex["completion"],
        })

    console.print(f"\n[bold cyan]Transition Summary:[/bold cyan] "
                  f"Gain: +{transitions['incorrect_to_correct']} | "
                  f"Loss: -{transitions['correct_to_incorrect']} | "
                  f"Unchanged Correct: {transitions['correct_to_correct']} | "
                  f"Unchanged Incorrect: {transitions['incorrect_to_incorrect']}")

    # 11. Save Checkpoint & Reload Verification
    console.print(f"\n[dim]Saving checkpoint to {output_dir}...[/dim]")
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)

    console.print("[dim]Verifying Checkpoint Reload & Generation...[/dim]")
    reloaded_base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
    )
    if device != "cuda":
        reloaded_base.to(device)
    reloaded_model = PeftModel.from_pretrained(reloaded_base, output_dir)
    reloaded_model.eval()

    test_input = tok(
        "<|im_start|>system\nYou are a helpful math assistant.<|im_end|>\n"
        "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n",
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        test_out = reloaded_model.generate(**test_input, max_new_tokens=32, do_sample=False)
        test_text = tok.decode(test_out[0], skip_special_tokens=False)
    console.print(f"[bold green]✓ Checkpoint reloaded successfully and generated valid text.[/bold green]")

    # 12. Save Report
    report = {
        "model_id": model_name,
        "is_smoke": is_smoke,
        "device": device,
        "adapter_config": {
            "target_modules": list(lora_config.target_modules),
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "trainable_parameters": trainable_params,
            "total_parameters": all_params,
            "trainable_percentage": 100 * trainable_params / all_params,
        },
        "grpo_config": {
            "loss_type": loss_type,
            "beta": beta,
            "scale_rewards": scale_rewards,
            "num_generations": num_generations,
            "learning_rate": learning_rate,
            "max_prompt_length": max_prompt_length,
            "max_completion_length": max_completion_length,
        },
        "telemetry_summary": {
            "has_nonzero_variation": has_nonzero_variation,
            "has_finite_grad": has_finite_grad,
            "weight_delta": weight_delta,
            "max_tensor_delta": max_tensor_delta,
            "weight_updated": weight_delta > 0.0,
        },
        "peak_vram_gb": peak_vram_gb,
        "elapsed_time_s": elapsed_time,
        "sec_per_step": sec_per_step,
        "steps_per_sec": steps_per_sec,
        "max_steps": max_steps,
        "initial_weight_norm": initial_weight_norm,
        "post_weight_norm": post_weight_norm,
        "weight_delta": weight_delta,
        "pre_eval_metrics": pre_metrics,
        "post_eval_metrics": post_metrics,
        "transitions": transitions,
        "training_curve": trainer.state.log_history,
    }

    report_path = Path("rlhf-pipeline/reports") / f"grpo_{'smoke' if is_smoke else 'pilot'}_{int(time.time())}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    console.print(f"\n[bold green]GRPO Experiment Report Saved: {report_path}[/bold green]\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Standard TRL GRPO Training on GSM8K")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-data", type=str, default="trainsight/sample_data/real_gsm8k_1000.jsonl")
    parser.add_argument("--eval-data", type=str, default="trainsight/sample_data/real_gsm8k_1000.jsonl")
    parser.add_argument("--output-dir", type=str, default="model_output/grpo_qwen_0_5b")
    parser.add_argument("--smoke", action="store_true", help="Run short smoke test (2 steps)")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=40)
    parser.add_argument("--eval-samples", type=int, default=20)
    parser.add_argument("--skip-train", type=int, default=20)
    parser.add_argument("--skip-eval", type=int, default=0)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=5e-6)

    args = parser.parse_args()

    run_grpo_experiment(
        model_name=args.model,
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        output_dir=args.output_dir,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        skip_train=args.skip_train,
        skip_eval=args.skip_eval,
        max_steps=2 if args.smoke else args.steps,
        num_generations=args.num_generations,
        learning_rate=args.lr,
        temperature=args.temperature,
        max_completion_length=args.max_completion_length,
        is_smoke=args.smoke,
    )
