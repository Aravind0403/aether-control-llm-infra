import os
import argparse
from typing import List, Dict, Any
from pathlib import Path

from rlhf_pipeline.rewards.math_reward import GRPOReasoningReward


def reward_function_format(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """HuggingFace TRL compatible reward function for reasoning format verification (<think> tags)."""
    evaluator = GRPOReasoningReward()
    rewards = []
    for comp in completions:
        text = comp[0]["content"] if isinstance(comp, list) else str(comp)
        rewards.append(evaluator.evaluate_format_reward(text))
    return rewards


def reward_function_accuracy(prompts: List[str], completions: List[str], answer: List[str], **kwargs) -> List[float]:
    """HuggingFace TRL compatible reward function for mathematical ground-truth accuracy."""
    evaluator = GRPOReasoningReward()
    rewards = []
    for comp, target in zip(completions, answer):
        text = comp[0]["content"] if isinstance(comp, list) else str(comp)
        rewards.append(evaluator.evaluate_accuracy_reward(text, target))
    return rewards


def run_trl_grpo_training(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir: str = "./results/grpo_qwen",
    max_steps: int = 50,
    learning_rate: float = 1e-6,
    num_generations: int = 4
):
    """
    Production PyTorch & HuggingFace TRL GRPO Fine-Tuning Execution Loop.
    Loads GSM8K dataset, configures GRPOTrainer with custom reward verifiers,
    and runs distributed gradient updates.
    """
    try:
        import torch
        from datasets import load_dataset
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from trl import GRPOTrainer, GRPOConfig
    except ImportError:
        print("⚠️ PyTorch, HuggingFace datasets/transformers, or TRL not installed in local environment.")
        print("💡 To run live PyTorch training: pip install torch transformers datasets trl accelerate deepspeed")
        return

    print(f"🚀 Initializing HuggingFace TRL GRPOTrainer on model: {model_name}")
    print(f"Parameters: Steps={max_steps} | LR={learning_rate} | Generations per Prompt (G)={num_generations}")

    # 1. Load HuggingFace GSM8K Dataset
    dataset = load_dataset("openai/gsm8k", "main", split="train[:500]")

    def format_gsm8k_prompt(example):
        return {
            "prompt": f"Solve the following math problem step by step. Enclose your reasoning in <think>...</think> and your final answer in <answer>...</answer>.\nQuestion: {example['question']}",
            "answer": example["answer"].split("####")[-1].strip()
        }

    dataset = dataset.map(format_gsm8k_prompt)

    # 2. Configure TRL GRPO Training Arguments
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        max_steps=max_steps,
        num_generations=num_generations,
        max_prompt_length=512,
        max_completion_length=256,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        logging_steps=5,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none"
    )

    # 3. Instantiate GRPOTrainer with custom reward functions
    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[reward_function_format, reward_function_accuracy],
        args=training_args,
        train_dataset=dataset
    )

    print("✨ Starting GRPOTrainer model weight updates...")
    trainer.train()
    print("✅ Training complete. Checkpoints saved to:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch HuggingFace TRL GRPO Fine-Tuning Script")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    run_trl_grpo_training(model_name=args.model, max_steps=args.steps)
