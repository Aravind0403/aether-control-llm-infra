#!/usr/bin/env python3
"""
GPU Paired Training Benchmark: Random Batching vs. Length-Grouped Batching.

Strict Experimental Protocol:
1. Fixed training configuration:
   - 1,000 real GSM8K training examples (formatted with Qwen chat template).
   - Qwen/Qwen2.5-1.5B-Instruct base model with bfloat16.
   - Strictly requires CUDA and BF16 (fails fast with explicit error if unavailable).
   - Identical LoRA configuration (r=16, alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]).
   - Fixed batch size = 8, gradient accumulation = 1 (no batch size adjustment).
   - Fixed optimizer (AdamW, lr=1e-5), loss reduction='mean'.
   - Resets ONLY the trainable LoRA adapter weights between runs (base model remains untouched on GPU).
2. Dynamic Padding & Attention Mask:
   - Passes explicit attention_mask built from original sequence lengths.
   - Mode A (Random): Sequences shuffled randomly with recorded seed, dynamic batch padding.
   - Mode B (Length-Grouped): Sequences sorted by length, chunked into batches of 8,
     and batch order is randomly shuffled to eliminate curriculum artifacts.
3. GPU Measurements:
   - torch.cuda.synchronize() for accurate step timing after warm-up (5 steps).
   - torch.cuda.empty_cache() and reset_peak_memory_stats() between runs.
   - Clear metric separation:
     * total_epoch_time_s: Includes 5 warm-up steps.
     * steady_state_step_ms: Excludes 5 warm-up steps.
     * steady_state_useful_tokens_per_sec: Useful tokens / steady-state elapsed time.
     * peak_allocated_gb: Active tensor memory footprint.
     * peak_reserved_gb: PyTorch caching allocator pool size (tracked separately).
4. Paired alternating execution across 5 runs (A-B, B-A, A-B, B-A, A-B).
   Sample standard deviations computed with ddof=1 across the 5 paired differences.
5. Pre-flight Smoke Check:
   Optionally runs 1 random and 1 grouped mini-run to confirm VRAM headroom and loss validity.
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "trainsight"))

from trainsight.preprocessing.sft_preprocessor import SFTPreprocessor, SFTExample


class GSM8KTokenizedDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        tokenizer_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_seq_len: int = 2048,
        truncation: bool = True,
    ):
        self.preprocessor = SFTPreprocessor(
            tokenizer_name=tokenizer_name,
            max_seq_len=max_seq_len,
            truncation=truncation,
        )
        self.tokenizer = self.preprocessor.tokenizer
        
        self.examples: List[SFTExample] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                ex = self.preprocessor.process_sample(item)
                if ex.is_valid:
                    self.examples.append(ex)
        
        self.lengths = [ex.total_len for ex in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> SFTExample:
        return self.examples[idx]


def build_batches(
    dataset: GSM8KTokenizedDataset,
    batch_size: int,
    mode: str,
    seed: int,
) -> List[List[int]]:
    """Builds index batches under either random or length-grouped collation."""
    n = len(dataset)
    indices = np.arange(n)
    rng = np.random.default_rng(seed)

    if mode == "random":
        shuffled = rng.permutation(indices)
        batches = [shuffled[i:i + batch_size].tolist() for i in range(0, n, batch_size)]
    elif mode == "length_grouped":
        # Sort indices by sequence length
        sorted_indices = sorted(range(n), key=lambda i: dataset.lengths[i])
        # Form batches of adjacent length
        batches = [sorted_indices[i:i + batch_size] for i in range(0, n, batch_size)]
        # CRITICAL: Randomize batch order to avoid shortest-to-longest curriculum artifacts
        rng.shuffle(batches)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return batches


def collate_and_pad_batch(
    batch_indices: List[int],
    dataset: GSM8KTokenizedDataset,
    pad_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Dynamically pads a batch using the shared preprocessor with answer-only loss masking."""
    batch_examples = [dataset[idx] for idx in batch_indices]
    return dataset.preprocessor.collate_batch(batch_examples, device=device)


def run_single_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: GSM8KTokenizedDataset,
    batches: List[List[int]],
    pad_token_id: int,
    device: torch.device,
    warmup_steps: int = 5,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Runs a training epoch through batches with GPU synchronization and explicit metric separation."""
    model.train()
    total_useful_tokens = 0
    total_padded_tokens = 0
    steady_useful_tokens = 0
    steady_padded_tokens = 0
    losses = []

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    t_start = time.perf_counter()
    step_times = []
    t_steady_start = None

    steps_to_run = batches if max_steps is None else batches[:max_steps]

    for step, batch_idx in enumerate(steps_to_run):
        if step == warmup_steps:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_steady_start = time.perf_counter()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_step_0 = time.perf_counter()

        input_ids, attention_mask, labels, u_tokens, p_tokens = collate_and_pad_batch(
            batch_idx, dataset, pad_token_id, device
        )
        total_useful_tokens += u_tokens
        total_padded_tokens += p_tokens

        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_step = time.perf_counter() - t_step_0

        losses.append(loss.item())

        if step >= warmup_steps:
            step_times.append(t_step)
            steady_useful_tokens += u_tokens
            steady_padded_tokens += p_tokens

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_total = time.perf_counter() - t_start
    t_steady_total = (time.perf_counter() - t_steady_start) if t_steady_start else t_total

    peak_allocated_gb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if device.type == "cuda"
        else 0.0
    )
    peak_reserved_gb = (
        torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        if device.type == "cuda"
        else 0.0
    )

    examples_processed = len(steps_to_run) * (len(batches[0]) if batches else 0)
    examples_per_sec = examples_processed / t_total if t_total > 0 else 0.0
    overall_useful_tokens_per_sec = total_useful_tokens / t_total if t_total > 0 else 0.0
    steady_useful_tokens_per_sec = steady_useful_tokens / t_steady_total if t_steady_total > 0 else 0.0
    avg_step_ms = (np.mean(step_times) * 1000.0) if step_times else 0.0

    return {
        "total_epoch_time_s": t_total,
        "steady_state_time_s": t_steady_total,
        "steady_state_step_ms": avg_step_ms,
        "examples_per_sec": examples_per_sec,
        "overall_useful_tokens_per_sec": overall_useful_tokens_per_sec,
        "steady_useful_tokens_per_sec": steady_useful_tokens_per_sec,
        "total_useful_tokens": total_useful_tokens,
        "total_padded_tokens": total_padded_tokens,
        "steady_useful_tokens": steady_useful_tokens,
        "steady_padded_tokens": steady_padded_tokens,
        "padding_waste_pct": ((total_padded_tokens - total_useful_tokens) / max(1, total_padded_tokens)) * 100.0,
        "peak_allocated_gb": peak_allocated_gb,
        "peak_reserved_gb": peak_reserved_gb,
        "mean_loss": float(np.mean(losses)),
        "first_loss": float(losses[0]) if losses else 0.0,
        "final_loss": float(losses[-1]) if losses else 0.0,
    }


def execute_paired_experiment(
    data_path: Path,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    batch_size: int = 4,
    num_pairs: int = 5,
    base_seed: int = 100,
    allow_cpu_dry_run: bool = False,
    smoke_check_only: bool = False,
) -> Dict[str, Any]:
    """Executes the full paired benchmark with strict GPU/BF16 enforcement and alternating order."""
    # 1. Enforce intended GPU hardware & precision requirements
    if not torch.cuda.is_available():
        if allow_cpu_dry_run:
            console.print("[bold yellow]⚠️ CUDA is unavailable. Running in CPU dry-run mode (--allow-cpu-dry-run).[/bold yellow]")
            device = torch.device("cpu")
            dtype = torch.float32
        else:
            raise RuntimeError(
                "CUDA is strictly required for this benchmark. torch.cuda.is_available() is False. "
                "To run locally for offline testing, pass --allow-cpu-dry-run."
            )
    else:
        device = torch.device("cuda")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                f"BF16 precision is strictly required on GPU ({torch.cuda.get_device_name(0)}), "
                "but torch.cuda.is_bf16_supported() is False. Aborting to prevent unintended FP32/FP16 fallback."
            )
        dtype = torch.bfloat16
        console.print(f"[bold green]✅ Verified GPU: {torch.cuda.get_device_name(0)} with native bfloat16 support.[/bold green]")

    console.print(f"[bold cyan]▶ Loading GSM8K dataset from {data_path.name}...[/bold cyan]")
    dataset = GSM8KTokenizedDataset(data_path, tokenizer_name=model_name)
    pad_token_id = dataset.tokenizer.pad_token_id

    console.print(f"[bold cyan]▶ Initializing base model {model_name} on {device} ({dtype})...[/bold cyan]")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)

    # Base model is frozen; only LoRA adapter is trained
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()

    # Save initial state dict of ONLY the trainable LoRA adapter weights (few megabytes, NOT full model)
    initial_lora_state = {k: v.cpu().clone() for k, v in get_peft_model_state_dict(lora_model).items()}

    def reset_lora_adapter():
        """Resets only the trainable LoRA adapter weights, avoiding large base model copies."""
        set_peft_model_state_dict(lora_model, {k: v.to(device) for k, v in initial_lora_state.items()})
        trainable_params = [p for p in lora_model.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable_params, lr=1e-5)

    # --------------------------------------------------------------------------
    # PRE-FLIGHT SMOKE CHECK
    # --------------------------------------------------------------------------
    if smoke_check_only or num_pairs >= 1:
        console.print("\n[bold yellow]🔍 Running Pre-Flight Smoke Check (1 Random, 1 Length-Grouped, 10 steps each)...[/bold yellow]")
        smoke_results = {}
        for mode in ["random", "length_grouped"]:
            opt = reset_lora_adapter()
            smoke_batches = build_batches(dataset, batch_size=batch_size, mode=mode, seed=999)
            res = run_single_epoch(lora_model, opt, dataset, smoke_batches, pad_token_id, device, warmup_steps=2, max_steps=10)
            smoke_results[mode] = res
            console.print(
                f"  • Smoke {mode.upper()}: Loss {res['first_loss']:.4f} ──► {res['final_loss']:.4f} | "
                f"Peak Allocated: {res['peak_allocated_gb']:.2f} GB | Reserved: {res['peak_reserved_gb']:.2f} GB | Status: OK"
            )
            assert not np.isnan(res['final_loss']), f"Smoke check failed: NaN loss encountered in {mode} mode"
            assert res['final_loss'] < 100.0, f"Smoke check failed: Exploding loss in {mode} mode"

        console.print("[bold green]✅ Pre-Flight Smoke Check passed: Valid losses and clean VRAM allocation confirmed.[/bold green]\n")
        if smoke_check_only:
            return {"smoke_check": "PASS", "results": smoke_results}

    # --------------------------------------------------------------------------
    # 5 PAIRED ALTERNATING RUNS
    # --------------------------------------------------------------------------
    pair_results = []
    for pair_idx in range(num_pairs):
        seed_rand = base_seed + pair_idx * 2
        seed_grp = base_seed + pair_idx * 2 + 1
        is_random_first = (pair_idx % 2 == 0)

        order_str = "Random ──► Length-Grouped" if is_random_first else "Length-Grouped ──► Random"
        console.print(f"[bold yellow]━━━ Executing Pair {pair_idx + 1}/{num_pairs} ({order_str}) ━━━[/bold yellow]")

        modes_to_run = ["random", "length_grouped"] if is_random_first else ["length_grouped", "random"]
        seeds = [seed_rand, seed_grp] if is_random_first else [seed_grp, seed_rand]

        pair_data = {}
        for m, s in zip(modes_to_run, seeds):
            opt = reset_lora_adapter()
            batches = build_batches(dataset, batch_size=batch_size, mode=m, seed=s)
            console.print(f"  • Running {m.upper()} (Seed: {s}, Batches: {len(batches)})...")
            res = run_single_epoch(lora_model, opt, dataset, batches, pad_token_id, device, warmup_steps=5)
            pair_data[m] = res
            console.print(
                f"    ↳ Total Time: {res['total_epoch_time_s']:.2f}s | Steady Step: {res['steady_state_step_ms']:.1f}ms | "
                f"Steady Useful: {res['steady_useful_tokens_per_sec']:,.0f} tok/s | Peak Allocated VRAM: {res['peak_allocated_gb']:.2f} GB"
            )

        rand = pair_data["random"]
        grp = pair_data["length_grouped"]

        padded_reduction = ((rand["total_padded_tokens"] - grp["total_padded_tokens"]) / rand["total_padded_tokens"]) * 100.0
        steady_tok_speedup = ((grp["steady_useful_tokens_per_sec"] - rand["steady_useful_tokens_per_sec"]) / rand["steady_useful_tokens_per_sec"]) * 100.0
        overall_tok_speedup = ((grp["overall_useful_tokens_per_sec"] - rand["overall_useful_tokens_per_sec"]) / rand["overall_useful_tokens_per_sec"]) * 100.0
        vram_allocated_delta_gb = grp["peak_allocated_gb"] - rand["peak_allocated_gb"]

        pair_results.append({
            "pair": pair_idx + 1,
            "order": order_str,
            "random": rand,
            "length_grouped": grp,
            "padded_reduction_pct": padded_reduction,
            "steady_useful_tok_s_gain_pct": steady_tok_speedup,
            "overall_useful_tok_s_gain_pct": overall_tok_speedup,
            "vram_allocated_delta_gb": vram_allocated_delta_gb,
        })

    # Summary Statistics with Sample SD (ddof=1)
    reductions = [p["padded_reduction_pct"] for p in pair_results]
    steady_speedups = [p["steady_useful_tok_s_gain_pct"] for p in pair_results]
    overall_speedups = [p["overall_useful_tok_s_gain_pct"] for p in pair_results]
    vram_allocated_deltas = [p["vram_allocated_delta_gb"] for p in pair_results]

    summary = {
        "num_pairs": num_pairs,
        "batch_size": batch_size,
        "mean_padded_reduction_pct": float(np.mean(reductions)),
        "sample_sd_padded_reduction_pct": float(np.std(reductions, ddof=1)) if len(reductions) > 1 else 0.0,
        "mean_steady_useful_tok_s_gain_pct": float(np.mean(steady_speedups)),
        "sample_sd_steady_useful_tok_s_gain_pct": float(np.std(steady_speedups, ddof=1)) if len(steady_speedups) > 1 else 0.0,
        "mean_overall_useful_tok_s_gain_pct": float(np.mean(overall_speedups)),
        "sample_sd_overall_useful_tok_s_gain_pct": float(np.std(overall_speedups, ddof=1)) if len(overall_speedups) > 1 else 0.0,
        "mean_peak_allocated_vram_delta_gb": float(np.mean(vram_allocated_deltas)),
        "sample_sd_peak_allocated_vram_delta_gb": float(np.std(vram_allocated_deltas, ddof=1)) if len(vram_allocated_deltas) > 1 else 0.0,
        "pairs": pair_results,
    }

    # Print Summary Table
    table = Table(title=f"📊 Paired GPU Training Results ({num_pairs} Alternating Runs, Sample SD ddof=1)", border_style="green")
    table.add_column("Pair", style="cyan", justify="center")
    table.add_column("First Mode", style="dim")
    table.add_column("Random (tok/s)", style="white")
    table.add_column("Grouped (tok/s)", style="white")
    table.add_column("Steady Useful tok/s Gain", style="bold green")
    table.add_column("Padded Token Reduction", style="bold cyan")
    table.add_column("Peak Allocated VRAM Δ", style="yellow")

    for p in pair_results:
        table.add_row(
            str(p["pair"]),
            p["order"].split(" ──► ")[0],
            f"{p['random']['steady_useful_tokens_per_sec']:,.0f}",
            f"{p['length_grouped']['steady_useful_tokens_per_sec']:,.0f}",
            f"+{p['steady_useful_tok_s_gain_pct']:.2f}%",
            f"-{p['padded_reduction_pct']:.2f}%",
            f"{p['vram_allocated_delta_gb']:+.2f} GB",
        )

    table.add_section()
    table.add_row(
        "SAMPLE MEAN ± s",
        "—",
        "—",
        "—",
        f"+{summary['mean_steady_useful_tok_s_gain_pct']:.2f}% ± {summary['sample_sd_steady_useful_tok_s_gain_pct']:.2f}%",
        f"-{summary['mean_padded_reduction_pct']:.2f}% ± {summary['sample_sd_padded_reduction_pct']:.2f}%",
        f"{summary['mean_peak_allocated_vram_delta_gb']:+.2f} ± {summary['sample_sd_peak_allocated_vram_delta_gb']:.2f} GB",
    )
    console.print(table)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run paired GPU training benchmark")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--pairs", type=int, default=5, help="Number of paired alternating runs")
    parser.add_argument("--allow-cpu-dry-run", action="store_true", help="Allow fallback to CPU for local testing")
    parser.add_argument("--smoke-check", action="store_true", help="Run only the smoke check (1 random, 1 grouped)")
    args = parser.parse_args()

    data_path = BASE_DIR / "trainsight" / "sample_data" / "real_gsm8k_1000.jsonl"
    report_file = BASE_DIR / "benchmarks" / "reports" / "gpu_paired_training_report.json"

    console.print(Panel("🚀 TrainSight GPU Paired Training Benchmark Harness", style="bold green"))
    results = execute_paired_experiment(
        data_path,
        batch_size=args.batch_size,
        num_pairs=args.pairs,
        allow_cpu_dry_run=args.allow_cpu_dry_run,
        smoke_check_only=args.smoke_check,
    )

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    console.print(f"[dim]Report saved to: {report_file}[/dim]")


if __name__ == "__main__":
    main()
