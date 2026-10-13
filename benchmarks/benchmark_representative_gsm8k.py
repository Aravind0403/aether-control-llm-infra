#!/usr/bin/env python3
"""
Representative Sequence Length Benchmark for Sequence Bin-Packing on Real GSM8K.

Evaluates SequenceBinPacker against 1,000 seeded Monte Carlo random collations
using actual tokenizer-measured lengths from 1,000 GSM8K training examples formatted
for Qwen/Qwen2.5-1.5B-Instruct.
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from transformers import AutoTokenizer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()
BASE_DIR = Path(__file__).resolve().parent.parent


def load_and_tokenize_gsm8k(
    data_path: Path,
    tokenizer_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> List[int]:
    """Tokenizes GSM8K prompt-completion sequences using the exact chat template formatting."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    lengths = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Standard formatting: Question prompt + Answer solution
            prompt = item.get("question", "")
            ans = item.get("answer", "")
            
            # Format using conversation messages
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ans},
            ]
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            token_ids = tokenizer.encode(formatted_text, truncation=False)
            lengths.append(len(token_ids))
    return lengths


def evaluate_batching(
    seq_lengths: List[int],
    batch_size: int,
    num_shuffles: int = 1000,
    seed: int = 42,
    drop_last: bool = False,
) -> Dict[str, Any]:
    """Evaluates sorted batching against seeded random shuffles with identical batching rules."""
    n = len(seq_lengths)
    arr_lengths = np.array(seq_lengths)
    total_useful_tokens = int(np.sum(arr_lengths))

    def _calc_padded_tokens(indices_order: np.ndarray) -> int:
        total = 0
        for i in range(0, n, batch_size):
            batch = indices_order[i:i + batch_size]
            if drop_last and len(batch) < batch_size:
                continue
            max_len = np.max(arr_lengths[batch])
            total += int(max_len * len(batch))
        return total

    # 1. Sorted / Length-Grouped Batching
    sorted_order = np.argsort(arr_lengths)
    sorted_padded_tokens = _calc_padded_tokens(sorted_order)
    sorted_waste_pct = ((sorted_padded_tokens - total_useful_tokens) / sorted_padded_tokens) * 100.0

    # 2. Random Shuffled Baseline (Recorded Seeds)
    rng = np.random.default_rng(seed)
    random_totals = []
    base_indices = np.arange(n)
    for _ in range(num_shuffles):
        shuffled = rng.permutation(base_indices)
        random_totals.append(_calc_padded_tokens(shuffled))

    random_totals = np.array(random_totals)
    mean_random = float(np.mean(random_totals))
    std_random = float(np.std(random_totals))
    min_random = float(np.min(random_totals))
    max_random = float(np.max(random_totals))
    p05_random = float(np.percentile(random_totals, 5))
    p95_random = float(np.percentile(random_totals, 95))

    random_waste_pct = ((mean_random - total_useful_tokens) / mean_random) * 100.0

    padded_token_reduction = ((mean_random - sorted_padded_tokens) / mean_random) * 100.0
    density_gain = ((mean_random / sorted_padded_tokens) - 1.0) * 100.0

    return {
        "batch_size": batch_size,
        "total_samples": n,
        "total_useful_tokens": total_useful_tokens,
        "sorted_padded_tokens": sorted_padded_tokens,
        "sorted_waste_pct": sorted_waste_pct,
        "random_mean": mean_random,
        "random_std": std_random,
        "random_min": min_random,
        "random_max": max_random,
        "random_p05": p05_random,
        "random_p95": p95_random,
        "random_waste_pct": random_waste_pct,
        "padded_token_reduction_pct": padded_token_reduction,
        "modeled_density_gain_pct": density_gain,
        "num_shuffles": num_shuffles,
        "seed": seed,
    }


def main():
    dataset_path = BASE_DIR / "trainsight" / "sample_data" / "real_gsm8k_1000.jsonl"
    console.print(f"[bold cyan]▶ Loading and tokenizing GSM8K dataset ({dataset_path.name}) with Qwen2.5 tokenizer...[/bold cyan]")
    
    t0 = time.perf_counter()
    seq_lengths = load_and_tokenize_gsm8k(dataset_path)
    tok_elapsed = time.perf_counter() - t0

    lengths_arr = np.array(seq_lengths)
    n = len(lengths_arr)

    # Dataset Distribution Table
    dist_table = Table(title=f"📊 Tokenized GSM8K Length Distribution ({n:,} sequences, {tok_elapsed:.2f}s)", border_style="cyan")
    dist_table.add_column("Statistic", style="cyan")
    dist_table.add_column("Value (Tokens)", style="bold white")
    dist_table.add_row("Total Sequences", f"{n:,}")
    dist_table.add_row("Total Useful Tokens", f"{np.sum(lengths_arr):,}")
    dist_table.add_row("Min Length", f"{np.min(lengths_arr):,}")
    dist_table.add_row("Mean Length (μ)", f"{np.mean(lengths_arr):.1f}")
    dist_table.add_row("Median Length (P50)", f"{np.median(lengths_arr):.1f}")
    dist_table.add_row("Std Dev Length (σ)", f"{np.std(lengths_arr):.1f}")
    dist_table.add_row("P90 Length", f"{np.percentile(lengths_arr, 90):.1f}")
    dist_table.add_row("P95 Length", f"{np.percentile(lengths_arr, 95):.1f}")
    dist_table.add_row("P99 Length", f"{np.percentile(lengths_arr, 99):.1f}")
    dist_table.add_row("Max Length", f"{np.max(lengths_arr):,}")
    console.print(dist_table)
    console.print()

    # Benchmark across representative batch sizes
    batch_sizes = [2, 4, 8, 16]
    res_table = Table(title="📈 Bin-Packing Offline Token Metrics vs. 1,000 Seeded Shuffles", border_style="green")
    res_table.add_column("Batch Size (B)", style="cyan", justify="center")
    res_table.add_column("Random Baseline Mean ± σ", style="white")
    res_table.add_column("Random 5th–95th Spread", style="dim")
    res_table.add_column("Sorted Padded Tokens", style="bold white")
    res_table.add_column("Padded Token Reduction", style="bold green")
    res_table.add_column("Modeled Useful Density Gain", style="bold green")

    all_results = []
    for bs in batch_sizes:
        res = evaluate_batching(seq_lengths, batch_size=bs, num_shuffles=1000, seed=42)
        all_results.append(res)
        res_table.add_row(
            f"B = {bs}",
            f"{res['random_mean']:,.0f} ± {res['random_std']:,.0f}",
            f"[{res['random_p05']:,.0f} – {res['random_p95']:,.0f}]",
            f"{res['sorted_padded_tokens']:,}",
            f"-{res['padded_token_reduction_pct']:.2f}%",
            f"+{res['modeled_density_gain_pct']:.2f}%",
        )

    console.print(res_table)
    console.print()

    # Save artifact
    output_path = BASE_DIR / "benchmarks" / "reports" / "gsm8k_representative_bin_packing.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": str(dataset_path),
            "tokenizer": "Qwen/Qwen2.5-1.5B-Instruct",
            "distribution": {
                "count": n,
                "useful_tokens": int(np.sum(lengths_arr)),
                "min": int(np.min(lengths_arr)),
                "mean": float(np.mean(lengths_arr)),
                "median": float(np.median(lengths_arr)),
                "std": float(np.std(lengths_arr)),
                "p95": float(np.percentile(lengths_arr, 95)),
                "max": int(np.max(lengths_arr)),
            },
            "batch_results": all_results,
        }, f, indent=2)
    console.print(f"[dim]Results saved to: {output_path}[/dim]")


if __name__ == "__main__":
    main()
