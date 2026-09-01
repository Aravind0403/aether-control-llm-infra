from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from trainsight.inspectors.sft_inspector import SFTInspector
from trainsight.inspectors.dpo_inspector import DPOInspector

app = typer.Typer(
    name="trainsight",
    help="Pre-training and post-training dataset validator CLI",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command("version")
def version():
    """Print trainsight CLI version."""
    console.print("[bold cyan]trainsight v0.1.0[/bold cyan]")


@app.command("profile")
def profile(
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Path to JSONL dataset file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    dataset_type: str = typer.Option(
        "sft",
        "--type",
        "-t",
        help="Dataset type: sft, dpo, or rl",
    ),
    max_seq_len: int = typer.Option(
        2048,
        "--max-seq-len",
        "-m",
        help="Maximum target sequence length for OOM risk assessment",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit with error code 1 if any warnings are found (ideal for K8s InitContainers / CI)",
    ),
    tokenizer: Optional[str] = typer.Option(
        None,
        "--tokenizer",
        "-k",
        help="Optional HuggingFace model ID for exact model-aligned BPE token counting (e.g. Qwen/Qwen2.5-1.5B-Instruct)",
    ),
    baseline: Optional[Path] = typer.Option(
        None,
        "--baseline",
        "-b",
        help="Optional baseline JSON report path for KS-test drift detection",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
):
    """Profile dataset quality, token distributions, and failure risks before training."""

    console.print()
    console.print(
        Panel(
            Text(f"🔍 Profiling dataset: {dataset.name}\nType: {dataset_type.upper()} | Max Target Length: {max_seq_len} tokens" + (f" | Tokenizer: {tokenizer}" if tokenizer else "") + (f" | Baseline: {baseline.name}" if baseline else ""), style="bold cyan"),
            title="[bold white]trainsight validator[/bold white]",
            border_style="cyan",
        )
    )

    if dataset_type.lower() == "sft":
        inspector = SFTInspector(max_seq_len_threshold=max_seq_len, tokenizer_name=tokenizer)
        baseline_seqs = None
        if baseline:
            import json
            try:
                with open(baseline, "r") as bf:
                    b_data = json.load(bf)
                    baseline_seqs = b_data.get("seq_lengths", [])
            except Exception:
                pass

        report = inspector.inspect_file(dataset, baseline_seq_lengths=baseline_seqs)

        table = Table(title="📊 SFT Dataset Metrics Summary", border_style="dim")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="bold white")
        table.add_column("Status", style="bold")

        table.add_row("Total Samples", str(report.total_samples), "✅ OK")
        table.add_row("Avg Sequence Length (μ)", f"{report.avg_seq_len:.1f} tokens", "ℹ️ Info")
        table.add_row("Std Dev Length (σ)", f"{report.std_seq_len:.1f} tokens", "ℹ️ Info")
        table.add_row("Variance Ratio (σ/μ)", f"{report.variance_ratio:.3f}", "⚠️ High Variance (>0.75)" if report.variance_ratio > 0.75 else "✅ Normal")
        table.add_row("P95 Sequence Length", f"{report.p95_seq_len:.1f} tokens", "ℹ️ Info")
        table.add_row("P99 Sequence Length", f"{report.p99_seq_len:.1f} tokens", "⚠️ Heavy Tail (>1000)" if report.p99_seq_len > 1000 else "✅ Normal")
        table.add_row("Max Sequence Length", f"{report.max_seq_len} tokens", "❌ Too Long" if report.max_seq_len > max_seq_len else "✅ OK")
        table.add_row("Predicted Padding Waste", f"{report.predicted_padding_waste_pct:.1f}%", "⚠️ High Waste (>30%)" if report.predicted_padding_waste_pct > 30 else "✅ Low Waste")
        table.add_row("Two-Factor Phase Quadrant", report.phase_quadrant, "⚠️ Risk Detected" if report.phase_quadrant != "Clean Execution" else "✅ Clean")
        table.add_row("OOM Risk Samples (>2048)", str(report.oom_risk_count), "❌ High Risk" if report.oom_risk_count > 0 else "✅ None")
        table.add_row("Duplicate Prompts", str(report.duplicate_count), "⚠️ Duplicates" if report.duplicate_count > 0 else "✅ Clean")
        table.add_row("Empty Completions", str(report.empty_completion_count), "❌ Corrupt" if report.empty_completion_count > 0 else "✅ Clean")
        table.add_row("Loss Mask Misalignments", str(report.loss_mask_misaligned_count), "⚠️ Misaligned" if report.loss_mask_misaligned_count > 0 else "✅ Clean")
        if baseline:
            table.add_row("KS Test Drift p-value", f"{report.ks_p_value:.4f}", "⚠️ Drift Alert!" if report.drift_alert else "✅ No Drift")

        console.print(table)
        console.print()

        if report.warnings:
            console.print(Panel("\n".join(report.warnings), title="[bold yellow]⚠️ Detected Issues[/bold yellow]", border_style="yellow"))
            console.print()

        if report.recommendations:
            console.print(Panel("\n".join([f"• {r}" for r in report.recommendations]), title="[bold green]💡 Actionable Recommendations[/bold green]", border_style="green"))
            console.print()

        if not report.warnings:
            console.print("[bold green]✨ SFT Dataset passed all validation checks![/bold green]\n")

        if strict and report.warnings:
            raise typer.Exit(code=1)

    elif dataset_type.lower() == "dpo":
        inspector = DPOInspector()
        report = inspector.inspect_file(dataset)

        table = Table(title="📊 DPO Preference Metrics Summary", border_style="dim")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="bold white")
        table.add_column("Status", style="bold")

        table.add_row("Total Preference Pairs", str(report.total_samples), "✅ OK")
        table.add_row("Avg Chosen Length", f"{report.avg_chosen_len:.1f} tokens", "ℹ️ Info")
        table.add_row("Avg Rejected Length", f"{report.avg_rejected_len:.1f} tokens", "ℹ️ Info")
        table.add_row("Length Bias Ratio (Chosen/Rejected)", f"{report.length_bias_ratio:.2f}", "⚠️ High Bias" if report.length_bias_ratio > 1.8 or report.length_bias_ratio < 0.55 else "✅ Balanced")
        table.add_row("Identical Pairs (Chosen == Rejected)", str(report.identical_pairs_count), "❌ Zero Gradient" if report.identical_pairs_count > 0 else "✅ None")
        table.add_row("Near-Identical Pairs (Sim > 90%)", str(report.near_identical_count), "⚠️ Low Margin" if report.near_identical_count > 0 else "✅ None")
        table.add_row("Preference Reversals (Chosen < Rejected)", str(report.preference_reversal_count), "❌ Inverted Ratings" if report.preference_reversal_count > 0 else "✅ None")
        table.add_row("Duplicate Prompts", str(report.duplicate_count), "⚠️ Duplicates" if report.duplicate_count > 0 else "✅ Clean")
        table.add_row("Missing Text Rows", str(report.empty_pair_count), "❌ Corrupt" if report.empty_pair_count > 0 else "✅ Clean")

        console.print(table)
        console.print()

        if report.warnings:
            console.print(Panel("\n".join(report.warnings), title="[bold yellow]⚠️ Detected Issues[/bold yellow]", border_style="yellow"))
            console.print()

        if report.recommendations:
            console.print(Panel("\n".join([f"• {r}" for r in report.recommendations]), title="[bold green]💡 Actionable Recommendations[/bold green]", border_style="green"))
            console.print()

        if not report.warnings:
            console.print("[bold green]✨ DPO Dataset passed all validation checks![/bold green]\n")

        if strict and report.warnings:
            raise typer.Exit(code=1)

    else:
        console.print(f"[bold red]Dataset type '{dataset_type}' inspector coming up next![/bold red]")


@app.command("fetch")
def fetch(
    dataset_name: str = typer.Option(
        "openai/gsm8k",
        "--repo",
        "-r",
        help="HuggingFace dataset repo (e.g. openai/gsm8k, openbmb/UltraFeedback)",
    ),
    split: str = typer.Option(
        "train[:1000]",
        "--split",
        "-s",
        help="Dataset split selection (e.g. train[:1000])",
    ),
    output: Path = typer.Option(
        Path("sample_data/fetched_dataset.jsonl"),
        "--output",
        "-o",
        help="Output destination path",
    ),
):
    """Fetch real production datasets directly from HuggingFace Hub for inspection."""
    try:
        from datasets import load_dataset
    except ImportError:
        console.print("[bold red]datasets library not installed. Install with: pip install datasets[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"📥 Fetching dataset [bold cyan]{dataset_name}[/bold cyan] ({split}) from HuggingFace Hub...")
    ds = load_dataset(dataset_name, "main" if "gsm8k" in dataset_name else None, split=split)

    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    import json
    with open(output, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row) + "\n")
            count += 1

    console.print(f"[bold green]✨ Successfully downloaded {count} real rows to {output}[/bold green]\n")


if __name__ == "__main__":
    app()
