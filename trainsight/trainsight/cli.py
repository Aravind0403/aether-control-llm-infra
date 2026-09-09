from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from trainsight.inspectors.sft_inspector import SFTInspector
from trainsight.inspectors.dpo_inspector import DPOInspector
from trainsight.simulators.collation_simulator import CollationSimulator
from trainsight.profilers.activation_profiler import (
    ActivationProfiler,
    ModelArchitectureSpec,
    DistributedStrategyConfig,
    HARDWARE_PRESETS,
)
from trainsight.profilers.meta_probe import SingleLayerMetaProbe
from trainsight.profilers.mbom import MBOMGenerator
from trainsight.profilers.efficiency import EfficiencyScorer
from trainsight.remediators.bin_packer import SequenceBinPacker
from trainsight.cache.signature import FastInvariantHasher
from trainsight.resilience.calibration import CalibrationMatrixManager
from trainsight.resilience.circuit_breaker import StorageCircuitBreaker

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
    runtime: bool = typer.Option(
        False,
        "--runtime",
        "-r",
        help="Enable execution-aware dynamic memory simulation & MBOM diagnostics",
    ),
    model_id: str = typer.Option(
        "Qwen/Qwen2.5-1.5B",
        "--model-id",
        "-M",
        help="Target HuggingFace model architecture or local config.json",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-B",
        help="Target micro-batch size for collation and dynamic activation profiling",
    ),
    world_size: int = typer.Option(
        8,
        "--world-size",
        "-w",
        help="Target distributed cluster GPU count for NCCL / ZeRO-3 buffer calculation",
    ),
    hardware: str = typer.Option(
        "l4-24gb",
        "--hardware",
        help="Target hardware preset (l4-24gb, a100-80gb, h100-80gb) or VRAM in GB",
    ),
    gradient_checkpointing: bool = typer.Option(
        True,
        "--gradient-checkpointing/--no-gradient-checkpointing",
        help="Toggle selective activation recomputation",
    ),
    optimizer: str = typer.Option(
        "adamw",
        "--optimizer",
        help="Optimizer state precision: adamw (FP32), adamw_8bit, sgd",
    ),
    deepspeed_config: Optional[Path] = typer.Option(
        None,
        "--deepspeed-config",
        help="Optional DeepSpeed JSON configuration path",
    ),
    auto_pack: Optional[Path] = typer.Option(
        None,
        "--auto-pack",
        help="Generate length-bucketed sequence packing index JSON file to eliminate padding waste",
    ),
    auto_scale: bool = typer.Option(
        False,
        "--auto-scale",
        help="Evaluate idle VRAM and provide proactive upward batch size acceleration advice",
    ),
    risk_accept: bool = typer.Option(
        False,
        "--risk-accept",
        help="Acknowledge non-zero OOM risk and allow pod execution with canary telemetry annotation",
    ),
    cache_dir: Optional[Path] = typer.Option(
        None,
        "--cache-dir",
        help="Directory for Tier 1 ~20KB dataset signature caching",
    ),
):
    """Profile dataset quality, token distributions, and failure risks before training."""

    console.print()
    console.print(
        Panel(
            Text(
                f"🔍 Profiling dataset: {dataset.name}\n"
                f"Type: {dataset_type.upper()} | Max Target Length: {max_seq_len} tokens"
                + (f" | Tokenizer: {tokenizer}" if tokenizer else "")
                + (f" | Model: {model_id} (BS: {batch_size}, GPUs: {world_size})" if runtime else "")
                + (f" | Baseline: {baseline.name}" if baseline else ""),
                style="bold cyan",
            ),
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

        # Tier 1 Caching Check
        if cache_dir and report.seq_lengths:
            sig = FastInvariantHasher.create_signature(dataset, report.seq_lengths, tokenizer_id=tokenizer)
            sig_file = cache_dir / f"{dataset.stem}_sig_{sig.signature_key[:12]}.json"
            FastInvariantHasher.save_signature_file(sig, sig_file)
            console.print(f"[dim]💾 Cached Tier 1 dataset signature to: {sig_file.name} (~{sig_file.stat().st_size / 1024:.1f} KB)[/dim]\n")

        # Execution Simulation (Collation + Activation Math + MBOM + Efficiency)
        mem_fits = True
        if (runtime or auto_pack or auto_scale) and report.seq_lengths:
            # 1. Collation Simulation
            col_sim = CollationSimulator(num_trials=1000, seed=42)
            col_res = col_sim.simulate(report.seq_lengths, batch_size=batch_size)

            col_table = Table(title=f"🎲 Monte-Carlo Batch Collation Profile (Batch Size: {batch_size})", border_style="magenta")
            col_table.add_column("Collation Parameter", style="magenta", no_wrap=True)
            col_table.add_column("Simulated Value", style="bold white")
            col_table.add_column("Collation Impact", style="bold")

            col_table.add_row("Worst-Case Batch Shape", f"[{batch_size}, {col_res.worst_case_seq_len}]", "⚠️ Peak Activation Tensor")
            col_table.add_row("Peak Batch Tokens (b × s)", f"{col_res.worst_case_batch_tokens:,} tokens", "⚠️ Dynamic Token Spike")
            col_table.add_row("Average Batch Tokens", f"{col_res.avg_batch_tokens:,.0f} tokens", "ℹ️ Expected Mean")
            col_table.add_row("P99 Batch Tokens", f"{col_res.p99_batch_tokens:,.0f} tokens", "ℹ️ Tail Bound")
            col_table.add_row("Collation Padding Waste", f"{col_res.padding_waste_pct:.1f}%", "⚠️ Zero-Value Compute" if col_res.padding_waste_pct > 25.0 else "✅ Efficient")
            console.print(col_table)
            console.print()

            # 2. Hardware VRAM & Calibration
            hw_key = hardware.lower().strip()
            vram_limit = HARDWARE_PRESETS.get(hw_key)
            if vram_limit is None:
                try:
                    vram_limit = float(hardware.replace("gb", "").replace("GB", "").strip())
                except ValueError:
                    vram_limit = 24.0

            cal_mgr = CalibrationMatrixManager()
            usable_vram = cal_mgr.get_usable_vram(vram_limit)
            if not cal_mgr.is_calibrated():
                console.print(f"[bold yellow]⚠️ Uncalibrated runtime detected. Applied 15% uncertainty headroom buffer: {usable_vram:.1f} GB usable on {vram_limit:.1f} GB GPU.[/bold yellow]\n")

            # 3. Model Architecture & Activation Profiler
            model_spec = ModelArchitectureSpec.from_model_id_or_config(model_id)
            if deepspeed_config:
                dist_cfg = DistributedStrategyConfig.from_deepspeed_json(deepspeed_config, world_size=world_size)
            else:
                dist_cfg = DistributedStrategyConfig(strategy="zero3" if world_size > 1 else "none", world_size=world_size)

            # Single-Layer Meta Probe fast test
            meta_probe = SingleLayerMetaProbe()
            probe_res = meta_probe.probe(model_spec)

            act_prof = ActivationProfiler(model_spec)
            mem_res = act_prof.profile_run(
                batch_size=batch_size,
                worst_case_seq_len=col_res.worst_case_seq_len,
                hardware_vram_gb=usable_vram,
                optimizer_type=optimizer,
                gradient_checkpointing=gradient_checkpointing,
                dist_config=dist_cfg,
            )
            mem_fits = mem_res.fits_in_vram

            # 4. MBOM Pedagogy Generation
            mbom_gen = MBOMGenerator()
            mbom_receipt = mbom_gen.generate(
                job_name=f"{model_spec.name} @ BS={batch_size}",
                mem_result=mem_res,
                collation_result=col_res,
                hardware_name=f"{hardware.upper()} ({vram_limit} GB)",
                total_steps=1000,
            )

            mbom_table = Table(title=f"🧾 Memory Bill of Materials (MBOM) | Hardware: {hardware.upper()}", border_style="cyan")
            mbom_table.add_column("VRAM Tenant Component", style="cyan", no_wrap=True)
            mbom_table.add_column("Footprint", style="bold white")
            mbom_table.add_column("% of VRAM", style="bold")
            mbom_table.add_column("Silicon Status / Explanation", style="dim")

            for c in mbom_receipt.components:
                mbom_table.add_row(c["name"], f"{c['footprint_gb']:.2f} GB", f"{c['pct_of_vram']:.1f}%", c["status"])

            mbom_table.add_row(
                "TOTAL PREDICTED PEAK",
                f"{mem_res.total_peak_gb:.2f} GB",
                f"{mem_res.utilization_pct:.1f}%",
                "✅ FITS SAFE" if mem_res.fits_in_vram else f"❌ OOM EXCEEDS BY {abs(mem_res.vram_headroom_gb):.2f} GB",
            )
            console.print(mbom_table)
            console.print()

            # MBOM Pedagogy Box
            pedagogy_text = f"[bold white]{mbom_receipt.pedagogy_title}[/bold white]\n\n{mbom_receipt.pedagogy_explanation}"
            if mbom_receipt.recommendations:
                pedagogy_text += "\n\n[bold green]Recommended Fixes:[/bold green]\n" + "\n".join([f"👉 {r}" for r in mbom_receipt.recommendations])
            if mbom_receipt.seed_delta_explanation:
                pedagogy_text += f"\n\n[bold yellow]{mbom_receipt.seed_delta_explanation}[/bold yellow]"

            console.print(Panel(pedagogy_text, title="[bold yellow]💡 Staff Engineering Diagnostic & Teaching Moment[/bold yellow]", border_style="yellow"))
            console.print()

            # 5. Efficiency Scorer (CES)
            eff_scorer = EfficiencyScorer()
            eff_grade = eff_scorer.evaluate(mem_res, collation_result=col_res, current_batch_size=batch_size)

            eff_table = Table(title="📈 Compute Efficiency Score (CES)", border_style="blue")
            eff_table.add_column("Efficiency Metric", style="blue")
            eff_table.add_column("Score", style="bold white")
            eff_table.add_column("FinOps Status", style="bold")

            eff_table.add_row("CES Overall Grade (0-100)", f"{eff_grade.score:.1f} / 100", f"Grade {eff_grade.letter_grade}")
            eff_table.add_row("VRAM Occupancy", f"{eff_grade.vram_occupancy_pct:.1f}%", "Occupied Silicon")
            eff_table.add_row("Padding Efficiency", f"{eff_grade.padding_efficiency_pct:.1f}%", "Non-Padded Compute")
            console.print(eff_table)
            console.print(f"[dim]{eff_grade.action_message}[/dim]\n")

            # 6. Auto-Pack Sequence Bin-Packer
            if auto_pack:
                packer = SequenceBinPacker(batch_size=batch_size)
                pack_res = packer.pack(report.seq_lengths, batch_size=batch_size)
                packer.write_packed_index_file(pack_res, auto_pack)

                pack_panel = (
                    f"✅ Sequence Bin-Packing Index Generated: [bold white]{auto_pack.name}[/bold white]\n"
                    f"• Padding waste reduced: {pack_res.original_padding_waste_pct:.1f}% ──► {pack_res.packed_padding_waste_pct:.1f}% "
                    f"(-{pack_res.waste_reduction_pct:.1f}% waste eliminated)\n"
                    f"• Estimated Training Speedup: [bold green]+{pack_res.estimated_throughput_gain_pct:.1f}% throughput[/bold green]\n"
                    f"• Invariant-Preserving Hyperparameter Translation: per_device_train_batch_size={pack_res.recommended_per_device_batch_size}, "
                    f"gradient_accumulation_steps={pack_res.recommended_gradient_accumulation_steps}"
                )
                console.print(Panel(pack_panel, title="[bold green]📦 Pre-Flight Sequence Bin-Packer (--auto-pack)[/bold green]", border_style="green"))
                console.print()

            # Fail-fast OOM verdict
            if not mem_fits:
                if risk_accept:
                    console.print(
                        "[bold yellow]⚠️ Signed Risk Accepted (--risk-accept): Pod canary status set "
                        "(trainsight.risk/accepted: true). Proceeding with scheduling.[/bold yellow]\n"
                    )
                else:
                    console.print(
                        f"[bold red]❌ PRE-FLIGHT DETERMINISTIC OOM HALT: Predicted peak allocation ({mem_res.total_peak_gb:.2f} GB) "
                        f"exceeds usable hardware capacity ({usable_vram:.2f} GB) by {abs(mem_res.vram_headroom_gb):.2f} GB!\n"
                        f"Apply the MBOM recommendations above or run with --auto-pack / --risk-accept to schedule.[/bold red]\n"
                    )
                    raise typer.Exit(code=1)

        if report.warnings:
            console.print(Panel("\n".join(report.warnings), title="[bold yellow]⚠️ Detected Issues[/bold yellow]", border_style="yellow"))
            console.print()

        if report.recommendations and not runtime:
            console.print(Panel("\n".join([f"• {r}" for r in report.recommendations]), title="[bold green]💡 Actionable Recommendations[/bold green]", border_style="green"))
            console.print()

        if not report.warnings and mem_fits:
            console.print("[bold green]✨ SFT Dataset passed all validation and runtime checks![/bold green]\n")

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
