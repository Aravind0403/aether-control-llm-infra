"""Direct vLLM Repeatability & Outlier Diagnostic Suite.

Protocol:
1. Backend: Direct vLLM only (standalone, unproxied).
2. Workload: Fixed tokenized prompt length and verified output length (min_tokens=max_tokens=64).
3. Arrival Process: 6 RPS Poisson over a fixed 180-second window (~1,080 requests per run).
4. Repetitions: 3 consecutive runs replaying the identical pre-generated trace.
5. Preparation: Warm-up before trials; inter-trial cooldown and drain.
6. Evidence:
   - Full per-request timing (scheduled, dispatch, TTFT, TPOT, E2E).
   - Dual perspectives: from_schedule (includes client lag) and from_dispatch (raw serving latency).
   - Actual prompt & completion token counts from returned usage payloads.
   - Goodput strictly calculated using the fixed 180.0s window denominator:
       Goodput = (Compliant requests within both TTFT and E2E targets) / 180.0
   - Attainment % = (Compliant requests / Total offered requests) * 100%.
   - Full persistence to benchmarks/reports/repeatability_diagnostic_report.json.
"""

import sys
import time
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx
import numpy as np
from rich.console import Console
from rich.table import Table

from vllm_engine.benchmarks.benchmark_serving import ServingBenchmarkRunner, BenchmarkSummary
from vllm_engine.benchmarks.load_generator import (
    run_open_loop_poisson_benchmark,
    generate_poisson_arrival_schedule_window,
    generate_synthetic_prompts,
    load_dataset_prompts,
)

console = Console()


async def run_single_diagnostic_trial(
    trial_idx: int,
    backend_url: str,
    model_name: str,
    prompts: List[str],
    arrival_schedule: List[float],
    window_duration_s: float,
    max_tokens: int = 64,
    min_tokens: int = 64,
    ttft_target_s: float = 0.25,
    e2e_target_s: float = 2.50,
) -> BenchmarkSummary:
    """Executes a single 180-second open-loop Poisson arrival run against direct vLLM."""
    runner = ServingBenchmarkRunner(
        api_url=backend_url,
        model_name=model_name,
        slo_target_s=e2e_target_s,
        ttft_target_s=ttft_target_s,
        e2e_target_s=e2e_target_s,
    )

    num_reqs = len(arrival_schedule)
    test_prompts = [prompts[i % len(prompts)] for i in range(num_reqs)]
    target_url = f"{backend_url.rstrip('/')}/v1/chat/completions"

    raw_results, duration = await run_open_loop_poisson_benchmark(
        api_url=target_url,
        model_name=model_name,
        prompts=test_prompts,
        target_rps=float(len(arrival_schedule)) / window_duration_s,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        arrival_schedule=arrival_schedule,
    )

    summary = runner.summarize_results(
        results=raw_results,
        total_duration_s=duration,
        arrival_window_s=window_duration_s,
    )
    return summary


async def run_repeatability_diagnostic(
    backend_url: str = "http://localhost:8000",
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    window_duration_s: float = 180.0,
    target_rps: float = 6.0,
    num_runs: int = 3,
    max_tokens: int = 64,
    min_tokens: int = 64,
    ttft_target_s: float = 0.25,
    e2e_target_s: float = 2.50,
    cooldown_s: float = 15.0,
    trace_seed: int = 42,
    workload_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Runs 3 repeated trials replaying the identical 180s arrival trace to diagnose stability."""
    # 1. Generate frozen prompt distribution
    if workload_path and workload_path.exists():
        prompts = load_dataset_prompts(workload_path, max_samples=100)
    else:
        # Fixed 128-token prompt text for complete reproducibility
        prompts = generate_synthetic_prompts(num_prompts=100, prompt_tokens=128)

    # 2. Generate deterministic Poisson arrival trace over the fixed [0, window_duration_s) window
    shared_schedule = generate_poisson_arrival_schedule_window(
        duration_s=window_duration_s,
        target_rps=target_rps,
        seed=trace_seed,
    )
    num_requests = len(shared_schedule)

    console.print(f"\n[bold cyan]🔬 Direct vLLM Repeatability & Stability Diagnostic[/bold cyan]")
    console.print(f"Backend: [yellow]{backend_url}[/yellow] | Model: [yellow]{model_name}[/yellow]")
    console.print(f"Fixed Window: [yellow]{window_duration_s:.1f}s[/yellow] | Target Rate: [yellow]{target_rps:.1f} RPS[/yellow]")
    console.print(f"Scheduled Requests per Run: [green]{num_requests}[/green] (Trace Seed: {trace_seed})")
    console.print(f"Verified Generation: min_tokens={min_tokens}, max_tokens={max_tokens}, temperature=0.0")
    console.print(f"Latency Targets: TTFT <= {ttft_target_s*1000:.0f}ms | E2E <= {e2e_target_s:.2f}s | Fixed Denominator = {window_duration_s:.1f}s\n")

    # 3. Warm-up
    console.print("[dim]Executing pre-trial warm-up (10 requests)...[/dim]")
    warmup_schedule = generate_poisson_arrival_schedule_window(duration_s=5.0, target_rps=2.0, seed=999)
    await run_single_diagnostic_trial(
        trial_idx=0,
        backend_url=backend_url,
        model_name=model_name,
        prompts=prompts[:10],
        arrival_schedule=warmup_schedule[:10],
        window_duration_s=5.0,
        max_tokens=16,
        min_tokens=16,
        ttft_target_s=ttft_target_s,
        e2e_target_s=e2e_target_s,
    )
    console.print("[dim]Warm-up finished. Draining backend before trial 1...[/dim]")
    await asyncio.sleep(5.0)

    # 4. Execute 3 identical runs
    run_summaries: List[BenchmarkSummary] = []
    trial_records = []

    table = Table(
        title=f"Direct vLLM Repeatability Runs ({num_runs} Trials, Identical 180s Trace @ 6 RPS)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Run #", justify="center", style="cyan")
    table.add_column("Completed", justify="right", style="green")
    table.add_column("Tokens (Mean)", justify="right")
    table.add_column("Sub. Lag P99 (ms)", justify="right", style="dim")
    table.add_column("TTFT P50 (ms)", justify="right")
    table.add_column("TTFT P99 (ms)", justify="right")
    table.add_column("E2E P50 (s)", justify="right")
    table.add_column("E2E P99 (s)", justify="right")
    table.add_column("Goodput (RPS)", justify="right", style="bold yellow")
    table.add_column("Attainment (%)", justify="right", style="bold green")

    for run_idx in range(num_runs):
        console.print(f"[bold magenta]▶ Executing Run {run_idx+1}/{num_runs}...[/bold magenta]")
        summary = await run_single_diagnostic_trial(
            trial_idx=run_idx + 1,
            backend_url=backend_url,
            model_name=model_name,
            prompts=prompts,
            arrival_schedule=shared_schedule,
            window_duration_s=window_duration_s,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            ttft_target_s=ttft_target_s,
            e2e_target_s=e2e_target_s,
        )
        run_summaries.append(summary)

        # Verify output tokens from returned usage
        comp_tokens = [r["completion_tokens"] for r in summary.raw_records if r.get("completion_tokens")]
        mean_tokens = float(np.mean(comp_tokens)) if comp_tokens else 0.0

        table.add_row(
            f"Run {run_idx+1}",
            f"{summary.completed_requests}/{summary.offered_requests}",
            f"{mean_tokens:.1f}",
            f"{summary.submission_lag_p99_ms:.1f}",
            f"{summary.ttft_p50 * 1000.0:.1f}",
            f"{summary.ttft_p99 * 1000.0:.1f}",
            f"{summary.e2e_p50:.2f}",
            f"{summary.e2e_p99:.2f}",
            f"{summary.goodput_rps:.2f}",
            f"{summary.goodput_attainment_pct:.1f}%",
        )

        trial_records.append({
            "run_index": run_idx + 1,
            "offered": summary.offered_requests,
            "completed": summary.completed_requests,
            "failed": summary.failed_requests,
            "mean_completion_tokens": mean_tokens,
            "submission_lag_p50_ms": summary.submission_lag_p50_ms,
            "submission_lag_p99_ms": summary.submission_lag_p99_ms,
            "ttft_p50_ms": summary.ttft_p50 * 1000.0,
            "ttft_p99_ms": summary.ttft_p99 * 1000.0,
            "e2e_p50_s": summary.e2e_p50,
            "e2e_p99_s": summary.e2e_p99,
            "goodput_rps": summary.goodput_rps,
            "goodput_attainment_pct": summary.goodput_attainment_pct,
            "raw_records": summary.raw_records,
        })

        if run_idx < num_runs - 1 and cooldown_s > 0:
            console.print(f"[dim]Draining and cooling down for {cooldown_s:.1f}s before next run...[/dim]")
            await asyncio.sleep(cooldown_s)

    console.print(table)

    # 5. Cross-trial stability statistics (Mean ± SD across the 3 runs)
    goodputs = [s.goodput_rps for s in run_summaries]
    attainments = [s.goodput_attainment_pct for s in run_summaries]
    ttft_p99s = [s.ttft_p99 * 1000.0 for s in run_summaries]
    e2e_p99s = [s.e2e_p99 for s in run_summaries]
    lags = [s.submission_lag_p99_ms for s in run_summaries]

    def _mean_sd(vals):
        return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    gp_m, gp_s = _mean_sd(goodputs)
    att_m, att_s = _mean_sd(attainments)
    ttft_m, ttft_s = _mean_sd(ttft_p99s)
    e2e_m, e2e_s = _mean_sd(e2e_p99s)
    lag_m, lag_s = _mean_sd(lags)

    console.print(f"\n[bold]Cross-Run Stability Metrics (Mean ± Sample SD, N={num_runs}):[/bold]")
    console.print(f"  Goodput:        [yellow]{gp_m:.2f} ± {gp_s:.2f} RPS[/yellow]")
    console.print(f"  Attainment:     [yellow]{att_m:.1f}% ± {att_s:.1f}%[/yellow]")
    console.print(f"  TTFT P99:       [yellow]{ttft_m:.1f} ± {ttft_s:.1f} ms[/yellow]")
    console.print(f"  E2E P99:        [yellow]{e2e_m:.2f} ± {e2e_s:.2f} s[/yellow]")
    console.print(f"  Sub. Lag P99:   [dim]{lag_m:.1f} ± {lag_s:.1f} ms[/dim]\n")

    report_payload = {
        "timestamp": time.time(),
        "backend": backend_url,
        "model": model_name,
        "window_duration_s": window_duration_s,
        "target_rps": target_rps,
        "num_runs": num_runs,
        "requests_per_run": num_requests,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "ttft_target_s": ttft_target_s,
        "e2e_target_s": e2e_target_s,
        "summary": {
            "goodput_rps_mean": gp_m,
            "goodput_rps_sd": gp_s,
            "attainment_pct_mean": att_m,
            "attainment_pct_sd": att_s,
            "ttft_p99_ms_mean": ttft_m,
            "ttft_p99_ms_sd": ttft_s,
            "e2e_p99_s_mean": e2e_m,
            "e2e_p99_s_sd": e2e_s,
            "submission_lag_p99_ms_mean": lag_m,
            "submission_lag_p99_ms_sd": lag_s,
        },
        "trials": trial_records,
    }

    out_file = output_path or Path("benchmarks/reports/repeatability_diagnostic_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report_payload, f, indent=2)

    console.print(f"Saved complete diagnostic audit to: [green]{out_file}[/green]\n")
    return report_payload


if __name__ == "__main__":
    asyncio.run(run_repeatability_diagnostic())
