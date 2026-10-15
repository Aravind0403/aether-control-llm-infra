"""Open-Loop Capacity Calibration for Standalone vLLM.

Calibrates the throughput-latency saturation knee of direct vLLM:
- Uncapped open-loop Poisson arrival schedules (no client concurrency throttling).
- Fixed workload parameters: prompt tokens ~128, max_tokens=64, temperature=0.0.
- Records load-generator submission lag to ensure client-side health.
- Evaluates dual latency targets (e.g. TTFT <= 800ms, E2E <= 1.5s).
- Defines capacity as the highest offered rate meeting >= 95% target attainment.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx
import numpy as np
from rich.console import Console
from rich.table import Table

from vllm_engine.benchmarks.benchmark_serving import ServingBenchmarkRunner, BenchmarkSummary
from vllm_engine.benchmarks.load_generator import (
    run_open_loop_poisson_benchmark,
    generate_synthetic_prompts,
    load_dataset_prompts,
)

console = Console()


async def calibrate_offered_rate(
    backend_url: str,
    model_name: str,
    prompts: List[str],
    target_rps: float,
    num_requests: int = 40,
    max_tokens: int = 64,
    ttft_target_s: float = 0.8,
    e2e_target_s: float = 1.5,
    timeout_s: float = 60.0,
) -> BenchmarkSummary:
    """Runs an open-loop Poisson evaluation at a specified offered arrival rate."""
    runner = ServingBenchmarkRunner(
        api_url=backend_url,
        model_name=model_name,
        slo_target_s=e2e_target_s,
        ttft_target_s=ttft_target_s,
        e2e_target_s=e2e_target_s,
    )

    test_prompts = [prompts[i % len(prompts)] for i in range(num_requests)]
    api_url = f"{backend_url.rstrip('/')}/v1/chat/completions"

    raw_results, duration = await run_open_loop_poisson_benchmark(
        api_url=api_url,
        model_name=model_name,
        prompts=test_prompts,
        target_rps=target_rps,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
    )

    summary = runner.summarize_results(raw_results, duration)
    return summary


async def run_capacity_calibration(
    backend_url: str = "http://localhost:8000",
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    workload_path: Optional[Path] = None,
    rates: Optional[List[float]] = None,
    requests_per_rate: int = 40,
    max_tokens: int = 64,
    ttft_target_s: float = 0.8,
    e2e_target_s: float = 1.5,
    attainment_threshold_pct: float = 95.0,
) -> Dict[str, Any]:
    """Sweeps offered rates to identify the saturation knee and capacity limit."""
    rates = rates or [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]

    # Prepare frozen prompt distribution
    if workload_path and workload_path.exists():
        prompts = load_dataset_prompts(workload_path, max_samples=100)
    else:
        prompts = generate_synthetic_prompts(num_prompts=100, prompt_tokens=128)

    console.print(f"\n[bold cyan]⚡ Initiating Open-Loop Capacity Calibration for {model_name}...[/bold cyan]")
    console.print(f"Backend URL: [yellow]{backend_url}[/yellow]")
    console.print(f"Frozen Workload: Input ~128 tok | Max Output = {max_tokens} tok | Temp = 0.0")
    console.print(f"Latency Targets: TTFT <= {ttft_target_s*1000:.0f}ms | E2E <= {e2e_target_s:.2f}s | Attainment Threshold >= {attainment_threshold_pct:.0f}%\n")

    # Warmup
    console.print("[dim]Executing pre-calibration warmup...[/dim]")
    await calibrate_offered_rate(
        backend_url=backend_url,
        model_name=model_name,
        prompts=prompts[:4],
        target_rps=2.0,
        num_requests=4,
        max_tokens=16,
        ttft_target_s=ttft_target_s,
        e2e_target_s=e2e_target_s,
    )
    await asyncio.sleep(1.0)

    table = Table(
        title="Direct vLLM Capacity Calibration (Open-Loop Poisson Arrivals)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Offered RPS", justify="right", style="cyan")
    table.add_column("Completed", justify="right", style="green")
    table.add_column("Lag P99 (ms)", justify="right", style="dim")
    table.add_column("TTFT P50 (ms)", justify="right")
    table.add_column("TTFT P99 (ms)", justify="right")
    table.add_column("E2E P99 (s)", justify="right")
    table.add_column("Goodput (RPS)", justify="right", style="bold yellow")
    table.add_column("Attainment (%)", justify="right", style="bold green")

    calibration_records = []
    calibrated_capacity_rps = rates[0]

    for rps in rates:
        summary = await calibrate_offered_rate(
            backend_url=backend_url,
            model_name=model_name,
            prompts=prompts,
            target_rps=rps,
            num_requests=requests_per_rate,
            max_tokens=max_tokens,
            ttft_target_s=ttft_target_s,
            e2e_target_s=e2e_target_s,
        )

        record = {
            "offered_rps": rps,
            "requests_offered": summary.offered_requests,
            "requests_completed": summary.completed_requests,
            "submission_lag_p99_ms": summary.submission_lag_p99_ms,
            "ttft_p50_ms": summary.ttft_p50 * 1000.0,
            "ttft_p99_ms": summary.ttft_p99 * 1000.0,
            "tpot_p50_ms": summary.tpot_p50 * 1000.0,
            "tpot_p99_ms": summary.tpot_p99 * 1000.0,
            "e2e_p99_s": summary.e2e_p99,
            "goodput_rps": summary.goodput_rps,
            "goodput_attainment_pct": summary.goodput_attainment_pct,
        }
        calibration_records.append(record)

        attainment_style = "green" if summary.goodput_attainment_pct >= attainment_threshold_pct else "red"
        table.add_row(
            f"{rps:.1f}",
            f"{summary.completed_requests}/{summary.offered_requests}",
            f"{summary.submission_lag_p99_ms:.1f}",
            f"{summary.ttft_p50 * 1000.0:.1f}",
            f"{summary.ttft_p99 * 1000.0:.1f}",
            f"{summary.e2e_p99:.2f}",
            f"{summary.goodput_rps:.2f}",
            f"[{attainment_style}]{summary.goodput_attainment_pct:.1f}%[/{attainment_style}]",
        )

        if summary.goodput_attainment_pct >= attainment_threshold_pct:
            calibrated_capacity_rps = rps

        await asyncio.sleep(1.0)

    console.print(table)
    console.print(f"\n[bold green]🎯 Calibrated System Capacity Limit: {calibrated_capacity_rps:.1f} RPS (Attainment >= {attainment_threshold_pct}%)[/bold green]\n")

    result_payload = {
        "timestamp": time.time(),
        "model": model_name,
        "max_tokens": max_tokens,
        "ttft_target_s": ttft_target_s,
        "e2e_target_s": e2e_target_s,
        "attainment_threshold_pct": attainment_threshold_pct,
        "calibrated_capacity_rps": calibrated_capacity_rps,
        "records": calibration_records,
    }

    out_file = Path("benchmarks/reports/capacity_calibration.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(result_payload, f, indent=2)

    console.print(f"Saved calibration data to: [green]{out_file}[/green]\n")
    return result_payload


if __name__ == "__main__":
    asyncio.run(run_capacity_calibration())
