"""Open-Loop Capacity Calibration for Standalone vLLM.

Calibrates the throughput-latency saturation knee of direct vLLM:
- Uncapped open-loop Poisson arrival schedules over fixed observation windows (e.g. W=60s).
- Active warmup burst (e.g. 10 requests) applied before every rate trial and after idle intervals.
- Fixed workload parameters: prompt tokens ~128, min_tokens=max_tokens=64, temperature=0.0.
- Synchronous background GPU telemetry logging (clocks, power, temp, utilization) via nvidia-smi.
- Dual latency perspectives: TTFT/E2E from dispatch (raw server) and from schedule (including client lag).
- Identifies empirical saturation boundary by sweeping candidate rates (e.g. 6, 10, 14, 18 RPS).
"""

import asyncio
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import httpx
import numpy as np
from rich.console import Console
from rich.table import Table

from vllm_engine.benchmarks.benchmark_serving import ServingBenchmarkRunner, BenchmarkSummary
from vllm_engine.benchmarks.load_generator import (
    run_open_loop_poisson_benchmark,
    generate_synthetic_prompts,
    load_dataset_prompts,
    generate_poisson_arrival_schedule_window,
    async_send_streaming_request,
)

console = Console()


async def sample_gpu_telemetry(interval_s: float = 0.5) -> List[Dict[str, Any]]:
    """Background task to poll nvidia-smi if available."""
    samples = []
    has_nvidia = shutil.which("nvidia-smi") is not None
    if not has_nvidia:
        return samples

    query_fields = "timestamp,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem"
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query_fields}",
        "--format=csv,noheader,nounits",
    ]

    try:
        while True:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0:
                line = stdout.decode().strip()
                if line:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 9:
                        samples.append({
                            "timestamp": time.time(),
                            "raw_time": parts[0],
                            "pstate": parts[1],
                            "gpu_temp_c": float(parts[2]) if parts[2] != "[Not Supported]" else None,
                            "gpu_util_pct": float(parts[3]) if parts[3] != "[Not Supported]" else None,
                            "mem_util_pct": float(parts[4]) if parts[4] != "[Not Supported]" else None,
                            "mem_used_mib": float(parts[5]) if parts[5] != "[Not Supported]" else None,
                            "power_w": float(parts[6]) if parts[6] != "[Not Supported]" else None,
                            "clock_sm_mhz": float(parts[7]) if parts[7] != "[Not Supported]" else None,
                            "clock_mem_mhz": float(parts[8]) if parts[8] != "[Not Supported]" else None,
                        })
            await asyncio.sleep(interval_s)
    except asyncio.CancelledError:
        pass
    except Exception:
        pass
    return samples


async def execute_standard_warmup(
    backend_url: str,
    model_name: str,
    num_warmup_requests: int = 10,
    warmup_tokens: int = 32,
):
    """Sends a standardized active warm-up burst outside the evaluation window."""
    target_url = f"{backend_url.rstrip('/')}/v1/chat/completions"
    console.print(f"  [dim]Applying standardized active warm-up ({num_warmup_requests} requests)...[/dim]")
    try:
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=40)
        async with httpx.AsyncClient(timeout=15.0, limits=limits) as client:
            tasks = []
            for i in range(num_warmup_requests):
                t = async_send_streaming_request(
                    client=client,
                    api_url=target_url,
                    model_name=model_name,
                    prompt=f"Standard warm-up prompt #{i+1} for engine stabilization.",
                    max_tokens=warmup_tokens,
                    min_tokens=warmup_tokens,
                )
                tasks.append(t)
            await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        console.print(f"  [dim yellow]Warning during warm-up: {e}[/dim yellow]")
    await asyncio.sleep(1.0)


async def calibrate_offered_rate(
    backend_url: str,
    model_name: str,
    prompts: List[str],
    target_rps: float,
    window_duration_s: float = 60.0,
    seed: int = 42,
    min_tokens: int = 64,
    max_tokens: int = 64,
    ttft_target_s: float = 0.25,
    e2e_target_s: float = 2.50,
    timeout_s: float = 60.0,
) -> Tuple[BenchmarkSummary, List[Dict[str, Any]]]:
    """Runs a fixed-window open-loop Poisson evaluation at a specified offered arrival rate."""
    runner = ServingBenchmarkRunner(
        api_url=backend_url,
        model_name=model_name,
        slo_target_s=e2e_target_s,
        ttft_target_s=ttft_target_s,
        e2e_target_s=e2e_target_s,
    )

    # Pre-generate Poisson arrival schedule strictly within [0, window_duration_s)
    schedule = generate_poisson_arrival_schedule_window(
        duration_s=window_duration_s,
        target_rps=target_rps,
        seed=seed,
    )
    num_requests = len(schedule)
    test_prompts = [prompts[i % len(prompts)] for i in range(num_requests)]
    api_url = f"{backend_url.rstrip('/')}/v1/chat/completions"

    # Start background GPU telemetry
    telemetry_samples = []
    telemetry_task = None
    if shutil.which("nvidia-smi"):
        telemetry_task = asyncio.create_task(sample_gpu_telemetry(interval_s=0.5))

    try:
        raw_results, duration = await run_open_loop_poisson_benchmark(
            api_url=api_url,
            model_name=model_name,
            prompts=test_prompts,
            target_rps=target_rps,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            arrival_schedule=schedule,
            timeout_s=timeout_s,
        )
    finally:
        if telemetry_task:
            telemetry_task.cancel()
            try:
                telemetry_samples = await telemetry_task
            except asyncio.CancelledError:
                pass

    summary = runner.summarize_results(raw_results, duration, arrival_window_s=window_duration_s)
    return summary, telemetry_samples


async def run_capacity_calibration(
    backend_url: str = "http://localhost:8000",
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    workload_path: Optional[Path] = None,
    rates: Optional[List[float]] = None,
    requests_per_rate: Optional[int] = None,
    window_duration_s: float = 60.0,
    min_tokens: int = 64,
    max_tokens: int = 64,
    ttft_target_s: float = 0.25,
    e2e_target_s: float = 2.50,
    attainment_threshold_pct: float = 95.0,
    cooldown_s: float = 5.0,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Sweeps candidate offered rates using fixed-duration windows to identify saturation boundary."""
    rates = rates or [6.0, 10.0, 14.0, 18.0]
    if requests_per_rate is not None and window_duration_s == 60.0:
        window_duration_s = max(float(requests_per_rate) / (rates[0] if rates else 10.0), 1.0)

    # Prepare frozen prompt distribution
    if workload_path and workload_path.exists():
        prompts = load_dataset_prompts(workload_path, max_samples=100)
    else:
        prompts = generate_synthetic_prompts(num_prompts=100, prompt_tokens=128)

    console.print(f"\n[bold cyan]⚡ Initiating Direct vLLM Capacity Recalibration...[/bold cyan]")
    console.print(f"Backend URL: [yellow]{backend_url}[/yellow] | Model: [yellow]{model_name}[/yellow]")
    console.print(f"Window / Rate: {window_duration_s:.1f}s | Output: {min_tokens}..{max_tokens} tokens")
    console.print(f"Candidate Rates: {rates}")
    console.print(f"SLO Targets: TTFT <= {ttft_target_s*1000:.0f}ms | E2E <= {e2e_target_s:.2f}s | Attainment >= {attainment_threshold_pct:.0f}%\n")

    table = Table(
        title="Direct vLLM Capacity Calibration (Fixed-Window Open-Loop Poisson)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Offered RPS", justify="right", style="cyan")
    table.add_column("Completed", justify="right", style="green")
    table.add_column("Lag P99 (ms)", justify="right", style="dim")
    table.add_column("TTFT P50 (ms)", justify="right")
    table.add_column("TTFT P99 (ms)", justify="right")
    table.add_column("E2E P50 (s)", justify="right")
    table.add_column("E2E P99 (s)", justify="right")
    table.add_column("Goodput (RPS)", justify="right", style="bold yellow")
    table.add_column("Attainment (%)", justify="right", style="bold green")

    calibration_records = []
    calibrated_capacity_rps = rates[0]
    first_failure_rate = None

    for idx, rps in enumerate(rates):
        console.print(f"[bold cyan]▶ Testing Offered Rate: {rps:.1f} RPS (Window: {window_duration_s}s)...[/bold cyan]")
        
        # 1. Standardized active warmup before EVERY rate trial (after idle)
        await execute_standard_warmup(backend_url, model_name, num_warmup_requests=10, warmup_tokens=32)

        # 2. Execute fixed-window evaluation
        summary, telemetry = await calibrate_offered_rate(
            backend_url=backend_url,
            model_name=model_name,
            prompts=prompts,
            target_rps=rps,
            window_duration_s=window_duration_s,
            seed=42 + int(rps * 10),
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            ttft_target_s=ttft_target_s,
            e2e_target_s=e2e_target_s,
        )

        mean_tokens = float(np.mean([r.get("completion_tokens", 0) for r in summary.raw_records])) if summary.raw_records else 0.0
        record = {
            "offered_rps": rps,
            "window_duration_s": window_duration_s,
            "requests_offered": summary.offered_requests,
            "requests_completed": summary.completed_requests,
            "requests_failed": summary.failed_requests,
            "mean_completion_tokens": mean_tokens,
            "submission_lag_p50_ms": summary.submission_lag_p50_ms,
            "submission_lag_p99_ms": summary.submission_lag_p99_ms,
            "ttft_p50_ms": summary.ttft_p50 * 1000.0,
            "ttft_p99_ms": summary.ttft_p99 * 1000.0,
            "tpot_p50_ms": summary.tpot_p50 * 1000.0,
            "tpot_p99_ms": summary.tpot_p99 * 1000.0,
            "e2e_p50_s": summary.e2e_p50,
            "e2e_p99_s": summary.e2e_p99,
            "goodput_rps": summary.goodput_rps,
            "goodput_attainment_pct": summary.goodput_attainment_pct,
            "gpu_telemetry_samples": len(telemetry),
            "raw_records": summary.raw_records,
        }
        calibration_records.append(record)

        attainment_style = "green" if summary.goodput_attainment_pct >= attainment_threshold_pct else "red"
        table.add_row(
            f"{rps:.1f}",
            f"{summary.completed_requests}/{summary.offered_requests}",
            f"{summary.submission_lag_p99_ms:.1f}",
            f"{summary.ttft_p50 * 1000.0:.1f}",
            f"{summary.ttft_p99 * 1000.0:.1f}",
            f"{summary.e2e_p50:.2f}",
            f"{summary.e2e_p99:.2f}",
            f"{summary.goodput_rps:.2f}",
            f"[{attainment_style}]{summary.goodput_attainment_pct:.1f}%[/{attainment_style}]",
        )

        if summary.goodput_attainment_pct >= attainment_threshold_pct:
            calibrated_capacity_rps = rps
        elif first_failure_rate is None:
            first_failure_rate = rps

        await asyncio.sleep(cooldown_s)

    console.print(table)
    console.print(f"\n[bold green]🎯 Highest Attaining Rate: {calibrated_capacity_rps:.1f} RPS (Attainment >= {attainment_threshold_pct}%)[/bold green]")
    if first_failure_rate:
        console.print(f"[bold red]⚠️ First Attainment Failure Boundary: {first_failure_rate:.1f} RPS[/bold red]\n")
    else:
        console.print("[bold yellow]ℹ️ No attainment failure observed up to maximum tested rate.[/bold yellow]\n")

    result_payload = {
        "timestamp": time.time(),
        "backend": backend_url,
        "model": model_name,
        "window_duration_s": window_duration_s,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "ttft_target_s": ttft_target_s,
        "e2e_target_s": e2e_target_s,
        "attainment_threshold_pct": attainment_threshold_pct,
        "calibrated_capacity_rps": calibrated_capacity_rps,
        "first_failure_boundary_rps": first_failure_rate,
        "records": calibration_records,
    }

    out_file = output_path or Path("benchmarks/reports/capacity_calibration.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(result_payload, f, indent=2)

    console.print(f"Saved calibration artifact to: [green]{out_file}[/green]\n")
    return result_payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Recalibrate direct vLLM serving capacity.")
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--rates", nargs="+", type=float, default=[6.0, 10.0, 14.0, 18.0])
    parser.add_argument("--window-duration-s", type=float, default=60.0)
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--ttft-target-s", type=float, default=0.25)
    parser.add_argument("--e2e-target-s", type=float, default=2.50)
    parser.add_argument("--cooldown-s", type=float, default=5.0)
    parser.add_argument("--output", default="benchmarks/reports/capacity_calibration.json")
    args = parser.parse_args()

    asyncio.run(run_capacity_calibration(
        backend_url=args.backend_url,
        model_name=args.model_name,
        rates=args.rates,
        window_duration_s=args.window_duration_s,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        ttft_target_s=args.ttft_target_s,
        e2e_target_s=args.e2e_target_s,
        cooldown_s=args.cooldown_s,
        output_path=Path(args.output),
    ))

