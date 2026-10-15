"""Four-Mode Serving Benchmark Pilot Runner.

Verifies:
1. Open-loop Poisson arrival schedules delivered as intended (submission lag P99 tracked).
2. Gateway additional latency measured accurately (Mode 2 vs Mode 1).
3. Full request accounting: offered == completed + shed_429 + failed across all 4 modes.
4. Goodput evaluated strictly as:
     Goodput = (Compliant Requests within all Latency Targets) / (Arrival Window Duration)
5. Attainment reported as:
     Attainment % = (Compliant Requests / Total Offered Requests) * 100%
"""

import sys
import time
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx
from rich.console import Console
from rich.table import Table

from vllm_engine.gateway.admission import AdmissionMode
from vllm_engine.benchmarks.benchmark_serving import ServingBenchmarkRunner, BenchmarkSummary
from vllm_engine.benchmarks.compare_modes import GatewayServerContext
from vllm_engine.benchmarks.load_generator import (
    run_open_loop_poisson_benchmark,
    generate_synthetic_prompts,
    load_dataset_prompts,
)

console = Console()


async def run_pilot(
    backend_url: str = "http://localhost:8000",
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    workload_path: Optional[Path] = None,
    target_rps: float = 8.0,
    requests_per_mode: int = 30,
    max_tokens: int = 64,
    ttft_target_s: float = 0.8,
    e2e_target_s: float = 1.5,
    max_concurrency: int = 4,
    max_queue: int = 16,
    deadline_ms: float = 400.0,
    gateway_port: int = 8080,
) -> Dict[str, Any]:
    """Runs a 4-mode pilot evaluation to validate benchmark harness and metrics accounting."""
    # Prepare frozen prompt distribution
    if workload_path and workload_path.exists():
        prompts = load_dataset_prompts(workload_path, max_samples=100)
    else:
        prompts = generate_synthetic_prompts(num_prompts=100, prompt_tokens=128)

    test_prompts = [prompts[i % len(prompts)] for i in range(requests_per_mode)]

    # Pre-generate deterministic shared arrival schedule with fixed seed
    from vllm_engine.benchmarks.load_generator import generate_poisson_arrival_schedule
    shared_arrival_schedule = generate_poisson_arrival_schedule(
        num_requests=requests_per_mode,
        target_rps=target_rps,
        seed=42,
    )
    arrival_window_s = shared_arrival_schedule[-1] - shared_arrival_schedule[0] if len(shared_arrival_schedule) > 1 else 1.0

    console.print(f"\n[bold cyan]🚀 Initiating 4-Mode Serving Pilot Experiment...[/bold cyan]")
    console.print(f"Backend: [yellow]{backend_url}[/yellow] | Model: [yellow]{model_name}[/yellow]")
    console.print(f"Offered Load: {target_rps:.1f} RPS | Requests/Mode: {requests_per_mode}")
    console.print(f"Shared Arrival Window: {arrival_window_s:.3f}s (Deterministic Seed: 42)")
    console.print(f"Targets: TTFT <= {ttft_target_s*1000:.0f}ms | E2E <= {e2e_target_s:.2f}s\n")

    runner = ServingBenchmarkRunner(
        api_url=backend_url,
        model_name=model_name,
        slo_target_s=e2e_target_s,
        ttft_target_s=ttft_target_s,
        e2e_target_s=e2e_target_s,
    )

    modes_config = [
        ("Mode 1: Direct vLLM", None, None),
        ("Mode 2: Forwarding Only", AdmissionMode.PASSTHROUGH, {}),
        (
            "Mode 3: Static Concurrency Cap",
            AdmissionMode.STATIC_CONCURRENCY,
            {"max_concurrency": max_concurrency},
        ),
        (
            "Mode 4: Bounded Queue + Deadline",
            AdmissionMode.BOUNDED_QUEUE,
            {
                "max_concurrency": max_concurrency,
                "max_queue": max_queue,
                "deadline_ms": deadline_ms,
            },
        ),
    ]

    pilot_results: Dict[str, BenchmarkSummary] = {}

    for label, adm_mode, kwargs in modes_config:
        console.print(f"[bold magenta]▶ Evaluating {label}...[/bold magenta]")

        if adm_mode is None:
            # Mode 1: Direct backend
            target_url = f"{backend_url.rstrip('/')}/v1/chat/completions"
            raw_res, duration = await run_open_loop_poisson_benchmark(
                api_url=target_url,
                model_name=model_name,
                prompts=test_prompts,
                target_rps=target_rps,
                max_tokens=max_tokens,
                arrival_schedule=shared_arrival_schedule,
            )
            summary = runner.summarize_results(raw_res, duration, arrival_window_s=arrival_window_s)
        else:
            # Modes 2, 3, 4: Ephemeral Gateway
            extra = kwargs or {}
            ctx = GatewayServerContext(
                backend_url=backend_url,
                admission_mode=adm_mode,
                max_concurrency=extra.get("max_concurrency", max_concurrency),
                max_queue=extra.get("max_queue", max_queue),
                deadline_ms=extra.get("deadline_ms", deadline_ms),
                port=gateway_port,
            )
            async with ctx as gw_url:
                target_url = f"{gw_url}/v1/chat/completions"
                raw_res, duration = await run_open_loop_poisson_benchmark(
                    api_url=target_url,
                    model_name=model_name,
                    prompts=test_prompts,
                    target_rps=target_rps,
                    max_tokens=max_tokens,
                    arrival_schedule=shared_arrival_schedule,
                )
                summary = runner.summarize_results(raw_res, duration, arrival_window_s=arrival_window_s)

        # Assert full request accounting
        accounted = summary.completed_requests + summary.shed_requests_429 + summary.failed_requests
        assert accounted == summary.offered_requests, (
            f"Accounting mismatch in {label}: offered={summary.offered_requests}, accounted={accounted}"
        )
        console.print(f"  ✅ Request Accounting: {summary.completed_requests} completed, {summary.shed_requests_429} shed (429), {summary.failed_requests} failed (Total: {summary.offered_requests})")
        console.print(f"  ✅ Submission Lag P99: {summary.submission_lag_p99_ms:.2f}ms (Client Health Verified)")

        pilot_results[label] = summary
        await asyncio.sleep(1.0)

    # Format Pilot Results Table
    table = Table(
        title="Aether 4-Mode Serving Pilot Results",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Evaluation Mode", style="cyan", width=25)
    table.add_column("Completed", justify="right", style="green")
    table.add_column("Shed (429)", justify="right", style="red")
    table.add_column("Lag P99 (ms)", justify="right", style="dim")
    table.add_column("TTFT P50 (ms)", justify="right")
    table.add_column("TTFT P99 (ms)", justify="right")
    table.add_column("E2E P99 (s)", justify="right")
    table.add_column("Goodput (RPS)", justify="right", style="bold yellow")
    table.add_column("Attainment (%)", justify="right", style="bold green")

    # Gateway overhead calculation
    m1 = pilot_results["Mode 1: Direct vLLM"]
    m2 = pilot_results["Mode 2: Forwarding Only"]
    ttft_delta_ms = (m2.ttft_p50 - m1.ttft_p50) * 1000.0

    for label, summary in pilot_results.items():
        table.add_row(
            label,
            f"{summary.completed_requests}/{summary.offered_requests}",
            f"{summary.shed_requests_429}",
            f"{summary.submission_lag_p99_ms:.1f}",
            f"{summary.ttft_p50 * 1000.0:.1f}",
            f"{summary.ttft_p99 * 1000.0:.1f}",
            f"{summary.e2e_p99:.2f}",
            f"{summary.goodput_rps:.2f}",
            f"{summary.goodput_attainment_pct:.1f}%",
        )

    console.print(table)
    console.print(f"\n[bold]Measured Gateway Forwarding Latency Delta (Mode 2 vs Mode 1 TTFT P50):[/bold] [yellow]{ttft_delta_ms:+.2f} ms[/yellow]\n")

    # Save pilot report artifact
    out_path = Path("benchmarks/reports/pilot_four_modes_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_dict = {
        "timestamp": time.time(),
        "target_rps": target_rps,
        "requests_per_mode": requests_per_mode,
        "ttft_target_s": ttft_target_s,
        "e2e_target_s": e2e_target_s,
        "shared_arrival_window_s": arrival_window_s,
        "gateway_forwarding_delta_p50_ms": ttft_delta_ms,
        "modes": {
            label: {
                "offered": s.offered_requests,
                "completed": s.completed_requests,
                "shed_429": s.shed_requests_429,
                "failed": s.failed_requests,
                "submission_lag_p99_ms": s.submission_lag_p99_ms,
                "ttft_p50_ms": s.ttft_p50 * 1000.0,
                "ttft_p99_ms": s.ttft_p99 * 1000.0,
                "e2e_p99_s": s.e2e_p99,
                "goodput_rps": s.goodput_rps,
                "goodput_attainment_pct": s.goodput_attainment_pct,
                "raw_records": s.raw_records,
            }
            for label, s in pilot_results.items()
        },
    }
    with open(out_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    console.print(f"Saved pilot results artifact to: [green]{out_path}[/green]\n")
    return report_dict


if __name__ == "__main__":
    asyncio.run(run_pilot())
