"""Four-Mode Paired Comparative Benchmark Suite Across Calibrated Traffic Regimes.

Executes scientific paired evaluation across four serving modes:
- Mode 1: Direct vLLM (Unconstrained Baseline)
- Mode 2: Aether Gateway Forwarding Only (Pure Proxy Overhead)
- Mode 3: Aether Gateway Static Concurrency Cap (Immediate 429 Shedding)
- Mode 4: Aether Gateway Bounded Queue + Deadline (Admission Control)

Across four calibrated traffic regimes:
- Regime A (Light Load): 10.0 RPS Poisson, W = 60.0s
- Regime B (Near-Capacity Knee): 35.0 RPS Poisson, W = 60.0s
- Regime C (Overload): 48.0 RPS Poisson, W = 60.0s
- Regime D (Traffic Burst): Base 10.0 RPS (0-20s), Surge 48.0 RPS (20-30s), Recovery 10.0 RPS (30-60s)

Key Rigor Controls:
1. Multi-seed paired comparison: N seeds per regime; each seed's exact trace replayed across all 4 modes.
2. Balanced mode execution order (round-robin) to eliminate thermal and caching order bias.
3. Standardized active warm-up burst applied before every mode evaluation outside the measurement window.
4. Continuous hardware telemetry logging (GPU clocks, power, temperature, utilization) via nvidia-smi.
5. Strict request accounting: offered == completed + shed_429 + failed.
6. Locked generation tokens: min_tokens = max_tokens = 64.
"""

import sys
import time
import json
import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rich.console import Console
from rich.table import Table

from vllm_engine.gateway.admission import AdmissionMode
from vllm_engine.benchmarks.benchmark_serving import ServingBenchmarkRunner, BenchmarkSummary
from vllm_engine.benchmarks.compare_modes import GatewayServerContext
from vllm_engine.benchmarks.calibrate_capacity import sample_gpu_telemetry, execute_standard_warmup
from vllm_engine.benchmarks.load_generator import (
    run_open_loop_poisson_benchmark,
    generate_synthetic_prompts,
    load_dataset_prompts,
    generate_poisson_arrival_schedule_window,
    generate_burst_arrival_schedule,
)
from vllm_engine.benchmarks.paired_experiment import compute_mean_and_sd

console = Console()


async def evaluate_single_mode_run(
    mode_label: str,
    adm_mode: Optional[AdmissionMode],
    mode_kwargs: Dict[str, Any],
    backend_url: str,
    gateway_port: int,
    model_name: str,
    prompts: List[str],
    arrival_schedule: List[float],
    window_duration_s: float,
    min_tokens: int = 64,
    max_tokens: int = 64,
    ttft_target_s: float = 0.25,
    e2e_target_s: float = 2.50,
) -> Tuple[BenchmarkSummary, List[Dict[str, Any]]]:
    """Evaluates a single mode trial with standardized warm-up and GPU telemetry."""
    runner = ServingBenchmarkRunner(
        api_url=backend_url,
        model_name=model_name,
        slo_target_s=e2e_target_s,
        ttft_target_s=ttft_target_s,
        e2e_target_s=e2e_target_s,
    )

    num_requests = len(arrival_schedule)
    test_prompts = [prompts[i % len(prompts)] for i in range(num_requests)]

    # 1. Warm-up
    await execute_standard_warmup(backend_url, model_name, num_warmup_requests=8, warmup_tokens=32)

    # 2. Setup GPU telemetry
    telemetry_samples = []
    telemetry_task = None
    if shutil.which("nvidia-smi"):
        telemetry_task = asyncio.create_task(sample_gpu_telemetry(interval_s=0.5))

    try:
        if adm_mode is None:
            # Mode 1: Direct backend
            target_url = f"{backend_url.rstrip('/')}/v1/chat/completions"
            raw_res, duration = await run_open_loop_poisson_benchmark(
                api_url=target_url,
                model_name=model_name,
                prompts=test_prompts,
                target_rps=float(num_requests) / window_duration_s,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                arrival_schedule=arrival_schedule,
            )
            summary = runner.summarize_results(raw_res, duration, arrival_window_s=window_duration_s)
        else:
            # Modes 2, 3, 4: Ephemeral Gateway
            ctx = GatewayServerContext(
                backend_url=backend_url,
                admission_mode=adm_mode,
                max_concurrency=mode_kwargs.get("max_concurrency", 36),
                max_queue=mode_kwargs.get("max_queue", 36),
                deadline_ms=mode_kwargs.get("deadline_ms", 250.0),
                port=gateway_port,
            )
            async with ctx as gw_url:
                target_url = f"{gw_url}/v1/chat/completions"
                raw_res, duration = await run_open_loop_poisson_benchmark(
                    api_url=target_url,
                    model_name=model_name,
                    prompts=test_prompts,
                    target_rps=float(num_requests) / window_duration_s,
                    max_tokens=max_tokens,
                    min_tokens=min_tokens,
                    arrival_schedule=arrival_schedule,
                )
                summary = runner.summarize_results(raw_res, duration, arrival_window_s=window_duration_s)
    finally:
        if telemetry_task:
            telemetry_task.cancel()
            try:
                telemetry_samples = await telemetry_task
            except asyncio.CancelledError:
                pass

    # Verify request accounting
    accounted = summary.completed_requests + summary.shed_requests_429 + summary.failed_requests
    assert accounted == summary.offered_requests, (
        f"Accounting mismatch in {mode_label}: offered={summary.offered_requests}, accounted={accounted}"
    )

    return summary, telemetry_samples


async def run_paired_regimes_experiment(
    backend_url: str = "http://localhost:8000",
    gateway_port: int = 8080,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    workload_path: Optional[Path] = None,
    seeds: Optional[List[int]] = None,
    window_duration_s: float = 60.0,
    min_tokens: int = 64,
    max_tokens: int = 64,
    max_concurrency: int = 36,
    max_queue: int = 36,
    deadline_ms: float = 250.0,
    ttft_target_s: float = 0.25,
    e2e_target_s: float = 2.50,
    cooldown_s: float = 5.0,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Executes the full paired 4-mode benchmark across all four traffic regimes."""
    seeds = seeds or [101, 102, 103, 104, 105]

    # Prepare frozen prompt distribution
    if workload_path and workload_path.exists():
        prompts = load_dataset_prompts(workload_path, max_samples=100)
    else:
        prompts = generate_synthetic_prompts(num_prompts=100, prompt_tokens=128)

    # 4 Serving Modes Catalog
    modes_catalog = [
        ("Mode 1: Direct vLLM", None, {}),
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

    # 4 Calibrated Traffic Regimes
    regimes_config = [
        {
            "name": "Regime A: Light Load (10 RPS)",
            "type": "poisson",
            "target_rps": 10.0,
            "window_duration_s": window_duration_s,
        },
        {
            "name": "Regime B: Near-Capacity Knee (35 RPS)",
            "type": "poisson",
            "target_rps": 35.0,
            "window_duration_s": window_duration_s,
        },
        {
            "name": "Regime C: Overload (48 RPS)",
            "type": "poisson",
            "target_rps": 48.0,
            "window_duration_s": window_duration_s,
        },
        {
            "name": "Regime D: Traffic Burst (10 RPS base, 48 RPS burst [20-30s])",
            "type": "burst",
            "base_rps": 10.0,
            "burst_rps": 48.0,
            "burst_start_s": 20.0,
            "burst_duration_s": 10.0,
            "window_duration_s": window_duration_s,
        },
    ]

    console.print(f"\n[bold cyan]🚀 Initiating 4-Mode Paired Comparative Benchmark Campaign[/bold cyan]")
    console.print(f"Backend: [yellow]{backend_url}[/yellow] | Model: [yellow]{model_name}[/yellow]")
    console.print(f"Frozen Admission Parameters: max_concurrency={max_concurrency}, max_queue={max_queue}, deadline_ms={deadline_ms}ms")
    console.print(f"SLO Targets: TTFT <= {ttft_target_s*1000:.0f}ms | E2E <= {e2e_target_s:.2f}s | Output: {min_tokens}..{max_tokens} tokens")
    console.print(f"Regimes: {len(regimes_config)} | Seeds per Regime: {len(seeds)} (Total Paired Runs: {len(regimes_config) * len(seeds) * 4})\n")

    campaign_results: Dict[str, Any] = {}

    for regime in regimes_config:
        regime_name = regime["name"]
        console.print(f"\n[bold green]════════════════════════════════════════════════════════════════════[/bold green]")
        console.print(f"[bold green]▶ Executing {regime_name}[/bold green]")
        console.print(f"[bold green]════════════════════════════════════════════════════════════════════[/bold green]")

        # Data collection structures for this regime: mode -> metric -> list of values across seeds
        regime_metrics: Dict[str, Dict[str, List[float]]] = {
            m[0]: {
                "offered": [],
                "completed": [],
                "shed_429": [],
                "failed": [],
                "submission_lag_p99_ms": [],
                "ttft_p50_ms": [],
                "ttft_p99_ms": [],
                "e2e_p50_s": [],
                "e2e_p99_s": [],
                "goodput_rps": [],
                "attainment_pct": [],
            }
            for m in modes_catalog
        }
        regime_trials_records = []

        for seed_idx, seed in enumerate(seeds):
            console.print(f"\n  [bold cyan]• Seed {seed_idx+1}/{len(seeds)} (seed={seed})[/bold cyan]")

            # Generate deterministic trace for this seed
            if regime["type"] == "burst":
                schedule = generate_burst_arrival_schedule(
                    duration_s=regime["window_duration_s"],
                    base_rps=regime["base_rps"],
                    burst_rps=regime["burst_rps"],
                    burst_start_s=regime["burst_start_s"],
                    burst_duration_s=regime["burst_duration_s"],
                    seed=seed,
                )
            else:
                schedule = generate_poisson_arrival_schedule_window(
                    duration_s=regime["window_duration_s"],
                    target_rps=regime["target_rps"],
                    seed=seed,
                )

            console.print(f"    Trace generated: {len(schedule)} requests over {regime['window_duration_s']:.1f}s")

            # Balanced round-robin order rotation across seeds
            rotated_modes = (
                modes_catalog[seed_idx % len(modes_catalog):] +
                modes_catalog[:seed_idx % len(modes_catalog)]
            )

            for label, adm_mode, kwargs in rotated_modes:
                console.print(f"    [dim]Evaluating: {label}...[/dim]")
                summary, telemetry = await evaluate_single_mode_run(
                    mode_label=label,
                    adm_mode=adm_mode,
                    mode_kwargs=kwargs,
                    backend_url=backend_url,
                    gateway_port=gateway_port,
                    model_name=model_name,
                    prompts=prompts,
                    arrival_schedule=schedule,
                    window_duration_s=regime["window_duration_s"],
                    min_tokens=min_tokens,
                    max_tokens=max_tokens,
                    ttft_target_s=ttft_target_s,
                    e2e_target_s=e2e_target_s,
                )

                # Record metrics
                m = regime_metrics[label]
                m["offered"].append(float(summary.offered_requests))
                m["completed"].append(float(summary.completed_requests))
                m["shed_429"].append(float(summary.shed_requests_429))
                m["failed"].append(float(summary.failed_requests))
                m["submission_lag_p99_ms"].append(summary.submission_lag_p99_ms)
                m["ttft_p50_ms"].append(summary.ttft_p50 * 1000.0)
                m["ttft_p99_ms"].append(summary.ttft_p99 * 1000.0)
                m["e2e_p50_s"].append(summary.e2e_p50)
                m["e2e_p99_s"].append(summary.e2e_p99)
                m["goodput_rps"].append(summary.goodput_rps)
                m["attainment_pct"].append(summary.goodput_attainment_pct)

                regime_trials_records.append({
                    "regime": regime_name,
                    "seed": seed,
                    "mode": label,
                    "summary": {
                        "offered": summary.offered_requests,
                        "completed": summary.completed_requests,
                        "shed_429": summary.shed_requests_429,
                        "failed": summary.failed_requests,
                        "submission_lag_p99_ms": summary.submission_lag_p99_ms,
                        "ttft_p50_ms": summary.ttft_p50 * 1000.0,
                        "ttft_p99_ms": summary.ttft_p99 * 1000.0,
                        "e2e_p50_s": summary.e2e_p50,
                        "e2e_p99_s": summary.e2e_p99,
                        "goodput_rps": summary.goodput_rps,
                        "attainment_pct": summary.goodput_attainment_pct,
                    },
                    "telemetry_samples": len(telemetry),
                    "raw_records": summary.raw_records,
                })

                console.print(f"      ✅ {label}: Completed={summary.completed_requests}/{summary.offered_requests}, Shed={summary.shed_requests_429}, Lag={summary.submission_lag_p99_ms:.1f}ms, TTFT P99={summary.ttft_p99*1000:.1f}ms, E2E P99={summary.e2e_p99:.2f}s, Goodput={summary.goodput_rps:.2f} RPS, Attainment={summary.goodput_attainment_pct:.1f}%")
                await asyncio.sleep(cooldown_s)

        # Aggregate summary table for this regime (mean ± sd across seeds)
        table = Table(
            title=f"Paired Experiment Summary: {regime_name} ({len(seeds)} Seeds Replayed)",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Serving Mode", style="cyan", width=25)
        table.add_column("Completed (Mean±SD)", justify="right")
        table.add_column("Shed 429 (Mean±SD)", justify="right")
        table.add_column("Lag P99 (ms)", justify="right", style="dim")
        table.add_column("TTFT P50 (ms)", justify="right")
        table.add_column("TTFT P99 (ms)", justify="right")
        table.add_column("E2E P50 (s)", justify="right")
        table.add_column("E2E P99 (s)", justify="right")
        table.add_column("Goodput (RPS)", justify="right", style="bold yellow")
        table.add_column("Attainment (%)", justify="right", style="bold green")

        regime_summary_dict = {}
        for label, _mode, _kwargs in modes_catalog:
            m = regime_metrics[label]
            comp_m, comp_s = compute_mean_and_sd(m["completed"])
            shed_m, shed_s = compute_mean_and_sd(m["shed_429"])
            lag_m, lag_s = compute_mean_and_sd(m["submission_lag_p99_ms"])
            t50_m, t50_s = compute_mean_and_sd(m["ttft_p50_ms"])
            t99_m, t99_s = compute_mean_and_sd(m["ttft_p99_ms"])
            e50_m, e50_s = compute_mean_and_sd(m["e2e_p50_s"])
            e99_m, e99_s = compute_mean_and_sd(m["e2e_p99_s"])
            gp_m, gp_s = compute_mean_and_sd(m["goodput_rps"])
            att_m, att_s = compute_mean_and_sd(m["attainment_pct"])

            regime_summary_dict[label] = {
                "completed": (comp_m, comp_s),
                "shed_429": (shed_m, shed_s),
                "submission_lag_p99_ms": (lag_m, lag_s),
                "ttft_p50_ms": (t50_m, t50_s),
                "ttft_p99_ms": (t99_m, t99_s),
                "e2e_p50_s": (e50_m, e50_s),
                "e2e_p99_s": (e99_m, e99_s),
                "goodput_rps": (gp_m, gp_s),
                "attainment_pct": (att_m, att_s),
            }

            table.add_row(
                label,
                f"{comp_m:.1f} ± {comp_s:.1f}",
                f"{shed_m:.1f} ± {shed_s:.1f}",
                f"{lag_m:.1f} ± {lag_s:.1f}",
                f"{t50_m:.1f} ± {t50_s:.1f}",
                f"{t99_m:.1f} ± {t99_s:.1f}",
                f"{e50_m:.2f} ± {e50_s:.2f}",
                f"{e99_m:.2f} ± {e99_s:.2f}",
                f"{gp_m:.2f} ± {gp_s:.2f}",
                f"{att_m:.1f}% ± {att_s:.1f}%",
            )

        console.print(table)
        campaign_results[regime_name] = {
            "regime_config": regime,
            "aggregate_summary": regime_summary_dict,
            "trials": regime_trials_records,
        }

    # Final Payload Assembly
    final_payload = {
        "timestamp": time.time(),
        "backend": backend_url,
        "model": model_name,
        "num_seeds": len(seeds),
        "seeds": seeds,
        "parameters": {
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "max_concurrency": max_concurrency,
            "max_queue": max_queue,
            "deadline_ms": deadline_ms,
            "ttft_target_s": ttft_target_s,
            "e2e_target_s": e2e_target_s,
        },
        "regimes": campaign_results,
    }

    out_file = output_path or Path("benchmarks/reports/paired_regimes_experiment_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(final_payload, f, indent=2)

    console.print(f"\n[bold green]🏆 Complete Paired Campaign Artifact Saved: {out_file}[/bold green]\n")
    return final_payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run 4-mode paired serving experiment across calibrated regimes.")
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--gateway-port", type=int, default=8080)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103, 104, 105])
    parser.add_argument("--window-duration-s", type=float, default=60.0)
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-concurrency", type=int, default=36)
    parser.add_argument("--max-queue", type=int, default=36)
    parser.add_argument("--deadline-ms", type=float, default=250.0)
    parser.add_argument("--ttft-target-s", type=float, default=0.25)
    parser.add_argument("--e2e-target-s", type=float, default=2.50)
    parser.add_argument("--cooldown-s", type=float, default=5.0)
    parser.add_argument("--output", default="benchmarks/reports/paired_regimes_experiment_report.json")
    args = parser.parse_args()

    asyncio.run(run_paired_regimes_experiment(
        backend_url=args.backend_url,
        gateway_port=args.gateway_port,
        model_name=args.model_name,
        seeds=args.seeds,
        window_duration_s=args.window_duration_s,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        max_concurrency=args.max_concurrency,
        max_queue=args.max_queue,
        deadline_ms=args.deadline_ms,
        ttft_target_s=args.ttft_target_s,
        e2e_target_s=args.e2e_target_s,
        cooldown_s=args.cooldown_s,
        output_path=Path(args.output),
    ))
