"""Multi-Trial Paired Comparative Serving Benchmark Suite.

Executes rigorous scientific comparative evaluation across inference architectures:
- N runs (default 6 independent runs) per load configuration.
- N distinct arrival traces generated via different random seeds.
- Paired comparison: every trace is replayed across all 4 modes.
- Rotates evaluation mode order to eliminate systemic thermal/caching bias.
- Consistent pre-trial warmup and inter-trial drain cooldown.
- Reports sample mean ± sample standard deviation (ddof=1) for all SLA metrics.
"""

import time
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rich.console import Console
from rich.table import Table

from vllm_engine.gateway.admission import AdmissionMode
from vllm_engine.benchmarks.benchmark_serving import ServingBenchmarkRunner, BenchmarkSummary
from vllm_engine.benchmarks.compare_modes import GatewayServerContext
from vllm_engine.benchmarks.load_generator import (
    generate_synthetic_prompts,
    run_open_loop_poisson_benchmark,
    async_send_streaming_request,
)
from vllm_engine.benchmarks.workload_prep import prepare_lmsys_workload, load_workload_file

console = Console()


def compute_mean_and_sd(values: List[float]) -> Tuple[float, float]:
    """Computes sample mean and sample standard deviation (ddof=1)."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    arr = np.array(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


class PairedExperimentRunner:
    """Orchestrates paired multi-run benchmark trials with randomized trace replays."""

    def __init__(
        self,
        backend_url: str,
        model_name: str,
        num_runs: int = 6,
        requests_per_run: int = 50,
        target_rps: float = 12.0,
        max_tokens: int = 64,
        max_concurrency: int = 4,
        max_queue: int = 16,
        deadline_ms: float = 400.0,
        cooldown_s: float = 3.0,
        workload_file: Optional[Path] = None,
        seeds: Optional[List[int]] = None,
    ):
        self.backend_url = backend_url
        self.model_name = model_name
        self.num_runs = num_runs
        self.requests_per_run = requests_per_run
        self.target_rps = target_rps
        self.max_tokens = max_tokens
        self.max_concurrency = max_concurrency
        self.max_queue = max_queue
        self.deadline_ms = deadline_ms
        self.cooldown_s = cooldown_s
        self.workload_file = workload_file
        self.seeds = seeds or [42 + i for i in range(num_runs)]
        self.runner = ServingBenchmarkRunner(backend_url, model_name)

    async def _warmup_backend(self, target_url: str):
        """Sends a few throwaway warmup requests to ensure CUDA kernels and KV caches are primed."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                for _ in range(2):
                    await async_send_streaming_request(
                        client=client,
                        api_url=target_url,
                        model_name=self.model_name,
                        prompt="Warmup query.",
                        max_tokens=8,
                    )
        except Exception:
            pass

    async def run_experiment(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Executes the full paired multi-run experiment."""
        console.print(f"\n[bold cyan]🔬 Initiating {self.num_runs}-Run Paired Serving Experiment...[/bold cyan]")
        console.print(f"Backend: [yellow]{self.backend_url}[/yellow] | Model: [yellow]{self.model_name}[/yellow]")
        console.print(f"Target RPS: {self.target_rps} | Requests/Run: {self.requests_per_run} | Runs: {self.num_runs}")
        console.print(f"Total Evaluations: {self.num_runs * 4} paired runs\n")

        # Base mode definitions
        modes_catalog = [
            ("Mode 1: Direct vLLM", None, None),
            ("Mode 2: Forwarding Only", AdmissionMode.PASSTHROUGH, {}),
            (
                "Mode 3: Static Concurrency Cap",
                AdmissionMode.STATIC_CONCURRENCY,
                {"max_concurrency": self.max_concurrency},
            ),
            (
                "Mode 4: Bounded Queue + Deadline",
                AdmissionMode.BOUNDED_QUEUE,
                {
                    "max_concurrency": self.max_concurrency,
                    "max_queue": self.max_queue,
                    "deadline_ms": self.deadline_ms,
                },
            ),
        ]

        # Structure to collect individual run metrics per mode
        # key: mode_name -> metric_name -> list of values across the runs
        run_data: Dict[str, Dict[str, List[float]]] = {
            m[0]: {
                "ttft_p50_ms": [],
                "ttft_p99_ms": [],
                "tpot_p99_ms": [],
                "e2e_p99_s": [],
                "throughput_rps": [],
                "token_throughput_tps": [],
                "completed_count": [],
                "shed_429_count": [],
                "trunc_fail_count": [],
                "submission_lag_p99_ms": [],
                "inter_event_jitter_p99_ms": [],
                "slo_attainment_pct": [],
            }
            for m in modes_catalog
        }

        # Pre-load or generate base workload
        base_workload = None
        if self.workload_file and self.workload_file.exists():
            base_workload = load_workload_file(self.workload_file)

        # Loop over the N independent runs
        for run_idx in range(self.num_runs):
            seed = self.seeds[run_idx]
            console.print(f"[bold magenta]▶ Executing Run {run_idx+1}/{self.num_runs} (Seed: {seed})[/bold magenta]")

            # Generate trace for this run
            if base_workload is not None:
                trace_prompts = base_workload[:self.requests_per_run]
            else:
                trace_prompts = prepare_lmsys_workload(
                    num_samples=self.requests_per_run,
                    seed=seed,
                    target_rps=self.target_rps,
                    max_tokens=self.max_tokens,
                )

            # Rotate mode execution order (round-robin) to avoid warming/thermal bias
            rotated_modes = (
                modes_catalog[run_idx % len(modes_catalog):] +
                modes_catalog[:run_idx % len(modes_catalog)]
            )

            for label, mode, kwargs in rotated_modes:
                console.print(f"  [dim]Evaluating: {label}...[/dim]")

                if mode is None:
                    # Mode 1: Direct backend
                    target_url = f"{self.backend_url.rstrip('/')}/v1/chat/completions"
                    await self._warmup_backend(target_url)

                    raw_res, dur = await run_open_loop_poisson_benchmark(
                        api_url=target_url,
                        model_name=self.model_name,
                        prompts=trace_prompts,
                        target_rps=self.target_rps,
                        max_tokens=self.max_tokens,
                    )
                    summary = self.runner.summarize_results(raw_res, dur)
                else:
                    # Modes 2, 3, 4: Ephemeral Gateway
                    extra = kwargs or {}
                    ctx = GatewayServerContext(
                        backend_url=self.backend_url,
                        admission_mode=mode,
                        max_concurrency=extra.get("max_concurrency", self.max_concurrency),
                        max_queue=extra.get("max_queue", self.max_queue),
                        deadline_ms=extra.get("deadline_ms", self.deadline_ms),
                    )
                    async with ctx as gw_url:
                        target_url = f"{gw_url}/v1/chat/completions"
                        await self._warmup_backend(target_url)

                        raw_res, dur = await run_open_loop_poisson_benchmark(
                            api_url=target_url,
                            model_name=self.model_name,
                            prompts=trace_prompts,
                            target_rps=self.target_rps,
                            max_tokens=self.max_tokens,
                        )
                        summary = self.runner.summarize_results(raw_res, dur)

                # Record metrics for this run
                trunc_fail = summary.truncated_requests + summary.errored_requests + summary.cancelled_requests
                metrics = run_data[label]
                metrics["ttft_p50_ms"].append(summary.ttft_p50 * 1000.0)
                metrics["ttft_p99_ms"].append(summary.ttft_p99 * 1000.0)
                metrics["tpot_p99_ms"].append(summary.tpot_p99 * 1000.0)
                metrics["e2e_p99_s"].append(summary.e2e_p99)
                metrics["throughput_rps"].append(summary.request_throughput_rps)
                metrics["token_throughput_tps"].append(summary.output_token_throughput_tps)
                metrics["completed_count"].append(float(summary.completed_requests))
                metrics["shed_429_count"].append(float(summary.shed_requests_429))
                metrics["trunc_fail_count"].append(float(trunc_fail))
                metrics["submission_lag_p99_ms"].append(summary.submission_lag_p99_ms)
                metrics["inter_event_jitter_p99_ms"].append(summary.inter_event_jitter_p99_ms)
                metrics["slo_attainment_pct"].append(summary.slo_attainment_pct)

                # Cooldown drain
                if self.cooldown_s > 0:
                    await asyncio.sleep(self.cooldown_s)

        # Aggregate across all runs: compute mean ± SD (ddof=1)
        summary_stats: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for mode_name, metrics in run_data.items():
            summary_stats[mode_name] = {}
            for metric_name, values in metrics.items():
                mean, sd = compute_mean_and_sd(values)
                summary_stats[mode_name][metric_name] = (mean, sd)

        return summary_stats


def format_paired_experiment_table(
    summary_stats: Dict[str, Dict[str, Tuple[float, float]]],
    num_runs: int,
) -> Table:
    """Formats a Rich table reporting Mean ± SD across paired runs."""
    table = Table(
        title=f"Aether Paired Serving Experiment ({num_runs} Independent Runs, Mean ± Sample SD)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Evaluation Mode", style="cyan", width=25)
    table.add_column("Completed", justify="right", style="green")
    table.add_column("Shed (429)", justify="right", style="red")
    table.add_column("Trunc/Fail", justify="right", style="magenta")
    table.add_column("Lag P99 (ms)", justify="right", style="dim")
    table.add_column("RPS", justify="right", style="green")
    table.add_column("Tok/s", justify="right", style="yellow")
    table.add_column("TTFT P50 (ms)", justify="right")
    table.add_column("TTFT P99 (ms)", justify="right")
    table.add_column("TPOT P99 (ms)", justify="right")
    table.add_column("E2E P99 (s)", justify="right", style="bold")
    table.add_column("SLO Met (%)", justify="right", style="bold green")

    for mode_name, stats in summary_stats.items():
        comp_m, comp_sd = stats["completed_count"]
        shed_m, shed_sd = stats["shed_429_count"]
        fail_m, fail_sd = stats["trunc_fail_count"]
        lag_m, lag_sd = stats["submission_lag_p99_ms"]
        rps_m, rps_sd = stats["throughput_rps"]
        tps_m, tps_sd = stats["token_throughput_tps"]
        ttft50_m, ttft50_sd = stats["ttft_p50_ms"]
        ttft99_m, ttft99_sd = stats["ttft_p99_ms"]
        tpot99_m, tpot99_sd = stats["tpot_p99_ms"]
        e2e99_m, e2e99_sd = stats["e2e_p99_s"]
        slo_m, slo_sd = stats["slo_attainment_pct"]

        table.add_row(
            mode_name,
            f"{comp_m:.1f} ± {comp_sd:.1f}",
            f"{shed_m:.1f} ± {shed_sd:.1f}",
            f"{fail_m:.1f} ± {fail_sd:.1f}",
            f"{lag_m:.1f} ± {lag_sd:.1f}",
            f"{rps_m:.2f} ± {rps_sd:.2f}",
            f"{tps_m:.1f} ± {tps_sd:.1f}",
            f"{ttft50_m:.1f} ± {ttft50_sd:.1f}",
            f"{ttft99_m:.1f} ± {ttft99_sd:.1f}",
            f"{tpot99_m:.1f} ± {tpot99_sd:.1f}",
            f"{e2e99_m:.2f} ± {e2e99_sd:.2f}",
            f"{slo_m:.1f}% ± {slo_sd:.1f}%",
        )

    return table


def format_paired_experiment_markdown(
    summary_stats: Dict[str, Dict[str, Tuple[float, float]]],
    num_runs: int,
) -> str:
    """Formats a clean Markdown table reporting Mean ± SD across paired runs."""
    lines = [
        f"### Aether Paired Serving Experiment ({num_runs} Independent Runs, Mean ± Sample SD)\n",
        "| Evaluation Mode | Completed | Shed (429) | Trunc/Fail | Lag P99 (ms) | RPS | Tok/s | TTFT P50 (ms) | TTFT P99 (ms) | TPOT P99 (ms) | E2E P99 (s) | SLO Met (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for mode_name, stats in summary_stats.items():
        comp_m, comp_sd = stats["completed_count"]
        shed_m, shed_sd = stats["shed_429_count"]
        fail_m, fail_sd = stats["trunc_fail_count"]
        lag_m, lag_sd = stats["submission_lag_p99_ms"]
        rps_m, rps_sd = stats["throughput_rps"]
        tps_m, tps_sd = stats["token_throughput_tps"]
        ttft50_m, ttft50_sd = stats["ttft_p50_ms"]
        ttft99_m, ttft99_sd = stats["ttft_p99_ms"]
        tpot99_m, tpot99_sd = stats["tpot_p99_ms"]
        e2e99_m, e2e99_sd = stats["e2e_p99_s"]
        slo_m, slo_sd = stats["slo_attainment_pct"]

        lines.append(
            f"| **{mode_name}** | {comp_m:.1f} ± {comp_sd:.1f} | {shed_m:.1f} ± {shed_sd:.1f} | {fail_m:.1f} ± {fail_sd:.1f} | "
            f"{lag_m:.1f} ± {lag_sd:.1f} | {rps_m:.2f} ± {rps_sd:.2f} | {tps_m:.1f} ± {tps_sd:.1f} | "
            f"{ttft50_m:.1f} ± {ttft50_sd:.1f} | {ttft99_m:.1f} ± {ttft99_sd:.1f} | {tpot99_m:.1f} ± {tpot99_sd:.1f} | "
            f"{e2e99_m:.2f} ± {e2e99_sd:.2f} | {slo_m:.1f}% ± {slo_sd:.1f}% |"
        )
    return "\n".join(lines)


def save_paired_experiment_json(
    summary_stats: Dict[str, Dict[str, Tuple[float, float]]],
    num_runs: int,
    output_path: Path,
):
    """Saves structured experiment results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "timestamp": time.time(),
        "num_runs": num_runs,
        "modes": {
            mode: {
                metric: {"mean": mean, "sd": sd}
                for metric, (mean, sd) in metrics.items()
            }
            for mode, metrics in summary_stats.items()
        }
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

