"""Automated 4-Mode Comparative Benchmark Suite.

Evaluates inference serving performance across 4 architectures:
1. Mode 1: Direct vLLM (backend baseline without gateway)
2. Mode 2: Aether Gateway Forwarding Only (quantifies gateway proxy latency overhead)
3. Mode 3: Aether Static Concurrency (simple concurrency cap, immediate 429 shedding)
4. Mode 4: Aether Bounded Queue + Deadline (bounded buffer absorbing bursts, shedding on deadline)
"""

import time
import asyncio
import socket
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
import uvicorn
import httpx

from vllm_engine.gateway.admission import AdmissionMode
from vllm_engine.gateway.proxy import create_gateway_app
from vllm_engine.benchmarks.benchmark_serving import ServingBenchmarkRunner, BenchmarkSummary
from vllm_engine.benchmarks.load_generator import (
    generate_synthetic_prompts,
    run_open_loop_poisson_benchmark,
)


def find_free_port() -> int:
    """Finds an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class GatewayServerContext:
    """Manages an ephemeral in-process Gateway server instance."""

    def __init__(
        self,
        backend_url: str,
        admission_mode: AdmissionMode,
        max_concurrency: int = 8,
        max_queue: int = 32,
        deadline_ms: float = 500.0,
        port: Optional[int] = None,
    ):
        self.backend_url = backend_url
        self.admission_mode = admission_mode
        self.max_concurrency = max_concurrency
        self.max_queue = max_queue
        self.deadline_ms = deadline_ms
        self.port = port or find_free_port()
        self.host = "127.0.0.1"
        self.server: Optional[uvicorn.Server] = None
        self.task: Optional[asyncio.Task] = None

    async def __aenter__(self):
        app = create_gateway_app(
            backend_url=self.backend_url,
            admission_mode=self.admission_mode,
            max_concurrency=self.max_concurrency,
            max_queue=self.max_queue,
            deadline_ms=self.deadline_ms,
        )
        config = uvicorn.Config(
            app=app,
            host=self.host,
            port=self.port,
            log_level="error",
            access_log=False,
        )
        self.server = uvicorn.Server(config)
        self.task = asyncio.create_task(self.server.serve())

        # Wait until the gateway server responds on /health
        base_url = f"http://{self.host}:{self.port}"
        async with httpx.AsyncClient() as client:
            for _ in range(60):
                try:
                    res = await client.get(f"{base_url}/health", timeout=0.2)
                    if res.status_code == 200:
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.05)

        return f"http://{self.host}:{self.port}"

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.server:
            self.server.should_exit = True
        if self.task:
            await self.task


def create_comparison_table(results: Dict[str, BenchmarkSummary]) -> Table:
    """Generates a formatted Rich comparison table across all evaluated modes."""
    table = Table(
        title="Aether Serving Architecture Comparative Benchmark (Open-Loop Poisson Arrival)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Evaluation Mode", style="cyan", width=25)
    table.add_column("Offered", justify="right", style="dim")
    table.add_column("Completed", justify="right", style="green")
    table.add_column("Shed (429)", justify="right", style="red")
    table.add_column("Trunc/Fail", justify="right", style="magenta")
    table.add_column("Lag P99\n(ms)", justify="right", style="dim")
    table.add_column("RPS", justify="right", style="green")
    table.add_column("Tokens\n(tok/s)", justify="right", style="yellow")
    table.add_column("TTFT P50\n(ms)", justify="right")
    table.add_column("TTFT P99\n(ms)", justify="right")
    table.add_column("TPOT P99\n(ms)", justify="right")
    table.add_column("E2E P99\n(s)", justify="right", style="bold")
    table.add_column("Jitter P99\n(ms)", justify="right")
    table.add_column("Queue P99\n(ms)", justify="right")
    table.add_column("SLO Met\n(%)", justify="right", style="bold green")

    for mode_name, summary in results.items():
        trunc_or_fail = summary.truncated_requests + summary.errored_requests + summary.cancelled_requests
        table.add_row(
            mode_name,
            str(summary.offered_requests),
            str(summary.completed_requests),
            str(summary.shed_requests_429),
            str(trunc_or_fail),
            f"{summary.submission_lag_p99_ms:.1f}",
            f"{summary.request_throughput_rps:.2f}",
            f"{summary.output_token_throughput_tps:.1f}",
            f"{summary.ttft_p50 * 1000:.1f}",
            f"{summary.ttft_p99 * 1000:.1f}",
            f"{summary.tpot_p99 * 1000:.1f}",
            f"{summary.e2e_p99:.2f}",
            f"{summary.inter_event_jitter_p99_ms:.1f}",
            f"{summary.queue_time_p99_ms:.1f}",
            f"{summary.slo_attainment_pct:.1f}%",
        )

    return table


async def run_comparison_suite(
    backend_url: str,
    model_name: str,
    num_requests: int = 30,
    target_rps: float = 10.0,
    max_tokens: int = 64,
    max_concurrency: int = 4,
    max_queue: int = 16,
    deadline_ms: float = 400.0,
    prompts: Optional[Union[List[str], List[Dict[str, Any]]]] = None,
    cooldown_s: float = 1.0,
    slo_target_s: float = 2.0,
) -> Dict[str, BenchmarkSummary]:
    """Executes comparative evaluation across all 4 modes."""
    if prompts is None:
        prompts = generate_synthetic_prompts(num_requests, prompt_tokens=64)

    results: Dict[str, BenchmarkSummary] = {}
    runner = ServingBenchmarkRunner(backend_url, model_name, slo_target_s=slo_target_s)

    modes_to_test = [
        ("Mode 1: Direct vLLM (Baseline)", None, None),
        ("Mode 2: Gateway Forwarding Only", AdmissionMode.PASSTHROUGH, {}),
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

    for label, mode, kwargs in modes_to_test:
        if mode is None:
            # Mode 1: Direct backend
            target_endpoint = f"{backend_url.rstrip('/')}/v1/chat/completions"
            raw_results, duration = await run_open_loop_poisson_benchmark(
                api_url=target_endpoint,
                model_name=model_name,
                prompts=prompts,
                target_rps=target_rps,
                max_tokens=max_tokens,
            )
            summary = runner.summarize_results(raw_results, duration)
            results[label] = summary
        else:
            # Modes 2, 3, 4: Ephemeral Gateway
            extra = kwargs or {}
            ctx = GatewayServerContext(
                backend_url=backend_url,
                admission_mode=mode,
                max_concurrency=extra.get("max_concurrency", max_concurrency),
                max_queue=extra.get("max_queue", max_queue),
                deadline_ms=extra.get("deadline_ms", deadline_ms),
            )
            async with ctx as gateway_url:
                target_endpoint = f"{gateway_url}/v1/chat/completions"
                raw_results, duration = await run_open_loop_poisson_benchmark(
                    api_url=target_endpoint,
                    model_name=model_name,
                    prompts=prompts,
                    target_rps=target_rps,
                    max_tokens=max_tokens,
                )
                summary = runner.summarize_results(raw_results, duration)
                results[label] = summary

        # Cooldown period between test runs to ensure backend queues drain completely
        if cooldown_s > 0:
            await asyncio.sleep(cooldown_s)

    return results
