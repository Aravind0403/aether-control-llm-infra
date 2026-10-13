import time
from pathlib import Path
from typing import Optional
import concurrent.futures
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from vllm_engine.config_loader import load_vllm_config, VLLMConfig
from vllm_engine.benchmarks.benchmark_serving import ServingBenchmarkRunner
from vllm_engine.benchmarks.load_generator import generate_synthetic_prompts, send_streaming_request

app = typer.Typer(help="vLLM Serving Engine CLI & Benchmark Suite")
console = Console()


@app.command()
def show_config(
    config_file: Path = typer.Option(Path(__file__).parent / "config.yaml", "--config", "-c", help="Path to config.yaml")
):
    """Displays current vLLM engine configuration and corresponding CLI launch flags."""
    try:
        config = load_vllm_config(config_file)
        console.print(Panel(f"[bold cyan]Model:[/bold cyan] {config.model}\n"
                            f"[bold cyan]VRAM Utilization:[/bold cyan] {config.gpu_memory_utilization * 100}%\n"
                            f"[bold cyan]Max Context Len:[/bold cyan] {config.max_model_len}\n"
                            f"[bold cyan]Chunked Prefill:[/bold cyan] {config.enable_chunked_prefill}\n"
                            f"[bold cyan]Prefix Caching:[/bold cyan] {config.enable_prefix_caching}",
                            title="vLLM Production Configuration"))

        cli_args = config.to_cli_args()
        console.print(f"\n[bold green]Generated Entrypoint Command:[/bold green]")
        console.print(f"[yellow]python3 -m vllm.entrypoints.openai.api_server {' '.join(cli_args)}[/yellow]\n")
    except Exception as e:
        console.print(f"[bold red]Error loading config:[/bold red] {e}")


@app.command()
def benchmark(
    host: str = typer.Option("http://localhost:8000", "--host", "-h", help="vLLM server endpoint URL"),
    model: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--model", "-m", help="Model name"),
    num_requests: int = typer.Option(20, "--num-requests", "-n", help="Total benchmark requests"),
    concurrency: int = typer.Option(4, "--concurrency", "-k", help="Concurrent worker threads"),
    max_tokens: int = typer.Option(128, "--max-tokens", help="Max tokens to generate per request"),
):
    """Runs standard industry SLA benchmark against a vLLM server."""
    console.print(f"\n[bold cyan]🚀 Starting vLLM SLA Benchmark...[/bold cyan]")
    console.print(f"Target: [bold yellow]{host}[/bold yellow] | Model: [bold yellow]{model}[/bold yellow]")
    console.print(f"Requests: {num_requests} | Concurrency: {concurrency}\n")

    prompts = generate_synthetic_prompts(num_requests)
    runner = ServingBenchmarkRunner(host, model)

    results = []
    start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_streaming_request, runner.api_url, model, prompt, max_tokens)
            for prompt in prompts
        ]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    total_duration = time.perf_counter() - start_time
    summary = runner.summarize_results(results, total_duration)

    table = Table(title="vLLM Serving Performance & SLA Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Duration", f"{summary.total_duration_s:.2f} s")
    table.add_row("Success / Total Requests", f"{summary.successful_requests} / {summary.total_requests}")
    table.add_row("Request Throughput", f"{summary.request_throughput_rps:.2f} req/s")
    table.add_row("Output Token Throughput", f"{summary.output_token_throughput_tps:.2f} tokens/s")
    table.add_row("TTFT (Prefill Latency) P50", f"{summary.ttft_p50 * 1000:.1f} ms")
    table.add_row("TTFT (Prefill Latency) P99", f"{summary.ttft_p99 * 1000:.1f} ms")
    table.add_row("TPOT (Decode Latency) P50", f"{summary.tpot_p50 * 1000:.1f} ms/token")
    table.add_row("TPOT (Decode Latency) P99", f"{summary.tpot_p99 * 1000:.1f} ms/token")
    table.add_row("E2E Latency P99", f"{summary.e2e_p99:.2f} s")

    console.print(table)


@app.command()
def gateway(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind host for gateway"),
    port: int = typer.Option(8001, "--port", "-p", help="Port for gateway"),
    backend: str = typer.Option("http://localhost:8000", "--backend", "-b", help="Upstream vLLM server URL"),
    mode: str = typer.Option("bounded_queue", "--mode", "-m", help="Admission mode: passthrough, static_concurrency, bounded_queue"),
    concurrency: int = typer.Option(8, "--concurrency", "-c", help="Max concurrent active slots"),
    max_queue: int = typer.Option(32, "--max-queue", "-q", help="Max queued requests before 429"),
    deadline_ms: float = typer.Option(500.0, "--deadline-ms", "-d", help="Queue wait deadline before 429 (ms)"),
    retry_after: int = typer.Option(5, "--retry-after", help="Retry-After header value for 429 (seconds)"),
):
    """Starts the Aether Admission Gateway streaming reverse proxy."""
    import uvicorn
    from vllm_engine.gateway.admission import AdmissionMode
    from vllm_engine.gateway.proxy import create_gateway_app

    adm_mode = AdmissionMode(mode)
    console.print(f"\n[bold cyan]🚀 Starting Aether Admission Gateway...[/bold cyan]")
    console.print(f"Listening on: [bold green]http://{host}:{port}[/bold green]")
    console.print(f"Backend vLLM: [bold yellow]{backend}[/bold yellow]")
    console.print(f"Mode: [bold magenta]{adm_mode.value}[/bold magenta] | Max Concurrency: {concurrency} | Max Queue: {max_queue} | Deadline: {deadline_ms}ms\n")

    app_instance = create_gateway_app(
        backend_url=backend,
        admission_mode=adm_mode,
        max_concurrency=concurrency,
        max_queue=max_queue,
        deadline_ms=deadline_ms,
        retry_after_s=retry_after,
    )

    uvicorn.run(app_instance, host=host, port=port, log_level="info")


@app.command()
def compare(
    backend: str = typer.Option("http://localhost:8000", "--backend", "-b", help="Target vLLM backend server URL"),
    model: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--model", "-m", help="Model name"),
    workload: Optional[Path] = typer.Option(None, "--workload", "-w", help="Path to reproducible JSONL workload file (e.g. lmsys_smoke_50.jsonl)"),
    num_requests: int = typer.Option(30, "--num-requests", "-n", help="Total benchmark requests per mode (if not using workload file)"),
    rps: float = typer.Option(10.0, "--rps", "-r", help="Target Poisson arrival rate (requests per second)"),
    concurrency: int = typer.Option(4, "--concurrency", "-c", help="Max concurrency slot limit for gateway"),
    max_queue: int = typer.Option(16, "--max-queue", "-q", help="Max queue capacity for bounded mode"),
    deadline_ms: float = typer.Option(400.0, "--deadline-ms", "-d", help="Queue deadline in ms for bounded mode"),
    max_tokens: int = typer.Option(128, "--max-tokens", help="Max output tokens"),
):
    """Runs 4-mode comparative SLA benchmark (Direct vs Gateway modes)."""
    import asyncio
    from vllm_engine.benchmarks.compare_modes import run_comparison_suite, create_comparison_table
    from vllm_engine.benchmarks.workload_prep import load_workload_file

    prompts = None
    if workload and workload.exists():
        items = load_workload_file(workload)
        prompts = items
        num_requests = len(items)
        console.print(f"[bold green]Loaded {len(items)} requests from workload file: {workload}[/bold green]")

    console.print(f"\n[bold cyan]🔬 Starting 4-Mode Comparative Serving Benchmark...[/bold cyan]")
    console.print(f"Backend: [bold yellow]{backend}[/bold yellow] | Model: [bold yellow]{model}[/bold yellow]")
    console.print(f"Schedule: Open-Loop Poisson @ {rps:.1f} req/s | Requests per mode: {num_requests}\n")

    results = asyncio.run(
        run_comparison_suite(
            backend_url=backend,
            model_name=model,
            num_requests=num_requests,
            target_rps=rps,
            max_tokens=max_tokens,
            max_concurrency=concurrency,
            max_queue=max_queue,
            deadline_ms=deadline_ms,
            prompts=prompts,
        )
    )

    table = create_comparison_table(results)
    console.print(table)


@app.command()
def mock_server(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Port for mock vLLM server"),
    chunk_delay_ms: float = typer.Option(15.0, "--chunk-delay-ms", help="Simulated decode delay per token event (ms)"),
):
    """Starts a local mock OpenAI/vLLM streaming server for local testing on Mac/CPU."""
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    import asyncio, json

    mock_app = FastAPI(title="Mock vLLM Backend")

    @mock_app.post("/v1/chat/completions")
    async def chat_completions(req: Request):
        body = await req.json()
        is_stream = body.get("stream", False)
        max_tokens = min(body.get("max_tokens", 32), 64)

        if not is_stream:
            return {
                "id": "chatcmpl-mock",
                "choices": [{"message": {"role": "assistant", "content": "Mock completion."}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": max_tokens, "total_tokens": 20 + max_tokens},
            }

        async def token_generator():
            words = ["The", " quick", " brown", " fox", " jumps", " over", " lazy", " dog", " with", " efficient", " attention", " kernels."]
            generated_tokens = 0
            for i in range(max_tokens):
                await asyncio.sleep(chunk_delay_ms / 1000.0)
                chunk_text = words[i % len(words)] + " "
                generated_tokens += 1
                data = {
                    "id": "chatcmpl-mock",
                    "choices": [{"delta": {"content": chunk_text}}],
                }
                yield f"data: {json.dumps(data)}\n\n"

            # Terminal usage chunk
            usage_data = {
                "id": "chatcmpl-mock",
                "choices": [],
                "usage": {"prompt_tokens": 20, "completion_tokens": generated_tokens, "total_tokens": 20 + generated_tokens},
            }
            yield f"data: {json.dumps(usage_data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(token_generator(), media_type="text/event-stream")

    @mock_app.get("/health")
    async def health():
        return {"status": "ok"}

    console.print(f"[bold cyan]🚀 Starting Mock vLLM Server on http://{host}:{port}...[/bold cyan]")
    uvicorn.run(mock_app, host=host, port=port, log_level="info")


@app.command()
def smoke(
    backend: str = typer.Option("http://localhost:8000", "--backend", "-b", help="Target vLLM backend server URL"),
    model: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--model", "-m", help="Model name"),
    port: int = typer.Option(8001, "--port", "-p", help="Gateway port for smoke tests"),
):
    """Runs the automated 6-step GPU pre-flight smoke test."""
    import asyncio
    from vllm_engine.benchmarks.gpu_smoke import run_gpu_smoke_test

    success = asyncio.run(run_gpu_smoke_test(backend_url=backend, model_name=model, gateway_port=port))
    if not success:
        raise typer.Exit(code=1)


@app.command()
def experiment(
    backend: str = typer.Option("http://localhost:8000", "--backend", "-b", help="Target vLLM backend server URL"),
    model: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--model", "-m", help="Model name"),
    runs: int = typer.Option(6, "--runs", "-N", help="Number of independent paired runs per mode"),
    num_requests: int = typer.Option(50, "--num-requests", "-n", help="Requests per run"),
    rps: float = typer.Option(12.0, "--rps", "-r", help="Target Poisson arrival rate"),
    concurrency: int = typer.Option(4, "--concurrency", "-c", help="Max concurrency limit for gateway"),
    max_queue: int = typer.Option(16, "--max-queue", "-q", help="Max queue capacity for bounded mode"),
    deadline_ms: float = typer.Option(400.0, "--deadline-ms", "-d", help="Queue deadline in ms"),
    max_tokens: int = typer.Option(64, "--max-tokens", help="Max output tokens"),
    cooldown: float = typer.Option(3.0, "--cooldown", help="Cooldown drain seconds between mode switches"),
    workload: Optional[Path] = typer.Option(None, "--workload", "-w", help="Path to base JSONL workload file"),
):
    """Runs multi-trial paired comparative experiment reporting Mean ± Sample SD across runs."""
    import asyncio
    from pathlib import Path
    from vllm_engine.benchmarks.paired_experiment import (
        PairedExperimentRunner,
        format_paired_experiment_table,
        format_paired_experiment_markdown,
        save_paired_experiment_json,
    )

    runner = PairedExperimentRunner(
        backend_url=backend,
        model_name=model,
        num_runs=runs,
        requests_per_run=num_requests,
        target_rps=rps,
        max_tokens=max_tokens,
        max_concurrency=concurrency,
        max_queue=max_queue,
        deadline_ms=deadline_ms,
        cooldown_s=cooldown,
        workload_file=workload,
    )

    summary_stats = asyncio.run(runner.run_experiment())
    table = format_paired_experiment_table(summary_stats, num_runs=runs)
    console.print(table)

    # Save JSON report
    report_path = Path("benchmarks/reports/paired_experiment_report.json")
    save_paired_experiment_json(summary_stats, num_runs=runs, output_path=report_path)
    console.print(f"\n[green]Saved structured results to: {report_path}[/green]\n")

    md_table = format_paired_experiment_markdown(summary_stats, num_runs=runs)
    console.print(md_table)


if __name__ == "__main__":
    app()



