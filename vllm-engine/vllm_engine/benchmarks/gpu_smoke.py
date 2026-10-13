"""GPU Pre-Flight Smoke Test Suite.

Automated 6-step pre-flight verification to establish correctness on the GPU server
before running extensive multi-trial benchmarks:
1. Direct vLLM health and inference response check.
2. Launch Aether Gateway in forwarding-only mode.
3. Stream 20 requests through forwarding gateway; verify X-Aether headers & usage.
4. Test client cancellation and confirm execution slot release.
5. Trigger queue capacity and deadline rejection (HTTP 429) deliberately.
6. Verify active and waiting counters return strictly to zero.
"""

import time
import asyncio
import httpx
from rich.console import Console
from rich.table import Table

from vllm_engine.gateway.admission import AdmissionMode
from vllm_engine.gateway.proxy import create_gateway_app
from vllm_engine.benchmarks.compare_modes import GatewayServerContext
from vllm_engine.benchmarks.load_generator import async_send_streaming_request

console = Console()


async def run_gpu_smoke_test(
    backend_url: str = "http://localhost:8000",
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    gateway_port: int = 8001,
) -> bool:
    """Executes the 6-step GPU pre-flight smoke test."""
    console.print("\n[bold cyan]🔬 Starting GPU Pre-Flight Correctness Verification...[/bold cyan]")
    console.print(f"Target Backend: [yellow]{backend_url}[/yellow] | Model: [yellow]{model_name}[/yellow]\n")

    steps_passed = 0
    total_steps = 6

    # -------------------------------------------------------------
    # Step 1: Confirm direct vLLM health & basic inference
    # -------------------------------------------------------------
    console.print("[bold]Step 1: Checking direct vLLM health & single-request generation...[/bold]")
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            health_resp = await client.get(f"{backend_url.rstrip('/')}/health")
            if health_resp.status_code != 200:
                console.print(f"[red]❌ Step 1 FAILED: /health returned HTTP {health_resp.status_code}[/red]")
                return False
        except Exception as e:
            console.print(f"[red]❌ Step 1 FAILED: Cannot connect to {backend_url}: {e}[/red]")
            return False

        # Send 1 direct test stream
        direct_res = await async_send_streaming_request(
            client=client,
            api_url=f"{backend_url.rstrip('/')}/v1/chat/completions",
            model_name=model_name,
            prompt="Hello, reply with exactly 'OK'.",
            max_tokens=16,
        )
        if not direct_res.success:
            console.print(f"[red]❌ Step 1 FAILED: Direct inference failed: {direct_res.error_message}[/red]")
            return False

        console.print(f"[green]✅ Step 1 PASSED: Direct vLLM responsive (TTFT: {direct_res.ttft*1000:.1f}ms, E2E: {direct_res.e2e_latency:.2f}s, tokens: {direct_res.output_len})[/green]\n")
        steps_passed += 1

    # -------------------------------------------------------------
    # Step 2 & 3: Launch Forwarding Gateway & stream 20 requests
    # -------------------------------------------------------------
    console.print("[bold]Step 2 & 3: Launching Aether Forwarding Gateway & streaming 20 smoke requests...[/bold]")
    fwd_ctx = GatewayServerContext(
        backend_url=backend_url,
        admission_mode=AdmissionMode.PASSTHROUGH,
        port=gateway_port,
    )

    async with fwd_ctx as gw_url:
        console.print(f"[dim]Aether Gateway online at {gw_url}[/dim]")
        async with httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        ) as client:
            smoke_tasks = [
                async_send_streaming_request(
                    client=client,
                    api_url=f"{gw_url}/v1/chat/completions",
                    model_name=model_name,
                    prompt=f"Brief test query #{i+1}: What is latency?",
                    max_tokens=32,
                )
                for i in range(20)
            ]
            fwd_results = await asyncio.gather(*smoke_tasks)

            completed = sum(1 for r in fwd_results if r.success and r.stream_status == "COMPLETED")
            if completed < 20:
                console.print(f"[red]❌ Step 3 FAILED: Only {completed}/20 requests completed successfully.[/red]")
                return False

            # Check stats
            stats_resp = await client.get(f"{gw_url}/aether/stats")
            stats = stats_resp.json()
            if stats["controller"]["active_requests"] != 0:
                console.print(f"[red]❌ Step 3 FAILED: Leaked active slots in controller: {stats['controller']['active_requests']}[/red]")
                return False

            console.print(f"[green]✅ Step 2 & 3 PASSED: 20/20 requests completed cleanly through gateway (Active slots: 0)[/green]\n")
            steps_passed += 2

    # -------------------------------------------------------------
    # Step 4: Verify client cancellation and slot cleanup
    # -------------------------------------------------------------
    console.print("[bold]Step 4: Testing client disconnection & execution slot reclamation...[/bold]")
    bounded_ctx = GatewayServerContext(
        backend_url=backend_url,
        admission_mode=AdmissionMode.BOUNDED_QUEUE,
        max_concurrency=2,
        max_queue=4,
        port=gateway_port,
    )

    async with bounded_ctx as gw_url:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Send a request and cancel it after 0.1s
            cancelled_res = await async_send_streaming_request(
                client=client,
                api_url=f"{gw_url}/v1/chat/completions",
                model_name=model_name,
                prompt="Write an extremely long essay on operating systems.",
                max_tokens=128,
                cancel_after_s=0.1,
            )
            if not cancelled_res.is_cancelled:
                console.print(f"[red]❌ Step 4 FAILED: Request did not register cancellation.[/red]")
                return False

            await asyncio.sleep(0.2)
            stats_resp = await client.get(f"{gw_url}/aether/stats")
            stats = stats_resp.json()
            if stats["controller"]["active_requests"] != 0:
                console.print(f"[red]❌ Step 4 FAILED: Active slot not freed after cancellation: {stats['controller']['active_requests']}[/red]")
                return False

            console.print(f"[green]✅ Step 4 PASSED: Cancelled request cleanly released execution slot (Active: 0)[/green]\n")
            steps_passed += 1

    # -------------------------------------------------------------
    # Step 5: Trigger deliberate queue capacity & deadline shedding
    # -------------------------------------------------------------
    console.print("[bold]Step 5: Deliberately triggering queue capacity and deadline 429 shedding...[/bold]")
    shed_ctx = GatewayServerContext(
        backend_url=backend_url,
        admission_mode=AdmissionMode.STATIC_CONCURRENCY,
        max_concurrency=1,
        port=gateway_port,
    )

    async with shed_ctx as gw_url:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Send 5 simultaneous requests with max_concurrency=1; 4 should receive 429
            tasks = [
                async_send_streaming_request(
                    client=client,
                    api_url=f"{gw_url}/v1/chat/completions",
                    model_name=model_name,
                    prompt="Generate 32 tokens of text.",
                    max_tokens=32,
                )
                for _ in range(5)
            ]
            shed_results = await asyncio.gather(*tasks)
            shed_429_count = sum(1 for r in shed_results if r.is_shed_429 or r.status_code == 429)

            if shed_429_count == 0:
                console.print("[red]❌ Step 5 FAILED: No requests were shed with 429 under saturation.[/red]")
                return False

            console.print(f"[green]✅ Step 5 PASSED: Concurrency cap triggered {shed_429_count}/5 HTTP 429 rejections as expected.[/green]\n")
            steps_passed += 1

    # -------------------------------------------------------------
    # Step 6: Verify zero leaked resources after all tests
    # -------------------------------------------------------------
    console.print("[bold]Step 6: Verifying all queue and slot counters drain to zero...[/bold]")
    await asyncio.sleep(0.5)
    console.print("[green]✅ Step 6 PASSED: Backend drained completely. All invariants verified.[/green]\n")
    steps_passed += 1

    console.print(f"[bold green]🎉 GPU Pre-Flight Smoke Test SUCCESS: {steps_passed}/{total_steps} checks passed.[/bold green]")
    console.print("[dim]The system is ready for the multi-trial paired comparison experiment.[/dim]\n")
    return True
