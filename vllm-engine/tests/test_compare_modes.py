import pytest
import asyncio
from vllm_engine.benchmarks.load_generator import (
    generate_poisson_arrival_schedule,
    run_open_loop_poisson_benchmark,
)
from vllm_engine.benchmarks.benchmark_serving import BenchmarkSummary
from vllm_engine.benchmarks.compare_modes import (
    create_comparison_table,
    run_comparison_suite,
)
try:
    from tests.test_gateway_proxy import create_mock_vllm_app
except ImportError:
    from test_gateway_proxy import create_mock_vllm_app
import uvicorn
import socket


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_poisson_arrival_schedule():
    # Empty
    assert generate_poisson_arrival_schedule(0, 10.0) == []

    # Burst (target_rps <= 0)
    burst = generate_poisson_arrival_schedule(5, 0.0)
    assert len(burst) == 5
    assert all(t == 0.0 for t in burst)

    # Standard Poisson process
    schedule = generate_poisson_arrival_schedule(10, target_rps=20.0)
    assert len(schedule) == 10
    assert schedule[0] == 0.0
    # Schedule must be monotonically non-decreasing
    for t1, t2 in zip(schedule[:-1], schedule[1:]):
        assert t2 >= t1


def test_create_comparison_table():
    sample_summary = BenchmarkSummary(
        total_requests=20,
        successful_requests=18,
        failed_requests=2,
        total_duration_s=2.5,
        request_throughput_rps=7.2,
        output_token_throughput_tps=120.0,
        ttft_p50=0.04,
        ttft_p90=0.08,
        ttft_p95=0.09,
        ttft_p99=0.12,
        tpot_p50=0.015,
        tpot_p90=0.018,
        tpot_p95=0.020,
        tpot_p99=0.022,
        e2e_p50=0.5,
        e2e_p90=0.8,
        e2e_p95=0.9,
        e2e_p99=1.1,
        offered_requests=20,
        admitted_requests=18,
        completed_requests=18,
        shed_requests_429=2,
        cancelled_requests=0,
        inter_event_jitter_p50_ms=1.2,
        inter_event_jitter_p99_ms=3.5,
        queue_time_p50_ms=0.5,
        queue_time_p99_ms=4.2,
    )
    table = create_comparison_table({"Mode 1: Baseline": sample_summary})
    assert table is not None
    assert len(table.rows) == 1


@pytest.mark.asyncio
async def test_full_comparison_suite_with_mock_backend():
    mock_app = create_mock_vllm_app()
    port = find_free_port()
    config = uvicorn.Config(app=mock_app, host="127.0.0.1", port=port, log_level="error", access_log=False)
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())

    await asyncio.sleep(0.1)
    backend_url = f"http://127.0.0.1:{port}"

    try:
        results = await run_comparison_suite(
            backend_url=backend_url,
            model_name="mock-qwen-model",
            num_requests=6,
            target_rps=15.0,
            max_tokens=16,
            max_concurrency=2,
            max_queue=4,
            deadline_ms=200.0,
            prompts=["Test query 1", "Test query 2", "Test query 3", "Test query 4", "Test query 5", "Test query 6"],
        )
        assert len(results) == 4
        for name, summary in results.items():
            assert summary.total_requests == 6
            assert summary.offered_requests == 6
    finally:
        server.should_exit = True
        await task
