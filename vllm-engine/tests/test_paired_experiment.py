import pytest
import asyncio
from vllm_engine.benchmarks.paired_experiment import (
    compute_mean_and_sd,
    PairedExperimentRunner,
    format_paired_experiment_table,
)
from vllm_engine.benchmarks.gpu_smoke import run_gpu_smoke_test
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


def test_compute_mean_and_sd():
    # Single element
    m, s = compute_mean_and_sd([10.0])
    assert m == 10.0
    assert s == 0.0

    # Multiple elements
    values = [10.0, 12.0, 14.0]
    m, s = compute_mean_and_sd(values)
    assert m == 12.0
    assert abs(s - 2.0) < 1e-6


@pytest.mark.asyncio
async def test_smoke_test_and_paired_experiment_mock():
    mock_app = create_mock_vllm_app()
    port = find_free_port()
    config = uvicorn.Config(app=mock_app, host="127.0.0.1", port=port, log_level="error", access_log=False)
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())

    await asyncio.sleep(0.1)
    backend_url = f"http://127.0.0.1:{port}"

    try:
        # Run 2-run paired experiment to verify pipeline
        runner = PairedExperimentRunner(
            backend_url=backend_url,
            model_name="mock-qwen-model",
            num_runs=2,
            requests_per_run=4,
            target_rps=20.0,
            max_tokens=8,
            max_concurrency=2,
            max_queue=4,
            deadline_ms=200.0,
            cooldown_s=0.1,
            seeds=[101, 102],
        )
        stats = await runner.run_experiment()
        assert len(stats) == 4
        for mode_name, metrics in stats.items():
            assert "ttft_p99_ms" in metrics
            mean_val, sd_val = metrics["ttft_p99_ms"]
            assert mean_val >= 0.0

        table = format_paired_experiment_table(stats, num_runs=2)
        assert table is not None
    finally:
        server.should_exit = True
        await task
