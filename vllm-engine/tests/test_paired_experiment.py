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


@pytest.mark.asyncio
async def test_open_loop_capacity_calibration_and_four_mode_pilot_mock():
    from vllm_engine.benchmarks.calibrate_capacity import run_capacity_calibration
    from vllm_engine.benchmarks.pilot_four_modes import run_pilot

    mock_app = create_mock_vllm_app()
    port = find_free_port()
    config = uvicorn.Config(app=mock_app, host="127.0.0.1", port=port, log_level="error", access_log=False)
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())

    await asyncio.sleep(0.1)
    backend_url = f"http://127.0.0.1:{port}"

    try:
        # Test capacity calibration
        calib = await run_capacity_calibration(
            backend_url=backend_url,
            model_name="mock-model",
            rates=[10.0, 20.0],
            requests_per_rate=4,
            max_tokens=8,
            ttft_target_s=0.5,
            e2e_target_s=1.0,
        )
        assert "calibrated_capacity_rps" in calib
        assert len(calib["records"]) == 2

        # Test 4-mode pilot
        gw_port = find_free_port()
        pilot = await run_pilot(
            backend_url=backend_url,
            model_name="mock-model",
            target_rps=20.0,
            requests_per_mode=4,
            max_tokens=8,
            ttft_target_s=0.5,
            e2e_target_s=1.0,
            gateway_port=gw_port,
        )
        assert len(pilot["modes"]) == 4
        for label, m in pilot["modes"].items():
            assert m["offered"] == 4
            assert m["offered"] == m["completed"] + m["shed_429"] + m["failed"]
            assert m["goodput_rps"] >= 0.0
            assert 0.0 <= m["goodput_attainment_pct"] <= 100.0
    finally:
        server.should_exit = True
        await task


@pytest.mark.asyncio
async def test_repeatability_diagnostic_mock():
    from vllm_engine.benchmarks.run_repeatability_diagnostic import run_repeatability_diagnostic

    mock_app = create_mock_vllm_app()
    port = find_free_port()
    config = uvicorn.Config(app=mock_app, host="127.0.0.1", port=port, log_level="error", access_log=False)
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())

    await asyncio.sleep(0.1)
    backend_url = f"http://127.0.0.1:{port}"

    try:
        diag = await run_repeatability_diagnostic(
            backend_url=backend_url,
            model_name="mock-model",
            window_duration_s=2.0,
            target_rps=5.0,
            num_runs=2,
            max_tokens=8,
            min_tokens=8,
            ttft_target_s=0.5,
            e2e_target_s=1.0,
            cooldown_s=0.1,
        )
        assert diag["num_runs"] == 2
        assert len(diag["trials"]) == 2
        for t in diag["trials"]:
            assert t["completed"] == t["offered"]
            assert len(t["raw_records"]) == t["completed"]
            r0 = t["raw_records"][0]
            assert "e2e_from_schedule_s" in r0
            assert "e2e_from_dispatch_s" in r0
            assert "submission_lag_ms" in r0
    finally:
        server.should_exit = True
        await task
