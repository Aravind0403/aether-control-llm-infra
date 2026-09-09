import pytest
from pathlib import Path
from trainsight.resilience.calibration import CalibrationMatrixManager
from trainsight.resilience.circuit_breaker import StorageCircuitBreaker


def test_calibration_uncertainty_buffer():
    mgr = CalibrationMatrixManager()

    # Known calibrated version gets tight 5% headroom (24 * 0.95 = 22.8 GB)
    usable_cal = mgr.get_usable_vram(hardware_vram_gb=24.0, runtime_signature="2.2.0")
    assert usable_cal == 22.8

    # Unknown/nightly gets 15% uncertainty buffer (24 * 0.85 = 20.4 GB)
    usable_uncal = mgr.get_usable_vram(hardware_vram_gb=24.0, runtime_signature="3.0.0-nightly")
    assert usable_uncal == 20.4
    assert usable_uncal < usable_cal


def test_storage_circuit_breaker(tmp_path: Path):
    cb = StorageCircuitBreaker(timeout_ms=1000.0)

    # Missing file returns exit code 1
    status_missing = cb.check_storage_access(tmp_path / "nonexistent.jsonl")
    assert status_missing.is_tripped is True
    assert status_missing.exit_code == 1

    # Existing file returns exit code 0
    valid_file = tmp_path / "valid.jsonl"
    valid_file.write_text("hello\n")
    status_ok = cb.check_storage_access(valid_file)
    assert status_ok.is_tripped is False
    assert status_ok.exit_code == 0
