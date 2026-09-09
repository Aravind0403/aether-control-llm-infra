import json
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel


class CalibrationProfile(BaseModel):
    version_id: str
    attn_factor_multiplier: float = 1.0
    checkpointing_factor_multiplier: float = 1.0
    calibrated_date: str = "2026-09-01"


# Built-in calibrated profiles for PyTorch / CUDA pairings
DEFAULT_CALIBRATION_MATRIX: Dict[str, Dict[str, Any]] = {
    "2.1.0": {"attn_factor_multiplier": 1.0, "checkpointing_factor_multiplier": 1.0},
    "2.1.2": {"attn_factor_multiplier": 1.0, "checkpointing_factor_multiplier": 1.0},
    "2.2.0": {"attn_factor_multiplier": 1.0, "checkpointing_factor_multiplier": 1.0},
    "2.2.1": {"attn_factor_multiplier": 1.0, "checkpointing_factor_multiplier": 1.0},
    "2.3.0": {"attn_factor_multiplier": 1.15, "checkpointing_factor_multiplier": 1.18},
    "2.4.0": {"attn_factor_multiplier": 1.12, "checkpointing_factor_multiplier": 1.15},
}


class CalibrationMatrixManager:
    """Manages empirical kernel calibration matrices and runtime uncertainty buffers."""

    def __init__(self, matrix_path: Optional[Path] = None):
        self.matrix: Dict[str, Dict[str, Any]] = dict(DEFAULT_CALIBRATION_MATRIX)
        if matrix_path and matrix_path.exists():
            try:
                with open(matrix_path, "r", encoding="utf-8") as f:
                    custom = json.load(f)
                self.matrix.update(custom)
            except Exception:
                pass

    def get_current_runtime_signature(self) -> str:
        """Detects current runtime PyTorch version."""
        try:
            import torch
            v = str(torch.__version__).split("+")[0]
            return v
        except Exception:
            return "unknown"

    def get_usable_vram(
        self,
        hardware_vram_gb: float,
        runtime_signature: Optional[str] = None,
    ) -> float:
        """Calculates usable VRAM, dynamically expanding uncertainty buffer for uncalibrated runtimes."""
        sig = runtime_signature or self.get_current_runtime_signature()

        if sig in self.matrix:
            # Calibrated runtime: tight 5% OS/CUDA driver headroom
            usable = hardware_vram_gb * 0.95
        else:
            # Uncalibrated or nightly runtime: 15% uncertainty buffer to absorb kernel shifts
            usable = hardware_vram_gb * 0.85

        return round(usable, 2)

    def is_calibrated(self, runtime_signature: Optional[str] = None) -> bool:
        sig = runtime_signature or self.get_current_runtime_signature()
        return sig in self.matrix
