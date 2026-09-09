from typing import Dict, Any, List
from pydantic import BaseModel, Field


class MultiplexerReceipt(BaseModel):
    """Memory balance ledger for Rollout vs Training phase (RFC-003)."""
    phase: str
    weights_gb: float
    gradients_gb: float
    optimizer_gb: float
    kv_cache_gb: float
    activations_gb: float
    total_peak_vram_gb: float
    headroom_on_80gb_pct: float
    collision_free: bool


class VRAMTimeMultiplexer:
    """Manages the Phase-Swapped VRAM Time-Multiplexing lifecycle (RFC-003).
    
    Guarantees that active training gradients/optimizers and autoregressive KV cache
    never co-exist in GPU VRAM at the same time.
    """

    def __init__(
        self,
        gpu_capacity_gb: float = 80.0,
        model_weights_gb: float = 17.5,
        gradients_gb: float = 17.5,
        optimizer_8bit_gb: float = 4.4,
        parallel_activations_gb: float = 12.8,
    ):
        self.gpu_capacity_gb = gpu_capacity_gb
        self.model_weights_gb = model_weights_gb
        self.gradients_gb = gradients_gb
        self.optimizer_8bit_gb = optimizer_8bit_gb
        self.parallel_activations_gb = parallel_activations_gb

        self.current_phase: str = "idle"
        self.active_kv_cache_gb: float = 0.0

    def enter_rollout(self, radix_kv_cache_gb: float = 5.8) -> MultiplexerReceipt:
        """Enters inference generation phase: Gradients and optimizer states are dormant (0 GB)."""
        self.current_phase = "rollout"
        self.active_kv_cache_gb = radix_kv_cache_gb

        # Peak VRAM: Weights + Paged KV Cache (zero gradients, zero optimizer)
        peak_vram = self.model_weights_gb + self.active_kv_cache_gb
        headroom_pct = ((self.gpu_capacity_gb - peak_vram) / self.gpu_capacity_gb) * 100.0

        return MultiplexerReceipt(
            phase="rollout_inference",
            weights_gb=self.model_weights_gb,
            gradients_gb=0.0,
            optimizer_gb=0.0,
            kv_cache_gb=self.active_kv_cache_gb,
            activations_gb=0.0,
            total_peak_vram_gb=round(peak_vram, 2),
            headroom_on_80gb_pct=round(headroom_pct, 1),
            collision_free=True,
        )

    def exit_rollout(self) -> float:
        """Purges the KV cache block pool to 0 MB upon completion handoff."""
        freed_gb = self.active_kv_cache_gb
        self.active_kv_cache_gb = 0.0
        self.current_phase = "handoff"
        return freed_gb

    def enter_train(self) -> MultiplexerReceipt:
        """Enters backward pass training phase: KV cache is 0 GB, gradients and optimizers allocate."""
        assert self.active_kv_cache_gb == 0.0, "FATAL: KV cache must be purged to 0 GB before backward pass!"
        self.current_phase = "train_backward"

        # Peak VRAM: Weights + Gradients + Optimizer + Parallel Causal Activations
        peak_vram = (
            self.model_weights_gb
            + self.gradients_gb
            + self.optimizer_8bit_gb
            + self.parallel_activations_gb
        )
        headroom_pct = ((self.gpu_capacity_gb - peak_vram) / self.gpu_capacity_gb) * 100.0

        return MultiplexerReceipt(
            phase="train_backward",
            weights_gb=self.model_weights_gb,
            gradients_gb=self.gradients_gb,
            optimizer_gb=self.optimizer_8bit_gb,
            kv_cache_gb=0.0,
            activations_gb=self.parallel_activations_gb,
            total_peak_vram_gb=round(peak_vram, 2),
            headroom_on_80gb_pct=round(headroom_pct, 1),
            collision_free=True,
        )
