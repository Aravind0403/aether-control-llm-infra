from pydantic import BaseModel
from typing import Dict, Any, Optional


class WatchdogEvaluation(BaseModel):
    step: int
    active_requests: int
    current_blocks_used: int
    pinned_baseline_blocks: int
    residual_leaked_blocks: int
    leak_percentage_of_pool: float
    is_leak_detected: bool
    readiness_probe_status: int  # 200 (Healthy) or 503 (Scheduled Rotation)
    action: str


class IdleResidualBlockWatchdog:
    """Monitors vllm:num_gpu_blocks_used during idle windows to catch C++ block leaks (VLLM-RFC-002).
    
    Residual_Blocks = vllm:num_gpu_blocks_used when (vllm:num_requests_running == 0)
    If Residual_Blocks exceeds 15% of total pool, flips readiness probe to 503 for zero-downtime drain.
    """

    def __init__(
        self,
        total_block_pool: int = 2000,
        pinned_baseline_blocks: int = 120,
        leak_threshold_pct: float = 0.15,
    ):
        self.total_block_pool = total_block_pool
        self.pinned_baseline_blocks = pinned_baseline_blocks
        self.leak_threshold_pct = leak_threshold_pct
        self.max_allowed_leak_blocks = int(total_block_pool * leak_threshold_pct)

    def evaluate(self, step: int, active_requests: int, blocks_used: int) -> WatchdogEvaluation:
        """Evaluates leak state. Only evaluates residual leak when active_requests == 0."""
        if active_requests > 0:
            return WatchdogEvaluation(
                step=step,
                active_requests=active_requests,
                current_blocks_used=blocks_used,
                pinned_baseline_blocks=self.pinned_baseline_blocks,
                residual_leaked_blocks=0,
                leak_percentage_of_pool=0.0,
                is_leak_detected=False,
                readiness_probe_status=200,
                action="Active traffic in flight; skipping idle residual evaluation.",
            )

        # Idle window (active_requests == 0)
        residual = max(0, blocks_used - self.pinned_baseline_blocks)
        leak_pct = residual / float(self.total_block_pool)

        if residual >= self.max_allowed_leak_blocks:
            return WatchdogEvaluation(
                step=step,
                active_requests=0,
                current_blocks_used=blocks_used,
                pinned_baseline_blocks=self.pinned_baseline_blocks,
                residual_leaked_blocks=residual,
                leak_percentage_of_pool=round(leak_pct * 100.0, 2),
                is_leak_detected=True,
                readiness_probe_status=503,
                action=(
                    f"🚨 LEAK THRESHOLD BREACHED: {residual} residual blocks leaked ({leak_pct*100:.1f}% of pool). "
                    "Tripped readiness probe to HTTP 503. Draining active streams (preStop: sleep 15) and rolling pod."
                ),
            )
        elif residual > 0:
            return WatchdogEvaluation(
                step=step,
                active_requests=0,
                current_blocks_used=blocks_used,
                pinned_baseline_blocks=self.pinned_baseline_blocks,
                residual_leaked_blocks=residual,
                leak_percentage_of_pool=round(leak_pct * 100.0, 2),
                is_leak_detected=False,
                readiness_probe_status=200,
                action=f"⚠️ Minor residual block drift ({residual} blocks). Monitoring baseline.",
            )
        else:
            return WatchdogEvaluation(
                step=step,
                active_requests=0,
                current_blocks_used=blocks_used,
                pinned_baseline_blocks=self.pinned_baseline_blocks,
                residual_leaked_blocks=0,
                leak_percentage_of_pool=0.0,
                is_leak_detected=False,
                readiness_probe_status=200,
                action="✅ Clean idle baseline: Allocated blocks match pinned roots exactly.",
            )
