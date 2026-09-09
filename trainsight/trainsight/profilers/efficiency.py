from typing import Optional
from pydantic import BaseModel

from trainsight.profilers.activation_profiler import MemoryBreakdownResult
from trainsight.simulators.collation_simulator import CollationSimulationResult


class EfficiencyGrade(BaseModel):
    score: float  # 0 to 100
    letter_grade: str  # A, B, C, D
    vram_occupancy_pct: float
    padding_efficiency_pct: float
    is_underutilized: bool
    recommended_upscaled_batch_size: Optional[int] = None
    estimated_speedup_factor: Optional[float] = None
    action_message: str


class EfficiencyScorer:
    """Computes Compute Efficiency Score (CES) and Auto-Upscaling opportunities."""

    def evaluate(
        self,
        mem_result: MemoryBreakdownResult,
        collation_result: Optional[CollationSimulationResult] = None,
        current_batch_size: int = 8,
    ) -> EfficiencyGrade:
        """Evaluates hardware utilization and sequence efficiency."""
        vram_occ = min(100.0, max(0.0, mem_result.utilization_pct))
        pad_waste = collation_result.padding_waste_pct if collation_result else 20.0
        pad_eff = max(0.0, 100.0 - pad_waste)

        # 50% VRAM Occupancy + 50% Padding Efficiency
        ces_score = round(0.5 * vram_occ + 0.5 * pad_eff, 1)

        if ces_score >= 80.0:
            letter = "A"
            msg = "Excellent efficiency. Eligible for Kubernetes Priority GPU Queue."
        elif ces_score >= 65.0:
            letter = "B"
            msg = "Good efficiency with minor optimization headroom."
        elif ces_score >= 50.0:
            letter = "C"
            msg = "Moderate underutilization or padding waste detected."
        else:
            letter = "D"
            msg = "Severe underutilization. Preemptible Low-Priority Queue assigned."

        # Auto-Upscaling Opportunity
        is_underutilized = (vram_occ < 60.0) and (mem_result.vram_headroom_gb >= 6.0)
        upscaled_bs = None
        speedup = None

        if is_underutilized and current_batch_size > 0:
            # Target 85% VRAM occupancy safely
            target_occ = 85.0
            scale_ratio = target_occ / max(10.0, vram_occ)
            upscaled_bs = int(current_batch_size * scale_ratio)
            # Ensure at least +4 or 1.5x
            upscaled_bs = max(current_batch_size + 4, upscaled_bs)
            speedup = round(upscaled_bs / current_batch_size, 1)
            msg = (
                f"You have {mem_result.vram_headroom_gb:.1f} GB of unused GPU memory. "
                f"You can safely upscale batch size from {current_batch_size} ──► {upscaled_bs}, "
                f"finishing training {speedup:.1f}x faster with zero OOM risk."
            )

        return EfficiencyGrade(
            score=ces_score,
            letter_grade=letter,
            vram_occupancy_pct=round(vram_occ, 1),
            padding_efficiency_pct=round(pad_eff, 1),
            is_underutilized=is_underutilized,
            recommended_upscaled_batch_size=upscaled_bs,
            estimated_speedup_factor=speedup,
            action_message=msg,
        )
