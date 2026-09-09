from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from trainsight.profilers.activation_profiler import MemoryBreakdownResult
from trainsight.simulators.collation_simulator import CollationSimulationResult


class MBOMReceipt(BaseModel):
    job_name: str
    hardware_name: str
    hardware_vram_gb: float
    total_predicted_peak_gb: float
    utilization_pct: float
    fits_in_vram: bool
    components: List[Dict[str, Any]]
    primary_hog_component: str
    pedagogy_title: str
    pedagogy_explanation: str
    recommendations: List[str]
    cumulative_risk_pct: float = 0.0
    predicted_blast_step: Optional[int] = None
    seed_delta_explanation: Optional[str] = None


class MBOMGenerator:
    """Dynamic Memory Bill of Materials (MBOM) & Pedagogy Engine.
    
    Translates physical silicon memory tenants into an undeniable,
    educational breakdown with Staff-level engineering recommendations.
    """

    def generate(
        self,
        job_name: str,
        mem_result: MemoryBreakdownResult,
        collation_result: Optional[CollationSimulationResult] = None,
        hardware_name: str = "24 GB VRAM",
        total_steps: int = 1000,
    ) -> MBOMReceipt:
        """Generates the structured MBOM receipt and pedagogical teaching moment."""
        vram = mem_result.available_vram_gb
        total_peak = mem_result.total_peak_gb

        # Build 5-tenant items
        c_weights = {
            "name": "1. Model Weights (BF16)",
            "footprint_gb": mem_result.weights_gb,
            "pct_of_vram": round((mem_result.weights_gb / vram) * 100.0, 1),
            "status": "✅ Fixed footprint (params × precision)",
        }
        c_grads = {
            "name": "2. Static Gradients",
            "footprint_gb": mem_result.gradients_gb,
            "pct_of_vram": round((mem_result.gradients_gb / vram) * 100.0, 1),
            "status": "✅ Fixed footprint",
        }
        
        opt_pct = round((mem_result.optimizer_gb / vram) * 100.0, 1)
        c_opt = {
            "name": f"3. Optimizer States ({mem_result.optimizer_type.upper()})",
            "footprint_gb": mem_result.optimizer_gb,
            "pct_of_vram": opt_pct,
            "status": "⚠️ THE MEMORY HOG" if opt_pct > 40.0 else "✅ Managed footprint",
        }

        act_pct = round((mem_result.activations_gb / vram) * 100.0, 1)
        pad_note = ""
        if collation_result and collation_result.padding_waste_pct > 25.0:
            pad_note = f" ({collation_result.padding_waste_pct:.1f}% is padding waste)"
        c_act = {
            "name": "4. Dynamic Activations",
            "footprint_gb": mem_result.activations_gb,
            "pct_of_vram": act_pct,
            "status": f"⚠️ Activation surge{pad_note}" if act_pct > 30.0 else "✅ Within normal limits",
        }

        c_comm = {
            "name": "5. Comm / NCCL Buffers",
            "footprint_gb": mem_result.comm_buffers_gb,
            "pct_of_vram": round((mem_result.comm_buffers_gb / vram) * 100.0, 1),
            "status": "✅ Pinned / Staging Buffers",
        }

        components = [c_weights, c_grads, c_opt, c_act, c_comm]

        # Identify primary hog
        tenant_map = {
            "Optimizer States": mem_result.optimizer_gb,
            "Dynamic Activations": mem_result.activations_gb,
            "Model Weights": mem_result.weights_gb,
            "Static Gradients": mem_result.gradients_gb,
            "Communication Buffers": mem_result.comm_buffers_gb,
        }
        primary_hog = max(tenant_map, key=tenant_map.get)

        # Pedagogical algebraic decision tree
        recommendations = []
        if mem_result.optimizer_gb > (total_peak * 0.45) and "8bit" not in mem_result.optimizer_type.lower():
            pedagogy_title = "Optimizer State Dominance (FP32 AdamW)"
            freed_gb = round(mem_result.optimizer_gb * (8.0 / 12.0), 1)
            pedagogy_explanation = (
                f"Why did you run out of memory? It wasn't just your batch size! "
                f"{(mem_result.optimizer_gb / total_peak)*100:.1f}% of your predicted peak is swallowed by AdamW's 32-bit momentum states. "
                f"Switching to 8-bit AdamW (`bitsandbytes.optim.AdamW8bit`) reduces optimizer memory from "
                f"{mem_result.optimizer_gb:.1f} GB to {round(mem_result.optimizer_gb - freed_gb, 1)} GB, "
                f"instantly freeing ~{freed_gb} GB of VRAM without lowering batch size."
            )
            recommendations.append(f"Switch to 8-bit AdamW (`--optimizer adamw_8bit`) to free {freed_gb} GB VRAM.")
        elif collation_result and collation_result.padding_waste_pct > 30.0:
            pedagogy_title = "Variable Padding Inefficiency (Sequence Length Variance)"
            pedagogy_explanation = (
                f"High sequence length variance caused dynamic batch collation to waste "
                f"{collation_result.padding_waste_pct:.1f}% of batch compute on padding zeros. "
                f"A single long sequence forced all {collation_result.batch_size} sequences in the batch to pad to "
                f"{collation_result.worst_case_seq_len} tokens."
            )
            recommendations.append("Apply Pre-Flight Sequence Bin-Packing (`trainsight profile --auto-pack`) to eliminate padding waste.")
        elif not mem_result.gradient_checkpointing and mem_result.activations_gb > (total_peak * 0.35):
            pedagogy_title = "Activation Memory Surge"
            pedagogy_explanation = (
                f"Dynamic activations account for {mem_result.activations_gb:.1f} GB. "
                f"Enabling FlashAttention-2 with Selective Activation Checkpointing will recompute intermediate QK projections on the backward pass, "
                f"reducing activation footprint by ~65%."
            )
            recommendations.append("Enable gradient checkpointing (`--gradient-checkpointing`) in training arguments.")
        else:
            pedagogy_title = "Balanced Memory Distribution"
            pedagogy_explanation = "Memory components are evenly distributed across static model state and dynamic batch activations."

        # Cumulative risk calculation
        cumulative_risk = 0.0
        predicted_blast_step = None
        seed_delta = None
        if collation_result and collation_result.empirical_oom_probability > 0.0:
            p_step = collation_result.empirical_oom_probability
            cumulative_risk = round((1.0 - ((1.0 - p_step) ** total_steps)) * 100.0, 1)
            predicted_blast_step = collation_result.predicted_blast_step
            seed_delta = (
                f"Seed lottery sensitivity: P(OOM) is {p_step * 100:.2f}% per step. "
                f"Cumulative probability of cluster crash within {total_steps} steps is {cumulative_risk}%."
            )

        return MBOMReceipt(
            job_name=job_name,
            hardware_name=hardware_name,
            hardware_vram_gb=mem_result.available_vram_gb,
            total_predicted_peak_gb=mem_result.total_peak_gb,
            utilization_pct=mem_result.utilization_pct,
            fits_in_vram=mem_result.fits_in_vram,
            components=components,
            primary_hog_component=primary_hog,
            pedagogy_title=pedagogy_title,
            pedagogy_explanation=pedagogy_explanation,
            recommendations=recommendations,
            cumulative_risk_pct=cumulative_risk,
            predicted_blast_step=predicted_blast_step,
            seed_delta_explanation=seed_delta,
        )
