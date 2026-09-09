import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from pydantic import BaseModel


class PackingResult(BaseModel):
    total_sequences: int
    num_batches: int
    batch_size: int
    original_padding_waste_pct: float
    packed_padding_waste_pct: float
    waste_reduction_pct: float
    estimated_throughput_gain_pct: float
    recommended_per_device_batch_size: int
    recommended_gradient_accumulation_steps: int
    packed_indices: List[List[int]]


class SequenceBinPacker:
    """Deterministic Length-Bucketed Sequence Bin-Packer.
    
    Groups sequences of similar length together to eliminate padding waste
    and provides invariant-preserving hyperparameter adjustments.
    """

    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size

    def pack(
        self,
        seq_lengths: List[int],
        batch_size: Optional[int] = None,
        grad_accum: int = 1,
    ) -> PackingResult:
        """Groups sequence indices into length-matched batches."""
        bs = batch_size or self.batch_size
        n = len(seq_lengths)
        if n == 0:
            raise ValueError("Sequence lengths list is empty.")

        # Sort indices by length
        sorted_indices = sorted(range(n), key=lambda i: seq_lengths[i])

        batches: List[List[int]] = []
        for i in range(0, n, bs):
            batches.append(sorted_indices[i:i + bs])

        # Calculate original random-order padding waste estimate
        # In a random shuffle, average max length in a batch is skewed toward high percentiles
        unpadded_sum = sum(seq_lengths)
        p95_len = float(np.percentile(seq_lengths, 95))
        avg_len = float(np.mean(seq_lengths))
        
        # Standard collation padding waste estimate
        random_padded_total = len(batches) * bs * max(avg_len, p95_len * 0.85)
        orig_waste_pct = max(0.0, ((random_padded_total - unpadded_sum) / max(1.0, random_padded_total)) * 100.0)

        # Packed collation padding waste
        packed_padded_total = 0
        for b in batches:
            max_in_b = max(seq_lengths[idx] for idx in b)
            packed_padded_total += max_in_b * len(b)

        packed_waste_pct = max(0.0, ((packed_padded_total - unpadded_sum) / max(1.0, packed_padded_total)) * 100.0)
        waste_reduction = max(0.0, orig_waste_pct - packed_waste_pct)

        # Throughput gain is proportional to avoided padding tokens
        if packed_padded_total > 0 and random_padded_total > 0:
            throughput_gain = min(150.0, max(0.0, ((random_padded_total / packed_padded_total) - 1.0) * 100.0))
        else:
            throughput_gain = 0.0

        # Invariant-preserving hyperparameter translation
        # If micro-batch is halved, grad_accum is doubled to preserve effective batch size: bs * accum
        safe_micro_bs = max(1, bs // 2)
        safe_accum = grad_accum * 2

        return PackingResult(
            total_sequences=n,
            num_batches=len(batches),
            batch_size=bs,
            original_padding_waste_pct=round(orig_waste_pct, 1),
            packed_padding_waste_pct=round(packed_waste_pct, 1),
            waste_reduction_pct=round(waste_reduction, 1),
            estimated_throughput_gain_pct=round(throughput_gain, 1),
            recommended_per_device_batch_size=safe_micro_bs,
            recommended_gradient_accumulation_steps=safe_accum,
            packed_indices=batches,
        )

    def write_packed_index_file(self, result: PackingResult, output_path: Path) -> Path:
        """Writes packed batch indices to a JSON file."""
        data = {
            "total_sequences": result.total_sequences,
            "num_batches": result.num_batches,
            "batch_size": result.batch_size,
            "waste_reduction_pct": result.waste_reduction_pct,
            "estimated_throughput_gain_pct": result.estimated_throughput_gain_pct,
            "batches": result.packed_indices,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return output_path
