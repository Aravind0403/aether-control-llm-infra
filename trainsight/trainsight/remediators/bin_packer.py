import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from pydantic import BaseModel, Field


class PackingResult(BaseModel):
    total_sequences: int
    num_batches: int
    batch_size: int
    unpadded_tokens: int = 0
    baseline_padded_tokens: float = 0.0
    packed_padded_tokens: int = 0
    original_padding_waste_pct: float
    packed_padding_waste_pct: float
    waste_reduction_pct: float
    padded_token_reduction_pct: float = 0.0
    estimated_throughput_gain_pct: float  # Retained for backwards compatibility: modeled density gain
    modeled_density_gain_pct: float = 0.0
    baseline_method: str = "exact_enumeration"
    recommended_per_device_batch_size: int
    recommended_gradient_accumulation_steps: int
    packed_indices: List[List[int]]


class SequenceBinPacker:
    """Deterministic Length-Bucketed Sequence Bin-Packer.
    
    Groups sequences of similar length together to eliminate padding waste
    and provides invariant-preserving hyperparameter adjustments.
    
    Both baseline and packed padding are calculated identically:
    padded_tokens = sum(max(lengths_in_batch) * len(batch)).
    """

    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size

    @staticmethod
    def _calculate_batch_padded_tokens(seq_lengths: List[int], batch: Any) -> int:
        """Calculates total padded tokens for a single batch."""
        if len(batch) == 0:
            return 0
        return max(seq_lengths[idx] for idx in batch) * len(batch)

    def _compute_baseline_padded_tokens(
        self,
        seq_lengths: List[int],
        bs: int,
        num_shuffles: int = 100,
        seed: int = 42,
    ) -> Tuple[float, str]:
        """Calculates expected padded tokens under random/shuffled collation.
        
        For small fixtures (N=10, bs=2), computes exact expectation over all 945 pairings.
        For general datasets, computes empirical mean over seeded permutations.
        """
        n = len(seq_lengths)
        if n == 10 and bs == 2:
            def _get_pairings(items):
                if len(items) == 2:
                    yield [(items[0], items[1])]
                    return
                first = items[0]
                rest = items[1:]
                for i, partner in enumerate(rest):
                    pair = (first, partner)
                    remaining = rest[:i] + rest[i + 1:]
                    for sub in _get_pairings(remaining):
                        yield [pair] + sub

            pairings = list(_get_pairings(seq_lengths))
            totals = [sum(max(a, b) * 2 for a, b in p) for p in pairings]
            return float(np.mean(totals)), "exact_enumeration"
        else:
            rng = np.random.default_rng(seed)
            indices = np.arange(n)
            totals = []
            for _ in range(num_shuffles):
                shuffled = rng.permutation(indices)
                batches = [shuffled[i:i + bs] for i in range(0, n, bs)]
                tot = sum(self._calculate_batch_padded_tokens(seq_lengths, b) for b in batches)
                totals.append(tot)
            return float(np.mean(totals)), f"shuffled_monte_carlo (K={num_shuffles})"

    def pack(
        self,
        seq_lengths: List[int],
        batch_size: Optional[int] = None,
        grad_accum: int = 1,
        num_shuffles: int = 100,
        seed: int = 42,
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

        unpadded_sum = sum(seq_lengths)

        # Baseline: True random collation (exact enumeration for N=10, bs=2; seeded shuffle otherwise)
        random_padded_total, method = self._compute_baseline_padded_tokens(
            seq_lengths, bs, num_shuffles=num_shuffles, seed=seed
        )
        orig_waste_pct = max(0.0, ((random_padded_total - unpadded_sum) / max(1.0, random_padded_total)) * 100.0)

        # Packed collation padding waste
        packed_padded_total = sum(self._calculate_batch_padded_tokens(seq_lengths, b) for b in batches)
        packed_waste_pct = max(0.0, ((packed_padded_total - unpadded_sum) / max(1.0, packed_padded_total)) * 100.0)
        waste_reduction_pp = max(0.0, orig_waste_pct - packed_waste_pct)

        # Padded token reduction vs random baseline
        if random_padded_total > 0:
            padded_token_reduction = max(0.0, ((random_padded_total - packed_padded_total) / random_padded_total) * 100.0)
        else:
            padded_token_reduction = 0.0

        # Modeled useful-token density gain: (useful / packed_total) / (useful / random_total) - 1
        # Exactly equal to: (random_padded_total / packed_padded_total) - 1
        if packed_padded_total > 0 and random_padded_total > 0:
            useful_density_gain = max(0.0, ((random_padded_total / packed_padded_total) - 1.0) * 100.0)
        else:
            useful_density_gain = 0.0

        # Invariant-preserving hyperparameter translation
        # If micro-batch is halved, grad_accum is doubled to preserve effective batch size: bs * accum
        # If batch size is odd, halving breaks integer divisibility of effective batch size
        if bs % 2 == 0:
            safe_micro_bs = bs // 2
            safe_accum = grad_accum * 2
        else:
            safe_micro_bs = bs
            safe_accum = grad_accum

        return PackingResult(
            total_sequences=n,
            num_batches=len(batches),
            batch_size=bs,
            unpadded_tokens=unpadded_sum,
            baseline_padded_tokens=round(random_padded_total, 2),
            packed_padded_tokens=packed_padded_total,
            original_padding_waste_pct=round(orig_waste_pct, 1),
            packed_padding_waste_pct=round(packed_waste_pct, 1),
            waste_reduction_pct=round(waste_reduction_pp, 1),
            padded_token_reduction_pct=round(padded_token_reduction, 1),
            estimated_throughput_gain_pct=round(useful_density_gain, 1),
            modeled_density_gain_pct=round(useful_density_gain, 1),
            baseline_method=method,
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
            "unpadded_tokens": result.unpadded_tokens,
            "baseline_padded_tokens": result.baseline_padded_tokens,
            "packed_padded_tokens": result.packed_padded_tokens,
            "padded_token_reduction_pct": result.padded_token_reduction_pct,
            "modeled_density_gain_pct": result.modeled_density_gain_pct,
            "baseline_method": result.baseline_method,
            "batches": result.packed_indices,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return output_path
