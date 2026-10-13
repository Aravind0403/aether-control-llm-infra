import pytest
from pathlib import Path
from trainsight.remediators.bin_packer import SequenceBinPacker


def test_bin_packer_reduces_waste(tmp_path: Path):
    seqs = ([50] * 50) + ([2000] * 50)
    packer = SequenceBinPacker(batch_size=10)

    result = packer.pack(seqs)

    assert result.total_sequences == 100
    assert result.num_batches == 10
    # Packed padding waste must be less than random order padding waste
    assert result.packed_padding_waste_pct <= result.original_padding_waste_pct
    assert result.padded_token_reduction_pct >= 0.0

    # Test file writing
    out_file = tmp_path / "packed.json"
    packer.write_packed_index_file(result, out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_bin_packer_exact_benchmark_profile():
    """Validates the exact 10-sequence benchmark workload against the exact-enumeration baseline.
    
    Workload: [120, 240, 800, 1500, 300, 950, 450, 1800, 210, 650]
    Useful tokens = 7,020.
    Sorted batches: [120, 210], [240, 300], [450, 650], [800, 950], [1500, 1800] -> 7,820 padded tokens.
    All 945 pairings mean = 10,286.67 padded tokens.
    """
    raw_lengths = [120, 240, 800, 1500, 300, 950, 450, 1800, 210, 650]
    packer = SequenceBinPacker(batch_size=2)
    res = packer.pack(raw_lengths)

    # 1. Input and Batch Grouping
    assert res.total_sequences == 10
    assert res.num_batches == 5
    assert res.batch_size == 2
    assert res.unpadded_tokens == 7020
    assert res.packed_indices == [[0, 8], [1, 4], [6, 9], [2, 5], [3, 7]]

    # 2. Padded Token Totals
    assert res.packed_padded_tokens == 7820
    assert res.baseline_padded_tokens == 10286.67
    assert res.baseline_method == "exact_enumeration"

    # 3. Corrected Defensible Comparison
    # Padded token reduction: (10286.67 - 7820) / 10286.67 = 23.98% -> 24.0%
    assert res.padded_token_reduction_pct == 24.0

    # Modeled useful-token density gain: 10286.67 / 7820 - 1 = 31.54% -> 31.5%
    assert res.modeled_density_gain_pct == 31.5
    assert res.estimated_throughput_gain_pct == 31.5


def test_bin_packer_odd_batch_size_preserves_effective_batch_size():
    """Validates hyperparameter adjustment under odd batch sizes to ensure effective batch size is preserved."""
    seqs = [100, 200, 300, 400, 500, 600]
    
    # Even batch size (bs=4, accum=2) -> micro=2, accum=4 -> effective = 4 * 2 = 8 == 2 * 4
    packer_even = SequenceBinPacker(batch_size=4)
    res_even = packer_even.pack(seqs, grad_accum=2)
    assert res_even.recommended_per_device_batch_size == 2
    assert res_even.recommended_gradient_accumulation_steps == 4
    assert res_even.recommended_per_device_batch_size * res_even.recommended_gradient_accumulation_steps == 4 * 2

    # Odd batch size (bs=3, accum=2) -> must not halve to 1 and double to 4 (which would yield effective 4 != 6)
    packer_odd = SequenceBinPacker(batch_size=3)
    res_odd = packer_odd.pack(seqs, grad_accum=2)
    assert res_odd.recommended_per_device_batch_size == 3
    assert res_odd.recommended_gradient_accumulation_steps == 2
    assert res_odd.recommended_per_device_batch_size * res_odd.recommended_gradient_accumulation_steps == 3 * 2
