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
    assert result.waste_reduction_pct >= 0.0

    # Test file writing
    out_file = tmp_path / "packed.json"
    packer.write_packed_index_file(result, out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0
