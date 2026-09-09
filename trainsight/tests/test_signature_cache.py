import pytest
from pathlib import Path
from trainsight.cache.signature import FastInvariantHasher


def test_signature_cache_creation_and_load(tmp_path: Path):
    dummy_file = tmp_path / "train.jsonl"
    dummy_file.write_text('{"prompt": "hi", "completion": "there"}\n' * 50)

    seq_lengths = [10, 20, 30, 40, 50]
    sig = FastInvariantHasher.create_signature(dummy_file, seq_lengths, tokenizer_id="qwen")

    assert sig.dataset_name == "train.jsonl"
    assert sig.total_samples == 5
    assert len(sig.sample_seq_lengths) == 5
    assert len(sig.histogram_counts) == 50

    cache_file = tmp_path / "sig_cache.json"
    FastInvariantHasher.save_signature_file(sig, cache_file)

    loaded = FastInvariantHasher.load_signature_file(cache_file)
    assert loaded is not None
    assert loaded.signature_key == sig.signature_key
    assert loaded.total_samples == sig.total_samples


def test_signature_changes_on_file_modification(tmp_path: Path):
    dummy_file = tmp_path / "test.jsonl"
    dummy_file.write_text("line 1\n")
    sig1 = FastInvariantHasher.compute_file_signature_key(dummy_file)

    # Append single line
    with open(dummy_file, "a") as f:
        f.write("line 2 appended\n")

    sig2 = FastInvariantHasher.compute_file_signature_key(dummy_file)
    # Fast invariant hashing must change if file size or mtime changes
    assert sig1 != sig2
