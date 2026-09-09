import hashlib
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
from pydantic import BaseModel


class DatasetSignature(BaseModel):
    signature_key: str
    dataset_name: str
    file_size_bytes: int
    total_samples: int
    min_seq_len: int
    max_seq_len: int
    avg_seq_len: float
    std_seq_len: float
    p50_seq_len: float
    p90_seq_len: float
    p95_seq_len: float
    p99_seq_len: float
    p99_9_seq_len: float
    variance_ratio: float
    histogram_counts: List[int]
    histogram_bin_edges: List[float]
    sample_seq_lengths: List[int]  # Compact representation (up to 1,000 items) for instant Monte Carlo simulation


class FastInvariantHasher:
    """O(1) Composite Invariant Hashing Engine for dataset files.
    
    Hashes the file size, modification timestamp, and the final 64KB (Parquet metadata
    footer or JSONL tail), producing an immutable signature in <10ms without reading
    the full multi-gigabyte file.
    """

    @staticmethod
    def compute_file_signature_key(file_path: Path, tokenizer_id: Optional[str] = None) -> str:
        """Computes O(1) composite invariant hash in <10ms."""
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {file_path}")

        stat = file_path.stat()
        file_size = stat.st_size
        mtime = int(stat.st_mtime_ns)

        hasher = hashlib.sha256()
        hasher.update(str(file_size).encode("utf-8"))
        hasher.update(str(mtime).encode("utf-8"))
        if tokenizer_id:
            hasher.update(tokenizer_id.encode("utf-8"))

        # Read the trailing 64 KB (where Parquet stores its footer metadata or JSONL ends)
        tail_read_size = min(64 * 1024, file_size)
        with open(file_path, "rb") as f:
            if file_size > tail_read_size:
                f.seek(file_size - tail_read_size)
            tail_bytes = f.read(tail_read_size)
            hasher.update(tail_bytes)

        return hasher.hexdigest()

    @classmethod
    def create_signature(
        cls,
        file_path: Path,
        seq_lengths: List[int],
        tokenizer_id: Optional[str] = None,
    ) -> DatasetSignature:
        """Constructs a compact ~20KB DatasetSignature object."""
        key = cls.compute_file_signature_key(file_path, tokenizer_id=tokenizer_id)
        arr = np.array(seq_lengths)

        counts, edges = np.histogram(arr, bins=50)

        # Compact sample for downstream Monte-Carlo replay (max 1000 items)
        if len(seq_lengths) > 1000:
            step = len(seq_lengths) // 1000
            compact_sample = seq_lengths[::step][:1000]
        else:
            compact_sample = seq_lengths

        avg_l = float(np.mean(arr))
        std_l = float(np.std(arr))
        var_r = float(std_l / avg_l) if avg_l > 0 else 0.0

        return DatasetSignature(
            signature_key=key,
            dataset_name=file_path.name,
            file_size_bytes=file_path.stat().st_size,
            total_samples=len(seq_lengths),
            min_seq_len=int(np.min(arr)),
            max_seq_len=int(np.max(arr)),
            avg_seq_len=avg_l,
            std_seq_len=std_l,
            p50_seq_len=float(np.percentile(arr, 50)),
            p90_seq_len=float(np.percentile(arr, 90)),
            p95_seq_len=float(np.percentile(arr, 95)),
            p99_seq_len=float(np.percentile(arr, 99)),
            p99_9_seq_len=float(np.percentile(arr, 99.9)),
            variance_ratio=var_r,
            histogram_counts=[int(c) for c in counts],
            histogram_bin_edges=[float(e) for e in edges],
            sample_seq_lengths=compact_sample,
        )

    @staticmethod
    def save_signature_file(signature: DatasetSignature, cache_path: Path) -> Path:
        """Saves signature to a ~20KB JSON cache artifact."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(signature.model_dump_json(indent=2))
        return cache_path

    @staticmethod
    def load_signature_file(cache_path: Path) -> Optional[DatasetSignature]:
        """Loads signature from cached JSON artifact if it exists."""
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return DatasetSignature(**data)
        except Exception:
            return None
