import tempfile
import json
from pathlib import Path
import pytest
from trainsight.inspectors.sft_inspector import SFTInspector
from trainsight.inspectors.dpo_inspector import DPOInspector


def test_with_vs_without_trainsight_sft():
    """Empirical proof test comparing raw un-gated dataset vs TrainSight-gated dataset."""
    # 1. Create a raw corrupted dataset with high variance, empty completions, and extreme length
    raw_samples = [
        {"prompt": "Short prompt", "completion": ""},  # Empty completion -> flat loss / wastes HBM
        {"prompt": "Unmasked prompt", "completion": "valid answer", "labels": [1, 2, 3, 4]},  # Missing -100 masking
        {"prompt": "Long prompt", "completion": "A " * 3000},  # 3000 tokens -> CUDA OOM risk
        {"prompt": "Normal prompt", "completion": "Normal completion text"}
    ]

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as raw_f:
        for s in raw_samples:
            raw_f.write(json.dumps(s) + "\n")
        raw_path = Path(raw_f.name)

    # 2. Inspect RAW dataset with TrainSight
    inspector = SFTInspector(max_seq_len_threshold=1000)
    raw_report = inspector.inspect_file(raw_path)

    # Verify TrainSight flags all failure modes
    assert raw_report.empty_completion_count > 0, "TrainSight must detect empty completions"
    assert raw_report.oom_risk_count > 0, "TrainSight must flag OOM risk samples > 1000 tokens"
    assert raw_report.loss_mask_misaligned_count > 0, "TrainSight must detect misaligned loss masks"
    assert len(raw_report.warnings) >= 3, "Raw dataset must produce multiple warnings"

    # 3. Create Sanitized dataset (TrainSight pipeline filtering)
    sanitized_samples = [
        s for s in raw_samples
        if s.get("completion")  # Remove empty completions
        and len((s.get("prompt", "") + s.get("completion", "")).split()) < 500  # Truncate OOM risk
    ]

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as clean_f:
        for s in sanitized_samples:
            clean_f.write(json.dumps(s) + "\n")
        clean_path = Path(clean_f.name)

    # 4. Inspect Clean dataset
    clean_report = inspector.inspect_file(clean_path)

    assert clean_report.empty_completion_count == 0
    assert clean_report.oom_risk_count == 0
    assert clean_report.phase_quadrant == "Clean Execution"

    # Cleanup
    raw_path.unlink()
    clean_path.unlink()


def test_with_vs_without_trainsight_dpo():
    """Empirical proof test comparing DPO dataset with preference reversals vs sanitized DPO dataset."""
    dpo_samples = [
        {"prompt": "Solve 2+2", "chosen": "4", "rejected": "5", "chosen_score": 1.0, "rejected_score": 0.0},
        {"prompt": "Solve 3+3", "chosen": "Wrong 5", "rejected": "Correct 6", "chosen_score": 0.0, "rejected_score": 1.0}, # Preference reversal!
        {"prompt": "Solve 4+4", "chosen": "8", "rejected": "8"}  # Identical pair -> zero gradient
    ]

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as dpo_f:
        for s in dpo_samples:
            dpo_f.write(json.dumps(s) + "\n")
        dpo_path = Path(dpo_f.name)

    inspector = DPOInspector()
    report = inspector.inspect_file(dpo_path)

    assert report.preference_reversal_count == 1, "TrainSight must detect preference reversals"
    assert report.identical_pairs_count == 1, "TrainSight must detect identical chosen/rejected pairs"

    dpo_path.unlink()
