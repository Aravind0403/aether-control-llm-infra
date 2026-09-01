import pytest
from pathlib import Path
from trainsight.inspectors.sft_inspector import SFTInspector


def test_sft_inspector_basic(tmp_path: Path):
    dataset_file = tmp_path / "test_data.jsonl"
    dataset_file.write_text(
        '{"prompt": "Hello", "completion": "World"}\n'
        '{"prompt": "Foo", "completion": "Bar"}\n'
    )

    inspector = SFTInspector(max_seq_len_threshold=100)
    report = inspector.inspect_file(dataset_file)

    assert report.total_samples == 2
    assert report.duplicate_count == 0
    assert report.empty_completion_count == 0
    assert report.oom_risk_count == 0


def test_sft_inspector_detects_duplicates_and_empty(tmp_path: Path):
    dataset_file = tmp_path / "test_data.jsonl"
    dataset_file.write_text(
        '{"prompt": "Hello", "completion": "World"}\n'
        '{"prompt": "Hello", "completion": "Different completion"}\n'
        '{"prompt": "Empty test", "completion": ""}\n'
    )

    inspector = SFTInspector(max_seq_len_threshold=100)
    report = inspector.inspect_file(dataset_file)

    assert report.total_samples == 3
    assert report.duplicate_count == 1
    assert report.empty_completion_count == 1
    assert len(report.warnings) >= 2


def test_sft_inspector_drift_detection(tmp_path: Path):
    dataset_file = tmp_path / "test_data.jsonl"
    dataset_file.write_text(
        '{"prompt": "Short", "completion": "A"}\n' * 50
    )

    inspector = SFTInspector()
    # Baseline with long prompts
    baseline_seqs = [1000] * 50
    report = inspector.inspect_file(dataset_file, baseline_seq_lengths=baseline_seqs)

    assert report.drift_alert is True
    assert report.ks_p_value < 0.05

