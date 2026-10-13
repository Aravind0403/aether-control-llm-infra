import tempfile
import json
from pathlib import Path
import pytest
from transformers import AutoTokenizer
from trainsight.inspectors.sft_inspector import SFTInspector


@pytest.fixture
def qwen_tokenizer_name():
    """Provides tokenizer name if accessible locally or in cache, else skips clean-pass test."""
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    try:
        AutoTokenizer.from_pretrained(model_name)
        return model_name
    except Exception as e:
        pytest.skip(f"Qwen tokenizer is not available in local environment: {e}")


def test_init_container_gate_blocks_on_corrupt_data(qwen_tokenizer_name):
    """Validates the defensive InitContainer gate contract:
    
    1. Clean data passes and allows downstream container to execute (exit 0).
    2. Corrupted data triggers validation failure, blocking pod execution (exit 1).
    """
    inspector = SFTInspector(max_seq_len_threshold=500, tokenizer_name=qwen_tokenizer_name)

    # 1. Clean dataset scenario -> Gate passes
    clean_data = [
        {"prompt": "Hello world", "completion": "Hello there!"}
    ]
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as f:
        for row in clean_data:
            f.write(json.dumps(row) + "\n")
        clean_file = Path(f.name)

    clean_report = inspector.inspect_file(clean_file)
    assert clean_report.empty_completion_count == 0
    assert clean_report.phase_quadrant == "Clean Execution"
    assert len(clean_report.warnings) == 0
    clean_file.unlink()

    # 2. Corrupted dataset scenario (empty completion + excessive length) -> Gate trips
    corrupt_data = [
        {"prompt": "Valid prompt", "completion": ""},  # Empty completion violates gate policy
        {"prompt": "Long prompt", "completion": "token " * 800}  # Breaches 500-token threshold
    ]
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as f:
        for row in corrupt_data:
            f.write(json.dumps(row) + "\n")
        corrupt_file = Path(f.name)

    corrupt_report = inspector.inspect_file(corrupt_file)
    
    # Verify that the gate triggers rejection
    assert corrupt_report.empty_completion_count > 0, "Gate must detect empty completions"
    assert corrupt_report.oom_risk_count > 0, "Gate must detect length threshold violations"
    assert corrupt_report.phase_quadrant != "Clean Execution"
    
    # In an InitContainer shell script, this maps to:
    # if [ "$phase_quadrant" != "Clean Execution" ] || [ $empty_count -gt 0 ] || [ $oom_count -gt 0 ]; then exit 1; fi
    is_blocked = (corrupt_report.phase_quadrant != "Clean Execution" or corrupt_report.empty_completion_count > 0 or corrupt_report.oom_risk_count > 0)
    assert is_blocked is True, "InitContainer gate must halt pod startup on corrupted data"

    corrupt_file.unlink()


def test_init_container_gate_blocks_on_unavailable_tokenizer(tmp_path: Path):
    """Asserts that an unavailable tokenizer blocks the gate. A missing tokenizer must never be treated as a clean inspection."""
    dataset_file = tmp_path / "clean_data.jsonl"
    dataset_file.write_text('{"prompt": "Hello world", "completion": "Hello there!"}\n')

    # Inspector instantiated with unavailable / unresolvable tokenizer
    inspector = SFTInspector(tokenizer_name="non_existent/unavailable_tokenizer_probe_12345")
    report = inspector.inspect_file(dataset_file)

    # Invariant: missing tokenizer must fail closed and never report Clean Execution
    assert report.phase_quadrant == "Unable to Inspect"
    assert report.phase_quadrant != "Clean Execution"
    assert len(report.warnings) > 0
    assert "Unable to inspect" in report.warnings[0]

    # In InitContainer logic (or CLI --strict), this condition halts the pod
    is_blocked = (report.phase_quadrant != "Clean Execution" or len(report.warnings) > 0)
    assert is_blocked is True, "InitContainer gate must halt pod startup when tokenizer is unavailable"
