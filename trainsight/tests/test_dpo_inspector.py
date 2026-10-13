import pytest
from pathlib import Path
from trainsight.inspectors.dpo_inspector import DPOInspector


def test_dpo_inspector_identical_pair(tmp_path: Path):
    dpo_file = tmp_path / "test_dpo.jsonl"
    dpo_file.write_text(
        '{"prompt": "Q1", "chosen": "Same text", "rejected": "Same text"}\n'
        '{"prompt": "Q2", "chosen": "Good answer", "rejected": "Bad answer"}\n'
    )

    inspector = DPOInspector()
    report = inspector.inspect_file(dpo_file)

    assert report.total_samples == 2
    assert report.identical_pairs_count == 1
    assert len(report.warnings) >= 1
    assert "IDENTICAL" in report.warnings[0]
    assert report.phase_quadrant == "Defects Detected"


def test_dpo_inspector_near_identical(tmp_path: Path):
    dpo_file = tmp_path / "test_dpo.jsonl"
    dpo_file.write_text(
        '{"prompt": "Q1", "chosen": "The quick brown fox jumps over dog", "rejected": "The quick brown fox leaps over dog"}\n'
    )

    inspector = DPOInspector(jaccard_similarity_threshold=0.7)
    report = inspector.inspect_file(dpo_file)

    assert report.total_samples == 1
    assert report.near_identical_count == 1
    assert any("near-identical" in a for a in report.advisories)


def test_dpo_inspector_detects_preference_reversal(tmp_path: Path):
    """Proves deliberate score inversion (chosen < rejected) is flagged as an actionable defect."""
    dpo_file = tmp_path / "reversal_dpo.jsonl"
    dpo_file.write_text(
        '{"prompt": "Explain gravity", '
        '"chosen": [{"role": "user", "content": "Explain gravity"}, {"role": "assistant", "content": "A force."}], '
        '"rejected": [{"role": "user", "content": "Explain gravity"}, {"role": "assistant", "content": "Gravity is the curvature of spacetime described by Einstein."}], '
        '"score_chosen": 3.5, "score_rejected": 9.0}\n'
    )

    inspector = DPOInspector()
    report = inspector.inspect_file(dpo_file)

    assert report.total_samples == 1
    assert report.preference_reversal_count == 1
    assert any("REVERSED" in w for w in report.warnings)
    assert report.phase_quadrant == "Defects Detected"


def test_dpo_inspector_detects_prompt_mismatch(tmp_path: Path):
    """Proves corrupt multi-turn messages where user content disagrees with prompt are detected."""
    dpo_file = tmp_path / "mismatch_dpo.jsonl"
    dpo_file.write_text(
        '{"prompt": "Write Python code", '
        '"chosen": [{"role": "user", "content": "Write Javascript code"}, {"role": "assistant", "content": "console.log(1);"}] ,'
        '"rejected": [{"role": "user", "content": "Write Python code"}, {"role": "assistant", "content": "print(1)"}], '
        '"score_chosen": 8.0, "score_rejected": 5.0}\n'
    )

    inspector = DPOInspector()
    report = inspector.inspect_file(dpo_file)

    assert report.total_samples == 1
    assert report.prompt_mismatch_count == 1
    assert any("do not match" in w for w in report.warnings)
    assert report.phase_quadrant == "Defects Detected"


def test_dpo_inspector_detects_missing_assistant_response(tmp_path: Path):
    """Proves empty or corrupted assistant completions are flagged."""
    dpo_file = tmp_path / "empty_dpo.jsonl"
    dpo_file.write_text(
        '{"prompt": "Tell me a joke", '
        '"chosen": [{"role": "user", "content": "Tell me a joke"}, {"role": "assistant", "content": ""}], '
        '"rejected": [{"role": "user", "content": "Tell me a joke"}, {"role": "assistant", "content": "A funny joke."}], '
        '"score_chosen": 8.0, "score_rejected": 5.0}\n'
    )

    inspector = DPOInspector()
    report = inspector.inspect_file(dpo_file)

    assert report.empty_pair_count == 1
    assert any("missing" in w for w in report.warnings)
    assert report.phase_quadrant == "Defects Detected"


def test_dpo_inspector_records_score_ties_as_advisory(tmp_path: Path):
    """Proves tied quality scores are treated as confidence advisories, not false-alarm zero-gradient defects."""
    dpo_file = tmp_path / "tied_dpo.jsonl"
    dpo_file.write_text(
        '{"prompt": "Solve 2+2", '
        '"chosen": [{"role": "user", "content": "Solve 2+2"}, {"role": "assistant", "content": "The answer to 2+2 is 4."}], '
        '"rejected": [{"role": "user", "content": "Solve 2+2"}, {"role": "assistant", "content": "The result of 2+2 is 4."}], '
        '"score_chosen": 8.5, "score_rejected": 8.5}\n'
    )

    inspector = DPOInspector()
    report = inspector.inspect_file(dpo_file)

    assert report.total_samples == 1
    assert report.score_tie_count == 1
    assert report.preference_reversal_count == 0
    assert report.identical_pairs_count == 0
    assert any("TIED" in a for a in report.advisories)
    # Distinct responses with tied scores and balanced lengths should pass clean execution
    assert report.phase_quadrant == "Clean Execution"
    assert len(report.warnings) == 0


def test_dpo_inspector_detects_severe_length_bias(tmp_path: Path):
    """Proves severe length bias (> 1.8 ratio) triggers a quality warning."""
    dpo_file = tmp_path / "bias_dpo.jsonl"
    long_text = "word " * 300
    short_text = "word " * 10
    dpo_file.write_text(
        f'{{"prompt": "Test", "chosen": "{long_text}", "rejected": "{short_text}"}}\n'
    )

    inspector = DPOInspector()
    report = inspector.inspect_file(dpo_file)

    assert report.length_bias_ratio > 1.8
    assert any("Severe length bias" in w for w in report.warnings)
    assert report.phase_quadrant == "Quality Warnings"


def test_dpo_inspector_fails_closed_on_missing_tokenizer(tmp_path: Path):
    """Proves inspector fails closed with 'Unable to Inspect' if requested tokenizer is unavailable."""
    dpo_file = tmp_path / "valid_dpo.jsonl"
    dpo_file.write_text('{"prompt": "Q", "chosen": "A", "rejected": "B"}\n')

    inspector = DPOInspector(tokenizer_name="nonexistent/unreachable_tokenizer_model_123")
    report = inspector.inspect_file(dpo_file)

    assert report.phase_quadrant == "Unable to Inspect"
    assert any("could not be loaded" in w for w in report.warnings)
    assert report.avg_chosen_len == 0.0


def test_dpo_inspector_reconciles_ultrafeedback_1000():
    """
    Milestone Reconciliation Test:
    Verifies DPOInspector exact output on the 1,000 deterministic slice of UltraFeedback Binarized:
    - 1,000 pairs evaluated
    - 7 identical assistant responses (zero-gradient defects)
    - 126 score ties (informational advisories)
    - 0 score reversals
    - 16 over-budget prompts (> 1024 tokens)
    - 0 prompt mismatches or missing fields
    """
    dataset_path = Path("trainsight/sample_data/real_ultrafeedback_1000.jsonl")
    assert dataset_path.exists(), f"Sample dataset missing at {dataset_path}"

    inspector = DPOInspector(tokenizer_name="Qwen/Qwen2.5-1.5B-Instruct")
    report = inspector.inspect_file(dataset_path)

    assert report.total_samples == 1000
    assert report.identical_pairs_count == 7
    assert report.score_tie_count == 126
    assert report.preference_reversal_count == 0
    assert report.over_budget_prompts_count == 16
    assert report.over_budget_completions_count == 0
    assert report.prompt_mismatch_count == 0
    assert report.empty_pair_count == 0

    assert report.avg_prompt_len == pytest.approx(186.6, abs=0.5)
    assert report.avg_chosen_len == pytest.approx(265.5, abs=0.5)
    assert report.avg_rejected_len == pytest.approx(233.7, abs=0.5)
    assert report.length_bias_ratio == pytest.approx(1.14, abs=0.02)

    assert report.phase_quadrant == "Defects Detected"
    assert any("7 pairs have IDENTICAL" in w for w in report.warnings)
    assert any("126 pairs have TIED" in a for a in report.advisories)
    assert any("16 prompts exceed" in a for a in report.advisories)
