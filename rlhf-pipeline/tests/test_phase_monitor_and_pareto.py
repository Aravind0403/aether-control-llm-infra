import pytest
import tempfile
import shutil
from pathlib import Path

from rlhf_pipeline.telemetry.phase_monitor import DecoupledTriadLogger, PhaseTransitionDetector
from rlhf_pipeline.checkpointing.pareto_checkpointer import ParetoCheckpointer
from rlhf_pipeline.telemetry.validation_oracle import ValidationOracle


def test_phase_transition_detector_banner():
    logger = DecoupledTriadLogger()
    detector = PhaseTransitionDetector(format_drop_threshold=0.15, accuracy_gain_threshold=0.03, window_size=5)

    # 5 steps of Phase 2
    for s in range(1, 6):
        logger.log_step(step=s, format_reward=0.50, accuracy_reward=0.35, validation_accuracy=0.35, average_thinking_length=180.0)

    # 5 steps of Phase 3 (format drops to 0.30, accuracy surges to 0.42)
    alert = None
    for s in range(6, 11):
        logger.log_step(step=s, format_reward=0.30, accuracy_reward=0.42, validation_accuracy=0.42, average_thinking_length=420.0)
        curr_alert = detector.evaluate(logger.history)
        if curr_alert:
            alert = curr_alert

    assert alert is not None
    assert "DO NOT KILL THIS JOB" in alert.banner_message
    assert alert.curr_val_acc > alert.prev_val_acc
    assert alert.curr_format < alert.prev_format


def test_pareto_checkpointer_classes():
    temp_dir = tempfile.mkdtemp()
    try:
        checkpointer = ParetoCheckpointer(checkpoint_dir=temp_dir)

        # Step 1: Baseline
        saved_1 = checkpointer.save_step(step=100, model_state={"layer": "w1"}, validation_accuracy=0.20, format_score=0.50)
        assert "best_accuracy" in saved_1
        assert "best_format" in saved_1
        assert "balanced_pareto" in saved_1
        assert "latest" in saved_1

        # Step 2: Format ATH, low accuracy
        saved_2 = checkpointer.save_step(step=200, model_state={"layer": "w2"}, validation_accuracy=0.22, format_score=0.99)
        assert "best_format" in saved_2
        assert checkpointer.best_format == 0.99

        # Step 3: Accuracy ATH, low format
        saved_3 = checkpointer.save_step(step=300, model_state={"layer": "w3"}, validation_accuracy=0.75, format_score=0.40)
        assert "best_accuracy" in saved_3
        assert "best_format" not in saved_3
        assert checkpointer.best_accuracy == 0.75

        # Manifest table rows
        rows = checkpointer.get_summary_table()
        assert len(rows) >= 3
    finally:
        shutil.rmtree(temp_dir)


def test_validation_oracle_early_stopping():
    oracle = ValidationOracle(eval_interval_steps=100, patience_steps=300, min_improvement=0.01)

    # Step 100: Best
    r100 = oracle.evaluate_step(100, precomputed_accuracy=0.40)
    assert r100.is_best
    assert not r100.should_stop

    # Step 200: Stall 1
    r200 = oracle.evaluate_step(200, precomputed_accuracy=0.402)  # < min_improvement
    assert not r200.is_best
    assert not r200.should_stop

    # Step 300: Stall 2
    r300 = oracle.evaluate_step(300, precomputed_accuracy=0.401)
    assert not r300.should_stop

    # Step 400: Stall 3 (reaches 300 steps patience) -> Early stopping triggered!
    r400 = oracle.evaluate_step(400, precomputed_accuracy=0.39)
    assert r400.should_stop
    assert "Early Stopping Triggered" in r400.reason
