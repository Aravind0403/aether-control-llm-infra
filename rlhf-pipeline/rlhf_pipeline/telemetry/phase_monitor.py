from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import numpy as np


class TriadStepRecord(BaseModel):
    """Decoupled Triad telemetry record for a single training step (RFC-002)."""
    step: int
    format_reward: float
    accuracy_reward: float
    composite_reward: float
    validation_accuracy: float
    average_thinking_length: float


class PhaseTransitionAlert(BaseModel):
    """Encapsulates alert payload when Phase 3 inflection occurs."""
    step: int
    format_drop_pct: float
    accuracy_gain_pct: float
    prev_format: float
    curr_format: float
    prev_val_acc: float
    curr_val_acc: float
    prev_think_len: float
    curr_think_len: float
    banner_message: str


class DecoupledTriadLogger:
    """Outlaws single-line reward logging by maintaining the Decoupled Triad over time."""

    def __init__(self):
        self.history: List[TriadStepRecord] = []

    def log_step(
        self,
        step: int,
        format_reward: float,
        accuracy_reward: float,
        validation_accuracy: float,
        average_thinking_length: float,
    ) -> TriadStepRecord:
        record = TriadStepRecord(
            step=step,
            format_reward=format_reward,
            accuracy_reward=accuracy_reward,
            composite_reward=format_reward + accuracy_reward,
            validation_accuracy=validation_accuracy,
            average_thinking_length=average_thinking_length,
        )
        self.history.append(record)
        return record


class PhaseTransitionDetector:
    """Monitors learning dynamics to detect creative destruction & format break-out (RFC-002)."""

    def __init__(
        self,
        format_drop_threshold: float = 0.15,
        accuracy_gain_threshold: float = 0.03,
        window_size: int = 5,
    ):
        self.format_drop_threshold = format_drop_threshold
        self.accuracy_gain_threshold = accuracy_gain_threshold
        self.window_size = window_size
        self.alert_triggered = False

    def evaluate(self, history: List[TriadStepRecord]) -> Optional[PhaseTransitionAlert]:
        """Evaluates whether the current step is experiencing a Phase 3 transition."""
        if len(history) < self.window_size * 2:
            return None

        recent = history[-self.window_size:]
        baseline = history[-self.window_size * 2 : -self.window_size]

        avg_baseline_format = float(np.mean([r.format_reward for r in baseline]))
        avg_recent_format = float(np.mean([r.format_reward for r in recent]))

        avg_baseline_acc = float(np.mean([r.validation_accuracy for r in baseline]))
        avg_recent_acc = float(np.mean([r.validation_accuracy for r in recent]))

        avg_baseline_len = float(np.mean([r.average_thinking_length for r in baseline]))
        avg_recent_len = float(np.mean([r.average_thinking_length for r in recent]))

        # Check condition: Format reward drops significantly WHILE validation accuracy rises
        format_delta = avg_recent_format - avg_baseline_format
        format_drop_pct = (abs(format_delta) / (avg_baseline_format + 1e-8)) if format_delta < 0 else 0.0
        acc_delta = avg_recent_acc - avg_baseline_acc

        if format_delta < -self.format_drop_threshold and acc_delta >= self.accuracy_gain_threshold:
            curr_step = recent[-1].step
            banner = self.format_banner(
                step=curr_step,
                prev_format=avg_baseline_format,
                curr_format=avg_recent_format,
                prev_acc=avg_baseline_acc,
                curr_acc=avg_recent_acc,
                prev_len=avg_baseline_len,
                curr_len=avg_recent_len,
            )
            self.alert_triggered = True
            return PhaseTransitionAlert(
                step=curr_step,
                format_drop_pct=format_drop_pct * 100.0,
                accuracy_gain_pct=(acc_delta / (avg_baseline_acc + 1e-8)) * 100.0,
                prev_format=avg_baseline_format,
                curr_format=avg_recent_format,
                prev_val_acc=avg_baseline_acc,
                curr_val_acc=avg_recent_acc,
                prev_think_len=avg_baseline_len,
                curr_think_len=avg_recent_len,
                banner_message=banner,
            )

        return None

    @staticmethod
    def format_banner(
        step: int,
        prev_format: float,
        curr_format: float,
        prev_acc: float,
        curr_acc: float,
        prev_len: float,
        curr_len: float,
    ) -> str:
        format_delta_pct = ((curr_format - prev_format) / (prev_format + 1e-8)) * 100.0
        acc_delta_pct = ((curr_acc - prev_acc) / (prev_acc + 1e-8)) * 100.0
        len_delta_pct = ((curr_len - prev_len) / (prev_len + 1e-8)) * 100.0

        return (
            "================================================================================\n"
            f"🚀 RLHF-OBSERVABILITY ALERT: PHASE-TRANSITION IN PROGRESS (Step {step})\n"
            "================================================================================\n"
            f"Metric                      Baseline                  Current                   Delta\n"
            "────────────────────────────────────────────────────────────────────────────────\n"
            f"Format Reward (<think>)     {prev_format:8.2f}                  {curr_format:8.2f}              {format_delta_pct:+6.1f}% ⚠️\n"
            f"Factual Math Accuracy       {prev_acc*100:7.1f}%                  {curr_acc*100:7.1f}%              {acc_delta_pct:+6.1f}% 🚀\n"
            f"Average Thinking Length     {prev_len:6.0f} tokens             {curr_len:6.0f} tokens           {len_delta_pct:+6.1f}% 🧠\n"
            "────────────────────────────────────────────────────────────────────────────────\n"
            "💡 OPERATOR NOTICE:\n"
            "\"DO NOT KILL THIS JOB. The model has broken out of the shallow format-overfitting\n"
            "local optimum. It is trading surface tag compliance for longer, deeper reasoning paths.\n"
            f"Factual correctness is UP {(curr_acc - prev_acc)*100:+.1f}%. This drop in total reward is healthy and expected.\"\n"
            "================================================================================"
        )
