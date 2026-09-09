from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field


class ValidationEvaluationResult(BaseModel):
    step: int
    validation_accuracy: float
    is_best: bool
    should_stop: bool
    reason: str


class ValidationOracle:
    """Governs model selection and early stopping strictly through held-out ground truth validation (RFC-002)."""

    def __init__(
        self,
        eval_interval_steps: int = 250,
        patience_steps: int = 500,
        min_improvement: float = 0.005,
    ):
        self.eval_interval_steps = eval_interval_steps
        self.patience_steps = patience_steps
        self.min_improvement = min_improvement

        self.best_accuracy = 0.0
        self.best_step = 0
        self.steps_since_improvement = 0
        self.history: List[ValidationEvaluationResult] = []

    def evaluate_step(
        self,
        step: int,
        eval_fn: Optional[Callable[[], float]] = None,
        precomputed_accuracy: Optional[float] = None,
    ) -> Optional[ValidationEvaluationResult]:
        """Runs validation evaluation if at interval, checking for plateau or new best."""
        if step % self.eval_interval_steps != 0 and step != 1:
            return None

        acc = precomputed_accuracy if precomputed_accuracy is not None else (eval_fn() if eval_fn else 0.0)

        is_best = False
        should_stop = False
        reason = "Normal validation step."

        if acc > self.best_accuracy + self.min_improvement:
            self.best_accuracy = acc
            self.best_step = step
            self.steps_since_improvement = 0
            is_best = True
            reason = f"New best validation accuracy: {acc * 100:.2f}% (Step {step})."
        else:
            self.steps_since_improvement += self.eval_interval_steps
            reason = f"No significant improvement. Stalled for {self.steps_since_improvement} steps."

            if self.steps_since_improvement >= self.patience_steps:
                should_stop = True
                reason = (
                    f"Early Stopping Triggered: Validation accuracy plateaued at "
                    f"{self.best_accuracy * 100:.1f}% for {self.patience_steps} steps. "
                    f"Automatically promoting checkpoint_best_accuracy.pt (from step {self.best_step})."
                )

        res = ValidationEvaluationResult(
            step=step,
            validation_accuracy=acc,
            is_best=is_best,
            should_stop=should_stop,
            reason=reason,
        )
        self.history.append(res)
        return res
