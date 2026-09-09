from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field
import re


class Distinct2DiversityReport(BaseModel):
    """Encapsulates Distinct-2 N-gram metrics across prompt completions (RFC-004)."""
    total_bigrams: int
    unique_bigrams: int
    distinct_2_ratio: float
    is_collapsed: bool


class EntropyRemediationState(BaseModel):
    """Telemetry report when exploration annealing engages or steps down."""
    step: int
    is_annealing_active: bool
    current_temperature: float
    entropy_loss_bonus_coeff: float
    consecutive_collapsed_steps: int
    action_taken: str


class Distinct2DiversityMonitor:
    """Calculates Distinct-2 Bigram Diversity across candidate rollouts in < 1ms (RFC-004)."""

    def __init__(self, collapse_threshold: float = 0.20):
        self.collapse_threshold = collapse_threshold

    def evaluate_completions(self, completions: List[str]) -> Distinct2DiversityReport:
        """Computes unique bigram ratio over all completions for a prompt group."""
        all_bigrams = []
        for comp in completions:
            tokens = re.findall(r"\b\w+\b", comp.lower())
            if len(tokens) >= 2:
                bigrams = list(zip(tokens[:-1], tokens[1:]))
                all_bigrams.extend(bigrams)

        total_bigrams = len(all_bigrams)
        if total_bigrams == 0:
            return Distinct2DiversityReport(
                total_bigrams=0,
                unique_bigrams=0,
                distinct_2_ratio=0.0,
                is_collapsed=True,
            )

        unique_bigrams = len(set(all_bigrams))
        ratio = unique_bigrams / float(total_bigrams)

        return Distinct2DiversityReport(
            total_bigrams=total_bigrams,
            unique_bigrams=unique_bigrams,
            distinct_2_ratio=round(ratio, 4),
            is_collapsed=bool(ratio < self.collapse_threshold),
        )


class DynamicEntropyGovernor:
    """Detects silent policy entropy collapse and self-heals by annealing temperature (RFC-004)."""

    def __init__(
        self,
        base_temperature: float = 0.70,
        annealed_temperature: float = 1.15,
        entropy_bonus_coeff: float = 0.05,
        collapse_steps_threshold: int = 5,
        anneal_duration_steps: int = 150,
        recovery_diversity_threshold: float = 0.50,
    ):
        self.base_temperature = base_temperature
        self.annealed_temperature = annealed_temperature
        self.entropy_bonus_coeff = entropy_bonus_coeff
        self.collapse_steps_threshold = collapse_steps_threshold
        self.anneal_duration_steps = anneal_duration_steps
        self.recovery_diversity_threshold = recovery_diversity_threshold

        self.consecutive_collapsed_steps: int = 0
        self.is_annealing_active: bool = False
        self.annealing_remaining_steps: int = 0
        self.current_temperature: float = base_temperature
        self.current_entropy_bonus: float = 0.0

    def step(self, step_num: int, distinct_2_ratio: float) -> EntropyRemediationState:
        """Evaluates diversity ratio at current step and adjusts temperature/entropy bonus."""
        action = "Nominal operation"

        # Check for collapse
        if distinct_2_ratio < 0.20:
            self.consecutive_collapsed_steps += 1
        else:
            self.consecutive_collapsed_steps = 0

        # Trigger annealing if collapsed for threshold steps
        if self.consecutive_collapsed_steps >= self.collapse_steps_threshold and not self.is_annealing_active:
            self.is_annealing_active = True
            self.annealing_remaining_steps = self.anneal_duration_steps
            self.current_temperature = self.annealed_temperature
            self.current_entropy_bonus = self.entropy_bonus_coeff
            action = (
                f"🚨 Entropy collapse detected ({distinct_2_ratio:.2f} < 0.20 for {self.consecutive_collapsed_steps} steps). "
                f"Re-igniting exploration: Boosted temperature to {self.current_temperature:.2f} and entropy bonus to {self.current_entropy_bonus:.2f}."
            )
        elif self.is_annealing_active:
            self.annealing_remaining_steps -= 1

            # Check for recovery
            if distinct_2_ratio >= self.recovery_diversity_threshold or self.annealing_remaining_steps <= 0:
                self.is_annealing_active = False
                self.current_temperature = self.base_temperature
                self.current_entropy_bonus = 0.0
                action = (
                    f"✅ Diversity recovered ({distinct_2_ratio:.2f} >= {self.recovery_diversity_threshold:.2f}). "
                    f"Stepping temperature back down to {self.base_temperature:.2f}."
                )
            else:
                action = f"Annealing active ({self.annealing_remaining_steps} steps remaining). Temp={self.current_temperature:.2f}."

        return EntropyRemediationState(
            step=step_num,
            is_annealing_active=self.is_annealing_active,
            current_temperature=round(self.current_temperature, 2),
            entropy_loss_bonus_coeff=round(self.current_entropy_bonus, 3),
            consecutive_collapsed_steps=self.consecutive_collapsed_steps,
            action_taken=action,
        )
