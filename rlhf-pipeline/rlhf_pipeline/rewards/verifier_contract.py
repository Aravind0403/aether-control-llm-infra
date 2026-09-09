from typing import List, Tuple, Callable
import time


class VerifierContractError(RuntimeError):
    """Raised when the verifier fails canonical equivalence tests on container startup."""
    pass


# 15 Canonical equivalence edge cases across standard math notations (RFC-004)
GOLDEN_CONTRACT: List[Tuple[str, str]] = [
    ("3/4", "0.75"),
    ("1,000", "1000"),
    (r"\frac{1}{2}", "0.5"),
    ("3.0e5", "300000"),
    ("42%", "0.42"),
    ("x = 5", "5"),
    ("0.500", "0.5"),
    ("$100", "100"),
    ("2/3", "4/6"),
    ("0.25", "1/4"),
    ("1.5e-2", "0.015"),
    ("50 kg", "50"),
    ("10 / 2", "5"),
    (r"\frac{3}{4}", "0.75"),
    ("100.0", "100"),
]


class VerifierContract:
    """Self-testing verifier contract running at container boot in < 5ms (RFC-004)."""

    @classmethod
    def verify(cls, eval_accuracy_fn: Callable[[str, str], float], target_weight: float = 1.0) -> float:
        """Runs the 15 canonical edge-case pairs. Halts if any test fails."""
        start_time = time.perf_counter()
        passed = 0

        for raw_pred, target in GOLDEN_CONTRACT:
            wrapped_completion = f"<think>Reasoning...</think> <answer>{raw_pred}</answer>"
            score = eval_accuracy_fn(wrapped_completion, target)
            if score < target_weight:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                raise VerifierContractError(
                    f"🚨 VERIFIER BUG DETECTED: Verifier failed canonical equivalence contract!\n"
                    f"Sample: raw_pred='{raw_pred}', target='{target}' -> scored {score:.2f} (expected {target_weight:.2f}).\n"
                    f"Halting container initialization to prevent reward corruption and compute waste."
                )
            passed += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return elapsed_ms
