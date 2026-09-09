import re
from typing import Tuple, Dict, Any, Optional
import sympy

from rlhf_pipeline.rewards.verifier_contract import VerifierContract, VerifierContractError


def clean_math_expr(s: str) -> str:
    """Canonicalizes raw mathematical string into standard algebraic form (RFC-004)."""
    s = s.strip().lower()
    # Strip currency and commas
    s = s.replace("$", "").replace(",", "")
    # Strip variable equations e.g. 'x = 5' -> '5'
    s = re.sub(r"^[a-z]\s*=\s*", "", s)
    # Strip common trailing units
    s = re.sub(r"\s*(kg|m|cm|meters|dollars|usd)$", "", s)
    # Convert percentages e.g. '42%' -> '0.42'
    if s.endswith("%"):
        num = s[:-1].strip()
        try:
            return str(float(num) / 100.0)
        except Exception:
            return f"({num})/100"
    # Convert LaTeX fractions \frac{a}{b} -> (a)/(b)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
    return s.strip()


def is_math_equivalent(pred: str, target: str) -> bool:
    """Evaluates mathematical equivalence using SymPy symbolic simplification (RFC-004)."""
    cp = clean_math_expr(pred)
    ct = clean_math_expr(target)

    # 1. Direct string match
    if cp == ct:
        return True

    # 2. Direct float match
    try:
        if float(cp) == float(ct):
            return True
    except ValueError:
        pass

    # 3. SymPy symbolic equivalence: simplify(p - t) == 0
    try:
        p = sympy.sympify(cp)
        t = sympy.sympify(ct)
        diff = sympy.simplify(p - t)
        if diff == 0:
            return True
    except Exception:
        pass

    return False


class GRPOReasoningReward:
    """Rule-based reward evaluator for DeepSeek-R1 style reasoning models (GSM8K/MATH).
    
    Hardened with RFC-004:
    - Zero-Bloat Verifier Contract (<5ms self-test on boot)
    - SymPy Symbolic Canonicalization
    """

    def __init__(
        self,
        accuracy_weight: float = 1.0,
        format_weight: float = 0.5,
        enforce_contract: bool = True,
    ):
        self.accuracy_weight = accuracy_weight
        self.format_weight = format_weight

        # Run RFC-004 self-testing contract
        if enforce_contract:
            self.contract_ms = VerifierContract.verify(self.evaluate_accuracy_reward, self.accuracy_weight)
        else:
            self.contract_ms = 0.0

    def extract_reasoning_and_answer(self, completion: str) -> Tuple[str, str, bool]:
        """Extracts text inside <think>...</think> and <answer>...</answer> tags."""
        think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)

        think_text = think_match.group(1).strip() if think_match else ""
        answer_text = answer_match.group(1).strip() if answer_match else ""

        # Format is valid if both think and answer tags exist in proper order
        format_valid = bool(think_match and answer_match and think_match.start() < answer_match.start())
        return think_text, answer_text, format_valid

    def evaluate_format_reward(self, completion: str) -> float:
        """Returns format reward (+0.5 if valid <think>/<answer> tags exist, 0.0 otherwise)."""
        _, _, format_valid = self.extract_reasoning_and_answer(completion)
        return self.format_weight if format_valid else 0.0

    def evaluate_accuracy_reward(self, completion: str, ground_truth: str) -> float:
        """Returns accuracy reward (+1.0 if extracted answer matches ground truth via SymPy)."""
        _, extracted_answer, _ = self.extract_reasoning_and_answer(completion)

        # Fallback: if no <answer> tag, search for trailing numbers
        if not extracted_answer:
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", completion)
            extracted_answer = numbers[-1] if numbers else ""

        if is_math_equivalent(extracted_answer, ground_truth):
            return self.accuracy_weight

        return 0.0

    def compute_total_reward(self, completion: str, ground_truth: str) -> Dict[str, float]:
        """Computes combined format + accuracy reward scores."""
        r_format = self.evaluate_format_reward(completion)
        r_accuracy = self.evaluate_accuracy_reward(completion, ground_truth)
        r_total = r_format + r_accuracy

        return {
            "total_reward": r_total,
            "format_reward": r_format,
            "accuracy_reward": r_accuracy,
        }
