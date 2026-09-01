import sys
import subprocess
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from typing import Dict, Any


def _run_code_sandbox(code_snippet: str, timeout_seconds: float = 2.0) -> bool:
    """Executes a Python code snippet in an isolated subprocess with strict timeout."""
    try:
        res = subprocess.run(
            [sys.executable, "-c", code_snippet],
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )
        return res.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


class CodeVerifierReward:
    """Process-isolated verifier for Python code execution with timeout protection."""

    def __init__(self, timeout_seconds: float = 2.0, max_workers: int = 4):
        self.timeout_seconds = timeout_seconds
        self.executor = ProcessPoolExecutor(max_workers=max_workers)

    def evaluate_code(self, code_snippet: str) -> float:
        """Evaluates python code snippet safely in an isolated worker process."""
        if not code_snippet or not code_snippet.strip():
            return 0.0

        try:
            future = self.executor.submit(_run_code_sandbox, code_snippet, self.timeout_seconds)
            passed = future.result(timeout=self.timeout_seconds + 0.5)
            return 1.0 if passed else 0.0
        except (TimeoutError, Exception):
            return 0.0

    def compute_reward(self, completion: str, test_code: str) -> Dict[str, float]:
        """Combines completion python code with unit test code and evaluates execution."""
        full_script = f"{completion}\n\n{test_code}"
        score = self.evaluate_code(full_script)
        return {
            "total_reward": score,
            "code_pass_reward": score
        }
