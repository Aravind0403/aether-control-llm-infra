import pytest
from rlhf_pipeline.rewards.code_reward import CodeVerifierReward


def test_code_verifier_reward_pass():
    verifier = CodeVerifierReward(timeout_seconds=2.0)
    code = "def add(a, b): return a + b"
    test = "assert add(2, 3) == 5"
    res = verifier.compute_reward(code, test)
    assert res["total_reward"] == 1.0


def test_code_verifier_reward_timeout():
    verifier = CodeVerifierReward(timeout_seconds=1.0)
    code = "import time\ntime.sleep(5)"
    test = "assert True"
    res = verifier.compute_reward(code, test)
    assert res["total_reward"] == 0.0, "Hanging code execution must time out and return 0.0 reward"
