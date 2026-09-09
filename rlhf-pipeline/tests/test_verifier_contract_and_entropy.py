import pytest
from rlhf_pipeline.rewards.verifier_contract import VerifierContract, VerifierContractError, GOLDEN_CONTRACT
from rlhf_pipeline.rewards.math_reward import GRPOReasoningReward, is_math_equivalent
from rlhf_pipeline.governor.entropy_governor import Distinct2DiversityMonitor, DynamicEntropyGovernor


def test_golden_contract_passes():
    reward_eval = GRPOReasoningReward(enforce_contract=False)
    elapsed_ms = VerifierContract.verify(reward_eval.evaluate_accuracy_reward, 1.0)
    assert elapsed_ms < 100.0  # < 5ms typically


def test_broken_verifier_raises_contract_error():
    # Broken evaluator that fails on fractions
    def broken_eval(comp: str, target: str) -> float:
        if "/" in comp:
            return 0.0
        return 1.0

    with pytest.raises(VerifierContractError) as exc_info:
        VerifierContract.verify(broken_eval, 1.0)
    assert "🚨 VERIFIER BUG DETECTED" in str(exc_info.value)


def test_sympy_equivalence_edge_cases():
    assert is_math_equivalent("3/4", "0.75")
    assert is_math_equivalent(r"\frac{1}{2}", "0.5")
    assert is_math_equivalent("3.0e5", "300000")
    assert is_math_equivalent("42%", "0.42")
    assert is_math_equivalent("x = 5", "5")
    assert is_math_equivalent("$1,000", "1000")
    assert is_math_equivalent("50 kg", "50")


def test_distinct_2_diversity_monitor():
    monitor = Distinct2DiversityMonitor(collapse_threshold=0.20)

    diverse = [
        "Let us calculate the total sum of numbers step by step.",
        "First we add the two quantities together to find the overall answer.",
        "The problem requires direct addition of both variables.",
    ]
    rep_div = monitor.evaluate_completions(diverse)
    assert not rep_div.is_collapsed
    assert rep_div.distinct_2_ratio > 0.50

    collapsed = [
        "Wait double check the answer is 42."
    ] * 8
    rep_col = monitor.evaluate_completions(collapsed)
    assert rep_col.is_collapsed
    assert rep_col.distinct_2_ratio <= 0.125


def test_dynamic_entropy_governor_auto_annealing():
    gov = DynamicEntropyGovernor(
        base_temperature=0.70,
        annealed_temperature=1.15,
        collapse_steps_threshold=3,
        recovery_diversity_threshold=0.50,
    )

    # 2 healthy steps
    st1 = gov.step(1, 0.85)
    st2 = gov.step(2, 0.80)
    assert not st1.is_annealing_active
    assert st2.current_temperature == 0.70

    # 3 collapsed steps -> trigger annealing on 3rd!
    gov.step(3, 0.10)
    gov.step(4, 0.12)
    st5 = gov.step(5, 0.08)
    assert st5.is_annealing_active
    assert st5.current_temperature == 1.15
    assert st5.entropy_loss_bonus_coeff > 0.0

    # Recover diversity
    st6 = gov.step(6, 0.65)
    assert not st6.is_annealing_active
    assert st6.current_temperature == 0.70
