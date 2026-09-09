import pytest
from rlhf_pipeline.rewards.consensus_jury import ConsensusJury, JudgeEvaluation
from rlhf_pipeline.rewards.self_consistency import SelfConsistencyConsensus, CycleConsistencyVerifier
from rlhf_pipeline.rewards.fluency_barrier import ReferencePPLFluencyBarrier


def test_consensus_jury_strong_consensus():
    jury = ConsensusJury(uncertainty_penalty_lambda=2.0)
    evals = [
        JudgeEvaluation(judge_name="J1", review_cot="Solid.", score=0.92),
        JudgeEvaluation(judge_name="J2", review_cot="Great.", score=0.90),
        JudgeEvaluation(judge_name="J3", review_cot="Sound.", score=0.88),
    ]
    verdict = jury.arbitrate(evals)
    assert not verdict.is_controversial
    assert verdict.mean_score == pytest.approx(0.90, abs=0.01)
    assert verdict.disagreement_std < 0.05
    assert verdict.final_penalized_reward > 0.80


def test_consensus_jury_bayesian_penalty_on_hacking():
    jury = ConsensusJury(uncertainty_penalty_lambda=2.0)
    evals = [
        JudgeEvaluation(judge_name="J1", review_cot="Hacked high score.", score=0.98),
        JudgeEvaluation(judge_name="J2", review_cot="Gibberish.", score=0.40),
        JudgeEvaluation(judge_name="J3", review_cot="Nonsense.", score=0.35),
    ]
    verdict = jury.arbitrate(evals)
    assert verdict.is_controversial
    assert verdict.disagreement_std > 0.20
    # The penalty heavily suppresses the final reward!
    assert verdict.final_penalized_reward < 0.15


def test_self_consistency_consensus_clustering():
    cons = SelfConsistencyConsensus(supermajority_threshold=0.60)
    completions = [
        "<answer>breach of contract</answer>",
        "<answer>breach of contract</answer>",
        "<answer>breach of contract</answer>",
        "<answer>no breach</answer>",
    ]
    res = cons.cluster_completions(completions)
    assert res.is_consensus_established
    assert res.pseudo_ground_truth == "breach of contract"
    assert res.supermajority_ratio == 0.75
    assert len(res.winning_completion_indices) == 3
    assert len(res.divergent_completion_indices) == 1


def test_cycle_consistency_verifier():
    verifier = CycleConsistencyVerifier(overlap_threshold=0.70, reward_weight=1.0)
    symptoms = "patient has fever cough fatigue shortness of breath"
    recon_good = "fever fatigue cough shortness of breath"
    recon_bad = "patient has fracture on left arm"

    res_good = verifier.evaluate_cycle(symptoms, recon_good)
    assert res_good.cycle_closed
    assert res_good.reward == 1.0

    res_bad = verifier.evaluate_cycle(symptoms, recon_bad)
    assert not res_bad.cycle_closed
    assert res_bad.reward == 0.0


def test_reference_ppl_fluency_barrier():
    barrier = ReferencePPLFluencyBarrier(tau_natural=14.0, alpha_penalty=0.05)

    natural = "In the second quarter of the fiscal year, revenue increased significantly across all departments."
    soup = "Revenue. $90B. Cook CEO. Up. Apple."

    res_nat = barrier.evaluate(1.0, natural)
    assert not res_nat.is_keyword_soup
    assert res_nat.fluency_penalty == 0.0
    assert res_nat.final_reward == 1.0

    res_soup = barrier.evaluate(1.0, soup)
    assert res_soup.is_keyword_soup
    assert res_soup.fluency_penalty > 1.0
    assert res_soup.final_reward < 0.0
