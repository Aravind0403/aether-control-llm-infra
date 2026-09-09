from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field
import numpy as np


class JudgeEvaluation(BaseModel):
    judge_name: str
    review_cot: str
    score: float  # [0.0, 1.0]


class JuryVerdict(BaseModel):
    individual_scores: Dict[str, float]
    mean_score: float
    disagreement_std: float
    uncertainty_penalty: float
    final_penalized_reward: float
    is_controversial: bool
    summary: str


class ConsensusJury:
    """Multi-Judge Consensus Jury with Bayesian Uncertainty Penalization (RFC-005).
    
    Prevents policy models from exploiting out-of-distribution (OOD) reward hacking
    in single neural reward models by ensembling multiple distinct judges and
    penalizing reward proportionally to inter-judge disagreement:
    
    R_jury = mu_jury - lambda * sigma_jury
    """

    def __init__(self, uncertainty_penalty_lambda: float = 2.0, disagreement_threshold: float = 0.20):
        self.uncertainty_penalty_lambda = uncertainty_penalty_lambda
        self.disagreement_threshold = disagreement_threshold

    def arbitrate(self, evaluations: List[JudgeEvaluation]) -> JuryVerdict:
        """Arbitrates multiple judge evaluations, computing mean, std, and penalized reward."""
        if not evaluations:
            raise ValueError("Consensus jury requires at least one judge evaluation.")

        scores = [max(0.0, min(1.0, e.score)) for e in evaluations]
        arr = np.array(scores, dtype=np.float64)

        mean_score = float(np.mean(arr))
        std_score = float(np.std(arr)) if len(scores) > 1 else 0.0

        # Disagreement penalty
        penalty = self.uncertainty_penalty_lambda * std_score
        penalized_reward = max(0.0, mean_score - penalty)
        is_controversial = bool(std_score >= self.disagreement_threshold)

        judge_dict = {e.judge_name: round(e.score, 3) for e in evaluations}

        if is_controversial:
            summary = (
                f"⚠️ High Judge Disagreement (σ={std_score:.3f} >= {self.disagreement_threshold:.2f}). "
                f"Policy penalized: raw μ={mean_score:.3f} ──► penalized R={penalized_reward:.3f} "
                f"(penalty=-{penalty:.3f})."
            )
        else:
            summary = (
                f"✅ Strong Consensus (σ={std_score:.3f}). "
                f"Raw μ={mean_score:.3f} ──► awarded R={penalized_reward:.3f}."
            )

        return JuryVerdict(
            individual_scores=judge_dict,
            mean_score=round(mean_score, 4),
            disagreement_std=round(std_score, 4),
            uncertainty_penalty=round(penalty, 4),
            final_penalized_reward=round(penalized_reward, 4),
            is_controversial=is_controversial,
            summary=summary,
        )
