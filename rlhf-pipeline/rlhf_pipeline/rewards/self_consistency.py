from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
from pydantic import BaseModel, Field
import re


class ConsensusClusterResult(BaseModel):
    """Result of unsupervised semantic clustering over candidate rollouts (RFC-005)."""
    pseudo_ground_truth: str
    cluster_counts: Dict[str, int]
    supermajority_ratio: float
    is_consensus_established: bool
    winning_completion_indices: List[int]
    divergent_completion_indices: List[int]


class CycleConsistencyResult(BaseModel):
    """Result of bidirectional reconstruction verification (RFC-005)."""
    original_input: str
    reconstructed_input: str
    token_overlap_ratio: float
    cycle_closed: bool
    reward: float


class SelfConsistencyConsensus:
    """Extracts pseudo-ground-truth from unsupervised candidate rollouts (RFC-005)."""

    def __init__(self, supermajority_threshold: float = 0.60):
        self.supermajority_threshold = supermajority_threshold

    @staticmethod
    def extract_conclusion(completion: str) -> str:
        """Extracts boxed or tagged conclusion, or normalizes text."""
        ans = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        if ans:
            return ans.group(1).strip().lower()
        # Fallback: extract last sentence or phrase
        sentences = [s.strip() for s in completion.strip().split(".") if s.strip()]
        return sentences[-1].lower() if sentences else completion.strip().lower()

    def cluster_completions(self, completions: List[str]) -> ConsensusClusterResult:
        """Clusters G completions by extracted conclusion to determine pseudo-ground-truth."""
        conclusions = [self.extract_conclusion(c) for c in completions]
        counts = Counter(conclusions)

        most_common, top_count = counts.most_common(1)[0]
        supermajority_ratio = top_count / float(len(completions))

        is_consensus = supermajority_ratio >= self.supermajority_threshold

        winning_indices = [i for i, c in enumerate(conclusions) if c == most_common]
        divergent_indices = [i for i, c in enumerate(conclusions) if c != most_common]

        return ConsensusClusterResult(
            pseudo_ground_truth=most_common,
            cluster_counts=dict(counts),
            supermajority_ratio=round(supermajority_ratio, 3),
            is_consensus_established=is_consensus,
            winning_completion_indices=winning_indices,
            divergent_completion_indices=divergent_indices,
        )


class CycleConsistencyVerifier:
    """Verifies reasoning quality by measuring bidirectional cycle consistency (RFC-005).
    
    Step 1: Input X -> Plan/Diagnosis Y
    Step 2: Plan/Diagnosis Y -> Reconstructed Input X'
    Verification: Token F1 overlap between X and X' >= threshold -> +1.0 Reward
    """

    def __init__(self, overlap_threshold: float = 0.70, reward_weight: float = 1.0):
        self.overlap_threshold = overlap_threshold
        self.reward_weight = reward_weight

    def evaluate_cycle(self, original_input: str, reconstructed_input: str) -> CycleConsistencyResult:
        """Calculates token overlap between original and reverse-reconstructed input."""
        orig_tokens = set(re.findall(r"\b\w+\b", original_input.lower()))
        recon_tokens = set(re.findall(r"\b\w+\b", reconstructed_input.lower()))

        if not orig_tokens:
            overlap = 1.0 if not recon_tokens else 0.0
        else:
            intersection = orig_tokens.intersection(recon_tokens)
            overlap = len(intersection) / float(len(orig_tokens))

        cycle_closed = bool(overlap >= self.overlap_threshold)
        reward = self.reward_weight if cycle_closed else 0.0

        return CycleConsistencyResult(
            original_input=original_input,
            reconstructed_input=reconstructed_input,
            token_overlap_ratio=round(overlap, 3),
            cycle_closed=cycle_closed,
            reward=reward,
        )
