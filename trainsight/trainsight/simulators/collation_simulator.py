import random
from typing import List, Optional
import numpy as np
from pydantic import BaseModel, Field


class CollationSimulationResult(BaseModel):
    batch_size: int
    num_trials: int
    worst_case_seq_len: int
    worst_case_batch_tokens: int
    avg_batch_tokens: float
    p95_batch_tokens: float
    p99_batch_tokens: float
    p99_9_batch_tokens: float
    padding_waste_pct: float
    unpadded_avg_tokens: float
    max_token_budget: Optional[int] = None
    empirical_oom_probability: float = 0.0
    predicted_blast_step: Optional[int] = None


class CollationSimulator:
    """Monte-Carlo batch collation replay engine.
    
    Simulates real PyTorch DataLoader collation across K random shuffles
    to discover worst-case sequence pairing and activation explosion risk.
    """

    def __init__(self, num_trials: int = 1000, seed: Optional[int] = 42):
        self.num_trials = num_trials
        self.seed = seed

    def simulate(
        self,
        seq_lengths: List[int],
        batch_size: int,
        token_budget_per_batch: Optional[int] = None,
        custom_seed: Optional[int] = None,
    ) -> CollationSimulationResult:
        """Runs Monte-Carlo collation simulation across sequence lengths."""
        if not seq_lengths:
            raise ValueError("Sequence lengths list cannot be empty for simulation.")
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")

        rng = random.Random(custom_seed if custom_seed is not None else self.seed)

        n_samples = len(seq_lengths)
        effective_k = min(n_samples, batch_size)

        batch_token_volumes: List[int] = []
        batch_worst_lens: List[int] = []
        batch_unpadded_tokens: List[int] = []

        # Run Monte-Carlo sampling of batches
        for _ in range(self.num_trials):
            # Sample micro-batch
            if n_samples >= effective_k:
                batch = rng.sample(seq_lengths, effective_k)
            else:
                batch = [rng.choice(seq_lengths) for _ in range(effective_k)]

            max_len_in_batch = max(batch)
            unpadded_sum = sum(batch)
            # PyTorch DataCollatorWithPadding pads every sequence in the micro-batch to max_len_in_batch
            padded_tokens = max_len_in_batch * batch_size

            batch_token_volumes.append(padded_tokens)
            batch_worst_lens.append(max_len_in_batch)
            batch_unpadded_tokens.append(unpadded_sum)

        batch_tokens_arr = np.array(batch_token_volumes)
        worst_tokens = int(np.max(batch_tokens_arr))
        worst_len = int(np.max(batch_worst_lens))
        avg_tokens = float(np.mean(batch_tokens_arr))
        p95_tokens = float(np.percentile(batch_tokens_arr, 95))
        p99_tokens = float(np.percentile(batch_tokens_arr, 99))
        p99_9_tokens = float(np.percentile(batch_tokens_arr, 99.9))

        avg_unpadded = float(np.mean(batch_unpadded_tokens))
        padding_waste = float(max(0.0, ((avg_tokens - avg_unpadded) / avg_tokens) * 100)) if avg_tokens > 0 else 0.0

        # Empirical OOM calculation against a token budget if provided
        oom_prob = 0.0
        predicted_blast_step = None
        if token_budget_per_batch is not None and token_budget_per_batch > 0:
            breaches = np.sum(batch_tokens_arr > token_budget_per_batch)
            oom_prob = float(breaches / len(batch_token_volumes))
            if oom_prob > 0.0:
                # Geometric expectation: 1 / p
                predicted_blast_step = max(1, int(round(1.0 / oom_prob)))

        return CollationSimulationResult(
            batch_size=batch_size,
            num_trials=self.num_trials,
            worst_case_seq_len=worst_len,
            worst_case_batch_tokens=worst_tokens,
            avg_batch_tokens=avg_tokens,
            p95_batch_tokens=p95_tokens,
            p99_batch_tokens=p99_tokens,
            p99_9_batch_tokens=p99_9_tokens,
            padding_waste_pct=padding_waste,
            unpadded_avg_tokens=avg_unpadded,
            max_token_budget=token_budget_per_batch,
            empirical_oom_probability=oom_prob,
            predicted_blast_step=predicted_blast_step,
        )
