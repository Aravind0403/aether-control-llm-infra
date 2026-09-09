import pytest
from trainsight.simulators.collation_simulator import CollationSimulator


def test_collation_simulator_worst_case_shape():
    # 50% 100 tokens, 50% 4000 tokens
    seqs = ([100] * 50) + ([4000] * 50)
    simulator = CollationSimulator(num_trials=500, seed=42)

    result = simulator.simulate(seqs, batch_size=32)

    assert result.batch_size == 32
    assert result.worst_case_seq_len == 4000
    assert result.worst_case_batch_tokens == 32 * 4000  # 128,000 tokens
    assert result.padding_waste_pct > 25.0


def test_collation_simulator_token_budget_breach():
    seqs = [100, 200, 300, 4000]
    simulator = CollationSimulator(num_trials=200, seed=42)

    # Budget of 10,000 tokens for batch of 4 -> max batch tokens is 4 * 4000 = 16,000 > 10,000
    result = simulator.simulate(seqs, batch_size=4, token_budget_per_batch=10000)

    assert result.empirical_oom_probability > 0.0
    assert result.predicted_blast_step is not None
