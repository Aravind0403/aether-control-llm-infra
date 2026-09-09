import pytest
from rlhf_pipeline.grpo_trainer import (
    GRPOTrainer,
    GRPOTrainingConfig,
    TeacherAnchorInjector,
)


def test_standard_intra_group_advantage():
    trainer = GRPOTrainer(GRPOTrainingConfig(group_size=4))
    rewards = [1.5, 0.5, 0.5, 0.0]
    advs, m, s, mode = trainer.compute_hierarchical_advantages(rewards)
    assert mode == "intra_group"
    assert s > 0
    assert advs[0] > 0  # Highest reward gets positive advantage
    assert advs[-1] < 0  # Lowest reward gets negative advantage


def test_zero_variance_incompetence_fallback():
    trainer = GRPOTrainer(GRPOTrainingConfig(group_size=4))
    # All failed [0, 0, 0, 0]
    rewards = [0.0, 0.0, 0.0, 0.0]
    batch_mean = 0.50
    batch_std = 0.50
    advs, m, s, mode = trainer.compute_hierarchical_advantages(
        rewards, batch_mean=batch_mean, batch_std=batch_std
    )
    assert mode == "batch_fallback"
    assert s == 0.0
    # Incompetence group gets negative advantage relative to batch!
    for a in advs:
        assert a == pytest.approx(-1.0, abs=1e-4)


def test_zero_variance_mastery_fallback():
    trainer = GRPOTrainer(GRPOTrainingConfig(group_size=4))
    # All passed [1.5, 1.5, 1.5, 1.5]
    rewards = [1.5, 1.5, 1.5, 1.5]
    batch_mean = 0.50
    batch_std = 0.50
    advs, m, s, mode = trainer.compute_hierarchical_advantages(
        rewards, batch_mean=batch_mean, batch_std=batch_std
    )
    assert mode == "batch_fallback"
    assert s == 0.0
    # Mastery group gets positive reinforcement relative to batch!
    for a in advs:
        assert a == pytest.approx(2.0, abs=1e-4)


def test_teacher_anchor_injection():
    injector = TeacherAnchorInjector(threshold_steps=3)
    completions = [["bad_1", "bad_2", "bad_3", "bad_4"]]
    cots = ["<think>Golden Step</think> <answer>42</answer>"]

    # Step 1: batch_std = 0
    comps, inj = injector.check_and_inject(0.0, completions, cots)
    assert not inj
    # Step 2: batch_std = 0
    comps, inj = injector.check_and_inject(0.0, completions, cots)
    assert not inj
    # Step 3: threshold reached -> injects anchor!
    comps, inj = injector.check_and_inject(0.0, completions, cots)
    assert inj
    assert comps[0][-1] == cots[0]


def test_batch_processing_end_to_end():
    trainer = GRPOTrainer(GRPOTrainingConfig(group_size=2, teacher_anchor_threshold=2))
    prompts = ["What is 2+2?", "What is 3+3?"]
    answers = ["4", "6"]
    completions = [
        ["<think>2+2=4</think><answer>4</answer>", "The answer is 5"],
        ["<think>3+3=6</think><answer>6</answer>", "The answer is 7"],
    ]
    batch_res = trainer.process_prompt_batch(prompts, answers, completions)
    assert len(batch_res.group_results) == 2
    assert batch_res.batch_mean_reward > 0
    assert not batch_res.anchor_injected
