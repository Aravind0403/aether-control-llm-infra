import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field
from rlhf_pipeline.rewards.math_reward import GRPOReasoningReward


class GRPOTrainingConfig(BaseModel):
    """Configuration parameters for GRPO fine-tuning pipeline."""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    group_size: int = Field(default=4, ge=2, description="Number of completions sampled per prompt (Group Size G)")
    kl_coeff: float = Field(default=0.04, ge=0.0, description="KL divergence penalty coefficient beta")
    clip_eps: float = Field(default=0.2, gt=0.0, description="PPO/GRPO ratio clipping parameter epsilon")
    learning_rate: float = Field(default=1e-6, description="Policy model learning rate")
    max_completion_length: int = Field(default=512, description="Max generated tokens per completion")
    teacher_anchor_threshold: int = Field(default=3, description="Consecutive zero-variance steps before teacher anchor activates")


class GRPOGroupResult(BaseModel):
    """Calculated advantages and rewards for a single sampled prompt group."""
    prompt: str
    ground_truth: str
    completions: List[str]
    rewards: List[float]
    advantages: List[float]
    group_mean_reward: float
    group_std_reward: float
    advantage_mode: str = "intra_group"  # 'intra_group', 'batch_fallback', or 'teacher_anchor'


class GRPOBatchResult(BaseModel):
    """Aggregated results across a mini-batch of prompt groups."""
    group_results: List[GRPOGroupResult]
    batch_mean_reward: float
    batch_std_reward: float
    anchor_injected: bool = False


class TeacherAnchorInjector:
    """Manages teacher-anchor injection during persistent cold-start or incompetence stalls (RFC-001)."""

    def __init__(self, threshold_steps: int = 3, eps: float = 1e-8):
        self.threshold_steps = threshold_steps
        self.eps = eps
        self.consecutive_zero_variance_steps = 0

    def check_and_inject(
        self,
        batch_std: float,
        completions_per_prompt: List[List[str]],
        golden_cots: Optional[List[str]] = None,
    ) -> Tuple[List[List[str]], bool]:
        """Injects a golden chain-of-thought rollout if batch variance flatlines."""
        if batch_std < self.eps:
            self.consecutive_zero_variance_steps += 1
        else:
            self.consecutive_zero_variance_steps = 0
            return completions_per_prompt, False

        if self.consecutive_zero_variance_steps >= self.threshold_steps and golden_cots:
            # Substitute the final completion of the first prompt with the golden anchor
            modified = [list(group) for group in completions_per_prompt]
            if modified and len(modified[0]) > 0:
                modified[0][-1] = golden_cots[0]
                return modified, True

        return completions_per_prompt, False


class GRPOTrainer:
    """Group Relative Policy Optimization (GRPO) Trainer Engine with RFC-001 Hardening."""

    def __init__(self, config: GRPOTrainingConfig):
        self.config = config
        self.reward_evaluator = GRPOReasoningReward()
        self.teacher_injector = TeacherAnchorInjector(threshold_steps=config.teacher_anchor_threshold)

    def compute_hierarchical_advantages(
        self,
        group_rewards: List[float],
        batch_mean: Optional[float] = None,
        batch_std: Optional[float] = None,
        eps: float = 1e-8,
    ) -> Tuple[List[float], float, float, str]:
        """Computes normalized advantages with automatic fallback to mini-batch baseline (RFC-001).
        
        Returns:
            advantages: List of advantage values.
            group_mean: Mean reward of the group.
            group_std: Standard deviation of the group.
            mode: 'intra_group' if group variance > eps, else 'batch_fallback' or 'zero_fallback'.
        """
        arr = np.array(group_rewards, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr))

        if std > eps:
            # Standard DeepSeek GRPO intra-group advantage
            advantages = [float((r - mean) / (std + eps)) for r in group_rewards]
            mode = "intra_group"
        elif batch_mean is not None and batch_std is not None and batch_std > eps:
            # Batch-Relative Momentum Fallback
            # All-failed -> negative penalty; All-succeeded -> positive reinforcement!
            advantages = [float((r - batch_mean) / (batch_std + eps)) for r in group_rewards]
            mode = "batch_fallback"
        else:
            # Fallback when entire batch has zero variance
            advantages = [0.0 for _ in group_rewards]
            mode = "zero_fallback"

        return advantages, mean, std, mode

    def compute_group_advantages(self, rewards: List[float], eps: float = 1e-8) -> Tuple[List[float], float, float]:
        """Legacy interface for backward compatibility."""
        advantages, mean, std, _ = self.compute_hierarchical_advantages(rewards, eps=eps)
        return advantages, mean, std

    def process_prompt_group(
        self,
        prompt: str,
        ground_truth: str,
        completions: List[str],
        batch_mean: Optional[float] = None,
        batch_std: Optional[float] = None,
    ) -> GRPOGroupResult:
        """Evaluates rewards and computes advantages across a sampled completion group."""
        if len(completions) != self.config.group_size:
            raise ValueError(f"Expected {self.config.group_size} completions per group, got {len(completions)}")

        rewards = []
        for comp in completions:
            eval_res = self.reward_evaluator.compute_total_reward(comp, ground_truth)
            rewards.append(eval_res["total_reward"])

        advantages, mean_r, std_r, mode = self.compute_hierarchical_advantages(
            rewards, batch_mean=batch_mean, batch_std=batch_std
        )

        return GRPOGroupResult(
            prompt=prompt,
            ground_truth=ground_truth,
            completions=completions,
            rewards=rewards,
            advantages=advantages,
            group_mean_reward=mean_r,
            group_std_reward=std_r,
            advantage_mode=mode,
        )

    def process_prompt_batch(
        self,
        prompts: List[str],
        ground_truths: List[str],
        completions_per_prompt: List[List[str]],
        golden_cots: Optional[List[str]] = None,
    ) -> GRPOBatchResult:
        """Processes a mini-batch of prompt groups with batch-level advantage normalization and teacher injection."""
        if len(prompts) != len(ground_truths) or len(prompts) != len(completions_per_prompt):
            raise ValueError("Mismatched dimensions in batch processing.")

        # Step 1: Compute raw rewards across all B x G completions
        all_rewards = []
        group_raw_rewards = []
        for p, gt, comps in zip(prompts, ground_truths, completions_per_prompt):
            grp_rewards = [self.reward_evaluator.compute_total_reward(c, gt)["total_reward"] for c in comps]
            group_raw_rewards.append(grp_rewards)
            all_rewards.extend(grp_rewards)

        # Batch-level reduction
        b_arr = np.array(all_rewards, dtype=np.float64)
        batch_mean = float(np.mean(b_arr))
        batch_std = float(np.std(b_arr))

        # Step 2: Check for cold-start teacher-anchor injection if variance flatlines
        active_completions, anchor_injected = self.teacher_injector.check_and_inject(
            batch_std, completions_per_prompt, golden_cots
        )

        if anchor_injected:
            # Re-evaluate rewards for the modified group
            group_raw_rewards[0] = [
                self.reward_evaluator.compute_total_reward(c, ground_truths[0])["total_reward"]
                for c in active_completions[0]
            ]
            all_rewards = [r for grp in group_raw_rewards for r in grp]
            b_arr = np.array(all_rewards, dtype=np.float64)
            batch_mean = float(np.mean(b_arr))
            batch_std = float(np.std(b_arr))

        # Step 3: Compute hierarchical advantages for each group
        group_results = []
        for p, gt, comps, rewards in zip(prompts, ground_truths, active_completions, group_raw_rewards):
            advs, m_r, s_r, mode = self.compute_hierarchical_advantages(
                rewards, batch_mean=batch_mean, batch_std=batch_std
            )
            if anchor_injected and comps == active_completions[0]:
                mode = "teacher_anchor"

            group_results.append(
                GRPOGroupResult(
                    prompt=p,
                    ground_truth=gt,
                    completions=comps,
                    rewards=rewards,
                    advantages=advs,
                    group_mean_reward=m_r,
                    group_std_reward=s_r,
                    advantage_mode=mode,
                )
            )

        return GRPOBatchResult(
            group_results=group_results,
            batch_mean_reward=batch_mean,
            batch_std_reward=batch_std,
            anchor_injected=anchor_injected,
        )
