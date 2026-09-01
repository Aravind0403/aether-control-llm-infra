import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from pydantic import BaseModel, Field


class DPOReport(BaseModel):
    total_samples: int
    avg_chosen_len: float
    avg_rejected_len: float
    identical_pairs_count: int
    near_identical_count: int
    empty_pair_count: int
    duplicate_count: int
    preference_reversal_count: int = 0
    length_bias_ratio: float
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class DPOInspector:
    """Inspector for Direct Preference Optimization (DPO) datasets."""

    def __init__(self, jaccard_similarity_threshold: float = 0.90):
        self.jaccard_similarity_threshold = jaccard_similarity_threshold

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text.strip()) // 4)

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Word-level Jaccard similarity to catch near-identical chosen/rejected pairs fast on CPU."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0

    def inspect_file(self, file_path: Path) -> DPOReport:
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        samples: List[Dict[str, Any]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not samples:
            return DPOReport(
                total_samples=0,
                avg_chosen_len=0.0,
                avg_rejected_len=0.0,
                identical_pairs_count=0,
                near_identical_count=0,
                empty_pair_count=0,
                duplicate_count=0,
                length_bias_ratio=1.0,
                warnings=["Dataset is empty or invalid JSONL."],
                recommendations=["Check dataset path and jsonl structure."],
            )

        chosen_lens = []
        rejected_lens = []
        identical_count = 0
        near_identical_count = 0
        empty_pair_count = 0
        seen_prompts = set()
        duplicate_count = 0
        preference_reversal_count = 0

        for sample in samples:
            prompt = sample.get("prompt") or ""
            chosen = sample.get("chosen") or ""
            rejected = sample.get("rejected") or ""

            # Check messages format if present (e.g., UltraFeedback / Anthropic HH)
            if isinstance(chosen, list):
                chosen = " ".join([m.get("content", "") for m in chosen])
            if isinstance(rejected, list):
                rejected = " ".join([m.get("content", "") for m in rejected])

            if not chosen or not rejected:
                empty_pair_count += 1
                continue

            # Preference Reversal Check (checking explicitly assigned chosen/rejected quality scores if present)
            c_score = sample.get("chosen_score") if sample.get("chosen_score") is not None else sample.get("chosen_rating")
            r_score = sample.get("rejected_score") if sample.get("rejected_score") is not None else sample.get("rejected_rating")
            if c_score is not None and r_score is not None:
                try:
                    if float(c_score) < float(r_score):
                        preference_reversal_count += 1
                except (ValueError, TypeError):
                    pass

            chosen_t = self._estimate_tokens(chosen)
            rejected_t = self._estimate_tokens(rejected)
            chosen_lens.append(chosen_t)
            rejected_lens.append(rejected_t)

            # Check identical / near-identical pairs
            if chosen.strip().lower() == rejected.strip().lower():
                identical_count += 1
            else:
                sim = self._jaccard_similarity(chosen, rejected)
                if sim >= self.jaccard_similarity_threshold:
                    near_identical_count += 1

            # Check duplicate prompt
            prompt_key = prompt.strip().lower()
            if prompt_key:
                if prompt_key in seen_prompts:
                    duplicate_count += 1
                else:
                    seen_prompts.add(prompt_key)

        avg_c = float(np.mean(chosen_lens)) if chosen_lens else 0.0
        avg_r = float(np.mean(rejected_lens)) if rejected_lens else 0.0
        length_bias_ratio = (avg_c / avg_r) if avg_r > 0 else 1.0

        warnings = []
        recommendations = []

        if preference_reversal_count > 0:
            warnings.append(f"⚠️ {preference_reversal_count} preference pairs have REVERSED ratings (chosen_score < rejected_score). DPO will optimize towards inferior responses!")
            recommendations.append("Swap chosen/rejected responses or filter inverted quality rating rows.")

        if identical_count > 0:
            warnings.append(f"⚠️ {identical_count} pairs have IDENTICAL chosen and rejected responses. DPO loss gradient will be 0!")
            recommendations.append("Remove identical chosen/rejected pairs from DPO dataset.")

        if near_identical_count > 0:
            warnings.append(f"⚠️ {near_identical_count} pairs are near-identical (Jaccard similarity > {self.jaccard_similarity_threshold * 100:.0f}%). DPO margin learning will stall.")
            recommendations.append("Filter low-preference-margin pairs before launch.")

        if length_bias_ratio > 1.8 or length_bias_ratio < 0.55:
            warnings.append(f"⚠️ Severe length bias detected (Chosen/Rejected avg length ratio = {length_bias_ratio:.2f}). Model will exploit length over quality!")
            recommendations.append("Apply length-normalized DPO loss (DisPo / SimPO) or rebalance dataset length distributions.")

        if empty_pair_count > 0:
            warnings.append(f"⚠️ {empty_pair_count} samples have missing chosen or rejected responses.")
            recommendations.append("Drop corrupted rows with missing preference text.")

        if duplicate_count > 0:
            warnings.append(f"⚠️ {duplicate_count} duplicate prompts detected.")
            recommendations.append("Deduplicate preference prompts to avoid over-fitting.")

        return DPOReport(
            total_samples=len(samples),
            avg_chosen_len=avg_c,
            avg_rejected_len=avg_r,
            identical_pairs_count=identical_count,
            near_identical_count=near_identical_count,
            empty_pair_count=empty_pair_count,
            duplicate_count=duplicate_count,
            preference_reversal_count=preference_reversal_count,
            length_bias_ratio=length_bias_ratio,
            warnings=warnings,
            recommendations=recommendations,
        )
