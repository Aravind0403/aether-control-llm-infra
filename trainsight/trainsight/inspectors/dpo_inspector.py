import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel, Field


class DPOReport(BaseModel):
    total_samples: int
    valid_samples_count: int = 0
    invalid_rows_count: int = 0
    empty_pair_count: int = 0
    prompt_mismatch_count: int = 0
    duplicate_count: int = 0

    # Exact token metrics
    avg_prompt_len: float = 0.0
    avg_chosen_len: float = 0.0
    avg_rejected_len: float = 0.0
    length_bias_ratio: float = 1.0

    # Anomaly / Defect metrics
    identical_pairs_count: int = 0
    near_identical_count: int = 0
    preference_reversal_count: int = 0
    score_tie_count: int = 0

    # Budget tracking (1024 prompt / 1024 completion / 2048 total)
    over_budget_prompts_count: int = 0
    over_budget_completions_count: int = 0
    over_budget_total_count: int = 0

    phase_quadrant: str = "Clean Execution"
    warnings: List[str] = Field(default_factory=list)
    advisories: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class DPOInspector:
    """
    Inspector for Direct Preference Optimization (DPO) datasets.

    Replaces character heuristics with deterministic, model-aligned checks:
    - Parses only assistant responses (chosen[1] and rejected[1] in message format).
    - Verifies user message agreement with top-level prompt.
    - Reads score_chosen and score_rejected on a 1-10 scale (or legacy score fields).
    - Measures exact token lengths using model tokenizer & chat template.
    - Evaluates training budgets (1024 prompt / 1024 completion / 2048 total).
    - Distinguishes actionable defects (identical pairs, reversals) from advisories (score ties, truncations).
    - Fails closed when the tokenizer cannot be loaded.
    """

    def __init__(
        self,
        tokenizer_name: Optional[str] = "Qwen/Qwen2.5-1.5B-Instruct",
        max_prompt_len: int = 1024,
        max_completion_len: int = 1024,
        max_total_len: int = 2048,
        jaccard_similarity_threshold: float = 0.90,
        strict: bool = False,
    ):
        self.tokenizer_name = tokenizer_name
        self.max_prompt_len = max_prompt_len
        self.max_completion_len = max_completion_len
        self.max_total_len = max_total_len
        self.jaccard_similarity_threshold = jaccard_similarity_threshold
        self.strict = strict

        self.tokenizer = None
        self.tokenizer_load_error = None

        if self.tokenizer_name:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            except Exception as e:
                self.tokenizer = None
                self.tokenizer_load_error = str(e)
                if self.strict:
                    raise RuntimeError(
                        f"Unable to inspect: Tokenizer '{self.tokenizer_name}' could not be loaded ({e}). "
                        "Exact DPO inspection requires the tokenizer."
                    ) from e

    def _estimate_tokens_fallback(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text.strip()) // 4)

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Word-level Jaccard similarity strictly on assistant responses."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0

    def _extract_assistant_and_validate_prompt(
        self, prompt_text: str, candidate_field: Any
    ) -> tuple[str, bool, bool]:
        """
        Extracts the assistant response text from candidate field (message list or string)
        and verifies user message agreement with prompt.

        Returns:
            (assistant_text, is_missing, is_prompt_mismatched)
        """
        if not candidate_field:
            return "", True, False

        if isinstance(candidate_field, str):
            text = candidate_field.strip()
            return text, len(text) == 0, False

        if isinstance(candidate_field, list):
            # Message format e.g. [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            user_text = ""
            asst_text = ""
            for msg in candidate_field:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user" and not user_text:
                    user_text = content
                elif role == "assistant" and not asst_text:
                    asst_text = content

            # Fallback if roles not explicitly matched
            if not asst_text and len(candidate_field) >= 2 and isinstance(candidate_field[1], dict):
                asst_text = candidate_field[1].get("content", "")
            if not user_text and len(candidate_field) >= 1 and isinstance(candidate_field[0], dict):
                user_text = candidate_field[0].get("content", "")

            prompt_mismatched = False
            if prompt_text and user_text:
                if user_text.strip() != prompt_text.strip():
                    prompt_mismatched = True

            return asst_text.strip(), len(asst_text.strip()) == 0, prompt_mismatched

        return str(candidate_field).strip(), False, False

    def inspect_file(self, file_path: Path) -> DPOReport:
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        samples: List[Dict[str, Any]] = []
        invalid_rows_count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    invalid_rows_count += 1
                    continue

        if not samples:
            return DPOReport(
                total_samples=0,
                invalid_rows_count=invalid_rows_count,
                phase_quadrant="Invalid Dataset",
                warnings=["Dataset is empty or contains no valid JSONL records."],
                recommendations=["Check dataset path and jsonl file content."],
            )

        # Fail closed if tokenizer was requested but failed to load
        if self.tokenizer_name and self.tokenizer is None:
            return DPOReport(
                total_samples=len(samples),
                invalid_rows_count=invalid_rows_count,
                phase_quadrant="Unable to Inspect",
                warnings=[
                    f"Unable to inspect: Tokenizer '{self.tokenizer_name}' could not be loaded ({self.tokenizer_load_error}). "
                    "Exact DPO inspection requires the tokenizer."
                ],
                recommendations=[
                    "Ensure the model tokenizer files are cached locally or pass a valid local tokenizer path."
                ],
            )

        prompt_lens = []
        chosen_lens = []
        rejected_lens = []

        identical_count = 0
        near_identical_count = 0
        empty_pair_count = 0
        prompt_mismatch_count = 0
        seen_prompts = set()
        duplicate_count = 0
        preference_reversal_count = 0
        score_tie_count = 0

        over_budget_prompts_count = 0
        over_budget_completions_count = 0
        over_budget_total_count = 0

        valid_samples_count = 0

        for sample in samples:
            prompt = sample.get("prompt") or ""
            prompt_str = prompt.strip() if isinstance(prompt, str) else str(prompt).strip()

            raw_chosen = sample.get("chosen")
            raw_rejected = sample.get("rejected")

            asst_chosen, chosen_missing, chosen_mismatched = self._extract_assistant_and_validate_prompt(
                prompt_str, raw_chosen
            )
            asst_rejected, rejected_missing, rejected_mismatched = self._extract_assistant_and_validate_prompt(
                prompt_str, raw_rejected
            )

            if not prompt_str or chosen_missing or rejected_missing:
                empty_pair_count += 1
                continue

            if chosen_mismatched or rejected_mismatched:
                prompt_mismatch_count += 1

            # Quality Score checks (reading score_chosen / score_rejected or fallbacks)
            c_score = sample.get("score_chosen")
            if c_score is None:
                c_score = sample.get("chosen_score") if sample.get("chosen_score") is not None else sample.get("chosen_rating")

            r_score = sample.get("score_rejected")
            if r_score is None:
                r_score = sample.get("rejected_score") if sample.get("rejected_score") is not None else sample.get("rejected_rating")

            if c_score is not None and r_score is not None:
                try:
                    c_val = float(c_score)
                    r_val = float(r_score)
                    if c_val < r_val:
                        preference_reversal_count += 1
                    elif c_val == r_val:
                        score_tie_count += 1
                except (ValueError, TypeError):
                    pass

            # Identical / Near-identical assistant response check
            if asst_chosen.strip().lower() == asst_rejected.strip().lower():
                identical_count += 1
            else:
                sim = self._jaccard_similarity(asst_chosen, asst_rejected)
                if sim >= self.jaccard_similarity_threshold:
                    near_identical_count += 1

            # Prompt deduplication check
            if prompt_str:
                prompt_key = prompt_str.lower()
                if prompt_key in seen_prompts:
                    duplicate_count += 1
                else:
                    seen_prompts.add(prompt_key)

            # Tokenization (exact BPE with template or fallback)
            if self.tokenizer is not None:
                try:
                    chat_prompt_str = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt_str}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    prompt_t = len(self.tokenizer.encode(chat_prompt_str))
                except Exception:
                    prompt_t = len(self.tokenizer.encode(prompt_str))

                chosen_t = len(self.tokenizer.encode(asst_chosen))
                rejected_t = len(self.tokenizer.encode(asst_rejected))
            else:
                prompt_t = self._estimate_tokens_fallback(prompt_str)
                chosen_t = self._estimate_tokens_fallback(asst_chosen)
                rejected_t = self._estimate_tokens_fallback(asst_rejected)

            prompt_lens.append(prompt_t)
            chosen_lens.append(chosen_t)
            rejected_lens.append(rejected_t)
            valid_samples_count += 1

            # Budget evaluation
            if prompt_t > self.max_prompt_len:
                over_budget_prompts_count += 1
            if chosen_t > self.max_completion_len or rejected_t > self.max_completion_len:
                over_budget_completions_count += 1
            if (prompt_t + chosen_t) > self.max_total_len or (prompt_t + rejected_t) > self.max_total_len:
                over_budget_total_count += 1

        avg_p = float(np.mean(prompt_lens)) if prompt_lens else 0.0
        avg_c = float(np.mean(chosen_lens)) if chosen_lens else 0.0
        avg_r = float(np.mean(rejected_lens)) if rejected_lens else 0.0
        length_bias_ratio = (avg_c / avg_r) if avg_r > 0 else 1.0

        warnings = []
        advisories = []
        recommendations = []

        # Actionable Defects (lead to non-clean quadrant)
        if identical_count > 0:
            warnings.append(
                f"⚠️ {identical_count} pairs have IDENTICAL chosen and rejected responses. "
                "DPO loss gradient will be zero; filter these pairs before training."
            )
            recommendations.append("Filter identical chosen/rejected pairs from the dataset.")

        if preference_reversal_count > 0:
            warnings.append(
                f"⚠️ {preference_reversal_count} preference pairs have REVERSED quality scores (score_chosen < score_rejected). "
                "DPO will optimize towards inferior responses!"
            )
            recommendations.append("Swap chosen/rejected responses or filter inverted quality rating rows.")

        if prompt_mismatch_count > 0:
            warnings.append(
                f"⚠️ {prompt_mismatch_count} pairs have user messages in chosen/rejected that do not match the prompt."
            )
            recommendations.append("Verify multi-turn message alignment with prompt.")

        if empty_pair_count > 0:
            warnings.append(f"⚠️ {empty_pair_count} samples have missing chosen, rejected, or prompt text.")
            recommendations.append("Drop corrupted rows with missing preference text.")

        if invalid_rows_count > 0:
            warnings.append(f"⚠️ {invalid_rows_count} malformed JSON lines were skipped.")
            recommendations.append("Validate JSON syntax.")

        if length_bias_ratio > 1.8 or length_bias_ratio < 0.55:
            warnings.append(
                f"⚠️ Severe length bias detected (Chosen/Rejected avg length ratio = {length_bias_ratio:.2f}). "
                "Model will exploit length over quality!"
            )
            recommendations.append("Apply length-normalized DPO loss (SimPO / DisPo) or rebalance dataset length distributions.")

        # Informational Advisories (confidence flags & training budget observations)
        if score_tie_count > 0:
            advisories.append(
                f"ℹ️ {score_tie_count} pairs have TIED quality scores (score_chosen == score_rejected). "
                "These are distinct responses with equal teacher confidence, not zero-gradient defects."
            )

        if over_budget_prompts_count > 0:
            advisories.append(
                f"ℹ️ {over_budget_prompts_count} prompts exceed the {self.max_prompt_len}-token budget and will be truncated during training."
            )

        if near_identical_count > 0:
            advisories.append(
                f"ℹ️ {near_identical_count} pairs are near-identical (Jaccard similarity > {self.jaccard_similarity_threshold * 100:.0f}%). "
                "DPO margin learning will be constrained."
            )

        if duplicate_count > 0:
            advisories.append(f"ℹ️ {duplicate_count} duplicate prompts detected.")

        # Phase Quadrant Assignment
        if (
            identical_count > 0
            or preference_reversal_count > 0
            or empty_pair_count > 0
            or prompt_mismatch_count > 0
            or invalid_rows_count > 0
        ):
            phase_quadrant = "Defects Detected"
        elif length_bias_ratio > 1.8 or length_bias_ratio < 0.55:
            phase_quadrant = "Quality Warnings"
        else:
            phase_quadrant = "Clean Execution"

        return DPOReport(
            total_samples=len(samples),
            valid_samples_count=valid_samples_count,
            invalid_rows_count=invalid_rows_count,
            empty_pair_count=empty_pair_count,
            prompt_mismatch_count=prompt_mismatch_count,
            duplicate_count=duplicate_count,
            avg_prompt_len=avg_p,
            avg_chosen_len=avg_c,
            avg_rejected_len=avg_r,
            length_bias_ratio=length_bias_ratio,
            identical_pairs_count=identical_count,
            near_identical_count=near_identical_count,
            preference_reversal_count=preference_reversal_count,
            score_tie_count=score_tie_count,
            over_budget_prompts_count=over_budget_prompts_count,
            over_budget_completions_count=over_budget_completions_count,
            over_budget_total_count=over_budget_total_count,
            phase_quadrant=phase_quadrant,
            warnings=warnings,
            advisories=advisories,
            recommendations=recommendations,
        )
