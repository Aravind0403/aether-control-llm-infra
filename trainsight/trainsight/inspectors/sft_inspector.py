"""
SFT Inspector & Pre-Flight Loss Mask Verifier.

Replaces inferred heuristics with deterministic checks:
- Verifies exact tokenization using the shared SFTPreprocessor
- Verifies answer-only loss masking (-100 for prompt and pad tokens)
- Counts disjoint anomalies: empty assistant targets, invalid rows, truncated answers, misaligned loss masks
- Computes real batch padding waste across proposed dynamic batches
- Fails closed when the tokenizer is unavailable (reports 'Unable to Inspect' with 0 estimated lengths)
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel, Field

from trainsight.preprocessing.sft_preprocessor import SFTPreprocessor, SFTExample


class SFTReport(BaseModel):
    total_samples: int
    valid_samples_count: int = 0
    invalid_rows_count: int = 0
    empty_completion_count: int = 0
    truncated_answers_count: int = 0
    loss_mask_misaligned_count: int = 0
    duplicate_count: int = 0
    length_exceeded_count: int = 0
    oom_risk_count: int = 0  # Retained as backwards-compatibility alias for length_exceeded_count
    
    avg_seq_len: float
    std_seq_len: float
    variance_ratio: float = 0.0
    min_seq_len: int
    max_seq_len: int
    p95_seq_len: float
    p99_seq_len: float
    avg_prompt_tokens: float = 0.0
    avg_target_tokens: float = 0.0

    # Real batch padding metrics (calculated over proposed batches)
    batch_size: int = 4
    actual_batch_padding_waste_pct: float = 0.0
    predicted_padding_waste_pct: float = 0.0  # Alias for actual_batch_padding_waste_pct
    total_useful_tokens: int = 0
    total_padded_tokens: int = 0

    drift_alert: bool = False
    ks_p_value: float = 1.0
    js_divergence: float = 0.0
    phase_quadrant: str = "Clean Execution"
    seq_lengths: List[int] = Field(default_factory=list)
    mbom: Optional[Any] = None
    efficiency: Optional[Any] = None
    collation: Optional[Any] = None
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class SFTInspector:
    """Inspector for Supervised Fine-Tuning (SFT) datasets."""

    def __init__(
        self,
        max_seq_len_threshold: int = 2048,
        tokenizer_name: Optional[str] = "Qwen/Qwen2.5-1.5B-Instruct",
        batch_size: int = 4,
        truncation: bool = True,
        strict: bool = False,
    ):
        self.max_seq_len_threshold = max_seq_len_threshold
        self.tokenizer_name = tokenizer_name or "Qwen/Qwen2.5-1.5B-Instruct"
        self.batch_size = batch_size
        self.truncation = truncation
        self.strict = strict
        self.preprocessor = None
        self.tokenizer_load_error = None
        try:
            self.preprocessor = SFTPreprocessor(
                tokenizer_name=self.tokenizer_name,
                max_seq_len=self.max_seq_len_threshold,
                truncation=self.truncation,
            )
        except Exception as e:
            self.preprocessor = None
            self.tokenizer_load_error = str(e)
            if self.strict:
                raise RuntimeError(
                    f"Unable to inspect: Tokenizer '{self.tokenizer_name}' could not be loaded ({e}). "
                    "Exact SFT inspection requires the tokenizer."
                ) from e

    def _safe_json_loads(self, line: str) -> Optional[Dict[str, Any]]:
        """Attempts standard json.loads and applies auto-fix recovery for common JSON corruptions."""
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            cleaned = line.replace("'", '"').rstrip(",")
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return None

    def inspect_file(
        self,
        file_path: Path,
        baseline_seq_lengths: Optional[List[int]] = None,
    ) -> SFTReport:
        """Inspects a dataset file using the exact shared preprocessor and loss mask verifier."""
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Fail closed if tokenizer is unavailable
        if self.preprocessor is None:
            return SFTReport(
                total_samples=0,
                valid_samples_count=0,
                invalid_rows_count=0,
                empty_completion_count=0,
                truncated_answers_count=0,
                loss_mask_misaligned_count=0,
                duplicate_count=0,
                length_exceeded_count=0,
                oom_risk_count=0,
                avg_seq_len=0.0,
                std_seq_len=0.0,
                min_seq_len=0,
                max_seq_len=0,
                p95_seq_len=0.0,
                p99_seq_len=0.0,
                phase_quadrant="Unable to Inspect",
                warnings=[
                    f"❌ Unable to inspect: Tokenizer '{self.tokenizer_name}' is unavailable ({self.tokenizer_load_error}). "
                    "Exact SFT input validation cannot proceed without the tokenizer."
                ],
                recommendations=[
                    f"Ensure tokenizer '{self.tokenizer_name}' is accessible locally or cached in HF_HOME before running SFT inspection."
                ],
            )

        raw_rows: List[Dict[str, Any]] = []
        invalid_json_lines = 0

        if file_path.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(str(file_path))
                raw_rows = table.to_pylist()
            except Exception:
                raw_rows = []
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = self._safe_json_loads(line)
                    if data is not None:
                        raw_rows.append(data)
                    else:
                        invalid_json_lines += 1

        if not raw_rows and invalid_json_lines == 0:
            return SFTReport(
                total_samples=0,
                valid_samples_count=0,
                invalid_rows_count=0,
                empty_completion_count=0,
                truncated_answers_count=0,
                loss_mask_misaligned_count=0,
                duplicate_count=0,
                length_exceeded_count=0,
                oom_risk_count=0,
                avg_seq_len=0.0,
                std_seq_len=0.0,
                min_seq_len=0,
                max_seq_len=0,
                p95_seq_len=0.0,
                p99_seq_len=0.0,
                phase_quadrant="Clean Execution",
                warnings=["Dataset file is empty."],
                recommendations=["Provide a populated training dataset file."],
            )

        processed_examples: List[SFTExample] = []
        invalid_rows = invalid_json_lines
        empty_completions = 0
        truncated_answers = 0
        loss_mask_misaligned = 0
        seen_prompts = set()
        duplicates = 0
        length_exceeded = 0

        prompt_lengths = []
        target_lengths = []
        total_lengths = []

        for row in raw_rows:
            prompt_str, completion_str, is_valid_fmt, err = self.preprocessor.extract_prompt_and_completion(row)

            # 1. Structural validity check (disjoint: bad rows stop here)
            if not is_valid_fmt or not prompt_str:
                invalid_rows += 1
                continue

            # 2. Duplicate prompt check on valid prompts
            p_key = prompt_str.strip().lower()
            if p_key in seen_prompts:
                duplicates += 1
            else:
                seen_prompts.add(p_key)

            # 3. Exact preprocessing and loss mask inspection
            ex = self.preprocessor.process_sample(row)
            processed_examples.append(ex)

            # 4. Disjoint anomaly tracking:
            if ex.has_empty_completion:
                empty_completions += 1
            elif ex.truncated_answer:
                truncated_answers += 1

            if ex.loss_mask_misaligned:
                loss_mask_misaligned += 1

            if ex.raw_len > self.max_seq_len_threshold:
                length_exceeded += 1

            prompt_lengths.append(ex.prompt_tokens_len)
            target_lengths.append(ex.completion_tokens_len)
            total_lengths.append(ex.total_len)

        total_samples = len(raw_rows) + invalid_json_lines
        valid_samples = len([ex for ex in processed_examples if not ex.has_empty_completion])

        total_lengths_arr = np.array(total_lengths) if total_lengths else np.array([0])
        avg_len = float(np.mean(total_lengths_arr)) if total_lengths else 0.0
        std_len = float(np.std(total_lengths_arr)) if total_lengths else 0.0
        var_ratio = float(std_len / avg_len) if avg_len > 0 else 0.0
        min_len = int(np.min(total_lengths_arr)) if total_lengths else 0
        max_len = int(np.max(total_lengths_arr)) if total_lengths else 0
        p95_len = float(np.percentile(total_lengths_arr, 95)) if total_lengths else 0.0
        p99_len = float(np.percentile(total_lengths_arr, 99)) if total_lengths else 0.0

        avg_p_len = float(np.mean(prompt_lengths)) if prompt_lengths else 0.0
        avg_t_len = float(np.mean(target_lengths)) if target_lengths else 0.0

        # Calculate actual padding waste across proposed batches
        bs = self.batch_size
        actual_padded_total = 0
        total_useful_tokens = int(np.sum(total_lengths_arr)) if total_lengths else 0
        for i in range(0, len(total_lengths), bs):
            chunk = total_lengths[i:i + bs]
            if chunk:
                actual_padded_total += max(chunk) * len(chunk)

        if actual_padded_total > 0:
            actual_pad_waste_pct = ((actual_padded_total - total_useful_tokens) / actual_padded_total) * 100.0
        else:
            actual_pad_waste_pct = 0.0

        # Concrete checks and warnings
        warnings = []
        recommendations = []

        if invalid_rows > 0:
            warnings.append(f"❌ {invalid_rows} invalid row(s) detected (unparseable JSON lines or missing prompts).")
            recommendations.append("Fix JSON syntax and remove records with missing prompt text.")

        if empty_completions > 0:
            warnings.append(f"❌ {empty_completions} sample(s) have empty assistant targets (loss is undefined on empty targets).")
            recommendations.append("Filter out rows with empty or missing assistant completions.")

        if truncated_answers > 0:
            warnings.append(f"⚠️ {truncated_answers} sample(s) have truncated assistant answers due to max sequence length limit ({self.max_seq_len_threshold} tokens).")
            recommendations.append(f"Consider increasing max_seq_len or pre-filtering prompts longer than {self.max_seq_len_threshold} tokens.")

        if loss_mask_misaligned > 0:
            warnings.append(f"❌ {loss_mask_misaligned} sample(s) have misaligned loss masks (unmasked prompt tokens or invalid custom labels).")
            recommendations.append("Ensure answer-only loss masking (labels = -100 for prompt and pad tokens) is strictly enforced.")

        if duplicates > 0:
            pct = (duplicates / max(1, total_samples)) * 100
            warnings.append(f"⚠️ {duplicates} duplicate prompt(s) detected ({pct:.1f}%).")
            recommendations.append("Deduplicate dataset prompts to avoid memorization and skewed gradients.")

        if length_exceeded > 0:
            pct = (length_exceeded / max(1, total_samples)) * 100
            warnings.append(f"⚠️ {length_exceeded} sample(s) ({pct:.1f}%) exceed the configured length limit ({self.max_seq_len_threshold} tokens).")
            recommendations.append(f"Apply truncation policy or filter sequences exceeding {self.max_seq_len_threshold} tokens.")

        if actual_pad_waste_pct > 20.0:
            recommendations.append(
                f"Collation Padding Advisory: {actual_pad_waste_pct:.1f}% zero-value padded tokens under sequential batching (B={bs}). "
                "Apply TrainSight SequenceBinPacker to eliminate padding waste."
            )

        # Drift test
        ks_p = 1.0
        drift_alert = False
        if baseline_seq_lengths and len(baseline_seq_lengths) > 0 and total_lengths:
            try:
                from scipy.stats import ks_2samp
                stat, p_val = ks_2samp(total_lengths, baseline_seq_lengths)
                ks_p = float(p_val)
                if ks_p < 0.05:
                    drift_alert = True
                    warnings.append(f"⚠️ Dataset Drift Detected! (KS Test p-value = {ks_p:.4f} < 0.05). Length distribution shifted.")
                    recommendations.append("Re-calibrate batch size and collation bounds against new length distribution.")
            except ImportError:
                pass

        # Phase Quadrant classification
        if invalid_rows > 0 or empty_completions > 0 or loss_mask_misaligned > 0 or truncated_answers > 0:
            phase_quadrant = "Validation Alerts Detected"
        elif duplicates > 0 or length_exceeded > 0:
            phase_quadrant = "Optimization Advisories"
        else:
            phase_quadrant = "Clean Execution"

        # Collation recommendations
        collation_dict = {
            "batch_size": bs,
            "sequential_padding_waste_pct": round(actual_pad_waste_pct, 2),
            "total_useful_tokens": total_useful_tokens,
            "total_padded_tokens": actual_padded_total,
            "suggested_bin_packing": actual_pad_waste_pct > 20.0,
        }

        return SFTReport(
            total_samples=total_samples,
            valid_samples_count=valid_samples,
            invalid_rows_count=invalid_rows,
            empty_completion_count=empty_completions,
            truncated_answers_count=truncated_answers,
            loss_mask_misaligned_count=loss_mask_misaligned,
            duplicate_count=duplicates,
            length_exceeded_count=length_exceeded,
            oom_risk_count=length_exceeded,
            avg_seq_len=round(avg_len, 2),
            std_seq_len=round(std_len, 2),
            variance_ratio=round(var_ratio, 4),
            min_seq_len=min_len,
            max_seq_len=max_len,
            p95_seq_len=round(p95_len, 2),
            p99_seq_len=round(p99_len, 2),
            avg_prompt_tokens=round(avg_p_len, 2),
            avg_target_tokens=round(avg_t_len, 2),
            batch_size=bs,
            actual_batch_padding_waste_pct=round(actual_pad_waste_pct, 2),
            predicted_padding_waste_pct=round(actual_pad_waste_pct, 2),
            total_useful_tokens=total_useful_tokens,
            total_padded_tokens=actual_padded_total,
            drift_alert=drift_alert,
            ks_p_value=round(ks_p, 4),
            phase_quadrant=phase_quadrant,
            seq_lengths=total_lengths,
            collation=collation_dict,
            warnings=warnings,
            recommendations=recommendations,
        )
