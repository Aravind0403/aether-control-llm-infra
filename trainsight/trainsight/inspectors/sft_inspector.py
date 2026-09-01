import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel, Field


class SFTReport(BaseModel):
    total_samples: int
    avg_seq_len: float
    std_seq_len: float
    variance_ratio: float = 0.0
    min_seq_len: int
    max_seq_len: int
    p95_seq_len: float
    p99_seq_len: float
    oom_risk_count: int
    empty_completion_count: int
    duplicate_count: int
    loss_mask_misaligned_count: int = 0
    drift_alert: bool = False
    ks_p_value: float = 1.0
    js_divergence: float = 0.0
    predicted_padding_waste_pct: float = 0.0
    phase_quadrant: str = "Clean Execution"
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class SFTInspector:
    """Inspector for Supervised Fine-Tuning (SFT) datasets."""

    def __init__(self, max_seq_len_threshold: int = 2048, tokenizer_name: Optional[str] = None):
        self.max_seq_len_threshold = max_seq_len_threshold
        self.tokenizer_name = tokenizer_name
        self.tokenizer = None
        if tokenizer_name:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception:
                self.tokenizer = None

    def _estimate_token_count(self, text: str) -> int:
        """Fast heuristic token counter (~4 chars per token) or model-aligned BPE tokenizer."""
        if not text:
            return 0
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text, truncation=False))
            except Exception:
                pass
        return max(1, len(text.strip()) // 4)

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

    def inspect_file(self, file_path: Path, baseline_seq_lengths: Optional[List[int]] = None) -> SFTReport:
        """Reads JSONL or Parquet dataset and computes token distribution, OOM risk, and anomalies."""
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        samples: List[Dict[str, Any]] = []

        if file_path.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(str(file_path))
                samples = table.to_pylist()
            except Exception as e:
                samples = []
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = self._safe_json_loads(line)
                    if data:
                        samples.append(data)

        if not samples:
            return SFTReport(
                total_samples=0,
                avg_seq_len=0.0,
                std_seq_len=0.0,
                min_seq_len=0,
                max_seq_len=0,
                p95_seq_len=0.0,
                p99_seq_len=0.0,
                oom_risk_count=0,
                empty_completion_count=0,
                duplicate_count=0,
                warnings=["Dataset is empty or contains invalid JSON lines."],
                recommendations=["Check dataset formatting. Ensure valid JSONL."],
            )

        seq_lengths = []
        empty_completions = 0
        seen_prompts = set()
        duplicates = 0
        oom_risk = 0
        loss_mask_misaligned = 0

        for sample in samples:
            # Handle standard keys: 'prompt'/'completion', 'instruction'/'output', 'question'/'answer', or 'messages'
            prompt_text = sample.get("prompt") or sample.get("instruction") or sample.get("question") or ""
            completion_text = sample.get("completion") or sample.get("output") or sample.get("response") or sample.get("answer") or ""

            if isinstance(sample.get("messages"), list):
                full_text = " ".join([m.get("content", "") for m in sample["messages"]])
                # Check loss masking alignment: must have at least one assistant completion turn
                assistant_turns = [m for m in sample["messages"] if m.get("role") == "assistant" and m.get("content")]
                if not assistant_turns:
                    loss_mask_misaligned += 1
            else:
                full_text = f"{prompt_text} {completion_text}"
                if not completion_text:
                    loss_mask_misaligned += 1

            # Check explicit labels if present
            labels = sample.get("labels")
            if isinstance(labels, list) and labels:
                # If labels contain no -100 masking tokens, prompt tokens are unmasked (loss computed on user prompt)
                if -100 not in labels:
                    loss_mask_misaligned += 1

            if not completion_text and not sample.get("messages"):
                empty_completions += 1

            token_count = self._estimate_token_count(full_text)
            seq_lengths.append(token_count)

            if token_count > self.max_seq_len_threshold:
                oom_risk += 1

            prompt_key = prompt_text.strip().lower()
            if prompt_key:
                if prompt_key in seen_prompts:
                    duplicates += 1
                else:
                    seen_prompts.add(prompt_key)

        seq_lengths_arr = np.array(seq_lengths)
        avg_len = float(np.mean(seq_lengths_arr))
        std_len = float(np.std(seq_lengths_arr))
        var_ratio = float(std_len / avg_len) if avg_len > 0 else 0.0
        min_len = int(np.min(seq_lengths_arr))
        max_len = int(np.max(seq_lengths_arr))
        p95_len = float(np.percentile(seq_lengths_arr, 95))
        p99_len = float(np.percentile(seq_lengths_arr, 99))

        # Two-Factor Model Quadrant & Waste Calculation
        predicted_pad_waste = float(min(60.0, max(5.0, var_ratio * 52.0)))
        if var_ratio > 0.75 and p99_len > 1000:
            quadrant = "Dual Catastrophic Failure / OOM Crash"
        elif var_ratio > 0.75:
            quadrant = "Padding Memory Waste Only"
        elif p99_len > 1000:
            quadrant = "Throughput Collapse & Entropy Inflation"
        else:
            quadrant = "Clean Execution"

        warnings = []
        recommendations = []

        if oom_risk > 0:
            pct = (oom_risk / len(samples)) * 100
            warnings.append(f"⚠️ {oom_risk} samples ({pct:.1f}%) exceed max target length ({self.max_seq_len_threshold} tokens). High OOM risk on GPU!")
            recommendations.append(f"Truncate or filter samples exceeding {self.max_seq_len_threshold} tokens before launch.")

        if var_ratio > 0.75:
            warnings.append(f"⚠️ High Sequence-Length Variance Ratio (σ/μ = {var_ratio:.2f} > 0.75). Predicted Padding Waste: {predicted_pad_waste:.1f}%.")
            recommendations.append("Enforce Sequence Length Bucket Packing / Sorting to recover up to +128.4% throughput.")

        if p99_len > 1000:
            warnings.append(f"⚠️ Heavy Context Tail (P99 = {p99_len:.0f} tokens > 1000). High risk of Throughput Collapse & Attention Entropy Inflation.")
            recommendations.append("Configure vLLM Chunked Prefill with --max-num-batched-tokens matching P99 to preserve low TPOT SLAs.")

        if empty_completions > 0:
            warnings.append(f"⚠️ {empty_completions} samples have empty or missing completions.")
            recommendations.append("Filter out entries with empty completions to prevent model learning dummy tokens.")

        if loss_mask_misaligned > 0:
            warnings.append(f"⚠️ {loss_mask_misaligned} samples have misaligned loss masks (unmasked prompt tokens or missing assistant targets).")
            recommendations.append("Ensure DataCollatorForCompletionOnlyLM or labels = -100 is applied to prompt turns.")

        if duplicates > 0:
            pct = (duplicates / len(samples)) * 100
            warnings.append(f"⚠️ {duplicates} duplicate prompts detected ({pct:.1f}%).")
            recommendations.append("Deduplicate dataset prompts to avoid over-fitting and biased gradient updates.")

        ks_p = 1.0
        js_div = 0.0
        drift_alert = False

        if baseline_seq_lengths and len(baseline_seq_lengths) > 0:
            try:
                from scipy.stats import ks_2samp
                stat, p_val = ks_2samp(seq_lengths, baseline_seq_lengths)
                ks_p = float(p_val)
                if ks_p < 0.05:
                    drift_alert = True
                    warnings.append(f"⚠️ Dataset Drift Detected! (KS Test p-value = {ks_p:.4f} < 0.05). Prompt distribution shifted.")
                    recommendations.append("Re-profile model max context limits and vLLM chunked prefill parameters.")
            except ImportError:
                pass

        return SFTReport(
            total_samples=len(samples),
            avg_seq_len=avg_len,
            std_seq_len=std_len,
            variance_ratio=var_ratio,
            min_seq_len=min_len,
            max_seq_len=max_len,
            p95_seq_len=p95_len,
            p99_seq_len=p99_len,
            oom_risk_count=oom_risk,
            empty_completion_count=empty_completions,
            duplicate_count=duplicates,
            loss_mask_misaligned_count=loss_mask_misaligned,
            drift_alert=drift_alert,
            ks_p_value=ks_p,
            js_divergence=js_div,
            predicted_padding_waste_pct=predicted_pad_waste,
            phase_quadrant=quadrant,
            warnings=warnings,
            recommendations=recommendations,
        )
