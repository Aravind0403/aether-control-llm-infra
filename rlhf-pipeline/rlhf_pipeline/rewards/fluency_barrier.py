from typing import Dict, Any, Tuple
from pydantic import BaseModel, Field
import re
import math


class FluencyEvaluationResult(BaseModel):
    raw_metrics_reward: float
    estimated_perplexity: float
    fluency_threshold: float
    fluency_penalty: float
    final_reward: float
    is_keyword_soup: bool
    diagnosis: str


class ReferencePPLFluencyBarrier:
    """Penalizes telegraphic keyword soup when perplexity under reference base model exceeds natural prose thresholds (RFC-005).
    
    Reward = R_verifiable - alpha * max(0, PPL - tau_natural)
    """

    def __init__(
        self,
        tau_natural: float = 14.0,
        alpha_penalty: float = 0.05,
        max_penalty: float = 2.0,
    ):
        self.tau_natural = tau_natural
        self.alpha_penalty = alpha_penalty
        self.max_penalty = max_penalty

    @staticmethod
    def estimate_linguistic_perplexity(text: str) -> float:
        """Estimates linguistic perplexity using grammatical structure, functional words, and fragment penalties."""
        cleaned = text.strip()
        words = re.findall(r"\b\w+\b", cleaned.lower())
        if not words:
            return 100.0

        # Functional words in natural English (prepositions, conjunctions, articles, pronouns)
        functional_words = {
            "the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "of",
            "and", "but", "or", "is", "was", "are", "were", "been", "it", "this",
            "that", "from", "as", "into", "reported", "under", "despite", "which"
        }

        func_count = sum(1 for w in words if w in functional_words)
        func_ratio = func_count / float(len(words))

        # Sentence fragments: split by periods
        sentences = [s.strip() for s in cleaned.split(".") if s.strip()]
        avg_sentence_len = len(words) / float(max(1, len(sentences)))

        # Natural prose typically has func_ratio between 0.35 and 0.55, and avg sentence length > 6 words.
        # Fragmented keyword soup has func_ratio < 0.15 and avg sentence length < 3 words.
        base_ppl = 9.0

        # Penalty for missing functional words
        if func_ratio < 0.20:
            deficit = (0.20 - func_ratio) / 0.20
            base_ppl += deficit * 50.0  # massive spike for telegraphic lists

        # Penalty for ultra-short choppy fragments
        if avg_sentence_len < 4.0:
            shortness = (4.0 - avg_sentence_len) / 4.0
            base_ppl += shortness * 30.0

        return round(base_ppl, 2)

    def evaluate(self, metrics_reward: float, completion: str) -> FluencyEvaluationResult:
        """Evaluates whether the completion passes the fluency barrier."""
        ppl = self.estimate_linguistic_perplexity(completion)

        excess_ppl = max(0.0, ppl - self.tau_natural)
        penalty = min(self.max_penalty, self.alpha_penalty * excess_ppl)
        final_reward = metrics_reward - penalty
        is_soup = bool(ppl > 30.0)

        if is_soup:
            diag = (
                f"🚨 Keyword soup detected (Estimated PPL={ppl:.1f} > τ={self.tau_natural:.1f}). "
                f"Applied fluency penalty of -{penalty:.2f} to raw reward {metrics_reward:.2f}."
            )
        else:
            diag = (
                f"✅ Fluent natural prose (Estimated PPL={ppl:.1f} <= τ={self.tau_natural:.1f}). "
                f"Zero fluency penalty applied."
            )

        return FluencyEvaluationResult(
            raw_metrics_reward=round(metrics_reward, 3),
            estimated_perplexity=round(ppl, 2),
            fluency_threshold=self.tau_natural,
            fluency_penalty=round(penalty, 3),
            final_reward=round(final_reward, 3),
            is_keyword_soup=is_soup,
            diagnosis=diag,
        )
