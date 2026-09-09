from pydantic import BaseModel
from typing import Dict, Any, Optional
import re


class CascadeRoutingDecision(BaseModel):
    model_target: str       # '7b_spot_fast_lane' or '70b_primary_deep_lane'
    is_spillover: bool
    estimated_latency_ms: float
    confidence: float
    reason: str


class ModelCascadeRouter:
    """Routes requests between Primary 70B and Warm 7B Spot Pool during traffic bursts (VLLM-RFC-001)."""

    def __init__(
        self,
        length_threshold_tokens: int = 64,
        confidence_threshold: float = 0.70,
    ):
        self.length_threshold_tokens = length_threshold_tokens
        self.confidence_threshold = confidence_threshold

    def route(self, prompt: str, primary_saturated: bool = False) -> CascadeRoutingDecision:
        """Determines routing destination based on query characteristics and cluster saturation."""
        tokens = re.findall(r"\b\w+\b", prompt)
        token_count = len(tokens)

        # Check for obvious reasoning indicators
        reasoning_keywords = {"prove", "calculate", "derive", "solve", "math", "cuda", "theorem"}
        has_reasoning = any(w.lower() in reasoning_keywords for w in tokens)

        if not primary_saturated and (has_reasoning or token_count > self.length_threshold_tokens):
            return CascadeRoutingDecision(
                model_target="70b_primary_deep_lane",
                is_spillover=False,
                estimated_latency_ms=280.0,
                confidence=0.95,
                reason="Complex query / nominal load routed to primary 70B.",
            )

        if primary_saturated and not has_reasoning and token_count <= self.length_threshold_tokens:
            return CascadeRoutingDecision(
                model_target="7b_spot_fast_lane",
                is_spillover=True,
                estimated_latency_ms=35.0,
                confidence=0.88,
                reason="Simple query diverted to warm 7B spot pool to absorb burst.",
            )

        if primary_saturated and has_reasoning:
            # Complex query during saturation stays in 70B queue or gets priority
            return CascadeRoutingDecision(
                model_target="70b_primary_deep_lane",
                is_spillover=False,
                estimated_latency_ms=500.0,
                confidence=0.90,
                reason="Reasoning query retained in 70B queue despite saturation.",
            )

        return CascadeRoutingDecision(
            model_target="7b_spot_fast_lane",
            is_spillover=False,
            estimated_latency_ms=40.0,
            confidence=0.85,
            reason="Short conversational query routed to 7B fast lane.",
        )
