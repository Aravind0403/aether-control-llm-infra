from pydantic import BaseModel, Field
from typing import Dict, Any


class RadixDeduplicationReport(BaseModel):
    """Encapsulates KV cache deduplication analytics across G completions (RFC-003)."""
    group_size: int
    prompt_tokens: int
    generation_tokens: int
    naive_total_tokens: int
    radix_total_tokens: int
    tokens_saved: int
    total_token_reduction_pct: float
    prompt_redundancy_eliminated_pct: float
    naive_kv_bytes_fp16: int
    radix_kv_bytes_fp16: int
    kv_cache_saved_mb: float


class RadixCacheSimulator:
    """Simulates SGLang / vLLM Radix Tree prefix caching across GRPO completion groups (RFC-003)."""

    def __init__(
        self,
        num_layers: int = 80,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        bytes_per_element: int = 2,  # FP16 / BF16
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.bytes_per_element = bytes_per_element

    def token_to_kv_bytes(self, num_tokens: int) -> int:
        """KV cache bytes for a given number of cached tokens across all layers."""
        # 2 * num_layers * num_kv_heads * num_tokens * head_dim * bytes_per_element
        return 2 * self.num_layers * self.num_kv_heads * num_tokens * self.head_dim * self.bytes_per_element

    def evaluate_group_savings(
        self,
        group_size: int = 8,
        prompt_tokens: int = 1024,
        generation_tokens: int = 1024,
    ) -> RadixDeduplicationReport:
        """Calculates exact memory and token savings when prefix sharing is activated."""
        # Naive: Each of the G completions stores its own copy of the prompt + generation
        naive_tokens = group_size * (prompt_tokens + generation_tokens)

        # Radix: Prompt is stored ONCE as the root node; G distinct branches for generations
        radix_tokens = (1 * prompt_tokens) + (group_size * generation_tokens)

        tokens_saved = naive_tokens - radix_tokens
        token_reduction_pct = (tokens_saved / naive_tokens) * 100.0

        # Prompt redundancy: (G - 1) copies of the prompt eliminated
        prompt_redundancy_eliminated_pct = ((group_size - 1) / group_size) * 100.0

        naive_bytes = self.token_to_kv_bytes(naive_tokens)
        radix_bytes = self.token_to_kv_bytes(radix_tokens)
        bytes_saved = naive_bytes - radix_bytes

        return RadixDeduplicationReport(
            group_size=group_size,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            naive_total_tokens=naive_tokens,
            radix_total_tokens=radix_tokens,
            tokens_saved=tokens_saved,
            total_token_reduction_pct=token_reduction_pct,
            prompt_redundancy_eliminated_pct=prompt_redundancy_eliminated_pct,
            naive_kv_bytes_fp16=naive_bytes,
            radix_kv_bytes_fp16=radix_bytes,
            kv_cache_saved_mb=bytes_saved / (1024.0 * 1024.0),
        )
