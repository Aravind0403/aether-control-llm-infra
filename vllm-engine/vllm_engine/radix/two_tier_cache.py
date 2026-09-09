from typing import Dict, List, Set, Optional, Any
from pydantic import BaseModel, Field
import hashlib
import time


class CacheBlockAllocation(BaseModel):
    block_id: int
    tenant_id: str
    is_pinned: bool
    token_count: int
    last_accessed: float


class TwoTierRadixCacheManager:
    """Manages Two-Tier Radix Cache Namespaces with Pinned System Roots & Tenant Quotas (VLLM-RFC-002).
    
    - Tier 1: High-Trust Global Pinned Roots (Immutable, immune to LRU eviction).
    - Tier 2: Dynamic Tenant Pool (Per-tenant quota capped at max 15% of blocks).
    """

    def __init__(self, total_blocks: int = 1000, max_tenant_quota_pct: float = 0.15):
        self.total_blocks = total_blocks
        self.max_tenant_blocks = int(total_blocks * max_tenant_quota_pct)

        self.pinned_blocks: Dict[str, CacheBlockAllocation] = {}
        self.tenant_allocations: Dict[str, Dict[int, CacheBlockAllocation]] = {}
        self.next_block_id: int = 1

    def pin_system_prompt(self, prompt_text: str, num_blocks: int) -> str:
        """Pins an enterprise system prompt as an immutable root node."""
        prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]

        for i in range(num_blocks):
            bid = self.next_block_id
            self.next_block_id += 1
            alloc = CacheBlockAllocation(
                block_id=bid,
                tenant_id="__SYSTEM__",
                is_pinned=True,
                token_count=16,
                last_accessed=time.time(),
            )
            self.pinned_blocks[f"{prompt_hash}_{i}"] = alloc

        return prompt_hash

    def allocate_tenant_blocks(self, tenant_id: str, requested_blocks: int) -> List[int]:
        """Allocates blocks for a tenant, strictly enforcing the 15% quota ceiling."""
        if tenant_id not in self.tenant_allocations:
            self.tenant_allocations[tenant_id] = {}

        current_blocks = self.tenant_allocations[tenant_id]
        allocated_ids = []

        for _ in range(requested_blocks):
            if len(current_blocks) >= self.max_tenant_blocks:
                # Quota reached: Evict OLDEST block belonging to THIS tenant only (LRU)
                oldest_id = min(current_blocks.keys(), key=lambda k: current_blocks[k].last_accessed)
                del current_blocks[oldest_id]

            bid = self.next_block_id
            self.next_block_id += 1
            alloc = CacheBlockAllocation(
                block_id=bid,
                tenant_id=tenant_id,
                is_pinned=False,
                token_count=16,
                last_accessed=time.time(),
            )
            current_blocks[bid] = alloc
            allocated_ids.append(bid)

        return allocated_ids

    def get_stats(self) -> Dict[str, Any]:
        pinned_count = len(self.pinned_blocks)
        dynamic_count = sum(len(b) for b in self.tenant_allocations.values())
        return {
            "total_blocks": self.total_blocks,
            "pinned_system_blocks": pinned_count,
            "dynamic_tenant_blocks": dynamic_count,
            "free_blocks": max(0, self.total_blocks - pinned_count - dynamic_count),
            "max_tenant_quota": self.max_tenant_blocks,
        }
