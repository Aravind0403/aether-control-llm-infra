import time
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel

from trainsight.cache.signature import DatasetSignature, FastInvariantHasher


class CircuitBreakerStatus(BaseModel):
    is_tripped: bool
    fallback_used: bool = False
    exit_code: int = 0
    pessimistic_penalty_applied: bool = False
    latency_ms: float = 0.0
    message: str = "Storage normal."


class StorageCircuitBreaker:
    """3-Tier Storage Circuit Breaker to prevent thundering herds on shared filesystems."""

    def __init__(self, timeout_ms: float = 1500.0, pessimistic_penalty_pct: float = 10.0):
        self.timeout_ms = timeout_ms
        self.pessimistic_penalty_pct = pessimistic_penalty_pct

    def check_storage_access(
        self,
        file_path: Path,
        cached_signature: Optional[DatasetSignature] = None,
    ) -> CircuitBreakerStatus:
        """Checks storage access within timeout limit, applying pessimistic fallback if timed out."""
        start = time.perf_counter()

        if not file_path.exists():
            return CircuitBreakerStatus(
                is_tripped=True,
                exit_code=1,
                message=f"File not found: {file_path}",
            )

        try:
            # Simulate stat/access test
            _ = file_path.stat().st_size
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if elapsed_ms > self.timeout_ms:
                # Storage degraded
                if cached_signature:
                    # Layer 1: Pessimistic Stale Fallback
                    return CircuitBreakerStatus(
                        is_tripped=True,
                        fallback_used=True,
                        exit_code=0,
                        pessimistic_penalty_applied=True,
                        latency_ms=round(elapsed_ms, 2),
                        message=(
                            f"⚠️ Storage latency exceeded {self.timeout_ms}ms ({elapsed_ms:.1f}ms). "
                            f"Fell back to cached signature with a {self.pessimistic_penalty_pct}% pessimistic penalty."
                        ),
                    )
                else:
                    # Storage degraded and no cache: Exit code 2 (Infrastructure issue)
                    return CircuitBreakerStatus(
                        is_tripped=True,
                        fallback_used=False,
                        exit_code=2,
                        latency_ms=round(elapsed_ms, 2),
                        message=(
                            f"🚨 Storage Circuit Breaker Tripped! Storage latency {elapsed_ms:.1f}ms > {self.timeout_ms}ms "
                            "and no local cache available. Exiting with code 2 (Infrastructure Degraded)."
                        ),
                    )

            return CircuitBreakerStatus(
                is_tripped=False,
                fallback_used=False,
                exit_code=0,
                latency_ms=round(elapsed_ms, 2),
                message="Storage accessible within SLA.",
            )

        except Exception as e:
            if cached_signature:
                return CircuitBreakerStatus(
                    is_tripped=True,
                    fallback_used=True,
                    exit_code=0,
                    pessimistic_penalty_applied=True,
                    message=f"Storage I/O exception ({e}). Fell back to cached profile with 10% safety penalty.",
                )
            return CircuitBreakerStatus(
                is_tripped=True,
                fallback_used=False,
                exit_code=2,
                message=f"Storage unreachable: {e}. Tripping circuit breaker with exit code 2.",
            )

    def apply_pessimistic_penalty(self, seq_lengths: List[int]) -> List[int]:
        """Inflates sequence lengths by pessimistic penalty percentage to absorb hidden variance."""
        multiplier = 1.0 + (self.pessimistic_penalty_pct / 100.0)
        return [int(round(s * multiplier)) for s in seq_lengths]
