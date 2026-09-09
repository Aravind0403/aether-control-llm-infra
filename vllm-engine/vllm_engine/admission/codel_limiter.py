from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import time


class AdmissionDecision(BaseModel):
    allowed: bool
    status_code: int
    queue_delay_ms: float
    retry_after_seconds: Optional[int] = None
    rate_limit_reason: Optional[str] = None
    route_target: str = "primary_70b"


class CoDelLoadShedder:
    """Controlled Delay (CoDel) Ingress Admission Controller (VLLM-RFC-001).
    
    Monitors live P99 queue delay. If queue wait exceeds target SLA (500ms),
    excess requests are immediately shed with HTTP 429 and Retry-After headers,
    preserving running request SLAs and preventing KV preemption storms.
    """

    def __init__(
        self,
        target_delay_ms: float = 500.0,
        retry_after_seconds: int = 15,
    ):
        self.target_delay_ms = target_delay_ms
        self.retry_after_seconds = retry_after_seconds

    def evaluate_request(self, current_queue_delay_ms: float) -> AdmissionDecision:
        """Evaluates whether to admit or shed an incoming request based on queue delay."""
        if current_queue_delay_ms <= self.target_delay_ms:
            return AdmissionDecision(
                allowed=True,
                status_code=200,
                queue_delay_ms=current_queue_delay_ms,
                route_target="primary_70b",
            )
        else:
            return AdmissionDecision(
                allowed=False,
                status_code=429,
                queue_delay_ms=current_queue_delay_ms,
                retry_after_seconds=self.retry_after_seconds,
                rate_limit_reason="QueueSaturationDelayExceeded",
                route_target="shed",
            )
