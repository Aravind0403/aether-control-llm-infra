import time
import uuid
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class GatewayRequestRecord:
    """Structured telemetry record for an individual gateway request."""
    request_id: str
    arrival_time: float
    queue_wait_ms: float
    ttft_ms: Optional[float] = None
    e2e_latency_ms: float = 0.0
    status_code: int = 200
    admitted: bool = True
    shed_reason: Optional[str] = None
    completion_tokens: int = 0
    is_estimated_tokens: bool = False
    client_disconnected: bool = False
    stream_status: str = "COMPLETED"  # COMPLETED, TRUNCATED, CANCELLED, SHED, ERROR
    error: Optional[str] = None
    inter_event_latencies_ms: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "arrival_time": self.arrival_time,
            "queue_wait_ms": round(self.queue_wait_ms, 2),
            "ttft_ms": round(self.ttft_ms, 2) if self.ttft_ms is not None else None,
            "e2e_latency_ms": round(self.e2e_latency_ms, 2),
            "status_code": self.status_code,
            "admitted": self.admitted,
            "shed_reason": self.shed_reason,
            "completion_tokens": self.completion_tokens,
            "is_estimated_tokens": self.is_estimated_tokens,
            "client_disconnected": self.client_disconnected,
            "stream_status": self.stream_status,
            "error": self.error,
        }


class GatewayTelemetryRecorder:
    """In-memory telemetry buffer and metric aggregator for gateway experiments."""

    def __init__(self, max_records: int = 10000):
        self.max_records = max_records
        self.records: List[GatewayRequestRecord] = []

    def generate_request_id(self) -> str:
        return f"aether-{uuid.uuid4().hex[:12]}"

    def record(self, rec: GatewayRequestRecord):
        if len(self.records) >= self.max_records:
            self.records.pop(0)
        self.records.append(rec)

    def clear(self):
        self.records.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Summarizes currently buffered telemetry records."""
        total = len(self.records)
        if total == 0:
            return {"total_requests": 0}

        admitted = [r for r in self.records if r.admitted]
        shed = [r for r in self.records if not r.admitted]
        completed = [r for r in admitted if r.status_code == 200 and not r.client_disconnected]
        disconnected = [r for r in self.records if r.client_disconnected]

        ttfts = [r.ttft_ms for r in completed if r.ttft_ms is not None and r.ttft_ms > 0]
        e2es = [r.e2e_latency_ms for r in completed]
        tokens = sum(r.completion_tokens for r in completed)

        import numpy as np

        def calc_p(values):
            if not values:
                return {"p50": 0.0, "p90": 0.0, "p99": 0.0}
            arr = np.array(values)
            return {
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
                "p99": float(np.percentile(arr, 99)),
            }

        return {
            "total_offered": total,
            "completed": len(completed),
            "shed": len(shed),
            "client_disconnected": len(disconnected),
            "total_completion_tokens": tokens,
            "ttft_ms": calc_p(ttfts),
            "e2e_latency_ms": calc_p(e2es),
        }
