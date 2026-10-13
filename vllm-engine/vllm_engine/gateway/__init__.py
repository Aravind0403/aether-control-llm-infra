"""Aether Admission Gateway package.

Provides bounded admission control, execution slot management, streaming reverse
proxying, and telemetry tracking for vLLM inference workloads.
"""

from vllm_engine.gateway.admission import (
    AdmissionMode,
    AdmissionResult,
    AdmissionShedReason,
    BoundedQueueController,
    QueueCapacityExceededError,
    QueueDeadlineExpiredError,
)
from vllm_engine.gateway.telemetry import (
    GatewayRequestRecord,
    GatewayTelemetryRecorder,
)
from vllm_engine.gateway.proxy import create_gateway_app

__all__ = [
    "AdmissionMode",
    "AdmissionResult",
    "AdmissionShedReason",
    "BoundedQueueController",
    "QueueCapacityExceededError",
    "QueueDeadlineExpiredError",
    "GatewayRequestRecord",
    "GatewayTelemetryRecorder",
    "create_gateway_app",
]
