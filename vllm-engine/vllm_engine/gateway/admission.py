import asyncio
import time
from contextlib import asynccontextmanager
from enum import Enum
from typing import Optional, NamedTuple


class AdmissionMode(str, Enum):
    PASSTHROUGH = "passthrough"
    STATIC_CONCURRENCY = "static_concurrency"
    BOUNDED_QUEUE = "bounded_queue"


class AdmissionShedReason(str, Enum):
    CAPACITY_EXCEEDED = "QueueCapacityExceeded"
    DEADLINE_EXPIRED = "QueueDeadlineExpired"


class AdmissionResult(NamedTuple):
    admitted: bool
    queue_wait_ms: float
    status_code: int
    shed_reason: Optional[AdmissionShedReason] = None
    retry_after_s: int = 5


class QueueCapacityExceededError(Exception):
    """Raised when queue depth exceeds max_queue capacity."""
    def __init__(self, retry_after_s: int = 5):
        super().__init__("Queue capacity exceeded; request shed with HTTP 429")
        self.retry_after_s = retry_after_s


class QueueDeadlineExpiredError(Exception):
    """Raised when request waits in queue longer than queue_deadline_ms."""
    def __init__(self, wait_ms: float, retry_after_s: int = 5):
        super().__init__(f"Queue deadline expired after {wait_ms:.1f}ms; request shed with HTTP 429")
        self.wait_ms = wait_ms
        self.retry_after_s = retry_after_s


class BoundedQueueController:
    """Bounded Queue & Execution Slot Admission Controller.
    
    Implements true bounded admission control:
    - Bounded concurrency: Limits active upstream vLLM requests to `max_concurrency`.
    - Bounded queue: If waiting queue reaches `max_queue`, sheds immediately with 429.
    - Queue deadline: If wait in queue reaches `deadline_ms` before slot acquisition, sheds with 429.
    - Holds execution slot until the upstream stream closes (guaranteed cleanup).
    """

    def __init__(
        self,
        mode: AdmissionMode = AdmissionMode.BOUNDED_QUEUE,
        max_concurrency: int = 8,
        max_queue: int = 32,
        deadline_ms: float = 500.0,
        retry_after_s: int = 5,
    ):
        self.mode = mode
        self.max_concurrency = max_concurrency
        self.max_queue = max_queue
        self.deadline_s = deadline_ms / 1000.0
        self.deadline_ms = deadline_ms
        self.retry_after_s = retry_after_s

        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._lock = asyncio.Lock()
        self._active_count = 0
        self._queued_count = 0

        # Telemetry metrics
        self.total_offered = 0
        self.total_admitted = 0
        self.total_shed_capacity = 0
        self.total_shed_deadline = 0

    @property
    def active_requests(self) -> int:
        return self._active_count

    @property
    def queued_requests(self) -> int:
        return self._queued_count

    @asynccontextmanager
    async def acquire(self):
        """Context manager for acquiring and guaranteed release of an execution slot."""
        wait_ms = await self.acquire_slot()
        try:
            yield wait_ms
        finally:
            self.release_slot()

    async def acquire_slot(self) -> float:
        """Acquires an execution slot according to the admission mode.
        
        Returns:
            queue_wait_ms: Time spent waiting in queue before acquiring slot.
            
        Raises:
            QueueCapacityExceededError: If queue is at capacity.
            QueueDeadlineExpiredError: If deadline elapsed while waiting.
        """
        arrival_t = time.perf_counter()

        async with self._lock:
            self.total_offered += 1

            if self.mode == AdmissionMode.PASSTHROUGH:
                self._active_count += 1
                self.total_admitted += 1
                return 0.0

            if self.mode == AdmissionMode.STATIC_CONCURRENCY:
                if self._active_count >= self.max_concurrency:
                    self.total_shed_capacity += 1
                    raise QueueCapacityExceededError(retry_after_s=self.retry_after_s)
                self._active_count += 1
                self.total_admitted += 1
                return 0.0

            if self.mode == AdmissionMode.BOUNDED_QUEUE:
                # If an execution slot is immediately free, acquire it directly without queueing
                if self._active_count < self.max_concurrency:
                    await self._semaphore.acquire()
                    self._active_count += 1
                    self.total_admitted += 1
                    return 0.0

                # All execution slots are busy; request must wait in queue
                if self._queued_count >= self.max_queue:
                    self.total_shed_capacity += 1
                    raise QueueCapacityExceededError(retry_after_s=self.retry_after_s)

                self._queued_count += 1

        # Request entered waiting queue; wait for semaphore slot with deadline
        in_queue = True
        try:
            try:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=self.deadline_s)
            except asyncio.TimeoutError:
                async with self._lock:
                    if in_queue:
                        self._queued_count = max(0, self._queued_count - 1)
                        self.total_shed_deadline += 1
                        in_queue = False
                wait_ms = (time.perf_counter() - arrival_t) * 1000.0
                raise QueueDeadlineExpiredError(wait_ms=wait_ms, retry_after_s=self.retry_after_s)

            # Slot acquired successfully from queue
            async with self._lock:
                if in_queue:
                    self._queued_count = max(0, self._queued_count - 1)
                    in_queue = False
                self._active_count += 1
                self.total_admitted += 1

            queue_wait_ms = (time.perf_counter() - arrival_t) * 1000.0
            return queue_wait_ms

        except (QueueCapacityExceededError, QueueDeadlineExpiredError):
            raise
        except BaseException:
            # Handles CancelledError and any unexpected exception
            async with self._lock:
                if in_queue:
                    self._queued_count = max(0, self._queued_count - 1)
                    in_queue = False
            raise

    def release_slot(self):
        """Releases the execution slot held by the request."""
        if self.mode == AdmissionMode.PASSTHROUGH:
            self._active_count = max(0, self._active_count - 1)
            return

        if self.mode == AdmissionMode.STATIC_CONCURRENCY:
            self._active_count = max(0, self._active_count - 1)
            return

        # Mode: BOUNDED_QUEUE
        self._active_count = max(0, self._active_count - 1)
        self._semaphore.release()
