import asyncio
import pytest
from vllm_engine.gateway.admission import (
    BoundedQueueController,
    AdmissionMode,
    QueueCapacityExceededError,
    QueueDeadlineExpiredError,
)


@pytest.mark.asyncio
async def test_passthrough_mode():
    controller = BoundedQueueController(mode=AdmissionMode.PASSTHROUGH)
    # Even with many concurrent requests, all should acquire immediately
    async with controller.acquire():
        assert controller.active_requests == 1
        async with controller.acquire():
            assert controller.active_requests == 2
    assert controller.active_requests == 0
    assert controller.queued_requests == 0


@pytest.mark.asyncio
async def test_static_concurrency_mode_limit_and_shed():
    controller = BoundedQueueController(
        mode=AdmissionMode.STATIC_CONCURRENCY,
        max_concurrency=2,
    )

    async with controller.acquire():
        assert controller.active_requests == 1
        async with controller.acquire():
            assert controller.active_requests == 2

            # 3rd concurrent request should immediately be shed with 429 (QueueCapacityExceededError)
            with pytest.raises(QueueCapacityExceededError) as exc_info:
                async with controller.acquire():
                    pass
            assert exc_info.value.retry_after_s == 5

    assert controller.active_requests == 0


@pytest.mark.asyncio
async def test_bounded_queue_capacity_shed():
    # max_concurrency=1, max_queue=2
    controller = BoundedQueueController(
        mode=AdmissionMode.BOUNDED_QUEUE,
        max_concurrency=1,
        max_queue=2,
        deadline_ms=5000.0,
    )

    # Slot 1 active
    async with controller.acquire():
        assert controller.active_requests == 1

        # Launch 2 background tasks that enter queue
        task1 = asyncio.create_task(controller.acquire().__aenter__())
        task2 = asyncio.create_task(controller.acquire().__aenter__())
        await asyncio.sleep(0.02)
        assert controller.queued_requests == 2

        # 3rd queued request should exceed queue capacity (max_queue=2)
        with pytest.raises(QueueCapacityExceededError):
            async with controller.acquire():
                pass

        # Cleanup background tasks
        task1.cancel()
        task2.cancel()
        await asyncio.gather(task1, task2, return_exceptions=True)


@pytest.mark.asyncio
async def test_bounded_queue_deadline_expiration():
    # max_concurrency=1, max_queue=5, deadline=50ms
    controller = BoundedQueueController(
        mode=AdmissionMode.BOUNDED_QUEUE,
        max_concurrency=1,
        max_queue=5,
        deadline_ms=50.0,
    )

    async with controller.acquire():
        assert controller.active_requests == 1

        # Second request waits in queue; holder stays active for 100ms
        with pytest.raises(QueueDeadlineExpiredError) as exc_info:
            async with controller.acquire():
                pass

        assert exc_info.value.wait_ms >= 45.0
        assert controller.queued_requests == 0

    assert controller.active_requests == 0


@pytest.mark.asyncio
async def test_bounded_queue_successful_drain():
    controller = BoundedQueueController(
        mode=AdmissionMode.BOUNDED_QUEUE,
        max_concurrency=1,
        max_queue=5,
        deadline_ms=500.0,
    )

    acquired_order = []

    async def worker(worker_id: int):
        async with controller.acquire():
            acquired_order.append(worker_id)
            await asyncio.sleep(0.05)

    await asyncio.gather(worker(1), worker(2))
    assert acquired_order == [1, 2]
    assert controller.active_requests == 0
    assert controller.queued_requests == 0


@pytest.mark.asyncio
async def test_slot_released_on_exception_or_cancellation():
    controller = BoundedQueueController(
        mode=AdmissionMode.BOUNDED_QUEUE,
        max_concurrency=1,
    )

    with pytest.raises(RuntimeError):
        async with controller.acquire():
            raise RuntimeError("Simulation failure")

    assert controller.active_requests == 0

    # Test cancellation
    async def cancelled_worker():
        async with controller.acquire():
            await asyncio.sleep(10.0)

    task = asyncio.create_task(cancelled_worker())
    await asyncio.sleep(0.02)
    assert controller.active_requests == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert controller.active_requests == 0


@pytest.mark.asyncio
async def test_bounded_queue_max_queue_zero():
    # When max_queue=0, free execution slots must be usable, but excess requests shed
    controller = BoundedQueueController(
        mode=AdmissionMode.BOUNDED_QUEUE,
        max_concurrency=2,
        max_queue=0,
    )

    # 1st and 2nd should acquire immediately
    async with controller.acquire():
        assert controller.active_requests == 1
        async with controller.acquire():
            assert controller.active_requests == 2

            # 3rd request has no slots and queue capacity is 0 -> sheds with 429
            with pytest.raises(QueueCapacityExceededError):
                async with controller.acquire():
                    pass

    assert controller.active_requests == 0
    assert controller.queued_requests == 0


@pytest.mark.asyncio
async def test_cancellation_while_queued_cleans_queue_counter():
    controller = BoundedQueueController(
        mode=AdmissionMode.BOUNDED_QUEUE,
        max_concurrency=1,
        max_queue=5,
        deadline_ms=5000.0,
    )

    async with controller.acquire():
        assert controller.active_requests == 1

        # Launch waiting worker in background
        async def waiting_worker():
            async with controller.acquire():
                pass

        wait_task = asyncio.create_task(waiting_worker())
        await asyncio.sleep(0.02)
        assert controller.queued_requests == 1

        # Cancel waiting task before it acquires slot
        wait_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await wait_task

        # Ensure queued counter was properly decremented
        assert controller.queued_requests == 0

    assert controller.active_requests == 0

