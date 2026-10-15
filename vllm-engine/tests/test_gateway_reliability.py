import asyncio
import json
import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, Response
import httpx
from starlette.requests import Request

from vllm_engine.gateway.admission import AdmissionMode
from vllm_engine.gateway.proxy import create_gateway_app, SSEEventParser, _is_content_event


# ---------------------------------------------------------------------------
# 1. SSEEventParser Unit & Boundary Tests
# ---------------------------------------------------------------------------

def test_sse_parser_split_across_chunk_boundaries():
    """Verify that SSEEventParser correctly reassembles events split across arbitrary network chunks."""
    parser = SSEEventParser()

    # Event 1: {"choices": [{"delta": {"content": "Hello"}}]}
    # Event 2: {"choices": [], "usage": {"completion_tokens": 10}}
    # Event 3: [DONE]
    raw_payload_1 = json.dumps({"choices": [{"delta": {"content": "Hello"}}]})
    raw_payload_2 = json.dumps({"choices": [], "usage": {"completion_tokens": 10}})

    full_stream = (
        f"data: {raw_payload_1}\n\n"
        f"data: {raw_payload_2}\n\n"
        "data: [DONE]\n\n"
    ).encode("utf-8")

    # Split stream into tiny 3-byte arbitrary network chunks
    chunk_size = 3
    collected_events = []
    for i in range(0, len(full_stream), chunk_size):
        chunk = full_stream[i : i + chunk_size]
        events = parser.feed(chunk)
        collected_events.extend(events)

    assert len(collected_events) == 3
    # Check event 1
    assert collected_events[0][1]["choices"][0]["delta"]["content"] == "Hello"
    # Check event 2
    assert collected_events[1][1]["usage"]["completion_tokens"] == 10
    # Check terminal [DONE]
    assert collected_events[2][0] == "[DONE]"
    assert collected_events[2][1] is None


def test_is_content_event_role_vs_content():
    """Verify that _is_content_event ignores role-only metadata and only accepts generated text."""
    role_only = {"choices": [{"delta": {"role": "assistant"}}]}
    empty_content = {"choices": [{"delta": {"role": "assistant", "content": ""}}]}
    none_content = {"choices": [{"delta": {"content": None}}]}
    real_content = {"choices": [{"delta": {"content": "First token"}}]}
    usage_only = {"choices": [], "usage": {"completion_tokens": 5}}

    assert _is_content_event(role_only) is False
    assert _is_content_event(empty_content) is False
    assert _is_content_event(none_content) is False
    assert _is_content_event(usage_only) is False
    assert _is_content_event(real_content) is True


# ---------------------------------------------------------------------------
# 2. Upstream Failure & Guaranteed Slot Release Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_slot_released_when_upstream_connection_fails():
    """Confirm admission slot is released when connecting upstream raises ConnectError."""
    # Gateway pointing to an unreachable backend address
    gateway_app = create_gateway_app(
        backend_url="http://127.0.0.1:54321",
        admission_mode=AdmissionMode.BOUNDED_QUEUE,
        max_concurrency=2,
        max_queue=4,
    )
    transport = httpx.ASGITransport(app=gateway_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # A. Streaming request to unreachable upstream
        resp_stream = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [], "stream": True},
        )
        assert resp_stream.status_code == 502
        assert gateway_app.state.controller.active_requests == 0
        assert gateway_app.state.controller.queued_requests == 0

        # B. Non-streaming unary request to unreachable upstream
        resp_unary = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [], "stream": False},
        )
        assert resp_unary.status_code == 502
        assert gateway_app.state.controller.active_requests == 0
        assert gateway_app.state.controller.queued_requests == 0


@pytest.mark.asyncio
async def test_slot_released_when_upstream_returns_500():
    """Confirm admission slot is released when upstream returns HTTP 500 error."""
    mock_backend = FastAPI()

    @mock_backend.post("/v1/chat/completions")
    async def failing_endpoint(req: Request):
        return Response(content="Internal Server Error", status_code=500)

    mock_transport = httpx.ASGITransport(app=mock_backend)
    gateway_app = create_gateway_app(
        backend_url="http://mock-vllm",
        admission_mode=AdmissionMode.BOUNDED_QUEUE,
        max_concurrency=1,
    )
    gateway_transport = httpx.ASGITransport(app=gateway_app)

    async with httpx.AsyncClient(transport=gateway_transport, base_url="http://test") as client:
        gateway_app.state.client = httpx.AsyncClient(transport=mock_transport, base_url="http://mock-vllm")

        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [], "stream": True},
        )
        assert resp.status_code == 500
        assert gateway_app.state.controller.active_requests == 0
        assert gateway_app.state.controller.queued_requests == 0


# ---------------------------------------------------------------------------
# 3. Client Cancellation & Disconnect Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_client_cancellation_during_streaming_releases_slot():
    """Verify that cancelling a client request during streaming releases the admission slot."""
    stream_started = asyncio.Event()

    mock_backend = FastAPI()

    @mock_backend.post("/v1/chat/completions")
    async def slow_stream(req: Request):
        async def slow_gen():
            stream_started.set()
            yield f"data: {json.dumps({'choices': [{'delta': {'content': 'Start'}}]})}\n\n"
            while True:
                await asyncio.sleep(0.05)
                yield f"data: {json.dumps({'choices': [{'delta': {'content': '.'}}]})}\n\n"

        return StreamingResponse(slow_gen(), media_type="text/event-stream")

    mock_transport = httpx.ASGITransport(app=mock_backend)
    gateway_app = create_gateway_app(
        backend_url="http://mock-vllm",
        admission_mode=AdmissionMode.BOUNDED_QUEUE,
        max_concurrency=1,
    )
    gateway_transport = httpx.ASGITransport(app=gateway_app)

    async with httpx.AsyncClient(transport=gateway_transport, base_url="http://test") as client:
        gateway_app.state.client = httpx.AsyncClient(transport=mock_transport, base_url="http://mock-vllm")

        req_task = asyncio.create_task(
            client.post("/v1/chat/completions", json={"model": "test", "stream": True})
        )
        await asyncio.sleep(0.05)
        assert gateway_app.state.controller.active_requests == 1

        # Client aborts / cancels request
        req_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await req_task

        await asyncio.sleep(0.05)
        # Slot MUST be released
        assert gateway_app.state.controller.active_requests == 0
        assert gateway_app.state.controller.queued_requests == 0


@pytest.mark.asyncio
async def test_client_cancellation_during_unary_releases_slot():
    """Verify that cancelling a client request during non-streaming call releases the admission slot."""
    mock_backend = FastAPI()

    @mock_backend.post("/v1/chat/completions")
    async def slow_unary(req: Request):
        await asyncio.sleep(2.0)
        return {"choices": [{"message": {"role": "assistant", "content": "Done"}}]}

    mock_transport = httpx.ASGITransport(app=mock_backend)
    gateway_app = create_gateway_app(
        backend_url="http://mock-vllm",
        admission_mode=AdmissionMode.BOUNDED_QUEUE,
        max_concurrency=1,
    )
    gateway_transport = httpx.ASGITransport(app=gateway_app)

    async with httpx.AsyncClient(transport=gateway_transport, base_url="http://test") as client:
        gateway_app.state.client = httpx.AsyncClient(transport=mock_transport, base_url="http://mock-vllm")

        req_task = asyncio.create_task(
            client.post("/v1/chat/completions", json={"model": "test", "stream": False})
        )
        await asyncio.sleep(0.05)
        assert gateway_app.state.controller.active_requests == 1

        # Client aborts / cancels
        req_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await req_task

        await asyncio.sleep(0.05)
        assert gateway_app.state.controller.active_requests == 0
        assert gateway_app.state.controller.queued_requests == 0


@pytest.mark.asyncio
async def test_client_cancellation_while_queued_cleans_counter():
    """Verify that client cancellation while waiting in admission queue decrements queued counter."""
    gateway_app = create_gateway_app(
        backend_url="http://mock-vllm",
        admission_mode=AdmissionMode.BOUNDED_QUEUE,
        max_concurrency=1,
        max_queue=5,
        deadline_ms=5000.0,
    )

    # Acquire the single available slot
    await gateway_app.state.controller.acquire_slot()
    assert gateway_app.state.controller.active_requests == 1

    # Now launch a 2nd task that will wait in the queue
    queued_task = asyncio.create_task(gateway_app.state.controller.acquire_slot())
    await asyncio.sleep(0.05)
    assert gateway_app.state.controller.queued_requests == 1

    # Cancel the waiting task
    queued_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued_task

    # Queue counter must return to 0
    assert gateway_app.state.controller.queued_requests == 0
    # Release the original slot
    gateway_app.state.controller.release_slot()
    assert gateway_app.state.controller.active_requests == 0


# ---------------------------------------------------------------------------
# 4. Accurate TTFT Timing Test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ttft_measures_at_first_content_event_not_role_header():
    """Verify that TTFT is recorded at the first generated content event, not the initial role-only event."""
    mock_backend = FastAPI()

    @mock_backend.post("/v1/chat/completions")
    async def chat_with_role_header(req: Request):
        async def token_stream():
            # Event 1: Initial role-only header (content is None) - emitted immediately
            role_data = {"id": "1", "choices": [{"delta": {"role": "assistant"}}]}
            yield f"data: {json.dumps(role_data)}\n\n"

            # Sleep 150ms before emitting the first real generated token
            await asyncio.sleep(0.15)
            content_data = {"id": "1", "choices": [{"delta": {"content": "Actual word"}}]}
            yield f"data: {json.dumps(content_data)}\n\n"

            # Usage and terminal [DONE]
            usage_data = {"choices": [], "usage": {"completion_tokens": 1}}
            yield f"data: {json.dumps(usage_data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(token_stream(), media_type="text/event-stream")

    mock_transport = httpx.ASGITransport(app=mock_backend)
    gateway_app = create_gateway_app(
        backend_url="http://mock-vllm",
        admission_mode=AdmissionMode.PASSTHROUGH,
    )
    gateway_transport = httpx.ASGITransport(app=gateway_app)

    async with httpx.AsyncClient(transport=gateway_transport, base_url="http://test") as client:
        gateway_app.state.client = httpx.AsyncClient(transport=mock_transport, base_url="http://mock-vllm")

        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [], "stream": True},
        )
        assert resp.status_code == 200
        _ = await resp.aread()

        records = gateway_app.state.recorder.records
        assert len(records) > 0
        rec = records[-1]

        # The role-only header arrived immediately (~0ms).
        # The content arrived after 150ms.
        # TTFT MUST be >= 120ms, proving that role-only header was NOT counted as the first token!
        assert rec.ttft_ms is not None
        assert rec.ttft_ms >= 120.0, f"TTFT was {rec.ttft_ms}ms, which incorrectly counted role-only metadata!"
        assert rec.completion_tokens == 1
        assert rec.stream_status == "COMPLETED"
        assert gateway_app.state.controller.active_requests == 0


# ---------------------------------------------------------------------------
# 5. End-to-End Split-Chunk Streaming & Reassembly Test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_end_to_end_streaming_with_split_chunks_and_usage():
    """Verify that split SSE chunks emitted by upstream are transparently streamed and telemetry records usage."""
    mock_backend = FastAPI()

    @mock_backend.post("/v1/chat/completions")
    async def fragmented_stream(req: Request):
        async def split_gen():
            # Emit fragmented bytes across network chunks
            yield b"data: {\"choices\": [{\"delta\": {\"c"
            await asyncio.sleep(0.01)
            yield b"ontent\": \"Hello\"}}]}\n\n"
            await asyncio.sleep(0.01)
            yield b"data: {\"choices\": [{\"delta\": {\"content\": \" World\"}}]}\n\n"
            await asyncio.sleep(0.01)
            # Split usage object across chunks
            yield b"data: {\"choices\": [], \"usage\": {\"completion_"
            await asyncio.sleep(0.01)
            yield b"tokens\": 2}}\n\n"
            # Split [DONE]
            yield b"data: [DO"
            await asyncio.sleep(0.01)
            yield b"NE]\n\n"

        return StreamingResponse(split_gen(), media_type="text/event-stream")

    mock_transport = httpx.ASGITransport(app=mock_backend)
    gateway_app = create_gateway_app(
        backend_url="http://mock-vllm",
        admission_mode=AdmissionMode.BOUNDED_QUEUE,
    )
    gateway_transport = httpx.ASGITransport(app=gateway_app)

    async with httpx.AsyncClient(transport=gateway_transport, base_url="http://test") as client:
        gateway_app.state.client = httpx.AsyncClient(transport=mock_transport, base_url="http://mock-vllm")

        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [], "stream": True},
        )
        assert resp.status_code == 200
        content = await resp.aread()
        assert b"Hello" in content
        assert b"World" in content
        assert b"[DONE]" in content

        records = gateway_app.state.recorder.records
        assert len(records) > 0
        rec = records[-1]
        assert rec.stream_status == "COMPLETED"
        assert rec.completion_tokens == 2
        assert rec.status_code == 200
        assert gateway_app.state.controller.active_requests == 0
