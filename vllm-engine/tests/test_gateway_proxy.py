import asyncio
import json
import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import httpx
from starlette.requests import Request

from vllm_engine.gateway.admission import AdmissionMode
from vllm_engine.gateway.proxy import create_gateway_app


def create_mock_vllm_app():
    """Creates a mock vLLM streaming server."""
    mock_app = FastAPI()

    @mock_app.post("/v1/chat/completions")
    async def chat_completions(req: Request):
        body = await req.json()
        is_stream = body.get("stream", False)
        
        if not is_stream:
            return {
                "id": "chatcmpl-mock",
                "choices": [{"message": {"role": "assistant", "content": "Mock answer"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }

        async def token_generator():
            chunks = ["Hello", " from", " mock", " vLLM", "!"]
            for chunk in chunks:
                await asyncio.sleep(0.01)
                data = {
                    "id": "chatcmpl-mock",
                    "choices": [{"delta": {"content": chunk}}],
                }
                yield f"data: {json.dumps(data)}\n\n"
            
            # Send terminal usage
            usage_data = {
                "id": "chatcmpl-mock",
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            yield f"data: {json.dumps(usage_data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(token_generator(), media_type="text/event-stream")

    @mock_app.get("/v1/models")
    async def models():
        return {"data": [{"id": "mock-qwen-model"}]}

    return mock_app


@pytest.mark.asyncio
async def test_gateway_health_and_stats():
    mock_app = create_mock_vllm_app()
    mock_transport = httpx.ASGITransport(app=mock_app)

    gateway_app = create_gateway_app(
        backend_url="http://mock-vllm",
        admission_mode=AdmissionMode.PASSTHROUGH,
    )

    # Use ASGITransport to test gateway app
    gateway_transport = httpx.ASGITransport(app=gateway_app)
    async with httpx.AsyncClient(transport=gateway_transport, base_url="http://test") as client:
        # Patch gateway's internal client to use mock_transport
        gateway_app.state.client = httpx.AsyncClient(transport=mock_transport, base_url="http://mock-vllm")

        resp = await client.get("/health")
        assert resp.status_code == 200
        health_data = resp.json()
        assert health_data["status"] == "ok"
        assert health_data["active_requests"] == 0

        stats_resp = await client.get("/aether/stats")
        assert stats_resp.status_code == 200
        stats = stats_resp.json()
        assert "controller" in stats
        assert "active_requests" in stats["controller"]
        assert "telemetry" in stats


@pytest.mark.asyncio
async def test_gateway_streaming_proxy_and_telemetry():
    mock_app = create_mock_vllm_app()
    mock_transport = httpx.ASGITransport(app=mock_app)

    gateway_app = create_gateway_app(
        backend_url="http://mock-vllm",
        admission_mode=AdmissionMode.BOUNDED_QUEUE,
        max_concurrency=4,
        max_queue=8,
    )

    gateway_transport = httpx.ASGITransport(app=gateway_app)
    async with httpx.AsyncClient(transport=gateway_transport, base_url="http://test") as client:
        gateway_app.state.client = httpx.AsyncClient(transport=mock_transport, base_url="http://mock-vllm")

        req_body = {
            "model": "mock-qwen-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }

        async with client.stream("POST", "/v1/chat/completions", json=req_body) as resp:
            assert resp.status_code == 200
            assert resp.headers["X-Aether-Admission"] == "ADMIT"
            assert "X-Aether-Request-ID" in resp.headers
            assert "X-Aether-Queue-Time-Ms" in resp.headers

            received_chunks = []
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and not line.endswith("[DONE]"):
                    payload = json.loads(line[6:])
                    if payload.get("choices") and payload["choices"][0].get("delta"):
                        received_chunks.append(payload["choices"][0]["delta"].get("content", ""))

        assert "".join(received_chunks) == "Hello from mock vLLM!"
        # After stream ends, slot is released
        assert gateway_app.state.controller.active_requests == 0


@pytest.mark.asyncio
async def test_gateway_429_on_capacity_shed():
    mock_app = FastAPI()

    @mock_app.post("/v1/chat/completions")
    async def slow_completions(req: Request):
        async def slow_stream():
            await asyncio.sleep(0.3)
            yield "data: [DONE]\n\n"
        return StreamingResponse(slow_stream(), media_type="text/event-stream")

    mock_transport = httpx.ASGITransport(app=mock_app)

    # 1 concurrent slot, 0 queue capacity -> 2nd request will immediately shed with 429
    gateway_app = create_gateway_app(
        backend_url="http://mock-vllm",
        admission_mode=AdmissionMode.STATIC_CONCURRENCY,
        max_concurrency=1,
    )

    gateway_transport = httpx.ASGITransport(app=gateway_app)
    async with httpx.AsyncClient(transport=gateway_transport, base_url="http://test") as client:
        gateway_app.state.client = httpx.AsyncClient(transport=mock_transport, base_url="http://mock-vllm")

        req_body = {"model": "test", "messages": [], "stream": True}

        # Start 1st request
        task1 = asyncio.create_task(client.post("/v1/chat/completions", json=req_body))
        await asyncio.sleep(0.02)

        # 2nd request should be shed with HTTP 429
        resp2 = await client.post("/v1/chat/completions", json=req_body)
        assert resp2.status_code == 429
        assert resp2.headers["X-Aether-Admission"] == "SHED"
        assert resp2.headers["X-Aether-Shed-Reason"] == "QueueCapacityExceeded"
        assert resp2.headers["Retry-After"] == "5"

        # Wait for task1 to finish
        resp1 = await task1
        assert resp1.status_code == 200
        assert gateway_app.state.controller.active_requests == 0


@pytest.mark.asyncio
async def test_gateway_non_streaming_and_stream_options_merging():
    received_requests = []
    mock_app = FastAPI()

    @mock_app.post("/v1/chat/completions")
    async def capture_completions(req: Request):
        body = await req.json()
        received_requests.append(body)
        return {"choices": [{"message": {"role": "assistant", "content": "Unary response"}}], "usage": {"completion_tokens": 4}}

    mock_transport = httpx.ASGITransport(app=mock_app)
    gateway_app = create_gateway_app(backend_url="http://mock-vllm", admission_mode=AdmissionMode.PASSTHROUGH)
    gateway_transport = httpx.ASGITransport(app=gateway_app)

    async with httpx.AsyncClient(transport=gateway_transport, base_url="http://test") as client:
        gateway_app.state.client = httpx.AsyncClient(transport=mock_transport, base_url="http://mock-vllm")

        # 1. Non-streaming request: should NOT inject stream_options
        resp_unary = await client.post("/v1/chat/completions", json={"model": "test", "messages": [], "stream": False})
        assert resp_unary.status_code == 200
        assert "stream_options" not in received_requests[0]
        assert gateway_app.state.controller.active_requests == 0

        # 2. Streaming request with existing stream_options: should merge include_usage: True
        resp_merge = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [], "stream": True, "stream_options": {"continuous_usage_stats": True}},
        )
        assert resp_merge.status_code == 200
        assert received_requests[1]["stream_options"]["continuous_usage_stats"] is True
        assert received_requests[1]["stream_options"]["include_usage"] is True


@pytest.mark.asyncio
async def test_gateway_truncated_stream_detection():
    mock_app = FastAPI()

    @mock_app.post("/v1/chat/completions")
    async def truncated_completions(req: Request):
        async def truncated_gen():
            data = {"choices": [{"delta": {"content": "Incomplete sentence"}}]}
            yield f"data: {json.dumps(data)}\n\n"
            # Closes abruptly WITHOUT sending data: [DONE]
        return StreamingResponse(truncated_gen(), media_type="text/event-stream")

    mock_transport = httpx.ASGITransport(app=mock_app)
    gateway_app = create_gateway_app(backend_url="http://mock-vllm", admission_mode=AdmissionMode.PASSTHROUGH)
    gateway_transport = httpx.ASGITransport(app=gateway_app)

    async with httpx.AsyncClient(transport=gateway_transport, base_url="http://test") as client:
        gateway_app.state.client = httpx.AsyncClient(transport=mock_transport, base_url="http://mock-vllm")

        resp = await client.post("/v1/chat/completions", json={"model": "test", "messages": [], "stream": True})
        assert resp.status_code == 200
        _ = await resp.aread()

        # Telemetry should reflect truncated status
        records = gateway_app.state.recorder.records
        assert len(records) > 0
        last_rec = records[-1]
        assert last_rec.stream_status == "TRUNCATED"
        assert gateway_app.state.controller.active_requests == 0

