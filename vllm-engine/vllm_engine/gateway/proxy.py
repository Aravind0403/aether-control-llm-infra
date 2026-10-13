import time
import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict, Any

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx

from vllm_engine.gateway.admission import (
    BoundedQueueController,
    AdmissionMode,
    QueueCapacityExceededError,
    QueueDeadlineExpiredError,
)
from vllm_engine.gateway.telemetry import (
    GatewayTelemetryRecorder,
    GatewayRequestRecord,
)

logger = logging.getLogger("aether.gateway")


def create_gateway_app(
    backend_url: str = "http://localhost:8000",
    admission_mode: AdmissionMode = AdmissionMode.BOUNDED_QUEUE,
    max_concurrency: int = 8,
    max_queue: int = 32,
    deadline_ms: float = 500.0,
    retry_after_s: int = 5,
    timeout_s: float = 120.0,
) -> FastAPI:
    """Factory creating an Aether Admission Gateway FastAPI application."""

    controller = BoundedQueueController(
        mode=admission_mode,
        max_concurrency=max_concurrency,
        max_queue=max_queue,
        deadline_ms=deadline_ms,
        retry_after_s=retry_after_s,
    )
    recorder = GatewayTelemetryRecorder()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Application-scoped reusable HTTPX client with connection pooling
        if not hasattr(app.state, "client") or app.state.client is None:
            app.state.client = httpx.AsyncClient(
                base_url=backend_url.rstrip("/"),
                timeout=httpx.Timeout(timeout_s, connect=5.0),
                limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
            )
        yield
        if hasattr(app.state, "client") and app.state.client is not None:
            await app.state.client.aclose()

    app = FastAPI(title="Aether LLM Admission Gateway", lifespan=lifespan)
    app.state.controller = controller
    app.state.recorder = recorder

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "active_requests": controller.active_requests,
            "queued_requests": controller.queued_requests,
            "mode": controller.mode.value,
        }

    @app.get("/aether/stats")
    async def stats():
        return {
            "controller": {
                "active_requests": controller.active_requests,
                "queued_requests": controller.queued_requests,
                "total_offered": controller.total_offered,
                "total_admitted": controller.total_admitted,
                "total_shed_capacity": controller.total_shed_capacity,
                "total_shed_deadline": controller.total_shed_deadline,
                "mode": controller.mode.value,
            },
            "telemetry": recorder.get_summary(),
        }

    @app.get("/v1/models")
    async def get_models():
        client: httpx.AsyncClient = app.state.client
        try:
            resp = await client.get("/v1/models")
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Backend unreachable: {e}")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        arrival_time = time.perf_counter()
        req_id = recorder.generate_request_id()

        # 1. Evaluate Admission & Bounded Queue
        try:
            queue_wait_ms = await controller.acquire_slot()
        except QueueCapacityExceededError as e:
            rec = GatewayRequestRecord(
                request_id=req_id,
                arrival_time=arrival_time,
                queue_wait_ms=0.0,
                status_code=429,
                admitted=False,
                shed_reason="QueueCapacityExceeded",
            )
            recorder.record(rec)
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": "Gateway queue capacity exceeded; shed under overload",
                        "type": "rate_limit_exceeded",
                        "code": "QueueCapacityExceeded",
                    }
                },
                headers={
                    "X-Aether-Request-ID": req_id,
                    "X-Aether-Admission": "SHED",
                    "X-Aether-Shed-Reason": "QueueCapacityExceeded",
                    "Retry-After": str(e.retry_after_s),
                },
            )
        except QueueDeadlineExpiredError as e:
            rec = GatewayRequestRecord(
                request_id=req_id,
                arrival_time=arrival_time,
                queue_wait_ms=e.wait_ms,
                status_code=429,
                admitted=False,
                shed_reason="QueueDeadlineExpired",
            )
            recorder.record(rec)
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": f"Gateway queue deadline expired after {e.wait_ms:.1f}ms",
                        "type": "rate_limit_exceeded",
                        "code": "QueueDeadlineExpired",
                    }
                },
                headers={
                    "X-Aether-Request-ID": req_id,
                    "X-Aether-Queue-Time-Ms": f"{e.wait_ms:.2f}",
                    "X-Aether-Admission": "SHED",
                    "X-Aether-Shed-Reason": "QueueDeadlineExpired",
                    "Retry-After": str(e.retry_after_s),
                },
            )

        # 2. Slot acquired - Prepare upstream request
        body = await request.body()
        is_streaming = False
        try:
            parsed_json = json.loads(body)
            is_streaming = bool(parsed_json.get("stream", False))
            if is_streaming:
                # Merge include_usage into existing stream_options, preserving any other fields
                stream_opts = parsed_json.get("stream_options")
                if isinstance(stream_opts, dict):
                    stream_opts["include_usage"] = True
                else:
                    parsed_json["stream_options"] = {"include_usage": True}
                body = json.dumps(parsed_json).encode("utf-8")
            else:
                # stream: false explicitly does not receive stream_options
                pass
        except Exception:
            pass

        client: httpx.AsyncClient = app.state.client
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)
        headers["X-Aether-Request-ID"] = req_id

        initial_response_headers = {
            "X-Aether-Request-ID": req_id,
            "X-Aether-Queue-Time-Ms": f"{queue_wait_ms:.2f}",
            "X-Aether-Admission": "ADMIT",
        }

        # 3. Handle Streaming vs Non-Streaming
        if is_streaming:
            return await _handle_streaming_request(
                client=client,
                controller=controller,
                recorder=recorder,
                request=request,
                req_id=req_id,
                arrival_time=arrival_time,
                queue_wait_ms=queue_wait_ms,
                headers=headers,
                body=body,
                initial_headers=initial_response_headers,
            )
        else:
            return await _handle_unary_request(
                client=client,
                controller=controller,
                recorder=recorder,
                req_id=req_id,
                arrival_time=arrival_time,
                queue_wait_ms=queue_wait_ms,
                headers=headers,
                body=body,
                initial_headers=initial_response_headers,
            )

    return app


async def _handle_unary_request(
    client: httpx.AsyncClient,
    controller: BoundedQueueController,
    recorder: GatewayTelemetryRecorder,
    req_id: str,
    arrival_time: float,
    queue_wait_ms: float,
    headers: dict,
    body: bytes,
    initial_headers: dict,
) -> Response:
    """Handles non-streaming forward with guaranteed slot release."""
    try:
        upstream_resp = await client.post("/v1/chat/completions", headers=headers, content=body)
        duration_ms = (time.perf_counter() - arrival_time) * 1000.0

        tokens = 0
        try:
            resp_data = upstream_resp.json()
            tokens = resp_data.get("usage", {}).get("completion_tokens", 0)
        except Exception:
            pass

        rec = GatewayRequestRecord(
            request_id=req_id,
            arrival_time=arrival_time,
            queue_wait_ms=queue_wait_ms,
            ttft_ms=duration_ms,
            e2e_latency_ms=duration_ms,
            status_code=upstream_resp.status_code,
            admitted=True,
            completion_tokens=tokens,
        )
        recorder.record(rec)

        response_headers = dict(upstream_resp.headers)
        response_headers.update(initial_headers)

        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=response_headers,
            media_type=upstream_resp.headers.get("content-type", "application/json"),
        )
    except Exception as e:
        rec = GatewayRequestRecord(
            request_id=req_id,
            arrival_time=arrival_time,
            queue_wait_ms=queue_wait_ms,
            status_code=502,
            admitted=True,
            error=str(e),
            e2e_latency_ms=(time.perf_counter() - arrival_time) * 1000.0,
        )
        recorder.record(rec)
        raise HTTPException(status_code=502, detail=f"Upstream communication failure: {e}")
    finally:
        controller.release_slot()


async def _handle_streaming_request(
    client: httpx.AsyncClient,
    controller: BoundedQueueController,
    recorder: GatewayTelemetryRecorder,
    request: Request,
    req_id: str,
    arrival_time: float,
    queue_wait_ms: float,
    headers: dict,
    body: bytes,
    initial_headers: dict,
) -> Response:
    """Initiates upstream stream and bridges it to downstream StreamingResponse."""
    req = client.build_request("POST", "/v1/chat/completions", headers=headers, content=body)
    upstream_resp = await client.send(req, stream=True)

    # If backend returned an error (e.g. 400 Bad Request, 500 Internal Error)
    if upstream_resp.status_code != 200:
        controller.release_slot()
        error_content = await upstream_resp.aread()
        await upstream_resp.aclose()
        rec = GatewayRequestRecord(
            request_id=req_id,
            arrival_time=arrival_time,
            queue_wait_ms=queue_wait_ms,
            status_code=upstream_resp.status_code,
            admitted=True,
            error=error_content.decode("utf-8", errors="replace"),
            e2e_latency_ms=(time.perf_counter() - arrival_time) * 1000.0,
        )
        recorder.record(rec)
        return Response(
            content=error_content,
            status_code=upstream_resp.status_code,
            headers=initial_headers,
            media_type=upstream_resp.headers.get("content-type", "application/json"),
        )

    # Downstream streaming generator that retains the upstream stream lifetime
    async def response_streamer() -> AsyncGenerator[bytes, None]:
        first_token_time: Optional[float] = None
        completion_tokens = 0
        client_disconnected = False
        saw_done = False
        inter_event_latencies = []
        last_event_time: Optional[float] = None

        try:
            async for chunk in upstream_resp.aiter_raw():
                # Client disconnect detection
                if await request.is_disconnected():
                    client_disconnected = True
                    logger.info("Client disconnected; aborting upstream stream", extra={"request_id": req_id})
                    break

                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                elif last_event_time is not None:
                    inter_event_latencies.append((now - last_event_time) * 1000.0)
                last_event_time = now

                # Extract completion_tokens from usage chunk and check for terminal [DONE]
                chunk_str = chunk.decode("utf-8", errors="ignore")
                for line in chunk_str.splitlines():
                    line_s = line.strip()
                    if line_s == "data: [DONE]":
                        saw_done = True
                    elif line_s.startswith("data: "):
                        try:
                            data = json.loads(line_s[6:].strip())
                            usage = data.get("usage")
                            if usage and "completion_tokens" in usage:
                                completion_tokens = usage["completion_tokens"]
                        except Exception:
                            pass

                yield chunk

        finally:
            await upstream_resp.aclose()
            controller.release_slot()

            end_time = time.perf_counter()
            ttft_ms = ((first_token_time - arrival_time) * 1000.0) if first_token_time else None
            e2e_ms = (end_time - arrival_time) * 1000.0

            # Determine exact completion status
            if client_disconnected:
                stream_status = "CANCELLED"
                status_code = 499
            elif saw_done:
                stream_status = "COMPLETED"
                status_code = 200
            else:
                stream_status = "TRUNCATED"
                status_code = 502

            rec = GatewayRequestRecord(
                request_id=req_id,
                arrival_time=arrival_time,
                queue_wait_ms=queue_wait_ms,
                ttft_ms=ttft_ms,
                e2e_latency_ms=e2e_ms,
                status_code=status_code,
                admitted=True,
                completion_tokens=completion_tokens,
                client_disconnected=client_disconnected,
                stream_status=stream_status,
                inter_event_latencies_ms=inter_event_latencies,
            )
            recorder.record(rec)

    response_headers = dict(initial_headers)
    response_headers["content-type"] = "text/event-stream"
    response_headers["cache-control"] = "no-cache"

    return StreamingResponse(
        response_streamer(),
        status_code=200,
        headers=response_headers,
        media_type="text/event-stream",
    )
