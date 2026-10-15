import asyncio
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


class SSEEventParser:
    """Delimited Server-Sent Events (SSE) parser that handles arbitrary byte chunk boundaries.
    
    Buffers incoming bytes until full double-newline boundaries are encountered,
    parsing data fields, usage objects, and the terminal [DONE] marker.
    """
    def __init__(self):
        self._buffer = ""

    def feed(self, chunk: bytes) -> list[tuple[str, Optional[dict]]]:
        """Feeds raw byte chunk and yields (raw_payload_text, parsed_json_or_none) for complete events."""
        text = chunk.decode("utf-8", errors="replace")
        self._buffer += text.replace("\r\n", "\n")

        events = []
        while "\n\n" in self._buffer:
            event_block, self._buffer = self._buffer.split("\n\n", 1)
            event_block = event_block.strip()
            if not event_block:
                continue

            data_lines = []
            for line in event_block.split("\n"):
                line_s = line.strip()
                if line_s.startswith("data:"):
                    data_lines.append(line_s[5:].strip())

            if not data_lines:
                continue

            payload_text = "\n".join(data_lines)
            if payload_text == "[DONE]":
                events.append(("[DONE]", None))
            else:
                try:
                    payload = json.loads(payload_text)
                    events.append((payload_text, payload))
                except Exception:
                    events.append((payload_text, None))

        return events


def _is_content_event(payload: Optional[dict]) -> bool:
    """Determines if an SSE chunk payload contains actual generated text.
    
    Returns True only when choices[0].delta.content contains non-empty text.
    Returns False for role-only headers (e.g. {"role": "assistant"}), empty strings, or usage-only events.
    """
    if not payload or not isinstance(payload, dict):
        return False
    choices = payload.get("choices")
    if not choices or not isinstance(choices, list) or len(choices) == 0:
        return False
    delta = choices[0].get("delta")
    if not delta or not isinstance(delta, dict):
        return False
    content = delta.get("content")
    return bool(content)


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

        # Slot is now held. Must guarantee release under any error or cancellation.
        slot_held = True
        try:
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
                resp = await _handle_streaming_request(
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
                # Slot release responsibility is handed over to the stream generator
                slot_held = False
                return resp
            else:
                resp = await _handle_unary_request(
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
                slot_held = False
                return resp

        except BaseException as e:
            if slot_held:
                controller.release_slot()
                slot_held = False
            if isinstance(e, HTTPException):
                raise
            if isinstance(e, asyncio.CancelledError):
                rec = GatewayRequestRecord(
                    request_id=req_id,
                    arrival_time=arrival_time,
                    queue_wait_ms=queue_wait_ms,
                    status_code=499,
                    admitted=True,
                    client_disconnected=True,
                    stream_status="CANCELLED",
                    e2e_latency_ms=(time.perf_counter() - arrival_time) * 1000.0,
                )
                recorder.record(rec)
                raise
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
            return JSONResponse(
                status_code=502,
                content={"error": {"message": f"Gateway request processing failure: {e}", "type": "gateway_error"}},
            )

    return app


async def _handle_unary_request(
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
    """Handles non-streaming forward with guaranteed slot release and client disconnect detection."""
    disconnect_event = asyncio.Event()

    async def poll_disconnect():
        while not disconnect_event.is_set():
            if await request.is_disconnected():
                disconnect_event.set()
                break
            await asyncio.sleep(0.05)

    disconnect_task = asyncio.create_task(poll_disconnect())

    try:
        post_task = asyncio.create_task(
            client.post("/v1/chat/completions", headers=headers, content=body)
        )
        wait_task = asyncio.create_task(disconnect_event.wait())

        done, pending = await asyncio.wait(
            [post_task, wait_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if wait_task in done:
            post_task.cancel()
            rec = GatewayRequestRecord(
                request_id=req_id,
                arrival_time=arrival_time,
                queue_wait_ms=queue_wait_ms,
                status_code=499,
                admitted=True,
                client_disconnected=True,
                stream_status="CANCELLED",
                e2e_latency_ms=(time.perf_counter() - arrival_time) * 1000.0,
            )
            recorder.record(rec)
            return JSONResponse(
                status_code=499,
                content={"error": {"message": "Client closed connection", "type": "client_closed"}},
                headers=initial_headers,
            )

        wait_task.cancel()
        upstream_resp = post_task.result()
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
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"Upstream communication failure: {e}", "type": "gateway_error"}},
            headers=initial_headers,
        )
    finally:
        disconnect_event.set()
        disconnect_task.cancel()
        try:
            await disconnect_task
        except asyncio.CancelledError:
            pass
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
    try:
        upstream_resp = await client.send(req, stream=True)
    except Exception as e:
        controller.release_slot()
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
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"Upstream communication failure: {e}", "type": "upstream_connect_error"}},
            headers=initial_headers,
        )

    # If backend returned an error (e.g. 400 Bad Request, 500 Internal Error)
    if upstream_resp.status_code != 200:
        try:
            error_content = await upstream_resp.aread()
        finally:
            await upstream_resp.aclose()
            controller.release_slot()

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
        sse_parser = SSEEventParser()

        disconnect_event = asyncio.Event()

        async def poll_disconnect():
            while not disconnect_event.is_set():
                if await request.is_disconnected():
                    disconnect_event.set()
                    break
                await asyncio.sleep(0.05)

        disconnect_task = asyncio.create_task(poll_disconnect())

        try:
            chunk_iterator = upstream_resp.aiter_raw().__aiter__()
            while True:
                chunk_task = asyncio.create_task(chunk_iterator.__anext__())
                wait_task = asyncio.create_task(disconnect_event.wait())

                done, pending = await asyncio.wait(
                    [chunk_task, wait_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if wait_task in done:
                    client_disconnected = True
                    chunk_task.cancel()
                    logger.info("Client disconnected; aborting upstream stream", extra={"request_id": req_id})
                    break

                wait_task.cancel()

                try:
                    chunk = chunk_task.result()
                except StopAsyncIteration:
                    break

                now = time.perf_counter()

                # Process chunk through incremental SSE parser
                events = sse_parser.feed(chunk)
                for raw_text, payload in events:
                    if raw_text == "[DONE]":
                        saw_done = True
                    elif payload:
                        usage = payload.get("usage")
                        if usage and "completion_tokens" in usage:
                            completion_tokens = usage["completion_tokens"]
                        if _is_content_event(payload):
                            if first_token_time is None:
                                first_token_time = now
                            elif last_event_time is not None:
                                inter_event_latencies.append((now - last_event_time) * 1000.0)
                            last_event_time = now

                yield chunk

        finally:
            disconnect_event.set()
            disconnect_task.cancel()
            try:
                await disconnect_task
            except asyncio.CancelledError:
                pass

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
