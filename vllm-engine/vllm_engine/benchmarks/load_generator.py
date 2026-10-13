import time
import json
import random
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import requests
import httpx

from vllm_engine.benchmarks.benchmark_serving import RequestBenchmarkResult


def generate_synthetic_prompts(num_prompts: int, prompt_tokens: int = 128) -> List[str]:
    """Generates synthetic prompts with approximately desired token length."""
    base_words = [
        "algorithm", "optimization", "pagedattention", "cuda", "vllm", "kernel",
        "throughput", "latency", "prefill", "decode", "batching"
    ]
    prompts = []
    for i in range(num_prompts):
        words_needed = prompt_tokens
        words = [random.choice(base_words) for _ in range(words_needed)]
        prompt = f"System test query #{i+1}: " + " ".join(words)
        prompts.append(prompt)
    return prompts


def load_dataset_prompts(dataset_path: Path, max_samples: int = 100) -> List[str]:
    """Loads real prompts from an SFT/DPO dataset JSONL file."""
    prompts = []
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                prompt = data.get("prompt") or data.get("instruction")
                if isinstance(data.get("messages"), list) and len(data["messages"]) > 0:
                    prompt = data["messages"][0].get("content", "")
                if prompt:
                    prompts.append(prompt)
                if len(prompts) >= max_samples:
                    break
            except json.JSONDecodeError:
                continue

    return prompts if prompts else generate_synthetic_prompts(max_samples)


def generate_poisson_arrival_schedule(num_requests: int, target_rps: float) -> List[float]:
    """Generates an open-loop arrival schedule (seconds from benchmark t=0) following a Poisson process.
    
    If target_rps <= 0, dispatches all requests simultaneously at t=0 (burst).
    Otherwise, inter-arrival times are drawn from an exponential distribution with mean 1/target_rps.
    """
    if num_requests <= 0:
        return []
    if target_rps <= 0:
        return [0.0] * num_requests

    inter_arrivals = np.random.exponential(scale=1.0 / target_rps, size=num_requests)
    inter_arrivals[0] = 0.0
    arrival_times = np.cumsum(inter_arrivals).tolist()
    return arrival_times


def send_streaming_request(
    api_url: str,
    model_name: str,
    prompt: Union[str, List[Dict[str, str]]],
    max_tokens: int = 128,
    timeout: float = 60.0,
    scheduled_time: float = 0.0,
    submission_lag_ms: float = 0.0,
) -> RequestBenchmarkResult:
    """Sends a single streaming request to standard OpenAI chat completions endpoint."""
    headers = {"Content-Type": "application/json"}
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0
    }

    start_time = time.perf_counter()
    first_event_time: float = 0.0
    event_arrival_times: List[float] = []
    events_received = 0
    prompt_tokens_reported = 0
    completion_tokens_reported = 0
    queue_time_ms = 0.0
    saw_done = False

    raw_prompt_len = sum(len(m.get("content", "")) for m in messages) // 4

    try:
        response = requests.post(api_url, headers=headers, json=payload, stream=True, timeout=timeout)
        queue_header = response.headers.get("X-Aether-Queue-Time-Ms")
        if queue_header:
            try:
                queue_time_ms = float(queue_header)
            except ValueError:
                pass

        if response.status_code == 429:
            return RequestBenchmarkResult(
                prompt_len=raw_prompt_len,
                output_len=0,
                success=False,
                status_code=429,
                is_shed_429=True,
                stream_status="SHED_429",
                queue_time_ms=queue_time_ms,
                error_message=f"HTTP 429 Shed: {response.text[:100]}",
                start_time=start_time,
                scheduled_time=scheduled_time,
                submission_lag_ms=submission_lag_ms,
                e2e_latency=time.perf_counter() - start_time
            )

        if response.status_code != 200:
            return RequestBenchmarkResult(
                prompt_len=raw_prompt_len,
                output_len=0,
                success=False,
                status_code=response.status_code,
                stream_status="ERROR",
                queue_time_ms=queue_time_ms,
                error_message=f"HTTP {response.status_code}: {response.text[:100]}",
                start_time=start_time,
                scheduled_time=scheduled_time,
                submission_lag_ms=submission_lag_ms,
                e2e_latency=time.perf_counter() - start_time
            )

        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8").strip()
            if line_str == "data: [DONE]":
                saw_done = True
                break
            if line_str.startswith("data: "):
                data_str = line_str[6:].strip()
                if data_str == "[DONE]":
                    saw_done = True
                    break
                try:
                    data = json.loads(data_str)
                    if "usage" in data and data["usage"]:
                        prompt_tokens_reported = data["usage"].get("prompt_tokens", 0)
                        completion_tokens_reported = data["usage"].get("completion_tokens", 0)

                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            now = time.perf_counter()
                            if first_event_time == 0.0:
                                first_event_time = now
                            event_arrival_times.append(now)
                            events_received += 1
                except json.JSONDecodeError:
                    continue

        end_time = time.perf_counter()
        e2e_latency = end_time - start_time
        ttft = (first_event_time - start_time) if first_event_time > 0 else e2e_latency

        final_output_len = completion_tokens_reported if completion_tokens_reported > 0 else events_received

        inter_event_latencies: List[float] = []
        inter_event_jitter_ms = 0.0
        if len(event_arrival_times) > 1:
            inter_event_latencies = [
                t2 - t1 for t1, t2 in zip(event_arrival_times[:-1], event_arrival_times[1:])
            ]
            tpot = float(np.mean(inter_event_latencies))
            inter_event_jitter_ms = float(np.std(inter_event_latencies) * 1000.0)
        else:
            tpot = 0.0

        # Rigorous completion detection: stream must have reached terminal [DONE]
        is_truncated = not saw_done
        stream_status = "COMPLETED" if saw_done else "TRUNCATED"
        success = saw_done

        return RequestBenchmarkResult(
            prompt_len=prompt_tokens_reported if prompt_tokens_reported > 0 else raw_prompt_len,
            output_len=final_output_len,
            success=success,
            status_code=200 if saw_done else 502,
            is_truncated=is_truncated,
            stream_status=stream_status,
            queue_time_ms=queue_time_ms,
            start_time=start_time,
            scheduled_time=scheduled_time,
            submission_lag_ms=submission_lag_ms,
            ttft=ttft,
            tpot=tpot,
            e2e_latency=e2e_latency,
            inter_event_latencies=inter_event_latencies,
            inter_event_jitter_ms=inter_event_jitter_ms,
            prompt_tokens=prompt_tokens_reported,
            completion_tokens=final_output_len,
        )

    except Exception as e:
        return RequestBenchmarkResult(
            prompt_len=raw_prompt_len,
            output_len=0,
            success=False,
            error_message=str(e),
            stream_status="ERROR",
            start_time=start_time,
            scheduled_time=scheduled_time,
            submission_lag_ms=submission_lag_ms,
            e2e_latency=time.perf_counter() - start_time
        )


async def async_send_streaming_request(
    client: httpx.AsyncClient,
    api_url: str,
    model_name: str,
    prompt: Union[str, List[Dict[str, str]]],
    max_tokens: int = 128,
    cancel_after_s: Optional[float] = None,
    scheduled_time: float = 0.0,
    submission_lag_ms: float = 0.0,
) -> RequestBenchmarkResult:
    """Asynchronously sends a single streaming request with optional client cancellation and submission lag tracking."""
    headers = {"Content-Type": "application/json"}
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0
    }

    start_time = time.perf_counter()
    first_event_time: float = 0.0
    event_arrival_times: List[float] = []
    events_received = 0
    prompt_tokens_reported = 0
    completion_tokens_reported = 0
    queue_time_ms = 0.0
    saw_done = False

    raw_prompt_len = sum(len(m.get("content", "")) for m in messages) // 4

    try:
        async with client.stream("POST", api_url, headers=headers, json=payload) as response:
            queue_header = response.headers.get("X-Aether-Queue-Time-Ms")
            if queue_header:
                try:
                    queue_time_ms = float(queue_header)
                except ValueError:
                    pass

            if response.status_code == 429:
                body = await response.aread()
                return RequestBenchmarkResult(
                    prompt_len=raw_prompt_len,
                    output_len=0,
                    success=False,
                    status_code=429,
                    is_shed_429=True,
                    stream_status="SHED_429",
                    queue_time_ms=queue_time_ms,
                    error_message=f"HTTP 429 Shed: {body.decode('utf-8')[:100]}",
                    start_time=start_time,
                    scheduled_time=scheduled_time,
                    submission_lag_ms=submission_lag_ms,
                    e2e_latency=time.perf_counter() - start_time
                )

            if response.status_code != 200:
                body = await response.aread()
                return RequestBenchmarkResult(
                    prompt_len=raw_prompt_len,
                    output_len=0,
                    success=False,
                    status_code=response.status_code,
                    stream_status="ERROR",
                    queue_time_ms=queue_time_ms,
                    error_message=f"HTTP {response.status_code}: {body.decode('utf-8')[:100]}",
                    start_time=start_time,
                    scheduled_time=scheduled_time,
                    submission_lag_ms=submission_lag_ms,
                    e2e_latency=time.perf_counter() - start_time
                )

            # Stream chunks
            async for line in response.aiter_lines():
                if cancel_after_s is not None and (time.perf_counter() - start_time) > cancel_after_s:
                    # Client aborts connection
                    return RequestBenchmarkResult(
                        prompt_len=raw_prompt_len,
                        output_len=events_received,
                        success=False,
                        status_code=499,
                        is_cancelled=True,
                        stream_status="CANCELLED",
                        queue_time_ms=queue_time_ms,
                        error_message="Client cancelled stream",
                        start_time=start_time,
                        scheduled_time=scheduled_time,
                        submission_lag_ms=submission_lag_ms,
                        e2e_latency=time.perf_counter() - start_time
                    )

                if not line:
                    continue
                line_str = line.strip()
                if line_str == "data: [DONE]":
                    saw_done = True
                    break
                if line_str.startswith("data: "):
                    data_str = line_str[6:].strip()
                    if data_str == "[DONE]":
                        saw_done = True
                        break
                    try:
                        data = json.loads(data_str)
                        if "usage" in data and data["usage"]:
                            prompt_tokens_reported = data["usage"].get("prompt_tokens", 0)
                            completion_tokens_reported = data["usage"].get("completion_tokens", 0)

                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                now = time.perf_counter()
                                if first_event_time == 0.0:
                                    first_event_time = now
                                event_arrival_times.append(now)
                                events_received += 1
                    except json.JSONDecodeError:
                        continue

        end_time = time.perf_counter()
        e2e_latency = end_time - start_time
        ttft = (first_event_time - start_time) if first_event_time > 0 else e2e_latency

        final_output_len = completion_tokens_reported if completion_tokens_reported > 0 else events_received

        inter_event_latencies: List[float] = []
        inter_event_jitter_ms = 0.0
        if len(event_arrival_times) > 1:
            inter_event_latencies = [
                t2 - t1 for t1, t2 in zip(event_arrival_times[:-1], event_arrival_times[1:])
            ]
            tpot = float(np.mean(inter_event_latencies))
            inter_event_jitter_ms = float(np.std(inter_event_latencies) * 1000.0)
        else:
            tpot = 0.0

        is_truncated = not saw_done
        stream_status = "COMPLETED" if saw_done else "TRUNCATED"
        success = saw_done

        return RequestBenchmarkResult(
            prompt_len=prompt_tokens_reported if prompt_tokens_reported > 0 else raw_prompt_len,
            output_len=final_output_len,
            success=success,
            status_code=200 if saw_done else 502,
            is_truncated=is_truncated,
            stream_status=stream_status,
            queue_time_ms=queue_time_ms,
            start_time=start_time,
            scheduled_time=scheduled_time,
            submission_lag_ms=submission_lag_ms,
            ttft=ttft,
            tpot=tpot,
            e2e_latency=e2e_latency,
            inter_event_latencies=inter_event_latencies,
            inter_event_jitter_ms=inter_event_jitter_ms,
            prompt_tokens=prompt_tokens_reported,
            completion_tokens=final_output_len,
        )

    except asyncio.CancelledError:
        return RequestBenchmarkResult(
            prompt_len=raw_prompt_len,
            output_len=events_received,
            success=False,
            is_cancelled=True,
            stream_status="CANCELLED",
            error_message="Task cancelled",
            start_time=start_time,
            scheduled_time=scheduled_time,
            submission_lag_ms=submission_lag_ms,
            e2e_latency=time.perf_counter() - start_time
        )
    except Exception as e:
        return RequestBenchmarkResult(
            prompt_len=raw_prompt_len,
            output_len=0,
            success=False,
            error_message=str(e),
            stream_status="ERROR",
            start_time=start_time,
            scheduled_time=scheduled_time,
            submission_lag_ms=submission_lag_ms,
            e2e_latency=time.perf_counter() - start_time
        )


async def run_open_loop_poisson_benchmark(
    api_url: str,
    model_name: str,
    prompts: Union[List[str], List[Dict[str, Any]]],
    target_rps: float,
    max_tokens: int = 128,
    timeout_s: float = 60.0,
    client: Optional[httpx.AsyncClient] = None,
) -> Tuple[List[RequestBenchmarkResult], float]:
    """Executes an open-loop benchmark following a Poisson arrival schedule.
    
    Requests are fired at predetermined timestamps regardless of whether earlier
    requests have completed, avoiding closed-loop concurrency throttling.
    Accurately measures submission lag to verify client-side queue health.
    """
    arrival_schedule = generate_poisson_arrival_schedule(len(prompts), target_rps)
    start_bench_time = time.perf_counter()
    results: List[Optional[RequestBenchmarkResult]] = [None] * len(prompts)

    own_client = False
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            limits=httpx.Limits(max_connections=500, max_keepalive_connections=100),
        )
        own_client = True

    try:
        async def _dispatch(idx: int, item: Union[str, Dict[str, Any]], scheduled_target_offset: float):
            target_abs_t = start_bench_time + scheduled_target_offset
            delay = target_abs_t - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)

            actual_dispatch_t = time.perf_counter()
            submission_lag_ms = max(0.0, (actual_dispatch_t - target_abs_t) * 1000.0)

            # Extract prompt and custom max_tokens if item is a structured workload dict
            if isinstance(item, dict):
                req_prompt = item.get("messages") or item.get("prompt")
                req_max_tokens = item.get("max_tokens", max_tokens)
            else:
                req_prompt = item
                req_max_tokens = max_tokens

            res = await async_send_streaming_request(
                client=client,
                api_url=api_url,
                model_name=model_name,
                prompt=req_prompt,
                max_tokens=req_max_tokens,
                scheduled_time=scheduled_target_offset,
                submission_lag_ms=submission_lag_ms,
            )
            results[idx] = res

        tasks = [
            asyncio.create_task(_dispatch(i, prompts[i], arrival_schedule[i]))
            for i in range(len(prompts))
        ]

        await asyncio.gather(*tasks)
        total_duration = time.perf_counter() - start_bench_time
        return [r for r in results if r is not None], total_duration
    finally:
        if own_client:
            await client.aclose()
