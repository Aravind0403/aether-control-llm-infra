import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import httpx
import numpy as np

from vllm_engine.benchmarks.load_generator import async_send_streaming_request, load_dataset_prompts


async def test_concurrency_level(
    client: httpx.AsyncClient,
    backend_url: str,
    model_name: str,
    prompts: List[str],
    concurrency: int,
    num_requests: int = 12,
    max_tokens: int = 32,
) -> Dict[str, Any]:
    api_url = f"{backend_url.rstrip('/')}/v1/chat/completions"
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def _worker(prompt: str):
        async with semaphore:
            res = await async_send_streaming_request(
                client=client,
                api_url=api_url,
                model_name=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            results.append(res)

    test_prompts = [prompts[i % len(prompts)] for i in range(num_requests)]
    t0 = time.perf_counter()
    tasks = [asyncio.create_task(_worker(p)) for p in test_prompts]
    await asyncio.gather(*tasks)
    total_time = time.perf_counter() - t0

    successful = [r for r in results if r.success]
    if not successful:
        return {"concurrency": concurrency, "error": "all failed"}

    ttfts = [r.ttft * 1000.0 for r in successful if r.ttft > 0]
    tpots = [r.tpot * 1000.0 for r in successful if r.tpot > 0]
    e2es = [r.e2e_latency for r in successful]
    tokens = sum(r.completion_tokens or r.output_len for r in successful)
    tok_per_sec = tokens / total_time if total_time > 0 else 0

    return {
        "concurrency": concurrency,
        "completed": len(successful),
        "total": len(results),
        "ttft_p50_ms": float(np.percentile(ttfts, 50)) if ttfts else 0.0,
        "ttft_p99_ms": float(np.percentile(ttfts, 99)) if ttfts else 0.0,
        "tpot_p50_ms": float(np.percentile(tpots, 50)) if tpots else 0.0,
        "tpot_p99_ms": float(np.percentile(tpots, 99)) if tpots else 0.0,
        "e2e_p99_s": float(np.percentile(e2es, 99)) if e2es else 0.0,
        "tok_s": float(tok_per_sec),
        "rps": float(len(successful) / total_time),
    }


async def main():
    backend = "http://localhost:8000"
    model = "Qwen/Qwen2.5-1.5B-Instruct"
    workload_file = Path("benchmarks/workloads/lmsys_smoke_50.jsonl")
    prompts = load_dataset_prompts(workload_file, max_samples=50)

    print(f"⚡ Calibrating Real vLLM Capacity on RTX 3090 ({model})...")
    print("-" * 80)
    print(f"{'Concurrency':<12} | {'TTFT P50':<10} | {'TTFT P99':<10} | {'TPOT P50':<10} | {'E2E P99':<10} | {'Tok/s':<10}")
    print("-" * 80)

    concurrency_levels = [1, 2, 4, 8, 12, 16]
    records = []

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=5.0),
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
    ) as client:
        # Warmup
        await test_concurrency_level(client, backend, model, prompts, concurrency=1, num_requests=2, max_tokens=16)

        for c in concurrency_levels:
            rec = await test_concurrency_level(client, backend, model, prompts, concurrency=c, num_requests=12, max_tokens=32)
            records.append(rec)
            print(f"{rec['concurrency']:<12} | {rec['ttft_p50_ms']:<8.1f}ms | {rec['ttft_p99_ms']:<8.1f}ms | {rec['tpot_p50_ms']:<8.1f}ms | {rec['e2e_p99_s']:<8.2f}s | {rec['tok_s']:<8.1f}")
            await asyncio.sleep(0.5)

    print("-" * 80)
    with open("benchmarks/reports/capacity_calibration.json", "w") as f:
        json.dump(records, f, indent=2)
    print("Saved calibration data to benchmarks/reports/capacity_calibration.json")


if __name__ == "__main__":
    asyncio.run(main())
