import time
import asyncio
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import requests


@dataclass
class RequestBenchmarkResult:
    """Individual HTTP inference request latency breakdown."""
    prompt_len: int
    output_len: int
    success: bool
    error_message: str = ""
    start_time: float = 0.0          # Actual client dispatch timestamp (perf_counter)
    scheduled_time: float = 0.0      # Planned arrival schedule timestamp
    submission_lag_ms: float = 0.0   # (start_time - scheduled_time) * 1000.0 (client pool queue lag)
    ttft: float = 0.0                # Time To First Content Event (seconds)
    tpot: float = 0.0                # Mean Time Per Content Event (seconds)
    e2e_latency: float = 0.0         # End to End Latency (seconds)
    status_code: int = 200
    is_shed_429: bool = False
    is_cancelled: bool = False
    is_truncated: bool = False
    stream_status: str = "COMPLETED" # COMPLETED, TRUNCATED, CANCELLED, SHED_429, ERROR, UNFINISHED
    queue_time_ms: float = 0.0
    inter_event_latencies: List[float] = field(default_factory=list) # Inter-content-event timing (seconds)
    inter_event_jitter_ms: float = 0.0 # Std dev of inter-content-event intervals in ms
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class BenchmarkSummary:
    """Aggregated SLA performance report across all test requests."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration_s: float
    request_throughput_rps: float
    output_token_throughput_tps: float
    
    # TTFT Percentiles (Prefill Latency / First Content Event)
    ttft_p50: float
    ttft_p90: float
    ttft_p95: float
    ttft_p99: float
    
    # TPOT Percentiles (Decode Latency / Content Event Interval)
    tpot_p50: float
    tpot_p90: float
    tpot_p95: float
    tpot_p99: float

    # E2E Percentiles
    e2e_p50: float
    e2e_p90: float
    e2e_p95: float
    e2e_p99: float

    # Detailed Flow Breakdown
    offered_requests: int = 0
    admitted_requests: int = 0
    completed_requests: int = 0
    truncated_requests: int = 0
    shed_requests_429: int = 0
    cancelled_requests: int = 0
    errored_requests: int = 0
    unfinished_requests: int = 0

    # Submission Lag (Open-Loop Client Health)
    submission_lag_p50_ms: float = 0.0
    submission_lag_p99_ms: float = 0.0

    # Inter-content-event timing jitter & gateway queue stats
    inter_event_jitter_p50_ms: float = 0.0
    inter_event_jitter_p99_ms: float = 0.0
    queue_time_p50_ms: float = 0.0
    queue_time_p99_ms: float = 0.0

    # Latency objective attainment (SLO target, e.g. E2E <= 2.0s)
    slo_target_s: float = 2.0
    slo_attainment_pct: float = 0.0

    # Explicit Goodput and Dual Latency Objectives
    ttft_target_s: float = 0.5
    e2e_target_s: float = 1.5
    arrival_window_s: float = 0.0
    goodput_rps: float = 0.0
    goodput_attainment_pct: float = 0.0

    # Backward compatibility fields
    token_jitter_p50_ms: Optional[float] = None
    token_jitter_p99_ms: Optional[float] = None

    def __post_init__(self):
        if self.token_jitter_p50_ms is not None and self.inter_event_jitter_p50_ms == 0.0:
            self.inter_event_jitter_p50_ms = self.token_jitter_p50_ms
        if self.token_jitter_p99_ms is not None and self.inter_event_jitter_p99_ms == 0.0:
            self.inter_event_jitter_p99_ms = self.token_jitter_p99_ms


class ServingBenchmarkRunner:
    """Standard OpenAI-compatible API load testing & SLA benchmark runner."""

    def __init__(
        self,
        api_url: str,
        model_name: str,
        slo_target_s: float = 2.0,
        ttft_target_s: float = 0.5,
        e2e_target_s: float = 1.5,
    ):
        self.api_url = api_url.rstrip("/") + "/v1/chat/completions"
        self.model_name = model_name
        self.slo_target_s = slo_target_s
        self.ttft_target_s = ttft_target_s
        self.e2e_target_s = e2e_target_s

    def calculate_percentiles(self, values: List[float]) -> Dict[str, float]:
        """Utility to calculate P50, P90, P95, P99 percentiles."""
        if not values:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
        arr = np.array(values)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    def summarize_results(
        self,
        results: List[RequestBenchmarkResult],
        total_duration_s: float,
        measurement_window_s: Optional[float] = None,
        arrival_window_s: Optional[float] = None,
    ) -> BenchmarkSummary:
        """Aggregates raw request latency metrics into a structured BenchmarkSummary."""
        completed = [r for r in results if r.stream_status == "COMPLETED" and r.success]
        truncated = [r for r in results if r.is_truncated or r.stream_status == "TRUNCATED"]
        shed_429 = [r for r in results if r.is_shed_429 or r.status_code == 429]
        cancelled = [r for r in results if r.is_cancelled or r.stream_status == "CANCELLED"]
        unfinished = [r for r in results if r.stream_status == "UNFINISHED"]
        errored = [
            r for r in results
            if not r.success and r not in completed and r not in truncated and r not in shed_429 and r not in cancelled and r not in unfinished
        ]

        offered_count = len(results)
        admitted_count = offered_count - len(shed_429)
        failed_count = len(truncated) + len(errored) + len(cancelled) + len(unfinished)

        window_duration = measurement_window_s if measurement_window_s and measurement_window_s > 0 else total_duration_s

        # Latency statistics on completed requests
        ttfts = [r.ttft for r in completed if r.ttft > 0]
        tpots = [r.tpot for r in completed if r.tpot > 0]
        e2es = [r.e2e_latency for r in completed]
        jitters = [r.inter_event_jitter_ms for r in completed if r.inter_event_jitter_ms > 0]
        queue_times = [r.queue_time_ms for r in results if r.queue_time_ms > 0]
        submission_lags = [r.submission_lag_ms for r in results if r.submission_lag_ms >= 0]

        ttft_stats = self.calculate_percentiles(ttfts)
        tpot_stats = self.calculate_percentiles(tpots)
        e2e_stats = self.calculate_percentiles(e2es)
        jitter_stats = self.calculate_percentiles(jitters)
        queue_stats = self.calculate_percentiles(queue_times)
        submission_lag_stats = self.calculate_percentiles(submission_lags)

        total_tokens = sum(r.output_len for r in completed)
        rps = len(completed) / window_duration if window_duration > 0 else 0.0
        tps = total_tokens / window_duration if window_duration > 0 else 0.0

        # SLO attainment: fraction of all offered requests completed within slo_target_s
        slo_met_count = sum(1 for r in completed if r.e2e_latency <= self.slo_target_s)
        slo_attainment = (slo_met_count / offered_count * 100.0) if offered_count > 0 else 0.0

        # Goodput: Requests completing successfully within ALL declared latency targets
        # Divided strictly by the duration of the arrival window
        compliant = [
            r for r in completed
            if (r.ttft <= self.ttft_target_s if self.ttft_target_s > 0 else True)
            and (r.e2e_latency <= self.e2e_target_s if self.e2e_target_s > 0 else True)
        ]
        if arrival_window_s is not None and arrival_window_s > 0:
            win_arrival = arrival_window_s
        else:
            sched_times = [r.scheduled_time for r in results if r.scheduled_time is not None]
            if len(sched_times) > 1 and max(sched_times) > min(sched_times):
                win_arrival = max(sched_times) - min(sched_times)
            else:
                win_arrival = window_duration

        goodput_rps = len(compliant) / win_arrival if win_arrival > 0 else 0.0
        goodput_attainment = (len(compliant) / offered_count * 100.0) if offered_count > 0 else 0.0

        return BenchmarkSummary(
            total_requests=offered_count,
            successful_requests=len(completed),
            failed_requests=failed_count,
            total_duration_s=total_duration_s,
            request_throughput_rps=rps,
            output_token_throughput_tps=tps,
            ttft_p50=ttft_stats["p50"],
            ttft_p90=ttft_stats["p90"],
            ttft_p95=ttft_stats["p95"],
            ttft_p99=ttft_stats["p99"],
            tpot_p50=tpot_stats["p50"],
            tpot_p90=tpot_stats["p90"],
            tpot_p95=tpot_stats["p95"],
            tpot_p99=tpot_stats["p99"],
            e2e_p50=e2e_stats["p50"],
            e2e_p90=e2e_stats["p90"],
            e2e_p95=e2e_stats["p95"],
            e2e_p99=e2e_stats["p99"],
            offered_requests=offered_count,
            admitted_requests=admitted_count,
            completed_requests=len(completed),
            truncated_requests=len(truncated),
            shed_requests_429=len(shed_429),
            cancelled_requests=len(cancelled),
            errored_requests=len(errored),
            unfinished_requests=len(unfinished),
            submission_lag_p50_ms=submission_lag_stats["p50"],
            submission_lag_p99_ms=submission_lag_stats["p99"],
            inter_event_jitter_p50_ms=jitter_stats["p50"],
            inter_event_jitter_p99_ms=jitter_stats["p99"],
            queue_time_p50_ms=queue_stats["p50"],
            queue_time_p99_ms=queue_stats["p99"],
            slo_target_s=self.slo_target_s,
            slo_attainment_pct=slo_attainment,
            ttft_target_s=self.ttft_target_s,
            e2e_target_s=self.e2e_target_s,
            arrival_window_s=win_arrival,
            goodput_rps=goodput_rps,
            goodput_attainment_pct=goodput_attainment,
        )
