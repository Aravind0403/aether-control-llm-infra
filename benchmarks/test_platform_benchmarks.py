"""Pytest Benchmark Invariant Assertions.

Validates that all SLAs and production invariants across TrainSight, RLHF, and vLLM
remain within their strict tolerances on every CI run.
"""
import pytest
from benchmarks.run_all_benchmarks import PlatformBenchmarkSuite


@pytest.fixture(scope="module")
def benchmark_results():
    suite = PlatformBenchmarkSuite(mode="analytical")
    return suite.run_all()


def test_stage_1_trainsight_invariants(benchmark_results):
    s1 = benchmark_results["stages"]["stage_1_trainsight"]
    # Meta probe under 50ms (typically < 10ms)
    assert s1["meta_probe"]["probe_latency_ms"] < 50.0
    # Invariant hashing under 10ms
    assert s1["signature_cache"]["hash_latency_ms"] < 10.0
    # Bin-packing throughput boost > 20%
    assert s1["bin_packing"]["throughput_boost_pct"] > 20.0


def test_stage_2_rlhf_invariants(benchmark_results):
    s2 = benchmark_results["stages"]["stage_2_rlhf"]
    # Verifier contract boot latency < 10ms
    assert s2["verifier_contract"]["latency_ms"] < 10.0
    assert s2["verifier_contract"]["pairs_tested"] >= 15
    # Radix prompt KV redundancy eliminated == 87.5% for G=8
    assert s2["radix_kv_deduplication"]["prompt_redundancy_eliminated_pct"] == 87.5
    # Rollout VRAM peak under 30 GB (leaving > 50GB headroom on 80GB GPU)
    assert s2["vram_time_multiplexing"]["rollout_peak_vram_gb"] < 30.0
    assert s2["vram_time_multiplexing"]["rollout_headroom_pct"] > 60.0
    # Train backward peak under 60 GB
    assert s2["vram_time_multiplexing"]["train_peak_vram_gb"] < 60.0
    # Bayesian penalty suppresses jury hacking
    assert s2["consensus_jury"]["is_hacking_suppressed"] is True
    # Reference fluency barrier penalizes high PPL
    assert s2["fluency_barrier"]["fluency_penalty"] > 0.0


def test_stage_3_vllm_invariants(benchmark_results):
    s3 = benchmark_results["stages"]["stage_3_vllm"]
    # CoDel load shedder returns HTTP 429 when latency > 500ms
    assert s3["codel_admission"]["nominal_ttft_status"] == 200
    assert "429" in s3["codel_admission"]["saturated_action"]
    # Model cascade router sends simple queries to 7B fast lane
    assert "7b" in s3["model_cascade"]["simple_query_target"]
    assert s3["model_cascade"]["simple_query_latency_ms"] < 100.0
    # Tenant blocks are capped at quota (150 blocks)
    assert s3["two_tier_radix_security"]["attacker_allocated_blocks"] == 150
    # Residual watchdog flips readiness probe to 503 upon leak
    assert s3["idle_residual_watchdog"]["clean_probe_status"] == 200
    assert s3["idle_residual_watchdog"]["leak_probe_status"] == 503
    # Semantic complexity oracle runs in < 2ms
    assert s3["semantic_oracle"]["oracle_latency_ms"] < 2.0
    # Spot handoff beats the 30s deadline with zero dropped requests
    assert s3["spot_handoff_timeline"]["replacement_ready_s"] < 30.0
    assert s3["spot_handoff_timeline"]["dropped_requests"] == 0


def test_stage_4_k8s_invariants(benchmark_results):
    s4 = benchmark_results["stages"]["stage_4_k8s"]
    # Driver mutex reduction >= 80%
    assert s4["dcgm_mutex_contention"]["driver_lock_reduction_pct"] >= 80.0
    assert s4["dcgm_mutex_contention"]["tiered_5000ms_p99_ttft_ms"] == 45.0
    # Shared memory read under 0.5ms (typically <0.1ms)
    assert s4["shared_memory_cache"]["read_latency_ms"] < 0.5
    assert s4["shared_memory_cache"]["storage_medium"] == "/dev/shm"
    assert s4["shared_memory_cache"]["is_driver_mutex_acquired"] is False
    # Kernel XID trap under 5.0ms with node quarantine taint applied
    assert s4["kernel_xid_trap"]["quarantine_latency_ms"] < 5.0
    assert s4["kernel_xid_trap"]["applied_node_taint"] == "node.kubernetes.io/unschedulable:NoSchedule"
    assert s4["kernel_xid_trap"]["zero_polling_overhead"] is True

