import pytest
from vllm_engine.admission.codel_limiter import CoDelLoadShedder
from vllm_engine.admission.model_cascade import ModelCascadeRouter
from vllm_engine.radix.two_tier_cache import TwoTierRadixCacheManager
from vllm_engine.radix.cache_primer import ProgressiveCachePrimer
from vllm_engine.watchdog.block_leak_watchdog import IdleResidualBlockWatchdog
from vllm_engine.routing.semantic_oracle import SemanticComplexityOracle
from vllm_engine.routing.spot_handoff import SpotHandoffSimulator


def test_codel_load_shedder():
    shedder = CoDelLoadShedder(target_delay_ms=500.0, retry_after_seconds=15)

    # 1. Under target
    dec_pass = shedder.evaluate_request(280.0)
    assert dec_pass.allowed
    assert dec_pass.status_code == 200
    assert dec_pass.route_target == "primary_70b"

    # 2. Over target -> Shed
    dec_shed = shedder.evaluate_request(650.0)
    assert not dec_shed.allowed
    assert dec_shed.status_code == 429
    assert dec_shed.retry_after_seconds == 15
    assert dec_shed.rate_limit_reason == "QueueSaturationDelayExceeded"


def test_model_cascade_router():
    router = ModelCascadeRouter(length_threshold_tokens=64)

    # Simple short query during saturation -> 7B spot fast lane
    dec_simple = router.route("What is the capital of France?", primary_saturated=True)
    assert dec_simple.model_target == "7b_spot_fast_lane"
    assert dec_simple.is_spillover

    # Complex math query during saturation -> stays in 70B
    dec_math = router.route("Solve equation 2x + 4 = 10", primary_saturated=True)
    assert dec_math.model_target == "70b_primary_deep_lane"


def test_two_tier_radix_cache():
    cache = TwoTierRadixCacheManager(total_blocks=1000, max_tenant_quota_pct=0.15)

    # Pin system prompt
    h = cache.pin_system_prompt("Enterprise System Prompt Catalog", num_blocks=120)
    assert len(cache.pinned_blocks) == 120

    # Tenant quota allocation
    t1_blocks = cache.allocate_tenant_blocks("tenant_123", requested_blocks=50)
    assert len(t1_blocks) == 50

    # Allocate up to and exceeding 150 blocks (15% of 1000)
    t1_excess = cache.allocate_tenant_blocks("tenant_123", requested_blocks=120)
    stats = cache.get_stats()
    # Tenant 123 cannot exceed 150 blocks total!
    assert len(cache.tenant_allocations["tenant_123"]) <= 150
    assert stats["pinned_system_blocks"] == 120


def test_progressive_cache_primer():
    primer = ProgressiveCachePrimer(num_pods=4)

    p1 = primer.prime_user_cache("user_a", "tenant_alpha")
    p2 = primer.prime_user_cache("user_b", "tenant_alpha")
    # Same tenant maps to same pod via consistent hashing
    assert p1.target_pod_id == p2.target_pod_id
    assert p1.is_warm


def test_idle_residual_block_watchdog():
    dog = IdleResidualBlockWatchdog(total_block_pool=2000, pinned_baseline_blocks=120, leak_threshold_pct=0.15)

    # Clean idle baseline
    ev1 = dog.evaluate(step=1, active_requests=0, blocks_used=120)
    assert not ev1.is_leak_detected
    assert ev1.readiness_probe_status == 200

    # Active requests in flight -> skip
    ev2 = dog.evaluate(step=2, active_requests=50, blocks_used=450)
    assert not ev2.is_leak_detected
    assert ev2.readiness_probe_status == 200

    # Severe leak (120 + 350 = 470 blocks used > 15% leak threshold of 300)
    ev3 = dog.evaluate(step=3, active_requests=0, blocks_used=470)
    assert ev3.is_leak_detected
    assert ev3.readiness_probe_status == 503  # Trips readiness probe!


def test_semantic_complexity_oracle():
    oracle = SemanticComplexityOracle()

    # Short query that cannot be gamed: 'Write CUDA kernel'
    res_cuda = oracle.classify("Write CUDA kernel")
    assert res_cuda.target_model == "70b_deep_lane"
    assert res_cuda.is_reasoning_heavy

    # Simple conversational query
    res_conv = oracle.classify("Translate hello to Spanish")
    assert res_conv.target_model == "7b_fast_lane"
    assert not res_conv.is_reasoning_heavy


def test_spot_handoff_simulator():
    sim = SpotHandoffSimulator()
    events = sim.simulate_preemption_event()
    assert len(events) == 5
    assert events[0].timestamp_s == 0.0
    assert events[-1].dropped_requests_count == 0
    assert events[-2].timestamp_s < 30.0  # Replacement up before 30s deadline
