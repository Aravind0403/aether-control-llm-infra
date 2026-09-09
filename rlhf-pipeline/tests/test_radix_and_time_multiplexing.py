import pytest
from rlhf_pipeline.engine.radix_cache import RadixCacheSimulator
from rlhf_pipeline.engine.time_multiplexer import VRAMTimeMultiplexer


def test_radix_cache_simulator_savings():
    sim = RadixCacheSimulator(num_layers=80, num_kv_heads=8, head_dim=128, bytes_per_element=2)
    # G=8, Prompt=1024, Gen=1024
    rep = sim.evaluate_group_savings(group_size=8, prompt_tokens=1024, generation_tokens=1024)

    assert rep.naive_total_tokens == 16384
    assert rep.radix_total_tokens == 9216
    assert rep.tokens_saved == 7168
    assert rep.total_token_reduction_pct == pytest.approx(43.75, abs=0.1)
    assert rep.prompt_redundancy_eliminated_pct == pytest.approx(87.5, abs=0.1)
    assert rep.kv_cache_saved_mb > 2000.0


def test_vram_time_multiplexer_lifecycle():
    mux = VRAMTimeMultiplexer(gpu_capacity_gb=80.0)

    # Phase 1: Rollout
    rollout_receipt = mux.enter_rollout(radix_kv_cache_gb=5.8)
    assert rollout_receipt.phase == "rollout_inference"
    assert rollout_receipt.gradients_gb == 0.0
    assert rollout_receipt.optimizer_gb == 0.0
    assert rollout_receipt.total_peak_vram_gb == pytest.approx(23.3, abs=0.1)
    assert rollout_receipt.headroom_on_80gb_pct > 70.0
    assert rollout_receipt.collision_free

    # Handoff: Flush KV cache
    freed = mux.exit_rollout()
    assert freed == pytest.approx(5.8, abs=0.1)
    assert mux.active_kv_cache_gb == 0.0

    # Phase 2: Train Backward
    train_receipt = mux.enter_train()
    assert train_receipt.phase == "train_backward"
    assert train_receipt.kv_cache_gb == 0.0
    assert train_receipt.gradients_gb == 17.5
    assert train_receipt.total_peak_vram_gb == pytest.approx(52.2, abs=0.1)
    assert train_receipt.headroom_on_80gb_pct > 30.0
    assert train_receipt.collision_free
