import pytest
from trainsight.profilers.activation_profiler import (
    ActivationProfiler,
    ModelArchitectureSpec,
    DistributedStrategyConfig,
)


def test_activation_profiler_qwen_1_5b():
    spec = ModelArchitectureSpec.from_model_id_or_config("qwen/qwen2.5-1.5b")
    assert spec.hidden_size == 1536
    assert spec.num_hidden_layers == 28

    profiler = ActivationProfiler(spec)

    # Calculate activations for batch size 8, sequence length 1024
    act_bytes_ckpt = profiler.calculate_activations_bytes(batch_size=8, seq_len=1024, gradient_checkpointing=True)
    act_bytes_no_ckpt = profiler.calculate_activations_bytes(batch_size=8, seq_len=1024, gradient_checkpointing=False)

    # Without checkpointing, activation memory should be significantly larger
    assert act_bytes_no_ckpt > act_bytes_ckpt

    # Test full 5-tenant profile run
    dist_cfg = DistributedStrategyConfig(strategy="zero3", world_size=8)
    mem = profiler.profile_run(
        batch_size=8,
        worst_case_seq_len=1024,
        hardware_vram_gb=24.0,
        optimizer_type="adamw",
        gradient_checkpointing=True,
        dist_config=dist_cfg,
    )

    assert mem.weights_gb > 0.0
    assert mem.gradients_gb > 0.0
    assert mem.optimizer_gb > 0.0
    assert mem.activations_gb > 0.0
    assert mem.comm_buffers_gb > 0.0
    assert mem.fits_in_vram is True


def test_activation_profiler_optimizer_8bit_reduction():
    spec = ModelArchitectureSpec.from_model_id_or_config("meta-llama/llama-3-8b")
    profiler = ActivationProfiler(spec)

    mem_fp32 = profiler.profile_run(batch_size=4, worst_case_seq_len=512, hardware_vram_gb=80.0, optimizer_type="adamw")
    mem_8bit = profiler.profile_run(batch_size=4, worst_case_seq_len=512, hardware_vram_gb=80.0, optimizer_type="adamw_8bit")

    # 8-bit AdamW should use ~1/3 of optimizer memory compared to FP32 AdamW
    assert mem_8bit.optimizer_gb < (mem_fp32.optimizer_gb * 0.4)
