import math
from typing import Dict, Any


def calculate_inference_memory(param_billions: float, seq_len: int = 2048, batch_size: int = 16, num_layers: int = 32, num_heads: int = 32, head_dim: int = 128) -> Dict[str, float]:
    """Calculates inference VRAM requirements (in GB) for FP16 model weights and PagedAttention KV-cache."""
    # FP16 Weights = 2 bytes per param
    weight_gb = param_billions * 2.0

    # KV Cache per token per layer = 2 * num_layers * num_heads * head_dim * 2 bytes (FP16)
    bytes_per_token = 2 * num_layers * num_heads * head_dim * 2
    kv_cache_gb = (batch_size * seq_len * bytes_per_token) / (1024 ** 3)

    # Overhead (activation buffers + vLLM block manager metadata ~10%)
    activation_overhead_gb = 0.5

    total_gb = weight_gb + kv_cache_gb + activation_overhead_gb
    return {
        "weight_gb": weight_gb,
        "kv_cache_gb": kv_cache_gb,
        "activation_overhead_gb": activation_overhead_gb,
        "total_gb": total_gb,
        "bytes_per_param": total_gb / param_billions
    }


def calculate_training_memory(param_billions: float, seq_len: int = 2048, batch_size: int = 4, num_layers: int = 32, hidden_dim: int = 4096) -> Dict[str, float]:
    """Calculates training VRAM requirements (in GB) for FP16 weights, FP16 gradients, FP32 Adam, and Activations."""
    # FP16 Weights = 2 bytes per param
    weight_gb = param_billions * 2.0

    # FP16 Gradients = 2 bytes per param
    grad_gb = param_billions * 2.0

    # Adam Optimizer States = 12 bytes per param (4 bytes FP32 master weight + 4 bytes m_t + 4 bytes v_t)
    opt_states_gb = param_billions * 12.0

    # Intermediate Activations (estimated for standard attention + MLP per layer)
    # Approx: batch_size * seq_len * hidden_dim * num_layers * 16 bytes (with activation checkpointing)
    activations_gb = (batch_size * seq_len * hidden_dim * num_layers * 2) / (1024 ** 3)

    total_gb = weight_gb + grad_gb + opt_states_gb + activations_gb
    return {
        "weight_gb": weight_gb,
        "grad_gb": grad_gb,
        "opt_states_gb": opt_states_gb,
        "activations_gb": activations_gb,
        "total_gb": total_gb,
        "bytes_per_param": total_gb / param_billions
    }


def print_comparison(model_name: str, param_b: float):
    print(f"\n================================================================================")
    print(f"📊 MEMORY & COMMUNICATION PROFILE: {model_name} ({param_b}B Parameters)")
    print(f"================================================================================\n")

    inf = calculate_inference_memory(param_b)
    train = calculate_training_memory(param_b)

    print("1. MEMORY FOOTPRINT COMPARISON:")
    print(f"   • Inference VRAM (vLLM PagedAttention): {inf['total_gb']:.2f} GB ({inf['bytes_per_param']:.2f} bytes/param)")
    print(f"     - Model Weights (FP16): {inf['weight_gb']:.2f} GB")
    print(f"     - KV-Cache Block Pool:  {inf['kv_cache_gb']:.2f} GB")
    print(f"     - Activation Overhead:  {inf['activation_overhead_gb']:.2f} GB")

    print(f"\n   • Training VRAM (Full PyTorch / Un-sharded): {train['total_gb']:.2f} GB ({train['bytes_per_param']:.2f} bytes/param)")
    print(f"     - Model Weights (FP16): {train['weight_gb']:.2f} GB")
    print(f"     - Gradients (FP16):     {train['grad_gb']:.2f} GB")
    print(f"     - Adam Opt States (FP32): {train['opt_states_gb']:.2f} GB")
    print(f"     - Activation Memory:    {train['activations_gb']:.2f} GB")

    ratio = train['total_gb'] / inf['total_gb']
    print(f"\n   👉 Training consumes {ratio:.1f}x MORE VRAM than Inference!")

    print("\n2. CROSS-GPU COMMUNICATION PATTERNS:")
    print("   • Training (DeepSpeed ZeRO-3 / DDP):")
    print("     - Forward Pass:  All-Gather parameters per layer (reconstruct shards)")
    print("     - Backward Pass: Reduce-Scatter gradients across nodes")
    print("     - Traffic:       ~1.5x model parameters transferred per step")
    print("   • Inference (vLLM Tensor Parallelism):")
    print("     - Prefill Phase:  All-Reduce GEMM outputs per layer (NVLink bound)")
    print("     - Decode Phase:   Low bandwidth, memory-bound token fetches")

    print("\n3. FAILURE BLAST RADIUS:")
    print("   • Training:  Stateful. 1 GPU failure crashes the entire step/job (SIGKILL). Requires checkpoint rollback.")
    print("   • Inference: Stateless. 1 Pod crash affects only in-flight requests on that pod. KEDA auto-heals.")
    print("================================================================================\n")


if __name__ == "__main__":
    print_comparison("Qwen2.5-1.5B", 1.5)
    print_comparison("Qwen2.5-70B", 70.0)
