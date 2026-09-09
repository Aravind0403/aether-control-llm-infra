import json
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


# Common production model presets (zero network call required)
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "qwen/qwen2.5-1.5b": {
        "num_hidden_layers": 28,
        "hidden_size": 1536,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "vocab_size": 151936,
        "total_params": 1540000000,
    },
    "qwen/qwen2.5-7b": {
        "num_hidden_layers": 28,
        "hidden_size": 3584,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "vocab_size": 152064,
        "total_params": 7610000000,
    },
    "meta-llama/llama-3-8b": {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "total_params": 8030000000,
    },
    "meta-llama/llama-3-70b": {
        "num_hidden_layers": 80,
        "hidden_size": 8192,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "total_params": 70600000000,
    },
    "mistralai/mixtral-8x7b": {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 32000,
        "total_params": 46700000000,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
    },
}

HARDWARE_PRESETS: Dict[str, float] = {
    "l4-24gb": 24.0,
    "a10g-24gb": 24.0,
    "a100-40gb": 40.0,
    "a100-80gb": 80.0,
    "h100-80gb": 80.0,
    "t4-16gb": 16.0,
    "v100-16gb": 16.0,
    "v100-32gb": 32.0,
}


class ModelArchitectureSpec(BaseModel):
    name: str = "custom-llm"
    num_hidden_layers: int = 28
    hidden_size: int = 1536
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    vocab_size: int = 151936
    total_params: int = 1540000000
    precision_bytes: int = 2  # FP16 or BF16 = 2 bytes
    num_local_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    expert_capacity_factor: float = 1.2

    @classmethod
    def from_model_id_or_config(cls, model_id_or_path: str) -> "ModelArchitectureSpec":
        """Resolves model spec from preset, local config file, or Hugging Face AutoConfig."""
        norm_key = model_id_or_path.lower().strip()
        for preset_name, spec in MODEL_PRESETS.items():
            if preset_name in norm_key or norm_key in preset_name:
                return cls(name=model_id_or_path, **spec)

        # Check local path
        p = Path(model_id_or_path)
        if p.exists() and p.is_file() and p.name.endswith(".json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                return cls(
                    name=p.stem,
                    num_hidden_layers=cfg.get("num_hidden_layers", 28),
                    hidden_size=cfg.get("hidden_size", 1536),
                    num_attention_heads=cfg.get("num_attention_heads", 12),
                    num_key_value_heads=cfg.get("num_key_value_heads", 2),
                    vocab_size=cfg.get("vocab_size", 151936),
                    total_params=cfg.get("total_params", 1540000000),
                    num_local_experts=cfg.get("num_local_experts"),
                    num_experts_per_tok=cfg.get("num_experts_per_tok"),
                )
            except Exception:
                pass

        # Try transformers AutoConfig if installed
        try:
            from transformers import AutoConfig
            hf_cfg = AutoConfig.from_pretrained(model_id_or_path)
            num_layers = getattr(hf_cfg, "num_hidden_layers", 28)
            hidden_sz = getattr(hf_cfg, "hidden_size", 1536)
            num_heads = getattr(hf_cfg, "num_attention_heads", 12)
            kv_heads = getattr(hf_cfg, "num_key_value_heads", num_heads)
            v_size = getattr(hf_cfg, "vocab_size", 151936)
            # Rough estimate of parameter count for standard transformer: 12 * L * h^2
            approx_params = getattr(hf_cfg, "total_params", None)
            if not approx_params:
                approx_params = int(12 * num_layers * (hidden_sz ** 2) + v_size * hidden_sz)

            return cls(
                name=model_id_or_path,
                num_hidden_layers=num_layers,
                hidden_size=hidden_sz,
                num_attention_heads=num_heads,
                num_key_value_heads=kv_heads,
                vocab_size=v_size,
                total_params=approx_params,
                num_local_experts=getattr(hf_cfg, "num_local_experts", None),
                num_experts_per_tok=getattr(hf_cfg, "num_experts_per_tok", None),
            )
        except Exception:
            pass

        # Fallback default
        return cls(name=model_id_or_path)


class DistributedStrategyConfig(BaseModel):
    strategy: str = "none"  # "none", "ddp", "zero3", "tensor_parallel"
    world_size: int = 1
    precision_bytes: int = 2
    nccl_buffsize_bytes: int = 4 * 1024 * 1024  # 4MB default
    nccl_channels: int = 16
    allgather_bucket_size: int = 200 * 1024 * 1024  # 200MB default
    reduce_bucket_size: int = 200 * 1024 * 1024     # 200MB default
    ddp_bucket_size_bytes: int = 25 * 1024 * 1024    # 25MB default

    @classmethod
    def from_deepspeed_json(cls, config_path: Path, world_size: int = 8) -> "DistributedStrategyConfig":
        if not config_path.exists():
            return cls(strategy="zero3", world_size=world_size)
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            z_cfg = d.get("zero_optimization", {})
            stage = z_cfg.get("stage", 3)
            allgather_b = z_cfg.get("allgather_bucket_size", 200000000)
            reduce_b = z_cfg.get("reduce_bucket_size", 200000000)
            return cls(
                strategy=f"zero{stage}",
                world_size=world_size,
                allgather_bucket_size=allgather_b,
                reduce_bucket_size=reduce_b,
            )
        except Exception:
            return cls(strategy="zero3", world_size=world_size)


class MemoryBreakdownResult(BaseModel):
    weights_gb: float
    gradients_gb: float
    optimizer_gb: float
    activations_gb: float
    comm_buffers_gb: float
    total_peak_gb: float
    available_vram_gb: float
    vram_headroom_gb: float
    fits_in_vram: bool
    utilization_pct: float
    optimizer_type: str
    gradient_checkpointing: bool


class ActivationProfiler:
    """Exact FlashAttention-2 / GQA analytical memory profiler."""

    def __init__(self, spec: ModelArchitectureSpec):
        self.spec = spec

    def calculate_activations_bytes(
        self,
        batch_size: int,
        seq_len: int,
        gradient_checkpointing: bool = True,
    ) -> int:
        """Computes activation memory for FlashAttention-2 with selective checkpointing."""
        L = self.spec.num_hidden_layers
        b = batch_size
        s = seq_len
        h = self.spec.hidden_size
        a = max(1, self.spec.num_attention_heads)
        n_kv = max(1, self.spec.num_key_value_heads)
        bytes_per_elem = self.spec.precision_bytes

        # Standard FlashAttention-2 with selective activation checkpointing
        if gradient_checkpointing:
            # Recomputation discards intermediate attention QK^T and softmax buffers
            # Layer factor: 10 + 2 * (n_kv / a)
            factor = 10.0 + 2.0 * (float(n_kv) / float(a))
        else:
            # Without activation checkpointing: store all Q, K, V, output projections and layernorm inputs
            factor = 34.0 + 4.0 * (float(n_kv) / float(a))

        # Check for Mixture-of-Experts (MoE) bounded capacity factor
        if self.spec.num_local_experts and self.spec.num_experts_per_tok:
            E = self.spec.num_local_experts
            k = self.spec.num_experts_per_tok
            C = self.spec.expert_capacity_factor
            T = b * s
            # Bounded tokens per expert on this rank
            t_expert_max = min(T * k, int(np.ceil((T * k / E) * C)))
            # MLP portion scales by active experts
            mlp_expert_factor = (t_expert_max / max(1, T)) * 4.0
            factor = factor + mlp_expert_factor

        activation_bytes = int(L * b * s * h * factor * bytes_per_elem)
        return activation_bytes

    def calculate_comm_overhead_bytes(
        self,
        dist_config: DistributedStrategyConfig,
        batch_size: int,
        seq_len: int,
    ) -> int:
        """Calculates physical NCCL ring buffer and framework aggregation buckets."""
        if dist_config.world_size <= 1:
            return 0

        # Physical NCCL pinned ring buffer
        peers = max(1, dist_config.world_size - 1)
        base_nccl = 2 * dist_config.nccl_buffsize_bytes * dist_config.nccl_channels * min(2, peers)

        strat = dist_config.strategy.lower()
        if "zero3" in strat:
            return base_nccl + dist_config.allgather_bucket_size + dist_config.reduce_bucket_size
        elif "tensor_parallel" in strat:
            # All-Reduce intermediate activation buffer: 2 * (b * s * h * precision_bytes)
            tp_buffer = 2 * (batch_size * seq_len * self.spec.hidden_size * self.spec.precision_bytes)
            return base_nccl + tp_buffer
        elif "ddp" in strat:
            return base_nccl + dist_config.ddp_bucket_size_bytes
        else:
            return base_nccl

    def profile_run(
        self,
        batch_size: int,
        worst_case_seq_len: int,
        hardware_vram_gb: float = 24.0,
        optimizer_type: str = "adamw",
        gradient_checkpointing: bool = True,
        dist_config: Optional[DistributedStrategyConfig] = None,
    ) -> MemoryBreakdownResult:
        """Performs full 5-tenant memory profiling."""
        if dist_config is None:
            dist_config = DistributedStrategyConfig()

        params = self.spec.total_params
        p_bytes = self.spec.precision_bytes

        # 1. Weights: params * 2 bytes (FP16/BF16)
        if "zero3" in dist_config.strategy.lower() and dist_config.world_size > 1:
            # ZeRO-3 shards model parameters across world_size GPUs
            m_weights = int((params * p_bytes) / dist_config.world_size)
        else:
            m_weights = int(params * p_bytes)

        # 2. Gradients: params * 2 bytes
        if "zero3" in dist_config.strategy.lower() and dist_config.world_size > 1:
            m_grads = int((params * p_bytes) / dist_config.world_size)
        else:
            m_grads = int(params * p_bytes)

        # 3. Optimizer States
        # FP32 AdamW: 12 bytes per parameter (4 bytes master weight + 4 bytes momentum + 4 bytes variance)
        # 8-bit AdamW: 4 bytes per parameter (4 bytes master or quantized states)
        # SGD with momentum: 4 bytes per parameter
        opt_key = optimizer_type.lower()
        if "8bit" in opt_key or "adamw8bit" in opt_key:
            opt_bytes_per_param = 4
        elif "sgd" in opt_key:
            opt_bytes_per_param = 4
        else:
            opt_bytes_per_param = 12

        if "zero3" in dist_config.strategy.lower() and dist_config.world_size > 1:
            m_opt = int((params * opt_bytes_per_param) / dist_config.world_size)
        else:
            m_opt = int(params * opt_bytes_per_param)

        # 4. Activations (worst-case batch shape)
        m_act = self.calculate_activations_bytes(
            batch_size=batch_size,
            seq_len=worst_case_seq_len,
            gradient_checkpointing=gradient_checkpointing,
        )

        # 5. Communication Buffers
        m_comm = self.calculate_comm_overhead_bytes(
            dist_config=dist_config,
            batch_size=batch_size,
            seq_len=worst_case_seq_len,
        )

        total_bytes = m_weights + m_grads + m_opt + m_act + m_comm
        total_gb = total_bytes / (1024 ** 3)
        vram_headroom_gb = hardware_vram_gb - total_gb
        fits = total_gb <= hardware_vram_gb
        utilization_pct = (total_gb / hardware_vram_gb) * 100.0 if hardware_vram_gb > 0 else 0.0

        return MemoryBreakdownResult(
            weights_gb=round(m_weights / (1024 ** 3), 2),
            gradients_gb=round(m_grads / (1024 ** 3), 2),
            optimizer_gb=round(m_opt / (1024 ** 3), 2),
            activations_gb=round(m_act / (1024 ** 3), 2),
            comm_buffers_gb=round(m_comm / (1024 ** 3), 2),
            total_peak_gb=round(total_gb, 2),
            available_vram_gb=round(hardware_vram_gb, 2),
            vram_headroom_gb=round(vram_headroom_gb, 2),
            fits_in_vram=fits,
            utilization_pct=round(utilization_pct, 1),
            optimizer_type=optimizer_type,
            gradient_checkpointing=gradient_checkpointing,
        )
