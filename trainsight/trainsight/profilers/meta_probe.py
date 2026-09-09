import time
import copy
from typing import Dict, Any, Optional, Tuple
from pydantic import BaseModel

from trainsight.profilers.activation_profiler import ModelArchitectureSpec, MemoryBreakdownResult, ActivationProfiler


class MetaProbeResult(BaseModel):
    model_name: str
    num_layers: int
    single_layer_vram_mb: float
    embeddings_vram_mb: float
    lm_head_vram_mb: float
    total_static_vram_gb: float
    probe_latency_ms: float
    is_meta_device_supported: bool
    fallback_used: bool = False
    warning: Optional[str] = None


class SingleLayerMetaProbe:
    """Zero-GPU Single-Layer Homogeneous Prober.
    
    Instantiates Layer 0 + Embeddings on torch.device('meta') to evaluate
    true tensor geometry in <15ms without allocating physical GPU/CPU memory,
    then projects linearly across all L layers.
    """

    @staticmethod
    def _calculate_module_meta_bytes(module: Any) -> int:
        """Calculates total tensor bytes for all parameters and buffers in an nn.Module."""
        total_bytes = 0
        for param in module.parameters():
            # numel * element_size
            total_bytes += param.numel() * param.element_size()
        for buf in module.buffers():
            total_bytes += buf.numel() * buf.element_size()
        return total_bytes

    def probe(self, spec: ModelArchitectureSpec) -> MetaProbeResult:
        """Runs single-layer meta-device probe in <15ms."""
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM
        except Exception:
            torch = None
            AutoConfig = None
            AutoModelForCausalLM = None

        start_t = time.perf_counter()

        if torch is not None and AutoConfig is not None:
            try:
                # Check if HuggingFace config can be loaded locally without remote network I/O
                try:
                    hf_config = AutoConfig.from_pretrained(spec.name, local_files_only=True)
                except Exception:
                    from transformers import LlamaConfig
                    hf_config = LlamaConfig(
                        vocab_size=spec.vocab_size,
                        hidden_size=spec.hidden_size,
                        num_hidden_layers=spec.num_hidden_layers,
                        num_attention_heads=spec.num_attention_heads,
                        num_key_value_heads=spec.num_key_value_heads,
                    )

                # Override to 1 layer for fast probing
                single_layer_cfg = copy.deepcopy(hf_config)
                single_layer_cfg.num_hidden_layers = 1

                with torch.device("meta"):
                    model_probe = AutoModelForCausalLM.from_config(single_layer_cfg)

                # Measure components
                layers_list = None
                embed_mod = None
                lm_head_mod = getattr(model_probe, "lm_head", None)

                base_model = getattr(model_probe, "model", getattr(model_probe, "transformer", None))
                if base_model:
                    layers_list = getattr(base_model, "layers", getattr(base_model, "h", None))
                    embed_mod = getattr(base_model, "embed_tokens", getattr(base_model, "wte", None))

                if layers_list and len(layers_list) > 0 and embed_mod is not None:
                    layer_0_bytes = self._calculate_module_meta_bytes(layers_list[0])
                    embed_bytes = self._calculate_module_meta_bytes(embed_mod)
                    lm_head_bytes = self._calculate_module_meta_bytes(lm_head_mod) if lm_head_mod else 0

                    L = spec.num_hidden_layers
                    total_bytes = embed_bytes + (L * layer_0_bytes) + lm_head_bytes
                    total_gb = total_bytes / (1024 ** 3)
                    latency_ms = (time.perf_counter() - start_t) * 1000.0

                    return MetaProbeResult(
                        model_name=spec.name,
                        num_layers=L,
                        single_layer_vram_mb=round(layer_0_bytes / (1024 ** 2), 2),
                        embeddings_vram_mb=round(embed_bytes / (1024 ** 2), 2),
                        lm_head_vram_mb=round(lm_head_bytes / (1024 ** 2), 2),
                        total_static_vram_gb=round(total_gb, 2),
                        probe_latency_ms=round(latency_ms, 2),
                        is_meta_device_supported=True,
                        fallback_used=False,
                    )
            except Exception:
                pass

        # Analytical fallback
        latency_ms = (time.perf_counter() - start_t) * 1000.0
        static_bytes = spec.total_params * spec.precision_bytes
        single_layer_bytes = int(static_bytes / max(1, spec.num_hidden_layers))

        return MetaProbeResult(
            model_name=spec.name,
            num_layers=spec.num_hidden_layers,
            single_layer_vram_mb=round(single_layer_bytes / (1024 ** 2), 2),
            embeddings_vram_mb=round((spec.vocab_size * spec.hidden_size * spec.precision_bytes) / (1024 ** 2), 2),
            lm_head_vram_mb=round((spec.vocab_size * spec.hidden_size * spec.precision_bytes) / (1024 ** 2), 2),
            total_static_vram_gb=round(static_bytes / (1024 ** 3), 2),
            probe_latency_ms=round(latency_ms, 2),
            is_meta_device_supported=False,
            fallback_used=True,
            warning="Meta-device probe bypassed; used analytical parameter scaling.",
        )
