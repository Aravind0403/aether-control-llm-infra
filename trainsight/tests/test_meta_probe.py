import pytest
from trainsight.profilers.activation_profiler import ModelArchitectureSpec
from trainsight.profilers.meta_probe import SingleLayerMetaProbe, META_PROBE_SLA_MS


def test_single_layer_meta_probe_speed():
    spec = ModelArchitectureSpec.from_model_id_or_config("qwen/qwen2.5-1.5b")
    probe = SingleLayerMetaProbe()

    # Warm-up pass to load underlying PyTorch meta-device structures
    probe.probe(spec)

    # Active probed pass
    res = probe.probe(spec)

    assert res.num_layers == 28
    assert res.total_static_vram_gb > 0.0
    assert META_PROBE_SLA_MS == 15.0
    assert res.probe_latency_ms < META_PROBE_SLA_MS
