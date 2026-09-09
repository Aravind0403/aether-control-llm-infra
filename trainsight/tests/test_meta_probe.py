import pytest
from trainsight.profilers.activation_profiler import ModelArchitectureSpec
from trainsight.profilers.meta_probe import SingleLayerMetaProbe


def test_single_layer_meta_probe_speed():
    spec = ModelArchitectureSpec.from_model_id_or_config("qwen/qwen2.5-1.5b")
    probe = SingleLayerMetaProbe()

    res = probe.probe(spec)

    assert res.num_layers == 28
    assert res.total_static_vram_gb > 0.0
    # Homogeneous probing should complete in under 500ms even on slow environments
    assert res.probe_latency_ms < 1000.0
