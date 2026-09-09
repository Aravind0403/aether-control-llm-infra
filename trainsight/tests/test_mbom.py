import pytest
from trainsight.profilers.activation_profiler import ModelArchitectureSpec, ActivationProfiler
from trainsight.profilers.mbom import MBOMGenerator
from trainsight.simulators.collation_simulator import CollationSimulator


def test_mbom_generator_pedagogy_optimizer_advice():
    spec = ModelArchitectureSpec.from_model_id_or_config("qwen/qwen2.5-1.5b")
    profiler = ActivationProfiler(spec)

    mem_res = profiler.profile_run(batch_size=32, worst_case_seq_len=2048, hardware_vram_gb=24.0, optimizer_type="adamw")
    col_sim = CollationSimulator(num_trials=100)
    col_res = col_sim.simulate([100, 500, 2048] * 20, batch_size=32)

    mbom_gen = MBOMGenerator()
    receipt = mbom_gen.generate(job_name="test-run", mem_result=mem_res, collation_result=col_res)

    assert len(receipt.components) == 5
    assert receipt.primary_hog_component in ["Optimizer States", "Dynamic Activations", "Model Weights"]
    assert len(receipt.recommendations) > 0
