import time

from gpu_sim.profiling import TensorStageProfiler


def test_tensor_stage_profiler_records_calls_and_time():
    profiler = TensorStageProfiler("cpu")
    with profiler.stage("draft"):
        time.sleep(0.001)

    metrics = profiler.as_dict()

    assert metrics["draft"]["calls"] == 1
    assert metrics["draft"]["elapsed_seconds"] > 0.0


def test_tensor_stage_profiler_accumulates_repeated_calls():
    profiler = TensorStageProfiler("cpu")
    with profiler.stage("lineup"):
        pass
    with profiler.stage("lineup"):
        pass

    assert profiler.as_dict()["lineup"]["calls"] == 2
