import itertools
import unittest

from compare_lowerings import (PATHS, RUNTIME_PATHS, MPP_PATHS, VIEW_PATHS, implementation_order, validate_native, validate_times, validate_mpp_tirx,
                               validate_tirx_runtime, validate_runtime_controls, gpu_samples)


class LoweringComparisonTest(unittest.TestCase):
    def test_gpu_controls_do_not_fall_back_to_host_or_instrumented_time(self):
        sample = dict(compute_ns=2000., compute_span_ns=2100., command_buffer_ns=3000.,
                      calibration_cpu_ns=1e6, calibration_gpu_ticks=24000., compute_passes=1, command_buffers=1)
        control = dict(method="metal_command_buffer_timestamps_v1", scope="sum_of_command_buffer_gpu_intervals",
                       encoder_instrumentation=False, repetitions=4,
                       throughput=[dict(command_buffer_ns=12000., command_buffers=1)] * 2,
                       latency=[dict(command_buffer_ns=5000., command_buffers=1)] * 2)
        device = dict(method="metal_compute_pass_timestamps_v1", scope="sum_of_compute_encoder_gpu_intervals",
                      host_samples_instrumented=False, repetitions=4, control=control,
                      throughput=[sample] * 2, latency=[sample] * 2)
        result = dict(device_timing=device, throughput_us=[100., 200.])
        self.assertEqual(gpu_samples(result, "torch", "throughput", 2), [3., 3.])
        self.assertEqual(gpu_samples(result, "tile_native_mpp", "latency", 2), [5., 5.])
        for invalid in ({}, {"throughput_us": [1., 2.]},
                        {"device_timing": {k: v for k, v in device.items() if k != "control"}},
                        {"device_timing": device | {"control": control | {"encoder_instrumentation": True}}}):
            with self.assertRaises(ValueError):
                gpu_samples(invalid, "tile_tirx", "throughput", 2)

    def test_hand_mpp_preserves_its_direct_command_buffer_timer(self):
        result = dict(gpu_throughput_us=[1., 2.], gpu_latency_us=[3., 4.], throughput_us=[100., 200.])
        self.assertEqual(gpu_samples(result, "handwritten_mpp", "throughput", 2), [1., 2.])
        self.assertEqual(gpu_samples(result, "handwritten_mpp", "latency", 2), [3., 4.])
        for values in ([True, 2.], [0., 2.], [float("nan"), 2.], [1.]):
            with self.assertRaises(ValueError):
                gpu_samples(result | {"gpu_throughput_us": values}, "handwritten_mpp", "throughput", 2)
        with self.assertRaises(ValueError):
            gpu_samples(result, "handwritten_mpp", "other", 2)

    def test_mpp_counterbalance(self):
        for paths in (PATHS + MPP_PATHS, PATHS + RUNTIME_PATHS + MPP_PATHS, PATHS + MPP_PATHS + VIEW_PATHS):
            for offset in range(2 * len(paths)):
                orders = [implementation_order(i, offset, paths) for i in range(2 * len(paths))]
                for name in paths:
                    self.assertEqual(sorted(o.index(name) for o in orders), sorted(list(range(len(paths))) * 2))
                for a, b in itertools.combinations(paths, 2):
                    self.assertEqual(sum(o.index(a) < o.index(b) for o in orders), len(paths))

    def test_mpp_codegen_identity(self):
        result = dict(metal_mpp=True, forward_readonly_tile_loads=False, simdgroup_intrinsics=0, mpp_intrinsics=1, matrix_intrinsics=1,
                      execution_plans=[dict(metal_mpp=True, cost_basis="metal_mpp_memory_v2")])
        validate_mpp_tirx(result)
        validate_mpp_tirx(result | dict(forward_readonly_tile_loads=True), views=True)
        with self.assertRaises(ValueError):
            validate_mpp_tirx(result, views=True)
        for change in (dict(metal_mpp=False), dict(forward_readonly_tile_loads=True), dict(simdgroup_intrinsics=1), dict(mpp_intrinsics=0),
                       dict(mpp_intrinsics=True), dict(matrix_intrinsics=2), dict(execution_plans=[]),
                       dict(execution_plans=[dict(metal_mpp=True, cost_basis="simdgroup_reference_geometry")])):
            with self.assertRaises(ValueError):
                validate_mpp_tirx(result | change)

    def test_counterbalance(self):
        for offset in range(10):
            orders = [implementation_order(i, offset) for i in range(10)]
            for name in PATHS:
                self.assertEqual(sorted(o.index(name) for o in orders), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
            for a, b in itertools.combinations(PATHS, 2):
                self.assertEqual(sum(o.index(a) < o.index(b) for o in orders), 5)

    def test_runtime_control_counterbalance(self):
        paths = PATHS + RUNTIME_PATHS
        for offset in range(14):
            orders = [implementation_order(i, offset, paths) for i in range(14)]
            for name in paths:
                self.assertEqual(sorted(o.index(name) for o in orders), sorted(list(range(7)) * 2))
            for a, b in itertools.combinations(paths, 2):
                self.assertEqual(sum(o.index(a) < o.index(b) for o in orders), 7)

    def test_runtime_control_identity(self):
        config = dict(gemm_block=(32, 64, 32), execution_scope="group", pipeline_window=1,
                      cooperative_matrix=True, no_vectorize=False, auto_vectorize=False, group_threads=256, copy_batch=8)
        result = dict(backend="metal", runtime="luisa", timing="synchronized_host_wall", fast_math=False,
                      operation="gemm", output_elements=32 * 32, mma_operations=1, execution_scope="group",
                      pipeline_window=1, cooperative_matrix=True, vectorize=True, auto_vectorize=False,
                      planner_threads=256, copy_batch=8, realized_threads=256, matrix_intrinsics=4)
        validate_tirx_runtime(result, (32, 32, 32), config, False)
        for key, value in (("runtime", "tvm"), ("fast_math", True), ("realized_threads", 128), ("copy_batch", 4), ("output_elements", 1)):
            with self.assertRaises(ValueError):
                validate_tirx_runtime(result | {key: value}, (32, 32, 32), config, False)

    def test_same_device_source_is_required(self):
        baseline = dict(valid=True, source_sha256="abc", measurement=dict(matrix_intrinsics=4, execution_plans=[dict(threads=256)]))
        control = dict(valid=True, source_sha256="abc", measurement=dict(matrix_intrinsics=4, realized_threads=256))
        row = dict(tile_tirx=baseline, tile_tirx_luisa=control)
        validate_runtime_controls(row)
        for changes in (dict(valid=False), dict(source_sha256="xyz"),
                        dict(measurement=dict(matrix_intrinsics=4, realized_threads=128)),
                        dict(measurement=dict(matrix_intrinsics=0, realized_threads=256))):
            with self.assertRaises(ValueError):
                validate_runtime_controls(row | {"tile_tirx_luisa": control | changes})

    def test_native_identity_and_precision(self):
        result = dict(implementation="tile_native_mpp", backend="metal", precision="fp32", fast_math=False,
                      relaxed_precision=False, timing="synchronized_host_wall", m=17, n=23, k=61,
                      block=[32, 32], execution_simdgroups=1, group_simdgroups=4, cohort_rows=2)
        config = (32, 32, 1, 1, 0, 1, 4, 2)
        validate_native(result, (17, 23, 61), config, 2)
        for key, value in (("implementation", "mpp_tensor_ops_matmul2d"), ("fast_math", True),
                           ("m", 18), ("cohort_rows", 4), ("timing", "gpu_event")):
            with self.assertRaises(ValueError):
                validate_native(result | {key: value}, (17, 23, 61), config, 2)
        with self.assertRaises(ValueError):
            validate_native(result, (17, 23, 61), (32, 32, 1, 1, 1, 1, 4, 2), 2)

    def test_host_samples_required(self):
        good = dict(throughput_us=[1, 2], latency_us=[3, 4])
        validate_times(good, 2)
        for bad in ([float("nan"), 2], [0, 1], [-1, 1], [True, 2], [1]):
            with self.assertRaises(ValueError):
                validate_times(good | {"throughput_us": bad}, 2)
        with self.assertRaises(ValueError):
            validate_times(dict(gpu_throughput_us=[1, 2], gpu_latency_us=[3, 4]), 2)


if __name__ == "__main__":
    unittest.main()
