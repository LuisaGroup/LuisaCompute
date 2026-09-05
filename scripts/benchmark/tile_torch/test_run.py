import argparse
import copy
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("tile_torch_benchmark", Path(__file__).with_name("run.py"))
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

REPEAT_SPEC = importlib.util.spec_from_file_location("tile_torch_repeat", Path(__file__).with_name("repeat.py"))
REPEAT = importlib.util.module_from_spec(REPEAT_SPEC)
with patch.dict(sys.modules, {"run": MODULE}):
    REPEAT_SPEC.loader.exec_module(REPEAT)

SYSTEM_SPEC = importlib.util.spec_from_file_location("tile_system_compare", Path(__file__).with_name("compare_system.py"))
SYSTEM = importlib.util.module_from_spec(SYSTEM_SPEC)
with patch.dict(sys.modules, {"run": MODULE, "repeat": REPEAT}):
    SYSTEM_SPEC.loader.exec_module(SYSTEM)


class SystemBaselineTests(unittest.TestCase):
    def test_three_implementation_orders_are_balanced_per_case(self):
        keys = [("metal", str(i)) for i in range(8)]
        for key in keys:
            orders = [MODULE.implementation_order(ordinal, True) for r in range(6)
                      for k, ordinal in SYSTEM.order_for_round(keys, r) if k == key]
            self.assertEqual(len(set(orders)), 6)
            for implementation in ("native", "torch", "system"):
                self.assertEqual([sum(o.index(implementation) == i for o in orders) for i in range(3)], [2, 2, 2])
            for a, b in (("native", "torch"), ("native", "system"), ("system", "torch")):
                self.assertEqual(sum(o.index(a) < o.index(b) for o in orders), 3)

    def test_two_implementation_replay_is_unchanged(self):
        for ordinal in range(6):
            self.assertEqual(MODULE.implementation_order(ordinal),
                             ("native", "torch") if ordinal % 2 == 0 else ("torch", "native"))

    @staticmethod
    def row(backend):
        return dict(backend=backend, operation="gemm", dtype="float32", layout="compact_row_major",
                    m=7, n=19, k=13, alpha=1, beta=0, transpose_left=False, transpose_right=False,
                    row_bytes=[52, 76, 76], repetitions=5, throughput_us=[2., 3.], latency_us=[6., 7.],
                    implementation="accelerate_cblas_sgemm" if backend == "cpu" else "mps_matrix_multiplication",
                    api_variant="classic_lp64" if backend == "cpu" else "MPSKernelOptionsNone",
                    storage="host" if backend == "cpu" else "private",
                    batch_policy="synchronous_calls" if backend == "cpu" else "one_command_buffer_per_batch")

    def test_system_api_shape_precision_and_layout_are_checked(self):
        case = MODULE.Case("gemm", 7, 19, 13)
        for backend in ("cpu", "metal"):
            row = self.row(backend)
            MODULE.validate_system_metadata(row, case, backend, 2)
            for key, value in (("backend", "cuda"), ("operation", "add"), ("m", 8), ("dtype", "float16"),
                               ("transpose_left", 0), ("beta", 1), ("row_bytes", [52, 80, 80]),
                               ("implementation", "torch"), ("api_variant", "reduced_precision"),
                               ("storage", "managed"), ("batch_policy", "single_call")):
                with self.subTest(backend=backend, key=key), self.assertRaisesRegex(RuntimeError, "metadata mismatch"):
                    MODULE.validate_system_metadata(row | {key: value}, case, backend, 2)

    def test_invalid_timings_or_unsupported_operations_fail_closed(self):
        case = MODULE.Case("gemm", 7, 19, 13)
        row = self.row("metal")
        for value in (None, 0, True, 100001):
            with self.assertRaisesRegex(RuntimeError, "repetition"):
                MODULE.validate_system_metadata(row | {"repetitions": value}, case, "metal", 2)
        for metric in ("throughput_us", "latency_us"):
            for values in ([], [1], [1, float("nan")], [1, float("inf")], [1, 0], [1, -1], [1, True]):
                with self.assertRaisesRegex(RuntimeError, "samples"):
                    MODULE.validate_system_metadata(row | {metric: values}, case, "metal", 2)
        with self.assertRaisesRegex(RuntimeError, "only CPU/Metal GEMM"):
            MODULE.validate_system_metadata(row, MODULE.Case("add", 7, 19), "metal", 2)

    def test_report_uses_round_medians_not_the_lucky_minimum(self):
        rows = [dict(backend="metal", name="example", valid=True, round=i, system_slowdown=t / 2,
                     native={"throughput_us_p50": t}, torch={"throughput_us_p50": 4}, system={"throughput_us_p50": 2})
                for i, t in enumerate([1, 2, 3, 4, 5, 100])]
        with patch.object(Path, "write_text") as write:
            SYSTEM.write_report({"metadata": {"rounds": 6}, "results": rows}, Path("unused"))
        markdown = write.call_args_list[-1].args[0]
        self.assertIn("6/6 | 3.500 | 4.000 | 2.000 | 1.750", markdown)
        self.assertIn("[0.500, 50.000]", markdown)

    def test_report_keeps_failed_rounds_and_withholds_complete_statistics(self):
        rows = [dict(backend="metal", name="example", valid=True, round=i) for i in range(5)]
        rows.append(dict(backend="metal", name="example", valid=False, round=5, error="bad output"))
        with patch.object(Path, "write_text") as write:
            SYSTEM.write_report({"metadata": {"rounds": 6}, "results": rows}, Path("unused"))
        raw = json.loads(write.call_args_list[0].args[0])
        self.assertEqual(len(raw["results"]), 6)
        markdown = write.call_args_list[-1].args[0]
        self.assertIn("5/6 | INCOMPLETE", markdown)
        self.assertIn("Failed measurements: 1", markdown)


class DeviceTimingTests(unittest.TestCase):
    @classmethod
    def controlled_timing(cls):
        result = cls.timing()
        control = dict(method="metal_command_buffer_timestamps_v1", scope="sum_of_command_buffer_gpu_intervals",
                       encoder_instrumentation=False, repetitions=4)
        for phase in ("throughput", "latency"):
            control[phase] = [dict(command_buffer_ns=s["command_buffer_ns"] / 2, command_buffers=s["command_buffers"])
                              for s in result[phase]]
        result["control"] = control
        return result

    @staticmethod
    def timing():
        sample = dict(compute_ns=2000., compute_span_ns=2100., command_buffer_ns=3000.,
                      calibration_cpu_ns=1e6, calibration_gpu_ticks=24000.,
                      compute_passes=1, command_buffers=1)
        return dict(method="metal_compute_pass_timestamps_v1", scope="sum_of_compute_encoder_gpu_intervals",
                    host_samples_instrumented=False, repetitions=4,
                    throughput=[sample.copy(), sample | {"compute_ns": 4000., "command_buffer_ns": 6000.}],
                    latency=[sample.copy(), sample.copy()])

    def test_gpu_denominator_is_independent_of_host_repetitions(self):
        result = dict(throughput_us=[10., 20.], latency_us=[100., 200.], repetitions=10000,
                      device_timing=self.timing())
        MODULE.summarize(result)
        self.assertEqual(result["throughput_us_p50"], 15.)
        self.assertEqual(result["device_timing"]["compute_throughput_us_p50"], .75)
        self.assertEqual(result["device_timing"]["compute_latency_us_p50"], 2.)
        self.assertEqual(result["device_timing"]["command_buffer_latency_us_p50"], 3.)

    def test_timing_method_instrumentation_scope_and_coverage_are_checked(self):
        for key, value in (("method", "host_clock"), ("scope", "single_kernel"),
                           ("host_samples_instrumented", True), ("repetitions", 0),
                           ("repetitions", True), ("repetitions", 65), ("latency", [])):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                MODULE.summarize_device_timing(self.timing() | {key: value}, 2)

    def test_invalid_gpu_samples_fail_closed(self):
        for key, value in (("compute_ns", 0), ("compute_ns", float("nan")),
                           ("compute_ns", 1e20), ("calibration_gpu_ticks", 0),
                           ("compute_passes", True), ("command_buffers", 1.5)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                result = copy.deepcopy(self.timing())
                result["throughput"][0][key] = value
                MODULE.summarize_device_timing(result, 2)

    def test_no_counter_control_keeps_distinct_scope_and_detects_probe_cost(self):
        result = self.controlled_timing()
        MODULE.summarize_device_timing(result, 2)
        self.assertEqual(result["control"]["command_buffer_throughput_us_p50"], .5625)
        self.assertEqual(result["control"]["command_buffer_latency_us_p50"], 1.5)
        self.assertEqual(result["counter_control_throughput_ratio"], 2.)
        for key, value in (("scope", "single_kernel"), ("encoder_instrumentation", True),
                           ("repetitions", True), ("repetitions", 8), ("throughput", [])):
            invalid = self.controlled_timing()
            invalid["control"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                MODULE.summarize_device_timing(invalid, 2)
        for key, value in (("command_buffer_ns", float("nan")), ("command_buffers", True)):
            invalid = self.controlled_timing()
            invalid["control"]["latency"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                MODULE.summarize_device_timing(invalid, 2)

    def test_control_report_includes_system_and_never_substitutes_probe_values(self):
        measured = dict(throughput_us=[10., 20.], latency_us=[100., 200.], device_timing=self.controlled_timing())
        MODULE.summarize(measured)
        lines = MODULE.device_control_report_lines([dict(name="example", valid=True, native=measured, torch=measured, system=measured)])
        for provider in ("native", "torch", "system"):
            self.assertIn(f"| example / {provider} | 0.562 | 1.500 | 15.000 | 150.000 | 2.000× |", lines)
        self.assertEqual(MODULE.device_control_report_lines([dict(name="bad", valid=False)]), [])

    def test_replay_gpu_report_uses_paired_ratios_and_withholds_missing_controls(self):
        def measurement(gpu, host):
            return dict(throughput_us_p50=host, latency_us_p50=host * 10,
                        device_timing=dict(control=dict(command_buffer_throughput_us_p50=gpu,
                                                        command_buffer_latency_us_p50=gpu + 1)))
        rows = [dict(backend="metal", name="example", round=i, variant=variant, valid=True,
                     native=measurement(value, 10), torch=measurement(5, 20))
                for i, values in enumerate(((8, 2), (80, 40)))
                for variant, value in zip(("reference", "candidate"), values)]
        report = dict(metadata=dict(rounds=2, metal_device_timing="helper"), results=rows)
        with patch.object(Path, "write_text") as write:
            REPEAT.write_report(report, Path("unused"))
        markdown = write.call_args_list[-1].args[0]
        self.assertIn("2/2 | 44.000 | 21.000 | 3.000× [2.000, 4.000] | 5.000", markdown)
        self.assertIn("| metal / example | 22.000 | 100.000 | 6.000 | 200.000 |", markdown)
        del rows[-1]["native"]["device_timing"]["control"]
        with patch.object(Path, "write_text") as write:
            REPEAT.write_report(report, Path("unused"))
        self.assertIn("1/2 | INCOMPLETE", write.call_args_list[-1].args[0])


class RepeatContractTests(unittest.TestCase):
    def test_cpu_model_checks_resolved_llvm_model_and_frozen_machine(self):
        MODULE.validate_cpu_target_policy({}, None, "cpu")
        generic = dict(cpu_target_policy="generic", cpu_model="generic")
        native = dict(cpu_target_policy="native", cpu_model="apple-m1")
        MODULE.validate_cpu_target_policy(generic, "generic", "cpu", b'"target-cpu"="generic"')
        MODULE.validate_cpu_target_policy(native, "native", "cpu", b'"target-cpu"="apple-m1"', "apple-m1")
        for row, policy, backend in (({}, "native", "cpu"), ({}, "generic", "cpu"),
                                     (native, None, "cpu"), (native, "native", "metal"),
                                     (generic, "native", "cpu"), (native, True, "cpu"),
                                     (native | {"cpu_model": "native"}, "native", "cpu"),
                                     (native | {"cpu_model": "generic"}, "native", "cpu")):
            with self.assertRaises(ValueError):
                MODULE.validate_cpu_target_policy(row, policy, backend)
        for source in (b'', b'"target-cpu"="generic"', b'"target-cpu"="apple-m1" "target-cpu"="generic"'):
            with self.assertRaisesRegex(ValueError, "LLVM CPU model"):
                MODULE.validate_cpu_target_policy(native, "native", "cpu", source)
        with self.assertRaisesRegex(ValueError, "frozen plan"):
            MODULE.validate_cpu_target_policy(native, "native", "cpu", expected_model="apple-m2")

    def test_optional_native_arguments_preserve_legacy_binaries_and_padding(self):
        self.assertEqual(MODULE.optional_native_arguments(argparse.Namespace()), [])
        old = dict(cpu_stack_bytes=8192, cpu_vector_lanes=64, cpu_input_views=True)
        expected = ["auto", "1", "tvm", "retain-subgroup-fences", "8192", "64", "forward-input-views"]
        self.assertEqual(MODULE.optional_native_arguments(argparse.Namespace(**old)), expected)
        self.assertEqual(MODULE.optional_native_arguments(argparse.Namespace(**old, cpu_model="native")), expected + ["native"])
        for model in ("generic", "native"):
            self.assertEqual(MODULE.optional_native_arguments(argparse.Namespace(cpu_model=model)),
                             ["auto", "1", "tvm", "retain-subgroup-fences", "0", "16", "retain-input-snapshots", model])
        self.assertEqual(MODULE.optional_native_arguments(argparse.Namespace(group_threads=128, copy_batch=4)), ["128", "4"])
        self.assertEqual(MODULE.optional_native_arguments(argparse.Namespace(cpu_matrix_backend="cblas")),
                         ["auto", "1", "tvm", "retain-subgroup-fences", "0", "16",
                          "retain-input-snapshots", "generic", "cblas"])
        self.assertEqual(MODULE.optional_native_arguments(argparse.Namespace(cpu_math_backend="accelerate")),
                         ["auto", "1", "tvm", "retain-subgroup-fences", "0", "16",
                          "retain-input-snapshots", "generic", "reference", "accelerate"])

    def test_element_reduction_mapping_options(self):
        prefix = ["auto", "1", "tvm", "retain-subgroup-fences", "0", "16",
                  "retain-input-snapshots", "generic", "reference", "reference", "preserve"]
        self.assertEqual(MODULE.optional_native_arguments(argparse.Namespace(reduction_programs_per_group=3)), prefix + ["3"])
        self.assertEqual(MODULE.optional_native_arguments(argparse.Namespace(element_grid="reference")), prefix + ["auto", "reference"])
        self.assertEqual(MODULE.optional_native_arguments(argparse.Namespace(reduction_unroll=4)), prefix + ["auto", "auto", "4"])
        self.assertEqual(MODULE.optional_native_arguments(argparse.Namespace(reduction_lane_elements=4)), prefix + ["auto", "auto", "1", "4"])
        self.assertEqual(MODULE.optional_native_arguments(argparse.Namespace(cache_reduction_inputs=True)), prefix + ["auto", "auto", "1", "1", "cache"])
        MODULE.validate_element_reduction_mapping({}, argparse.Namespace())
        args = argparse.Namespace(reduction_programs_per_group=3, element_grid="auto")
        native = {"reduction_programs_per_group": 3, "fuse_gpu_elementwise": True,
                  "execution_plans": [{"reduction_programs_per_group": 3}]}
        MODULE.validate_element_reduction_mapping(native, args)
        for change in ({"reduction_programs_per_group": 0}, {"fuse_gpu_elementwise": False},
                       {"execution_plans": []}, {"execution_plans": [{"reduction_programs_per_group": 1}]}):
            with self.assertRaises(ValueError):
                MODULE.validate_element_reduction_mapping(native | change, args)

    def test_lane_element_request_requires_realization(self):
        args = argparse.Namespace(reduction_lane_elements=4)
        native = {"reduction_lane_elements": 4, "execution_plans": [{"reduction_lane_elements": 4}]}
        MODULE.validate_element_reduction_mapping(native, args)
        for change in ({"reduction_lane_elements": True}, {"reduction_lane_elements": 1},
                       {"execution_plans": []}, {"execution_plans": [{}]}):
            with self.assertRaisesRegex(ValueError, "lane elements"):
                MODULE.validate_element_reduction_mapping(native | change, args)

    def test_reduction_input_cache_request_is_exact(self):
        args = argparse.Namespace(cache_reduction_inputs=True)
        MODULE.validate_element_reduction_mapping({"cache_reduction_inputs": True}, args)
        for value in (False, 1, None, "cache"):
            with self.assertRaisesRegex(ValueError, "input-cache"):
                MODULE.validate_element_reduction_mapping({"cache_reduction_inputs": value}, args)
        with self.assertRaisesRegex(ValueError, "input-cache"):
            MODULE.validate_element_reduction_mapping({}, args)

    def test_row_shapes_cover_independent_width_and_program_count(self):
        shapes = MODULE.parse_row_shapes("1x4096,1024x4096,1024x127")
        cases = MODULE.make_cases(["sum", "add"], True, shapes)
        self.assertEqual([(c.m, c.n) for c in cases], shapes * 2)
        for text in ("", "0x8", "8x", "1x2,1x2", "1x16385", "1x2x3"):
            with self.assertRaises(ValueError):
                MODULE.parse_row_shapes(text)

    def test_reduction_search_product_is_bounded(self):
        args = argparse.Namespace(tuning_candidates=[], gemm_block=(8, 8, 16), pipeline_window=2,
                                  packing_tuning_candidates=[0, 4], unroll_tuning_candidates=[1, 4],
                                  max_tuning_candidates=4)
        choices = MODULE.program_candidates(args)
        self.assertEqual([choice[-3:] for choice in choices], [(0, 1, 1), (0, 4, 1), (4, 1, 1), (4, 4, 1)])
        args.max_tuning_candidates = 3
        with self.assertRaisesRegex(ValueError, "budget"):
            MODULE.program_candidates(args)
        args.max_tuning_candidates = 8
        args.lane_tuning_candidates = [1, 4]
        self.assertEqual(len(MODULE.program_candidates(args)), 8)
        args.max_tuning_candidates = 7
        with self.assertRaisesRegex(ValueError, "budget"):
            MODULE.program_candidates(args)
        args.lane_tuning_candidates = [3]
        with self.assertRaisesRegex(ValueError, "lane elements"):
            MODULE.program_candidates(args)
        self.assertEqual(MODULE.reduction_candidates("0,4,0", 0, 8), [0, 4])
        self.assertEqual(MODULE.reduction_candidates(None, 1, 16), [])
        for value in ("", "0", "17", "1,,4", "-1", "x"):
            with self.assertRaises(ValueError):
                MODULE.reduction_candidates(value, 1, 16)

    def test_cpu_matrix_policy_requires_realized_external_call(self):
        MODULE.validate_cpu_matrix_policy({}, "reference", "cpu", "gemm")
        MODULE.validate_cpu_matrix_policy({"cpu_matrix_backend": "cblas", "external_matrix_calls": 1},
                                          "cblas", "cpu", "gemm")
        for row, requested, backend, operation in (
                ({}, "cblas", "cpu", "gemm"),
                ({"cpu_matrix_backend": "cblas", "external_matrix_calls": 2}, "cblas", "cpu", "gemm"),
                ({"cpu_matrix_backend": "reference", "external_matrix_calls": 1}, "reference", "cpu", "gemm"),
                ({"cpu_matrix_backend": "cblas", "external_matrix_calls": 1}, "cblas", "metal", "gemm"),
                ({"cpu_matrix_backend": "cblas", "external_matrix_calls": 1}, "cblas", "cpu", "add")):
            with self.assertRaises(ValueError):
                MODULE.validate_cpu_matrix_policy(row, requested, backend, operation)

    def test_cpu_math_policy_requires_realized_exp_provider(self):
        MODULE.validate_cpu_math_policy({}, "reference", "cpu", "softmax")
        MODULE.validate_cpu_math_policy({"cpu_math_backend": "accelerate", "external_vector_math_calls": 1},
                                        "accelerate", "cpu", "softmax")
        MODULE.validate_cpu_math_policy({"cpu_math_backend": "accelerate", "external_vector_math_calls": 1},
                                        "accelerate", "cpu", "sum")
        for row, requested, backend, operation in (
                ({}, "accelerate", "cpu", "softmax"),
                ({"cpu_math_backend": "reference", "external_vector_math_calls": 1}, "reference", "cpu", "softmax"),
                ({"cpu_math_backend": "accelerate", "external_vector_math_calls": 1}, "accelerate", "metal", "softmax"),
                ({"cpu_math_backend": "accelerate", "external_vector_math_calls": 0}, "accelerate", "cpu", "softmax")):
            with self.assertRaises(ValueError):
                MODULE.validate_cpu_math_policy(row, requested, backend, operation)

    def test_replay_retains_resolved_cpu_model_without_upgrading_legacy_plans(self):
        row = self.row()
        row["backend"] = "cpu"
        for policy, model in ((None, None), ("generic", "generic"), ("native", "apple-m1")):
            current = dict(row, native=dict(row["native"]))
            if policy is not None:
                current["native"].update(cpu_target_policy=policy, cpu_model=model)
            with patch.object(Path, "read_text", return_value=json.dumps({"results": [current]})):
                config = REPEAT.load_plan(Path("unused.json"), {"gemm"})["cpu", "gemm_17x19x13"]
            self.assertEqual(config["cpu_model"], policy)
            self.assertEqual(config["expected_cpu_model"], model)

    def test_replay_retains_cpu_math_backend(self):
        row = self.row()
        row["backend"] = "cpu"
        row["native"].update(cpu_math_backend="accelerate", external_vector_math_calls=0)
        with patch.object(Path, "read_text", return_value=json.dumps({"results": [row]})):
            config = REPEAT.load_plan(Path("unused.json"), {"gemm"})["cpu", "gemm_17x19x13"]
        self.assertEqual(config["cpu_math_backend"], "accelerate")

    def test_shared_vector_override_preserves_input_view_ablation(self):
        plans = [{("cpu", "case"): dict(auto_vectorize=True, no_vectorize=False, cpu_vector_lanes=16,
                                       cpu_input_views=forward)} for forward in (False, True)]
        for plan in plans:
            REPEAT.override_cpu_vector_lanes(plan, 64)
        self.assertEqual([p["cpu", "case"]["cpu_vector_lanes"] for p in plans], [64, 64])
        self.assertEqual([p["cpu", "case"]["cpu_input_views"] for p in plans], [False, True])
        original = dict(auto_vectorize=True, no_vectorize=False, cpu_vector_lanes=16)
        for backend, config, budget in (("cpu", original, True), ("cpu", original, 48), ("metal", original, 64),
                                       ("cpu", original | {"auto_vectorize": False}, 64),
                                       ("cpu", original | {"no_vectorize": True}, 64)):
            plan = {("cpu", "first"): dict(original), (backend, "invalid"): dict(config)}
            with self.assertRaises(ValueError):
                REPEAT.override_cpu_vector_lanes(plan, budget)
            self.assertEqual(plan["cpu", "first"], original)

    def test_cpu_input_views_are_not_mpp_or_an_implicit_reference_policy(self):
        native = dict(backend="cpu", metal_mpp=False, forward_readonly_tile_loads=True)
        MODULE.validate_tirx_realization(native, "simdgroup", True)
        for invalid in (native | {"backend": "metal"}, native | {"metal_mpp": True},
                        native | {"forward_readonly_tile_loads": False}, native | {"forward_readonly_tile_loads": 1}):
            with self.assertRaises(ValueError):
                MODULE.validate_tirx_realization(invalid, "simdgroup", True)
        for realization in ("mpp", "mpp-views"):
            with self.assertRaises(ValueError):
                MODULE.validate_tirx_realization(native, realization, True)
        with self.assertRaises(ValueError):
            MODULE.validate_tirx_realization(native, "simdgroup")
        with self.assertRaises(ValueError):
            MODULE.validate_tirx_realization(native, "simdgroup", 1)

    def test_replay_preserves_cpu_views_separately_from_matrix_realization(self):
        row = self.row()
        row["backend"] = "cpu"
        row["native"].update(forward_readonly_tile_loads=True, metal_mpp=False)
        with patch.object(Path, "read_text", return_value=json.dumps({"results": [row]})):
            config = REPEAT.load_plan(Path("unused.json"), {"gemm"})["cpu", "gemm_17x19x13"]
        self.assertTrue(config["cpu_input_views"])
        self.assertEqual(config["matrix_realization"], "simdgroup")
        for key, value in (("forward_readonly_tile_loads", 1), ("forward_readonly_tile_loads", None), ("metal_mpp", True)):
            invalid = dict(row, native=row["native"] | {key: value})
            with patch.object(Path, "read_text", return_value=json.dumps({"results": [invalid]})):
                with self.assertRaises(ValueError):
                    REPEAT.load_plan(Path("unused.json"), {"gemm"})

    def test_generic_input_views_are_explicit_and_replayed_on_metal(self):
        native = dict(backend="metal", metal_mpp=False, forward_readonly_tile_loads=True)
        MODULE.validate_tirx_realization(native, "simdgroup", input_views=True)
        for invalid in (native | {"metal_mpp": True}, native | {"forward_readonly_tile_loads": 1},
                        native | {"backend": "cuda"}):
            with self.assertRaises(ValueError):
                MODULE.validate_tirx_realization(invalid, "simdgroup", input_views=True)
        with self.assertRaises(ValueError):
            MODULE.validate_tirx_realization(native, "simdgroup")
        self.assertEqual(MODULE.optional_native_arguments(argparse.Namespace(input_views=True))[-1], "forward-input-views")
        row = self.row()
        row["backend"] = "metal"
        row["native"].update(native)
        with patch.object(Path, "read_text", return_value=json.dumps({"results": [row]})):
            config = REPEAT.load_plan(Path("unused.json"), {"gemm"})["metal", "gemm_17x19x13"]
        self.assertTrue(config["input_views"])
        self.assertFalse(config["cpu_input_views"])
        self.assertEqual(config["matrix_realization"], "simdgroup")

    def test_cpu_vector_budget_must_match_and_preserve_legacy_default(self):
        MODULE.validate_cpu_vector_policy({}, 16)
        for budget in (16, 32, 64, 128):
            MODULE.validate_cpu_vector_policy({"cpu_vector_lanes": budget}, budget)
        for actual in (None, False, 16, 256, "64"):
            with self.assertRaisesRegex(ValueError, "CPU vector lanes"):
                MODULE.validate_cpu_vector_policy({"cpu_vector_lanes": actual}, 64)
        for requested in (None, False, 0, 8, 48, 129):
            with self.assertRaises(ValueError):
                MODULE.validate_cpu_vector_policy({"cpu_vector_lanes": requested}, requested)

    def test_replay_requires_cpu_auto_vectorization_for_cartesian_packs(self):
        row = self.row()
        row["backend"] = "cpu"
        row["native"].update(vectorize=True, auto_vectorize=True)
        for budget in (None, 16, 32, 64, 128):
            if budget is not None:
                row["native"]["cpu_vector_lanes"] = budget
            with patch.object(Path, "read_text", return_value=json.dumps({"results": [row]})):
                config = REPEAT.load_plan(Path("unused.json"), {"gemm"})["cpu", "gemm_17x19x13"]
            self.assertEqual(config["cpu_vector_lanes"], budget or 16)
        for backend, auto, vectorize, budget in (("metal", True, True, 64), ("cpu", False, True, 64),
                                                ("cpu", True, False, 64), ("cpu", True, True, 48),
                                                ("cpu", True, True, True), ("cpu", True, True, None)):
            row["backend"] = backend
            row["native"].update(auto_vectorize=auto, vectorize=vectorize, cpu_vector_lanes=budget)
            with patch.object(Path, "read_text", return_value=json.dumps({"results": [row]})):
                with self.assertRaisesRegex(ValueError, "CPU vector-lane"):
                    REPEAT.load_plan(Path("unused.json"), {"gemm"})

    def test_cpu_stack_budget_must_be_reported_exactly(self):
        MODULE.validate_cpu_storage_policy({}, 0)
        MODULE.validate_cpu_storage_policy({"cpu_stack_bytes": 8192}, 8192)
        for actual in (None, False, 1, 65537, "8192"):
            with self.assertRaisesRegex(ValueError, "CPU stack budget"):
                MODULE.validate_cpu_storage_policy({"cpu_stack_bytes": actual}, 8192)
        with self.assertRaises(ValueError):
            MODULE.validate_cpu_storage_policy({}, 8192)
        with self.assertRaises(ValueError):
            MODULE.validate_cpu_storage_policy({}, False)

    def test_replay_preserves_cpu_stack_budget_and_legacy_zero(self):
        row = self.row()
        row["backend"] = "cpu"
        for budget in (None, 0, 8192):
            if budget is not None:
                row["native"]["cpu_stack_bytes"] = budget
            with patch.object(Path, "read_text", return_value=json.dumps({"results": [row]})):
                config = REPEAT.load_plan(Path("unused.json"), {"gemm"})["cpu", "gemm_17x19x13"]
            self.assertEqual(config["cpu_stack_bytes"], budget or 0)
        for backend, budget in (("metal", 8192), ("cpu", True), ("cpu", -1), ("cpu", 65537), ("cpu", None)):
            row["backend"] = backend
            row["native"]["cpu_stack_bytes"] = budget
            with patch.object(Path, "read_text", return_value=json.dumps({"results": [row]})):
                with self.assertRaisesRegex(ValueError, "CPU stack budget"):
                    REPEAT.load_plan(Path("unused.json"), {"gemm"})

    def test_subgroup_fence_policy_requires_a_realization_proof(self):
        MODULE.validate_subgroup_policy({}, False)  # legacy default
        isolated = dict(elide_independent_subgroup_barriers=True,
                        execution_plans=[dict(independent_subgroups=True, group_barrier_sites_before=2, group_barrier_sites_after=0)])
        MODULE.validate_subgroup_policy(isolated, True)
        MODULE.validate_subgroup_policy(isolated | dict(execution_plans=[dict(independent_subgroups=False, group_barrier_sites_before=6, group_barrier_sites_after=5)]), True)
        for bad in ({}, isolated | dict(elide_independent_subgroup_barriers=1),
                    isolated | dict(execution_plans=[]), isolated | dict(execution_plans=[{}]),
                    isolated | dict(execution_plans=[dict(independent_subgroups=True, group_barrier_sites_before=2, group_barrier_sites_after=1)]),
                    isolated | dict(execution_plans=[dict(independent_subgroups=True, group_barrier_sites_before=2, group_barrier_sites_after=False)]),
                    isolated | dict(execution_plans=[dict(independent_subgroups=False, group_barrier_sites_before=2, group_barrier_sites_after=3)])):
            with self.assertRaises(ValueError):
                MODULE.validate_subgroup_policy(bad, True)
        with self.assertRaises(ValueError):
            MODULE.validate_subgroup_policy(isolated, False)

    def test_replay_preserves_subgroup_fence_policy(self):
        row = self.row()
        row["native"].update(metal_mpp=True, forward_readonly_tile_loads=True, elide_independent_subgroup_barriers=True)
        with patch.object(Path, "read_text", return_value=json.dumps({"results": [row]})):
            config = REPEAT.load_plan(Path("unused.json"), {"gemm"})["metal", "gemm_17x19x13"]
        self.assertIs(config["elide_independent_subgroup_barriers"], True)
        for changes in (dict(elide_independent_subgroup_barriers=1), dict(elide_independent_subgroup_barriers=None),
                        dict(forward_readonly_tile_loads=False)):
            bad = row | dict(native=row["native"] | changes)
            with patch.object(Path, "read_text", return_value=json.dumps({"results": [bad]})):
                with self.assertRaisesRegex(ValueError, "subgroup-fence"):
                    REPEAT.load_plan(Path("unused.json"), {"gemm"})

    def test_fingerprints_cover_both_runtimes_and_extra_compiler(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binaries = []
            for variant in ("before", "after"):
                folder = root / variant
                folder.mkdir()
                binary = folder / "benchmark"
                binary.write_bytes(b"same executable")
                binaries.append(binary)
                (folder / "runtime.dylib").write_bytes(variant.encode())
                (folder / "unrelated.txt").write_bytes(b"not an artifact")
            compiler = root / "compiler.dylib"
            compiler.write_bytes(b"compiler")
            hashes = REPEAT.artifact_hashes(binaries, [compiler])
            self.assertEqual(len(hashes), 5)
            self.assertEqual(hashes[str(compiler.resolve())], hashlib.sha256(b"compiler").hexdigest())
            runtime = binaries[1].parent / "runtime.dylib"
            runtime.write_bytes(b"changed without relinking executable")
            self.assertNotEqual(hashes, REPEAT.artifact_hashes(binaries, [compiler]))

    def test_fingerprints_deduplicate_symlinks_and_reject_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "benchmark"
            binary.write_bytes(b"executable")
            library = root / "runtime.1.dylib"
            library.write_bytes(b"runtime")
            (root / "runtime.dylib").symlink_to(library.name)
            self.assertEqual(len(REPEAT.artifact_hashes([binary, binary], [library])), 2)
            with self.assertRaises(FileNotFoundError):
                REPEAT.artifact_hashes([binary], [root / "missing.dylib"])

    def test_variant_and_framework_orders_are_both_balanced(self):
        keys = [("metal", str(i)) for i in range(8)]
        rounds = [list(REPEAT.order_for_round(keys, i)) for i in range(4)]
        for key in keys:
            first_variants = [next(v for k, v, _ in r if k == key) for r in rounds]
            self.assertEqual(first_variants.count("candidate"), 2)
            for variant in ("reference", "candidate"):
                orders = [order for r in rounds for k, v, order in r if (k, v) == (key, variant)]
                self.assertEqual(sorted(orders), [0, 0, 1, 1])
        self.assertEqual([r[0][0] for r in rounds], keys[:4])

    @staticmethod
    def row():
        return {"case": {"operation": "gemm", "m": 17, "n": 19, "k": 13}, "backend": "metal",
                "valid": True, "block": [16, 32, 32], "native": {
                    "execution_scope": "group", "pipeline_window": 1, "cooperative_matrix": True,
                    "vectorize": True, "auto_vectorize": False, "throughput_us_p50": 1.0,
                    "shared_tile_materialization": "preserve"}}

    def test_plan_uses_recorded_configuration_but_not_recorded_score(self):
        row = self.row()
        with patch.object(Path, "read_text", return_value=json.dumps({"results": [row]})):
            plan = REPEAT.load_plan(Path("unused.json"), {"gemm"})
        config = plan["metal", "gemm_17x19x13"]
        self.assertEqual(config["gemm_block"], (16, 32, 32))
        self.assertEqual(config["pipeline_window"], 1)
        self.assertNotIn("throughput_us_p50", config)
        self.assertEqual(config["group_threads"], 0)
        self.assertEqual(config["copy_batch"], 1)

    def test_replay_preserves_copy_batch_policy(self):
        row = self.row()
        row["native"]["copy_batch"] = 4
        with patch.object(Path, "read_text", return_value=json.dumps({"results": [row]})):
            config = REPEAT.load_plan(Path("unused.json"), {"gemm"})["metal", "gemm_17x19x13"]
        self.assertEqual(config["copy_batch"], 4)
        for invalid in (True, 0, 17, None):
            row["native"]["copy_batch"] = invalid
            with patch.object(Path, "read_text", return_value=json.dumps({"results": [row]})):
                with self.assertRaisesRegex(ValueError, "copy-batch"):
                    REPEAT.load_plan(Path("unused.json"), {"gemm"})

    def test_replay_preserves_exact_group_thread_constraint(self):
        row = self.row()
        row["native"]["planner_threads"] = 128
        with patch.object(Path, "read_text", return_value=json.dumps({"results": [row]})):
            config = REPEAT.load_plan(Path("unused.json"), {"gemm"})["metal", "gemm_17x19x13"]
        self.assertEqual(config["group_threads"], 128)
        for invalid in (True, -1, 0x100000000, None):
            row["native"]["planner_threads"] = invalid
            with patch.object(Path, "read_text", return_value=json.dumps({"results": [row]})):
                with self.assertRaisesRegex(ValueError, "group-thread"):
                    REPEAT.load_plan(Path("unused.json"), {"gemm"})

    def test_replay_preserves_metal_subgroup_reduction_policy(self):
        row = self.row()
        row["case"] = {"operation": "softmax", "m": 64, "n": 4096, "k": 1}
        row["block"] = [1, 4096, 1]
        row["native"] |= {
            "execution_scope": "auto", "cooperative_matrix": False,
            "metal_subgroup_reductions": True, "metal_mpp": False,
            "forward_readonly_tile_loads": True,
            "execution_plans": [{"optimized": True, "threads": 256,
                                 "independent_subgroups": False}],
        }
        with patch.object(Path, "read_text", return_value=json.dumps({"results": [row]})):
            config = REPEAT.load_plan(Path("unused.json"), {"softmax"})["metal", "softmax_64x4096"]
        self.assertIs(config["metal_subgroup_reductions"], True)
        self.assertEqual(config["matrix_realization"], "simdgroup")
        self.assertIs(config["cpu_input_views"], False)
        self.assertEqual(config["reduction_lane_elements"], 1)
        self.assertIs(config["cache_reduction_inputs"], False)
        cached = row | {"native": row["native"] | {"cache_reduction_inputs": True}}
        with patch.object(Path, "read_text", return_value=json.dumps({"results": [cached]})):
            config = REPEAT.load_plan(Path("unused.json"), {"softmax"})["metal", "softmax_64x4096"]
        self.assertIs(config["cache_reduction_inputs"], True)
        for value in (1, None, "cache"):
            invalid = row | {"native": row["native"] | {"cache_reduction_inputs": value}}
            with patch.object(Path, "read_text", return_value=json.dumps({"results": [invalid]})):
                with self.assertRaisesRegex(ValueError, "input-cache"):
                    REPEAT.load_plan(Path("unused.json"), {"softmax"})
        packed = copy.deepcopy(row)
        packed["native"]["reduction_lane_elements"] = 4
        packed["native"]["execution_plans"][0]["reduction_lane_elements"] = 4
        with patch.object(Path, "read_text", return_value=json.dumps({"results": [packed]})):
            config = REPEAT.load_plan(Path("unused.json"), {"softmax"})["metal", "softmax_64x4096"]
        self.assertEqual(config["reduction_lane_elements"], 4)
        for bad_width in (True, 3, 8):
            bad = packed | {"native": packed["native"] | {"reduction_lane_elements": bad_width}}
            with patch.object(Path, "read_text", return_value=json.dumps({"results": [bad]})):
                with self.assertRaisesRegex(ValueError, "lane elements"):
                    REPEAT.load_plan(Path("unused.json"), {"softmax"})
        layernorm = row | {
            "case": {"operation": "layernorm", "m": 64, "n": 4096, "k": 1}}
        with patch.object(Path, "read_text", return_value=json.dumps({"results": [layernorm]})):
            config = REPEAT.load_plan(Path("unused.json"), {"layernorm"})[
                "metal", "layernorm_64x4096"]
        self.assertIs(config["metal_subgroup_reductions"], True)
        for changes in ({"metal_subgroup_reductions": 1},
                        {"forward_readonly_tile_loads": False},
                        {"metal_mpp": True},
                        {"execution_plans": [{"optimized": False, "threads": 256}]}):
            bad = row | {"native": row["native"] | changes}
            with patch.object(Path, "read_text", return_value=json.dumps({"results": [bad]})):
                with self.assertRaisesRegex(ValueError, "subgroup-reduction"):
                    REPEAT.load_plan(Path("unused.json"), {"softmax"})

    def test_plan_does_not_guess_historical_vectorization_semantics(self):
        row = self.row()
        del row["native"]["auto_vectorize"]
        with patch.object(Path, "read_text", return_value=json.dumps({"results": [row]})):
            with self.assertRaisesRegex(ValueError, "auto_vectorize"):
                REPEAT.load_plan(Path("unused.json"), {"gemm"})

    def test_plan_rejects_invalid_duplicate_and_empty_selections(self):
        for rows in ([self.row(), self.row()], [self.row() | {"valid": False}], []):
            with self.subTest(rows=rows), patch.object(Path, "read_text", return_value=json.dumps({"results": rows})):
                with self.assertRaises(ValueError):
                    REPEAT.load_plan(Path("unused.json"), {"gemm"})


class BenchmarkContractTests(unittest.TestCase):
    def test_gelu_add_uses_elementwise_shapes_and_blocks(self):
        cases = MODULE.make_cases(["gelu_add"])
        self.assertEqual([(c.m, c.n) for c in cases], [(1, 127), (17, 257), (128, 1024), (4096, 256)])
        self.assertTrue(all(MODULE.block_shape(c, (8, 8, 8)) == (1, 256, 1) for c in cases))
        self.assertEqual(MODULE.tolerance("gelu_add"), (2e-6, 2e-5))

    def test_shape_matrix_is_not_only_squares(self):
        cases = MODULE.make_cases(["gemm"])
        self.assertEqual(len(cases), 8)
        self.assertTrue(any(c.m == c.n == c.k == 1024 for c in cases))
        self.assertTrue(any(c.m > c.n for c in cases))
        self.assertTrue(any(c.m < c.n for c in cases))
        self.assertTrue(any(c.m % 8 and c.n % 8 and c.k % 16 for c in cases))

    def test_reduction_sizes_include_wide_and_single_row(self):
        cases = MODULE.make_cases(["sum", "softmax", "rmsnorm", "layernorm", "residual_layernorm", "cross_entropy"])
        self.assertTrue(any(c.m == 1 for c in cases))
        self.assertTrue(any(c.n == 4096 for c in cases))
        self.assertEqual(sum(c.operation == "rmsnorm" for c in cases), 4)
        self.assertEqual(sum(c.operation == "layernorm" for c in cases), 4)
        self.assertEqual(sum(c.operation == "residual_layernorm" for c in cases), 4)
        self.assertEqual(sum(c.operation == "cross_entropy" for c in cases), 4)

    def test_percentiles(self):
        self.assertEqual(MODULE.percentile([9, 1, 5], 0.5), 5)
        self.assertAlmostEqual(MODULE.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9], 0.9), 8.2)
        with self.assertRaises(ValueError):
            MODULE.percentile([float("nan")], 0.5)

    def test_tolerances_are_shared_and_add_is_exact(self):
        self.assertEqual(MODULE.tolerance("add"), (0.0, 0.0))
        self.assertEqual(MODULE.tolerance("gemm"), (1e-4, 1e-4))
        self.assertEqual(MODULE.tolerance("layernorm"), (1e-5, 2e-5))
        self.assertEqual(MODULE.tolerance("residual_layernorm"), (1e-5, 2e-5))

    def test_shared_tile_materialization_is_explicit_and_padded(self):
        self.assertEqual(
            MODULE.optional_native_arguments(
                argparse.Namespace(shared_tile_materialization="expensive-only")),
            ["auto", "1", "tvm", "retain-subgroup-fences", "0", "16",
             "retain-input-snapshots", "generic", "reference", "reference",
             "expensive-only"])
        MODULE.validate_shared_tile_materialization(
            {"shared_tile_materialization": "preserve"}, "preserve")
        for native, requested in (({}, "preserve"),
                                  ({"shared_tile_materialization": "preserve"}, "expensive-only"),
                                  ({"shared_tile_materialization": "preserve"}, "unknown")):
            with self.assertRaises(ValueError):
                MODULE.validate_shared_tile_materialization(native, requested)

    def test_unknown_operation_is_not_silently_skipped(self):
        with self.assertRaises(ValueError):
            MODULE.make_cases(["gemm", "unknown"])

    def test_native_mapping_metadata_matches_the_request(self):
        case = MODULE.Case("gemm", 7, 17, 37)
        native = dict(backend="metal", operation="gemm", execution_scope="group", pipeline_window=2, mma_operations=1,
                      cooperative_matrix=False, matrix_intrinsics=0, vectorize=True, auto_vectorize=False)
        MODULE.validate_native_metadata(native, case, "metal", "group")
        for key, value in (("backend", "cpu"), ("operation", "add"), ("execution_scope", "worker"), ("mma_operations", 0)):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                MODULE.validate_native_metadata(native | {key: value}, case, "metal", "group")
        del native["execution_scope"]
        with self.assertRaisesRegex(RuntimeError, "execution-scope"):
            MODULE.validate_native_metadata(native, case, "metal", "group")

    def test_native_pipeline_window_matches_the_request(self):
        case = MODULE.Case("gemm", 7, 17, 37)
        native = dict(backend="metal", operation="gemm", execution_scope="worker", mma_operations=1,
                      cooperative_matrix=False, matrix_intrinsics=0, vectorize=True, auto_vectorize=False)
        for window in (1, 2):
            MODULE.validate_native_metadata(native | {"pipeline_window": window}, case, "metal", "worker", window)
            with self.assertRaisesRegex(RuntimeError, "pipeline-window"):
                MODULE.validate_native_metadata(native | {"pipeline_window": 3 - window}, case, "metal", "worker", window)
        with self.assertRaisesRegex(RuntimeError, "pipeline-window"):
            MODULE.validate_native_metadata(native, case, "metal", "worker")

    def test_group_thread_constraint_and_realization_are_both_checked(self):
        case = MODULE.Case("gemm", 17, 19, 13)
        for threads in (31, 48, 64, 128, 256):
            native = dict(backend="metal", operation="gemm", execution_scope="group", pipeline_window=2,
                          cooperative_matrix=True, matrix_intrinsics=int(threads % 32 == 0), mma_operations=1,
                          vectorize=True, auto_vectorize=False, planner_threads=threads, execution_plans=[{"threads": threads}])
            arguments = dict(cooperative_matrix=True, group_threads=threads)
            MODULE.validate_native_metadata(native, case, "metal", "group", **arguments)
            with self.assertRaisesRegex(RuntimeError, "group-thread"):
                MODULE.validate_native_metadata(native | {"planner_threads": threads + 1}, case, "metal", "group", **arguments)
            for plans in ([], None, [{"threads": threads + 1}]):
                with self.assertRaisesRegex(RuntimeError, "realized group threads"):
                    MODULE.validate_native_metadata(native | {"execution_plans": plans}, case, "metal", "group", **arguments)

    def test_subgroup_reduction_metadata_accepts_cooperating_subgroups(self):
        case = MODULE.Case("rmsnorm", 128, 1024)
        native = dict(backend="metal", operation="rmsnorm", execution_scope="auto", pipeline_window=2,
                      cooperative_matrix=False, matrix_intrinsics=0, vectorize=True, auto_vectorize=False,
                      metal_subgroup_reductions=True, execution_plans=[{
                          "optimized": True, "threads": 128, "independent_subgroups": False}])
        MODULE.validate_native_metadata(native, case, "metal", "auto", metal_subgroup_reductions=True)
        for plans in ([], None, [{"optimized": False, "threads": 128, "independent_subgroups": False}],
                      [{"optimized": True, "threads": 48, "independent_subgroups": False}],
                      [{"optimized": True, "threads": 128, "independent_subgroups": "false"}]):
            with self.assertRaisesRegex(RuntimeError, "SIMD-group reduction"):
                MODULE.validate_native_metadata(native | {"execution_plans": plans}, case, "metal", "auto",
                                                metal_subgroup_reductions=True)

    def test_reduction_device_limit_and_ownership_features(self):
        case = MODULE.Case("rmsnorm", 17, 257)
        plan = dict(optimized=True, threads=96, independent_subgroups=False,
                    programs=17, reduction_programs_per_group=1, reduction_subgroups_per_program=3,
                    reduction_threadgroups=17, reduction_scalar_rounds=8.0, reduction_lane_utilization=257 / 384)
        native = dict(backend="metal", operation="rmsnorm", execution_scope="auto", pipeline_window=2,
                      cooperative_matrix=False, matrix_intrinsics=0, vectorize=True, auto_vectorize=False,
                      metal_subgroup_reductions=True, metal_max_threads=1024, execution_plans=[plan])
        MODULE.validate_native_metadata(native, case, "metal", "auto", metal_subgroup_reductions=True)
        for field, value in (("reduction_threadgroups", 3), ("reduction_threadgroups", True),
                             ("reduction_programs_per_group", 2), ("reduction_subgroups_per_program", 2),
                             ("threads", 1056), ("reduction_scalar_rounds", float("nan")),
                             ("reduction_lane_utilization", 1.1), ("reduction_lane_utilization", False)):
            with self.assertRaisesRegex(RuntimeError, "ownership features"):
                MODULE.validate_native_metadata(native | {"execution_plans": [plan | {field: value}]}, case,
                                                "metal", "auto", metal_subgroup_reductions=True)
        for limit in (None, True, 16):
            with self.assertRaisesRegex(RuntimeError, "device thread limit"):
                MODULE.validate_native_metadata(native | {"metal_max_threads": limit}, case,
                                                "metal", "auto", metal_subgroup_reductions=True)
        with self.assertRaisesRegex(RuntimeError, "ownership features"):
            MODULE.validate_native_metadata(native | {"metal_max_threads": 64}, case,
                                            "metal", "auto", metal_subgroup_reductions=True)

    def test_copy_batch_request_and_plan_are_both_checked(self):
        case = MODULE.Case("gemm", 17, 19, 13)
        native = dict(backend="metal", operation="gemm", execution_scope="group", pipeline_window=2,
                      cooperative_matrix=False, matrix_intrinsics=0, mma_operations=1,
                      vectorize=True, auto_vectorize=False, copy_batch=4, execution_plans=[{"max_copy_batch": 4}])
        MODULE.validate_native_metadata(native, case, "metal", "group", copy_batch=4)
        for policy in (True, 1, None):
            with self.assertRaisesRegex(RuntimeError, "copy-batch policy"):
                MODULE.validate_native_metadata(native | {"copy_batch": policy}, case, "metal", "group", copy_batch=4)
        for plans in ([], None, [{"max_copy_batch": 1}]):
            with self.assertRaisesRegex(RuntimeError, "copy-batch plan"):
                MODULE.validate_native_metadata(native | {"execution_plans": plans}, case, "metal", "group", copy_batch=4)

    def test_matrix_capability_is_not_confused_with_emitted_instructions(self):
        case = MODULE.Case("gemm", 7, 17, 37)
        for enabled in (False, True):
            for backend, scope, block in (("cpu", "worker", (8, 8, 16)),
                                          ("metal", "worker", (8, 8, 16)),
                                          ("metal", "group", (3, 5, 7)),
                                          ("metal", "group", (8, 8, 16))):
                calls = int(enabled and backend == "metal" and scope == "group" and block == (8, 8, 16))
                native = dict(backend=backend, operation="gemm", execution_scope=scope, pipeline_window=2,
                              mma_operations=1, cooperative_matrix=enabled, matrix_intrinsics=calls, vectorize=True, auto_vectorize=False)
                arguments = (case, backend, scope, 2, enabled, block)
                with self.subTest(enabled=enabled, backend=backend, scope=scope, block=block):
                    MODULE.validate_native_metadata(native, *arguments)
                    with self.assertRaisesRegex(RuntimeError, "cooperative-matrix"):
                        MODULE.validate_native_metadata(native | {"cooperative_matrix": not enabled}, *arguments)
                    with self.assertRaisesRegex(RuntimeError, "matrix-intrinsic"):
                        MODULE.validate_native_metadata(native | {"matrix_intrinsics": 1 - calls}, *arguments)
                    for invalid in (None, -1, 1.5, True):
                        with self.assertRaisesRegex(RuntimeError, "matrix-intrinsic"):
                            MODULE.validate_native_metadata(native | {"matrix_intrinsics": invalid}, *arguments)

    def test_native_vectorization_metadata_matches_the_request(self):
        case = MODULE.Case("gemm", 7, 17, 37)
        native = dict(backend="cpu", operation="gemm", execution_scope="worker", pipeline_window=2,
                      cooperative_matrix=False, matrix_intrinsics=0, mma_operations=1, auto_vectorize=False)
        for enabled in (False, True):
            MODULE.validate_native_metadata(native | {"vectorize": enabled}, case, "cpu", "worker", vectorize=enabled)
            for invalid in (None, int(enabled), not enabled):
                with self.subTest(enabled=enabled, invalid=invalid), self.assertRaisesRegex(RuntimeError, "vectorization"):
                    MODULE.validate_native_metadata(native | {"vectorize": invalid}, case, "cpu", "worker", vectorize=enabled)

    def test_automatic_vectorization_is_a_separate_explicit_option(self):
        case = MODULE.Case("gemm", 7, 17, 37)
        native = dict(backend="cpu", operation="gemm", execution_scope="worker", pipeline_window=2,
                      cooperative_matrix=False, matrix_intrinsics=0, mma_operations=1, vectorize=True)
        for enabled in (False, True):
            MODULE.validate_native_metadata(native | {"auto_vectorize": enabled}, case, "cpu", "worker", auto_vectorize=enabled)
            for invalid in (None, int(enabled), not enabled):
                with self.subTest(enabled=enabled, invalid=invalid), self.assertRaisesRegex(RuntimeError, "automatic-vectorization"):
                    MODULE.validate_native_metadata(native | {"auto_vectorize": invalid}, case, "cpu", "worker", auto_vectorize=enabled)

    def test_jit_candidates_are_explicit_and_deduplicated(self):
        self.assertEqual(MODULE.tuning_candidates((8, 8, 16), 2, None, None), [])
        self.assertEqual(MODULE.tuning_candidates((8, 8, 16), 2, "8,8,16;16,32,32;8,8,16", "1,2,1"),
                         [((8, 8, 16), 1), ((8, 8, 16), 2), ((16, 32, 32), 1), ((16, 32, 32), 2)])
        self.assertEqual(MODULE.tuning_candidates((8, 8, 16), 2, None, "1"), [((8, 8, 16), 1)])
        for block in ("", "8,8", "0,8,16", "8,8,16,32", "8,a,16"):
            with self.subTest(block=block), self.assertRaises(ValueError):
                MODULE.parse_gemm_block(block)
        with self.assertRaises(ValueError):
            MODULE.tuning_candidates((8, 8, 16), 2, None, "0,2")
        self.assertEqual(
            MODULE.materialization_candidates(
                "preserve", "preserve,expensive-only,preserve"),
            ["preserve", "expensive-only"])
        self.assertEqual(
            MODULE.materialization_candidates("preserve", None), [])
        for value in ("", "unknown", "preserve,unknown"):
            with self.assertRaises(ValueError):
                MODULE.materialization_candidates("preserve", value)

    def test_jit_search_revalidates_winner_instead_of_publishing_minimum(self):
        args = argparse.Namespace(tuning_candidates=[((8, 8, 16), 2), ((16, 32, 32), 1), ((64, 64, 64), 2)],
                                  gemm_block=(8, 8, 16), pipeline_window=2)
        calls = []
        ordinals = []

        def measure(torch, np, candidate, case, backend, ordinal):
            calls.append((candidate.gemm_block, candidate.pipeline_window))
            ordinals.append(ordinal)
            if candidate.gemm_block == (64, 64, 64):
                raise RuntimeError("capacity")
            score = 100.0 if len(calls) == 4 else (2.0 if candidate.gemm_block == (8, 8, 16) else 1.0)
            return {"valid": True, "native": {"throughput_us_p50": score}}

        with patch.object(MODULE, "run_case", side_effect=measure), contextlib.redirect_stdout(io.StringIO()):
            result = MODULE.run_tuned_case(None, None, args, MODULE.Case("gemm", 32, 32, 32), "cpu", 1)
        self.assertEqual(calls[0], ((16, 32, 32), 1))  # Rotated trial order.
        self.assertEqual(calls[-1], ((16, 32, 32), 1))
        self.assertEqual(len(calls), 4)  # Includes a fresh recapture/JIT after selection.
        self.assertEqual(ordinals[-1], 1)  # Fresh comparison preserves alternating framework order.
        self.assertEqual(result["native"]["throughput_us_p50"], 100.0)
        self.assertEqual(sum(trial["valid"] for trial in result["tuning"]["trials"]), 2)
        self.assertEqual(args.gemm_block, (8, 8, 16))  # No mutation of caller configuration.

    def test_mapping_candidates_are_bounded_deduplicated_and_opt_in(self):
        self.assertEqual(MODULE.mapping_candidates(128, 4, None, None), [])
        self.assertEqual(MODULE.mapping_candidates(128, 4, "0,256,0", "1,4,1"),
                         [(0, 1), (0, 4), (256, 1), (256, 4)])
        self.assertEqual(MODULE.mapping_candidates(128, 4, None, "8"), [(128, 8)])
        self.assertEqual(MODULE.mapping_candidates(128, 4, "64", None), [(64, 4)])
        for value in ("", "auto", "-1", "4294967296"):
            with self.subTest(threads=value), self.assertRaises(ValueError):
                MODULE.mapping_candidates(128, 4, value, None)
        for value in ("", "0", "17", "1.5"):
            with self.subTest(batch=value), self.assertRaises(ValueError):
                MODULE.mapping_candidates(128, 4, None, value)

    def test_joint_candidate_product_and_compile_budget(self):
        args = argparse.Namespace(tuning_candidates=[((32, 64, 32), 1), ((64, 64, 32), 2)],
                                  mapping_tuning_candidates=[(128, 4), (256, 8)], max_tuning_candidates=4)
        self.assertEqual(MODULE.joint_candidates(args), [((32, 64, 32), 1, 128, 4, "preserve"), ((32, 64, 32), 1, 256, 8, "preserve"),
                                                       ((64, 64, 32), 2, 128, 4, "preserve"), ((64, 64, 32), 2, 256, 8, "preserve")])
        for budget in (0, 3):
            args.max_tuning_candidates = budget
            with self.assertRaisesRegex(ValueError, "budget"):
                MODULE.joint_candidates(args)
        args = argparse.Namespace(tuning_candidates=[], gemm_block=(16, 32, 32), pipeline_window=1,
                                  mapping_tuning_candidates=[(128, 4), (256, 4)])
        self.assertEqual(MODULE.joint_candidates(args), [((16, 32, 32), 1, 128, 4, "preserve"), ((16, 32, 32), 1, 256, 4, "preserve")])

    def test_materialization_candidates_join_the_jit_product(self):
        args = argparse.Namespace(
            tuning_candidates=[], gemm_block=(8, 8, 16), pipeline_window=2,
            mapping_tuning_candidates=[], group_threads=0, copy_batch=1,
            materialization_tuning_candidates=["preserve", "expensive-only"],
            max_tuning_candidates=2)
        self.assertEqual(
            MODULE.joint_candidates(args),
            [((8, 8, 16), 2, 0, 1, "preserve"),
             ((8, 8, 16), 2, 0, 1, "expensive-only")])

    def test_joint_search_revalidates_the_entire_selected_mapping(self):
        args = argparse.Namespace(tuning_candidates=[], gemm_block=(32, 64, 32), pipeline_window=1,
                                  group_threads=0, copy_batch=1, mapping_tuning_candidates=[(128, 1), (256, 4)])
        calls = []

        def measure(torch, np, candidate, case, backend, ordinal):
            calls.append((candidate.gemm_block, candidate.pipeline_window, candidate.group_threads, candidate.copy_batch))
            score = 100.0 if len(calls) == 3 else 1.0 if candidate.copy_batch == 4 else 2.0
            return dict(valid=True, native=dict(throughput_us_p50=score))

        with patch.object(MODULE, "run_case", side_effect=measure), contextlib.redirect_stdout(io.StringIO()):
            result = MODULE.run_tuned_case(None, None, args, MODULE.Case("gemm", 128, 128, 128), "metal", 0)
        self.assertEqual(calls, [((32, 64, 32), 1, 128, 1), ((32, 64, 32), 1, 256, 4), ((32, 64, 32), 1, 256, 4)])
        self.assertEqual(result["native"]["throughput_us_p50"], 100.0)
        selected = result["tuning"]["trials"][result["tuning"]["selected_trial"]]
        self.assertEqual((selected["group_threads"], selected["copy_batch"]), (256, 4))
        self.assertEqual((args.group_threads, args.copy_batch), (0, 1))

    def test_reduction_mapping_search_jits_each_width_and_revalidates(self):
        args = argparse.Namespace(tuning_candidates=[], gemm_block=(8, 8, 16), pipeline_window=1,
                                  group_threads=0, copy_batch=1,
                                  mapping_tuning_candidates=[(32, 1), (128, 1), (256, 1)])
        calls = []

        def measure(torch, np, candidate, case, backend, ordinal):
            calls.append((case.operation, candidate.group_threads, ordinal))
            score = 50.0 if len(calls) == 4 else {32: 8.0, 128: 3.0, 256: 5.0}[candidate.group_threads]
            return dict(valid=True, native=dict(throughput_us_p50=score))

        case = MODULE.Case("softmax", 64, 4096, 1)
        with patch.object(MODULE, "run_case", side_effect=measure), contextlib.redirect_stdout(io.StringIO()):
            result = MODULE.run_tuned_case(None, None, args, case, "metal", 0)
        self.assertEqual(calls, [("softmax", 32, 0), ("softmax", 128, 1),
                                 ("softmax", 256, 2), ("softmax", 128, 0)])
        self.assertEqual(result["native"]["throughput_us_p50"], 50.0)
        self.assertEqual(result["tuning"]["selected_trial"], 1)
        self.assertEqual(args.group_threads, 0)

    def test_gpu_jit_objective_is_explicit_and_replays_the_selected_lane_layout(self):
        args = argparse.Namespace(tuning_candidates=[], gemm_block=(8, 8, 16), pipeline_window=1,
                                  lane_tuning_candidates=[1, 4], tuning_metric="gpu-control")
        calls = []

        def measure(torch, np, candidate, case, backend, ordinal):
            calls.append(candidate.reduction_lane_elements)
            gpu = 20.0 if len(calls) == 3 else 2.0 if calls[-1] == 4 else 8.0
            return {"valid": True, "native": {
                "throughput_us_p50": 1.0 if calls[-1] == 1 else 3.0,
                "execution_plans": [{"normalized_kernel_cost": 1.0 if calls[-1] == 1 else 2.0}],
                "device_timing": {"control": {"encoder_instrumentation": False,
                                               "method": "metal_command_buffer_timestamps_v1",
                                               "command_buffer_throughput_us_p50": gpu}}}}

        with patch.object(MODULE, "run_case", side_effect=measure), contextlib.redirect_stdout(io.StringIO()):
            result = MODULE.run_tuned_case(None, None, args, MODULE.Case("rmsnorm", 64, 4096, 1), "metal", 0)
        self.assertEqual(calls, [1, 4, 4])
        self.assertEqual(result["tuning"]["selection_metric"], "native_gpu_command_buffer_throughput_us_p50")
        self.assertEqual(result["tuning"]["model_regret"], 3.0)
        self.assertEqual(MODULE.tuning_score(result, "gpu-control"), 20.0)
        for control in ({}, {"method": "metal_command_buffer_timestamps_v1", "encoder_instrumentation": True},
                        {"method": "metal_command_buffer_timestamps_v1", "encoder_instrumentation": False,
                         "command_buffer_throughput_us_p50": float("nan")}):
            with self.assertRaises(ValueError):
                MODULE.tuning_score({"valid": True, "native": {"throughput_us_p50": 1.0,
                                                               "device_timing": {"control": control}}}, "gpu-control")

    def test_jit_search_reports_model_regret_without_overriding_measurement(self):
        args = argparse.Namespace(tuning_candidates=[((32, 128, 1024), 1), ((128, 32, 1024), 1)],
                                  gemm_block=(64, 64, 1024), pipeline_window=1)
        calls = []

        def measure(torch, np, candidate, case, backend, ordinal):
            calls.append(candidate.gemm_block)
            if len(calls) == 3:
                return {"valid": True, "native": {"throughput_us_p50": 100.0}}
            timing, model = ((10.0, 1.0) if candidate.gemm_block[0] == 32 else (8.0, 2.0))
            return {"valid": True, "native": {"throughput_us_p50": timing,
                                                "execution_plans": [{"normalized_kernel_cost": model}]}}

        with patch.object(MODULE, "run_case", side_effect=measure), contextlib.redirect_stdout(io.StringIO()):
            result = MODULE.run_tuned_case(None, None, args, MODULE.Case("gemm", 1024, 1024, 1024), "metal", 0)
        tuning = result["tuning"]
        self.assertEqual(tuning["model_selected_trial"], 0)
        self.assertEqual(tuning["selected_trial"], 1)
        self.assertAlmostEqual(tuning["model_regret"], 0.25)
        self.assertEqual(calls[-1], (128, 32, 1024))

    def test_jit_search_cannot_hide_failed_candidates_or_revalidation(self):
        args = argparse.Namespace(tuning_candidates=[((8, 8, 16), 2)], gemm_block=(8, 8, 16), pipeline_window=2)
        case = MODULE.Case("gemm", 32, 32, 32)
        for failure in (RuntimeError("oracle"), {"valid": True, "native": {"throughput_us_p50": float("nan")}}):
            effect = [failure] if isinstance(failure, Exception) else lambda *args: failure
            with patch.object(MODULE, "run_case", side_effect=effect), contextlib.redirect_stdout(io.StringIO()):
                result = MODULE.run_tuned_case(None, None, args, case, "cpu", 0)
            self.assertFalse(result["valid"])
            self.assertFalse(result["tuning"]["trials"][0]["valid"])
        with patch.object(MODULE, "run_case", side_effect=[{"valid": True, "native": {"throughput_us_p50": 1.0}},
                                                          RuntimeError("fresh oracle")]), contextlib.redirect_stdout(io.StringIO()):
            result = MODULE.run_tuned_case(None, None, args, case, "cpu", 0)
        self.assertFalse(result["valid"])
        self.assertIn("revalidation", result["error"])
        self.assertTrue(result["tuning"]["trials"][0]["valid"])


if __name__ == "__main__":
    unittest.main()
