import argparse
import copy
from contextlib import redirect_stderr
import io
import os
import unittest
from unittest.mock import patch

from compare_llm import check_metadata, configure_probe_environment, make_summary, parse_arguments, parse_case, reference, shapes_for, validate_output


class LlmBenchmarkTests(unittest.TestCase):
    def test_shapes_and_invalid_cases(self):
        self.assertEqual(shapes_for("rope", (17, 258)), ([(17, 258), (17, 129), (17, 129)], (17, 258)))
        self.assertEqual(parse_case("attention:1,4,2,1,128,64,64")[0], "attention")
        for text in ("rope:1,7", "swiglu:0,4", "attention:1,3,2,1,4,4,4", "attention:1,2,1,5,4,4,4", "swiglu:65536,65536", "unknown:1,2"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_case(text)

    def test_rope_oracle_and_complete_check(self):
        import numpy as np
        x = np.array([[1, 2, 3, 4]], np.float32)
        c, s = np.array([[1, 0]], np.float32), np.array([[0, 1]], np.float32)
        expected = reference("rope", (1, 4), [x, c, s])
        np.testing.assert_array_equal(expected, [[1, -4, 3, 2]])
        self.assertEqual(validate_output(expected, expected)["elements"], 4)
        for actual in (expected[:, :3], expected + 1, np.full_like(expected, np.nan)):
            with self.assertRaises(ValueError):
                validate_output(actual, expected)

    def test_attention_bottom_right_mask_and_gqa(self):
        import numpy as np
        dims = (1, 4, 2, 2, 3, 1, 1)
        q = np.zeros((1, 4, 2, 1), np.float32)
        k = np.zeros((1, 2, 3, 1), np.float32)
        v = np.array([[[[1], [3], [8]], [[2], [6], [10]]]], np.float32)
        expected = reference("attention", dims, [q, k, v])
        np.testing.assert_allclose(expected[0, :2, :, 0], [[2, 4], [2, 4]])
        np.testing.assert_allclose(expected[0, 2:, :, 0], [[4, 6], [4, 6]])

    def test_metadata_fail_closed(self):
        row = dict(implementation="tile_xir_simd", backend="cpu", precision="fp32", fast_math=False,
                   relaxed_precision=False, runtime="luisa", timing="synchronized_host_wall",
                   batch_policy="one_runtime_command_list_per_batch", operation="rope", dimensions=[1, 4],
                   attention_block=[1, 1], input_shapes=[[1, 4], [1, 2], [1, 2]], output_shape=[1, 4],
                   correctness=dict(checks=2, elements_per_check=4, guard_elements_per_check=34, atol=5e-5, rtol=5e-5),
                   repetitions=10, throughput_us=[1., 2.], latency_us=[3., 4.])
        check_metadata(row, "cpu", "rope", (1, 4), (1, 1), 2)
        current = row | dict(source_reduction_policy="unordered_tree", reduction_candidate_setting="not_applicable")
        check_metadata(current, "cpu", "rope", (1, 4), (1, 1), 2)
        for key, value in (("source_reduction_policy", "fold_left"), ("reduction_candidate_setting", "enabled")):
            with self.assertRaises(ValueError):
                check_metadata(current | {key: value}, "cpu", "rope", (1, 4), (1, 1), 2)
        for tree, threads in ((True, 0), (False, 128)):
            with self.assertRaises(ValueError):
                check_metadata(row, "cpu", "rope", (1, 4), (1, 1), 2, tree, threads)
        explicit = row | dict(reduction_tree=True, requested_group_threads=128)
        check_metadata(explicit, "cpu", "rope", (1, 4), (1, 1), 2, True, 128)
        for key, value in (("reduction_tree", 1), ("requested_group_threads", 128.0)):
            with self.assertRaises(ValueError):
                check_metadata(explicit | {key: value}, "cpu", "rope", (1, 4), (1, 1), 2, True, 128)
        with self.assertRaises(ValueError):
            check_metadata(row, "cpu", "rope", (1, 4), (1, 1), 2, forward_input_views=True)
        check_metadata(row | dict(requested_input_views=True), "cpu", "rope", (1, 4), (1, 1), 2, forward_input_views=True)
        for value in (1, False, "true"):
            with self.assertRaises(ValueError):
                check_metadata(row | dict(requested_input_views=value), "cpu", "rope", (1, 4), (1, 1), 2, forward_input_views=True)
        for key, value in (("fast_math", 0), ("operation", "swiglu"), ("repetitions", 0), ("throughput_us", [float("nan"), 2])):
            bad = copy.deepcopy(row)
            bad[key] = value
            with self.assertRaises(ValueError):
                check_metadata(bad, "cpu", "rope", (1, 4), (1, 1), 2)

    def test_paired_ratios_and_missing_case_not_omitted(self):
        rows = [dict(operation="rope", dimensions=[1, 4], round=r, path=p, valid=True,
                     measurement=dict(throughput_us_p50=t, latency_us_p50=2 * t))
                for r, a, b in ((0, 2, 4), (1, 9, 3)) for p, t in (("native", a), ("torch", b))]
        result = make_summary(rows, 2)[0]
        self.assertEqual(result["throughput_us_p50"]["paired_native_over_torch_median"], 1.75)
        self.assertEqual(result["throughput_us_p50"]["slower_rounds"], 1)
        rows[0]["valid"] = False
        self.assertEqual(make_summary(rows, 2), [dict(operation="rope", dimensions=[1, 4], complete=False)])

    def test_attention_decomposition_acknowledged(self):
        dims = (1, 2, 1, 1, 3, 4, 5)
        inputs, output = shapes_for("attention", dims)
        row = dict(implementation="tile_tirx_metal", backend="metal", precision="fp32", fast_math=False,
                   relaxed_precision=False, runtime="luisa", timing="synchronized_host_wall",
                   batch_policy="one_runtime_command_list_per_batch", operation="attention", dimensions=list(dims),
                   attention_block=[1, 3], input_shapes=[list(s) for s in inputs], output_shape=list(output),
                   correctness=dict(checks=2, elements_per_check=10, guard_elements_per_check=34, atol=5e-5, rtol=5e-5),
                   repetitions=10, throughput_us=[1., 2.], latency_us=[3., 4.])
        check = lambda result, mode: check_metadata(result, "metal", "attention", dims, (1, 3), 2, attention_qk=mode)
        check(row, "mma")  # Legacy default is still accepted.
        check(row | dict(attention_qk="reduce"), "reduce")
        for bad in (row, row | dict(attention_qk="mma"), row | dict(attention_qk=True)):
            with self.assertRaises(ValueError):
                check(bad, "reduce")
        with self.assertRaises(ValueError):
            check(row | dict(attention_qk="reduce"), "mma")

    def test_attention_pv_default_backcompat_and_explicit_acknowledgment(self):
        dims = (1, 2, 1, 1, 3, 4, 5)
        inputs, output = shapes_for("attention", dims)
        row = dict(implementation="tile_tirx_metal", backend="metal", precision="fp32", fast_math=False,
                   relaxed_precision=False, runtime="luisa", timing="synchronized_host_wall",
                   batch_policy="one_runtime_command_list_per_batch", operation="attention", dimensions=list(dims),
                   attention_block=[1, 3], input_shapes=[list(s) for s in inputs], output_shape=list(output),
                   correctness=dict(checks=2, elements_per_check=10, guard_elements_per_check=34, atol=5e-5, rtol=5e-5),
                   repetitions=10, throughput_us=[1., 2.], latency_us=[3., 4.])
        check = lambda result, **modes: check_metadata(result, "metal", "attention", dims, (1, 3), 2, **modes)
        check(copy.deepcopy(row))  # Old binaries/artifacts need not report the default.
        check(row | dict(attention_qk="mma", attention_pv="mma"))
        check(row | dict(attention_pv="reduce"), attention_pv="reduce")
        check(row | dict(attention_qk="reduce", attention_pv="reduce"), attention_qk="reduce", attention_pv="reduce")
        check(row | dict(attention_qk="reduce", attention_pv="mma"), attention_qk="reduce")
        for policy in (None, "mma", True, 1, "not_applicable", "invalid"):
            candidate = row if policy is None else row | dict(attention_pv=policy)
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                check(candidate, attention_pv="reduce")
        with self.assertRaises(ValueError):
            check(row | dict(attention_pv="reduce"))
        # Neither an acknowledged PV probe nor its math-equivalent source
        # decomposition relaxes the complete oracle and timing contract.
        for key, value in (("repetitions", 0), ("throughput_us", [1.]), ("latency_us", [float("nan"), 1.]),
                           ("correctness", row["correctness"] | dict(elements_per_check=9)),
                           ("correctness", row["correctness"] | dict(guard_elements_per_check=0)),
                           ("correctness", row["correctness"] | dict(atol=1e-2))):
            candidate = row | dict(attention_pv="reduce") | {key: value}
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                check(candidate, attention_pv="reduce")

    def test_attention_pv_non_attention_is_not_applicable(self):
        row = dict(implementation="tile_tirx_metal", backend="metal", precision="fp32", fast_math=False,
                   relaxed_precision=False, runtime="luisa", timing="synchronized_host_wall",
                   batch_policy="one_runtime_command_list_per_batch", operation="rope", dimensions=[1, 4],
                   attention_block=[1, 1], input_shapes=[[1, 4], [1, 2], [1, 2]], output_shape=[1, 4],
                   correctness=dict(checks=2, elements_per_check=4, guard_elements_per_check=34, atol=5e-5, rtol=5e-5),
                   repetitions=10, throughput_us=[1., 2.], latency_us=[3., 4.])
        check = lambda result, **modes: check_metadata(result, "metal", "rope", (1, 4), (1, 1), 2, **modes)
        check(copy.deepcopy(row))
        check(row | dict(attention_pv="not_applicable"))
        check(row | dict(attention_pv="not_applicable"), attention_pv="reduce")
        for policy in (None, "mma", "reduce", True):
            candidate = row if policy is None else row | dict(attention_pv=policy)
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                check(candidate, attention_pv="reduce")

    def test_attention_probe_parser_guards(self):
        common = ["--native", "not-resolved-native", "--build-dir", "not-resolved-build", "--output", "not-created"]
        attention = "attention:1,2,1,1,3,4,5"
        with patch("compare_llm.subprocess.run") as process:
            default = parse_arguments(common + ["--backend", "metal", "--case", attention])
            self.assertEqual((default.attention_qk, default.attention_pv), ("mma", "mma"))
            for qk in ("mma", "reduce"):
                for pv in ("mma", "reduce"):
                    selected = parse_arguments(common + ["--backend", "metal", "--case", attention,
                                                         "--case", "rope:1,4", "--attention-qk", qk, "--attention-pv", pv])
                    self.assertEqual((selected.attention_qk, selected.attention_pv), (qk, pv))
            process.assert_not_called()  # Parsing never builds or launches.
        invalid = [(["--backend", "cpu", "--case", attention, "--attention-pv", "reduce"], "PV decomposition"),
                   (["--backend", "metal", "--case", "rope:1,4", "--attention-pv", "reduce"], "PV decomposition"),
                   (["--backend", "metal", "--case", attention, "--attention-pv", "invalid"], "invalid choice"),
                   (["--backend", "cpu", "--case", attention, "--attention-qk", "reduce"], "QK decomposition")]
        for options, message in invalid:
            output = io.StringIO()
            with self.subTest(options=options), redirect_stderr(output), self.assertRaises(SystemExit):
                parse_arguments(common + options)
            self.assertIn(message, output.getvalue())

    def test_attention_probe_environment_is_sanitized_and_independent(self):
        common = ["--native", "unused", "--build-dir", "unused", "--output", "unused", "--backend", "metal",
                  "--case", "attention:1,2,1,1,3,4,5"]
        for qk in ("mma", "reduce"):
            for pv in ("mma", "reduce"):
                args = parse_arguments(common + ["--attention-qk", qk, "--attention-pv", pv])
                inherited = {"LUISA_TILE_BENCH_ATTENTION_QK": "stale-qk", "LUISA_TILE_BENCH_ATTENTION_PV": "stale-pv",
                             "LUISA_TILE_BENCH_UNRELATED": "stale", "LUISA_SIMD_WARP_WIDTH": "64"}
                with self.subTest(qk=qk, pv=pv), patch.dict(os.environ, inherited, clear=True):
                    removed = configure_probe_environment(args)
                    self.assertEqual(removed, inherited)
                    for key, mode in (("LUISA_TILE_BENCH_ATTENTION_QK", qk), ("LUISA_TILE_BENCH_ATTENTION_PV", pv)):
                        if mode == "mma":
                            self.assertNotIn(key, os.environ)
                        else:
                            self.assertEqual(os.environ[key], mode)
                    self.assertNotIn("LUISA_TILE_BENCH_UNRELATED", os.environ)
                    self.assertEqual(os.environ["LUISA_SIMD_WARP_WIDTH"], "8")


if __name__ == "__main__":
    unittest.main()
