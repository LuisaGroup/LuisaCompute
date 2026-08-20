#!/usr/bin/env python3

from __future__ import annotations

import array
import importlib.util
import math
import pathlib
import sys
import tempfile
import unittest


SCRIPT_DIRECTORY = pathlib.Path(__file__).resolve().parent
MODULE_SPEC = importlib.util.spec_from_file_location(
    "simd_ispc_run", SCRIPT_DIRECTORY / "run.py"
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError("failed to load standalone SIMD/ISPC driver")
RUN = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = RUN
MODULE_SPEC.loader.exec_module(RUN)


class CpuSetTest(unittest.TestCase):

    def test_parses_ranges_and_individual_cpus(self):
        self.assertEqual(RUN.parse_cpu_set("0-2,5,7-8"), {0, 1, 2, 5, 7, 8})

    def test_rejects_descending_range(self):
        with self.assertRaisesRegex(Exception, "descending"):
            RUN.parse_cpu_set("4-2")


class CompileArgumentsTest(unittest.TestCase):

    def test_removes_translation_unit_outputs_and_dependency_flags(self):
        compiler, arguments = RUN.reusable_compile_arguments(
            [
                "clang++",
                "-DVALUE=1",
                "-MD",
                "-MF",
                "dependency.d",
                "-MT",
                "target.o",
                "-o",
                "target.o",
                "-c",
                "/project/benchmark_simd_gemm.cpp",
                "-O3",
            ]
        )
        self.assertEqual(compiler, "clang++")
        self.assertEqual(arguments, ["-DVALUE=1", "-O3"])


class ResultParsingTest(unittest.TestCase):

    def test_extracts_record_from_backend_log_noise(self):
        record = RUN.parse_result(
            "backend log\n"
            "simd_ispc_suite,implementation=luisa,backend=simd,width=8,"
            "workers=16,workload=gemm,items=65536,dispatches=128,"
            "median_seconds=0.5,rate_unit=gflop_per_second,median_rate=8.5,"
            "checksum=abc,samples_seconds=0.5;0.6\n"
        )
        self.assertEqual(record["implementation"], "luisa")
        self.assertEqual(record["width"], 8)
        self.assertEqual(record["median_rate"], 8.5)
        self.assertEqual(record["samples_seconds"], [0.5, 0.6])


class OrderAndStatisticsTest(unittest.TestCase):

    def setUp(self):
        self.variants = [
            RUN.Variant(f"w{width}", "luisa", width, "simd")
            for width in (1, 2, 4, 8)
        ]

    def test_balanced_order_is_a_permutation_each_round(self):
        expected = {variant.name for variant in self.variants}
        orders = [RUN.balanced_order(self.variants, index) for index in range(8)]
        for order in orders:
            self.assertEqual({variant.name for variant in order}, expected)
        self.assertNotEqual(orders[0], orders[1])
        for variant in self.variants:
            positions = [order.index(variant) for order in orders]
            self.assertEqual(
                [positions.count(index) for index in range(len(self.variants))],
                [2] * len(self.variants),
            )
        for lhs_index, lhs in enumerate(self.variants):
            for rhs in self.variants[lhs_index + 1 :]:
                lhs_first = sum(
                    order.index(lhs) < order.index(rhs) for order in orders
                )
                self.assertEqual(lhs_first, len(orders) // 2)

    def test_two_variant_order_really_alternates(self):
        variants = self.variants[:2]
        orders = [RUN.balanced_order(variants, index) for index in range(6)]
        self.assertEqual(
            [[variant.name for variant in order] for order in orders],
            [["w1", "w2"], ["w2", "w1"]] * 3,
        )

    def test_paired_ratio_uses_log_space(self):
        summary = RUN.paired_ratio_summary([2.0, 8.0], [1.0, 2.0])
        self.assertAlmostEqual(summary["geomean"], math.sqrt(8.0))
        self.assertLess(summary["ci95_low"], summary["geomean"])
        self.assertGreater(summary["ci95_high"], summary["geomean"])


class NumericValidationTest(unittest.TestCase):

    @staticmethod
    def write_floats(path: pathlib.Path, values: list[float]) -> None:
        payload = array.array("f", values)
        with path.open("wb") as stream:
            payload.tofile(stream)

    def test_absolute_plus_relative_path_tolerance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            reference = root / "reference.bin"
            candidate = root / "candidate.bin"
            self.write_floats(reference, [0.0, 1.0, 100.0])
            self.write_floats(candidate, [1.0e-6, 1.000001, 100.001])
            comparison = RUN.compare_path_dumps(
                reference, candidate, 2.0e-5, 2.0e-5
            )
            self.assertEqual(comparison["violation_count"], 0)
            self.assertGreater(comparison["maximum_absolute_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
