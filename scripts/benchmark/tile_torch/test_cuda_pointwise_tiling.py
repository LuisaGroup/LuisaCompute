"""Host-only checks for explicit pointwise feature scheduling and receipts."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cuda_matrix import tensor_receipts, validate_case
from cuda_torch_baseline import expected_shapes, load_packet, pointwise_schedule_algorithm


class CudaPointwiseTilingTests(unittest.TestCase):
    @staticmethod
    def case(op="swiglu", width=65, bd=32, precision="fp32", rows=3):
        return dict(id="feature-test", operation=op, dimensions=[rows, width], tile=[1, bd, 1],
                    precision=precision, seed=19, pattern="random", fast_math=False)

    def packet(self, directory, case):
        storage, dtype = {"fp32": ("float32", "<f4"), "fp16": ("float16", "<f2"),
                          "bf16": ("bfloat16", "<u2")}[case["precision"]]
        inputs, output = expected_shapes(case["operation"], case["dimensions"])
        entries = []
        for i, sizes in enumerate(inputs):
            name = f"input{i}.bin"
            np.zeros(sizes, dtype=dtype).tofile(directory / name)
            entries.append(dict(name=f"input{i}", path=name, shape=list(sizes), storage_dtype=storage))
        np.zeros(output, dtype="<f8").tofile(directory / "expected.f64")
        np.full(output, 5e-5, dtype="<f8").tofile(directory / "bound.f64")
        semantics = dict(accumulation="float32")
        if case["operation"] == "rope": semantics["rope_pairing"] = "half_split"
        if case["operation"] == "gelu_residual": semantics["gelu_approximation"] = "tanh"
        manifest = dict(case, schema=1, backend="cuda", endianness="little", semantics=semantics,
            algorithm=pointwise_schedule_algorithm(case["operation"], case["dimensions"], case["tile"], "cuda"),
            inputs=entries, output=dict(path="output.bin", shape=list(output), storage_dtype=storage),
            expected=dict(path="expected.f64", bound_path="bound.f64", storage_dtype="float64"))
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_pointwise_feature_and_whole_row_schedules(self):
        for op in ("swiglu", "gelu_residual", "rope"):
            for precision in ("fp32", "fp16", "bf16"):
                for rows in (1, 3):
                    for bd in (32, 128):
                        case = self.case(op, 130 if op == "rope" else 65, bd, precision, rows)
                        self.assertEqual(validate_case(case, 0)["tile"], case["tile"])
                        expected = "feature_tiled_pointwise_fp32_compute" if bd == 32 else "whole_row_tile_fp32_compute"
                        self.assertEqual(pointwise_schedule_algorithm(op, case["dimensions"], case["tile"]), expected)

    def test_reductions_and_row_blocking_not_relaxed(self):
        for op in ("rmsnorm", "layernorm", "softmax", "masked_softmax", "scan", "scan_ordered", "reduce_sum", "reduce_max"):
            with self.subTest(op=op), self.assertRaises(ValueError): validate_case(self.case(op), 0)
        for op in ("swiglu", "gelu_residual", "rope"):
            case = self.case(op, 130 if op == "rope" else 65)
            for tile in ([4, 32, 1], [1, 32, 2], [1, 0, 1], [1, 32768, 1]):
                with self.subTest(op=op, tile=tile), self.assertRaises(ValueError):
                    validate_case({**case, "tile": tile}, 0)
        with self.assertRaises(ValueError): validate_case(self.case("rope", 65), 0)

    def test_real_typed_packets_and_realized_algorithm_receipts(self):
        for op in ("swiglu", "gelu_residual", "rope"):
            for precision in ("fp32", "fp16", "bf16"):
                for bd in (32, 128):
                    case = self.case(op, 130 if op == "rope" else 65, bd, precision)
                    with self.subTest(op=op, precision=precision, bd=bd), tempfile.TemporaryDirectory() as temporary:
                        path = self.packet(Path(temporary), case)
                        packet = load_packet(path)
                        self.assertEqual(packet["expected"].shape, (3, case["dimensions"][1]))
                        self.assertEqual(len(tensor_receipts(path, case)[1]), 5)
                        manifest = json.loads(path.read_text())
                        manifest["algorithm"] = ("whole_row_tile_fp32_compute" if bd == 32 else "feature_tiled_pointwise_fp32_compute")
                        path.write_text(json.dumps(manifest))
                        with self.assertRaisesRegex(ValueError, "algorithm mismatch"): load_packet(path)
                        with self.assertRaisesRegex(ValueError, "algorithm mismatch"): tensor_receipts(path, case)

    def test_cuda_grid_guard_does_not_constrain_simd(self):
        with self.assertRaisesRegex(ValueError, "launch grid"):
            pointwise_schedule_algorithm("swiglu", [1, 65536], [1, 1, 1], "cuda")
        self.assertEqual(pointwise_schedule_algorithm("swiglu", [1, 65536], [1, 1, 1], "simd"),
                         "feature_tiled_pointwise_fp32_compute")


if __name__ == "__main__": unittest.main()
