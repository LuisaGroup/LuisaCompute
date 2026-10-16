"""Host-only BMM packet/CLI checks; imports neither Torch nor CUDA."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from cuda_matrix import validate_case
from cuda_torch_baseline import expected_shapes, load_packet, make_program, validate_output, verify_packet


class CudaBmmTests(unittest.TestCase):
    @staticmethod
    def case(**updates):
        result = dict(id="bmm-host", operation="bmm", dimensions=[3, 31, 37, 19],
                      tile=[16, 16, 8], precision="fp32", seed=19, pattern="random", fast_math=False)
        result.update(updates)
        return result

    def test_valid_schedules_and_precisions(self):
        for precision in ("fp32", "fp16", "bf16"):
            for dimensions, tile in (([32, 16, 16, 32], [16, 16, 16]),
                                     ([8, 128, 128, 128], [32, 32, 32]),
                                     ([3, 31, 37, 19], [16, 16, 8]),
                                     ([65536, 1, 1, 1], [1, 1, 1]),
                                     ([1, 65535, 1, 1], [1, 1, 1]),
                                     ([1, 1, 65535, 1], [1, 1, 1])):
                with self.subTest(precision=precision, dimensions=dimensions):
                    source = self.case(precision=precision, dimensions=dimensions, tile=tile)
                    self.assertEqual(validate_case(source, 0), source)

    def test_dimension_schedule_and_overflow_rejections(self):
        invalid = [dict(dimensions=[3, 31, 19]), dict(dimensions=[3, 31, 37, 19, 1]),
                   dict(dimensions=[True, 31, 37, 19]), dict(dimensions=[0, 31, 37, 19]),
                   dict(dimensions=[-1, 31, 37, 19]), dict(dimensions=[3.0, 31, 37, 19]),
                   dict(dimensions=[65537, 1, 1, 1]),
                   dict(dimensions=[65536, 65536, 65536, 65536]),
                   dict(dimensions=[2, 2048, 2048, 2049]),  # total work > 2^34, allocations fit
                   dict(dimensions=[257, 256, 1, 256]),  # A > 2^24, work < 2^28
                   dict(dimensions=[257, 1, 256, 256]),  # B > 2^24
                   dict(dimensions=[257, 256, 256, 1]),  # output > 2^24
                   dict(dimensions=[1, 65536, 1, 1], tile=[1, 1, 1]),
                   dict(dimensions=[1, 1, 65536, 1], tile=[1, 1, 1]),
                   dict(tile=[0, 16, 8]), dict(tile=[16, -1, 8]), dict(tile=[16, 16, True]),
                   dict(tile=[16, 16, 3]), dict(tile=[256, 16, 8]), dict(tile=[16, 256, 8]),
                   dict(tile=[16, 16, 512]), dict(tile=[16, 16]), dict(precision="tf32")]
        for updates in invalid:
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                validate_case(self.case(**updates), 0)

    def test_contiguous_batch_shapes_without_broadcast(self):
        self.assertEqual(expected_shapes("bmm", (3, 31, 37, 19)),
                         ([(3, 31, 19), (3, 19, 37), (1,)], (3, 31, 37)))

    @staticmethod
    def stored(values, precision):
        values = np.asarray(values, dtype="<f4")
        if precision == "fp32":
            return values, values.copy()
        if precision == "fp16":
            encoded = values.astype("<f2")
            return encoded, encoded.astype("<f4")
        bits = values.view("<u4")
        encoded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype("<u2")
        return encoded, (encoded.astype("<u4") << 16).view("<f4")

    def packet(self, directory, precision="fp32"):
        # Distinct batches and non-dyadic source values expose broadcast and dtype drift.
        a = (np.arange(30, dtype="<f4").reshape(3, 2, 5) - 12) / np.float32(13)
        b = (np.arange(60, dtype="<f4").reshape(3, 5, 4) - 17) / np.float32(19)
        arrays = [self.stored(value, precision) for value in (a, b, np.zeros(1, dtype="<f4"))]
        expected = np.matmul(arrays[0][1].astype("<f8"), arrays[1][1].astype("<f8"))
        absolute = np.matmul(np.abs(arrays[0][1].astype("<f8")), np.abs(arrays[1][1].astype("<f8")))
        q, k = 12, 5
        bounds = (q * 2**-24 / (1 - q * 2**-24) + 8 * k * 2**-53) * absolute + q * 2**-149
        if precision != "fp32":
            unit, eta = (2**-11, 2**-24) if precision == "fp16" else (2**-8, 2**-133)
            bounds += unit * (np.abs(expected) + bounds) + eta
        storage = dict(fp32="float32", fp16="float16", bf16="bfloat16")[precision]
        manifest = dict(schema=1, operation="bmm", dimensions=[3, 2, 4, 5], tile=[2, 4, 8], precision=precision,
                        endianness="little", semantics=dict(accumulation="float32",
                            batch_layout="contiguous_bmk_bkn_bmn", batch_broadcast=False,
                            contraction="mma_fused_reassociation_allowed",
                            input_quantization="round_to_nearest_even_before_fp64_oracle",
                            output_rounding="round_to_nearest_even"), inputs=[],
                        output=dict(path="output.bin", storage_dtype=storage, shape=[3, 2, 4]),
                        expected=dict(path="expected.f64", bound_path="bounds.f64", storage_dtype="float64"))
        for i, (encoded, _) in enumerate(arrays):
            name = f"input{i}"
            encoded.tofile(directory / f"{name}.bin")
            manifest["inputs"].append(dict(name=name, path=f"{name}.bin", storage_dtype=storage, shape=list(encoded.shape)))
        expected.astype("<f8").tofile(directory / "expected.f64")
        bounds.astype("<f8").tofile(directory / "bounds.f64")
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path, manifest, arrays

    def test_actual_storage_and_all_batch_oracle(self):
        for precision in ("fp32", "fp16", "bf16"):
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as temporary:
                path, _, arrays = self.packet(Path(temporary), precision)
                packet = load_packet(path)
                for actual, (encoded, decoded) in zip(packet["inputs"], arrays):
                    np.testing.assert_array_equal(actual.view("<u4"), decoded.view("<u4"))
                    self.assertEqual(encoded.dtype.itemsize, 4 if precision == "fp32" else 2)
                rounded = self.stored(packet["expected"], precision)[1]
                result = validate_output(packet, rounded)
                self.assertEqual(result["elements"], 24)
                self.assertEqual(result["failed_elements"], 0)
                for batch in (0, 1, 2):
                    bad = rounded.copy()
                    bad[batch, -1, -1] += 1
                    with self.subTest(batch=batch), self.assertRaisesRegex(ValueError, "oracle mismatch"):
                        validate_output(packet, bad)
                verify_packet(packet)

    def test_semantics_shapes_and_dtype_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, original, _ = self.packet(Path(temporary))
            changes = [("dimensions", [1, 2, 4, 5]), ("precision", "tf32")]
            for key, value in changes:
                manifest = copy.deepcopy(original)
                manifest[key] = value
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.subTest(key=key), self.assertRaises(ValueError):
                    load_packet(path)
            for key, value in (("batch_layout", "broadcast"), ("batch_broadcast", True),
                               ("batch_broadcast", 0), ("batch_broadcast", None),
                               ("accumulation", "float16"), ("contraction", "tf32"),
                               ("input_quantization", "none"), ("output_rounding", "truncate")):
                manifest = copy.deepcopy(original)
                manifest["semantics"][key] = value
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    load_packet(path)
            for part in ("inputs", "output"):
                manifest = copy.deepcopy(original)
                entry = manifest[part][0] if part == "inputs" else manifest[part]
                entry["storage_dtype"] = "float16"
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.subTest(part=part), self.assertRaises(ValueError):
                    load_packet(path)

    def test_packet_limits_fail_before_tensor_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, original, _ = self.packet(Path(temporary))
            for dims in ([65536] * 4, [257, 256, 1, 256], [1, 2, 3], [1, 2, 3, 0]):
                manifest = copy.deepcopy(original)
                manifest["dimensions"] = dims
                manifest["inputs"][0]["path"] = "missing-file-must-not-be-read.bin"
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.subTest(dimensions=dims), self.assertRaises(ValueError):
                    load_packet(path)

    def test_full_bytes_finite_bounds_and_readonly_receipts(self):
        for defect in ("truncated", "nonfinite", "negative_bound", "mutated"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                path, _, _ = self.packet(directory)
                packet = load_packet(path)
                if defect == "truncated":
                    (directory / "input0.bin").write_bytes(bytes(3))
                elif defect == "nonfinite":
                    np.full((3, 2, 5), np.nan, dtype="<f4").tofile(directory / "input0.bin")
                elif defect == "negative_bound":
                    np.full((3, 2, 4), -1, dtype="<f8").tofile(directory / "bounds.f64")
                else:
                    (directory / "input0.bin").write_bytes(bytes(120))
                    with self.assertRaisesRegex(ValueError, "changed"):
                        verify_packet(packet)
                    continue
                with self.assertRaises(ValueError):
                    load_packet(path)

    def test_torch_bmm_receives_original_typed_objects(self):
        for precision in ("fp32", "fp16", "bf16"):
            with self.subTest(precision=precision):
                inputs = [object(), object(), object()]
                output, calls = object(), []

                def bmm(a, b):
                    calls.append((a, b))
                    return output

                packet = dict(manifest=dict(operation="bmm", precision=precision, dimensions=[3, 2, 4, 5]))
                invoke, description = make_program(SimpleNamespace(bmm=bmm), packet, inputs)
                self.assertFalse(calls)
                self.assertIs(invoke(), output)
                self.assertEqual(calls, [(inputs[0], inputs[1])])
                self.assertEqual(description, "functional bmm")


if __name__ == "__main__":
    unittest.main()
