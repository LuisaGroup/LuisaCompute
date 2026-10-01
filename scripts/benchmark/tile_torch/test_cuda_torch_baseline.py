import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cuda_torch_baseline import expected_shapes, load_packet, validate_output, verify_packet


class CudaTorchPacketTests(unittest.TestCase):
    def fixture(self, directory, ranking=False):
        if ranking:
            operation, dimensions = "topk", [1, 3, 2]
            arrays = [np.array([[5, 5, 1]], dtype="<f4"), np.zeros(1, "<f4"), np.zeros(1, "<f4")]
            expected = np.array([[5, 5]], dtype="<f8")
            semantics = dict(accumulation="float32", descending=True, stable=True, tie_break="original_index_ascending")
        else:
            operation, dimensions = "reduce_sum", [2, 2]
            arrays = [np.array([[1, 2], [3, 4]], dtype="<f4")] * 3
            expected = np.array([[3], [7]], dtype="<f8")
            semantics = dict(accumulation="float32")
        manifest = dict(schema=1, operation=operation, dimensions=dimensions, precision="fp32",
                        endianness="little", semantics=semantics, inputs=[],
                        output=dict(path="output.f32", storage_dtype="float32", shape=list(expected.shape)),
                        expected=dict(path="expected.f64", bound_path="per_element_bound.f64", storage_dtype="float64"))
        for i, array in enumerate(arrays):
            name = f"input{i}"
            array.tofile(directory / f"{name}.f32")
            manifest["inputs"].append(dict(name=name, path=f"{name}.f32", shape=list(array.shape), storage_dtype="float32"))
        expected.tofile(directory / "expected.f64")
        np.full_like(expected, 0 if ranking else 1e-6).tofile(directory / "per_element_bound.f64")
        if ranking:
            np.array([[0, 1]], dtype="<i8").tofile(directory / "expected_indices.i64")
            manifest["indices"] = dict(path="output_indices.i64", expected_path="expected_indices.i64",
                                       shape=[1, 2], storage_dtype="int64")
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path, manifest

    def test_complete_bounds_are_authoritative(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self.fixture(Path(temporary))
            packet = load_packet(path)
            self.assertEqual(validate_output(packet, packet["expected"])["failed_elements"], 0)
            for bad in (packet["expected"] + 1e-4, np.full((2, 1), np.nan), packet["expected"][:1]):
                with self.assertRaises(ValueError):
                    validate_output(packet, bad)

    def test_input_mutation_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self.fixture(Path(temporary))
            packet = load_packet(path)
            verify_packet(packet)
            (Path(temporary) / "input0.f32").write_bytes(bytes(16))
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_packet(packet)

    def test_rejects_truncated_nonfinite_and_negative_bound(self):
        for failure in ("truncated", "nan", "negative_bound"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                path, _ = self.fixture(directory)
                if failure == "truncated":
                    (directory / "input0.f32").write_bytes(bytes(3))
                elif failure == "nan":
                    np.full((2, 2), np.nan, dtype="<f4").tofile(directory / "input0.f32")
                else:
                    np.full((2, 1), -1, dtype="<f8").tofile(directory / "per_element_bound.f64")
                with self.assertRaises(ValueError):
                    load_packet(path)

    def test_standard_ties_are_valid_but_not_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self.fixture(Path(temporary), ranking=True)
            packet = load_packet(path)
            permutation = np.array([[1, 0]], dtype=np.int64)
            result = validate_output(packet, packet["expected"], permutation)
            self.assertFalse(result["exact_indices_and_ties"])
            with self.assertRaisesRegex(ValueError, "index mismatch"):
                validate_output(packet, packet["expected"], permutation, ranking_contract="stable")
            self.assertTrue(validate_output(packet, packet["expected"], packet["indices"], ranking_contract="stable")["exact_indices_and_ties"])

    def test_standard_ranking_still_checks_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self.fixture(Path(temporary), ranking=True)
            packet = load_packet(path)
            for bad in ([[0, 0]], [[0, 3]], [[0, 2]], [[-1, 0]]):
                with self.subTest(indices=bad), self.assertRaises(ValueError):
                    validate_output(packet, packet["expected"], np.array(bad, dtype=np.int64))

    def test_shape_and_semantic_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, original = self.fixture(Path(temporary))
            for update in (dict(dimensions=[1, 4]), dict(endianness="big"), dict(precision="tf32"),
                           dict(semantics={"accumulation": "float16"})):
                changed = copy.deepcopy(original)
                changed.update(update)
                path.write_text(json.dumps(changed))
                with self.subTest(update=update), self.assertRaises(ValueError):
                    load_packet(path)

    def test_shapes_do_not_treat_operation_dimensions_as_tensor_volume(self):
        self.assertEqual(expected_shapes("gemm", [4096, 4096, 4096]),
                         ([(4096, 4096), (4096, 4096), (1,)], (4096, 4096)))
        self.assertEqual(expected_shapes("attention", [1, 4, 2, 1, 127, 64, 32]),
                         ([(1, 4, 1, 64), (1, 2, 127, 64), (1, 2, 127, 32)], (1, 4, 1, 32)))
        with self.assertRaisesRegex(ValueError, "N=1"):
            expected_shapes("gemv", [17, 2, 31])


if __name__ == "__main__":
    unittest.main()
