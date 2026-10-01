import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cuda_torch_baseline import generated_calls, expected_shapes, load_packet, validate_output, verify_packet


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

    def test_actual_narrow_storage_is_decoded_exactly(self):
        for precision, storage_dtype, dtype, values in (
            ("fp16", "float16", "<f2", [0.0, -0.0, 1.0, 2**-24]),
            ("bf16", "bfloat16", "<u2", [0, 0x8000, 0x3f80, 1]),
        ):
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                path, manifest = self.fixture(directory)
                manifest["precision"] = precision
                manifest["output"]["storage_dtype"] = storage_dtype
                raw = np.array(values, dtype=dtype).reshape(2, 2)
                decoded = raw.astype("<f4") if precision == "fp16" else (raw.astype("<u4") << 16).view("<f4")
                for entry in manifest["inputs"]:
                    entry["path"] = entry["name"] + "." + precision
                    entry["storage_dtype"] = storage_dtype
                    raw.tofile(directory / entry["path"])
                path.write_text(json.dumps(manifest))
                packet = load_packet(path)
                np.testing.assert_array_equal(packet["inputs"][0].view("<u4"), decoded.view("<u4"))
                self.assertEqual((directory / manifest["inputs"][0]["path"]).stat().st_size, 8)
                # An FP32 surrogate, or a nonfinite BF16 bit pattern, is refused.
                for entry in manifest["inputs"]:
                    entry["storage_dtype"] = "float32"
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "storage dtype"):
                    load_packet(path)

    def test_narrow_nonfinite_bits_are_rejected(self):
        for precision, dtype, storage_dtype, value in (("fp16", "<f2", "float16", np.inf),
                                                     ("bf16", "<u2", "bfloat16", 0x7fc1)):
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                path, manifest = self.fixture(directory)
                manifest["precision"] = precision
                manifest["output"]["storage_dtype"] = storage_dtype
                for entry in manifest["inputs"]:
                    entry["storage_dtype"] = storage_dtype
                    np.full((2, 2), value, dtype=dtype).tofile(directory / entry["path"])
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    load_packet(path)

    def test_generated_call_evidence_ignores_comments_and_strings(self):
        calls = generated_calls("# torch.ops.aten.mm(x)\nfragment = 'extern_kernels.mm(x)'\na = async_compile.triton('kernel', 'cuda source')\nb = kernel.run(x)\n")
        self.assertEqual(calls, {"async_compile.triton", "kernel.run"})
        self.assertEqual(generated_calls("extern_kernels.mm(x)\ntorch.ops.aten.sort.default(x)"),
                         {"extern_kernels.mm", "torch.ops.aten.sort.default"})

    def tensorcore_fixture(self, directory):
        unit, eta = 2**-11, 2**-24
        gamma = 6 * 2**-24 / (1 - 6 * 2**-24)
        probability = (1 + gamma) / (1 - gamma) * (unit + eta)
        arrays = [np.zeros((1, 1, 1, 2), "<f2"), np.zeros((1, 1, 2, 2), "<f2"),
                  np.array([[[[1, 1], [-1, -1]]]], dtype="<f2")]
        dimensions = [1, 1, 1, 1, 2, 2, 2]
        expected = np.zeros((1, 1, 1, 2), "<f8")
        strict = np.full_like(expected, 5e-5 + unit*5e-5 + eta/2)
        probability = np.full_like(expected, probability)
        manifest = dict(schema=1, operation="attention_tensorcore", dimensions=dimensions, precision="fp16",
            endianness="little", inputs=[], output=dict(path="output.f16", shape=list(expected.shape), storage_dtype="float16"),
            semantics=dict(accumulation="float32", causal=True, query_positions="last_Q_in_K", attention_scale=float(np.float32(1/np.sqrt(np.float32(2))))),
            precision_contract=dict(name="attention_single_narrow_probability_v1", stage="unnormalized_probability_before_pv_per_kv_block",
                                    rounding="rne", u_T=unit, eta_T=eta, u32=2**-24),
            expected=dict(path="expected.f64", bound_path="per_element_bound.f64", strict_bound_path="strict_bound.f64",
                          probability_rounding_bound_path="probability_rounding_bound.f64", storage_dtype="float64"))
        for i, value in enumerate(arrays):
            name = f"input{i}"
            value.tofile(directory / f"{name}.f16")
            manifest["inputs"].append(dict(name=name, path=f"{name}.f16", shape=list(value.shape), storage_dtype="float16"))
        for name, value in (("expected.f64", expected), ("strict_bound.f64", strict),
                            ("probability_rounding_bound.f64", probability), ("per_element_bound.f64", strict+(1+unit)*probability)):
            value.tofile(directory / name)
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path, manifest

    def test_tensorcore_contract_preserves_strict_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path, _ = self.tensorcore_fixture(directory)
            packet = load_packet(path)
            actual = np.full_like(packet["expected"], 2e-4)
            result = validate_output(packet, actual)
            self.assertEqual(result["failed_elements"], 0)
            self.assertEqual(result["strict_correctness"]["failed_elements"], 2)
            self.assertFalse(result["strict_correctness"]["primary_acceptance"])
            (directory / "strict_bound.f64").write_bytes(bytes(16))
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_packet(packet)

    def test_tensorcore_bound_contract_fails_closed(self):
        for failure in ("missing", "negative", "composition", "constants", "precision"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                path, manifest = self.tensorcore_fixture(directory)
                if failure == "missing":
                    del manifest["expected"]["strict_bound_path"]
                elif failure == "negative":
                    np.full((1, 1, 1, 2), -1.0, "<f8").tofile(directory / "strict_bound.f64")
                elif failure == "composition":
                    np.full((1, 1, 1, 2), 9.0, "<f8").tofile(directory / "per_element_bound.f64")
                elif failure == "constants":
                    manifest["precision_contract"]["u_T"] = 1.0
                elif failure == "precision":
                    manifest["precision"] = "fp32"
                path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
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
