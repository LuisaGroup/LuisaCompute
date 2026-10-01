"""Host-only embedding mixed-input ABI, exact IDs and value-bit checks."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from cuda_matrix import tensor_receipts, validate_case
from cuda_torch_baseline import load_packet, make_program, prepare_cpu_inputs, validate_output, verify_packet


class CudaEmbeddingTests(unittest.TestCase):
    @staticmethod
    def case(**changes):
        value = dict(id="embedding-fixture", operation="embedding", dimensions=[7, 5, 5],
                     tile=[1, 4, 1], precision="fp32", seed=19, pattern="adversarial", fast_math=False)
        value.update(changes)
        return value

    def packet(self, directory, precision="fp32"):
        case = self.case(precision=precision)
        table = ((np.arange(35, dtype="<f4")-17)*.125).reshape(7, 5)
        table[0, 0] = -0.; table[0, 1] = 0.; table[6, 0] = -0.; table[6, 1] = 0.
        ids = np.array([6, 0, 3, 6, 0], dtype="<i8")
        storage, suffix = {"fp32": ("float32", "f32"), "fp16": ("float16", "f16"),
                           "bf16": ("bfloat16", "bf16")}[precision]
        raw = ((table.view("<u4") >> 16).astype("<u2") if precision == "bf16" else
               table.astype("<f2" if precision == "fp16" else "<f4"))
        decoded = ((raw.astype("<u4") << 16).view("<f4") if precision == "bf16" else raw.astype("<f4"))
        raw.tofile(directory / ("input0."+suffix)); ids.tofile(directory / "input1.i64")
        expected = decoded[ids]
        expected.astype("<f8").tofile(directory / "expected.f64")
        np.zeros_like(expected, dtype="<f8").tofile(directory / "bound.f64")
        manifest = dict(case, schema=1, endianness="little", algorithm="uniform_int64_row_gather",
            semantics=dict(accumulation="none", index_dtype="int64", index_bounds="reject_invalid",
                           gather_axis=0, value_preservation="storage_bits", index_pattern="repeated_boundary_rows"),
            inputs=[dict(name="input0", path="input0."+suffix, shape=[7, 5], storage_dtype=storage),
                    dict(name="input1", path="input1.i64", shape=[5], storage_dtype="int64")],
            output=dict(path="output."+suffix, shape=[5, 5], storage_dtype=storage),
            expected=dict(path="expected.f64", bound_path="bound.f64", storage_dtype="float64"))
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path, case, expected

    def test_small_large_tail_shapes_and_three_storage_precisions(self):
        for precision in ("fp32", "fp16", "bf16"):
            for vocabulary, width, tokens, bd in [(7, 1, 1, 1), (4096, 64, 1, 64),
                  (4096, 1024, 32, 256), (4096, 4096, 512, 256), (17, 65, 37, 32)]:
                row = self.case(dimensions=[vocabulary, width, tokens], tile=[1, bd, 1], precision=precision)
                self.assertEqual(validate_case(row, 0)["precision"], precision)
        self.assertEqual(validate_case(self.case(tile=[1, 3, 1]), 0)["tile"], [1, 3, 1])

    def test_shape_allocation_schedule_pattern_fail_closed(self):
        for updates in (dict(dimensions=[7, 5]), dict(dimensions=[0, 5, 5]),
                        dict(dimensions=[65536, 4096, 1]), dict(dimensions=[7, 4096, 65536]),
                        dict(tile=[2, 4, 1]), dict(tile=[1, 4, 2]), dict(tile=[1, 32768, 1]),
                        dict(pattern="cancellation"), dict(ranking_algorithm="full_sort_prefix")):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                validate_case(self.case(**updates), 0)

    def test_mixed_packet_storage_is_real_and_exact(self):
        for precision in ("fp32", "fp16", "bf16"):
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as temporary:
                path, case, expected = self.packet(Path(temporary), precision)
                packet = load_packet(path)
                self.assertEqual(packet["inputs"][1].dtype, np.dtype("int64"))
                self.assertEqual((path.parent / "input1.i64").stat().st_size, 5*8)
                entry = packet["manifest"]["inputs"][0]
                self.assertEqual((path.parent / entry["path"]).stat().st_size, 35*(4 if precision=="fp32" else 2))
                self.assertEqual(validate_output(packet, expected)["max_abs_error"], 0)
                self.assertIn("input1.i64", tensor_receipts(path, case)[1])
                verify_packet(packet)

    def test_invalid_ids_rejected_exactly_before_torch(self):
        for invalid in (-1, 7, 2**24+1, 2**53+1, 2**63-1):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temporary:
                path, _, _ = self.packet(Path(temporary))
                ids = np.array([6, 0, invalid, 6, 0], dtype="<i8")
                ids.tofile(path.parent / "input1.i64")
                with self.assertRaisesRegex(ValueError, "outside"):
                    load_packet(path)

    def test_input_descriptors_are_not_globally_relaxed(self):
        for position, wrong_type in ((0, "int64"), (0, "float16"), (1, "float32"), (1, "int32")):
            with self.subTest(position=position, dtype=wrong_type), tempfile.TemporaryDirectory() as temporary:
                path, case, _ = self.packet(Path(temporary))
                data = json.loads(path.read_text()); data["inputs"][position]["storage_dtype"] = wrong_type
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError): load_packet(path)
                with self.assertRaises(ValueError): tensor_receipts(path, case)
        with tempfile.TemporaryDirectory() as temporary:
            path, _, _ = self.packet(Path(temporary))
            data = json.loads(path.read_text()); data["inputs"][1]["shape"] = [1, 5]
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): load_packet(path)

    def test_cuda_feature_grid_and_bad_semantics_rejected_before_data_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _, _ = self.packet(Path(temporary))
            original = json.loads(path.read_text())
            for changes, message in ((dict(backend="cuda", dimensions=[1, 65536, 1], tile=[1, 1, 1]), "launch grid"),
                                     (dict(pattern="cancellation"), "pattern"),
                                     (dict(semantics={**original["semantics"], "index_bounds":"wrap"}), "contract")):
                path.write_text(json.dumps({**original, **changes}))
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                    load_packet(path)

    def test_oracle_and_values_preserve_signed_zero_and_all_elements(self):
        for precision in ("fp32", "fp16", "bf16"):
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as temporary:
                path, _, expected = self.packet(Path(temporary), precision)
                packet = load_packet(path)
                changed = expected.copy(); changed[0, 0] = 0.
                with self.assertRaisesRegex(ValueError, "storage bits"): validate_output(packet, changed)
                changed = expected.copy(); changed[-1, -1] += .125
                with self.assertRaisesRegex(ValueError, "oracle mismatch"): validate_output(packet, changed)
                altered = expected.astype("<f8"); altered[0, 0] = 0.
                altered.tofile(path.parent / "expected.f64")
                with self.assertRaisesRegex(ValueError, "oracle mismatch"): load_packet(path)

    def test_nonzero_bound_and_index_file_mutation_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _, _ = self.packet(Path(temporary))
            packet = load_packet(path)
            ids = packet["inputs"][1].copy(); ids[0] = 0
            ids.tofile(path.parent / "input1.i64")
            with self.assertRaisesRegex(ValueError, "changed"): verify_packet(packet)
            np.ones((5, 5), dtype="<f8").tofile(path.parent / "bound.f64")
            with self.assertRaisesRegex(ValueError, "zero-bound"): load_packet(path)

    def test_torch_expression_is_normal_typed_index_select(self):
        table, ids, result = object(), object(), object()
        calls = []
        def select(*args): calls.append(args); return result
        invoke, description = make_program(SimpleNamespace(index_select=select),
            dict(manifest=dict(operation="embedding", dimensions=[7, 5, 5])), [table, ids])
        self.assertIs(invoke(), result)
        self.assertEqual(calls, [(table, 0, ids)])
        self.assertIn("int64_ids", description)

    def test_cpu_conversion_never_casts_index_tensor_to_float(self):
        class Tensor:
            def __init__(self, array): self.array = array; self.dtype = array.dtype
            def to(self, *, dtype):
                self_test.assertNotEqual(self.array.dtype, np.dtype("int64"))
                return Tensor(self.array.astype(dtype))
            def float(self):
                self_test.assertNotEqual(self.array.dtype, np.dtype("int64"))
                return Tensor(self.array.astype("<f4"))
            def numpy(self): return self.array
        self_test = self
        # This is a conversion-only identity test, not a valid embedding lookup.
        ids = np.array([2**53+1, 2**63-1], dtype="<i8")
        packet = dict(manifest=dict(operation="embedding"), inputs=[np.array([[-0., 1.]], "<f4"), ids])
        tensors, restored = prepare_cpu_inputs(SimpleNamespace(from_numpy=Tensor, int64=np.dtype("int64")), packet, np.dtype("float32"))
        np.testing.assert_array_equal(restored[1], ids)
        self.assertEqual(tensors[1].dtype, np.dtype("int64"))
        self.assertEqual(restored[0].view("<u4")[0, 0], 0x80000000)


if __name__ == "__main__":
    unittest.main()
