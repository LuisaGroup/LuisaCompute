"""Host-only argmax ABI, stable-index and bit-correspondence checks."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from cuda_matrix import tensor_receipts, validate_case
from cuda_torch_baseline import load_packet, make_program, validate_output, verify_packet


class CudaArgmaxTests(unittest.TestCase):
    @staticmethod
    def case(**changes):
        row = dict(id="argmax-fixture", operation="argmax", dimensions=[4, 5],
                   tile=[1, 8, 1], precision="fp32", seed=19, pattern="adversarial", fast_math=False)
        row.update(changes)
        return row

    def packet(self, directory, precision="fp32"):
        case = self.case(precision=precision)
        # Stable winners include negative zero, a tied nonzero maximum and a tail.
        x = np.array([[-0., 0., -0., 0., -0.], [5, -1, 5, 2, 5],
                      [-4, -3, -2, -1, 7], [-1.25]*5], dtype="<f4")
        storage, suffix = {"fp32": ("float32", "f32"), "fp16": ("float16", "f16"),
                           "bf16": ("bfloat16", "bf16")}[precision]
        arrays = [x, np.zeros(1, "<f4"), np.zeros(1, "<f4")]
        manifest = dict(case, schema=1, endianness="little", algorithm="stable_first_index_argmax",
                        inputs=[], semantics=dict(accumulation="float32", descending=True, stable=True,
                        tie_break="original_index_ascending"),
                        output=dict(path="output."+suffix, shape=[4, 1], storage_dtype=storage),
                        expected=dict(path="expected.f64", bound_path="bound.f64", storage_dtype="float64"),
                        indices=dict(path="output_indices.i64", expected_path="expected_indices.i64",
                                     shape=[4, 1], storage_dtype="int64"))
        decoded = []
        for i, array in enumerate(arrays):
            raw = ((array.view("<u4") >> 16).astype("<u2") if precision == "bf16" else
                   array.astype("<f2" if precision == "fp16" else "<f4"))
            value = ((raw.astype("<u4") << 16).view("<f4") if precision == "bf16" else raw.astype("<f4"))
            decoded.append(value)
            name = f"input{i}"
            raw.tofile(directory / (name+"."+suffix))
            manifest["inputs"].append(dict(name=name, path=name+"."+suffix, shape=list(raw.shape), storage_dtype=storage))
        indices = np.array([[0], [0], [4], [0]], dtype="<i8")
        values = np.take_along_axis(decoded[0], indices, axis=-1)
        values.astype("<f8").tofile(directory / "expected.f64")
        np.zeros((4, 1), "<f8").tofile(directory / "bound.f64")
        indices.tofile(directory / "expected_indices.i64")
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path, case, values, indices

    def test_parser_accepts_three_precisions_and_small_large_tail_shapes(self):
        for precision in ("fp32", "fp16", "bf16"):
            for r, n in ((1, 1), (4, 33), (17, 65), (128, 1024), (1, 4096), (4, 8192)):
                case = self.case(precision=precision, dimensions=[r, n], tile=[1, 1 << (n-1).bit_length(), 1])
                self.assertEqual(validate_case(case, 0)["precision"], precision)
        # Generic parser does not impose a CUDA-native-only power-of-two rule.
        self.assertEqual(validate_case(self.case(tile=[1, 5, 1]), 0)["tile"], [1, 5, 1])

    def test_parser_rejects_invalid_shapes_and_unrelated_ranking_switch(self):
        for change in (dict(dimensions=[4, 5, 1]), dict(dimensions=[4, 0]),
                       dict(tile=[1, 4, 1]), dict(tile=[2, 8, 1]), dict(tile=[1, 8, 2]),
                       dict(dimensions=[65536, 1024], tile=[1, 1024, 1]),
                       dict(ranking_algorithm="full_sort_prefix")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_case(self.case(**change), 0)

    def test_real_typed_packets_have_value_and_int64_oracles(self):
        for precision in ("fp32", "fp16", "bf16"):
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as temporary:
                path, case, values, indices = self.packet(Path(temporary), precision)
                packet = load_packet(path)
                verify_packet(packet)
                tensor_receipts(path, case)
                result = validate_output(packet, values, indices)
                self.assertTrue(result["exact_indices_and_ties"])
                self.assertEqual(result["max_abs_error"], 0)
                self.assertEqual((path.parent / packet["manifest"]["inputs"][0]["path"]).stat().st_size,
                                 20 * (4 if precision == "fp32" else 2))
                self.assertEqual(packet["indices"].dtype, np.dtype("int64"))

    def test_first_tied_index_is_required_even_under_standard_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _, values, indices = self.packet(Path(temporary))
            packet = load_packet(path)
            indices[1, 0] = 2  # Same maximum, wrong first index.
            with self.assertRaisesRegex(ValueError, "index mismatch"):
                validate_output(packet, values, indices, ranking_contract="standard")

    def test_signed_zero_bit_and_index_dtype_are_not_relaxed(self):
        for precision in ("fp32", "fp16", "bf16"):
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as temporary:
                path, _, values, indices = self.packet(Path(temporary), precision)
                packet = load_packet(path)
                with self.assertRaisesRegex(ValueError, "must be int64"):
                    validate_output(packet, values, indices.astype("<i4"))
                changed = values.copy(); changed[0, 0] = 0.0
                with self.assertRaisesRegex(ValueError, "bit correspondence"):
                    validate_output(packet, changed, indices)

    def test_manifest_semantics_and_realized_algorithm_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, case, _, _ = self.packet(Path(temporary))
            original = json.loads(path.read_text())
            for change in ({"algorithm": "full_sort"}, {"indices": {**original["indices"], "storage_dtype": "float32"}}):
                path.write_text(json.dumps({**original, **change}))
                with self.assertRaises(ValueError):
                    tensor_receipts(path, case)
            altered = copy.deepcopy(original); altered["semantics"]["stable"] = False
            path.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):
                load_packet(path)

    def test_torch_expression_is_value_first_index_pair_without_casting(self):
        x, output = object(), (object(), object())
        calls = []
        def maximum(value, **kwargs):
            calls.append((value, kwargs))
            return output
        invoke, description = make_program(SimpleNamespace(max=maximum),
            dict(manifest=dict(operation="argmax", dimensions=[4, 5])), [x])
        self.assertIs(invoke(), output)
        self.assertEqual(calls, [(x, dict(dim=-1, keepdim=True))])
        self.assertIn("first int64 index", description)

    def test_two_reduction_model_masks_tail_infinities_and_retains_zero_bits(self):
        for row in ([-np.inf]*33, [np.inf]*33, [-0., 0.]*17, [0., -0.]*17,
                    [-3.]*32+[7.]):
            source = np.array(row, dtype="<f4")
            padded = np.zeros(1 << (len(row)-1).bit_length(), dtype="<f4")
            padded[:len(row)] = source
            valid = np.arange(padded.size) < len(row)
            peak = np.max(np.where(valid, padded, -np.inf))
            winner = np.min(np.where(valid & (padded == peak), np.arange(padded.size, dtype="<i8"), np.iinfo(np.int64).max))
            first = 0
            for i in range(1, len(row)):
                if source[i] > source[first]:
                    first = i
            self.assertEqual(int(winner), first)
            self.assertEqual(source[winner].view("<u4"), source[first].view("<u4"))


if __name__ == "__main__":
    unittest.main()
