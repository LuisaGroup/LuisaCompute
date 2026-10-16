"""Host-only explicit packed-sort identity and unchanged stable Torch contract."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import test_cuda_matrix as existing
from cuda_matrix import tensor_receipts, validate_case
from cuda_torch_baseline import make_program


class CudaPackedSortTests(unittest.TestCase):
    def test_explicit_choice_and_default_are_distinct(self):
        row = existing.CudaMatrixTests.case()
        self.assertEqual(validate_case(row, 0)['ranking_algorithm'], 'full_sort_prefix')
        for op in ('sort', 'topk'):
            for precision in ('fp32', 'fp16', 'bf16'):
                row = existing.CudaMatrixTests.case(operation=op, precision=precision, dimensions=[3, 33, 33 if op == 'sort' else 7],
                                                    tile=[1, 64, 1], ranking_algorithm='packed_fp32')
                self.assertEqual(validate_case(row, 0)['ranking_algorithm'], 'packed_fp32')

    def test_non_ranking_operation_cannot_claim_packed_sort(self):
        with self.assertRaisesRegex(ValueError, 'only applicable to ranking'):
            validate_case(existing.CudaMatrixTests.case(operation='rmsnorm', ranking_algorithm='packed_fp32'), 0)

    def test_manifest_requires_actual_packed_algorithm_label(self):
        row = existing.CudaMatrixTests.case(ranking_algorithm='packed_fp32')
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            existing.CudaMatrixTests.write_fixture(directory, row)
            with self.assertRaisesRegex(ValueError, 'realized ranking algorithm'):
                tensor_receipts(directory / 'manifest.json', row)
            path = directory / 'manifest.json'
            manifest = json.loads(path.read_text())
            manifest['algorithm'] = 'stable_packed_fp32_full_sort_prefix'
            path.write_text(json.dumps(manifest))
            self.assertTrue(tensor_receipts(path, row))

    def test_torch_keeps_original_dtype_and_stable_sort(self):
        for op in ('sort', 'topk'):
            calls = []
            source = np.asarray([[3., 3., -1., 2.]], dtype=np.float16)
            def sort(value, **kwargs):
                calls.append((value, kwargs))
                return np.asarray([[3., 3., 2., -1.]], dtype=np.float16), np.asarray([[0, 1, 3, 2]], dtype=np.int64)
            packet = dict(manifest=dict(operation=op, dimensions=[1, 4, 2], precision='fp16', ranking_algorithm='packed_fp32'))
            invoke, _ = make_program(SimpleNamespace(sort=sort), packet, [source, object(), object()], ranking_contract='stable')
            values, indices = invoke()
            self.assertIs(calls[0][0], source)
            self.assertEqual(calls[0][1], dict(dim=-1, descending=True, stable=True))
            self.assertEqual(values.dtype, np.dtype('float16'))
            self.assertEqual(indices.tolist(), [[0, 1]] if op == 'topk' else [[0, 1, 3, 2]])


if __name__ == '__main__':
    unittest.main()
