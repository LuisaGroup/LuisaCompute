"""Metadata-only matrix work/allocation/grid boundaries; never allocates large tensors."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cuda_matrix import validate_case
from cuda_torch_baseline import check_matrix_limits, load_packet


class CudaMatrixLimitTests(unittest.TestCase):
    @staticmethod
    def case(operation, dims, tile, precision="fp32"):
        return dict(id="matrix-limits", operation=operation, dimensions=dims, tile=tile,
                    precision=precision, seed=19, pattern="random", fast_math=False)

    def check_both(self, operation, dims, tile, accepted):
        for precision in ("fp32", "fp16", "bf16"):
            with self.subTest(op=operation, dimensions=dims, tile=tile, precision=precision):
                row = self.case(operation, dims, tile, precision)
                if accepted:
                    self.assertEqual(validate_case(row, 0), row)
                    check_matrix_limits(operation, dims, tile)
                else:
                    with self.assertRaises(ValueError):
                        validate_case(row, 0)
                    with self.assertRaises(ValueError):
                        check_matrix_limits(operation, dims, tile)

    def test_requested_large_shapes_and_exact_work_limit(self):
        for op, dims, tile in [("gemm", [1024,1024,1024], [32,32,32]),
                               ("gemm", [2048,2048,2048], [64,64,32]),
                               ("gemm", [128,4096,4096], [32,64,32]),
                               ("gemm", [4096,4096,1024], [32,32,32]),
                               ("bmm", [2,2048,2048,2048], [64,64,32]),
                               ("gemv", [65536,1,256], [1,1,256]),
                               ("bmm", [65536,1,1,1], [1,1,1])]:
            self.check_both(op, dims, tile, True)

    def test_work_one_step_over_limit_with_all_allocations_still_valid(self):
        self.check_both("gemm", [4096,4096,1025], [32,32,32], False)
        self.check_both("bmm", [2,2048,2048,2049], [64,64,32], False)

    def test_each_tensor_limit_is_independent_of_work_limit(self):
        for op, shapes in [("gemm", [[4097,1,4096], [1,4097,4096], [4097,4096,1]]),
                           ("bmm", [[2,2049,1,4096], [2,1,2049,4096], [2,2049,4096,1]]),
                           ("gemv", [[65536,1,257]])]:
            for dims in shapes:
                self.check_both(op, dims, [32,32,32], False)

    def test_positive_integer_dimension_and_schedule_bounds(self):
        for dims in ([0,1,1], [-1,1,1], [True,1,1], [1.0,1,1], [65537,1,1], [2**63-1]*3):
            self.check_both("gemm", dims, [16,16,8], False)
        for tile in ([0,16,8], [256,16,8], [16,256,8], [16,16,512], [16,16,True]):
            self.check_both("gemm", [31,37,19], tile, False)
        self.check_both("gemv", [37,1,129], [16,1,1024], True)
        self.check_both("gemv", [37,2,129], [16,1,1024], False)
        self.check_both("bmm", [1,31,37,19], [16,16,512], False)

    def test_legacy_non_power_of_two_gemm_and_gemv_schedules(self):
        for op,dims,tile in [("gemm", [31,37,19], [3,5,7]),
                             ("gemv", [37,1,129], [3,1,65])]:
            self.check_both(op,dims,tile,True)
        self.check_both("bmm",[1,31,37,19],[3,5,7],False)

    def test_cuda_grid_axes_boundaries(self):
        self.check_both("gemm", [65536,1,1], [1,1,1], True)
        self.check_both("gemm", [1,65535,1], [1,1,1], True)
        self.check_both("gemm", [1,65536,1], [1,1,1], True)
        self.check_both("gemm", [1,65536,1], [1,2,1], True)
        # Case selection is route-neutral: a mixed matrix must preserve SIMD.
        for backend in (None, "simd"):
            check_matrix_limits("gemm", [1,65536,1], [1,1,1], backend)
        with self.assertRaisesRegex(ValueError,"grid"):
            check_matrix_limits("gemm", [1,65536,1], [1,1,1], "cuda")
        check_matrix_limits("gemm", [1,65536,1], [1,2,1], "cuda")
        for dims in ([1,65536,1,1], [1,1,65536,1]):
            self.check_both("bmm", dims, [1,1,1], False)
            self.check_both("bmm", dims, [2,2,1], True)

    def test_nonmatrix_work_budgets_are_not_increased(self):
        for op,dims,tile in [("rmsnorm", [65536,8192], [1,8192,1]),
                             ("sort", [1024,4096,4096], [1,4096,1]),
                             ("attention", [1,4,1,1024,1024,64,64], [16,32,1])]:
            with self.subTest(op=op), self.assertRaises(ValueError):
                validate_case(self.case(op,dims,tile),0)

    def test_packet_limits_and_shape_drift_fail_before_tensor_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"manifest.json"
            base=dict(schema=1, backend="cuda", operation="gemm", dimensions=[2,2,2], tile=[2,2,2], precision="fp32",
                      endianness="little", semantics=dict(accumulation="float32"),
                      inputs=[dict(name=f"input{i}",path="must-not-read.bin",storage_dtype="float32",shape=s)
                              for i,s in enumerate([[2,2],[2,2],[1]])],
                      output=dict(path="output.f32",storage_dtype="float32",shape=[2,2]))
            malformed=[]
            for dims in ([4096,4096,1025], [4097,4096,1], [1,65536,1]):
                m=copy.deepcopy(base); m["dimensions"]=dims
                if dims==[1,65536,1]: m["tile"]=[1,1,1]
                malformed.append(m)
            m=copy.deepcopy(base); m.pop("tile"); malformed.append(m)
            m=copy.deepcopy(base); m["inputs"][0]["shape"]=[65536,65536]; malformed.append(m)
            m=copy.deepcopy(base); m["output"]["shape"]=[4097,4096]; malformed.append(m)
            for manifest in malformed:
                path.write_text(json.dumps(manifest),encoding="utf-8")
                with self.subTest(manifest=manifest), patch("numpy.fromfile", side_effect=AssertionError("tensor read before metadata checks")) as read:
                    with self.assertRaises(ValueError): load_packet(path)
                    read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
