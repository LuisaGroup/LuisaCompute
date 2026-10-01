"""Metadata-only optional row-blocking tests; no Torch/CUDA import or large tensors."""
import unittest
from cuda_matrix import validate_case


class CudaRowBlockingTests(unittest.TestCase):
    @staticmethod
    def case(operation='scan', rows=17, width=65, block_rows=4, padded=128, precision='fp32'):
        return dict(id='blocked-row-host',operation=operation,dimensions=[rows,width],
                    tile=[block_rows,padded,1],precision=precision,seed=20261001,pattern='cancellation',fast_math=False)

    def test_explicit_schedules_keep_every_dtype_and_row_operation(self):
        for operation in ('scan','reduce_sum','reduce_max'):
            for precision in ('fp32','fp16','bf16'):
                for block_rows in (1,4,8):
                    row=self.case(operation=operation,precision=precision,block_rows=block_rows)
                    self.assertEqual(validate_case(row,0),row)

    def test_tails_and_fewer_rows_than_one_program_are_not_rejected(self):
        for rows,width,padded in ((1,1,1),(3,65,128),(17,65,128),(128,65,128),(128,8192,8192)):
            for block_rows in (1,4,8):
                validate_case(self.case(rows=rows,width=width,padded=padded,block_rows=block_rows),0)

    def test_other_row_families_keep_single_row_contract(self):
        for operation in ('scan_ordered','rmsnorm','layernorm','softmax','masked_softmax','swiglu','gelu_residual','rope'):
            width,padded=(64,32) if operation=='rope' else (65,128)
            validate_case(self.case(operation=operation,width=width,padded=padded,block_rows=1),0)
            for block_rows in (4,8):
                with self.subTest(operation=operation,block_rows=block_rows),self.assertRaisesRegex(ValueError,'row schedule'):
                    validate_case(self.case(operation=operation,width=width,padded=padded,block_rows=block_rows),0)

    def test_invalid_block_width_allocation_and_dimension_gates_remain(self):
        for block_rows in (0,-1,2,3,16,True,4.0):
            with self.subTest(block_rows=block_rows),self.assertRaises(ValueError):
                validate_case(self.case(block_rows=block_rows),0)
        for row in (self.case(padded=64),self.case(padded=32768),
                    self.case(rows=65536,width=1024,padded=1024),self.case(rows=0)):
            with self.subTest(row=row),self.assertRaises(ValueError):validate_case(row,0)

    def test_non_power_of_two_columns_remain_cross_backend_compatible(self):
        # Native capability is checked by the real compiler, not a route-neutral parser.
        for block_rows in (1,4,8):validate_case(self.case(padded=65,block_rows=block_rows),0)


if __name__=='__main__':unittest.main()
