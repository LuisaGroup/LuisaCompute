import copy
import unittest

import numpy as np

from compare_mpp import DEFAULT_CONFIG, METRICS, oracle, parse_config, parse_shape, pick_winners, validate_metadata, validate_output


class MppBenchmarkTests(unittest.TestCase):
    def test_shapes(self):
        self.assertEqual(parse_shape("127x193x61"), (127, 193, 61))
        for text in ("0,1,1", "1,2", "1,2,2147483648"):
            with self.assertRaises(ValueError):
                parse_shape(text)

    def test_scope_constraints(self):
        self.assertEqual(parse_config("64,64,4,1,0,1"), DEFAULT_CONFIG)
        self.assertEqual(parse_config("32,32,1,1,0,1,8,2")[-2:], (8, 2))
        for text in ("7,32,1,1,0,1", "8,8,1,1,0,1", "32,32,2,1,0,1,4,1",
                     "32,32,1,1,0,1,8,3", "32,32,1,2,0,1", "32,32,1,1,0,1,8,0"):
            with self.assertRaises(ValueError):
                parse_config(text)

    def test_full_oracle_checks_tail_and_nan(self):
        expected = oracle(np, (7, 19, 13))
        actual = expected.astype(np.float32)
        self.assertEqual(validate_output(np, actual, expected)["checked_elements"], 133)
        for bad in (np.nan, actual[-1, -1] + 1):
            broken = actual.copy()
            broken[-1, -1] = bad
            with self.assertRaises(ValueError):
                validate_output(np, broken, expected)
        with self.assertRaises(ValueError):
            validate_output(np, actual[:, :-1], expected)

    def test_invalid_candidates_cannot_win_and_gpu_is_selection_metric(self):
        rows = [dict(shape=[32, 32, 32], config=[i] * 8, valid=valid,
                     measurement=dict(gpu_throughput_us_p50=gpu, throughput_us_p50=host))
                for i, valid, gpu, host in ((1, False, 0.1, 0.1), (2, True, 2, 10), (3, True, 3, 4))]
        rows.append(dict(shape=[32, 32, 32], config=None, valid=True,
                         measurement=dict(gpu_throughput_us_p50=0.01)))
        self.assertEqual(pick_winners(rows), {"32x32x32": [2] * 8})
        self.assertEqual(pick_winners([rows[0], rows[-1]]), {})

    def test_precise_mpp_metadata(self):
        result = dict(m=128, n=128, k=128, backend="metal", implementation="mpp_tensor_ops_matmul2d",
                      precision="fp32", relaxed_precision=False, fast_math=False, block=[64, 64],
                      execution_simdgroups=4, cooperative_output=True, static_reduction=False,
                      inline_tensors=True, group_simdgroups=4, cohort_rows=1)
        result.update({metric: [1., 2., 3.] for metric in METRICS})
        validate_metadata(result, (128, 128, 128), DEFAULT_CONFIG, 3)
        for field, value in (("precision", "fp16"), ("relaxed_precision", True), ("fast_math", True),
                             ("group_simdgroups", 8), ("cohort_rows", 2), ("n", 127),
                             ("gpu_throughput_us", [1., np.nan, 2.]), ("throughput_us", [1.])):
            changed = copy.deepcopy(result)
            changed[field] = value
            with self.assertRaises(ValueError):
                validate_metadata(changed, (128, 128, 128), DEFAULT_CONFIG, 3)


if __name__ == "__main__":
    unittest.main()
