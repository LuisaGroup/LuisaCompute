import copy
import argparse
import unittest

from compare_xir import check_metadata, positive_triple


class XIRMetadataTests(unittest.TestCase):
    def row(self):
        return dict(implementation="tile_xir_simd", backend="cpu", precision="fp32", fast_math=False,
                    relaxed_precision=False, block=[1, 1, 8], m=32, n=32, k=32, planner_policy="canonical",
                    timing="synchronized_host_wall", batch_policy="one_runtime_command_list_per_batch",
                    realization="XIR SSA W8 64 workers/block; uncalibrated cost 1, root order [0,1]",
                    repetitions=4, throughput_us=[1.0, 2.0], latency_us=[3.0, 4.0])

    def test_accepts_exact_contract(self):
        check_metadata(self.row(), (32, 32, 32), "canonical", 2)

    def test_explicit_tile_contract(self):
        row = self.row() | {"block": [4, 1, 8], "planner_policy": "planned"}
        check_metadata(row, (32, 32, 32), "planned", 2, (4, 1, 8))
        with self.assertRaises(ValueError):
            check_metadata(row, (32, 32, 32), "planned", 2)

    def test_shape_parser(self):
        self.assertEqual(positive_triple("127,193,61"), (127, 193, 61))
        for text in ("", "1,2", "1,2,3,4", "1,0,3", "-1,2,3", "1,x,3"):
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                positive_triple(text)

    def test_rejects_wrong_realization(self):
        for key, value in {"implementation": "tile_tirx", "fast_math": True, "m": 31,
                           "realization": "XIR SSA W8 32 workers/block; uncalibrated cost 1, root order [0,1]"}.items():
            row = self.row() | {key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_metadata(row, (32, 32, 32), "canonical", 2)

    def test_rejects_invalid_samples(self):
        for values in ([], [1.0], [1.0, float("nan")], [1.0, -1.0], [0.0, 1.0]):
            row = copy.deepcopy(self.row())
            row["throughput_us"] = values
            with self.subTest(values=values), self.assertRaises(ValueError):
                check_metadata(row, (32, 32, 32), "canonical", 2)


if __name__ == "__main__":
    unittest.main()
