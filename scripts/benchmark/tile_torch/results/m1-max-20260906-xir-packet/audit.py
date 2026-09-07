#!/usr/bin/env python3
"""Recompute the frozen packet-proof comparison; never trust stored summaries."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory):
    report = json.loads((directory / "results.json").read_text())
    meta = report["metadata"]
    shapes = [(32, 32, 32), (128, 128, 128), (512, 512, 512), (1024, 1024, 1024),
              (128, 2048, 512), (127, 193, 61)]
    policies = ("planned", "baseline", "torch")
    orders = list(itertools.permutations(policies))
    assert meta["rounds"] == 6 and meta["samples"] == 5 and meta["block"] == [1, 1, 8]
    assert meta["control"] == "baseline" and meta["requested_threads"] == 8
    assert meta["artifacts_unchanged"] is True
    assert all(digest(Path(path)) == sha for path, sha in meta["artifacts_sha256"].items())
    rows = report["results"]
    assert len(rows) == 108
    inventory = {}
    for i, row in enumerate(rows):
        shape = tuple(row["shape"])
        r, policy = row["round"], row["policy"]
        assert shape in shapes and r in range(6) and policy in policies
        key = (shape, r, policy)
        assert key not in inventory and row["valid"] is True
        inventory[key] = row
        order = orders[(r + shapes.index(shape)) % 6]
        assert tuple(row["order"]) == order
        case_order = shapes[r % 6:] + shapes[:r % 6]
        assert i == r * 18 + case_order.index(shape) * 3 + order.index(policy)
        measurement = row["measurement"]
        for metric in ("throughput_us", "latency_us"):
            values = measurement[metric]
            assert len(values) == 5 and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values)
            assert math.isclose(statistics.median(values), measurement[metric + "_p50"], rel_tol=1e-12)
        assert digest(directory / row["output"]) == row["output_sha256"]
        if policy != "torch":
            assert digest(directory / row["source"]) == row["source_sha256"]
            assert measurement["block"] == [1, 1, 8] and measurement["planner_policy"] == "planned"
            assert measurement["fast_math"] is False and measurement["relaxed_precision"] is False
            assert measurement["timing"] == "synchronized_host_wall"
            counts = re.search(r"contiguous reads=(\d+), broadcasts=(\d+)", measurement["realization"])
            expected_counts = (8, 8) if policy == "planned" and shape[-2] % 8 == 0 else (0, 0)
            assert counts and tuple(map(int, counts.groups())) == expected_counts
    summary = []
    checked = 0
    for shape in shapes:
        m, n, k = shape
        values = lambda count, seed: (((np.arange(count, dtype=np.int64) * seed + 17) % 127 - 63) / 64).astype(np.float32)
        expected = values(m * k, 5).reshape(m, k).astype(np.float64) @ values(k * n, 11).reshape(k, n).astype(np.float64)
        errors = []
        for r in range(6):
            native = [inventory[shape, r, p] for p in policies[:2]]
            # Same chosen execution map and relative-work score; only the
            # emitter's proven memory classifications may change.
            plans = [re.sub(r"contiguous reads=\d+, broadcasts=\d+", "COUNTS", v["measurement"]["realization"]) for v in native]
            assert plans[0] == plans[1]
            for policy in policies:
                row = inventory[shape, r, policy]
                actual = np.fromfile(directory / row["output"], dtype=np.float32).reshape(m, n)
                diff = np.abs(actual.astype(np.float64) - expected)
                assert np.isfinite(actual).all() and np.all(diff <= 1e-4 + 1e-4 * np.abs(expected))
                assert row["atol"] == row["rtol"] == 1e-4 and row["checked_elements"] == m * n
                assert float(diff.max()) == row["max_abs_error"]
                errors.append(float(diff.max()))
                checked += m * n
        result = {"shape": shape, "max_abs_error": max(errors), "metrics": {}}
        for metric in ("throughput_us", "latency_us"):
            times = {p: [statistics.median(inventory[shape, r, p]["measurement"][metric]) for r in range(6)] for p in policies}
            detail = {"median_us": {p: statistics.median(v) for p, v in times.items()}}
            for p in ("baseline", "torch"):
                ratios = [a / b for a, b in zip(times["planned"], times[p])]
                detail["planned_over_" + p] = {"median": statistics.median(ratios), "min": min(ratios), "max": max(ratios),
                                                  "slower_rounds": sum(r > 1 for r in ratios)}
            result["metrics"][metric] = detail
        result["compile_ms"] = {p: statistics.median(inventory[shape, r, p]["measurement"]["compile_ms"] for r in range(6)) for p in policies[:2]}
        result["identical_llvm_between_arms"] = all(inventory[shape, r, "planned"]["source_sha256"] == inventory[shape, r, "baseline"]["source_sha256"] for r in range(6))
        summary.append(result)
    return {"rows": len(rows), "checked_elements": checked, "artifacts": len(meta["artifacts_sha256"]),
            "report_sha256": digest(directory / "results.json"), "summary": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).parent / "replay")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
