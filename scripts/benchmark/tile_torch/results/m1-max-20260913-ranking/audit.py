#!/usr/bin/env python3
"""Read-only audit of retained ranking outputs; does not execute kernels.

Extract evidence.tar.gz into a new directory and pass that directory. Checks
all payload values/indices using an independent NumPy stable-order oracle.
Runtime guard metadata is a receipt, not retained guarded allocation bytes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    checked = []
    for cohort in ("smoke", "matrix"):
        path = root / cohort
        report = json.loads((path / "results.json").read_text())
        assert report["status"] == "completed"
        assert report["source_unchanged"] and report["binary_unchanged"]
        assert len(report["results"]) == (5 if cohort == "smoke" else 60)
        assert all(row["status"] == "OK" for row in report["results"])
        for row in report["results"]:
            rows, columns, count = row["dimensions"]
            key = row["case"]
            inputs = path / (key + ".input.f32")
            assert sha(inputs) == report["input_sha256"][key]
            x = np.fromfile(inputs, dtype="<f4").reshape(rows, columns)
            reference_input = (((np.arange(columns)[None, :] * 37 + np.arange(rows)[:, None] * 17) % 31) - 15) * .25
            assert np.array_equal(x, reference_input)
            visit = path / f"{key}-r{row['round']}-{row['route']}"
            values = visit / "output.values.f32"
            indices = visit / "output.indices.i64"
            assert sha(values) == row["output_sha256"]["values"]
            assert sha(indices) == row["output_sha256"]["indices"]
            v = np.fromfile(values, dtype="<f4").reshape(rows, count)
            i = np.fromfile(indices, dtype="<i8").reshape(rows, count)
            assert np.isfinite(x).all() and np.isfinite(v).all()
            assert (i >= 0).all() and (i < columns).all()
            assert all(np.unique(selected).size == count for selected in i)
            assert np.array_equal(v, np.take_along_axis(x, i, axis=1))
            order = np.argsort(-x if row["direction"] == "descending" else x, axis=1, stable=True)[:, :count]
            assert np.array_equal(v, np.take_along_axis(x, order, axis=1))
            stable = row["route"].startswith(("xir-", "tirx-")) or row["operation"] == "sort"
            if stable:
                assert np.array_equal(i, order)
            measurement = row["measurement"]
            for source, metric in (("throughput_us", "e2e_batch_us"), ("latency_us", "e2e_single_us")):
                samples = measurement[source]
                assert len(samples) == report["samples"]
                assert all(type(t) in (int, float) and np.isfinite(t) and t > 0 for t in samples)
                assert statistics.median(samples) == row["metrics"][metric]
            checked.append({"cohort": cohort, "case": key, "route": row["route"], "elements": rows * count})
    return {"status": "passed", "output_pairs": len(checked), "checks": checked,
            "scope": "Full retained inputs/values/indices, payload hashes and host medians. GPU schemas were validated by the frozen driver, not independently re-audited here. Runtime guards are execution receipts only; no native execution or performance claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2))
