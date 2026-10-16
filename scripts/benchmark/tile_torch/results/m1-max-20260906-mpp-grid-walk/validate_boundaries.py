#!/usr/bin/env python3
"""Check uint32 launch arithmetic boundaries before/without large allocation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile

from experiment import BUILD, CONFIG, RECTANGLES, ROOT, digest, measure, oracle


def main():
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[variable] = "8"
    import numpy as np
    binary = BUILD / "benchmark_tile_mpp"
    result = {"binary": str(binary), "sha256_before": digest(binary), "valid_outputs": [], "rejected_requests": []}
    # 2 * INT32_MAX fits uint32 but not int32. Small physical grids exercise
    # that expression without allocating a large tensor or padding the launch.
    RECTANGLES["boundary"] = (2**31 - 1, 2**31 - 1)
    for shape in ((1, 1, 1), (129, 33, 7)):
        row = measure(np, shape, "boundary", oracle(np, shape), CONFIG, 1, 1, 1)
        result["valid_outputs"].append(row)
    with tempfile.TemporaryDirectory(prefix="mpp-walk-boundaries-") as temporary:
        output = Path(temporary) / "output.f32"
        requests = [((1, 1, 1), (0, 1), "positive int32"),
                    ((1, 1, 1), (1, 0), "positive int32"),
                    ((1, 65, 1), (2**31 - 1, 1), "uint32 program coordinates"),
                    ((2**31 - 1, 2**31 - 1, 1), (1, 1), "uint32 program coordinates")]
        for shape, walk, reason in requests:
            command = [str(binary), "fp32", *map(str, shape), "1", "1", "1", str(output),
                       "32", "32", "1", "1", "0", "0", "1", "4", "4", *map(str, walk)]
            process = subprocess.run(command, capture_output=True, text=True, timeout=30)
            row = dict(command=command, returncode=process.returncode, stderr=process.stderr,
                       passed=process.returncode == 1 and reason in process.stderr and not output.exists())
            result["rejected_requests"].append(row)
    result["sha256_after"] = digest(binary)
    result["passed"] = (result["sha256_before"] == result["sha256_after"] and
                        all(row["valid"] for row in result["valid_outputs"]) and
                        all(row["passed"] for row in result["rejected_requests"]))
    (ROOT / "boundaries.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "valid_outputs": len(result["valid_outputs"]),
                      "rejected_requests": len(result["rejected_requests"])}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
