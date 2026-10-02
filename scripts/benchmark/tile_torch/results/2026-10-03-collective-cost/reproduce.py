"""Verify this portable evidence package and reproduce all three offline fits.

Requires Python 3.11+ and NumPy. Does not load Torch/CUDA, compile, or run kernels.
"""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code/collective-worker-calibration-v3-prep"))
from public_model import numerical_report
import calibrate_v3 as v3


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    receipts = read(ROOT / "receipts.json")
    for name, receipt in receipts["public_files"].items():
        path = ROOT / name
        data = path.read_bytes()
        assert len(data) == receipt["bytes"] and sha(data) == receipt["sha256"], f"published file changed: {name}"
    summaries = [(ROOT / f"inputs/summary-{index}.json").read_bytes() for index in (0, 1)]
    packets = [(sha(data), json.loads(data)) for data in summaries]
    device = read(ROOT / "inputs/device.json")
    actual = [v3.v1.calibrate(packets[0][1], device), v3.v2.calibrate(packets, device), v3.calibrate(packets, device)]
    results = []
    for version, report in enumerate(actual, 1):
        expected = read(ROOT / f"models/v{version}.json")
        observed = numerical_report(report, version)
        assert observed == expected["numerical_report"], f"v{version} numerical model/decision reproduction differs"
        results.append(dict(version=version, exact_numerical_projection_equal=True,
                            training_rows=len(observed["training_observations"]),
                            logo_groups=len(observed["leave_one_geometry_group_out"]["folds"])))
    # The original opaque profile ID includes unredacted input hashes and local
    # provenance. It is retained only as provenance, never presented as a hash
    # of a redacted public file or newly derived report.
    result = dict(status="passed", public_files_verified=len(receipts["public_files"]), versions=results,
                  v3_geometry_equal_weight_logo=numerical_report(actual[2], 3)["geometry_equal_weight_logo"]["policy_geomean"],
                  original_v3_profile_id=receipts["original_v3_profile_id"],
                  independent_heldout_measured=False, execution="offline CPU fits only")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
