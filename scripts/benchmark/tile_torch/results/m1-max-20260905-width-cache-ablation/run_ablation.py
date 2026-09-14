"""Fixed 192/416 x reload/cache diagnostic, not another winner search.

Reuse run.py's executed validation, fresh JIT and dual timing. Analyze every
trial; the harness's extra model-selected measurement is retained but excluded
from the factorial effects. Four permutations balance each cell's trial slot
and native/Torch order within each operation. Never rebuild during the run.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
BENCHMARK = HERE.parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    profile = HERE.parent / "m1-max-20260905-service-policy-validation/calibration.json"
    previous = json.loads((HERE.parent / "m1-max-20260905-service-policy-replay/results.json").read_text())
    artifacts = dict.fromkeys(previous["metadata"]["artifacts_sha256"])
    artifacts.update(dict.fromkeys([str(Path(__file__).resolve()), str(BENCHMARK / "run.py")]))
    artifacts = {path: digest(path) for path in artifacts}
    orders = [("192,416", "reload,cache"), ("416,192", "reload,cache"),
              ("416,192", "cache,reload"), ("192,416", "cache,reload")]
    manifest = {"protocol": "fixed factorial; all trials, no timing-selected substitution",
                "shape": [37, 1537], "operations": ["softmax", "rmsnorm", "layernorm"],
                "orders": orders, "samples": 9, "sample_ms": 30, "warmup_ms": 200,
                "artifacts_sha256": artifacts, "rounds": []}
    path = HERE / "manifest.json"
    # Refuse to overwrite evidence. Resuming must inspect the live process and
    # existing rounds, not silently start another collection at the same path.
    with path.open("x") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")
    for index, (widths, caches) in enumerate(orders):
        command = [sys.executable, str(BENCHMARK / "run.py"),
                   "--native", str(ROOT / "cmake-build-tirx/bin/benchmark_tile_tirx"),
                   "--output", str(HERE / f"round-{index + 1}"),
                   "--backends", "metal", "--operations", "softmax,rmsnorm,layernorm",
                   "--row-shapes", "37x1537", "--pipeline-window", "1",
                   "--metal-subgroup-reductions", "--reduction-programs-per-group", "1",
                   "--reduction-lane-elements", "4", "--reduction-unroll", "1",
                   "--tune-group-threads", widths, "--tune-reduction-input-caches", caches,
                   "--reduction-cost-profile", json.loads(profile.read_text())["native_argument"],
                   "--tuning-metric", "model", "--samples", "9", "--sample-ms", "30",
                   "--warmup-ms", "200", "--threads", "8", "--capture-sources",
                   "--metal-device-timing", str(ROOT / "cmake-build-tirx/bin/libluisa-benchmark-metal-timing.dylib")]
        print(f"Fixed ablation round {index + 1}/4", flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        unchanged = all(digest(p) == h for p, h in artifacts.items())
        manifest["rounds"].append({"command": command, "exit_code": result.returncode,
                                   "artifacts_unchanged": unchanged})
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        if result.returncode or not unchanged:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
