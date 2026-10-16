#!/usr/bin/env python3
"""Old/new source and correctness controls; these pilots make no speed claims."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
BUILD = Path("/tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin")
OLD = Path("/tmp/luisa-element-multi-baseline.WdH4Ct")
TVM = Path("/tmp/luisa-tvm-mpp.VaKmzx/build/lib")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = ROOT / "controls.json"
    if receipt.exists():
        raise ValueError("refusing to overwrite a control receipt")
    artifacts = [p for directory in (BUILD, OLD, TVM) for p in directory.iterdir()
                 if p.is_file() and (p.suffix in (".so", ".dylib") or p.name == "benchmark_tile_tirx")]
    before = {str(p.resolve()): digest(p) for p in artifacts}
    report = dict(artifacts_before=before, runs=[], source_controls=[],
                  interpretation="correctness and generated-source identity; pilot timings are not optimization claims")
    cohorts = [
        ("pointwise", "metal", "add,gelu_add", "37x1537,1024x4096", []),
        ("reductions", "metal", "softmax,rmsnorm,layernorm", "37x1537,1024x4096",
         ["--metal-subgroup-reductions", "--reduction-lane-elements", "4", "--cache-reduction-inputs"]),
        ("cpu", "cpu", "sigmoid_pair,gelu_pair", "1x127,37x1537", []),
        ("large-old", "metal", "sigmoid_pair,gelu_pair", "1024x4096,4096x4096", []),
    ]
    for cohort, backend, operations, shapes, flags in cohorts:
        for variant in (("old",) if cohort == "large-old" else ("old", "new")):
            binary = (OLD if variant == "old" else BUILD) / "benchmark_tile_tirx"
            output = ROOT / f"control-{cohort}-{variant}"
            command = [sys.executable, str(REPO / "scripts/benchmark/tile_torch/run.py"), "--native", str(binary),
                       "--backends", backend, "--operations", operations, "--row-shapes", shapes,
                       "--samples", "3", "--sample-ms", "10", "--warmup-ms", "40",
                       "--element-grid", "auto", "--capture-sources", "--output", str(output), *flags]
            search = f"{OLD}:{TVM}:{BUILD}" if variant == "old" else str(TVM)
            env = dict(os.environ, DYLD_LIBRARY_PATH=search)
            tested = subprocess.run(command, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, timeout=600)
            (ROOT / f"control-{cohort}-{variant}.log").write_text(tested.stdout)
            rows = json.loads((output / "results.json").read_text())["results"] if (output / "results.json").exists() else []
            expected = len(operations.split(",")) * len(shapes.split(","))
            passed = tested.returncode == 0 and len(rows) == expected and all(r["valid"] is True for r in rows)
            entry = dict(cohort=cohort, variant=variant, command=command, loader_search=search,
                         exit_code=tested.returncode, expected_cases=expected, passed=passed)
            report["runs"].append(entry)
            receipt.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(entry), flush=True)
            if not passed:
                print(tested.stdout, flush=True)
    if all(r["passed"] for r in report["runs"]):
        for cohort, _, _, _, _ in cohorts:
            old_rows = json.loads((ROOT / f"control-{cohort}-old/results.json").read_text())["results"]
            new_path = ROOT / ("new-reference" if cohort == "large-old" else f"control-{cohort}-new") / "results.json"
            new_rows = {r["name"]: r for r in json.loads(new_path.read_text())["results"]}
            for old in old_rows:
                new = new_rows[old["name"]]
                same = old["native_source_sha256"] == new["native_source_sha256"]
                report["source_controls"].append(dict(cohort=cohort, name=old["name"], identical=same,
                                                     old=old["native_source_sha256"], new=new["native_source_sha256"]))
    report["artifacts_after"] = {str(p.resolve()): digest(p) for p in artifacts}
    report["artifacts_unchanged"] = before == report["artifacts_after"]
    report["passed"] = (report["artifacts_unchanged"] and all(r["passed"] for r in report["runs"]) and
                        len(report["source_controls"]) == 18 and all(r["identical"] for r in report["source_controls"]))
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("artifacts")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
