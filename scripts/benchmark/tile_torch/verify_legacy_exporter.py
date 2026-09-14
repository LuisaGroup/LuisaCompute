#!/usr/bin/env python3
"""Prove parameterization preserves the original lowering at original shapes."""
import argparse
import json
import os
from pathlib import Path

from run_legacy_matrix import run_command, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LUISA_", "DYLD_"))}
    originals = dict(copy=(8192, 512, 1), add=(2048, 2048, 1), saxpy=(8192, 512, 1),
                     clamp=(8192, 512, 1), exp=(8192, 512, 1), rmsnorm=(2048, 256, 1),
                     sum=(8192, 512, 1), max=(8192, 512, 1), min=(8192, 512, 1),
                     abssum=(8192, 512, 1), absmax=(8192, 512, 1), cumsum=(1024, 256, 1),
                     cummax=(8192, 512, 1), transpose=(72, 72, 1),
                     gemm=(4096, 4096, 4096), gemm_fp16=(512, 512, 512))
    report = dict(cases=[], exporter_sha256={name: sha(args.bin_dir/name) for name in ["emit_legacy_tile", "emit_legacy_tile_sized"]})
    for op, dims in originals.items():
        case = dict(operation=op, dimensions=dims, runs=[])
        report["cases"].append(case)
        for route in ["original", "sized"]:
            executable = args.bin_dir / ("emit_legacy_tile" if route == "original" else "emit_legacy_tile_sized")
            prefix = output / f"{op}-{route}"
            command = [executable, op, prefix] + ([] if route == "original" else list(dims))
            result = run_command(command, Path(str(prefix) + ".log"), env, 60)
            case["runs"].append(result)
            if result.get("exit_code") != 0:
                (output/"results.json").write_text(json.dumps(report, indent=2)+"\n")
                raise RuntimeError(f"legacy capture failed: {op}/{route}")
            result["ast_sha256"] = sha(Path(str(prefix) + ".ast.json"))
            result["metadata"] = json.loads((output/result["log"]).read_text().strip().splitlines()[-1])
        a, b = case["runs"]
        case["same_ast_and_launch"] = a["ast_sha256"] == b["ast_sha256"] and a["metadata"]["dispatch"] == b["metadata"]["dispatch"] and a["metadata"]["block"] == b["metadata"]["block"]
        (output/"results.json").write_text(json.dumps(report, indent=2)+"\n")
        if not case["same_ast_and_launch"]:
            raise RuntimeError(f"dimension parameterization changed original lowering: {op}")
        print(op, "identical AST and launch", flush=True)
    report["passed"] = True
    (output/"results.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
