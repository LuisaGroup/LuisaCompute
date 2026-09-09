#!/usr/bin/env python3
"""Inspect real CPU dispatches and native samples, not a performance benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    if min(args.threads, args.rows, args.width) <= 0:
        parser.error("threads, rows and width must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(args.output.resolve() / "inductor")
    import torch
    torch.manual_seed(20260908)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    package = Path(torch.__file__).parent
    library = package / "lib/libtorch_cpu.dylib"
    with library.open("rb") as f:
        library_hash = hashlib.file_digest(f, "sha256").hexdigest()
    report = dict(purpose="dispatch and code inspection; profiler times are not benchmark results",
                  version=torch.__version__, commit=torch.version.git_version,
                  library=str(library), library_sha256=library_hash,
                  threads=args.threads, rows=args.rows, width=args.width,
                  cpu_capability=torch.backends.cpu.get_cpu_capability(),
                  build_config=torch.__config__.show(),
                  parallel_info=torch.__config__.parallel_info(), cases=[])
    x, u = torch.randn(args.rows, args.width), torch.randn(args.rows, args.width)
    gamma, beta = torch.randn(args.width), torch.randn(args.width)
    a, b = torch.randn(1024, 1024), torch.randn(1024, 1024)
    cases = {
        "softmax": lambda: torch.softmax(x, dim=-1),
        "layernorm": lambda: torch.nn.functional.layer_norm(x, (args.width,), gamma, beta, 1e-5),
        "rmsnorm": lambda: torch.nn.functional.rms_norm(x, (args.width,), gamma, 1e-5),
        "swiglu": lambda: torch.nn.functional.silu(x) * u,
        "gemm": lambda: torch.mm(a, b),
    }
    for name, invoke in cases.items():
        for _ in range(5):
            invoke()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as profile:
            output = invoke()
        item = dict(name=name, shape=list(output.shape), output_finite=bool(torch.isfinite(output).all()),
                    operators=[dict(name=e.key, calls=e.count, shapes=e.input_shapes)
                               for e in profile.key_averages(group_by_input_shape=True)])
        trace = args.output / (name + ".trace.json")
        profile.export_chrome_trace(str(trace))
        sample_path = args.output.resolve() / (name + ".sample.txt")
        with (args.output / (name + ".sample.log")).open("w") as log:
            sampler = subprocess.Popen(["sample", str(os.getpid()), "2", "1", "-file", str(sample_path)], stdout=log, stderr=log)
            start, iterations = time.monotonic(), 0
            while sampler.poll() is None and time.monotonic() - start < 30:
                invoke()
                iterations += 1
            if sampler.poll() is None:
                sampler.kill()
            item.update(sample_exit_code=sampler.wait(), sampled_invocations=iterations)
        report["cases"].append(item)
        (args.output / "inspection.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, "sample exit", item["sample_exit_code"], "operators", [e["name"] for e in item["operators"]], flush=True)
    if args.compile:
        for name in ("softmax", "rmsnorm", "swiglu"):
            item = dict(name=name)
            try:
                compiled = torch.compile(cases[name], backend="inductor", fullgraph=True)
                actual, expected = compiled(), cases[name]()
                torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)
                item["correctness"] = "complete eager comparison passed"
            except Exception as error:
                item["error"] = str(error)
            report.setdefault("compiled", []).append(item)
            (args.output / "inspection.json").write_text(json.dumps(report, indent=2) + "\n")
            print("compiled", name, item.get("correctness", item.get("error")), flush=True)
    if any(not item["output_finite"] or item["sample_exit_code"] != 0 for item in report["cases"]):
        raise SystemExit("incomplete eager inspection; see inspection.json")
    if any("error" in item for item in report.get("compiled", [])):
        raise SystemExit("incomplete compiled inspection; see inspection.json")


if __name__ == "__main__":
    main()
