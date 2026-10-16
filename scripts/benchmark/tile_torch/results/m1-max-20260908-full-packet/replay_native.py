#!/usr/bin/env python3
"""Replay real Luisa ORC objects and an actual one-thread Inductor entry.

This is single-thread native-entry wall time, not the eight-thread Runtime
benchmark and not a hardware cycle counter. All compilation and allocation
is outside timing; the C++ loop retains native-call/launch-record-reset cost.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compare_llm import reference, validate_output


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="actual capture directory")
    parser.add_argument("--candidate", type=Path, required=True, help="actual capture directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[5]
    source = Path(__file__).resolve().parent.parent / "m1-max-20260908-xir-packet-local/replay_native.cpp"
    helper_path = args.output / "replay.dylib"
    commands = []

    def run(command):
        commands.append(command)
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)

    run(["clang++", "-std=c++20", "-O3", "-dynamiclib", "-I" + str(root / "src"), str(source), "-o", str(helper_path)])
    helper_library = ctypes.CDLL(str(helper_path))
    helper = helper_library.replay_native
    pointer, u32 = ctypes.c_void_p, ctypes.c_uint32
    helper.argtypes = [pointer, u32, u32, u32, u32, u32, pointer, pointer, pointer, pointer,
                       ctypes.c_size_t, u32, pointer, ctypes.POINTER(ctypes.c_uint64)]
    helper.restype = ctypes.c_int
    entries, libraries, hashes = {}, [], {}
    arrays = expected = input_hashes = dimensions = None
    for name in ("baseline", "candidate"):
        directory = getattr(args, name).resolve()
        measurement = json.loads((directory / "measurement.json").read_text())
        if measurement["operation"] != "rmsnorm" or len(measurement["dimensions"]) != 2:
            raise ValueError("this replay validates only the inspected RMSNorm ABI")
        if dimensions is not None and measurement["dimensions"] != dimensions:
            raise ValueError("capture dimensions differ")
        dimensions = measurement["dimensions"]
        rows, columns = dimensions
        llvm = (directory / "kernel.ll").read_text()
        symbol = "llm_rows.packet_batch.blocks"
        if f"define dso_local void @{symbol}(" not in llvm or "W8," not in measurement["realization"]:
            raise ValueError("not the verified W8 block-batch entry")
        block = int(re.search(r"W8, (\d+) workers/block", measurement["realization"])[1])
        workspace = re.search(r"private_workspace_bytes=(\d+)", measurement["realization"])
        workspace = int(workspace[1]) if workspace else 0
        objects = list((directory / "object").glob("*.o"))
        if len(objects) != 1:
            raise ValueError("expected one actual ORC object")
        undefined = run(["/opt/homebrew/opt/llvm@21/bin/llvm-nm", "--undefined-only", "--just-symbol-name", str(objects[0])])
        system_symbols = set(undefined.split())
        # Full-packet constant propagation can expose an exact contiguous copy
        # which LLVM lowers to libc memcpy. Keep that emitted call in timing,
        # link the actual ORC object normally, and reject every other import.
        # LLJIT's default ProcessSymbols dylib resolves the same host libc.
        if sys.platform != "darwin" or system_symbols - {"_memcpy"}:
            raise ValueError("unsupported ORC imports: " + undefined)
        library_path = args.output / (name + ".dylib")
        run(["clang++", "-dynamiclib", str(objects[0]), "-o", str(library_path)])
        library = ctypes.CDLL(str(library_path))
        libraries.append(library)
        local = re.search(r"local_lanes=(\d+)", measurement["realization"])
        local = int(local[1]) if local else 1
        if local not in (1, 8):
            raise ValueError("only whole-program or full W8 local distribution is admitted")
        entries[name] = (ctypes.cast(getattr(library, symbol), pointer), 0, block, workspace, local)
        hashes[name] = dict(object_sha256=digest(objects[0]), library_sha256=digest(library_path),
                            llvm_sha256=digest(directory / "kernel.ll"), source_capture=str(directory),
                            undefined_system_symbols=sorted(system_symbols),
                            linked_libraries=run(["otool", "-L", str(library_path)]))
        paths = [directory / f"output.f32.input{i}.f32" for i in range(3)]
        current = [digest(p) for p in paths]
        if input_hashes is not None and input_hashes != current:
            raise ValueError("capture input bits differ")
        input_hashes = current
        arrays = [np.fromfile(p, dtype=np.float32).reshape(s) for p, s in zip(paths, ((rows, columns), (1, columns), (1, columns)))]
        expected = reference("rmsnorm", (rows, columns), arrays)

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(args.output / "inductor")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    x, gamma = torch.from_numpy(arrays[0]), torch.from_numpy(arrays[1][0])
    compiled = torch.compile(lambda x, gamma: torch.nn.functional.rms_norm(x, (columns,), gamma, 1e-5), fullgraph=True)
    validate_output(compiled(x, gamma).numpy(), expected)
    sources = [p for p in (args.output / "inductor").rglob("*.cpp") if 'extern "C"' in p.read_text() and "out_ptr1" in p.read_text()]
    if len(sources) != 1:
        raise ValueError("expected one generated RMSNorm C++ entry")
    generated = sources[0].read_text()
    if not re.search(r"void\s+kernel\(const float\* in_ptr0,\s*const float\* in_ptr1,\s*float\* out_ptr0,\s*float\* out_ptr1\)", generated):
        raise ValueError("Inductor signature changed; inspect before replay")
    if "#pragma omp parallel" in generated:
        raise ValueError("expected the one-thread native entry without a parallel team")
    torch_path = sources[0].with_suffix(".so")
    library = ctypes.CDLL(str(torch_path))
    libraries.append(library)
    entries["inductor"] = (ctypes.cast(library.kernel, pointer), 1, 1, 0, 1)
    hashes["inductor"] = dict(source=str(sources[0]), source_sha256=digest(sources[0]),
                               library_sha256=digest(torch_path), torch_version=torch.__version__,
                               torch_git_version=torch.version.git_version, config=torch.__config__.show())
    output = np.full(rows * columns + 34, -731.25, np.float32)
    partial = np.full(rows + 34, -731.25, np.float32)
    report = dict(dimensions=dimensions, metric="single_thread_native_entry_host_wall_us", runtime_excluded=True,
                  hardware_cycles=False, cpu_threads=1, packet_width=8,
                  warmup_ms=100, sample_target_ms=30, samples=7,
                  boundary="C++ loop, native call including compiler-emitted libc memcpy; Luisa also resets mutable launch block/thread indices; no Runtime/Python/allocations in timed region",
                  input_sha256=input_hashes, artifacts=hashes, commands=commands,
                  helper_source_sha256=digest(source), helper_sha256=digest(helper_path), results=[])
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    # All six orders balance three variants. This is a fixed diagnostic replay,
    # not timing-based schedule selection or a multithread scalability claim.
    import itertools
    for round_id, order in enumerate(itertools.permutations(entries)):
        for name in order:
            output[17:-17] = np.nan
            partial[17:-17] = np.nan
            samples = np.empty(7, np.float64)
            repetitions = ctypes.c_uint64()
            address, abi, block, workspace, local = entries[name]
            code = helper(address, abi, rows, columns, block, local, arrays[0].ctypes.data, arrays[1].ctypes.data,
                          partial[17:-17].ctypes.data, output[17:-17].ctypes.data, workspace,
                          samples.size, samples.ctypes.data, ctypes.byref(repetitions))
            if code or not np.isfinite(samples).all() or np.any(samples <= 0):
                raise ValueError("native replay failed")
            check = validate_output(output[17:-17].reshape(rows, columns), expected)
            for data in (output, partial):
                if not np.all(data[:17] == -731.25) or not np.all(data[-17:] == -731.25):
                    raise ValueError("native replay overwrote guard")
            row = dict(variant=name, round=round_id, order=order, samples_us=samples.tolist(),
                       median_us=statistics.median(samples), repetitions=repetitions.value,
                       correctness=check, guard_elements=68)
            report["results"].append(row)
            (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            print(name, round_id, row["median_us"], "us", flush=True)
    report["summary_us"] = {name: statistics.median(r["median_us"] for r in report["results"] if r["variant"] == name) for name in entries}
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

