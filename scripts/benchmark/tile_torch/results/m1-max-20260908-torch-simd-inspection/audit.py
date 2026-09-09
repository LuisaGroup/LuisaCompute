#!/usr/bin/env python3
"""Verify local CPU code-inspection evidence; do not derive timing rankings."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import struct


ROOT = Path(__file__).resolve().parent
COMMIT = "08187d9e0fba026dc8217405802ab5381dc88d90"
LIBRARY_HASH = "ca979f3619acb92240c2ef46d7581b1a388761e0fccfa7dc491a7eaa153d584a"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(relative):
    path = ROOT / relative
    if path.suffix == ".gz":
        return gzip.decompress(path.read_bytes()).decode()
    return path.read_text()


def object_text_bytes(relative):
    """Read __text directly from the captured Mach-O, independently of objdump."""
    data = gzip.decompress((ROOT / relative).read_bytes())
    magic, cpu, _, _, commands, _, _, _ = struct.unpack_from("<8I", data)
    require(magic == 0xFEEDFACF and cpu == 0x0100000C, "not an ARM64 Mach-O")
    offset = 32
    for _ in range(commands):
        command, size = struct.unpack_from("<2I", data, offset)
        require(size >= 8 and offset + size <= len(data), "bad Mach-O command")
        if command == 0x19:  # LC_SEGMENT_64
            sections = struct.unpack_from("<I", data, offset + 64)[0]
            require(72 + sections * 80 <= size, "bad Mach-O section count")
            for index in range(sections):
                section = offset + 72 + index * 80
                if data[section:section + 16].rstrip(b"\0") == b"__text":
                    length = struct.unpack_from("<Q", data, section + 40)[0]
                    start = struct.unpack_from("<I", data, section + 48)[0]
                    require(start + length <= len(data), "truncated machine code")
                    return length
        offset += size
    raise ValueError("no __text in captured object")


def audit():
    ops = {"softmax", "layernorm", "rmsnorm", "swiglu", "gemm"}
    for shape in ("1024x4096", "64x256"):
        row = json.loads(read(f"aten/inspection-{shape}.json"))
        require(row["commit"] == COMMIT and row["library_sha256"] == LIBRARY_HASH, "Torch binary identity")
        require(row["version"] == "2.14.0" and row["threads"] == 8, "Torch configuration")
        require({x["name"] for x in row["cases"]} == ops and len(row["cases"]) == 5, "eager coverage")
        for item in row["cases"]:
            require(item["sample_exit_code"] == 0 and item["output_finite"] is True, "incomplete eager inspection")
            expected = [1024, 1024] if item["name"] == "gemm" else list(map(int, shape.split("x")))
            require(item["shape"] == expected, "unexpected shape")
        require(len(row["compiled"]) == 3, "compiled coverage")
        require({x["name"] for x in row["compiled"]} == {"softmax", "rmsnorm", "swiglu"}, "compiled names")
        for item in row["compiled"]:
            require(item.get("correctness") == "complete eager comparison passed" and "error" not in item, "compiled correctness")
            source = read(f"inductor/{shape}/{item['name']}.cpp")
            require("#pragma omp parallel num_threads(8)" in source, "OpenMP realization")
            require("static_cast<int64_t>(4LL)" in source and "Vectorized<float>::loadu" in source, "four-lane source")
    config = json.loads(read("aten/inspection-64x256.json"))
    require(config["cpu_capability"] == "DEFAULT" and "BLAS_INFO=accelerate" in config["build_config"], "provider config")
    for op in ops:
        trace = json.loads(read(f"aten/{op}.trace.json"))
        require(any(x.get("name", "").startswith("aten::") for x in trace["traceEvents"]), "missing ATen trace")
    require("Sleef_expf4_u10" in read("aten/softmax.sample.txt"), "sampled vector exp")
    require("LayerNormKernelImplInternal<float" in read("aten/layernorm.sample.txt"), "sampled FP32 norm")
    require("at::native::cpublas::gemm" in read("aten/gemm.sample.txt") and "SGEMM  (in libBLAS.dylib)" in read("aten/gemm.sample.txt"), "sampled BLAS provider")
    rms_ops = next(x["operators"] for x in config["cases"] if x["name"] == "rmsnorm")
    require({"aten::pow", "aten::sum", "aten::rsqrt", "aten::mul"} <= {x["name"] for x in rms_ops}, "composite eager RMSNorm")
    require("fmax.4s" in read("aten/softmax-fp32.asm") and "_Sleef_expf4_u10" in read("aten/softmax-fp32.asm"), "ATen NEON code")
    require("fmul.4s" in read("aten/layernorm-fp32.asm"), "ATen norm vector code")

    assembly = read("xir/rmsnorm-64x256.s.gz")
    loop = assembly.split("LBB0_2308:", 1)[1].split("b.ne\tLBB0_2308", 1)[0]
    comparisons = list(map(int, re.findall(r"cmp\s+x10, #(\d+)", loop)))
    require(comparisons == list(range(257)), "dynamic select chain and loop bound")
    object_assembly = read("xir/rmsnorm-64x256.object.asm.gz")
    require("cmp\tx10, #0x100" in object_assembly and "b.ne\t0xcbec" in object_assembly, "actual object loop")
    require("sp, sp, #0x8, lsl #12" in object_assembly and "sp, sp, #0x100" in object_assembly and "[sp, #-0x50]!" in object_assembly, "actual stack frame")
    for op in ("rmsnorm", "swiglu"):
        row = json.loads(read(f"xir/{op}-64x256.json"))
        require(row["operation"] == op and row["dimensions"] == [64, 256], "XIR case identity")
        require(row["correctness"]["checks"] == 2 and row["correctness"]["elements_per_check"] == 16384 and row["correctness"]["guard_elements_per_check"] == 34, "XIR complete checks")
        require(row["fast_math"] is False and "contiguous reads=0" in row["realization"] and "root order [0]" in row["realization"], "XIR mapping")
    swiglu_ir = read("xir/swiglu-64x256.ll.gz")
    exp_calls = len(re.findall(r"call <8 x float> @__luisa_cpu_native_exp_f32_v8_u10", swiglu_ir))
    require(exp_calls == 256, "existing varying vector math and static expansion")
    require("expf" not in read("xir/swiglu-64x256.s.gz"), "unexpected scalar exp call")
    small_worker = read("inductor/64x256/rmsnorm-worker.asm")
    require("ldr\tq1, [x8], #0x10" in small_worker and "fmul.4s" in small_worker and "faddp.2s" in small_worker, "Inductor inner vector loop")
    require("fsqrt\ts0, s1" in small_worker and "b.lo\t0x9cc" in small_worker, "Inductor output loop")
    failure = json.loads(read("xir/large-inspection-status.json"))
    require(failure["shell_exit_code"] == 143 and failure["numerical_output_checked"] is False, "retain failed large diagnostic")
    require("emit_assembly_copy" in read("xir/rmsnorm-1024x4096-compile.sample.txt") and "MachineSinking" in read("xir/rmsnorm-1024x4096-compile.sample.txt"), "diagnostic-only stack")

    return dict(verdict="code_inspection_verified_no_performance_ranking", eager_cases=10,
                compiled_complete_comparisons=6, xir_complete_output_checks=4,
                rmsnorm_static_selection_choices=256, rmsnorm_dynamic_iterations=256,
                rmsnorm_stack_frame_bytes=32768 + 256 + 80,
                rmsnorm_object_text_bytes=object_text_bytes("xir/rmsnorm-64x256.o.gz"),
                swiglu_object_text_bytes=object_text_bytes("xir/swiglu-64x256.o.gz"),
                swiglu_static_vector_exp_calls=exp_calls, retained_incomplete_large_diagnostics=1,
                limitation="Source, samples and disassembly establish realization, not timing causality or a throughput ratio.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    result["artifact_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(ROOT.rglob("*")) if p.is_file()
                                 and p.name != "audit.json" and "__pycache__" not in p.parts}
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
        print(result["verdict"])
    else:
        print(text)


if __name__ == "__main__":
    main()
