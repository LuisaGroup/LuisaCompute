"""Offline audit of the frozen opt-in 2x2 MMA attention pilot.

Usage: python -B audit.py EXTRACTED_EXPERIMENT_ROOT
Requires NumPy. Reads archived bytes only; never imports run.py or executes,
links, disassembles, or loads native code. Prints JSON; writes no files.
All six cases, twelve captures and seventy-two ABBA visits are mandatory.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import tarfile

import numpy as np


ORIGINAL_ROOT = "luisa-attention-mma-2d.HIXgEL"
RUN_SHA256 = "05df3824530feaafce9d53f2f0eeffc055d8c6e29220f181d93934b09aeaea91"
ATOL = RTOL = 5e-5
CASES = [
    ("prefill-q4-u0", (1, 4, 2, 32, 65, 32, 32), (4, 16), 0),
    ("prefill-q4-u8", (1, 4, 2, 32, 65, 32, 32), (4, 16), 8),
    ("heldout-q8-u0", (1, 4, 2, 64, 129, 32, 48), (8, 16), 0),
    ("heldout-batch-gqa-q4-u8", (2, 6, 2, 17, 67, 40, 48), (4, 16), 8),
    ("heldout-gqa-q8-u8", (1, 6, 2, 33, 131, 48, 40), (8, 16), 8),
    ("decode-m1-u8-control", (1, 8, 2, 1, 2053, 80, 96), (1, 16), 8),
]
VARIANTS = (("off", False), ("on", True))
RESOURCE_FIELDS = (
    "static_snapshot_bytes_per_worker", "static_snapshot_allocations", "snapshot_budget",
    "private_workspace_bytes", "local_lanes", "max_unrolled_tile_elements",
    "max_unrolled_region_work", "unordered_reduction_partitions",
    "requested_max_unrolled_mma_terms", "requested_mma_output_block", "blocks_per_task",
)
OBSERVED_FIELDS = (
    "blocked_mmas", "two_dimensional_mmas", "rolled_mmas", "full_packet_specializations",
    "full_packet_cloned_instructions", "contiguous_private_reads", "contiguous_private_writes", "interleaved_private_arrays",
)
COMMON_FILES = (
    "input0.f32", "input1.f32", "input2.f32", "expected.f64", "native_tile.py",
    "native_tile_replay.cpp", "backends/simd/llvm/llvm_schedule_codegen.h",
)
SOURCE_FILES = {
    "native_tile.py": "scripts/benchmark/tile_torch/native_tile.py",
    "native_tile_replay.cpp": "scripts/benchmark/tile_torch/native_tile_replay.cpp",
    "backends/simd/llvm/llvm_schedule_codegen.h": "src/backends/simd/llvm/llvm_schedule_codegen.h",
}
BOUNDARY = ("Common C++ timer; native entries, launch resets, block traversal and compiler-emitted libc/allocations included. "
            "Runtime/Python/JIT/caller allocation/validation excluded.")
HASHES = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, context):
    require(type(actual) is type(expected) and actual == expected,
            f"mismatch {context}: {actual!r} vs {expected!r}")


def fingerprint(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA256")
    return value


def digest(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"missing/nonregular artifact: {path}")
    stat = path.stat()
    identity = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if identity not in HASHES:
        with path.open("rb") as file:
            HASHES[identity] = hashlib.file_digest(file, "sha256").hexdigest()
    return HASHES[identity]


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def load(path):
    value = json.loads(Path(path).read_text(), object_pairs_hook=unique_object)
    def finite(item):
        if type(item) is float:
            require(math.isfinite(item), "nonfinite JSON number: " + str(path))
        elif isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)
    finite(value)
    return value


def local(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "unsafe artifact path: " + str(relative))
    result = root / relative
    require(result.resolve().is_relative_to(root.resolve()), "escaped artifact path: " + str(result))
    return result


def original(root, prepared, text):
    path = Path(text)
    if ORIGINAL_ROOT in path.parts:
        return local(root, Path(*path.parts[path.parts.index(ORIGINAL_ROOT) + 1:]))
    snapshots = {"native_tile.py": "native_tile.py", "native_tile_replay.cpp": "native_tile_replay.cpp",
                 "llvm_schedule_codegen.h": "backends/simd/llvm/llvm_schedule_codegen.h"}
    require(path.name in snapshots, "unmapped original identity (live reads forbidden): " + text)
    return local(prepared, snapshots[path.name])


def controls():
    return dict(
        cases=[dict(case=name, dimensions=list(dims), block=list(block), cap=cap) for name, dims, block, cap in CASES],
        variants=[dict(name=name, enable_mma_2d_blocking=enabled) for name, enabled in VARIANTS],
        requested_mma_output_block=4, full_packet_specialization=True,
        packet_width=8, workers_per_block=32, local_lanes=1,
        max_unrolled_tile_elements=64, max_unrolled_region_work=4096,
        require_ab_root_order_equal=True, require_ab_blocks_per_task_equal=True,
        attention_qk="mma", attention_pv="mma", precision="fp32", fast_math=False,
        native_cycles=3, native_samples=7, native_warmup_ms=30, native_target_ms=15,
        capture_samples=3, capture_sample_ms=1, capture_warmup_ms=1,
        process_timeout_s=60, metric="single_thread_native_entry_host_wall_us",
        independent_oracle="NumPy dense FP64 GQA, bottom-right causal, FP32 scale",
        require_ab_bitwise_output=True, decode_control_requires_identical_source_and_object=True,
        admission="Actual two_dimensional_mmas is recorded, not forced to 2 for an enabled request.",
    )


def provenance(root):
    record = load(root / "provenance.json")
    equal(digest(root / "sources.tar.gz"), record["source_archive_sha256"], "source archive fingerprint")
    expected, members = record["source_sha256"], {}
    require(type(expected) is dict and expected, "missing source inventory")
    archived_patch = None
    with tarfile.open(root / "sources.tar.gz", "r:gz") as archive:
        for member in archive:
            name = member.name
            path = Path(name)
            require(not path.is_absolute() and ".." not in path.parts, "unsafe source member")
            if member.isdir():
                continue
            require(member.isfile() and name not in members and name in expected, "unlisted/nonregular/duplicate source member: " + name)
            file = archive.extractfile(member)
            require(file is not None, "unreadable source member: " + name)
            if name == "_provenance/owned-changes-from-main-head.patch":
                archived_patch = file.read()
                members[name] = hashlib.sha256(archived_patch).hexdigest()
            else:
                members[name] = hashlib.file_digest(file, "sha256").hexdigest()
            equal(members[name], fingerprint(expected[name]), "source member " + name)
    equal(set(members), set(expected), "complete source inventory")
    equal(expected["_experiment/run.py"], digest(root / "run.py"), "archived experiment source")
    equal(expected["_experiment/archive_sources.py"], digest(root / "archive_sources.py"), "archiver source")
    equal(record["build_configuration_sha256"], expected["_build/CMakeCache.txt"], "frozen build cache")
    for name, value in record["build_files_sha256"].items():
        equal(value, expected[name], "build configuration/archive " + name)
    for path, row in record["main_isolate_equality"].items():
        equal(row["equal"], True, "main/isolate equality " + path)
        equal(row["main_sha256"], row["isolated_sha256"], "main/isolate bytes " + path)
        equal(row["isolated_sha256"], expected[path], "isolate/archive source " + path)
    require(record["main_isolate_equality"], "missing main/isolate check")
    equal(record["source_snapshot_not_loader_attestation"], True, "provenance boundary")
    binaries = record["binary_sha256"]
    require(type(binaries) is dict and "benchmark_tile_xir" in binaries, "missing historical producer fingerprint")
    for name, value in binaries.items():
        require(Path(name).name == name, "non-basename producer fingerprint")
        fingerprint(value)
    # Preserve the original malformed auxiliary patch and its failure receipt.
    # Full source members remain authoritative; check the exact one-LF repair
    # from bytes without invoking git or replacing any frozen archive member.
    correction = load(root / "provenance-correction-receipt.json")
    equal(correction["status"], "corrected_by_supplement_only", "supplement-only provenance correction")
    for name, value in correction["frozen_files_unchanged"].items():
        equal(digest(local(root, name)), value, "unchanged frozen artifact")
    require(archived_patch is not None, "missing auxiliary source patch")
    equal(len(archived_patch), correction["archived_patch"]["bytes"], "original patch byte extent")
    equal(hashlib.sha256(archived_patch).hexdigest(), correction["archived_patch"]["sha256"], "original patch hash")
    equal(correction["archived_patch"]["returncode"], 128, "retained original patch parse failure")
    exact = local(root, correction["exact_patch"]["file"])
    equal(digest(exact), correction["exact_patch"]["sha256"], "supplemental patch hash")
    equal(exact.read_bytes(), archived_patch + b"\n", "exact terminal-LF-only patch repair")
    equal(exact.stat().st_size, correction["exact_patch"]["bytes"], "supplemental patch extent")
    equal(correction["exact_patch"]["returncode"], 0, "supplemental patch parse receipt")
    equal({line.split("\t", 2)[2] for line in correction["exact_patch"]["numstat"]}, set(record["main_isolate_equality"]), "supplemental patch owned-file inventory")
    equal(correction["safety"]["actual_patch_applied"], False, "no actual patch application")
    return record


def plan(root, frozen):
    equal(digest(root / "run.py"), RUN_SHA256, "predeclared runner SHA256")
    declarations = {}
    for node in ast.parse((root / "run.py").read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("CASES", "VARIANTS"):
                    require(target.id not in declarations, "duplicate runner declaration")
                    declarations[target.id] = ast.literal_eval(node.value)
    equal(declarations, {"CASES": CASES, "VARIANTS": VARIANTS}, "literal frozen case/variant declarations")
    record = load(root / "plan.json")
    equal(record["controls"], controls(), "complete frozen experimental controls")
    equal(record["llvm"], "/opt/homebrew/opt/llvm/bin", "declared LLVM tools path")
    equal(record["identity_boundary"], "Producer fingerprint and helper sources only; not a loader-closure attestation.", "plan attestation boundary")
    identities = record["frozen_sha256"]
    by_name = {Path(path).name: fingerprint(value) for path, value in identities.items()}
    equal(len(by_name), len(identities), "unambiguous frozen identities")
    expected = {"run.py": RUN_SHA256, "benchmark_tile_xir": frozen["binary_sha256"]["benchmark_tile_xir"],
                **{Path(relative).name: frozen["source_sha256"][source] for relative, source in SOURCE_FILES.items()}}
    equal(by_name, expected, "plan/provenance source and producer identities")
    return record


def array(path, shape, dtype):
    dtype = np.dtype(dtype)
    require(Path(path).stat().st_size == math.prod(shape) * dtype.itemsize, "tensor byte extent: " + str(path))
    value = np.fromfile(path, dtype=dtype).reshape(shape)
    require(np.isfinite(value).all(), "nonfinite tensor: " + str(path))
    return value


def oracle(dims, inputs):
    b, h, kh, q, keys, d, dv = dims
    require(all(type(v) is int and v > 0 for v in dims) and h % kh == 0 and q <= keys, "invalid GQA dimensions")
    query, key, value = [x.astype(np.float64) for x in inputs]
    key, value = np.repeat(key, h // kh, axis=1), np.repeat(value, h // kh, axis=1)
    scale = float(np.float32(1) / np.sqrt(np.float32(d)))
    scores = (query @ np.swapaxes(key, -1, -2)) * scale
    causal = np.arange(keys)[None, :] <= np.arange(q)[:, None] + keys - q
    scores = np.where(causal, scores, -1e30)
    weights = np.where(causal, np.exp(scores - scores.max(axis=-1, keepdims=True)), 0)
    result = (weights / weights.sum(axis=-1, keepdims=True)) @ value
    require(np.isfinite(result).all(), "independent oracle is nonfinite")
    return result


def output_check(path, expected):
    actual = array(path, expected.shape, "<f4").astype(np.float64)
    error = np.abs(actual - expected)
    require(np.all(error <= ATOL + RTOL * np.abs(expected)), "complete output mismatch: " + str(path))
    return float(error.max())


def functions(source):
    matches = re.findall(r"^(define[^\n]*@([-a-zA-Z$._0-9]+)\([^\n]*\)[^\n]*\{\n.*?^\})", source, re.M | re.S)
    require(len(matches) == len({name for _, name in matches}), "duplicate LLVM definition")
    return {name: body for body, name in matches}


def realization(metadata, source):
    text, result = metadata["realization"], {}
    for key in RESOURCE_FIELDS + OBSERVED_FIELDS:
        found = re.findall(r"\b" + key + r"=(\d+)\b", text)
        require(len(found) == 1, "missing/duplicate metadata field: " + key)
        result[key] = int(found[0])
    for key in ("requested_mma_2d_blocking", "fast_math", "ordered_reduction"):
        found = re.findall(r"\b" + key + r"=(true|false)\b", text)
        require(len(found) == 1, "missing/duplicate boolean metadata: " + key)
        result[key] = found[0] == "true"
    equal(re.findall(r"\bW(\d+), (\d+) workers/block\b", text), [("8", "32")], "actual packet/block ABI")
    orders = re.findall(r"\broot order \[([0-9]+(?:\s*,\s*[0-9]+)*)\]", text)
    require(len(orders) == 1, "missing/duplicate root order")
    result["root_order"] = [int(value.strip()) for value in orders[0].split(",")]
    equal(sorted(result["root_order"]), list(range(len(result["root_order"]))), "root permutation")
    blocks, cfg = re.findall(r"\bSchedule blocks=(\d+)", text), re.findall(r"\bdirect CFG=(true|false)", text)
    require(len(blocks) == len(cfg) == 1, "missing/duplicate codegen counters")
    result.update(schedule_blocks=int(blocks[0]), direct_cfg=cfg[0] == "true")
    defs = functions(source)
    clones = [name for name in defs if name.endswith(".full_packet")]
    count, instructions = result["full_packet_specializations"], result["full_packet_cloned_instructions"]
    require(count in (0, 1) and len(clones) == count, "actual LLVM clone definitions disagree with metadata")
    require((count == 0 and instructions == 0) or (count == 1 and 0 < instructions <= 4096), "invalid clone admission diagnostics")
    if count:
        equal(clones, ["llm_attention.full_packet"], "actual specialized function")
        require("%active_lane_count" not in defs[clones[0]], "full-packet body retains variable lane count")
        require(re.search(r"call void @llm_attention\.full_packet\(", source), "specialization has no call site")
    result["actual_full_packet_definitions"] = clones
    instruction = re.compile(r"^  (?:%[-a-zA-Z$._0-9]+ = |(?:tail |musttail |notail )?call\b|store\b|ret\b|br\b|switch\b|unreachable\b|fence\b|invoke\b|resume\b|indirectbr\b|catchret\b|cleanupret\b)")
    result["llvm_text_statistics"] = {
        name: dict(utf8_bytes=len(body.encode()), instruction_start_lines=sum(bool(instruction.match(line)) for line in body.splitlines()),
                   basic_block_labels=len(re.findall(r"^[-a-zA-Z$._0-9]+:", body, re.M)))
        for name, body in defs.items()
    }
    return result


def command(folder, stem, process=True):
    record = load(folder / (stem + ".command.json"))
    equal(record["returncode"], 0, "raw command exit: " + stem)
    require("error" not in record, "raw command error: " + stem)
    require(not (folder / (stem + ".stderr")).read_bytes(), "nonempty raw stderr: " + str(folder / stem))
    require((folder / (stem + ".stdout")).is_file(), "missing raw stdout: " + stem)
    if process:
        require(record["finished"] >= record["started"], "unfinished process: " + stem)
        equal(record["timeout_s"], 60, "bounded process timeout")
        equal(record["process_group_isolated"], True, "isolated process group")
    return record


def capture(root, frozen, name, dims, block, cap, variant, enabled):
    folder, prepared = root / f"{name}-{variant}", root / f"prepared-{name}-{variant}"
    manifest = load(prepared / "prepared.json")
    for key, value in dict(status="prepared", format="native-tile-entry-v1", name=variant, capture_kind="llm").items():
        equal(manifest.get(key), value, "prepared " + key)
    inventory = {str(path.relative_to(prepared)) for path in prepared.rglob("*") if path.is_file() and path.name != "prepared.json"}
    equal(set(manifest["files"]), inventory, "complete prepared inventory")
    for relative, expected in manifest["files"].items():
        equal(digest(local(prepared, relative)), fingerprint(expected), "prepared file " + relative)
    for path, expected in manifest["original_sha256"].items():
        equal(digest(original(root, prepared, path)), fingerprint(expected), "original capture/helper " + path)
    for relative, source in SOURCE_FILES.items():
        equal(manifest["files"][relative], frozen["source_sha256"][source], "prepared/frozen helper " + relative)
    metadata = load(folder / "capture.stdout")
    equal(manifest["metadata"], metadata, "captured/prepared metadata")
    b, h, kh, q, keys, d, dv = dims
    shapes = [[b, h, q, d], [b, kh, keys, d], [b, kh, keys, dv]]
    output_shape = [b, h, q, dv]
    expected_metadata = dict(implementation="tile_xir_simd", backend="cpu", operation="attention", dimensions=list(dims),
        attention_block=list(block), attention_qk="mma", attention_pv="mma", precision="fp32", fast_math=False,
        relaxed_precision=False, runtime="luisa", source_kind="tile_lowering_source", timing="synchronized_host_wall",
        batch_policy="one_runtime_command_list_per_batch", source_reduction_policy="unordered_tree",
        reduction_candidate_setting="not_applicable", requested_group_threads=0, requested_input_views=False,
        reduction_tree=False, input_shapes=shapes, output_shape=output_shape)
    for key, value in expected_metadata.items():
        equal(metadata.get(key), value, f"{name}/{variant}/metadata/{key}")
    for key, value in dict(checks=2, elements_per_check=math.prod(output_shape), guard_elements_per_check=34, atol=ATOL, rtol=RTOL).items():
        equal(metadata["correctness"][key], value, "capture validation receipt " + key)
    for key in ("throughput_us", "latency_us"):
        require(len(metadata[key]) == 3 and all(type(v) in (int, float) and v > 0 for v in metadata[key]), "invalid capture timing receipt")
    inputs = [array(prepared / f"input{i}.f32", shape, "<f4") for i, shape in enumerate(shapes)]
    independent = oracle(dims, inputs)
    expected = array(prepared / "expected.f64", output_shape, "<f8")
    require(np.allclose(expected, independent, atol=1e-12, rtol=1e-12), "independent dense FP64 oracle mismatch")
    error = output_check(prepared / "captured.f32", independent)
    output_check(folder / "output.f32", independent)
    for key, value in dict(packet_width=8, block=[32, 1, 1], input_files=["input0.f32", "input1.f32", "input2.f32"], output_elements=math.prod(output_shape), atol=ATOL, rtol=RTOL).items():
        equal(manifest[key], value, "prepared ABI " + key)
    equal(manifest["dispatch"], metadata["dispatch"], "prepared actual dispatch")
    source = (prepared / "kernel.ll").read_text()
    symbols = re.findall(r"^define dso_local void @(llm_attention\.packet_batch(?:\.blocks)?)\(", source, re.M)
    require(len(symbols) == 1, "ambiguous packet-batch entry")
    equal(manifest["symbol"], symbols[0], "actual native entry")
    equal(manifest["abi"], 0 if symbols[0].endswith(".blocks") else 2, "native ABI discriminator")
    realized = realization(metadata, source)
    equal(manifest["workspace_bytes"], realized["private_workspace_bytes"], "native workspace")
    for key, value in dict(local_lanes=1, max_unrolled_tile_elements=64, max_unrolled_region_work=4096,
                           requested_mma_output_block=4, requested_max_unrolled_mma_terms=cap,
                           requested_mma_2d_blocking=enabled, fast_math=False).items():
        equal(realized[key], value, "exact realized request/control " + key)
    if not enabled or block[0] == 1:
        equal(realized["two_dimensional_mmas"], 0, "disabled/decode negative control")
    require("mma_unroll_cost=unmodeled" in metadata["realization"] and "mma_blocking_cost=unmodeled" in metadata["realization"], "unmodeled native-cost boundary changed")
    receipt = command(folder, "capture")
    equal(Path(receipt["command"][0]).name, "benchmark_tile_xir", "producer filename")
    equal(receipt["command"][1:-1], ["llm", "attention", ",".join(map(str, dims)), *map(str, block), "3", "1", "1"], "exact capture CLI")
    equal(original(root, prepared, receipt["command"][-1]), folder / "output.f32", "capture output path")
    expected_env = dict(LUISA_TILE_BENCH_XIR_BACKEND="simd", LUISA_TILE_BENCH_ATTENTION_QK="mma", LUISA_TILE_BENCH_ATTENTION_PV="mma",
        LUISA_TILE_BENCH_XIR_MMA_OUTPUT_BLOCK="4", LUISA_TILE_BENCH_XIR_MMA_UNROLL_TERMS=str(cap),
        LUISA_TILE_BENCH_XIR_MMA_2D_BLOCKING="1" if enabled else "0", LUISA_TILE_BENCH_XIR_BLOCK_SIZE="32",
        LUISA_TILE_BENCH_XIR_LOCAL_LANES="1", LUISA_TILE_BENCH_XIR_REGION_WORK="4096", LUISA_SIMD_ENABLE_FULL_PACKET_SPECIALIZATION="1")
    environment = receipt["environment"]
    equal(set(environment), set(expected_env) | {"LUISA_TILE_BENCH_DUMP_SOURCE", "LUISA_SIMD_DUMP_ASSEMBLY_DIR"}, "complete recorded capture controls")
    for key, value in expected_env.items():
        equal(environment[key], value, "capture environment " + key)
    equal(original(root, prepared, environment["LUISA_TILE_BENCH_DUMP_SOURCE"]), folder / "output.f32.source.txt", "dump source path")
    equal(original(root, prepared, environment["LUISA_SIMD_DUMP_ASSEMBLY_DIR"]), folder / "objects", "dump object path")
    prep = command(folder, "prepare")
    equal(Path(prep["command"][1]).name, "native_tile.py", "prepare runner")
    equal(prep["environment"], {}, "prepare environment receipt")
    suffix = [str(original(root, prepared, value)) if ORIGINAL_ROOT in Path(value).parts else value for value in prep["command"][2:]]
    equal(suffix, ["prepare", "--capture-kind", "llm", "--llvm", "/opt/homebrew/opt/llvm/bin", "--prefix", str(folder / "output.f32"),
                   "--log", str(folder / "capture.stdout"), "--objects", str(folder / "objects"), "--output", str(prepared), "--name", variant], "exact preparation command")
    tools = {Path(path).name: path for path in manifest["tool_sha256"]}
    equal(set(tools), {"clang++", "llvm-nm"}, "recorded tool names")
    for value in manifest["tool_sha256"].values():
        fingerprint(value)
    frozen_tools = {Path(path).name: info for path, info in frozen["tool_fingerprints"].items()}
    for path, value in manifest["tool_sha256"].items():
        equal(value, frozen_tools[Path(path).name]["sha256"], "prepared/provenance tool fingerprint " + path)
    tool_commands = {
        "compiler": [tools["clang++"], "--version"],
        "link": [tools["clang++"], "-dynamiclib", str(prepared / "kernel.o"), "-o", str(prepared / "kernel.dylib")],
        "helper": [tools["clang++"], "-std=c++20", "-O3", "-dynamiclib", "-I" + str(prepared), str(prepared / "native_tile_replay.cpp"), "-o", str(prepared / "replay.dylib")],
        "imports": [tools["llvm-nm"], "--undefined-only", "--just-symbol-name", str(prepared / "kernel.o")],
        "exports": [tools["llvm-nm"], "--defined-only", "--extern-only", "--just-symbol-name", str(prepared / "kernel.o")],
    }
    for stem, expected_argv in tool_commands.items():
        actual = []
        for value in command(prepared, stem, process=False)["argv"]:
            prefix = "-I" if value.startswith("-I") else ""
            text = value[len(prefix):]
            actual.append(prefix + str(original(root, prepared, text)) if ORIGINAL_ROOT in Path(text).parts else value)
        equal(actual, expected_argv, "exact archived tool command " + stem)
    equal((prepared / "imports.stdout").read_text().split(), manifest["imports"], "raw object imports")
    require(set(manifest["imports"]) <= {"_memcpy", "_memset", "_bzero", "___chkstk_darwin"}, "unadmitted imports")
    require("_" + manifest["symbol"] in (prepared / "exports.stdout").read_text().split(), "object export receipt misses native entry")
    copies = {"kernel.ll": folder / "output.f32.source.txt", "captured.f32": folder / "output.f32", "expected.f64": folder / "output.f32.expected.f64",
              **{f"input{i}.f32": folder / f"output.f32.input{i}.f32" for i in range(3)}}
    for relative, path in copies.items():
        equal(digest(path), manifest["files"][relative], "raw/prepared copy " + relative)
    objects = list((folder / "objects").glob("*.o"))
    require(len(objects) == 1, "ambiguous captured ORC object")
    equal(digest(objects[0]), manifest["files"]["kernel.o"], "actual captured object identity")
    return dict(manifest=manifest, prepared=prepared, folder=folder, receipt=receipt, expected=independent,
                realized=realized, error=error, source=source)


def pair(left, right, decode):
    a, b = left["manifest"], right["manifest"]
    for key in ("symbol", "abi", "packet_width", "block", "dispatch", "workspace_bytes", "input_files", "output_elements", "atol", "rtol", "tool_sha256"):
        equal(a[key], b[key], "fixed pair native ABI/tool " + key)
    for key in RESOURCE_FIELDS + ("root_order", "fast_math", "ordered_reduction"):
        equal(left["realized"][key], right["realized"][key], "fixed pair execution/resource " + key)
    for relative in COMMON_FILES + ("captured.f32",):
        equal(a["files"][relative], b["files"][relative], "bitwise pair input/helper/output " + relative)
    if decode:
        for relative in ("kernel.ll", "kernel.o"):
            equal(a["files"][relative], b["files"][relative], "decode code-identity negative control " + relative)


def replay(root, name, dims, block, cap, left, right):
    directory = root / f"replay-{name}"
    result = load(directory / "results.json")
    entries = {"off": left, "on": right}
    for key, value in dict(status="passed", artifacts_unchanged=True, cpu_threads=1, metric="single_thread_native_entry_host_wall_us",
                           order_policy="ABBA per cycle; two matched pairs per cycle", boundary=BOUNDARY).items():
        equal(result[key], value, "replay contract " + key)
    for key, value in dict(command="replay", cycles=3, samples=7, warmup_ms=30, target_ms=15, validate_only=False).items():
        equal(result["options"][key], value, "replay options " + key)
    equal(result["runner_sha256"], left["manifest"]["files"]["native_tile.py"], "replay runner fingerprint")
    expected_manifests = [entry["prepared"] / "prepared.json" for entry in (left, right)]
    equal([original(root, root, value) for value in result["options"]["prepared"]], expected_manifests, "ABBA entry order")
    equal(original(root, root, result["options"]["output"]), directory, "replay output path")
    equal({original(root, root, key): value for key, value in result["prepared_sha256"].items()}, {path: digest(path) for path in expected_manifests}, "replay manifest identities")
    receipt = command(root, "replay-" + name)
    equal(receipt["environment"], {}, "replay environment receipt")
    equal(Path(receipt["command"][1]).name, "native_tile.py", "replay runner name")
    expected_argv = ["replay", "--prepared", result["options"]["prepared"][0], "--prepared", result["options"]["prepared"][1],
                     "--output", result["options"]["output"], "--cycles", "3", "--samples", "7", "--warmup-ms", "30", "--target-ms", "15"]
    normalize = lambda values: [str(original(root, root, value)) if ORIGINAL_ROOT in Path(value).parts else value for value in values]
    equal(normalize(receipt["command"][2:]), normalize(expected_argv), "exact ABBA command")
    visits = result["visits"]
    equal(len(visits), 12, "complete ABBA visits")
    ratios, maximum = [], 0.0
    for cycle in range(3):
        rows = visits[cycle * 4:cycle * 4 + 4]
        equal([row["variant"] for row in rows], ["off", "on", "on", "off"], "exact ABBA order")
        for position, row in enumerate(rows):
            for key, value in dict(cycle=cycle, position=position, valid=True, returncode=0, all_guards_passed=True, inputs_unchanged=True).items():
                equal(row[key], value, "ABBA visit " + key)
            samples = row["samples_us"]
            require(len(samples) == 7 and all(type(value) in (int, float) and value > 0 for value in samples), "invalid finite positive timing samples")
            require(type(row["repetitions"]) is int and row["repetitions"] > 0, "invalid native repetitions")
            equal(row["median_us"], statistics.median(samples), "recomputed visit median")
            equal(row["output"], f"visit-{cycle}-{position}.f32", "visit output filename")
            path = local(directory, row["output"])
            equal(digest(path), row["output_sha256"], "visit output hash")
            equal(digest(path), entries[row["variant"]]["manifest"]["files"]["captured.f32"], "bitwise capture/replay output")
            maximum = max(maximum, output_check(path, entries[row["variant"]]["expected"]))
            for key, value in dict(elements=left["manifest"]["output_elements"], atol=ATOL, rtol=RTOL).items():
                equal(row["correctness"][key], value, "visit correctness receipt " + key)
        ratios.extend((rows[1]["median_us"] / rows[0]["median_us"], rows[2]["median_us"] / rows[3]["median_us"]))
    paired = dict(baseline="off", candidate="on", pairs=ratios, median=statistics.median(ratios), minimum=min(ratios), maximum=max(ratios))
    equal(result["candidate_over_baseline"], paired, "all recomputed paired statistics")
    medians = {name: statistics.median(row["median_us"] for row in visits if row["variant"] == name) for name in entries}
    equal(result["summary_us"], medians, "recomputed cohort medians")
    return dict(case=name, dimensions=list(dims), attention_block=list(block), cap=cap, medians_us=medians,
                paired_on_over_off=paired, paired_wins=sum(value < 1 for value in ratios),
                outputs_bitwise_equal=True, replay_max_abs_error_vs_independent_fp64=maximum,
                decode_source_and_object_identical=block[0] == 1,
                observed_off=left["realized"], observed_on=right["realized"])


def validation(root):
    directory = root / "validation"
    clean = lambda path: re.sub(r"\x1b\[[0-9;]*m", "", path.read_text())
    summary = clean(directory / "tests-retry1.log")
    expected = {"test_simd_llvm_schedule_codegen", "test_tile_xir_target_info", "test_tile_xir_runtime", "test_tile_xir_llm"}
    passed = set(re.findall(r"Test #\d+: (\w+)\s+\.+\s+Passed", summary))
    equal(passed, expected, "all four full CTest pass receipts")
    require("100% tests passed out of 4" in summary and "FAILED" not in summary, "full CTest cohort summary")
    host_files = sorted(directory.glob("host-*.log"))
    equal(len(host_files), 7, "seven bounded host validation receipts")
    assertions = {}
    for path in host_files:
        text = clean(path)
        counts = re.findall(r"Suite 'global': all tests passed\s+\((\d+) asserts in (\d+) tests\)", text)
        require(len(counts) == 1 and all(int(value) > 0 for value in counts[0]), "nonempty host test pass: " + path.name)
        assertions[path.name] = dict(assertions=int(counts[0][0]), tests=int(counts[0][1]))
    initial = clean(directory / "build.log")
    require("FAILED:" in initial and "use of undeclared identifier 'format'" in initial, "retained initial build failure")
    retry = clean(directory / "build-retry1.log")
    require("FAILED:" not in retry and "ninja: build stopped" not in retry, "retry build log still reports failure")
    machine = (directory / "machine.log").read_text()
    require(machine.strip(), "missing machine receipt")
    return dict(ctest_passes=sorted(passed), host_receipts=assertions,
                initial_build_failure_retained=True, machine_receipt=machine,
                interpretation="Checks saved validation receipts and hashes; does not rerun the build or tests.")


def audit(root):
    root = root.resolve()
    frozen = provenance(root)
    declaration = plan(root, frozen)
    plan_sha = digest(root / "plan.json")
    captures_record, replays_record = load(root / "captures.json"), load(root / "replays.json")
    for record in (captures_record, replays_record):
        equal(record["status"], "complete", "complete cohort (no partial acceptance)")
        equal(record["plan_sha256"], plan_sha, "frozen cohort plan")
        require(record["finished"] >= record["started"], "unfinished cohort")
    equal(replays_record["runner_sha256"], RUN_SHA256, "cohort runner identity")
    require(declaration["frozen_unix"] <= captures_record["started"] <= frozen["frozen_unix"] <= captures_record["finished"] <= replays_record["started"],
            "plan/source/capture/competitive replay chronology differs from correction receipt")
    equal(replays_record["declared_cases"], controls()["cases"], "all predeclared replay cases")
    equal([row["case"] for row in captures_record["cases"]], [row[0] for row in CASES], "all twelve captures' six cases")
    equal([row["case"] for row in replays_record["experiments"]], [row[0] for row in CASES], "all six replay rows, including regressions")
    captures, rows, expected_commands, evidence = {}, [], set(), set()
    for case_index, (name, dims, block, cap) in enumerate(CASES):
        capture_row = captures_record["cases"][case_index]
        equal(capture_row["status"], "OK", "accepted capture pair")
        equal([row["label"] for row in capture_row["variants"]], [f"{name}-off", f"{name}-on"], "capture variant order")
        for variant_index, (variant, enabled) in enumerate(VARIANTS):
            entry = capture(root, frozen, name, dims, block, cap, variant, enabled)
            captures[name, variant] = entry
            variant_row = capture_row["variants"][variant_index]
            equal(variant_row["prepared_manifest_sha256"], digest(entry["prepared"] / "prepared.json"), "capture matrix manifest hash")
            equal(variant_row["two_dimensional_mmas"], entry["realized"]["two_dimensional_mmas"], "capture matrix actual 2D count")
            actual_controls = {key: entry["realized"][key] for key in ("local_lanes", "max_unrolled_tile_elements", "max_unrolled_region_work", "root_order", "blocks_per_task")}
            actual_controls.update(packet_width=8, workers_per_block=32)
            equal(variant_row["execution_controls"], actual_controls, "capture matrix execution controls")
            producer_paths = [path for path in declaration["frozen_sha256"] if Path(path).name == "benchmark_tile_xir"]
            equal(entry["receipt"]["command"][0], producer_paths[0], "frozen producer command association")
            for folder, stems in ((entry["folder"], ("capture", "prepare")), (entry["prepared"], ("compiler", "helper", "link", "imports", "exports"))):
                expected_commands.update(folder / (stem + ".command.json") for stem in stems)
                evidence.update(path for path in folder.rglob("*") if path.is_file())
        left, right = captures[name, "off"], captures[name, "on"]
        pair(left, right, block[0] == 1)
        row = replay(root, name, dims, block, cap, left, right)
        rows.append(row)
        cohort_row = replays_record["experiments"][case_index]
        for key, value in dict(case=name, dimensions=list(dims), block=list(block), cap=cap, packet=True).items():
            equal(cohort_row[key], value, "cohort row " + key)
        for summary in (capture_row["correctness"], cohort_row["correctness"]):
            for key, value in dict(independent_fp64=True, elements=left["manifest"]["output_elements"], atol=ATOL, rtol=RTOL,
                                   ab_bitwise_equal=True, captured_output_sha256=left["manifest"]["files"]["captured.f32"], decode_code_identical=block[0] == 1).items():
                equal(summary[key], value, "cohort correctness receipt " + key)
            equal(summary["execution_controls"], actual_controls, "cohort correctness execution controls")
        expected_commands.add(root / f"replay-{name}.command.json")
        evidence.update(path for path in (root / f"replay-{name}").rglob("*") if path.is_file())
        evidence.update(root / f"replay-{name}.{suffix}" for suffix in ("command.json", "stdout", "stderr"))
    equal(set(root.rglob("*.command.json")), expected_commands, "complete raw command inventory")
    equal(len(expected_commands), 90, "12 captures + 12 prepares + 60 tool commands + 6 replays")
    equal(len(captures), 12, "capture count")
    equal(len(rows), 6, "paired experiment count")
    validation_summary = validation(root)
    evidence.update(path for path in (root / "validation").rglob("*") if path.is_file())
    evidence.update(root / name for name in ("run.py", "audit.py", "archive_sources.py", "plan.json", "captures.json", "replays.json", "provenance.json", "sources.tar.gz",
                                           "owned-changes-exact.patch", "provenance-corrections.md", "provenance-correction-receipt.json"))
    raw_hashes = {str(path.relative_to(root)): digest(path) for path in sorted(evidence)}
    return dict(status="passed", independent_numpy_oracles=12, capture_output_checks=24, replay_output_checks=72,
                timer_samples=504, raw_commands_checked=90, all_declared_and_negative_rows_retained=True,
                run_sha256=RUN_SHA256, audit_sha256=digest(Path(__file__)), plan_sha256=plan_sha,
                validation=validation_summary,
                chronology=dict(plan_frozen=declaration["frozen_unix"], captures_started=captures_record["started"],
                                source_frozen=frozen["frozen_unix"], captures_finished=captures_record["finished"],
                                replays_started=replays_record["started"], replays_finished=replays_record["finished"],
                                source_snapshot_during_capture=True),
                provenance=dict(source_archive_sha256=frozen["source_archive_sha256"], source_members_verified=len(frozen["source_sha256"]),
                    prepared_helpers_match_frozen_source=True, main_isolate_files_checked=len(frozen["main_isolate_equality"]),
                    recorded_binary_sha256=frozen["binary_sha256"], recorded_tool_sha256=next(iter(captures.values()))["manifest"]["tool_sha256"],
                    producer_and_tool_bytes_independently_verified=False, live_files_read=False, loader_closure_attested=False),
                captures={f"{name}-{variant}": dict(observed=data["realized"], kernel_object_sha256=data["manifest"]["files"]["kernel.o"],
                    prepared_manifest_sha256=digest(data["prepared"] / "prepared.json"), capture_max_abs_error_vs_independent_fp64=data["error"])
                    for (name, variant), data in captures.items()}, experiments=rows, raw_evidence_sha256=raw_hashes,
                limitations=[
                    "Single-thread actual native-entry host-wall includes launch resets/traversal/compiler helpers, not pure hardware time or Runtime E2E; no Torch/MPS comparison.",
                    "Only six predeclared cases: paired medians and full min/max ranges are descriptive, not confidence intervals or a default-policy recommendation.",
                    "Requested full-packet enable is fixed; actual LLVM clone definition/call sites and diagnostics are checked separately. Changed clone admission is part of the combined 2D realization effect.",
                    "Schedule/MMA/codegen counters are producer diagnostics. Textual LLVM statistics are not machine instruction counts or independent semantic attribution of every loop.",
                    "All capture/replay payload elements are checked against independently recomputed dense FP64 causal GQA; guard/input immutability additionally rely on frozen C++ receipts, since guard allocations are not retained.",
                    "Prepared source/object/dylib/helper bytes and exact historical link receipts are checked, but the audit does not execute, relink, disassemble or load them.",
                    "Producer/tool binary fingerprints and main/isolate equality are internally checked provenance records, not live binary-byte or loader-closure attestations; complete inherited environment is not retained.",
                ])


if __name__ == "__main__":
    require(len(sys.argv) == 2, "usage: audit.py EXTRACTED_EXPERIMENT_ROOT")
    print(json.dumps(audit(Path(sys.argv[1])), indent=2, allow_nan=False))
