"""Read-only MMA-only term-cap attention audit; never executes native artifacts.

Usage: python -B audit.py EXPERIMENT_DIRECTORY
Requires NumPy. All 12 captures and six ABBA experiments are mandatory.
Only JSON is printed; the caller may save it after successful completion.
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


ORIGINAL_ROOT_NAME = "luisa-attention-mma-roll.LHgRyP"
ATOL = RTOL = 5e-5
HASHES = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


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
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object,
                      parse_constant=lambda value: require(False, f"nonfinite JSON constant: {value}"))


def equal(actual, expected, context):
    require(type(actual) is type(expected) and actual == expected, f"mismatch {context}: {actual!r} vs {expected!r}")


def local(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, f"unsafe artifact path: {relative}")
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()), f"escaped artifact path: {path}")
    return path


def original_path(root, prepared, text):
    path = Path(text)
    marker = next((m for m in (ORIGINAL_ROOT_NAME,) if m in path.parts), None)
    if marker:
        offset = path.parts.index(marker) + 1
        return local(root, Path(*path.parts[offset:]))
    snapshots = {
        "native_tile.py": "native_tile.py", "native_tile_replay.cpp": "native_tile_replay.cpp",
        "llvm_schedule_codegen.h": "backends/simd/llvm/llvm_schedule_codegen.h",
    }
    require(path.name in snapshots, f"unmapped original source identity: {text}")
    return prepared / snapshots[path.name]


def array(path, shape, dtype):
    dtype = np.dtype(dtype)
    require(Path(path).stat().st_size == math.prod(shape) * dtype.itemsize, f"tensor byte extent: {path}")
    value = np.fromfile(path, dtype=dtype).reshape(shape)
    require(np.isfinite(value).all(), f"nonfinite tensor: {path}")
    return value


def oracle(dims, inputs):
    # Independent dense FP64 expression, not the captured blockwise online
    # softmax or the replay runner's reuse of the captured expected buffer.
    b, h, kh, q, keys, d, dv = dims
    x, k, v = [value.astype(np.float64) for value in inputs]
    k, v = np.repeat(k, h // kh, axis=1), np.repeat(v, h // kh, axis=1)
    scale = float(np.float32(1) / np.sqrt(np.float32(d)))
    scores = (x @ np.swapaxes(k, -1, -2)) * scale
    causal = np.arange(keys)[None, :] <= np.arange(q)[:, None] + keys - q
    scores = np.where(causal, scores, -1e30)
    weight = np.where(causal, np.exp(scores - scores.max(axis=-1, keepdims=True)), 0)
    return (weight / weight.sum(axis=-1, keepdims=True)) @ v


def check_output(path, expected):
    value = array(path, expected.shape, "<f4").astype(np.float64)
    error = np.abs(value - expected)
    require(np.all(error <= ATOL + RTOL * np.abs(expected)), f"complete output mismatch: {path}")
    return float(error.max())


def declared_cases(root):
    definitions = []
    for node in ast.parse((root / "run.py").read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CASES" for t in node.targets):
            definitions.append(ast.literal_eval(node.value))
    # Admission is deliberately frozen to the predeclared pilot, not winners
    # discovered by inspecting results.
    expected = [
        ("decode-r4-off", (1, 8, 2, 1, 2053, 80, 96), (1, 16), 4, False),
        ("decode-r4-on", (1, 8, 2, 1, 2053, 80, 96), (1, 16), 4, True),
        ("mha-r1-off", (1, 8, 8, 1, 2048, 64, 64), (1, 16), 1, False),
        ("mha-r1-on", (1, 8, 8, 1, 2048, 64, 64), (1, 16), 1, True),
        ("prefill-r1-on", (1, 4, 2, 32, 65, 32, 32), (4, 16), 1, True),
        ("decode-tail-r4-on", (1, 6, 2, 1, 2053, 80, 96), (1, 16), 4, True),
    ]
    equal(definitions, [expected], "complete declared six-case pilot")
    return definitions[0]


def functions(source):
    matches = re.findall(r"^(define[^\n]*@([-a-zA-Z$._0-9]+)\([^\n]*\)[^\n]*\{\n.*?^\})", source, re.M | re.S)
    require(len(matches) == len({name for _, name in matches}), "duplicate LLVM function name")
    return {name: body for body, name in matches}


def normalized_realization(value):
    return re.sub(r"full_packet_(?:specializations|cloned_instructions)=\d+", "FULL_PACKET_COUNTER", value)


def specialization(metadata, source):
    result = {}
    for key in ("full_packet_specializations", "full_packet_cloned_instructions"):
        matches = re.findall(r"\b" + key + r"=(\d+)\b", metadata["realization"])
        require(len(matches) == 1, f"missing/duplicate {key}")
        result[key] = int(matches[0])
    count, instructions = result.values()
    require(count in (0, 1) and ((count == 0 and instructions == 0) or (count == 1 and 0 < instructions <= 4096)), "invalid observed clone count/budget")
    defs = functions(source)
    clone = "llm_attention.full_packet"
    require((clone in defs) == (count == 1), "clone metadata does not match actual LLVM definitions")
    if count:
        require("%active_lane_count" not in defs[clone], "specialized body retained dynamic lane count")
        require(re.search(r"call void @llm_attention\.full_packet\(", source), "clone has no call site")
    return result


def audit_capture(root, name, dims, block, width, packet, cap):
    variant = f"u{cap}"
    folder = root / f"{name}-{variant}"
    prepared = root / f"prepared-{name}-{variant}"
    manifest = load(prepared / "prepared.json")
    for key, value in dict(status="prepared", format="native-tile-entry-v1", name=f"u{cap}", capture_kind="llm").items():
        equal(manifest.get(key), value, f"{name}/{variant}/{key}")
    declared = manifest["files"]
    require(set(declared) == {str(path.relative_to(prepared)) for path in prepared.rglob("*") if path.is_file() and path.name != "prepared.json"}, "prepared inventory has missing/extra files")
    for relative, expected in declared.items():
        equal(digest(local(prepared, relative)), expected, str(prepared / relative))
    for text, expected in manifest["original_sha256"].items():
        equal(digest(original_path(root, prepared, text)), expected, text)
    for text, expected in manifest["tool_sha256"].items():
        # Tool executables are not archived. Check their recorded identities
        # and command associations, never substitute a later live installation
        # for historical evidence. Pairwise tool identities are checked below.
        require(type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected),
                "malformed recorded tool fingerprint " + text)
    metadata = load(folder / "capture.stdout")
    equal(manifest["metadata"], metadata, "captured metadata copied to preparation")
    b, h, kh, q, keys, d, dv = dims
    input_shapes = [[b, h, q, d], [b, kh, keys, d], [b, kh, keys, dv]]
    output_shape = [b, h, q, dv]
    expected_metadata = dict(implementation="tile_xir_simd", backend="cpu", operation="attention", dimensions=list(dims),
                             attention_block=list(block), attention_qk="mma", attention_pv="mma", precision="fp32",
                             fast_math=False, relaxed_precision=False, runtime="luisa", source_kind="tile_lowering_source",
                             timing="synchronized_host_wall", batch_policy="one_runtime_command_list_per_batch",
                             source_reduction_policy="unordered_tree", reduction_candidate_setting="not_applicable",
                             requested_group_threads=0, requested_input_views=False, reduction_tree=False,
                             input_shapes=input_shapes, output_shape=output_shape)
    for key, value in expected_metadata.items():
        equal(metadata.get(key), value, f"{name}/{variant}/metadata/{key}")
    for key, value in dict(checks=2, elements_per_check=math.prod(output_shape), guard_elements_per_check=34,
                           atol=ATOL, rtol=RTOL).items():
        equal(metadata["correctness"].get(key), value, "capture oracle receipt " + key)
    for key in ("throughput_us", "latency_us"):
        require(len(metadata[key]) == 3 and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in metadata[key]), "bad capture timing receipt")
    inputs = [array(prepared / f"input{i}.f32", shape, "<f4") for i, shape in enumerate(input_shapes)]
    independent = oracle(dims, inputs)
    expected = array(prepared / "expected.f64", output_shape, "<f8")
    require(np.allclose(expected, independent, atol=1e-12, rtol=1e-12), f"independent FP64 oracle mismatch {name}/{variant}")
    error = check_output(prepared / "captured.f32", independent)
    check_output(folder / "output.f32", independent)
    for key, value in dict(packet_width=8, block=[32, 1, 1], input_files=["input0.f32", "input1.f32", "input2.f32"],
                           output_elements=math.prod(output_shape), atol=ATOL, rtol=RTOL).items():
        equal(manifest.get(key), value, "prepared contract " + key)
    equal(manifest["dispatch"], metadata["dispatch"], "actual dispatch")
    source = (prepared / "kernel.ll").read_text()
    exported = re.findall(r"^define dso_local void @(llm_attention\.packet_batch(?:\.blocks)?)\(", source, re.M)
    require(len(exported) == 1, "missing/ambiguous exported packet entry")
    equal(manifest["symbol"], exported[0], "actual packet entry")
    equal(manifest["abi"], 0 if exported[0].endswith(".blocks") else 2, "actual native entry ABI")
    workspace = re.findall(r"\bprivate_workspace_bytes=(\d+)", metadata["realization"])
    require(len(workspace) == 1 and manifest["workspace_bytes"] == int(workspace[0]), "actual workspace differs from replay")
    require(set(manifest["imports"]) <= {"_memcpy", "_memset", "_bzero", "___chkstk_darwin"}, "unexpected native imports")
    command = load(folder / "capture.command.json")
    equal(command.get("returncode"), 0, "capture process")
    arguments = command["command"]
    require(Path(arguments[0]).name == "benchmark_tile_xir", "wrong native capture executable")
    equal(arguments[1:-1], ["llm", "attention", ",".join(map(str, dims)), *map(str, block), "3", "1", "1"], "capture CLI")
    equal(original_path(root, prepared, arguments[-1]), folder / "output.f32", "capture prefix")
    flag = f"LUISA_SIMD_{'ENABLE' if packet else 'DISABLE'}_FULL_PACKET_SPECIALIZATION"
    environment = command["environment"]
    equal(set(environment), {"LUISA_TILE_BENCH_XIR_BACKEND", "LUISA_TILE_BENCH_ATTENTION_QK", "LUISA_TILE_BENCH_ATTENTION_PV",
                             "LUISA_TILE_BENCH_DUMP_SOURCE", "LUISA_SIMD_DUMP_ASSEMBLY_DIR", "LUISA_TILE_BENCH_XIR_MMA_OUTPUT_BLOCK", "LUISA_TILE_BENCH_XIR_MMA_UNROLL_TERMS", flag}, "recorded environment key set")
    for key, value in {flag: "1", "LUISA_TILE_BENCH_XIR_MMA_UNROLL_TERMS": str(cap), "LUISA_TILE_BENCH_XIR_MMA_OUTPUT_BLOCK": str(width), "LUISA_TILE_BENCH_XIR_BACKEND": "simd", "LUISA_TILE_BENCH_ATTENTION_QK": "mma", "LUISA_TILE_BENCH_ATTENTION_PV": "mma"}.items():
        equal(environment[key], value, key)
    equal(original_path(root, prepared, environment["LUISA_TILE_BENCH_DUMP_SOURCE"]), folder / "output.f32.source.txt", "source export path")
    equal(original_path(root, prepared, environment["LUISA_SIMD_DUMP_ASSEMBLY_DIR"]), folder / "objects", "object export path")
    for stem in ("prepare",):
        equal(load(folder / f"{stem}.command.json")["returncode"], 0, f"{stem} exit")
    for path in prepared.glob("*.command.json"):
        equal(load(path)["returncode"], 0, str(path))
    tools = {Path(text).name: text for text in manifest["tool_sha256"]}
    equal(set(tools), {"clang++", "llvm-nm"}, "native replay tool set")
    expected_commands = {
        "compiler": [tools["clang++"], "--version"],
        "link": [tools["clang++"], "-dynamiclib", str(prepared / "kernel.o"), "-o", str(prepared / "kernel.dylib")],
        "helper": [tools["clang++"], "-std=c++20", "-O3", "-dynamiclib", "-I" + str(prepared),
                   str(prepared / "native_tile_replay.cpp"), "-o", str(prepared / "replay.dylib")],
        "imports": [tools["llvm-nm"], "--undefined-only", "--just-symbol-name", str(prepared / "kernel.o")],
        "exports": [tools["llvm-nm"], "--defined-only", "--extern-only", "--just-symbol-name", str(prepared / "kernel.o")],
    }
    for stem, expected_argv in expected_commands.items():
        actual_argv = []
        for argument in load(prepared / f"{stem}.command.json")["argv"]:
            prefix = "-I" if argument.startswith("-I") else ""
            text = argument[len(prefix):]
            if ORIGINAL_ROOT_NAME in Path(text).parts:
                argument = prefix + str(original_path(root, prepared, text))
            actual_argv.append(argument)
        equal(actual_argv, expected_argv, "exact link/helper/tool flags " + stem)
    for relative, original in {
        "kernel.ll": folder / "output.f32.source.txt", "captured.f32": folder / "output.f32",
        "expected.f64": folder / "output.f32.expected.f64",
        **{f"input{i}.f32": folder / f"output.f32.input{i}.f32" for i in range(3)},
    }.items():
        equal(digest(original), manifest["files"][relative], "original vs prepared copy " + relative)
    objects = list((folder / "objects").glob("*.o"))
    require(len(objects) == 1, "ambiguous captured ORC object")
    equal(digest(objects[0]), manifest["files"]["kernel.o"], "actual captured ORC object")
    realized = realization_fields(metadata["realization"])
    equal(realized["requested_mma_output_block"], width, "requested MMA output blocking")
    equal(realized["requested_max_unrolled_mma_terms"], cap, "requested MMA-only term cap acknowledgement")
    require("mma_unroll_cost=unmodeled" in metadata["realization"], "unroll cost-model status changed")
    require("mma_blocking_cost=unmodeled" in metadata["realization"], "blocking cost-model status changed")
    spec = specialization(metadata, source)
    if not packet:
        equal(spec["full_packet_specializations"], 0, "disabled packet specialization")
    return dict(manifest=manifest, command=command, source=source, expected=independent,
                realized=realized, prepared=prepared,
                specialization=specialization(metadata, source), capture_max_abs_error=error)


RESOURCE_FIELDS = ("static_snapshot_bytes_per_worker", "static_snapshot_allocations", "snapshot_budget",
                   "private_workspace_bytes", "local_lanes", "max_unrolled_tile_elements", "max_unrolled_region_work",
                   "unordered_reduction_partitions")
OBSERVED_FIELDS = ("requested_mma_output_block", "blocked_mmas", "full_packet_specializations",
                   "full_packet_cloned_instructions", "contiguous_private_reads", "contiguous_private_writes",
                   "requested_max_unrolled_mma_terms", "rolled_mmas")
CONTROLS = ("implementation", "backend", "operation", "dimensions", "attention_block", "attention_qk", "attention_pv",
            "precision", "fast_math", "relaxed_precision", "requested_group_threads", "requested_input_views",
            "reduction_tree", "source_reduction_policy", "input_shapes", "output_shape", "dispatch")
COMMON_FILES = ("input0.f32", "input1.f32", "input2.f32", "expected.f64", "native_tile.py",
                "native_tile_replay.cpp", "backends/simd/llvm/llvm_schedule_codegen.h")
BOUNDARY = ("Common C++ timer; native entries, launch resets, block traversal and compiler-emitted libc/allocations included. "
            "Runtime/Python/JIT/caller allocation/validation excluded.")


def realization_fields(text):
    result = {}
    for key in (*RESOURCE_FIELDS, *OBSERVED_FIELDS):
        found = re.findall(r"\b" + key + r"=(\d+)\b", text)
        require(len(found) == 1, "missing/duplicate realization field " + key)
        result[key] = int(found[0])
    blocks = re.findall(r"\bSchedule blocks=(\d+)", text)
    direct = re.findall(r"\bdirect CFG=(true|false)", text)
    require(len(blocks) == len(direct) == 1, "missing/duplicate CFG diagnostics")
    result["schedule_blocks"] = int(blocks[0])
    result["direct_cfg"] = direct[0] == "true"
    return result


def normalize_variant_realization(text):
    for key in OBSERVED_FIELDS:
        text = re.sub(r"\b" + key + r"=\d+", key + "=VARIANT", text)
    text = re.sub(r"\bSchedule blocks=\d+", "Schedule blocks=VARIANT", text)
    return re.sub(r"\bdirect CFG=(?:true|false)", "direct CFG=VARIANT", text)


def compare_common(left, right, context):
    for key in CONTROLS:
        equal(left["metadata"][key], right["metadata"][key], context + " source/math " + key)
    for key in ("symbol", "abi", "packet_width", "block", "dispatch", "workspace_bytes", "input_files",
                "output_elements", "atol", "rtol", "tool_sha256"):
        equal(left[key], right[key], context + " native contract " + key)
    for relative in COMMON_FILES:
        equal(left["files"][relative], right["files"][relative], context + " input/oracle/helper " + relative)


def audit_replay(root, name, dims, block, width, packet, baseline, candidate):
    experiment = name
    replay_dir = root / f"replay-{experiment}"
    replay = load(replay_dir / "results.json")
    entries = {"u0": baseline, "u8": candidate}
    left, right = baseline["manifest"], candidate["manifest"]
    for key, value in dict(status="passed", artifacts_unchanged=True, cpu_threads=1,
                           metric="single_thread_native_entry_host_wall_us",
                           order_policy="ABBA per cycle; two matched pairs per cycle", boundary=BOUNDARY).items():
        equal(replay.get(key), value, experiment + " replay " + key)
    for key, value in dict(command="replay", cycles=3, samples=7, warmup_ms=30, target_ms=15, validate_only=False).items():
        equal(replay["options"][key], value, "exact replay option " + key)
    equal(replay["runner_sha256"], left["files"]["native_tile.py"], "replay runner identity")
    manifests = [entry["prepared"] / "prepared.json" for entry in (baseline, candidate)]
    equal([original_path(root, root, text) for text in replay["options"]["prepared"]], manifests, "replay variant order")
    equal(original_path(root, root, replay["options"]["output"]), replay_dir, "replay output location")
    equal({original_path(root, root, text): sha for text, sha in replay["prepared_sha256"].items()},
          {path: digest(path) for path in manifests}, "replay prepared hashes")
    process = load(root / f"replay-{experiment}.command.json")
    equal(process.get("returncode"), 0, "replay process exit")
    require(process["finished"] >= process["started"], "unfinished replay process")
    argv = process["command"]
    equal(argv[2:], ["replay", "--prepared", replay["options"]["prepared"][0], "--prepared", replay["options"]["prepared"][1],
                     "--output", replay["options"]["output"], "--cycles", "3", "--samples", "7", "--warmup-ms", "30", "--target-ms", "15"],
          "replay command/options match")
    require(Path(argv[1]).name == "native_tile.py", "wrong replay runner filename")
    visits = replay["visits"]
    require(len(visits) == 12, "missing/extra ABBA visits")
    ratios, hashes, maximum = [], {variant: set() for variant in entries}, 0.0
    for cycle in range(3):
        rows = visits[cycle * 4:cycle * 4 + 4]
        equal([row["variant"] for row in rows], ["u0", "u8", "u8", "u0"], "ABBA order")
        for position, row in enumerate(rows):
            for key, value in dict(cycle=cycle, position=position, valid=True, returncode=0,
                                   all_guards_passed=True, inputs_unchanged=True).items():
                equal(row.get(key), value, "visit " + key)
            samples = row["samples_us"]
            require(len(samples) == 7 and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in samples), "invalid samples")
            require(type(row["repetitions"]) is int and row["repetitions"] > 0, "invalid repetitions")
            equal(row["median_us"], statistics.median(samples), "exact visit median")
            equal(row["output"], f"visit-{cycle}-{position}.f32", "visit output identity")
            output = local(replay_dir, row["output"])
            equal(digest(output), row["output_sha256"], "replay output hash")
            maximum = max(maximum, check_output(output, entries[row["variant"]]["expected"]))
            hashes[row["variant"]].add(row["output_sha256"])
            for key, value in dict(elements=left["output_elements"], atol=ATOL, rtol=RTOL).items():
                equal(row["correctness"][key], value, "visit oracle receipt " + key)
        ratios.extend((rows[1]["median_us"] / rows[0]["median_us"], rows[2]["median_us"] / rows[3]["median_us"]))
    require(all(len(value) == 1 for value in hashes.values()), "fixed output changed across visits")
    for variant, entry in entries.items():
        equal(next(iter(hashes[variant])), entry["manifest"]["files"]["captured.f32"], "capture vs replay bitwise output")
    expected = dict(baseline="u0", candidate="u8", pairs=ratios, median=statistics.median(ratios), minimum=min(ratios), maximum=max(ratios))
    equal(replay["candidate_over_baseline"], expected, "exact paired statistics")
    medians = {variant: statistics.median(row["median_us"] for row in visits if row["variant"] == variant) for variant in entries}
    equal(replay["summary_us"], medians, "exact cohort medians")
    return dict(case=name, dimensions=list(dims), attention_block=list(block), packet=packet, output_block=width,
                medians_us=medians, paired_u8_over_u0=expected["median"],
                pair_range=[min(ratios), max(ratios)], paired_wins=sum(r < 1 for r in ratios),
                observed_baseline=baseline["realized"], observed_candidate=candidate["realized"],
                kernel_object_sha256={variant: entry["manifest"]["files"]["kernel.o"] for variant, entry in entries.items()},
                outputs_bitwise_identical=hashes["u0"] == hashes["u8"],
                llvm_statistics={variant: llvm_statistics(entry["source"]) for variant, entry in entries.items()},
                replay_max_abs_error_vs_independent_fp64=maximum)



def llvm_statistics(source):
    """Textual raw LLVM statistics, not optimized machine instruction counts."""
    result = {}
    # Canonical producer IR uses one-line starts for instructions. Continuation
    # switch entries, comments and labels are excluded, not counted as opcodes.
    instruction = re.compile(
        r"^  (?:%[-a-zA-Z$._0-9]+ = |(?:tail |musttail |notail )?call\b|"
        r"store\b|ret\b|br\b|switch\b|unreachable\b|fence\b|"
        r"invoke\b|resume\b|indirectbr\b|catchret\b|cleanupret\b)")
    for name, body in functions(source).items():
        lines = body.splitlines()
        result[name] = {
            "utf8_bytes": len(body.encode()),
            "text_lines": len(lines),
            "instruction_start_lines": sum(bool(instruction.match(line)) for line in lines),
            "basic_block_labels": len(re.findall(r"^[-a-zA-Z$._0-9]+:", body, re.M)),
            "branch_or_switch_starts": len(re.findall(r"^  (?:br|switch)\b", body, re.M)),
        }
    return result


def math_helpers(source):
    # Resolve LLVM attribute group identities so unrelated declaration insertion
    # cannot cause a spurious mathematical-helper difference.
    attrs = dict(re.findall(r"^attributes (#\d+) = (.*)$", source, re.M))
    def expand(body):
        return re.sub(r"#\d+\b", lambda match: attrs[match[0]], body)
    return {name: expand(body) for name, body in functions(source).items()
            if name.startswith("__luisa_cpu_native_")}


def audit_provenance(root, captures):
    provenance = load(root / "provenance.json")
    archive = root / "sources.tar.gz"
    equal(digest(archive), provenance["source_archive_sha256"], "frozen source archive")
    expected = provenance["source_sha256"]
    members = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            path = Path(member.name)
            require(not path.is_absolute() and ".." not in path.parts, "unsafe source archive member")
            if member.isdir():
                continue
            require(member.isfile() and member.name not in members, "nonregular/duplicate source archive member")
            require(member.name in expected, "unlisted source archive member " + member.name)
            data = tar.extractfile(member)
            require(data is not None, "unreadable source archive member")
            members[member.name] = hashlib.file_digest(data, "sha256").hexdigest()
            equal(members[member.name], expected[member.name], "source archive member " + member.name)
    equal(set(members), set(expected), "complete frozen source inventory")
    prepared_sources = {
        "native_tile.py": "scripts/benchmark/tile_torch/native_tile.py",
        "native_tile_replay.cpp": "scripts/benchmark/tile_torch/native_tile_replay.cpp",
        "backends/simd/llvm/llvm_schedule_codegen.h": "src/backends/simd/llvm/llvm_schedule_codegen.h",
    }
    for capture in captures.values():
        for relative, frozen in prepared_sources.items():
            equal(capture["manifest"]["files"][relative], expected[frozen], "prepared helper vs frozen source")
    # This archive contains generated objects/dylibs and frozen source, not the
    # producer build closure. Historical producer hashes are provenance records,
    # not independently attested binary bytes. Never read the mutable live build
    # directory: it may already hold a different experiment's producer.
    fingerprints = provenance["binary_sha256"]
    require(type(fingerprints) is dict and fingerprints, "missing producer fingerprint records")
    for name, expected_sha in fingerprints.items():
        require(type(name) is str and Path(name).name == name and
                type(expected_sha) is str and re.fullmatch(r"[0-9a-f]{64}", expected_sha),
                "malformed recorded producer fingerprint " + str(name))
    producers = {Path(capture["command"]["command"][0]).name for capture in captures.values()}
    equal(producers, {"benchmark_tile_xir"}, "consistent recorded capture producer")
    require(producers <= fingerprints.keys(), "recorded producer absent from provenance")
    return dict(provenance_sha256=digest(root / "provenance.json"),
                source_archive_sha256=digest(archive), source_members_verified=len(members),
                prepared_helpers_match_frozen_source=True,
                external_live_files_read=False,
                producer_binary_bytes_independently_verified=False,
                recorded_binary_sha256=fingerprints,
                recorded_producer_sha256=fingerprints["benchmark_tile_xir"],
                recorded_tool_sha256=next(iter(captures.values()))["manifest"]["tool_sha256"],
                tool_executable_bytes_independently_verified=False)


def audit(root):
    root = root.resolve()
    cases = declared_cases(root)
    record = load(root / "replays.json")
    expected_records = [dict(case=name, packet=packet, width=width, dimensions=list(dims), block=list(block))
                        for name, dims, block, width, packet in cases]
    equal(record["experiments"], expected_records, "all six predeclared replay rows, including regressions")
    require(record["finished"] >= record["started"], "unfinished replay cohort")
    equal(record["runner_sha256"], digest(root / "run.py"), "experiment runner identity")
    captures = {}
    for name, dims, block, width, packet in cases:
        for cap in (0, 8):
            capture = audit_capture(root, name, dims, block, width, packet, cap)
            captures[name, cap] = capture
            # Freeze storage/resource budget separately from the MMA-only cap.
            equal(capture["realized"]["max_unrolled_tile_elements"], 64, "fixed Tile storage threshold")
            equal(capture["realized"]["max_unrolled_region_work"], 4096, "fixed region expansion budget")
            require(math_helpers(capture["source"]), "missing native math helpers")
            # Captures, prepare receipts and replay raw diagnostics stay visible.
            folder = root / f"{name}-u{cap}"
            for stem in ("capture", "prepare"):
                receipt = load(folder / f"{stem}.command.json")
                require(receipt["finished"] >= receipt["started"], "unfinished command " + stem)
                require(not (folder / f"{stem}.stderr").read_bytes(), "nonempty raw diagnostics " + stem)
            prep = load(folder / "prepare.command.json")["command"]
            expected_suffix = [
                "prepare", "--capture-kind", "llm", "--llvm", "/opt/homebrew/opt/llvm/bin",
                "--prefix", str(folder / "output.f32"), "--log", str(folder / "capture.stdout"),
                "--objects", str(folder / "objects"), "--output", str(capture["prepared"]),
                "--name", f"u{cap}",
            ]
            actual_suffix = [str(original_path(root, root, arg)) if ORIGINAL_ROOT_NAME in Path(arg).parts else arg
                             for arg in prep[2:]]
            equal(actual_suffix, expected_suffix, "complete native preparation options")
            require(Path(prep[1]).name == "native_tile.py", "wrong preparation runner")
        baseline, candidate = captures[name, 0], captures[name, 8]
        compare_common(baseline["manifest"], candidate["manifest"], name)
        for key in RESOURCE_FIELDS:
            equal(baseline["realized"][key], candidate["realized"][key], "equal static resource capacity " + key)
        equal(normalize_variant_realization(baseline["manifest"]["metadata"]["realization"]),
              normalize_variant_realization(candidate["manifest"]["metadata"]["realization"]),
              "fixed planner fields outside observed lowering counters")
        equal(baseline["command"]["command"][:-1], candidate["command"]["command"][:-1],
              "same native producer and CLI math configuration")
        equal(math_helpers(baseline["source"]), math_helpers(candidate["source"]), "same native math helpers and attributes")
        # No assertion that an object changes, a clone is admitted, a rolled
        # count increases, output becomes bitwise different, or timing improves.
    provenance = audit_provenance(root, captures)
    rows = [audit_replay(root, name, dims, block, width, packet, captures[name, 0], captures[name, 8])
            for name, dims, block, width, packet in cases]
    for name, *_ in cases:
        require(not (root / f"replay-{name}.stderr").read_bytes(), "nonempty replay raw diagnostics")
    equal(len(captures), 12, "complete capture count")
    equal(len(rows), 6, "complete experiment count")
    # Digest raw command/log/result/source/assembly/tensors in this CPU cohort,
    # including files not named by a prepared manifest. This is a new audit
    # inventory, not falsely described as a historical producer attestation.
    evidence = [root / "run.py", root / "replays.json"]
    for name, *_ in cases:
        for cap in (0, 8):
            evidence.extend(path for path in (root / f"{name}-u{cap}").rglob("*") if path.is_file())
            evidence.append(root / f"prepared-{name}-u{cap}/prepared.json")
        evidence.extend(path for path in (root / f"replay-{name}").rglob("*") if path.is_file())
        evidence.extend(root / f"replay-{name}.{suffix}" for suffix in ("command.json", "stdout", "stderr"))
    raw_hashes = {str(path.relative_to(root)): digest(path) for path in sorted(set(evidence))}
    return dict(status="passed", independent_numpy_oracles=12, capture_output_checks=24,
                replay_output_checks=72, timer_samples=504,
                all_declared_cases_and_negative_results_retained=True,
                run_sha256=digest(root / "run.py"), audit_sha256=digest(Path(__file__)),
                provenance=provenance, static_resource_capacity_checked=list(RESOURCE_FIELDS),
                captures={f"{name}-u{cap}": dict(realized=data["realized"],
                          llvm_statistics=llvm_statistics(data["source"]),
                          kernel_object_sha256=data["manifest"]["files"]["kernel.o"],
                          prepared_manifest_sha256=digest(data["prepared"] / "prepared.json"),
                          capture_max_abs_error_vs_independent_fp64=data["capture_max_abs_error"])
                          for (name, cap), data in captures.items()},
                experiments=rows, raw_evidence_sha256=raw_hashes,
                distinct_files_hashed=len({key[0] for key in HASHES}),
                limitations=[
                    "Single-thread native-entry host-wall includes launch resets, traversal and compiler-emitted helpers; not pure hardware execution or runtime-dispatch E2E.",
                    "This six-case pilot changes only requested MMA contraction unroll cap 0 to 8 at fixed output blocking and packet-enable option; it is not a calibrated cost-model or default-policy recommendation.",
                    "Rolled/blocked MMA and Schedule counts are producer diagnostics. Actual LLVM function bodies, clone definition/call sites, hashes and textual instruction counts are independently inspected, not a proof mapping every loop to a semantic MMA.",
                    "Cap0 may already roll contractions over the fixed general expansion threshold; no assumption of zero baseline rolled MMAs is made.",
                    "Static snapshot/workspace capacities match; register allocation, cache behavior, code size and stack traffic can differ.",
                    "Packet realization may differ even when requested enable is held fixed. A restored/rejected clone is part of this combined lowering effect, not isolated rolling at equal cloned code.",
                    "FP64 output validation checks every captured and replayed element. Guard/input-immutability checks additionally rely on C++ receipts, not retained guard allocations.",
                    "Source archive and prepared helper/object bytes are verified; producer and LLVM tool executable fingerprints are historical records checked for internal consistency, not independent binary-byte or loader-closure attestations. No live build or LLVM installation is read. Complete inherited environment is not recorded.",
                    "Saved kernel.o bytes match the captured ORC object; replayed dylib and helper bytes match manifests, with exact link commands checked, but the audit does not execute, relink or disassemble native code.",
                    "Six paired ratios per row and their min/max are descriptive, not confidence intervals. All negative/unchanged rows remain; no Torch/MPS competitiveness conclusion is implied."
                ])


if __name__ == "__main__":
    require(len(sys.argv) == 2, "usage: audit.py EXPERIMENT_DIRECTORY")
    print(json.dumps(audit(Path(sys.argv[1])), indent=2, allow_nan=False))
