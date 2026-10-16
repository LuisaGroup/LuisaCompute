"""Read-only MMA-output-block attention audit; no native code executed or loaded.

Usage: python audit.py EXPERIMENT_DIRECTORY PREVIOUS_FULL_PACKET_DIRECTORY
Requires NumPy. All 12 captures and 8 paired experiments remain mandatory.
Helper validation/oracle code is reused from the separate full-packet audit.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

import numpy as np


ORIGINAL_ROOT_NAME = "luisa-attention-mma-block.PNrrvz"
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
    marker = next((m for m in (ORIGINAL_ROOT_NAME, "luisa-attention-full-packet.0c45Je") if m in path.parts), None)
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
    equal(definitions, [[("decode", (1, 8, 2, 1, 2053, 80, 96), (1, 16)),
                         ("prefill", (1, 4, 2, 32, 65, 32, 32), (4, 16))]], "declared two-case matrix")
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


def audit_capture(root, name, dims, block, packet, width):
    variant = f"{packet}-r{width}"
    folder = root / f"{name}-{variant}"
    prepared = root / f"prepared-{name}-{variant}"
    manifest = load(prepared / "prepared.json")
    for key, value in dict(status="prepared", format="native-tile-entry-v1", name=f"r{width}", capture_kind="llm").items():
        equal(manifest.get(key), value, f"{name}/{variant}/{key}")
    declared = manifest["files"]
    require(set(declared) == {str(path.relative_to(prepared)) for path in prepared.rglob("*") if path.is_file() and path.name != "prepared.json"}, "prepared inventory has missing/extra files")
    for relative, expected in declared.items():
        equal(digest(local(prepared, relative)), expected, str(prepared / relative))
    for text, expected in manifest["original_sha256"].items():
        equal(digest(original_path(root, prepared, text)), expected, text)
    for text, expected in manifest["tool_sha256"].items():
        # LLVM installs clang++ as a symlink to clang; attest the bytes of its
        # resolved executable, while prepared artifacts remain nonsymlink-only.
        equal(digest(Path(text).resolve()), expected, "live tool identity " + text)
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
    flag = f"LUISA_SIMD_{'ENABLE' if packet == 'on' else 'DISABLE'}_FULL_PACKET_SPECIALIZATION"
    environment = command["environment"]
    equal(set(environment), {"LUISA_TILE_BENCH_XIR_BACKEND", "LUISA_TILE_BENCH_ATTENTION_QK", "LUISA_TILE_BENCH_ATTENTION_PV",
                             "LUISA_TILE_BENCH_DUMP_SOURCE", "LUISA_SIMD_DUMP_ASSEMBLY_DIR", "LUISA_TILE_BENCH_XIR_MMA_OUTPUT_BLOCK", flag}, "recorded environment key set")
    for key, value in {flag: "1", "LUISA_TILE_BENCH_XIR_MMA_OUTPUT_BLOCK": str(width), "LUISA_TILE_BENCH_XIR_BACKEND": "simd", "LUISA_TILE_BENCH_ATTENTION_QK": "mma", "LUISA_TILE_BENCH_ATTENTION_PV": "mma"}.items():
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
    equal(realized["blocked_mmas"], 0 if width == 1 else 1, "actual blocked MMA count")
    require("mma_blocking_cost=unmodeled" in metadata["realization"], "blocking cost-model status changed")
    spec = specialization(metadata, source)
    if packet == "off":
        equal(spec["full_packet_specializations"], 0, "disabled packet specialization")
    return dict(manifest=manifest, command=command, source=source, expected=independent,
                realized=realized, prepared=prepared,
                specialization=specialization(metadata, source), capture_max_abs_error=error)


RESOURCE_FIELDS = ("static_snapshot_bytes_per_worker", "static_snapshot_allocations", "snapshot_budget",
                   "private_workspace_bytes", "local_lanes", "max_unrolled_tile_elements", "max_unrolled_region_work",
                   "unordered_reduction_partitions")
OBSERVED_FIELDS = ("requested_mma_output_block", "blocked_mmas", "full_packet_specializations",
                   "full_packet_cloned_instructions", "contiguous_private_reads", "contiguous_private_writes")
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
    return result


def normalize_blocking_realization(text):
    for key in OBSERVED_FIELDS:
        text = re.sub(r"\b" + key + r"=\d+", key + "=VARIANT", text)
    return re.sub(r"\bSchedule blocks=\d+", "Schedule blocks=VARIANT", text)


def compare_common(left, right, context):
    for key in CONTROLS:
        equal(left["metadata"][key], right["metadata"][key], context + " source/math " + key)
    for key in ("symbol", "abi", "packet_width", "block", "dispatch", "workspace_bytes", "input_files",
                "output_elements", "atol", "rtol", "tool_sha256"):
        equal(left[key], right[key], context + " native contract " + key)
    for relative in COMMON_FILES:
        equal(left["files"][relative], right["files"][relative], context + " input/oracle/helper " + relative)


def previous_baseline(previous_root, name, packet, current):
    prepared = previous_root / f"prepared-{name}-{packet}"
    old = load(prepared / "prepared.json")
    equal(old["status"], "prepared", "previous baseline status")
    equal(old["name"], packet, "previous baseline variant")
    for relative, expected in old["files"].items():
        equal(digest(local(prepared, relative)), expected, "previous manifest file " + relative)
    for text, expected in old["original_sha256"].items():
        equal(digest(original_path(previous_root, prepared, text)), expected, "previous original " + text)
    compare_common(old, current["manifest"], f"{name}/{packet} old vs R1")
    for relative in ("kernel.ll", "kernel.o", "captured.f32"):
        equal(old["files"][relative], current["manifest"]["files"][relative], "R1 old baseline byte identity " + relative)
    # New implementation reports three new fields; no old measurements are
    # mixed into the new paired statistics.
    stripped = re.sub(r"requested_mma_output_block=1; blocked_mmas=0; mma_blocking_cost=unmodeled; ", "",
                      current["manifest"]["metadata"]["realization"])
    equal(stripped, old["metadata"]["realization"], "R1 realization excluding newly introduced diagnostics")
    return dict(previous_manifest_sha256=digest(prepared / "prepared.json"), current_manifest_sha256=digest(current["prepared"] / "prepared.json"),
                kernel_ll_sha256=old["files"]["kernel.ll"], kernel_object_sha256=old["files"]["kernel.o"],
                inputs_oracle_helpers_identical=True, output_identical=True)


def audit_replay(root, name, packet, width, baseline, candidate):
    experiment = f"{name}-{packet}-r{width}"
    replay_dir = root / f"replay-{experiment}"
    replay = load(replay_dir / "results.json")
    entries = {"r1": baseline, f"r{width}": candidate}
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
        equal([row["variant"] for row in rows], ["r1", f"r{width}", f"r{width}", "r1"], "ABBA order")
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
    expected = dict(baseline="r1", candidate=f"r{width}", pairs=ratios, median=statistics.median(ratios), minimum=min(ratios), maximum=max(ratios))
    equal(replay["candidate_over_baseline"], expected, "exact paired statistics")
    medians = {variant: statistics.median(row["median_us"] for row in visits if row["variant"] == variant) for variant in entries}
    equal(replay["summary_us"], medians, "exact cohort medians")
    return dict(case=name, packet=packet, output_block=width, medians_us=medians, paired_candidate_over_r1=expected["median"],
                pair_range=[min(ratios), max(ratios)], paired_wins=sum(r < 1 for r in ratios),
                observed_baseline=baseline["realized"], observed_candidate=candidate["realized"],
                kernel_object_sha256={variant: entry["manifest"]["files"]["kernel.o"] for variant, entry in entries.items()},
                outputs_bitwise_identical=hashes["r1"] == hashes[f"r{width}"],
                replay_max_abs_error_vs_independent_fp64=maximum)


def audit(root, previous_root):
    root, previous_root = root.resolve(), previous_root.resolve()
    cases = declared_cases(root)
    record = load(root / "replays.json")
    expected_records = [dict(case=name, packet=packet, width=width, dimensions=list(dims), block=list(block))
                        for name, dims, block in cases for packet in ("off", "on") for width in (2, 4)]
    equal(record["experiments"], expected_records, "complete eight-row predeclared replay matrix")
    require(record["finished"] >= record["started"], "unfinished replay cohort")
    captures, baseline_identity = {}, {}
    for name, dims, block in cases:
        for packet in ("off", "on"):
            for width in (1, 2, 4):
                captures[name, packet, width] = audit_capture(root, name, dims, block, packet, width)
            r1 = captures[name, packet, 1]
            baseline_identity[f"{name}-{packet}"] = previous_baseline(previous_root, name, packet, r1)
            for width in (2, 4):
                candidate = captures[name, packet, width]
                compare_common(r1["manifest"], candidate["manifest"], f"{name}/{packet}/R{width}")
                for key in RESOURCE_FIELDS:
                    equal(r1["realized"][key], candidate["realized"][key], "static capacity " + key)
                equal(normalize_blocking_realization(r1["manifest"]["metadata"]["realization"]),
                      normalize_blocking_realization(candidate["manifest"]["metadata"]["realization"]), "all unchanged planner realization fields")
                equal(r1["command"]["command"][:-1], candidate["command"]["command"][:-1], "identical native executable/CLI")
                # The generic kernel changes intentionally. Native exp helper
                # body/attributes must not change, nor may math flags drift.
                defs = functions(r1["source"])
                cdefs = functions(candidate["source"])
                math_bodies = lambda d: {k: v for k, v in d.items() if k.startswith("__luisa_cpu_native_")}
                equal(math_bodies(defs), math_bodies(cdefs), "native math helper bodies")
                equal(re.findall(r"^attributes .*", r1["source"], re.M), re.findall(r"^attributes .*", candidate["source"], re.M), "LLVM attributes")
                require(r1["manifest"]["files"]["kernel.o"] != candidate["manifest"]["files"]["kernel.o"], "blocking did not change actual object")
        for width in (1, 2, 4):
            off, on = captures[name, "off", width], captures[name, "on", width]
            compare_common(off["manifest"], on["manifest"], f"{name}/R{width} packet off vs on")
            equal(normalized_realization(off["manifest"]["metadata"]["realization"]),
                  normalized_realization(on["manifest"]["metadata"]["realization"]), "only packet realization fields change with enable")
            if on["specialization"]["full_packet_specializations"] == 0:
                for relative in ("kernel.ll", "kernel.o"):
                    equal(off["manifest"]["files"][relative], on["manifest"]["files"][relative], "no-trigger packet control " + relative)
    rows = [audit_replay(root, name, packet, width, captures[name, packet, 1], captures[name, packet, width])
            for name, _, _ in cases for packet in ("off", "on") for width in (2, 4)]
    equal(len(captures), 12, "capture coverage")
    equal(len(rows), 8, "pair coverage")
    run_sha = digest(root / "run.py")
    return dict(status="passed", independent_numpy_oracles=12, capture_output_checks=24, replay_output_checks=96,
                timer_samples=672, all_declared_cases_and_negative_results_retained=True,
                distinct_files_hashed=len({key[0] for key in HASHES}), run_sha256=run_sha,
                previous_baseline_identities=baseline_identity, static_resource_capacity_checked=list(RESOURCE_FIELDS),
                captures={f"{name}-{packet}-r{width}": dict(realized=data["realized"],
                          kernel_object_sha256=data["manifest"]["files"]["kernel.o"],
                          capture_max_abs_error_vs_independent_fp64=data["capture_max_abs_error"])
                          for (name, packet, width), data in captures.items()},
                experiments=rows,
                limitations=[
                    "Single-thread native-entry host-wall, including launch reset/traversal and generated helpers; not hardware-only execution time.",
                    "MMA output blocking intentionally changes generic LLVM bodies. Source/math metadata, native math helpers, inputs, complete output oracles and resource capacities are checked; this is not a transformation-legality proof.",
                    "Static snapshot/workspace capacity is unchanged; this does not assert unchanged register pressure, occupancy, generated instruction counts or stack traffic.",
                    "Packet enable is held fixed, but realized specialization may stop triggering when the blocked body exceeds its budget. Counts remain visible; such a row is the combined effect, not isolated blocking with equal cloned code.",
                    "Guards/input immutability are C++ receipts, not saved guard allocations independently inspected by NumPy.",
                    "Capture command records identify paths and selected LUISA flags, not historical executable/dylib closure or complete inherited environment.",
                    "Prior R1 code/input/output identity is checked; prior timing is not pooled into these new matched comparisons. Six-pair ranges are not confidence intervals; no Torch/MPS parity claim."
                ])


if __name__ == "__main__":
    require(len(sys.argv) == 3, "usage: audit.py EXPERIMENT_DIRECTORY PREVIOUS_FULL_PACKET_DIRECTORY")
    print(json.dumps(audit(Path(sys.argv[1]), Path(sys.argv[2])), indent=2, allow_nan=False))
