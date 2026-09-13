"""Read-only full-packet attention audit; prints JSON, never modifies evidence.

Usage: python audit.py EXPERIMENT_DIRECTORY
Requires NumPy. No benchmark/compiler/library is executed or loaded. All ten
declared cases, including unchanged and negative controls, remain mandatory.
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


ORIGINAL_ROOT_NAME = "luisa-attention-full-packet.0c45Je"
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
    if ORIGINAL_ROOT_NAME in path.parts:
        offset = path.parts.index(ORIGINAL_ROOT_NAME) + 1
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
    definitions = {}
    # Parse literals; never import/execute the experiment orchestrator.
    for node in ast.parse((root / "run.py").read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("CASES", "HELDOUT"):
                    definitions[target.id] = ast.literal_eval(node.value)
    require(set(definitions) == {"CASES", "HELDOUT"}, "missing declared experiment matrices")
    require(len(definitions["CASES"]) == 4 and len(definitions["HELDOUT"]) == 6, "case coverage differs from 4 + 6")
    cases = [(group, name, tuple(dims), tuple(block)) for key, group in (("CASES", "screen"), ("HELDOUT", "heldout"))
             for name, dims, block in definitions[key]]
    require(len({name for _, name, _, _ in cases}) == 10, "duplicate case name")
    for group, name, dims, block in cases:
        require(len(dims) == 7 and len(block) == 2 and all(type(v) is int and v > 0 for v in (*dims, *block)), f"bad case {name}")
        require(dims[1] % dims[2] == 0 and dims[4] >= dims[3], f"bad attention semantics {name}")
    return cases


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


def audit_capture(root, name, dims, block, variant):
    folder = root / f"{name}-{variant}"
    prepared = root / f"prepared-{name}-{variant}"
    manifest = load(prepared / "prepared.json")
    for key, value in dict(status="prepared", format="native-tile-entry-v1", name=variant, capture_kind="llm").items():
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
    flag = f"LUISA_SIMD_{'ENABLE' if variant == 'on' else 'DISABLE'}_FULL_PACKET_SPECIALIZATION"
    environment = command["environment"]
    equal(set(environment), {"LUISA_TILE_BENCH_XIR_BACKEND", "LUISA_TILE_BENCH_ATTENTION_QK", "LUISA_TILE_BENCH_ATTENTION_PV",
                             "LUISA_TILE_BENCH_DUMP_SOURCE", "LUISA_SIMD_DUMP_ASSEMBLY_DIR", flag}, "recorded environment key set")
    for key, value in {flag: "1", "LUISA_TILE_BENCH_XIR_BACKEND": "simd", "LUISA_TILE_BENCH_ATTENTION_QK": "mma", "LUISA_TILE_BENCH_ATTENTION_PV": "mma"}.items():
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
    return dict(manifest=manifest, command=command, source=source, expected=independent,
                specialization=specialization(metadata, source), capture_max_abs_error=error)


def audit(root):
    root = root.resolve()
    cases = declared_cases(root)
    experiment_records = {}
    for group, filename in (("screen", "replays.json"), ("heldout", "replays-heldout.json")):
        record = load(root / filename)
        wanted = [(name, list(dims), list(block)) for cohort, name, dims, block in cases if cohort == group]
        equal([(e["case"], e["dimensions"], e["block"]) for e in record["experiments"]], wanted, filename + " coverage")
        require(record["finished"] >= record["started"], "unfinished cohort")
        experiment_records.update({entry["case"]: entry for entry in record["experiments"]})
    summaries, checked_visits, captured = [], 0, 0
    controls = ("operation", "dimensions", "attention_block", "attention_qk", "attention_pv", "precision", "fast_math", "relaxed_precision",
                "requested_group_threads", "requested_input_views", "reduction_tree", "source_reduction_policy", "input_shapes", "output_shape", "dispatch")
    common_files = ("input0.f32", "input1.f32", "input2.f32", "expected.f64", "native_tile.py", "native_tile_replay.cpp", "backends/simd/llvm/llvm_schedule_codegen.h")
    for group, name, dims, block in cases:
        pair = {variant: audit_capture(root, name, dims, block, variant) for variant in ("off", "on")}
        captured += 2
        left, right = pair["off"]["manifest"], pair["on"]["manifest"]
        for key in controls:
            equal(left["metadata"][key], right["metadata"][key], f"{name} source/math control {key}")
        for key in ("symbol", "abi", "packet_width", "block", "dispatch", "workspace_bytes", "input_files", "output_elements", "atol", "rtol", "tool_sha256"):
            equal(left[key], right[key], f"{name} replay ABI/options {key}")
        for key in common_files:
            equal(left["files"][key], right["files"][key], f"{name} input/oracle/helper source {key}")
        equal(pair["off"]["command"]["command"][:-1], pair["on"]["command"]["command"][:-1], "native executable/CLI controls")
        equal(normalized_realization(left["metadata"]["realization"]), normalized_realization(right["metadata"]["realization"]), "all other planner/codegen realization fields")
        definitions = {variant: functions(pair[variant]["source"]) for variant in ("off", "on")}
        untouched = lambda defs: {key: value for key, value in defs.items() if key not in ("llm_attention.full_packet", "llm_attention.packet_batch", "llm_attention.packet_batch.blocks")}
        equal(untouched(definitions["off"]), untouched(definitions["on"]), "generic kernel and native math function bodies")
        for function in ("llm_attention.packet_batch", "llm_attention.packet_batch.blocks"):
            require((function in definitions["off"]) == (function in definitions["on"]), "wrapper set changed")
            if function not in definitions["off"]:
                continue
            restored = re.sub(r"@llm_attention\.full_packet\(([^\n]*)\)", r"@llm_attention(\1, i32 8)", definitions["on"][function])
            equal(restored, definitions["off"][function], "only full-packet call retargeting in " + function)
        equal(re.findall(r"^attributes .*", pair["off"]["source"], re.M), re.findall(r"^attributes .*", pair["on"]["source"], re.M), "LLVM math/function attributes")
        equal(pair["off"]["specialization"]["full_packet_specializations"], 0, "off must not specialize")
        triggered = pair["on"]["specialization"]["full_packet_specializations"] != 0
        object_identical = left["files"]["kernel.o"] == right["files"]["kernel.o"]
        if not triggered:
            require(object_identical and left["files"]["kernel.ll"] == right["files"]["kernel.ll"], "non-trigger control changed actual object/LLVM")
        else:
            require(not object_identical, "triggered specialization did not change actual object")
        replay_dir = root / f"replay-{name}"
        replay = load(replay_dir / "results.json")
        for key, value in dict(status="passed", artifacts_unchanged=True, cpu_threads=1,
                               metric="single_thread_native_entry_host_wall_us", order_policy="ABBA per cycle; two matched pairs per cycle").items():
            equal(replay.get(key), value, "replay " + key)
        equal(replay["boundary"], "Common C++ timer; native entries, launch resets, block traversal and compiler-emitted libc/allocations included. Runtime/Python/JIT/caller allocation/validation excluded.", "timing boundary")
        for key, value in dict(command="replay", cycles=3, samples=7, warmup_ms=30, target_ms=15, validate_only=False).items():
            equal(replay["options"][key], value, "exact replay option " + key)
        equal(replay["runner_sha256"], left["files"]["native_tile.py"], "replay runner identity")
        prepared_paths = [root / f"prepared-{name}-{variant}" / "prepared.json" for variant in ("off", "on")]
        equal([original_path(root, root, text) for text in replay["options"]["prepared"]], prepared_paths, "replay variant order")
        equal({original_path(root, root, text): sha for text, sha in replay["prepared_sha256"].items()}, {path: digest(path) for path in prepared_paths}, "replay manifest hashes")
        process = load(root / f"replay-{name}.command.json")
        equal(process.get("returncode"), 0, "replay process exit")
        equal(process["command"], experiment_records[name]["command"], "cohort command receipt")
        equal(process["command"][2:], ["replay", "--prepared", *replay["options"]["prepared"][:1], "--prepared", *replay["options"]["prepared"][1:],
                                      "--output", replay["options"]["output"], "--cycles", "3", "--samples", "7", "--warmup-ms", "30", "--target-ms", "15"], "replay command/options match")
        visits = replay["visits"]
        require(len(visits) == 12, "missing/extra ABBA visits")
        ratios, variant_hashes = [], {"off": set(), "on": set()}
        maximum = 0.0
        for cycle in range(3):
            rows = visits[cycle * 4:cycle * 4 + 4]
            equal([row["variant"] for row in rows], ["off", "on", "on", "off"], "ABBA order")
            for position, row in enumerate(rows):
                for key, value in dict(cycle=cycle, position=position, valid=True, returncode=0, all_guards_passed=True, inputs_unchanged=True).items():
                    equal(row.get(key), value, "visit " + key)
                samples = row["samples_us"]
                require(len(samples) == 7 and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in samples), "invalid native timer samples")
                require(type(row["repetitions"]) is int and row["repetitions"] > 0, "invalid native timer repetitions")
                equal(row["median_us"], statistics.median(samples), "exact visit median")
                output = local(replay_dir, row["output"])
                equal(row["output"], f"visit-{cycle}-{position}.f32", "visit output identity")
                equal(digest(output), row["output_sha256"], "replay output hash")
                maximum = max(maximum, check_output(output, pair[row["variant"]]["expected"]))
                variant_hashes[row["variant"]].add(row["output_sha256"])
                for key, value in dict(elements=left["output_elements"], atol=ATOL, rtol=RTOL).items():
                    equal(row["correctness"][key], value, "visit oracle receipt " + key)
                checked_visits += 1
            ratios.extend((rows[1]["median_us"] / rows[0]["median_us"], rows[2]["median_us"] / rows[3]["median_us"]))
        require(all(len(values) == 1 for values in variant_hashes.values()), "fixed entry output changed between visits")
        for variant in ("off", "on"):
            equal(next(iter(variant_hashes[variant])), pair[variant]["manifest"]["files"]["captured.f32"], "capture vs replay bitwise output")
        expected_ratio = dict(baseline="off", candidate="on", pairs=ratios, median=statistics.median(ratios), minimum=min(ratios), maximum=max(ratios))
        equal(replay["candidate_over_baseline"], expected_ratio, "exact paired statistics")
        medians = {variant: statistics.median(row["median_us"] for row in visits if row["variant"] == variant) for variant in ("off", "on")}
        equal(replay["summary_us"], medians, "exact cohort medians")
        summaries.append(dict(group=group, case=name, dimensions=list(dims), attention_block=list(block),
                              specialization={variant: pair[variant]["specialization"] for variant in ("off", "on")},
                              object_identical=object_identical, outputs_bitwise_identical=variant_hashes["off"] == variant_hashes["on"],
                              medians_us=medians, paired_on_over_off=expected_ratio["median"], pair_range=[min(ratios), max(ratios)],
                              paired_wins=sum(value < 1 for value in ratios), replay_max_abs_error_vs_independent_fp64=maximum))
    require(captured == 20 and checked_visits == 120 and len(summaries) == 10, "incomplete full experiment")
    return dict(status="passed", independent_numpy_oracles=captured, capture_output_checks=2 * captured,
                replay_output_checks=checked_visits, timer_samples=checked_visits * 7,
                distinct_files_hashed=len({key[0] for key in HASHES}), run_sha256=digest(root / "run.py"),
                controls_checked=list(controls), all_declared_cases_retained=True,
                limitations=["Native-entry host-wall timing includes launch resets, block traversal and compiler-emitted helper/libc work; not hardware-only execution time.",
                             "Buffer/workspace/input guards are C++ execution receipts, not independently saved guard allocations.",
                             "Generic LLVM body/math/ABI and wrapper retargeting are checked; this audit is not a proof of the optimizer's transformation legality.",
                             "Capture commands record selected LUISA flags and executable paths, not historical executable/dylib bytes or the complete inherited process environment.",
                             "Original external source identities are verified against frozen prepared snapshots; tool identities require the recorded LLVM executables to remain available.",
                             "Screen and held-out cohorts retain separate labels; ranges describe six matched pairs, not confidence intervals or Torch/MPS parity."],
                experiments=summaries)


if __name__ == "__main__":
    require(len(sys.argv) == 2, "usage: audit.py EXPERIMENT_DIRECTORY")
    print(json.dumps(audit(Path(sys.argv[1])), indent=2, allow_nan=False))
