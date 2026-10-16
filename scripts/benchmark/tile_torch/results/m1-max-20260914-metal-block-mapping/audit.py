"""Offline audit of the fixed-BQ Metal block experiment; never executes native code.

Accept the original raw root with --helpers-dir, or an extracted package root
containing raw/, tensors/ and sources/. Requires NumPy for the complete FP64
oracle, but neither Torch nor a live build/toolchain. --output never overwrites.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import sys

CASES = [
    ["attention-small-q4", "attention", [1, 4, 2, 16, 33, 32, 32], [4, 16], 16, 32],
    ["attention-small-q1", "attention", [1, 4, 2, 16, 33, 32, 32], [1, 16], 64, 64],
    ["attention-prefill-q4", "attention", [1, 4, 2, 64, 128, 64, 64], [4, 16], 64, 64],
    ["attention-prefill-q1", "attention", [1, 4, 2, 64, 128, 64, 64], [1, 16], 256, 256],
    ["rmsnorm-tail", "rmsnorm", [257, 1025], [1, 1], 257, 256],
    ["swiglu-wide-grid", "swiglu", [8192, 65], [1, 1], 8192, 256],
]
ORDER = [0, 32, 32, 0] * 2
HELPERS = ("metal4_timing.py", "compare_llm.py", "run.py", "repeat.py")
TEST_HELPERS = ("test_metal4_timing.py", "test_compare_llm.py")
METRICS = tuple(f"{phase}_{kind}" for phase in ("throughput", "latency") for kind in (
    "instrumented_dispatch_ns", "feedback_only_command_buffer_ns_per_dispatch", "host_wall_us_per_dispatch"))
STRUCTURE = ("local_lanes", "static_snapshot_bytes_per_worker", "static_snapshot_allocations",
             "max_unrolled_tile_elements", "max_unrolled_region_work", "unordered_reduction_partitions",
             "fused_reduction_loads", "fused_reduction_expressions", "fused_pointwise_regions", "deferred_maps",
             "blocked_mmas", "two_dimensional_mmas", "rolled_mmas", "fast_math", "ordered_reduction", "strict_mma")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(path):
    require(path.is_file() and not path.is_symlink(), "expected a regular nonsymlink file: " + str(path))
    return dict(bytes=path.stat().st_size, sha256=sha(path))


def contained(root, relative):
    relative = PurePosixPath(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "unsafe relative path")
    path = root / str(relative)
    require(path.resolve().is_relative_to(root.resolve()), "path escapes evidence root")
    return path


def one_named(mapping, name):
    matches = [(p, value) for p, value in mapping.items() if Path(p).name == name]
    require(len(matches) == 1, "missing/ambiguous identity: " + name)
    return matches[0]


def normalize_tmp(value):
    # macOS /tmp is the /private/tmp alias. Compare recorded lexical paths
    # without consulting the live filesystem during a portable offline audit.
    return value.removeprefix("/private") if value.startswith("/private/tmp/") else value


def validate_plan(raw):
    plan, result = read(raw / "plan.json"), read(raw / "results.json")
    require(plan["cases"] == CASES and plan["block_order"] == ORDER, "predeclared experiment changed")
    require((plan["samples"], plan["repetitions"], plan["local_lanes"]) == (3, 8, 1), "protocol changed")
    require(result["status"] == "complete" and result["artifacts_unchanged"] is True and len(result["rows"]) == 48,
            "experiment incomplete or artifacts changed")
    require(not (raw / "failure.json").exists(), "failed visit cannot be accepted as complete")
    exported = read(raw / "source-export.json")
    require(exported["exports"][0]["commit"].startswith(plan["source_head"]), "source HEAD mismatch")
    require(exported["exports"][0]["destination"] == plan["selected_source"], "source export destination mismatch")
    probe_path, probe_sha = one_named(plan["frozen_sha256"], "probe.py")
    require(sha(raw / "probe.py") == probe_sha, "frozen runner changed")
    return plan, result, Path(probe_path).parent


def helper_sources(raw, plan, helper_dir):
    tested = read(raw / "python-protocol/tested-sources.json")
    require({Path(p).name for p in tested} == set(HELPERS + TEST_HELPERS) and len(tested) == 6,
            "expected the six tested Python sources")
    for name in HELPERS + TEST_HELPERS:
        _, expected = one_named(tested, name)
        require(sha(helper_dir / name) == expected, "tested helper changed: " + name)
        if name in HELPERS:
            require(expected == one_named(plan["frozen_sha256"], name)[1], "tested/captured helper mismatch")
    return tested


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--helpers-dir", type=Path)
    parser.add_argument("--inventory", type=Path, help="optional package inventory for every extracted member")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = args.evidence.resolve(strict=True)
    extracted = (base / "raw").is_dir()
    raw = base / "raw" if extracted else base
    helpers = (args.helpers_dir or base / "sources/helpers").resolve(strict=True)
    plan, result, original_raw = validate_plan(raw)
    helper_sources(raw, plan, helpers)
    if args.inventory:
        require(extracted, "inventory validation requires an extracted root")
        inventory = read(args.inventory)
        for name, receipt in inventory["members"].items():
            require(fingerprint(contained(base, name)) == receipt, "archive member mismatch: " + name)
        present = {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
        require(present == set(inventory["members"]), "extracted member set differs from inventory")
    selected_root = Path(plan["selected_source"])
    for original, expected in plan["selected_source_sha256"].items():
        require(Path(original).is_relative_to(selected_root), "selected source outside declared root")
        path = base / "sources/selected" / Path(original).relative_to(selected_root) if extracted else Path(original)
        require(fingerprint(path)["sha256"] == expected, "selected source changed: " + original)
    sys.path.insert(0, str(helpers))
    sys.dont_write_bytecode = True
    import metal4_timing as timing

    def command_receipt(label, status="passed", expected_command=None, final_runner=False, host_warning_false_positive=False):
        folder = contained(raw, label)
        command, outcome = read(folder / "command.json"), read(folder / "result.json")
        process = read(folder / "process.process.json")
        require(outcome["status"] == status and outcome["command"] == command["command"], "command status/argv mismatch")
        if expected_command is not None:
            require(list(map(normalize_tmp, command["command"])) == list(map(normalize_tmp, expected_command)),
                    "unexpected command: " + label)
        require(outcome["started"] == command["started"] and outcome["finished"] >= outcome["started"], "bad command interval")
        require(process["process"] == outcome["process"] and not process["process"]["timed_out"], "process receipt mismatch")
        require((process["process"]["exit_code"] == 0) == (status == "passed" or host_warning_false_positive), "wrong command exit status")
        require(process["logs"] == outcome["process_logs"], "process log receipt mismatch")
        content = {}
        for channel, info in process["logs"].items():
            path = contained(folder, info["path"])
            require(fingerprint(path) == {k: info[k] for k in ("bytes", "sha256")}, "raw command log changed")
            content[channel] = path.read_bytes()
        diagnostics = timing.gpu_failure_diagnostics(content["stdout"], content["stderr"])
        require(diagnostics == process["gpu_failure_diagnostics"] == outcome["gpu_failure_diagnostics"], "diagnostic receipt differs")
        if host_warning_false_positive:
            require(label == "build-head" and status == "failed" and len(diagnostics) == 2,
                    "only the retained host-build false positive is exempted")
            for diagnostic, warning in zip(diagnostics, ("__auto_type", "_Generic")):
                require(diagnostic["channel"] == "stdout" and "/reproc/reproc/src/error.posix.c:" in diagnostic["excerpt"] and
                        "warning:" in diagnostic["excerpt"] and warning in diagnostic["excerpt"], "unexpected build diagnostic")
        else:
            require(not diagnostics, "GPU failure diagnostic")
        if final_runner:
            require(command.get("host_only") is False, "native visit bypasses GPU failure guard")
            for original, expected in command["source_sha256"].items():
                require(plan["frozen_sha256"].get(original) == expected, "capture runner/helper identity mismatch")
        return command, outcome, content

    old_command, old_build, _ = command_receipt("build", "failed")
    require(old_command["command"][:2] == ["cmake", "--build"], "missing original failed build")
    config, config_result, _ = command_receipt("configure-head")
    initial_build, initial_result, _ = command_receipt("build-head", "failed", host_warning_false_positive=True)
    build, build_result, _ = command_receipt("build-head-confirm")
    binary, _ = one_named(plan["frozen_sha256"], "benchmark_tile_xir")
    build_root = str(Path(binary).parent.parent)
    require(build["command"] == ["cmake", "--build", build_root, "-j", "6"], "not a full selected build")
    require(initial_build["command"] == build["command"] and build.get("host_only") is True,
            "host confirmation differs from the full build")
    require(config["command"][config["command"].index("-S") + 1] == str(selected_root) and
            config["command"][config["command"].index("-B") + 1] == build_root, "configure source/build differs")
    unit, unit_result, logs = command_receipt("python-protocol")
    require(unit["command"][1:] == ["-m", "unittest", "test_metal4_timing", "test_compare_llm"], "unexpected unit command")
    require(re.search(rb"Ran 40 tests in [^\n]+\n\nOK\s*$", logs["stderr"]), "40 Python tests did not pass")
    command_receipt("power-assertions")
    require(config_result["finished"] <= initial_result["started"] <= initial_result["finished"] <=
            build_result["started"] <= build_result["finished"] <= plan["started"] and
            unit_result["finished"] <= plan["started"], "build/test gate completed after experiment freeze")

    visits, cases, inputs, outputs, structures, checksums = [], [], {}, {}, {}, {}
    artifacts = None
    last_finished = plan["started"]
    expected_commands = {"build/command.json", "configure-head/command.json", "build-head/command.json", "build-head-confirm/command.json",
                         "python-protocol/command.json", "power-assertions/command.json"}
    for case_index, (case, op, dims, block, dispatch, auto_block) in enumerate(CASES):
        current = []
        for position, requested in enumerate(ORDER):
            row = result["rows"][case_index * 8 + position]
            label = f"{case}-{position}-block{requested}"
            relative = label + "/cohort/results.json"
            require((row["case"], row["position"], row["requested_block"], row["result"]) ==
                    (case, position, requested, relative), "visit identity/order mismatch")
            expected_commands.add(label + "/command.json")
            command = read(raw / label / "command.json")["command"]
            expected = [command[0], one_named(plan["frozen_sha256"], "metal4_timing.py")[0], "--binary", binary,
                        "--case", op + ":" + ",".join(map(str, dims)), "--attention-block", *map(str, block),
                        "--local-lanes", "1", "--block-size", str(requested), "--rounds", "1", "--samples", "3",
                        "--repetitions", "8", "--timeout", "60", "--output", str(original_raw / label / "cohort")]
            _, outcome, _ = command_receipt(label, expected_command=expected, final_runner=True)
            require(outcome["started"] >= last_finished, "overlapping visits")
            last_finished = outcome["finished"]
            cohort_path = raw / relative
            cohort = read(cohort_path)
            require(cohort["cohort_valid"] is cohort["gpu_diagnostics_valid"] is cohort["artifacts_unchanged"] is True,
                    "invalid cohort")
            require(cohort["artifacts_before"] == cohort["artifacts_after"], "changed capture artifact")
            if artifacts is None:
                artifacts = cohort["artifacts_before"]
            require(cohort["artifacts_before"] == artifacts, "artifacts differ across visits")
            for original, info in artifacts.items():
                require(plan["frozen_sha256"].get(original) == info["sha256"], "artifact outside/mismatching frozen set")
            require(len(cohort["results"]) == 1, "unexpected extra visit")
            captured = cohort["results"][0]
            require(captured["status"] == "OK" and captured["valid"] is True and captured["exit_code"] == 0, "capture failed")
            require((captured["operation"], captured["dimensions"], captured["round"], captured["requested_local_lanes"],
                     captured["requested_block_size"]) == (op, dims, 0, 1, requested), "captured controls mismatch")
            protocol = cohort["protocol"]
            for key, value in dict(rounds=1, samples=3, host_repetitions=8, device_repetitions=8, target_ms=20,
                                   warmup_ms=10, attention_block=block, attention_qk="mma", attention_pv="mma",
                                   requested_block_size=requested, zero_overhead_kernel_time=False, serial=True).items():
                require(protocol[key] == value, "protocol denies " + key)
            declared = {"LUISA_TILE_BENCH_METAL4_TIMING": "1", "LUISA_TILE_BENCH_FIXED_REPETITIONS": "8",
                        "LUISA_TILE_BENCH_XIR_BACKEND": "metal4"}
            if requested:
                declared["LUISA_TILE_BENCH_XIR_BLOCK_SIZE"] = str(requested)
            require(cohort["declared_environment"] == declared and captured["environment"] ==
                    {**declared, "LUISA_TILE_BENCH_XIR_LOCAL_LANES": "1"}, "unexpected benchmark overrides")
            stdout, stderr = (contained(cohort_path.parent, captured[key]) for key in ("stdout", "stderr"))
            require(not timing.gpu_failure_diagnostics(stdout.read_bytes(), stderr.read_bytes()), "capture GPU diagnostic")
            payload = read(stdout)
            actual = requested or auto_block
            require(timing.validate_local_lanes(payload["realization"], 1) == captured["actual_local_lanes"] == 1, "wrong local lanes")
            require(timing.validate_block_size(payload, requested) == captured["actual_block_size"] == row["actual_block"] == actual,
                    "wrong actual block")
            require(payload["dispatch"] == [dispatch, 1, 1] and row["dispatch"] == dispatch, "wrong dispatch geometry")
            metrics = timing.validate(payload, op, dims, 3, 8, block)
            require(set(metrics) == set(METRICS) and metrics == captured["metrics"] == row["metrics"], "timing recomputation differs")
            original_output = Path(captured["command"][-1])
            tensor_root = Path(cohort["temporary_tensor_root"])
            require(original_output.is_relative_to(tensor_root), "output outside declared tensor root")
            require(captured["command"] == [binary, "llm", op, ",".join(map(str, dims)), *map(str, block), "3", "20", "10",
                                            str(original_output)], "native command differs")
            tensor_base = base / "tensors" / label if extracted else tensor_root
            output = contained(tensor_base, original_output.relative_to(tensor_root).as_posix())
            original_paths = [original_output, *(Path(str(original_output) + f".input{i}.f32") for i in range(3))]
            require(set(captured["tensor_receipts"]) == set(map(str, original_paths)), "tensor receipt set differs")
            for original in original_paths:
                path = contained(tensor_base, original.relative_to(tensor_root).as_posix())
                require(fingerprint(path) == captured["tensor_receipts"][str(original)], "tensor hash/size mismatch")
            input_hash = [captured["tensor_receipts"][str(p)]["sha256"] for p in original_paths[1:]]
            require(inputs.setdefault((op, tuple(dims)), input_hash) == input_hash, "input changed between variants")
            output_hash = sha(output)
            require(outputs.setdefault(case, output_hash) == output_hash == row["output_sha256"], "output not bitwise equal")
            oracle = timing.validate_exports(output, op, dims)
            require(oracle == captured["independent_correctness"], "independent FP64 oracle summary differs")
            realization = payload["realization"]
            require(realization == captured["realization"], "row realization mismatch")
            structure = {}
            for key in STRUCTURE:
                found = re.findall(r"(?:^|;)\s*" + re.escape(key) + r"=([^;]+)", realization)
                require(len(found) == 1, "ambiguous/missing structural field")
                structure[key] = found[0]
            order = re.findall(r"; root order (\[[^]]*\])", realization)
            require(len(order) == 1, "ambiguous root order")
            structure["root_order"] = order[0]
            require(structures.setdefault(case, structure) == structure == row["structure"], "structural realization changed")
            checksum = payload["device_timing"]["throughput"][0]["dispatches"][0]["shader_checksum"]
            require(checksums.setdefault((case, actual), checksum) == checksum == row["checksum"], "shader identity changed")
            groups = (dispatch + actual - 1) // actual
            require(row["threadgroups"] == groups, "threadgroup count differs")
            useful_packets = (dispatch + 31) // 32
            allocated_packets = groups * (actual // 32)
            item = dict(case=case, position=position, requested_block=requested, actual_block=actual, dispatch=dispatch,
                        threadgroups=groups, useful_packets=useful_packets, allocated_packets=allocated_packets,
                        padding_packets=allocated_packets - useful_packets, padding_threads=groups * actual - dispatch,
                        metrics=metrics, structure=structure, checksum=checksum, oracle=oracle, output_sha256=output_hash)
            current.append(item)
            visits.append(item)
        ratios = {}
        for metric in METRICS:
            values = [v["metrics"][metric]["median"] for v in current]
            pairs = [values[pinned] / values[auto] for pinned, auto in ((1, 0), (2, 3), (5, 4), (6, 7))]
            ratios[metric] = dict(pairs=pairs, median=statistics.median(pairs), minimum=min(pairs), maximum=max(pairs))
        cases.append(dict(case=case, operation=op, dimensions=dims, attention_block=block, visits=current,
                          pinned32_over_automatic=ratios, output_bitwise_equal=True))
    require(result["finished"] >= last_finished, "result predates final visit")
    actual_commands = {p.relative_to(raw).as_posix() for p in raw.rglob("command.json")}
    require(actual_commands == expected_commands, "unexpected or missing command receipts")
    report = dict(status="passed", visits=48, tensor_receipts=192, metric_sample_values=48 * 3 * 6,
                  adjacent_pairs_per_case=4, python_tests=40, retained_failed_builds=1,
                  retained_host_build_wrapper_false_positives=1, full_build_process_exit_code=0, confirmation_exit_code=0, cases=cases,
                  maximum_abs_error=max(v["oracle"]["max_abs_error"] for v in visits),
                  boundary="Complete exported tensors checked against independent FP64 math. Instrumented dispatch, feedback-only command-buffer and host timings remain distinct. Four adjacent pinned32/auto pairs per case; no cross-framework ranking, live loader proof, or zero-overhead timing claim.")
    if args.output:
        with args.output.open("x") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(status="passed", visits=48, tensor_receipts=192, metric_sample_values=864)))
    else:
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
