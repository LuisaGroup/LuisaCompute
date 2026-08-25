#!/usr/bin/env python3
"""Build and run the standalone Luisa SIMD/ISPC comparison suite.

This driver intentionally does not invoke CMake or modify a CMake cache. It
uses compile flags and libraries from an existing LuisaCompute build, while
the explicitly supplied ISPC compiler writes objects only to a temporary
directory.
"""

from __future__ import annotations

import argparse
import array
import dataclasses
import datetime as dt
import json
import math
import os
import pathlib
import platform
import shlex
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Sequence


SCRIPT_DIRECTORY = pathlib.Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[2]
RESULT_PREFIX = "simd_ispc_suite,"
EXACT_WORKLOADS = frozenset(
    {"mandelbrot", "masked_stream", "aos_to_soa", "gemm"}
)
ALL_WORKLOADS = (
    "mandelbrot",
    "masked_stream",
    "aos_to_soa",
    "gemm",
    "path_trace",
)
ISPC_TARGETS = {
    "avx2-i32x4": ("avx2_w4", 4),
    "avx2-i32x8": ("avx2_w8", 8),
    "avx512skx-x4": ("avx512_w4", 4),
    "avx512skx-x8": ("avx512_w8", 8),
    "avx512skx-x16": ("avx512_w16", 16),
}
LUISA_WIDTHS = (1, 2, 4, 8, 16)


@dataclasses.dataclass(frozen=True)
class Variant:
    name: str
    implementation: str
    width: int
    backend: str
    target: str | None = None


def comma_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def integer_list(text: str) -> list[int]:
    try:
        return [int(item) for item in comma_list(text)]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_cpu_set(text: str) -> set[int]:
    result: set[int] = set()
    try:
        for item in comma_list(text):
            if "-" in item:
                begin_text, end_text = item.split("-", 1)
                begin = int(begin_text)
                end = int(end_text)
                if begin > end:
                    raise ValueError(f"invalid descending CPU range '{item}'")
                result.update(range(begin, end + 1))
            else:
                result.add(int(item))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not result or min(result) < 0:
        raise argparse.ArgumentTypeError("CPU set must contain non-negative IDs")
    return result


def display_command(command: Sequence[str]) -> str:
    return shlex.join(str(item) for item in command)


def run_checked(
    command: Sequence[str],
    *,
    cwd: pathlib.Path,
    timeout: float | None = None,
    environment: dict[str, str] | None = None,
    affinity: set[int] | None = None,
    quiet: bool = False,
) -> subprocess.CompletedProcess[str]:
    if not quiet:
        print(f"+ {display_command(command)}", flush=True)

    def set_affinity() -> None:
        if affinity is not None:
            os.sched_setaffinity(0, affinity)

    result = subprocess.run(
        [str(item) for item in command],
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
        preexec_fn=set_affinity if affinity is not None else None,
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, file=sys.stderr, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        raise RuntimeError(
            f"command exited with status {result.returncode}: "
            f"{display_command(command)}"
        )
    return result


def capture_version(command: Sequence[str], cwd: pathlib.Path) -> str:
    result = run_checked(command, cwd=cwd, quiet=True)
    output = (result.stdout + result.stderr).strip()
    return output.splitlines()[0] if output else "unknown"


def load_compile_command(build_directory: pathlib.Path) -> list[str]:
    database_path = build_directory / "compile_commands.json"
    if not database_path.is_file():
        raise RuntimeError(f"missing compilation database: {database_path}")
    database = json.loads(database_path.read_text(encoding="utf-8"))
    candidates = [
        entry
        for entry in database
        if pathlib.Path(entry.get("file", "")).name == "benchmark_simd_gemm.cpp"
    ]
    if not candidates:
        raise RuntimeError(
            "compile_commands.json has no benchmark_simd_gemm.cpp entry; "
            "build the benchmark_simd_gemm target first"
        )
    entry = candidates[0]
    if "arguments" in entry:
        return [str(item) for item in entry["arguments"]]
    return shlex.split(entry["command"])


def reusable_compile_arguments(command: Sequence[str]) -> tuple[str, list[str]]:
    if not command:
        raise RuntimeError("empty C++ compile command")
    compiler = command[0]
    result: list[str] = []
    index = 1
    options_with_value = {"-o", "-MF", "-MT", "-MQ"}
    options_without_value = {"-c", "-MD", "-MMD", "-MP"}
    while index < len(command):
        argument = command[index]
        if argument in options_with_value:
            index += 2
            continue
        if argument in options_without_value:
            index += 1
            continue
        if argument.endswith("benchmark_simd_gemm.cpp"):
            index += 1
            continue
        result.append(argument)
        index += 1
    return compiler, result


def require_file(path: pathlib.Path, description: str) -> pathlib.Path:
    if not path.is_file():
        raise RuntimeError(f"missing {description}: {path}")
    return path


def shared_library(binary_directory: pathlib.Path, stem: str) -> pathlib.Path:
    exact = binary_directory / f"{stem}.so"
    if exact.is_file():
        return exact
    candidates = sorted(binary_directory.glob(f"{stem}.so.*"))
    if not candidates:
        raise RuntimeError(f"missing shared library {stem}.so in {binary_directory}")
    return candidates[0]


def build_runners(
    args: argparse.Namespace,
    temporary_directory: pathlib.Path,
    command_log: list[list[str]],
) -> tuple[pathlib.Path, pathlib.Path, str]:
    build_directory = args.build_dir.resolve()
    binary_directory = (args.runtime_dir or build_directory / "bin").resolve()
    require_file(binary_directory / "libluisa-backend-simd.so", "SIMD backend")
    if args.include_fallback:
        require_file(
            binary_directory / "libluisa-backend-fallback.so", "fallback backend"
        )

    original_compile_command = load_compile_command(build_directory)
    compiler, compile_arguments = reusable_compile_arguments(original_compile_command)
    if args.cxx:
        compiler = str(args.cxx)
    compiler_version = capture_version([compiler, "--version"], REPOSITORY_ROOT)

    objects: list[pathlib.Path] = []
    for target, (suffix, _) in ISPC_TARGETS.items():
        output = temporary_directory / f"benchmark_{suffix}.o"
        command = [
            str(args.ispc),
            str(SCRIPT_DIRECTORY / "benchmark.ispc"),
            "-o",
            str(output),
            "-O2",
            "--pic",
            "--arch=x86-64",
            f"--target={target}",
            "--math-lib=default",
            "--opt=disable-fma",
            "--wno-perf",
            f"-DLUISA_ISPC_SUFFIX=_{suffix}",
        ]
        if args.cpu:
            command.append(f"--cpu={args.cpu}")
        command_log.append(command)
        run_checked(command, cwd=REPOSITORY_ROOT)
        objects.append(output)

    luisa_runner = temporary_directory / "luisa_runner"
    luisa_object = temporary_directory / "luisa_runner.o"
    command = [
        compiler,
        *compile_arguments,
        "-c",
        str(SCRIPT_DIRECTORY / "luisa_runner.cpp"),
        "-o",
        str(luisa_object),
    ]
    command_log.append(command)
    run_checked(command, cwd=REPOSITORY_ROOT)

    link_libraries = [
        shared_library(binary_directory, name)
        for name in (
            "libluisa-osl",
            "libluisa-coro",
            "libluisa-xir",
            "libluisa-dsl",
            "libluisa-runtime",
            "libluisa-ast",
            "libluisa-core",
            "libglslang",
        )
    ]
    command = [
        compiler,
        "-O3",
        "-DNDEBUG",
        str(luisa_object),
        *(str(path) for path in link_libraries),
        "-ldl",
        "-pthread",
        f"-Wl,-rpath,{binary_directory}",
        "-o",
        str(luisa_runner),
    ]
    command_log.append(command)
    run_checked(command, cwd=REPOSITORY_ROOT)

    thread_pool_directory = REPOSITORY_ROOT / "src/backends/simd/runtime"
    thread_pool_source = require_file(
        thread_pool_directory / "simd_thread_pool.cpp", "SIMD thread-pool source"
    )
    require_file(thread_pool_directory / "simd_thread_pool.h", "SIMD thread-pool header")
    ispc_runner = temporary_directory / "ispc_runner"
    command = [
        compiler,
        *compile_arguments,
        f"-I{thread_pool_directory}",
        str(SCRIPT_DIRECTORY / "ispc_runner.cpp"),
        str(thread_pool_source),
        *(str(path) for path in objects),
        str(shared_library(binary_directory, "libluisa-core")),
        "-pthread",
        f"-Wl,-rpath,{binary_directory}",
        "-o",
        str(ispc_runner),
    ]
    command_log.append(command)
    run_checked(command, cwd=REPOSITORY_ROOT)
    return luisa_runner, ispc_runner, compiler_version


def parse_result(output: str) -> dict[str, Any]:
    lines = [line for line in output.splitlines() if line.startswith(RESULT_PREFIX)]
    if len(lines) != 1:
        raise RuntimeError(
            f"expected one '{RESULT_PREFIX}' record, found {len(lines)}"
        )
    fields: dict[str, Any] = {}
    for field in lines[0].split(",")[1:]:
        key, value = field.split("=", 1)
        fields[key] = value
    for key in ("width", "workers", "items", "dispatches"):
        fields[key] = int(fields[key])
    for key in ("median_seconds", "median_rate"):
        fields[key] = float(fields[key])
    fields["samples_seconds"] = [
        float(item) for item in fields["samples_seconds"].split(";")
    ]
    return fields


def balanced_order(variants: Sequence[Variant], round_index: int) -> list[Variant]:
    if not variants:
        return []
    cycle, shift = divmod(round_index, len(variants))
    if cycle % 2 == 1:
        # Offset the reversed cycle once so the last rotation of the forward
        # cycle cannot cancel the reversal. In particular, two variants must
        # alternate A/B, B/A instead of repeating A/B every round.
        shift = (shift + 1) % len(variants)
    order = list(variants[shift:]) + list(variants[:shift])
    if cycle % 2 == 1:
        order.reverse()
    return order


def benchmark_command(
    variant: Variant,
    workload: str,
    dump_path: pathlib.Path | None,
    *,
    luisa_runner: pathlib.Path,
    ispc_runner: pathlib.Path,
    runtime_directory: pathlib.Path,
    workers: int,
) -> list[str]:
    if variant.implementation == "luisa":
        command = [
            str(luisa_runner),
            str(runtime_directory),
            variant.backend,
            str(variant.width),
            str(workers),
            workload,
        ]
    else:
        command = [
            str(ispc_runner),
            str(workers),
            str(variant.target),
            workload,
        ]
    if dump_path is not None:
        command.append(str(dump_path))
    return command


def geometric_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def t_critical_95(sample_count: int) -> float:
    table = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
        13: 2.179,
        14: 2.160,
        15: 2.145,
        16: 2.131,
        17: 2.120,
        18: 2.110,
        19: 2.101,
        20: 2.093,
        21: 2.086,
        22: 2.080,
        23: 2.074,
        24: 2.069,
        25: 2.064,
        26: 2.060,
        27: 2.056,
        28: 2.052,
        29: 2.048,
        30: 2.045,
    }
    return table.get(sample_count, 1.960)


def paired_ratio_summary(
    numerator: Sequence[float], denominator: Sequence[float]
) -> dict[str, Any]:
    if len(numerator) != len(denominator) or not numerator:
        raise ValueError("paired ratio inputs must have equal non-zero lengths")
    ratios = [a / b for a, b in zip(numerator, denominator, strict=True)]
    logs = [math.log(value) for value in ratios]
    center = statistics.fmean(logs)
    if len(logs) == 1:
        low = high = math.exp(center)
    else:
        radius = (
            t_critical_95(len(logs))
            * statistics.stdev(logs)
            / math.sqrt(len(logs))
        )
        low = math.exp(center - radius)
        high = math.exp(center + radius)
    return {
        "geomean": math.exp(center),
        "ci95_low": low,
        "ci95_high": high,
        "ratios": ratios,
    }


def load_float_dump(path: pathlib.Path) -> array.array[float]:
    values = array.array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, path.stat().st_size // values.itemsize)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def compare_path_dumps(
    reference_path: pathlib.Path,
    candidate_path: pathlib.Path,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    reference = load_float_dump(reference_path)
    candidate = load_float_dump(candidate_path)
    if len(reference) != len(candidate):
        raise RuntimeError(
            f"path_trace output size mismatch: {len(reference)} != {len(candidate)}"
        )
    maximum_absolute = 0.0
    maximum_relative = 0.0
    violation_count = 0
    for expected, observed in zip(reference, candidate, strict=True):
        absolute = abs(observed - expected)
        relative = absolute / max(abs(expected), 1.0e-30)
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
        if absolute > absolute_tolerance + relative_tolerance * abs(expected):
            violation_count += 1
    return {
        "element_count": len(reference),
        "maximum_absolute_error": maximum_absolute,
        "maximum_relative_error": maximum_relative,
        "violation_count": violation_count,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
    }


def make_variants(args: argparse.Namespace) -> list[Variant]:
    variants: list[Variant] = []
    if args.include_fallback:
        variants.append(Variant("fallback", "luisa", 0, "fallback"))
    variants.extend(
        Variant(f"simd-w{width}", "luisa", width, "simd")
        for width in args.luisa_widths
    )
    variants.extend(
        Variant(f"ispc-{target}", "ispc", ISPC_TARGETS[target][1], "ispc", target)
        for target in args.ispc_targets
    )
    return variants


def validate_arguments(args: argparse.Namespace) -> None:
    args.build_dir = args.build_dir.resolve()
    args.ispc = args.ispc.resolve()
    if args.runtime_dir:
        args.runtime_dir = args.runtime_dir.resolve()
    require_file(args.ispc, "ISPC executable")
    if args.process_rounds < 1:
        raise RuntimeError("--process-rounds must be positive")
    if args.workers < 1:
        raise RuntimeError("--workers must be positive")
    invalid_widths = sorted(set(args.luisa_widths) - set(LUISA_WIDTHS))
    if invalid_widths:
        raise RuntimeError(f"unsupported Luisa widths: {invalid_widths}")
    invalid_targets = sorted(set(args.ispc_targets) - set(ISPC_TARGETS))
    if invalid_targets:
        raise RuntimeError(f"unsupported ISPC targets: {invalid_targets}")
    invalid_workloads = sorted(set(args.workloads) - set(ALL_WORKLOADS))
    if invalid_workloads:
        raise RuntimeError(f"unsupported workloads: {invalid_workloads}")
    if not args.luisa_widths and not args.ispc_targets and not args.include_fallback:
        raise RuntimeError("no benchmark variants selected")
    if args.cpus is not None and args.workers > len(args.cpus):
        print(
            f"warning: {args.workers} workers share {len(args.cpus)} pinned CPUs",
            file=sys.stderr,
        )


def git_metadata() -> dict[str, Any]:
    commit = capture_version(["git", "rev-parse", "HEAD"], REPOSITORY_ROOT)
    status = run_checked(
        ["git", "status", "--short"], cwd=REPOSITORY_ROOT, quiet=True
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status": status}


def benchmark_suite(
    args: argparse.Namespace,
    temporary_directory: pathlib.Path,
    luisa_runner: pathlib.Path,
    ispc_runner: pathlib.Path,
    variants: Sequence[Variant],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    runtime_directory = (args.runtime_dir or args.build_dir / "bin").resolve()
    environment = os.environ.copy()
    loader_paths = [str(runtime_directory), *(str(path) for path in args.loader_path)]
    previous_loader_path = environment.get("LD_LIBRARY_PATH")
    if previous_loader_path:
        loader_paths.append(previous_loader_path)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(loader_paths)

    validation_records: list[dict[str, Any]] = []
    dump_paths: dict[tuple[str, str], pathlib.Path] = {}
    for workload in args.workloads:
        print(f"\n== validate {workload} ==", flush=True)
        for variant in variants:
            dump_path = temporary_directory / f"{workload}-{variant.name}.bin"
            dump_paths[(workload, variant.name)] = dump_path
            command = benchmark_command(
                variant,
                workload,
                dump_path,
                luisa_runner=luisa_runner,
                ispc_runner=ispc_runner,
                runtime_directory=runtime_directory,
                workers=args.workers,
            )
            result = run_checked(
                command,
                cwd=REPOSITORY_ROOT,
                timeout=args.timeout,
                environment=environment,
                affinity=args.cpus,
            )
            parsed = parse_result(result.stdout)
            parsed.update({"variant": variant.name, "command": command})
            validation_records.append(parsed)
            print(f"  {variant.name}: checksum={parsed['checksum']}", flush=True)

    validation: dict[str, Any] = {}
    for workload in args.workloads:
        workload_records = [
            record
            for record in validation_records
            if record["workload"] == workload
        ]
        checksums_by_variant = {
            record["variant"]: record["checksum"] for record in workload_records
        }
        if workload in EXACT_WORKLOADS:
            unique_checksums = set(checksums_by_variant.values())
            if len(unique_checksums) != 1:
                raise RuntimeError(
                    f"{workload} is not bit-exact across variants: "
                    f"{checksums_by_variant}"
                )
            validation[workload] = {
                "mode": "bit_exact",
                "checksum": next(iter(unique_checksums)),
                "variant_count": len(checksums_by_variant),
            }
            continue

        reference_name = "fallback" if args.include_fallback else variants[0].name
        reference_path = dump_paths[(workload, reference_name)]
        comparisons: dict[str, Any] = {}
        for variant in variants:
            comparison = compare_path_dumps(
                reference_path,
                dump_paths[(workload, variant.name)],
                args.path_absolute_tolerance,
                args.path_relative_tolerance,
            )
            comparisons[variant.name] = comparison
            if comparison["violation_count"] != 0:
                raise RuntimeError(
                    f"{workload} output from {variant.name} exceeded tolerance: "
                    f"{comparison}"
                )
        validation[workload] = {
            "mode": "absolute_plus_relative",
            "reference": reference_name,
            "comparisons": comparisons,
        }

    records: list[dict[str, Any]] = []
    for workload in args.workloads:
        print(f"\n== measure {workload} ==", flush=True)
        for round_index in range(args.process_rounds):
            order = balanced_order(variants, round_index)
            print(
                f"process round {round_index + 1}/{args.process_rounds}: "
                + ", ".join(variant.name for variant in order),
                flush=True,
            )
            for variant in order:
                command = benchmark_command(
                    variant,
                    workload,
                    None,
                    luisa_runner=luisa_runner,
                    ispc_runner=ispc_runner,
                    runtime_directory=runtime_directory,
                    workers=args.workers,
                )
                result = run_checked(
                    command,
                    cwd=REPOSITORY_ROOT,
                    timeout=args.timeout,
                    environment=environment,
                    affinity=args.cpus,
                )
                parsed = parse_result(result.stdout)
                parsed.update(
                    {
                        "variant": variant.name,
                        "process_round": round_index,
                        "command": command,
                    }
                )
                records.append(parsed)
                print(
                    f"  {variant.name}: "
                    f"{parsed['median_rate']:.3f} {parsed['rate_unit']}",
                    flush=True,
                )
    for workload in args.workloads:
        expected_checksums = {
            record["variant"]: record["checksum"]
            for record in validation_records
            if record["workload"] == workload
        }
        for record in records:
            if record["workload"] != workload:
                continue
            expected = expected_checksums[record["variant"]]
            if record["checksum"] != expected:
                raise RuntimeError(
                    f"{workload} output from {record['variant']} changed in "
                    f"process round {record['process_round']}: "
                    f"{record['checksum']} != {expected}"
                )
    return records, validation_records, validation


def summarize(
    records: Sequence[dict[str, Any]], variants: Sequence[Variant], workloads: Sequence[str]
) -> dict[str, Any]:
    summary: dict[str, Any] = {"workloads": {}, "ispc_over_luisa": {}}
    variant_by_name = {variant.name: variant for variant in variants}
    for workload in workloads:
        values: dict[str, list[float]] = {}
        for variant in variants:
            selected = sorted(
                (
                    record
                    for record in records
                    if record["workload"] == workload
                    and record["variant"] == variant.name
                ),
                key=lambda record: record["process_round"],
            )
            values[variant.name] = [
                record["median_rate"] for record in selected
            ]
        workload_summary: dict[str, Any] = {}
        fallback_values = values.get("fallback")
        for name, samples in values.items():
            row: dict[str, Any] = {
                "implementation": variant_by_name[name].implementation,
                "width": variant_by_name[name].width,
                "rate_unit": selected[0]["rate_unit"],
                "process_median_rate": statistics.median(samples),
                "process_geomean_rate": geometric_mean(samples),
                "process_rate_values": samples,
            }
            if fallback_values is not None:
                row["speedup_over_fallback"] = paired_ratio_summary(
                    samples, fallback_values
                )
            workload_summary[name] = row
        summary["workloads"][workload] = workload_summary

        comparisons: dict[str, Any] = {}
        for ispc_target, (_, width) in ISPC_TARGETS.items():
            ispc_name = f"ispc-{ispc_target}"
            luisa_name = f"simd-w{width}"
            if ispc_name in values and luisa_name in values:
                comparisons[f"{ispc_name}/{luisa_name}"] = paired_ratio_summary(
                    values[ispc_name], values[luisa_name]
                )
        summary["ispc_over_luisa"][workload] = comparisons
    return summary


def print_summary(summary: dict[str, Any], workloads: Sequence[str]) -> None:
    for workload in workloads:
        print(f"\n### {workload}\n")
        units = {
            row["rate_unit"]
            for row in summary["workloads"][workload].values()
        }
        if len(units) != 1:
            raise RuntimeError(f"mixed rate units for {workload}: {sorted(units)}")
        unit = next(iter(units))
        print(f"| Variant | Width | {unit} | vs fallback | 95% CI |")
        print("|---|---:|---:|---:|---:|")
        for name, row in summary["workloads"][workload].items():
            speedup = row.get("speedup_over_fallback")
            if speedup is None:
                speedup_text = ci_text = "n/a"
            else:
                speedup_text = f"{speedup['geomean']:.3f}x"
                ci_text = (
                    f"[{speedup['ci95_low']:.3f}, {speedup['ci95_high']:.3f}]"
                )
            print(
                f"| {name} | {row['width']} | "
                f"{row['process_median_rate']:.3f} | "
                f"{speedup_text} | {ci_text} |"
            )
        comparisons = summary["ispc_over_luisa"][workload]
        if comparisons:
            print("\nISPC / same-width Luisa SIMD:\n")
            for name, value in comparisons.items():
                print(
                    f"- {name}: {value['geomean']:.3f}x "
                    f"(95% CI [{value['ci95_low']:.3f}, "
                    f"{value['ci95_high']:.3f}])"
                )


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and measure standalone Luisa SIMD/ISPC controls without "
            "invoking or modifying the project CMake build graph."
        )
    )
    parser.add_argument(
        "--build-dir",
        type=pathlib.Path,
        required=True,
        help="existing LuisaCompute build containing compile_commands.json",
    )
    parser.add_argument(
        "--runtime-dir",
        type=pathlib.Path,
        help="directory containing backend and shared libraries (default: BUILD/bin)",
    )
    parser.add_argument(
        "--ispc",
        type=pathlib.Path,
        required=True,
        help="explicit ISPC executable; no environment or CMake variable is read",
    )
    parser.add_argument("--cxx", type=pathlib.Path, help="override the build's C++ compiler")
    parser.add_argument("--cpu", help="ISPC --cpu value, for example znver5")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--cpus",
        type=parse_cpu_set,
        help="pin every benchmark process, for example 0-15 or 0,2,4,6",
    )
    parser.add_argument(
        "--workloads", type=comma_list, default=list(ALL_WORKLOADS)
    )
    parser.add_argument(
        "--luisa-widths", type=integer_list, default=list(LUISA_WIDTHS)
    )
    parser.add_argument(
        "--ispc-targets", type=comma_list, default=list(ISPC_TARGETS)
    )
    parser.add_argument(
        "--include-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--process-rounds", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--loader-path",
        action="append",
        type=pathlib.Path,
        default=[],
        help="additional dynamic-loader directory; may be repeated",
    )
    parser.add_argument("--path-absolute-tolerance", type=float, default=2.0e-5)
    parser.add_argument("--path-relative-tolerance", type=float, default=2.0e-5)
    parser.add_argument("--output", type=pathlib.Path, help="write raw JSON here")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    return parser


def main() -> int:
    parser = make_argument_parser()
    args = parser.parse_args()
    try:
        validate_arguments(args)
        ispc_version = capture_version([str(args.ispc), "--version"], REPOSITORY_ROOT)
        variants = make_variants(args)
        command_log: list[list[str]] = []
        temporary_context: tempfile.TemporaryDirectory[str] | None = None
        if args.keep_temp:
            temporary_directory = pathlib.Path(
                tempfile.mkdtemp(prefix="luisa-simd-ispc-")
            )
        else:
            temporary_context = tempfile.TemporaryDirectory(
                prefix="luisa-simd-ispc-"
            )
            temporary_directory = pathlib.Path(temporary_context.name)
        print(f"temporary build: {temporary_directory}", flush=True)
        luisa_runner, ispc_runner, compiler_version = build_runners(
            args, temporary_directory, command_log
        )
        if args.build_only:
            print(f"built {luisa_runner} and {ispc_runner}")
            if temporary_context is not None:
                temporary_context.cleanup()
            return 0

        records, validation_records, validation = benchmark_suite(
            args, temporary_directory, luisa_runner, ispc_runner, variants
        )
        summary = summarize(records, variants, args.workloads)
        result = {
            "schema": 1,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "repository": git_metadata(),
            "host": {
                "hostname": platform.node(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "affinity": sorted(args.cpus)
                if args.cpus is not None
                else (
                    sorted(os.sched_getaffinity(0))
                    if hasattr(os, "sched_getaffinity")
                    else None
                ),
            },
            "configuration": {
                "build_directory": str(args.build_dir),
                "runtime_directory": str(
                    (args.runtime_dir or args.build_dir / "bin").resolve()
                ),
                "workers": args.workers,
                "process_rounds": args.process_rounds,
                "workloads": args.workloads,
                "luisa_widths": args.luisa_widths,
                "ispc_targets": args.ispc_targets,
                "include_fallback": args.include_fallback,
                "ispc_cpu": args.cpu,
                "math_mode": "precise",
                "cmake_invoked": False,
                "cmake_cache_modified": False,
            },
            "tools": {
                "ispc": str(args.ispc),
                "ispc_version": ispc_version,
                "cxx_version": compiler_version,
            },
            "build_commands": command_log,
            "validation": validation,
            "validation_records": validation_records,
            "records": records,
            "summary": summary,
        }
        print_summary(summary, args.workloads)
        if args.output:
            output_path = args.output.resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"\nraw result: {output_path}")
        if args.keep_temp:
            print(f"temporary build retained: {temporary_directory}")
        elif temporary_context is not None:
            temporary_context.cleanup()
        return 0
    except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
