#!/usr/bin/env python3
"""Print complete paired tables, retaining current timings beside legacy Error."""
import argparse
import collections
import json
from pathlib import Path


def tables(report):
    if not report.get("finished_unix") or not report.get("artifacts_unchanged"):
        raise ValueError("only a completed, unchanged experiment can supply published tables")
    groups = {(row["case"], row["route"]): row for row in report["summary"]}
    expected_rounds = report["options"]["rounds"]
    planned = {(row["case"], row["route"]) for row in report["planned"]}
    coverage = collections.defaultdict(collections.Counter)

    def value(case, route):
        key = "gpu_us" if "metal" in route else "batched_e2e_us"
        row = groups.get((case, route), {})
        if row.get("passed") != expected_rounds:
            return "Error" if (case, route) in planned else "—"
        samples = row[key]
        if len(samples) != expected_rounds or any(v <= 0 for v in samples):
            raise ValueError("invalid per-round samples")
        low, high = min(samples), max(samples)
        return f"{low:,.2f}–{high:,.2f}" if low != high else f"{low:,.2f}"

    lines = ["### Full baseline/current matrix", "",
             "Cells are the range of the two per-visit medians in **µs**. Metal uses GPU",
             "compute-encoder intervals; SIMD uses batched E2E host wall. `Error` never",
             "removes the other implementation's successful measurement. Failure reasons",
             "and timeouts are in the raw visit records; no Error receives a speedup ratio.", "",
             "| Operation / M×N×K | Legacy Metal | Current Metal/TIRx | Legacy SIMD | Current XIR/SIMD |",
             "|---|---:|---:|---:|---:|"]
    for case in report["cases"]:
        label = case["operation"] + " " + "×".join(map(str, case["shape"]))
        lines.append("| " + " | ".join([label] + [value(case["id"], route) for route in ["legacy-metal", "metal", "legacy-simd", "simd"]]) + " |")
    lines += ["", "### Same pipelined source: additional Metal routes", "",
              "These rows retain the 16×16×32 K-pipeline capture. Native MPP does not",
              "support this source structure; its `Error` cells are not replaced by the",
              "different full-K program below. An MPP-enabled TIRx policy can retain a",
              "non-MPP fallback for an ineligible dtype; inspect `realization` in the log.", "",
              "| Operation / M×N×K | Native MPP | TIRx, MPP-enabled policy |",
              "|---|---:|---:|"]
    for case in report["cases"]:
        if case["operation"].startswith("gemm"):
            lines.append("| " + " | ".join([case["operation"] + " " + "×".join(map(str, case["shape"])), value(case["id"], "metal-native"), value(case["id"], "metal-tirx-mpp")]) + " |")
    lines += ["", "### Same full-K FP32 source: three Metal routes", "",
              "This is a **separately staged 32×32 full-K capture**, not a planner rewrite",
              "of the preceding pipeline. All three columns use the same capture and inputs.", "",
              "| M×N×K | TIRx | Native MPP | TIRx/MPP |", "|---|---:|---:|---:|"]
    for case in report["cases"]:
        if case["operation"] == "gemm":
            lines.append("| " + " | ".join(["×".join(map(str, case["shape"]))] + [value(case["id"], route) for route in ["metal-direct", "metal-native-direct", "metal-tirx-mpp-direct"]]) + " |")
    for case, route in planned:
        row = groups.get((case, route), {})
        coverage[route]["passed" if row.get("passed") == expected_rounds else "error"] += 1
    lines += ["", "### Route coverage", "", "| Requested route | Both visits pass | Error |", "|---|---:|---:|"]
    for route, count in sorted(coverage.items()):
        lines.append(f"| {route} | {count['passed']} | {count['error']} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(tables(json.loads(args.results.read_text())), end="")


if __name__ == "__main__":
    main()
