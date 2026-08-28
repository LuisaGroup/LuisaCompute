#!/usr/bin/env python3

"""Summarize matched old-Metal and Metal4 iOS benchmark reports."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _load_reports(paths: list[Path], expected_backend: str) -> list[dict[str, Any]]:
    reports = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("backend") != expected_backend:
            raise ValueError(
                f"{path}: expected backend {expected_backend!r}, "
                f"got {report.get('backend')!r}")
        if not report.get("success"):
            raise ValueError(f"{path}: benchmark did not succeed")
        reports.append(report)
    return reports


def _assert_matched(reports: list[dict[str, Any]]) -> None:
    reference = reports[0]
    matched_fields = ("device", "system", "source", "resolution", "spp",
                      "max_spp_per_dispatch")
    for report in reports[1:]:
        for field in matched_fields:
            if report.get(field) != reference.get(field):
                raise ValueError(
                    f"reports are not matched: field {field!r} differs "
                    f"({reference.get(field)!r} vs {report.get(field)!r})")


def _values(reports: list[dict[str, Any]], field: str,
            *, warm_only: bool) -> list[float]:
    values = []
    for report in reports:
        runs = report["runs"][1:] if warm_only else report["runs"][:1]
        values.extend(float(run[field]) for run in runs)
    if not values:
        raise ValueError("each report must contain at least two runs")
    return values


def _median(reports: list[dict[str, Any]], field: str,
            *, warm_only: bool) -> float:
    return statistics.median(_values(reports, field, warm_only=warm_only))


def _metrics(reports: list[dict[str, Any]]) -> dict[str, float]:
    render_ms = _median(reports, "render_ms", warm_only=True)
    process_first = _values(
        reports, "shader_compile_ms", warm_only=False)
    spp = float(reports[0]["spp"])
    return {
        "process_first_compile_min_ms": min(process_first),
        "process_first_compile_max_ms": max(process_first),
        "warm_compile_ms": _median(
            reports, "shader_compile_ms", warm_only=True),
        "warm_render_ms": render_ms,
        "warm_spp_per_second": spp * 1000.0 / render_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metal", type=Path, action="append", required=True,
                        help="old-Metal JSON report; may be repeated")
    parser.add_argument("--metal4", type=Path, action="append", required=True,
                        help="Metal4 JSON report; may be repeated")
    args = parser.parse_args()

    metal = _load_reports(args.metal, "metal")
    metal4 = _load_reports(args.metal4, "metal4")
    reports = metal + metal4
    _assert_matched(reports)

    metal_metrics = _metrics(metal)
    metal4_metrics = _metrics(metal4)
    print("| backend | process-first compile range (ms) | "
          "in-process cache hit (ms) | stable render (ms) | stable spp/s |")
    print("|---|---:|---:|---:|---:|")
    for name, metrics in (("metal", metal_metrics),
                          ("metal4", metal4_metrics)):
        print(
            f"| {name} | {metrics['process_first_compile_min_ms']:.3f}.."
            f"{metrics['process_first_compile_max_ms']:.3f} | "
            f"{metrics['warm_compile_ms']:.3f} | "
            f"{metrics['warm_render_ms']:.3f} | "
            f"{metrics['warm_spp_per_second']:.3f} |")

    render_delta = (
        metal4_metrics["warm_render_ms"] /
        metal_metrics["warm_render_ms"] - 1.0) * 100.0
    print()
    print(f"Metal4 stable-render time delta: {render_delta:+.3f}%")

    metal_run = metal[-1]["runs"][-1]
    metal4_run = metal4[-1]["runs"][-1]
    print(
        "Image evidence: "
        f"nonblack={metal_run.get('nonblack_pixels')}/"
        f"{metal4_run.get('nonblack_pixels')}, "
        f"mean-luma delta="
        f"{float(metal4_run.get('mean_luma', 0.0)) - float(metal_run.get('mean_luma', 0.0)):+.9g}, "
        f"bit-exact={metal_run.get('pixel_sha256') == metal4_run.get('pixel_sha256')}")


if __name__ == "__main__":
    main()
