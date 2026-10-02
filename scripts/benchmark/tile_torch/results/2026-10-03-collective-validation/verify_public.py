"""Verify published byte receipts and recompute reported ratios; no GPU or NumPy."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    receipt = read(ROOT / "receipts.json")
    for name, item in receipt["files"].items():
        path = ROOT / name
        require(path.is_file() and path.parent == ROOT, "Missing or escaped public file")
        data = path.read_bytes()
        require(len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"], "Public bytes changed: " + name)
    counts = {}
    for kind in ("cost", "streaming"):
        path = ROOT / (kind + ".json")
        if not path.exists():
            continue
        report = read(path)
        require(report["status"] == "completed_validated", "Unvalidated report")
        counts[kind] = len(report["cases"])
        for row in report["cases"]:
            for phase in row["cohorts"].values():
                require(phase["status"] == "validated", "Invalid phase")
                for route in phase["routes"].values():
                    if "event_us" not in route:
                        require(route["status"] == "not_requested", "Unexpected missing timing")
                        continue
                    samples = route["event_us"]["samples"]
                    require(len(samples) == 7 and all(math.isfinite(x) and x > 0 for x in samples), "Invalid raw samples")
                    require(statistics.median(samples) == route["event_us"]["p50"], "Median changed")
                require(phase["fixture_sha256"] == row["cohorts"]["default"]["fixture_sha256"], "Matched fixture changed")
            for name, comparison in row["comparisons"].items():
                require(comparison["status"] == "valid_matched_pair", "Invalid comparison")
                base = "recheck" if name == "repeat_vs_recheck" else "default"
                candidate = "cost-repeat" if name == "repeat_vs_recheck" else name
                a, b = (row["cohorts"][phase] for phase in (base, candidate))
                ratio = b["routes"]["native"]["event_us"]["p50"] / a["routes"]["native"]["event_us"]["p50"]
                require(ratio == comparison["candidate_over_default"], "Ratio changed")
                for source, value in a["source_files"].items():
                    require(value["normalized_source_sha256"] == b["source_files"][source]["original_source_sha256"], "Source proof changed")
            if kind == "cost":
                require(row["repeated_selection_identical"], "Repeat selection changed")
        if kind == "streaming":
            require(sum(row["historical_default_preservation"]["status"] == "byte_identical_default_preserved" for row in report["cases"]) == 3,
                    "Missing historical default source proof")
    cost = read(ROOT / "cost.json")
    stats = read(ROOT / "cost-statistics.json")
    for label, selected in (("all_12", None), ("selected_4", True), ("retained_default_8", False)):
        rows = [row for row in cost["cases"] if selected is None or row["comparisons"]["cost"]["specialization_selected"] is selected]
        require(len(rows) == stats[label]["cases"], "Incorrect subgroup count")
        for key, pair in (("first_cost_over_default", "cost"), ("repeat_cost_over_recheck", "repeat_vs_recheck")):
            mean = math.exp(sum(math.log(row["comparisons"][pair]["candidate_over_default"]) for row in rows) / len(rows))
            require(mean == stats[label][key], "Geometric mean changed")
    print(json.dumps(dict(status="passed", public_files=len(receipt["files"]), inventories=counts,
                          scope="Public byte receipts and numerical/source-proof consistency only; original runtime checks are retained evidence, not rerun.")))


if __name__ == "__main__":
    main()
