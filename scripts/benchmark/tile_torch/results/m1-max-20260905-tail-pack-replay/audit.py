"""Audit fixed-plan tail-pack lowering A/B without benchmark statistics.

Reuse the preceding independent access/ownership oracle, not production cost
or statistics code. Keep all rounds, unchanged-source controls and regressions.
"""

from collections import defaultdict
import importlib.util
import json
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
ORACLE = HERE.parent / "m1-max-20260905-service-policy-validation/audit.py"
spec = importlib.util.spec_from_file_location("service_oracle", ORACLE)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def read(row, metric, provider="native"):
    kind, phase = metric.split("_")
    item = row[provider]
    return oracle.gpu(item, phase) if kind == "gpu" else median(item[phase + "_us"])


def main():
    report = json.loads((HERE / "results.json").read_text())
    metadata = report["metadata"]
    assert metadata["rounds"] == 4 and metadata["samples"] == 9
    assert metadata["sample_ms"] == 30 and metadata["warmup_ms"] == 200
    assert metadata["artifacts_unchanged"] is True
    frozen_file = HERE.parent / "m1-max-20260905-service-policy-plan/results.json"
    frozen = {r["name"]: r for r in json.loads(frozen_file.read_text())["results"]}
    assert len(report["results"]) == 96 and set(frozen) == oracle.CASES
    for source in metadata["source_reports"].values():
        assert source["sha256"] == oracle.digest(frozen_file)
    variants = metadata["native_variants"]
    assert variants["reference"]["sha256"] == "26f6c817d6aebba2c011b2bea3faacf786dbf677f37700261067d065d81b15f9"
    assert variants["reference"]["adjacent_tile_library_sha256"]["libluisa-tile-bridge-tirx.dylib"] == "886bda1ebbff189396d1a0b8cfc8f79a38396703e6a29796dc8ebf327fd902b1"
    assert variants["candidate"]["adjacent_tile_library_sha256"]["libluisa-tile-bridge-tirx.dylib"] != variants["reference"]["adjacent_tile_library_sha256"]["libluisa-tile-bridge-tirx.dylib"]
    grouped = defaultdict(dict)
    sources = defaultdict(set)
    orders = defaultdict(list)
    for row in report["results"]:
        oracle.validate(row, HERE, 9, "candidate")  # Both sides use the frozen service policy.
        name, variant, index = row["name"], row["variant"], row["round"]
        assert variant in ("reference", "candidate") and 0 <= index < 4
        assert variant not in grouped[name, index]
        grouped[name, index][variant] = row
        assert row["native"]["execution_plans"] == frozen[name]["native"]["execution_plans"]
        assert row["native"]["cache_reduction_inputs"] == frozen[name]["native"]["cache_reduction_inputs"]
        sources[name, variant].add(row["native_source_sha256"])
        orders[name, variant].append(row["implementation_order"][0])
        if variant == "reference":
            assert row["native_source_sha256"] == frozen[name]["native_source_sha256"]
    assert len(grouped) == 48 and all(len(pair) == 2 for pair in grouped.values())
    assert all(len(hashes) == 1 for hashes in sources.values())
    assert all(sorted(order) == ["native", "native", "torch", "torch"] for order in orders.values())
    summary = {"validated_outputs": 192, "artifacts": len(metadata["artifacts_sha256"]), "cases": []}
    for name in sorted(frozen):
        first = [next(iter(grouped[name, index])) for index in range(4)]
        assert sorted(first) == ["candidate", "candidate", "reference", "reference"]
        reference = next(iter(sources[name, "reference"]))
        candidate = next(iter(sources[name, "candidate"]))
        record = {"name": name, "same_source": reference == candidate,
                  "source_sha256": {"reference": reference, "candidate": candidate},
                  "source_if_statements": {v: (HERE / "sources" / f"{h}.metal").read_text().count("if (")
                                           for v, h in (("reference", reference), ("candidate", candidate))}}
        for metric in ("gpu_throughput", "e2e_throughput", "gpu_latency", "e2e_latency"):
            pairs = [grouped[name, index] for index in range(4)]
            gain = [read(pair["reference"], metric) / read(pair["candidate"], metric) for pair in pairs]
            torch = [read(pair["candidate"], metric) / read(pair["candidate"], metric, "torch") for pair in pairs]
            record[metric] = {"reference": median(read(p["reference"], metric) for p in pairs),
                              "candidate": median(read(p["candidate"], metric) for p in pairs),
                              "torch": median(read(p["candidate"], metric, "torch") for p in pairs),
                              "gain": median(gain), "min": min(gain), "max": max(gain), "pairs": gain,
                              "candidate_over_torch": {"median": median(torch), "min": min(torch), "max": max(torch)}}
        summary["cases"].append(record)
    (HERE / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"PASS: 192 executed output validations; 12 identical plans; {summary['artifacts']} stable artifacts")
    for row in summary["cases"]:
        gpu, host = row["gpu_throughput"], row["e2e_throughput"]
        print(f"{row['name']}: source {'same' if row['same_source'] else 'changed'}; "
              f"GPU {gpu['gain']:.3f} [{gpu['min']:.3f}, {gpu['max']:.3f}]; "
              f"E2E {host['gain']:.3f} [{host['min']:.3f}, {host['max']:.3f}]")


if __name__ == "__main__":
    main()
