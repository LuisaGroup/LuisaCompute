"""Read-only offline check of fixed and query-tile Metal4 captures; no GPU work.

Use the original raw root plus --helpers-dir for a local audit, or an extracted
evidence directory containing raw/, tensors/ and sources/helpers/. Numerical
oracles read every input/output value. Live binaries are never loaded/executed.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("out-of-scope relative path")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--helpers-dir", type=Path)
    parser.add_argument("--output", type=Path, help="new audit JSON path; never overwrites a prior result")
    args = parser.parse_args()
    base = args.evidence.resolve(strict=True)
    extracted = (base / "raw").is_dir()
    raw = base / "raw" if extracted else base
    helper_dir = (args.helpers_dir or base / "sources/helpers").resolve(strict=True)
    sys.path.insert(0, str(helper_dir))
    import metal4_timing as timing

    plan, result = read(raw / "query-tiles-plan.json"), read(raw / "query-tiles-results.json")
    assert result["status"] == "complete" and result["artifacts_unchanged"] is True
    assert plan["query_tile_order"] == [4, 1, 1, 4, 4, 1, 1, 4]
    assert (plan["key_tile"], plan["local_lanes"], plan["samples_per_visit"], plan["repetitions"]) == (16, 1, 3, 8)
    assert len(result["rows"]) == 16 and len(plan["cases"]) == 2
    for name in ("metal4_timing.py", "compare_llm.py", "run.py", "repeat.py"):
        matches = [v for p, v in plan["frozen_sha256"].items() if Path(p).name == name]
        assert len(matches) == 1 and sha(helper_dir / name) == matches[0]
    assert sha(raw / "query_tiles.py") == next(v for p, v in plan["frozen_sha256"].items() if Path(p).name == "query_tiles.py")

    common_artifacts, input_hashes, visits = None, {}, []

    def check_cohort(relative, label):
        nonlocal common_artifacts
        path = contained(raw, relative)
        cohort = read(path)
        assert cohort["cohort_valid"] is cohort["gpu_diagnostics_valid"] is cohort["artifacts_unchanged"] is True
        assert cohort["artifacts_before"] == cohort["artifacts_after"]
        artifacts = cohort["artifacts_before"]
        if common_artifacts is None:
            common_artifacts = artifacts
        else:
            assert artifacts == common_artifacts
        # Compare captured artifact receipts to the experiment's frozen set
        # where present. This is internal consistency, not live loader proof.
        for original, info in artifacts.items():
            if original in plan["frozen_sha256"]:
                assert info["sha256"] == plan["frozen_sha256"][original]
        assert cohort["protocol"]["host_repetitions"] == cohort["protocol"]["device_repetitions"] == 8
        assert cohort["protocol"]["samples"] == 3
        checked = []
        for row in cohort["results"]:
            assert row["status"] == "OK" and row["valid"] is True and row["exit_code"] == 0
            stdout = contained(path.parent, row["stdout"])
            stderr = contained(path.parent, row["stderr"])
            assert not timing.gpu_failure_diagnostics(stdout.read_bytes(), stderr.read_bytes())
            payload = read(stdout)
            op, dims = row["operation"], row["dimensions"]
            block = cohort["protocol"]["attention_block"] if op == "attention" else [1, 1]
            actual = timing.validate_local_lanes(payload["realization"], row["requested_local_lanes"])
            assert actual == row["actual_local_lanes"] == 1
            recomputed = timing.validate(payload, op, dims, 3, 8, block)
            assert recomputed == row["metrics"]
            original_output = Path(row["command"][-1])
            declared_tensor_root = Path(cohort["temporary_tensor_root"])
            assert original_output.is_relative_to(declared_tensor_root)
            tensor_base = base / "tensors" / label if extracted else declared_tensor_root
            output = tensor_base / original_output.relative_to(declared_tensor_root)
            expected_paths = [original_output, *(Path(str(original_output) + f".input{i}.f32") for i in range(3))]
            assert set(row["tensor_receipts"]) == set(map(str, expected_paths))
            for original in expected_paths:
                local = tensor_base / original.relative_to(declared_tensor_root)
                receipt = row["tensor_receipts"][str(original)]
                assert local.stat().st_size == receipt["bytes"] and sha(local) == receipt["sha256"]
            inputs = [row["tensor_receipts"][str(p)]["sha256"] for p in expected_paths[1:]]
            identity = (op, tuple(dims))
            assert input_hashes.setdefault(identity, inputs) == inputs
            oracle = timing.validate_exports(output, op, dims)
            assert oracle == row["independent_correctness"]
            group = re.findall(r"(?:^|;)\s*(\d+) threads/group(?:;|$)", payload["realization"])
            assert len(group) == 1
            field = lambda name: re.findall(r"(?:^|;)\s*" + name + r"=(\d+)(?=;|$)", payload["realization"])
            snapshots, allocations = field("static_snapshot_bytes_per_worker"), field("static_snapshot_allocations")
            assert len(snapshots) == len(allocations) == 1
            block_size = payload["device_timing"]["throughput"][0]["dispatches"][0]["block_size"]
            dispatch_size = payload["dispatch"]
            item = dict(operation=op, dimensions=dims, block=block, round=row["round"], metrics=recomputed,
                        threads_per_group=int(group[0]), snapshot_bytes=int(snapshots[0]), snapshot_allocations=int(allocations[0]),
                        dispatch_size=dispatch_size, block_size=block_size,
                        threadgroup_count=math.prod((d + b - 1) // b for d, b in zip(dispatch_size, block_size)),
                        dispatches_per_throughput_sample=8, dispatches_per_latency_sample=1,
                        local_lanes=actual, oracle=oracle, output_sha256=row["tensor_receipts"][str(original_output)]["sha256"])
            checked.append(item)
            visits.append(item)
        return checked

    fixed = check_cohort("metal4-fixed/cohort/results.json", "metal4-fixed")
    assert len(fixed) == 4
    query = []
    for case_index, (case, dims) in enumerate(plan["cases"]):
        case_visits = []
        for position, bq in enumerate(plan["query_tile_order"]):
            row = result["rows"][case_index * 8 + position]
            assert (row["case"], row["position"], row["query_tile"]) == (case, position, bq)
            label = f"query-{case}-{position}-bq{bq}"
            assert row["result"] == label + "/cohort/results.json"
            checked = check_cohort(row["result"], label)
            assert len(checked) == 1 and checked[0]["dimensions"] == dims and checked[0]["block"] == [bq, 16]
            assert checked[0]["metrics"] == row["metrics"]
            case_visits.append(checked[0])
        ratios = {}
        for metric in case_visits[0]["metrics"]:
            values = [v["metrics"][metric]["median"] for v in case_visits]
            pairs = [values[b] / values[a] for b, a in ((1, 0), (2, 3), (5, 4), (6, 7))]
            ratios[metric] = dict(pairs=pairs, median=statistics.median(pairs), minimum=min(pairs), maximum=max(pairs))
        query.append(dict(case=case, dimensions=dims, visits=case_visits, bq1_over_bq4=ratios,
                          output_bitwise_equal=len({v["output_sha256"] for v in case_visits}) == 1))
    report = dict(status="passed", fixed_visits=4, query_visits=16, tensor_receipts=80,
                          query_samples=16 * 3 * 6, query=query,
                          maximum_abs_error=max(v["oracle"]["max_abs_error"] for v in visits),
                          boundary="Full independent FP64 outputs and retained input/artifact receipts checked; no native execution or loader-closure claim. Ratios are adjacent BQ1/BQ4 within two ABBA cycles, not a cross-framework ranking.")
    if args.output:
        with args.output.open("x") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(status="passed", fixed_visits=4, query_visits=16, tensor_receipts=80)))
    else:
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
