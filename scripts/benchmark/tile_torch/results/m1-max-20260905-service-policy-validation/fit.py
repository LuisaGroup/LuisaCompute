"""Freeze a small nonnegative service policy using only the prior search.

The six-dimensional NNLS solve enumerates all active sets after column scaling.
No kernel name or shape-specific coefficient is a model input. Capacity is
selected by leave-one-shape-out ranking regret, then all training rows are fit.
Cross-validation here is model selection, not independent acceptance evidence.
"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "m1-max-20260905-resource-map-search/results.json"
CAPACITIES = (64, 128, 256, 512, 1024, 2048)
HOLDOUT = ((37, 1537), (256, 3072), (768, 6144), (64, 12289))
FIELDS = ("dispatch", "scalar_round", "collective", "global_program_byte", "global_worker_byte", "private_worker_byte")


def features(plan, capacity):
    worker = plan["reduction_payload_accesses_per_worker"]
    program = plan["reduction_payload_accesses_per_program"]
    waves = max(1.0, plan["reduction_threadgroups"] * plan["reduction_subgroups_per_program"] * plan["reduction_programs_per_group"] / capacity)
    return [1.0, plan["reduction_scalar_rounds"] * waves,
            plan["reduction_operations"] * plan["reduction_subgroups_per_program"] * waves,
            plan["programs"] * (program["global_read_bytes"] + program["global_write_bytes"]),
            worker["global_read_bytes"] + worker["global_write_bytes"],
            (worker["private_read_bytes"] + worker["private_write_bytes"]) * waves]


def nnls(matrix, target):
    """Exact active-set enumeration for this bounded six-feature problem."""
    scale = np.linalg.norm(matrix, axis=0)
    scale[scale == 0] = 1.0
    matrix = matrix / scale
    best_loss, best = float(target @ target), np.zeros(matrix.shape[1])
    for bits in range(1, 1 << matrix.shape[1]):
        active = [i for i in range(matrix.shape[1]) if bits >> i & 1]
        coefficients = np.linalg.lstsq(matrix[:, active], target, rcond=None)[0]
        if np.any(coefficients < -1e-10):
            continue
        full = np.zeros(matrix.shape[1])
        full[active] = np.maximum(0, coefficients)
        loss = float(np.sum((matrix @ full - target) ** 2))
        if loss < best_loss:
            best_loss, best = loss, full / scale
    return best


def load_training():
    report = json.loads(SOURCE.read_text())
    records = []
    rejected = 0
    for row in report["results"]:
        seen = set()
        for trial in row["tuning"]["trials"]:
            key = trial["group_threads"], trial["cache_reduction_inputs"]
            assert key not in seen
            seen.add(key)
            if not trial["valid"]:
                rejected += 1
                continue
            measured = trial["measurement"]
            native = measured["native"]
            assert measured["valid"] is True and measured["backend"] == "metal"
            assert native["reduction_lane_elements"] == 4 and native["reduction_unroll_factor"] == 1
            assert native["reduction_programs_per_group"] == 1 and len(native["execution_plans"]) == 1
            plan = native["execution_plans"][0]
            assert plan["reduction_payload_accesses_known"] is True
            control = native["device_timing"]["control"]
            assert control["encoder_instrumentation"] is False
            assert control["method"] == "metal_command_buffer_timestamps_v1"
            actual = median(s["command_buffer_ns"] for s in control["throughput"]) / control["repetitions"] / 1000
            assert np.isclose(actual, control["command_buffer_throughput_us_p50"], rtol=1e-12)
            records.append(dict(case=row["name"], shape=(row["case"]["m"], row["case"]["n"]),
                                threads=key[0], cache=key[1], plan=plan, actual_us=actual))
        assert len(seen) == 10
    assert len(records) == 101 and rejected == 19
    assert {r["shape"] for r in records}.isdisjoint(HOLDOUT)
    return records


def ranking(records, predicted, selected):
    output = []
    for name in sorted({records[i]["case"] for i in selected}):
        indices = [i for i in selected if records[i]["case"] == name]
        best = min(indices, key=lambda i: predicted[i])
        measured_best = min(indices, key=lambda i: records[i]["actual_us"])
        record = records[best]
        output.append(dict(case=name, threads=record["threads"], cache=record["cache"],
                           prediction_us=float(predicted[best]), measured_us=record["actual_us"],
                           best_measured_us=records[measured_best]["actual_us"],
                           regret=record["actual_us"] / records[measured_best]["actual_us"] - 1.0))
    return output


def fit():
    records = load_training()
    targets = np.array([r["actual_us"] for r in records])
    shapes = sorted({r["shape"] for r in records})
    candidates = []
    for capacity in CAPACITIES:
        matrix = np.array([features(r["plan"], capacity) for r in records])
        folds = []
        for held_shape in shapes:
            train = np.array([r["shape"] != held_shape for r in records])
            coefficients = nnls(matrix[train] / targets[train, None], np.ones(sum(train)))
            predicted = matrix @ coefficients
            folds.extend(ranking(records, predicted, [i for i in range(len(records)) if not train[i]]))
        coefficients = nnls(matrix / targets[:, None], np.ones(len(records)))
        predicted = matrix @ coefficients
        candidates.append(dict(capacity=capacity, coefficients=coefficients.tolist(),
                               cv_mean_regret=mean(fold["regret"] for fold in folds),
                               cv_max_regret=max(fold["regret"] for fold in folds), folds=folds,
                               training_relative_rmse=float(np.sqrt(np.mean((predicted / targets - 1) ** 2))),
                               training_picks=ranking(records, predicted, range(len(records)))))
    winner = min(candidates, key=lambda item: (item["cv_mean_regret"], item["capacity"]))
    profile = ",".join(["service-v1", str(winner["capacity"]), *(format(c, ".17g") for c in winner["coefficients"])])
    return dict(schema="reduction_service_calibration_v1", source=str(SOURCE.relative_to(HERE.parent)),
                source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                fit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                numpy_version=np.__version__, training_valid_trials=len(records), training_rejected_trials=19,
                training_shapes=shapes, numerical_policy="FP32 subgroup trees, V4/U1/P1, private budget 64",
                fit_objective="sum of squared per-trial relative timing errors; nonnegative coefficients",
                capacity_selection="lowest mean leave-one-shape-out per-case regret, then smallest capacity",
                timing="no-counter command-buffer GPU batched throughput us/op; not isolated-kernel time",
                coefficient_names=FIELDS, native_argument=profile, selected_capacity=winner["capacity"],
                candidates=candidates, holdout_shapes=HOLDOUT,
                holdout_operations=["softmax", "rmsnorm", "layernorm"],
                holdout_selection="old automatic width with reload vs service automatic width and model-only reload/cache JIT product",
                holdout_replay=dict(rounds=4, samples=9, sample_ms=30, warmup_ms=200),
                defaults_changed=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fit()
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(result["native_argument"])
    for row in result["candidates"]:
        print(row["capacity"], row["cv_mean_regret"], row["cv_max_regret"], row["training_relative_rmse"])
