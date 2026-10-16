"""Validate the bundled compact acceptance projection; never compile or run GPU code."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re

ACCEPTANCE_SHA256 = "2a75645f402776080752295549cc26ebabda77af415c45e9eb9054f2d138068f"
SUMMARY_SHA256 = "bc68167de69362de036cc47ae57d0791147664e136e9513c1080f9602404987e"
COEFFICIENTS = dict(scalar_round=1, collective=2, group_setup=16,
                    global_access_byte=0, private_access_byte=0,
                    preferred_concurrent_programs=64)
RESOURCE_FIELDS = {"registers", "static_shared_bytes", "local_bytes", "max_threads"}
DEMAND_FIELDS = {"global_read_bytes", "global_write_bytes", "private_read_bytes", "private_write_bytes"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected):
    require(isinstance(actual, (int, float)) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            f"score mismatch: {actual!r} != {expected!r}")


def receipt(value):
    require(set(value) == {"reference", "sha256", "bytes"}, "receipt fields")
    require(isinstance(value["reference"], str) and value["reference"].startswith(".deps/"), "relative reference")
    require(re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None, "SHA256 syntax")
    require(type(value["bytes"]) is int and value["bytes"] > 0, "receipt length")


def candidate(document, requested, resolved):
    require(document["schema"] == 1 and document["scope"] == "actual_ReductionCandidate_callback", "callback scope")
    require(document["policy"] == "unchanged_AnalyticExecutionCostPolicy"
            and document["coefficient_override"] is False and document["overflow"] is False, "policy changed")
    require(document["requested_unroll"] == requested
            and document["selected_resolved_unroll"] == resolved, "request/selection")
    require(len(document["records"]) == 1, "one exact candidate per acceptance run")
    r = document["records"][0]
    require(r["candidate_index"] == 0 and r["threads"] == 128
            and r["programs_per_group"] == 2 and r["subgroups_per_program"] == 2
            and r["lane_elements"] == 1 and r["unroll_factor"] == resolved, "geometry/resolved unroll")
    require(type(r["programs"]) is int and r["programs"] > 0
            and r["threadgroups"] == (r["programs"] + 1) // 2, "program grid")
    minimum = r["source_constant_striped_index_min_unroll"]
    require(type(minimum) is int and 1 <= minimum <= 64, "known sufficient unroll")
    require(requested != 0 or resolved == minimum, "automatic resolution does not follow observed fact")
    require(r["analytic_coefficients"] == COEFFICIENTS, "original coefficients")
    access = 0.0
    if r["payload_accesses_known"] is True:
        for field in ("payload_accesses_per_program", "payload_accesses_per_worker"):
            demand = r[field]
            require(set(demand) == DEMAND_FIELDS and all(type(x) is int and x >= 0 for x in demand.values()), "known demand")
        d = r["payload_accesses_per_worker"]
        access = ((d["global_read_bytes"] + d["global_write_bytes"]) * COEFFICIENTS["global_access_byte"]
                  + (d["private_read_bytes"] + d["private_write_bytes"]) * COEFFICIENTS["private_access_byte"])
    else:
        require(r["payload_accesses_known"] is False and r["payload_accesses_per_program"] is None
                and r["payload_accesses_per_worker"] is None, "unknown demand must remain null")
    score = (access + r["scalar_rounds"] * COEFFICIENTS["scalar_round"]
             + r["reductions"] * r["subgroups_per_program"] * COEFFICIENTS["collective"]
             + COEFFICIENTS["group_setup"] / r["programs_per_group"])
    waves = max(1, math.ceil(r["programs"] / max(1, COEFFICIENTS["preferred_concurrent_programs"])))
    close(r["cost"]["program_score"], score)
    close(r["cost"]["concurrent_waves"], waves)
    close(r["cost"]["kernel_score"], score * waves)
    return r


def validate(data):
    require(data["schema"] == 1 and data["status"] == "compact_projection_of_validated_acceptance", "compact scope")
    impl = data["implementation"]
    require(impl["default_unroll_factor"] == 1 and impl["cuda_automatic_request"] == 0
            and impl["manual_control"] == impl["unroll_cap"] == 64, "implementation bounds")
    require(impl["unchanged_policy"] == "AnalyticExecutionCostPolicy", "unchanged cost policy")
    require(data["reported_validation_counts"] == dict(admissions=24, numerical_validations=24,
            explicit_unsupported=0, actual_graph_nodes=2400), "reported count projection")
    for value in data["upstream"].values():
        receipt(value)
    require(data["upstream"]["summary"]["sha256"] == SUMMARY_SHA256, "upstream summary identity")
    build = data["build_validation"]
    require(build["fresh_full_build"] is False and len(build["blocked_attempts"]) == 2, "targeted build boundary")
    for attempt in build["blocked_attempts"]:
        require(attempt["reported_exit_code"] == 4551, "preserved blocked build")
        receipt(attempt["log_receipt"])
    receipt(build["bridge_log"])
    expected = {"host": (921, 12), "planner": (14429, 14), "gpu": (2184696, 12)}
    require(len(build["tests"]) == len(expected), "test census")
    for test in build["tests"]:
        require(test["role"] in expected and (test["assertions"], test["test_groups"]) == expected.pop(test["role"])
                and test["reported_exit_code"] == 0, "reported targeted validation")
        receipt(test["log_receipt"])
    require(data["preserved_failure"]["status"] == "failed", "v1 failure retained")
    seen, cases, graph_nodes = set(), {}, 0
    require(len(data["pairs"]) == 12, "12 separate math-mode pairs")
    for pair in data["pairs"]:
        key = (pair["case"], pair["fast_math"])
        require(type(pair["fast_math"]) is bool and key not in seen and pair["geometry"] == [128, 2], "unique pair geometry")
        seen.add(key)
        cases.setdefault(pair["case"], []).append(pair)
        require(pair["manual_resolved_unroll"] == 64 and pair["source_bytes_equal"] is True, "reported source parity")
        require(pair["reported_resource_fields_equal"] is True and pair["known_resource_values_equal"] is True, "known resource parity")
        require(pair["resources"]["auto"] == pair["resources"]["manual"], "resource attributes differ")
        resources = pair["resources"]["auto"]
        require(set(resources) == RESOURCE_FIELDS, "resource set")
        for value in resources.values():
            require(set(value) == {"value", "known", "cuda_status"} and value["known"] == "1"
                    and value["cuda_status"] == "0" and value["value"].isdigit(), "recorded queried resource")
        records = {}
        for label, requested, resolved in (("auto", 0, pair["auto_resolved_unroll"]), ("manual", 64, 64)):
            source = pair["source_receipts"][label]
            receipt(source)
            receipt(pair["case_receipts"][label])
            records[label] = candidate(pair["candidates"][label], requested, resolved)
            record = records[label]
            observation = pair["reported_execution_observation"][label]
            require(observation["entry"] == "workload_rows_kernel" and observation["grid"] == [record["threadgroups"], 1, 1]
                    and observation["block"] == [128, 1, 1] and observation["dynamic_shared_bytes"] == 0, "actual launch projection")
            require(observation["actual_graph_nodes_checked"] == 100
                    and observation["same_module_function_identity_checked"] is True
                    and observation["actual_final_pointers_checked"] is True, "graph observation")
            require(observation["requested_unroll_factor"] == requested
                    and observation["candidate_resolved_unroll_factor"] == observation["actual_plan_unroll_factor"] == resolved
                    and observation["unroll_evidence"] == "actual_ReductionCandidate_and_emitted_GroupPlan", "callback/plan binding")
            policy = "fast-elements-preserved-reductions-v1" if pair["fast_math"] else "strict-no-contract-v1"
            require(observation["math_policy"] == policy, "math-mode pairing")
            graph_nodes += observation["actual_graph_nodes_checked"]
        require(pair["auto_min_unroll"] == [records["auto"]["source_constant_striped_index_min_unroll"]], "minimum projection")
        a, b = pair["source_receipts"]["auto"], pair["source_receipts"]["manual"]
        require(a["sha256"] == b["sha256"] and pair["source_byte_lengths"] == [a["bytes"], b["bytes"]]
                and a["bytes"] == b["bytes"], "paired source receipt identity")
        require({k: v for k, v in records["auto"].items() if k != "unroll_factor"}
                == {k: v for k, v in records["manual"].items() if k != "unroll_factor"}, "cost facts changed beyond unroll")
        for value in pair["fixture_sha256"].values():
            require(re.fullmatch(r"[0-9a-f]{64}", value) is not None, "fixture digest")
    require(len(cases) == 6 and all({p["fast_math"] for p in group} == {False, True} for group in cases.values()), "six complete strict/fast fixtures")
    require(graph_nodes == 2400, "graph count")
    for case, group in cases.items():
        require(len({p["auto_resolved_unroll"] for p in group}) == 1, "strict/fast resolution consistency")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("acceptance.json"))
    args = parser.parse_args()
    raw = args.input.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ACCEPTANCE_SHA256, "bundled acceptance bytes changed")
    def reject_constant(value):
        raise ValueError(f"non-finite JSON number: {value}")
    cases = validate(json.loads(raw, parse_constant=reject_constant))
    print("Compact projection validated: 24 reported validations, 12 source/resource pairs, 2,400 reported graph nodes.")
    for name, group in cases.items():
        print(f"{name}: U0 -> {group[0]['auto_resolved_unroll']}, manual U64; strict/fast kept separate")
    print("Original GPU kernels, CUDA source files and full tensor oracle were not rerun or re-read by this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
