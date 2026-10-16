# CUDA TIRx automatic unroll acceptance

An explicit CUDA unroll request of **0** resolved to a sufficient factor from the actual striped-materialization analysis. In this acceptance run, all **12 strict/fast pairs** produced byte-identical CUDA source for automatic unroll and the manual **64** control. Their four queried CUDA Driver resource attributes also matched. The default unroll factor remains **1**; this result does not enable a new default schedule or change the analytic cost policy.

The closed v2 run contains six norm fixtures, each tested with T128/P2/L1, automatic/manual unroll, and strict/fast math separately: **24 numerical validations and 2,400 observed graph nodes**. Before preparing this compact projection, the upstream CPU summary re-read all saved logical outputs with the original full FP64 oracle and error bounds, and checked original runtime read-only/guard reports, source identities, and recorded launch/argument proofs. It did not independently re-read the guard allocations. GPU kernels were run by the original acceptance queue, not by that CPU summary.

| Fixture | Automatic factor, both math modes | Registers strict / fast | Static shared bytes |
|---|---:|---:|---:|
| RMSNorm 128×1024 BF16 | 9 | 34 / 39 | 16 |
| RMSNorm 256×1536 FP16 | 17 | 40 / 40 | 16 |
| RMSNorm 7×2051 BF16 | 33 | 54 / 56 | 16 |
| Softmax 1024×512 FP16 | 5 | 28 / 23 | 80 |
| Softmax 32×512 FP16 | 5 | 28 / 23 | 80 |
| Softmax 9×769 FP16 | 9 | 40 / 31 | 80 |

Both members of every pair reported local bytes 0 and maximum threads 128. Maximum threads is a queried bound, not measured physical occupancy. The automatic factor is sufficient for constant source indices under the current lowering; it is not a claim about the minimum physical register allocation, actual spill traffic, or the globally fastest schedule. Unknown requirements or requirements above the CUDA cap of 64 reject the automatic candidate. Strict and fast observations remain separate; this packet does not assert identical math policies across them.

No dispatch-speed or compilation-speed claim is made from these correctness smokes. No Torch timings are used. The later P4 admissions are outside this packet. The failed v1 run, caused by a relative child artifact path, remains recorded separately and contributes no accepted sample; v2 used absolute output paths.

## Build and implementation scope

The implementation was committed as `cce8d442b`; exact tested source/binary receipts establish provenance, rather than attributing an earlier build to the later commit timestamp. The LLVM22 bridge and private acceptance executable were rebuilt with MSVC/CMake. Separate targeted validation passed 921 host assertions in 12 groups, 14,429 planner assertions in 14 groups, and 2,184,696 CUDA assertions in 12 groups.

This was **not a fresh full-project build**. Two full/aggregate build attempts stopped with exit 4551 when Windows blocked the device-library embed helper. Existing common production binaries retain prior full-build provenance, while the rebuilt bridge and imported private tests have independent receipts. Additional LLVM23 bridge/host checks are outside this packet; it does not claim LLVM23 GPU validation.

## Reproduce the compact projection

Run with Python 3 (standard library only):

```sh
python reproduce.py
```

This checks the exact bundled JSON digest, the 24/12/2,400 census, recorded source/resource pairing, observed callback-to-plan resolutions and graph descriptors, and the unchanged analytic score arithmetic. It reads only `acceptance.json`. It **does not compile code, run GPU kernels, re-read the unbundled CUDA source or tensors, or independently reproduce numerical correctness**. The source and tensor hashes are provenance receipts for the previously validated upstream evidence. Large source files, tensors, cubins and raw logs are intentionally not bundled.

The public JSON is a compact projection, not the upstream measurement file:

- `acceptance.json` SHA256: `2a75645f402776080752295549cc26ebabda77af415c45e9eb9054f2d138068f`.
- Original closed summary SHA256: `bc68167de69362de036cc47ae57d0791147664e136e9513c1080f9602404987e`.
- Upstream queue, readiness, targeted build/test, fixture, source and per-case receipts are retained inside the JSON using repository-relative references.
