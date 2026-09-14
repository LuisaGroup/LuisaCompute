# Bounded-K MPP view validation and replay

This is a capability-envelope change, not a coefficient fit. Before timing,
freeze the new implementation after its complete-output regression tests.
The previous binaries and TVM v2 libraries are preserved at
`/tmp/luisa-mpp-bounds-baseline.398l4W`; retain their rejected fixed requests as
admission evidence, not speed ratios against a nonexistent execution.

Replay four M×N×K shapes: 128×128×61, 1024×1024×1537,
4096×4096×11008, and 8192³. The first three have K tails; the last is an
unchanged aligned scaling control. The view schedule remains the previously
fixed 128×32×1024, 128 threads, ordered pipeline, copy batch 1. This run does
not search parameters or select a schedule from these new timings.

Use all seven existing `compare_lowerings.py` paths: native Tile→MPP,
ordinary TIRx SIMD-group, handwritten MPP, direct MPS, eager Torch MPS,
TIRx MPP with materialized inputs, and TIRx MPP input views. Native/handwritten
MPP retain 32,32,1,1,0,1,4,4; non-view TIRx retains 32×32×32 and 128 threads.
Run 14 balanced orders with rotating shape order, five samples, 20 ms target
windows, 100 ms warmup, and a 300-second per-process timeout. Include every
failure and regression. No concurrent tests, builds or profilers during timing.

FP32 compact row-major C=A×B, alpha=1, beta=0, no transpose/prepacking or
relaxed input precision. Native/handwritten MPP disable fast math; TVM Metal
retains its existing fast-math setting. Validate every output against the
unchanged FP64 oracle, atol=rtol=1e-4. Regression tests separately use
non-dyadic inputs, all A/B transpositions and nonzero initial accumulators.

Report host-wall throughput and synchronized latency separately from the
no-counter GPU command-buffer batch and single-call intervals. The latter
include command-buffer work/gaps, not isolated shader-instruction time.
Compilation, allocation, transfers and the oracle are untimed. Fingerprint
executables, adjacent libraries, the timing helper and explicit TVM/FFI
libraries before and after replay. Retain generated source identities.

The pre-existing local change to Metal barrier flags in `cooperative.cpp`
(3→2) is present in both frozen baseline and candidate builds. It is not part
of this patch and is not staged into its commit. Do not claim the whole dirty
worktree passes all tests; two older source-string assertions expect flag 3.

Scope boundary: only a common positive, zero-padded K suffix is newly
forwarded. M/N tails, unequal K masks, extra predicates, nonzero fill, mutable
inputs and unproved address maps retain the strict snapshot path. The broad
MPS/Torch parity goal remains open regardless of this cohort's outcome.
