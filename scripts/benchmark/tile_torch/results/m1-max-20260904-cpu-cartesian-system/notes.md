# Cartesian CPU packing against Torch and BLAS, 2026-09-04 UTC

The CPU library gap remains large. FP32 1024³ median times are **7.820 ms Tile,
0.975 ms Torch, 0.984 ms Accelerate BLAS**; the median paired Tile/Torch ratio
is **7.990×**, Tile/BLAS **7.845×**. Every tested shape remains slower than both
libraries. The [full eight-shape table](results.md) retains all slow rounds.

This is a fresh six-order comparison, not a ratio against historical library
times. The [separate fixed-geometry A/B](../m1-max-20260904-cpu-cartesian-replay/notes.md)
isolates the single-row-to-Cartesian policy change and contains its regressions.
Do not interpret either comparison as a solved CPU cost model.

## Fixed policy and comparison boundary

All eight shapes use the same 4×16×32 tile, worker binding, pipeline window 2,
automatic vectorization, 8192-byte stack payload budget, and 64-scalar logical
Cartesian pack budget. The latter is not a hardware vector width. Each kernel
is freshly captured and compiled with the generic LLVM target; no shape search,
source rewriting, library-call substitution, or CPU prepacking is introduced.

Torch is the installed eager CPU implementation with a preallocated output;
the independent system binary calls classic LP64 Accelerate `cblas_sgemm`.
Both are comparison baselines, not our lowering implementation. No particular
undocumented Apple matrix instruction or library-internal worker count is
inferred from timing. All paths use compact FP32 row-major C=A×B, alpha=1,
beta=0, identical input values, and complete FP64 output references.

## Validation

- Six rounds × eight shapes × three implementations: **144/144 outputs valid**,
  33,798,528 checked elements, maximum absolute error zero for the deterministic
  dyadic inputs. Additional non-dyadic/cancellation/alias regressions are
  described with the A/B report; zero dyadic error alone is not an accuracy proof.
- All six implementation permutations are present for every shape. No invalid,
  slow, or late round is removed. Timings are per-round p50 medians; paired
  ratio ranges are not confidence intervals.
- Seven samples × 30 ms; 200 ms warmup; eight requested threads. Warm timing
  is synchronized host wall time including API dispatch and internal scratch
  costs, excluding JIT, setup, input/output allocation, and transfers.
- All 20 executable/shared-library paths remained unchanged and were
  independently rehashed after timing. All 48 raw LLVM source hashes were
  independently checked. Commands, versions, overrides, samples, and compiler
  artifacts are in [results.json](results.json).
- Both Luisa build trees were fully built before tests/timing. CTest was 23/25
  for each, with the same two preexisting Metal fence-text assertion failures;
  no new failure was hidden. No build, test, or profiler overlapped timing.
  Ordinary OS/user activity remains uncontrolled.

The [additional 12 add/sum/softmax cases](../m1-max-20260904-cpu-cartesian-ops/results.md)
also passed full checks (24 outputs; maximum error 3.54e-9). Their single-pass
timings are exploratory, not evidence of a packing speedup or library parity.

## Consequences for planning

The realization now exposes multi-row B reuse, but the end-to-end cost still
includes snapshot packing, per-program work granularity, cache reuse across
programs, scalar/conditional tails, and accumulator/register pressure. Those
features need separate measurements and candidate families. A larger vector
budget is neither an occupancy estimate nor a substitute for that model.

Original CPU/Metal TIRx and the independent Metal MPP paths remain intact.
This CPU run does not retime the existing seven-way MPS/MPP/Torch report.
