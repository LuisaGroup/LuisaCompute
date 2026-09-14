# Fixed-group reduction mapping protocol

## Question and frozen comparisons

Does reducing each row's private work through two cooperating SIMD groups
outweigh the cost of group synchronization and fewer packed rows? Does this
explicit mapping also improve on the current automatic execution family?

This protocol is recorded before timing. Use the unchanged executable and
libraries validated with commit `356c5d53c`. No implementation, coefficient,
private budget or default change is part of the initial experiment.

All variants use FP32 Metal on this Apple M1 Max, V=4, ordered unroll U=1,
preserved shared SSA, immutable input caching and the 64-scalar private bound.
The operator cohort is softmax, RMSNorm and affine LayerNorm at:

```text
37x769, 1024x1024, 16384x257, 4096x1024
```

This covers a partially packed/ragged case, two repeated widths with different
program counts, and many short ragged rows. It is not an operator/device
holdout, a production LLM suite, or an exhaustive mapping search.

Two independent four-round paired comparisons are predeclared:

1. **Fixed total group size:** `reference` is S=1/P=8/W=32/T=256;
   `candidate` is S=2/P=4/W=64/T=256. P counts programs per group; S counts
   cooperating SIMD groups per program; W=32*S; T=W*P. The second mapping
   needs group fences/partials and halves the worker stripe. Changing W can
   change floating-point tree order, under the existing reassociation option;
   full-output FP64 tolerances remain unchanged.
2. **Automatic execution control:** `automatic` sets T=0/P=0 and lets the
   unchanged analytic solver choose from its established family, with exactly
   the same resource policy. Compare it to the same predeclared S=2/P=4
   candidate, not a winner selected from comparison 1. This is the automatic
   execution family with caching enabled, not all-default CompileOptions.

Neither comparison may drop a case, choose a winner after its timing, pool
unpaired phases/rounds, or multiply gains from other experiments. Keep the
complete two comparisons even if the first one loses. Do not infer a new
automatic default or a calibrated grouping model from this finite cohort.

## Measurements and evidence

Three fixed parameter pilots (`reference/`, `candidate/`, `automatic/`) use
three host samples, 10 ms sample windows and 100 ms warm-up. They validate
complete outputs and freeze exact plans/generated-source hashes. Pilot times
are not the acceptance evidence; pilots have no GPU timing phase.

Run `repeat.py` sequentially for `fixed-replay/` and `automatic-replay/`, each
with four counterbalanced fresh-capture/JIT rounds, nine samples, 30 ms target
sample duration and 200 ms warm-up. Alternate both variant-first and
native/Torch-first orders. Capture generated Metal and fingerprint the native
binary, adjacent Tile libraries, TVM compiler/runtime/Metal runtime/FFI and the
Metal timing helper. Run no builds, tests or profilers concurrently.

Report separately:

- warm batched host-wall throughput, including dispatch and amortized sync;
- individually synchronized end-to-end latency;
- uninstrumented command-buffer GPU throughput and single-call time; and
- instrumented compute-pass samples as diagnostics only.

The GPU control includes work/gaps within completed command buffers and is
not an isolated-kernel timestamp. Never subtract independently sampled phase
medians to estimate dispatch cost. Retain repetitions and all raw samples.

Native output is preallocated. Eager Torch MPS softmax uses `out=`; functional
normalization returns allocated output in timing. Record allocation/math
policy qualifications near every external comparison. Direct MPS/MPP, CUDA,
CPU and compiled Torch are not measured here. Preserve all regressions.

An independent audit must enumerate ownership/resource facts and the complete
automatic candidate costs, validate frozen source/plan identity and round
balance, and recompute all four paired metrics from raw samples. All complete
outputs must pass the runner's independent FP64 oracle before accepting a
measurement. The post-hoc audit checks these records, not nonexistent archived
output arrays. Reuse the last code validation only after confirming the
current binary hashes; do not present its 31/33 local CTest boundary as a new
all-green test run.

## Report contract

The user-selected delivery surface remains the existing Sphinx documentation,
`docs/source/performance/tile/reductions.md`, with methods/raw evidence in this
directory. The `build-report` technical roles map to a result-led summary,
precisely labeled paired findings, definitions and experimental design,
robustness/limitations, and next questions. Preserve earlier report sections.
Do not create a parallel HTML/MCP application or change the site's theme.

Use a neutral exact-lookup table for the twelve operator/shape cases: paired
gain medians and observed min/max are all required for acceptance, and GPU/E2E
must remain distinguishable. This is the `visualize-data` tables/audit-detail
choice rather than a ranking chart; ordering is operator then predeclared
shape, not measured speed. Ranges are not confidence intervals. Inspect the
result in the existing Sphinx page and check source/compatibility links.
