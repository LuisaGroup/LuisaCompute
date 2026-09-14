# Cooperating-program packing experiment

## Question and preselected comparison

Can multiple independently cooperating reduction programs share a physical
Metal threadgroup profitably, without changing each row's worker ownership,
input caching, private stripes, or floating-point recurrence?

This is a fixed mapping-family experiment, not a shape search, new default,
or claim that the selected width is optimal. Both variants use 256 workers
per program, four consecutive elements per worker, one-way stripe unrolling,
immutable input caching, the 64-scalar private budget, and the analytic policy.
Reference: one program and 256 threads per group. Candidate: two programs and
512 threads per group. The cooperating width stays eight SIMD groups per row.

The cases were chosen before candidate timing: softmax, RMSNorm, and LayerNorm
at 37x1537, 256x3072, 768x6144, and 1024x4096. This includes an odd program
count and an element tail, plus larger fully packed cases. Retain every case,
round, and regression; do not select a winner from pilot timings.

## Measurement and acceptance

`reference/` and `candidate/` are fresh correctness-checked parameter pilots.
Their timings are not used in the paired comparison. The reference pilot
predates a formatting-only library rebuild; the replay must check both
variants with the same post-build binaries and frozen generated sources.

`replay/` will contain four counterbalanced fresh-JIT rounds, nine samples,
30 ms target sample duration, and 200 ms warm-up. Use the repository `repeat.py`
runner, recording full native/Torch outputs, exact launch plans, source hashes,
and adjacent/runtime/compiler artifact hashes. Do not run builds, tests, or
profilers concurrently with the measurements.

Keep warm batched host-wall throughput, individually synchronized host-wall
latency, instrumented compute-pass diagnostics, and uninstrumented GPU
command-buffer controls separate. Command-buffer time includes GPU work/gaps
and possible blits; it is not an isolated kernel timestamp. Never subtract
independent phase medians or multiply gains from earlier experiments.

Native outputs are preallocated. Torch softmax uses `out=`; functional norms
return newly allocated outputs. Torch is eager on MPS, not a fused compiled
graph or a direct MPS matrix baseline. The native-to-native comparison holds
the API and allocation policy fixed. All outputs must pass the runner's full
independent FP64 oracle before accepting timing records.

The implementation gate is a full CMake build and all Tile CTests, with the
two known unrelated local barrier-source assertion failures retained. New
coverage includes 36 numeric packing configurations, six typed raw-IR proof
cases, unconditional-fence inspection, guard rows, exact/automatic cooperating
widths, and cost accounting for inactive-tail reads.

## Report shape

Update the existing Sphinx `docs/source/performance/tile/reductions.md` and
`docs/source/internals/tile/reductions.md`, as requested by the user; do not
create a parallel report application. The technical summary leads, followed
by exact comparison/units, per-case results, limitations, and next actions.
Use a compact neutral table for exact operator/shape lookup, with per-round
ranges and GPU/E2E phases kept distinct; a chart is omitted because all cases
and paired extrema are needed for acceptance, not a visually ranked winner.
Saved JSON, the independent audit, source files, and this protocol supply the
methods/provenance roles of the technical-report specification. Preserve all
previous sections and caveats in the existing report.
