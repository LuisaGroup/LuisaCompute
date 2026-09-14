# Native Tile Runtime: first five-path replay

The [full results](results.md) contain eight shapes, ten counterbalanced rounds
and five implementations: **400/400 complete outputs passed** the common FP64
oracle. This is one Apple M1 Max, FP32, one deterministic benchmark input
pattern. The native runtime tests separately cover non-dyadic inputs,
transposes, changed inputs, offset views and guard regions.

## What the comparison establishes

- Actual TileIR→MPP code now runs through Luisa's Metal backend, normal shader
  handle, Buffer/BufferView bindings and Stream. The handwritten MPP numbers
  are a separate implementation, never relabeled as compiler performance.
- At 1024³, median host-wall throughput is about 295 µs native, 319 µs TIRx,
  272 µs handwritten MPP, 279 µs MPS and 291 µs Torch. The paired native/MPS
  ratio is 1.058. The gap is not closed.
- At 513×257×129, TIRx is about 23 µs versus 32 µs native. Keeping TIRx as an
  independently exercised path is necessary; native is not uniformly better.
- Matched native/handwritten MPP configurations still differ in generated
  entry/argument plumbing, root grid realization and host dispatch runtime.
  Their timing difference cannot be attributed to Runtime overhead alone.
  A matched native GPU-interval measurement and generated-code/grid comparison
  should precede changes to planner costs.
- All table comparisons use host wall time. The available MPS/handwritten
  MPP GPU intervals cannot be substituted for missing native/TIRx GPU timers.

## Build and regression status

The complete `cmake-build-tirx` build passed with Metal and TIRx enabled.
Native codegen/runtime tests passed, including Luisa and Metal validation;
the existing SIMT DSL-sugar kernel also passed. The comparison drivers passed
35 Python tests. CPU BLAS and Metal MPS self-tests passed.

Existing TIRx CPU/Metal regression was **19/21**, not all green. Two generated
source checks still expect `metal::mem_flags(3)`, while a pre-existing local
change in `cooperative.cpp` emits `2`. That hunk was not overwritten or hidden
by loosening tests. No numeric assertion failed in that run. The checkout also
contains uncommitted private-prefetch work; the frozen TIRx plan used here has
window 1 in every case, and all reported `prefetched_pipeline_loops` are zero.
Raw metadata explicitly records the dirty checkout and artifact hashes.

## Artifact provenance and final hardening

All measured binaries and shared libraries remained unchanged during the
campaign. Local copies are retained at
`/tmp/luisa-tile-native-measured.gQffCN/`; their original path/hash inventory is
in [results.json](results.json). In particular:

```text
benchmark_tile_native:
a08909b7214af05814c2391c84c77eb79b9d8397307a3ae20316c9863df6d5ff
libluisa-backend-metal.so:
ded392cb63c9012e6c1ac0d957498aad1f389d4bb60e0421970c638d532c90fe
```

After timing, a compile-time input check was added to reject detached or
parentless Tile functions before module verification. The full build and
native tests passed again. All eight shapes were then numerically rechecked;
their generated-source hashes match the measured versions. These post-check
timings are not substituted into the replay. The final backend binary hash
therefore differs from the frozen measurement artifact, despite unchanged
generated kernels for these cases.

The native factory currently implements only the documented MPP subset.
Adapting TIRx output to that same Runtime factory, general Machine TileIR,
native K pipelines and tile epilogues remain follow-up implementation work.
