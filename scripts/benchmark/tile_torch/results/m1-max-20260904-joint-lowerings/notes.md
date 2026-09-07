# Five-path replay after joint TIRx tuning

[The complete table](results.md) contains eight shapes, ten counterbalanced
rounds and five implementations. **400/400 complete outputs passed** the
common FP64 oracle; executable/shared-library hashes were unchanged throughout.
All compared table entries are synchronized host-wall batched throughput,
including dispatch, not GPU-only times. Compilation and transfers are excluded.

## What changed

The outer Staged/JIT benchmark can now search block shape, pipeline window,
group width and copy-batch limit jointly, with an explicit candidate budget.
The [search](../m1-max-20260904-joint-search/results.md) attempted 128 candidates:
108 passed and 20 were rejected; all eight selected schedules passed fresh
post-selection validation. A separate
[four-round same-binary replay](../m1-max-20260904-joint-replay/results.md)
then checked the old and selected schedules without searching again.

This five-path run freezes those TIRx settings and the previously selected
MPP descriptors/cohorts. No emitter or cost-model default changed. The
contraction-unroll experiment was reverted; its regressions and the other
negative probes remain in the [structural experiment report](../m1-max-20260904-tirx-structure.md).

## Results and remaining gaps

- The same-binary A/B replay improves TIRx 128³ from 9.821 to 6.841 µs,
  with a paired speedup median of 1.453×. This is relative to the old TIRx
  schedule, not relative to MPS or Torch.
- In this five-path replay, 128³ takes 6.827 µs through TIRx versus 9.478 µs
  through native MPP. On 513×257×129, the corresponding times are 22.783 and
  31.778 µs. Native is not a uniformly better replacement for TIRx.
- At 1024³, TIRx is 320.019 µs, native MPP 295.407 µs, handwritten MPP
  272.073 µs, MPS 278.567 µs and Torch 291.049 µs. The paired native/MPS
  ratio is 1.061. The large-square performance objective remains open.
- 512³ TIRx is 53.313 µs versus Torch 48.668 µs. Its changed copy limit did
  not establish an improvement in independent A/B replay. A search winner
  is not automatically a demonstrated optimization.

All complete output checks use one deterministic dyadic input pattern.
Separate kernel tests cover non-dyadic values, transposes, tails, repeated
input updates and offset views. FP32 tensor types and passing this benchmark
do not imply identical compiler fast-math settings or bitwise equivalence;
see the [math-policy qualification](../../README.md#five-path-tile-lowering-comparison).

## Validation and scope

The full Metal/TIRx CMake build passed. The CPU and Metal matrix suites plus
the planner suite passed **3/3** after restoring the serial emitter, and the
benchmark-driver suite passed **38/38** (with NumPy available).

The two pre-existing fence-source regression failures described in the
[first replay notes](../m1-max-20260904-native-lowerings/notes.md) remain
unresolved; their unowned `mem_flags(3)` to `mem_flags(2)` edit was not touched.
This is not an all-green claim for the whole TIRx suite. All selected pipelines
here have window one; the uncommitted late-prefetch path is inactive.

The TIRx benchmark still launches through TVM's runtime, while native MPP
launches through Luisa Runtime. Adapting TIRx to the same
`DeviceInterface::create_tile_kernel` factory and adding comparable native/TIRx
GPU-interval measurements remain implementation work. Until then, the
end-to-end differences cannot be assigned to codegen or Runtime overhead
alone. These results do not justify changing global planner coefficients.
