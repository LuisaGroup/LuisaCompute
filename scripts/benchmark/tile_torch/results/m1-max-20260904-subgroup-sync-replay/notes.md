# Subgroup isolation: a legal transformation is not always faster

The new whole-group proof permits removing compiler-owned group fences from
independent MPP subgroup programs. A frozen-geometry A/B test found a **512³
regression in every round**. Accordingly, production code exposes elision as
the **default-off** `PlannerOptions::elide_independent_subgroup_barriers`
choice, separate from `GroupPlan::independent_subgroups`, the proof result.
Neither the existing TIRx path nor default MPP view forwarding changes its
fence policy. No size lookup table or calibrated MPP cost model is introduced.

The [four-round table](results.md) and [raw results](results.json) retain all
64 native and 64 Torch outputs. All 30,043,136 output elements pass the shared
FP64 oracle, with maximum observed absolute error zero on these deterministic
dyadic inputs. Separate C++ regressions use non-dyadic inputs, nonzero C and
changed inputs. This is not a claim of exact FP32 arithmetic in general.

## The isolated variable

Reference and candidate use the identical [frozen view plan](../m1-max-20260904-tirx-views-plan.json),
SHA-256 `89d030ed65de2122d72736985bb0b598a78b13bcff7d8b71c820453b872dd259`.
There is no search. Shapes rotate, both compiler versions receive both A/B
positions, and each receives both native/Torch orders. Each native process
captures/JIT-compiles afresh. Timing uses seven 30-ms target samples after
200-ms warmup, synchronized device-resident **host wall**, including dispatch,
not GPU event time. Compilation, allocation and transfers are excluded.

The reference executable and all its adjacent libraries were copied before
the changes and verified against the preceding view report's 22 artifact
hashes. Versioned glslang/SPIRV symlink aliases were restored in the snapshot;
the first loader diagnostic had exposed those missing aliases. A successful
diagnostic launch then confirmed old adjacent Luisa libraries and the
unchanged patched TVM libraries. Diagnostic timings are not in this dataset.

The candidate here is the prototype that enabled the new proof inside the
existing coalescing option. It predates the separate default-off flag and
`independent_subgroups` metadata. Its actual fence counts and generated
sources remain archived; do not infer its behavior from a missing modern
policy field. After this experiment, the option was separated, both builds
and test cohorts rerun, and the seven-path comparison explicitly requests
elision with the new interface. Historical results were not edited to claim
the modern default.

All **38** recorded executables/library paths matched their before/after
hashes. All **14** content-addressed Metal sources were rehashed. A read-only
source diff confirms that the six interior shapes differ only by deletion
of the two static compiler barrier sites. The two ragged shapes have identical
source across versions. Geometry, copies, arithmetic and compiler options
are therefore controlled in this A/B; numerical noise for identical-source
ragged cases is not attributed to the transformation.

## Result and interpretation

For 512³, medians were 56.571 µs before and 62.624 µs after; the median paired
speedup is 0.902×, about an 11% increase in paired time. Its BK=32 kernel has
16 outer K steps: one static removed fence executes at every iteration.
For 1024³, medians were 383.786 and 379.720 µs; paired speedup is only 1.011×,
with a 0.995–1.072× range. That full-K kernel has no surviving outer K loop.
Neither result supports a universal “fewer barriers is faster” rule.

Apple's [MPP programming guide, §2.3.4](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
describes periodic group barriers as a cache-working-set tuning technique.
This offers a plausible mechanism for the regression, not measured proof of
its cause on M1. A cache/issue model would need controlled measurements of
K granularity, working set and cohort geometry, not a negative constant per
removed barrier. External-library timings are targets, not solver candidates.

Absolute times in this session are higher than the earlier view report,
including Torch and unmodified controls. Do not compare historical medians
as an implementation A/B or infer a specific thermal cause. The read-only
`pmset -g therm` check reported no recorded thermal/performance warning; that
is not a GPU clock measurement. Only within-run paired evidence is used here.

## Safety and implementation

The matrix emitter supplies identities for known synchronous, subgroup-private
cooperative-tensor operations and one partitioned output. The forwarding pass
supplies immutable noalias input identities. The entire body must contain only
these operations, uniform constant serial loops, no-ops and compiler fences,
ending in one output store outside all loops. Additional effects, an output
consumer, a store on a backedge, shared/manual resources, branches, escapes,
explicit synchronization or unknown statements reject the proof. A scope
string or zero shared-memory byte count alone is never permission.

Forwarding now iterates to a finite fixed point. Anonymous-axis relabeling had
introduced a second snapshot copy; every iteration rechecks the same complete
effect, dominance, bounds and nonescape conditions before eliminating another
compiler allocation. The regression keeps the zero-shared-storage assertion
instead of requiring users to reuse named axes to make the optimization work.

Both final Luisa configurations completed full builds. The native/TIRx/system
cohort remains **23/25** in each: only the pre-existing `mem_flags(3)` checks
in `test_tile_tirx_cooperative_metal` and `test_tile_tirx_memory_metal` fail
against the unowned worktree change to `mem_flags(2)`. Neither that edit nor
the assertions was overwritten. New positive/negative isolation tests pass,
including default retention, disabled coalescing, two pipeline windows,
anonymous shapes, extra global writes, post-store consumers and backedges.
The Python driver tests pass **47/47**. Original CPU/Metal tests also run with
unpatched TVM; unsupported MPP requests stay explicit failures.

The overall CPU/Torch and Metal/MPS performance goal remains open. This
experiment establishes a safe candidate and a measured rejection of its
unconditional use, not a performance-goal completion.

## Reproduce with the explicit policy

After a complete build and correctness validation, use one current binary
for both variants to compare modern default retention against explicit elision:

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/repeat.py \
  --reference scripts/benchmark/tile_torch/results/m1-max-20260904-tirx-views-plan.json \
  --candidate scripts/benchmark/tile_torch/results/m1-max-20260904-tirx-views-plan.json \
  --native /tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/benchmark_tile_tirx \
  --candidate-subgroup-fences elide --capture-sources \
  --compiler-artifact /tmp/luisa-tvm-mpp.VaKmzx/build/lib/libtvm_compiler.dylib \
  --compiler-artifact /tmp/luisa-tvm-mpp.VaKmzx/build/lib/libtvm_runtime.dylib \
  --compiler-artifact /tmp/luisa-tvm-mpp.VaKmzx/build/lib/libtvm_runtime_metal.dylib \
  --compiler-artifact /tmp/luisa-tvm-mpp.VaKmzx/build/lib/libtvm_ffi.dylib \
  --rounds 4 --samples 7 --sample-ms 30 --warmup-ms 200 \
  --output /tmp/tile-subgroup-fence-replay
```

Use a new output directory. This reproduces the two policies, not the exact
historical executable bytes; the latter are identified by the recorded hashes.
