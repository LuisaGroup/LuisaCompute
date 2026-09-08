# Closed matrix epilogues: legality is generic; profitability is not yet

September 7, 2026, Apple M1 Max, FP32. The new optional TIRx→MPP realization
removes compiler-owned same-element epilogue storage without recognizing
activation names. It remains **off by default**. Fixed schedules improve
ragged ReLU/GELU GPU time by median 15.54%/14.41%, but 4096³ GELU regresses
35.14% GPU and 32.42% batched E2E in four paired rounds. This is not general
MPS/Torch parity or a CPU/SIMD/native-MPP improvement.

The reader-facing explanation belongs in the existing Sphinx
[matrix internals](../../../../../docs/source/internals/tile/matrix.md),
[current status](../../../../../docs/source/performance/tile/index.md) and
[route results](../../../../../docs/source/performance/tile/results.md).
This directory is their reproducibility record, not another documentation
hierarchy. The status page is shortened to current route conclusions;
historical timings remain with the detailed route/validation records.

## Contract and attribution

The bridge proves a closed scalar DAG over one MMA accumulator element and
literal constants. Dominating, compiler-owned pure Tile producers become
scalar bindings; manual storage, free variables, extra memory operands,
neighbor/transposed reads and additional observations retain fallback.
Final layout/bounds/store-order proofs remain separate. Pure operations use
typed TIRx arithmetic/calls, not `gemm_relu`/`gemm_gelu` production intrinsics.
Those names exist only to select benchmark graphs.

The optional fifth native TVM patch supplies typed MPP capacity, validity,
element load and element store operations with pure/read-state/write effects.
They operate on a declared FP32 fragment; they neither allocate nor synchronize
or infer lane-to-matrix layouts. The planner removes proved storage bytes but
does not erase scalar math cost. No coefficient was fitted or default enabled.
TVM may still expand bound scalar expressions later; instruction reuse and
live-state/spill costs are open, not established causes of the timing losses.

## Protocol and evidence

[Protocol](protocol.md) fixes 64×64×4096 blocks, 256 workers, pipeline window 1,
default cost coefficients and MPP read-only views for every matrix graph.
The same final compiler differs only in `fuse_matrix_epilogues`. Four rounds
cover both variant orders and both native/Torch orders, at seven samples,
20 ms host sample windows and 100 ms warmup. Neither per-shape schedule search
nor selected minima appear in the comparison. Device activity is not isolated.

The graphs are GEMM; `max(v, 0)`; and tanh-GELU of
`v = 0.125 * (A @ B) + 0.25`. The driver checks every output element against
an FP64 expression at atol=rtol=1e-4. Torch uses preallocated output/intermediate
storage and eager `mm.out`, in-place scale/shift and activation.out. It is not
compiled fused Torch or MPSGraph. Plain MPS/BLAS GEMM is not substituted for
a matching fused-expression baseline.

The [replay](replay/results.json) retains all 144 rows, raw GPU/host samples,
plans, source hashes and JIT/setup times. Its [independent audit](replay/audit.json)
checks 288 complete-output receipts covering 882,766,800 elements, 72 pairs,
30 generated sources and unchanged artifacts. It independently reconstructs
GPU divisors, sample counts and medians. It audits recorded complete checks;
the scalar maximum-error receipt is not a replacement for the driver's
elementwise tolerance test. [Six timing views](replay/results.md) retain
all shapes and rounds through paired medians/ranges and slower counts.

Primary GPU timing is the no-counter command-buffer interval, including work
and intra-buffer gaps, not isolated kernel time. Instrumented compute-pass
intervals are separate diagnostics. Batched E2E and single-call E2E remain
distinct. Ratios are medians of within-round pairs, not ratios of pooled
medians, confidence intervals or cumulative gains across experiments.

Six plain-GEMM controls have byte-identical enabled/disabled source; their
individual GPU ratios still span 0.790–1.049. The identical 4096³ source
favors the enabled arm in all four GPU rounds. ReLU/GELU's ragged GPU/Torch
median ratios are approximately 0.976 but lose two rounds each. Both larger
squares lose to Torch. Tiny ReLU improves GPU batch time without improving
batched E2E. All negative/slow rows remain; no cost fit or broad ranking follows.

The [default-off controls](controls/results.json) compare a frozen pre-extension
binary/bridge/TVM stack with the final disabled path. [Source audit](controls/audit.json)
validates 56 complete native/Torch outputs (630,684 element checks): ten Metal
sources match byte-for-byte, and four CPU sources differ only in bijective
TBAA object-address labels. Instructions, alias graph and width/offset suffixes
are unchanged. The nonmatrix controls are Add, paired GELU, sum and softmax;
they are compatibility checks, not new reduction or CPU performance claims.

## Failed probes and verification boundary

[Initial baseline](baseline/results.json) and [initial experimental compiler](fused/results.json)
each retain 18 attempted rows, including three ragged runtime failures at the
pre-existing automatic 1024-worker binding. These sequential developmental
screens are not causal acceptance evidence. Baseline ragged Metal source
compiles independently, and 256 workers run successfully; the complete fixed
replay validates the latter numerically. This does not repair auto-1024 or
establish the pipeline/resource failure's cause. Do not discard those failures
or promote a universal 256-worker policy.

[Initial full check](check/results.json) and [final full check](final-check/results.json)
retain complete build/test logs. Both have 17/19 passing invocations, with
5,565 Metal matrix assertions, 14,411 planner assertions and 1,054,543
assertions across passing suites. Matrix tests include ordinary clamp,
polynomial and GELU expressions, two non-dyadic input sets, masks, tails,
transposed/offset destinations, nonzero state, zero-K, observations and manual
memory. Raw ABI tests exercise nonzero fragment ordinals and invalid types.

The two red suites are the existing Metal memory/cooperative source-string
checks (two plus one failed assertions). Both measured stacks include the
pre-existing user edit `metal::mem_flags(3)` → `metal::mem_flags(2)`; it is
excluded from the epilogue commit. [Check audit](check/audit.json) and
[final check audit](final-check/audit.json) explicitly preserve this boundary.
No all-green whole-worktree claim is made. The post-replay native source edit
only updates benchmark usage text; the final build/check has separate hashes.
All [105 benchmark Python tests](python-tests.json) pass, including the new
graph/oracle, preallocation, case identity and backward-compatible CLI checks.

## Documentation QA

The existing Sphinx surface was checked in headless Chrome 152.0.7977.76 at
1280- and 390-pixel widths. All eight section screenshots were visually
inspected: status text, result table, ownership diagram and verification
boundary remain readable. Narrow tables/code scroll inside the theme without
page-level overflow; a separate right-edge screenshot checks access to the
last result column. [Browser receipts](docs-qa.json) retain dimensions and
the absence of page-script errors. Screenshots are local review artifacts.

The local checker passes 48 HTML pages, 3,718 local links/assets and 199
compatibility anchors. The strict fresh Sphinx build still exits 1 with ten
missing-Doxygen-XML warnings in the existing API reference: Doxygen/XML are
unavailable here. No warnings were suppressed and no full API-build success
is claimed. The existing toctree and published section anchors are preserved.

## Reproduce and audit

The native dependencies use TVM `c7b458e946bc4266915da582457476bdcd9705ae`
and TVM FFI `12dbf053b3d9ba4ebd9da3123b1aeca79cf74229`, built with Metal and
LLVM enabled, MLIR disabled. The four existing MPP patches precede
`metal-mpp-element-v1.patch`; see the
[patch instructions](../../../../../src/tile/bridge/tirx/patches/README.md).
The replay records Torch 2.14.0 (`08187d9e0fba026dc8217405802ab5381dc88d90`),
Clang version, exact native commands and compiler/library hashes. The frozen
baseline includes its own ABI-coherent Luisa bridge and TVM libraries.

```sh
uv run --offline --no-project --python 3.13 --with numpy --with torch python \
  scripts/benchmark/tile_torch/results/m1-max-20260907-fragment-epilogue/experiment.py \
  --phase replay --build /path/to/luisa-build --tvm-lib /path/to/tvm/lib \
  --output /path/to/new-empty-replay
uv run --offline --no-project --python 3.13 python \
  scripts/benchmark/tile_torch/results/m1-max-20260907-fragment-epilogue/audit.py \
  replay --directory /path/to/new-empty-replay
```

The experiment builds the complete selected configuration before native runs,
fingerprints artifacts before/after, and retains failed rows. Do not run other
benchmarks, builds or profilers during timing. `controls` additionally needs
`--baseline` pointing to a frozen pre-extension executable/library directory.
The auditor deliberately corrupts cohorts, output counts/tolerances, sources,
artifact stability, GPU divisors, order, worker counts and scalar work; every
corruption must fail. Audit outputs describe the evidence, never tune a policy.
