# Bounded direct MPP output — protocol fixed before implementation

The current compiler rejects direct accumulator output whenever the destination
copy has a nontrivial bounds guard. Even with immutable direct A/B views, this
retains a shared C tile and group synchronization. Test whether composing a
proved rectangular output domain with the subgroup's cooperative coordinates
can eliminate that resource without changing the Tile DSL or math policy.

The admission rule must be based on affine projections, guards, ownership and
closed recurrence observations, never an operator name or benchmark size.
Existing fully in-bounds stores retain their ABI/code generation. A separately
versioned optional TVM capability may admit a bounded destination rectangle;
older TVM, arbitrary masks, negative origins, manual/observed carry storage and
unproved layouts must keep the existing realization. Empty subgroup outputs
must not construct an invalid pointer or write an element. Input and output
rectangles are independent: a padded input does not authorize dropping output.

Before editing, a full selected Luisa build completed successfully. The old
benchmark, Tile libraries, TVM libraries and three external TVM source files
are frozen in `/tmp/luisa-mpp-store-baseline.u37ixE`. The selected build is
`/tmp/luisa-tvm-mpp.VaKmzx/luisa-build`; patched TVM is in the adjacent `build`.
The user-owned barrier edit in `cooperative.cpp` must remain unchanged and
must be present in both compiler variants.

## Correctness gates

- Low-level native TIRx stores: full/partial/empty M and N; row/column-major
  output; distinct output offsets and sentinels; static and dynamic lengths;
  non-dyadic, nonfinite and signed-zero payloads; malformed typed ABI rejection.
- High-level recurrence: aligned and ragged tiles; small physical dimensions;
  several subgroup rectangles; one/multiple K steps; nonzero initializers;
  transposed output; exact output bounds and untouched sentinel regions.
- Rejection: negative origins, extra masks, manual/observed accumulators,
  alternate output uses and missing extension capability. CPU and ordinary
  SIMD-group paths remain unaffected.
- A complete selected-tree build precedes every native test/benchmark phase;
  no measurements overlap compiler builds. Full output checks, not sample
  checks, are required before accepting timings.

## Performance comparison

Use FP32 preallocated outputs, fixed 128 threads, copy batch 1, pipeline
window 1, and the same logical `128x32` Tile shape. Use a single `BK=4096`
candidate for both variants, without retuning on measured results. Six shapes:
`129x257x61`, `1025x1025x1024`, `2049x4097x1025`, `4097x4097x4096`,
`1024x1024x1024`, `4096x4096x4096`. The last two are unchanged aligned controls.
If a fixed request rejects, retain the rejection rather than substituting a
different schedule. Compare old/new/new/old in a pilot, then six independent
fresh-JIT paired rounds with reversed/rotated case and variant ordering.

Record nine samples, 30 ms calibrated sample windows, 100 ms warmup, exact
commands, compiler hashes and generated source. Keep batched/single-call E2E,
no-counter command-buffer GPU time and instrumented compute-pass diagnostics
separate. The no-counter interval includes work and command-buffer gaps; it is
not isolated pure kernel time. Torch and direct MPS comparisons retain their
fusion, fast-math, allocation and timing differences. Do not fit cost-model
coefficients, promote a new default or claim cross-device parity from this
cohort. Structural code-generation change and measured profitability are
separate acceptance questions.

## Correctness pilot diagnoses (before timing the candidate)

The first complete matrix run failed. The arbitrary-mask fixture used a
predicated BufferStore, unsupported by native Metal/CPU code generation;
represent the same even-column mask as an IfThenElse with bitwise parity.
The MPP zero-K fixture also retained an unsupported 8x24 local rectangle;
use a legal 16x24 rectangle without removing zero-K semantic coverage.
Add case/IR/source diagnostics for the ragged transposed-output admission and
two zero-shared-byte assertions before deciding whether those fail in the
implementation, the proof, or the expectation. No legality predicate is
relaxed. The original pilot logs are retained; HEAD 9ed989af2 plus the bounded
store worktree diff is the rollback reference, excluding user-owned edits.

The diagnostic rerun confirmed that the unsupported fixtures were fixed.
The remaining shared allocations were exactly the old-output history snapshot
(1536/2048 bytes), not C, and must remain. Transposed 37x37 output had identical
bounds in a different conjunction order. Add bidirectional matching of proved
equivalent nontrivial conjuncts, preserving the general fallback for unmatched
clauses; assert the required history resource instead of zero total resources.
