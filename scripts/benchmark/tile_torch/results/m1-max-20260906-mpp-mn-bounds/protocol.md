# Bounded M/N memory inputs: semantic gate before performance

Starting point: Luisa `20ccf9744`, pinned TVMx plus the existing MPP v2 and
bounded-K patches. The unrelated local barrier flag 3-to-2 edit remains
unchanged. This experiment does not modify the execution distribution or
precision policy.

## Question and candidate

Can a bounded inline tensor represent the exact positive-zero-padded M/N
inputs of a Tile MMA, including completely inactive subgroup rectangles,
without nominal A/B shared snapshots? Nominal cooperative output shape and
the reduction recurrence must stay unchanged. A partial input is not a
license to discard observable output elements or assume finite operands.

First extend the optional native C++ TVMx memory-input contract in isolation
with actual M/N/K lengths. M/N may be zero, K must be positive and shared by
A/B. Retain the v2 and bounded-K-v1 forms verbatim. No production bridge
selection is enabled until the semantic probe passes.

## Required checks

1. Low-level native TIRx-to-MPP probe: all nominal output elements, including
   padded rows/columns, nonzero C, multiply and multiply-accumulate, transposed
   A/B, zero and partial M/N, and short K.
2. Compare non-dyadic finite inputs to an independently computed FP64 result.
   Also check observable `0 * infinity` and NaN propagation; an empty input
   dimension must not silently replace that arithmetic with zero.
3. Reject negative/oversized lengths, wrong scalar types and invalid leading
   strides before a Metal launch. Dynamic lengths remain proved caller
   preconditions, not claims of runtime validation.
4. If successful, derive canonical padded rectangles in the bridge, retain
   noalias/immutability/dominance checks and validate ragged, multi-iteration,
   transposed and mixed-effect kernels. Preserve all existing fallback tests.
5. Complete full CMake builds before binary tests. Preserve a v2+bounded-K
   compiler and old bridge at `/tmp/luisa-mpp-mn-baseline.gFce8i` for compatibility
   and same-schedule comparison.

Performance is a later gate, not inferred from source reduction or compilation.
Any timing must keep GPU command-buffer and E2E metrics separate, validate
complete outputs, retain fresh MPS/Torch controls, source/library hashes and
all order reversals. No default schedule or cost coefficients are promoted
from a semantic probe or unstable timing.

## Empty-extent diagnostic and revised emitter

The initial 56-contract/168-output probe failed all 72 empty-M/N outputs;
nonempty rectangles passed, including Inf/NaN cases. A diagnostic replay
showed that M=0/N=16, NN, accumulate, finite inputs returned -4.454064 at
(0,0), instead of the unchanged C value 1.1022274494171143. The emitted inline
tensor had extents (7,0), yet the result included its allocated sentinel row.
This establishes that the naive empty-view encoding is not usable here; it
does not establish a general SDK bug or its internal cause.

Revised strategy: use ordinary MPP only for strictly positive actual M/N.
For an empty operand, realize the exact zero-product case directly in the
cooperative output fragments. Read only valid elements of the nonempty
operand. Integer FP32 classification preserves 0*Inf/NaN and signed-zero
addition even under module-level fast-math. It requires no A/B staging and
retains the original nominal output and nonzero C. Keep all empty and
nonfinite tests; do not redefine them as successful rejections.

The first archived `integrated-pilot` invocation used an unsupported wildcard
filter: it ran zero assertions, so its exit-zero receipt is invalid evidence.
The corrected runner requires a positive assertion count and uses an exact
test name (or explicitly runs the entire suite). `integrated-exact` passed
235 assertions; it does not replace the final post-change regression run.

## Predeclared performance comparison

Use `experiment.py` after correctness/compatibility checks. Five FP32 NN GEMMs:
129×257×61, 1025×1025×1024, 2049×4097×1025, 4097×4097×4096, and aligned 1024³.
Both compilers receive the same three requests: BM=128, BN=32, BK=16/1024/4096,
128 threads, window=1, copy batch=1. Record rejected candidates explicitly.
Order is old-forward, new-forward, new-reverse, old-reverse, reversing both
shapes and candidate order. Select by uninstrumented command-buffer GPU time
and perform a fresh selected replay, each with fresh MPS/PyTorch controls.
Use five samples, 20 ms target samples, 100 ms warmup, and complete FP64-oracle
validation. Small-BK common candidates provide a same-schedule comparison;
newly admitted large-BK candidates have no old timing denominator. Aligned
1024³ is a source-identity control. No other performance workloads or builds
are run concurrently. CPU/SIMD and other operators are outside this checkpoint.

Report surface: the existing Sphinx Markdown architecture/results pages,
explicitly requested by the user; no parallel analytics app or new doc tree.
The technical summary, evidence, measurement definitions, method, limitations,
and next questions are split between those pages and these linked source notes.

Preflight observed active desktop/background processes; this is not an
isolated-device session. Keep order reversals and fresh controls, report
variation, and do not attribute it to a particular thermal/cache mechanism.
These are exploratory measurements, not a cost-policy calibration dataset.

The initial root-level `old-forward` run failed dynamic loading of unchanged
Luisa support libraries (`libluisa-osl.dylib`), before code generation. All 15
failures are loader failures, **not** candidate admission or numerical results.
The corrected run lives under `replay/` and appends the current build's support
library directory after the frozen old compiler/bridge directory. All those
support libraries are inventoried as well. The original failure record remains.

## Second emitter, before its measurement

The complete `replay/` ABBA run exposed a strong regression on the two smaller
ragged shapes, despite correct outputs and successful large-BK admission.
The original emitter is retained in `initial-per-element.patch`. Its empty
operand handling rereads each nonempty row/column for every nominal output
element. The follow-up replaces that redundant scan with a subgroup-striped
scan and integer `simd_or`/`simd_and`, then distributes the classification using
the cooperative fragment's public coordinates. It also gives fully in-bounds
subgroups the static-M/N tensor representation. Both branches retain actual K
and all nonfinite/signed-zero checks. These are two simultaneous emitter
changes, so this comparison cannot attribute improvement to one alone.

After a full build and the same complete matrix semantic suite, rerun the
identical ABBA protocol under `cooperative-replay/`, retaining the old compiler
and fresh library controls. Do not replace the first run or treat a rejected
large-BK old candidate as having a timing. This is still a nonisolated,
two-order diagnostic, not production-policy acceptance.

## Third emitter: direct classification distribution

`cooperative-replay/` still regressed on the small ragged case. It traversed
every output fragment once for every nominal row/column. The next emitter
assigns one nonempty row/column to each lane, scans K once, packs its sign and
nonfinite classification into one uint, and shuffles that classification to
each output coordinate. It needs only ceil(max(local M,N)/32) fragment walks.
Shuffles execute in every lane, outside output-validity predicates; no opaque
MPP lane layout is assumed. Keep the static interior branch from emitter two.
The intermediate implementation is preserved in `cooperative-scan.patch`.
Repeat the same semantic gate and identical ABBA protocol in `shuffle-replay/`.
