# Multi-output pointwise fusion: protocol

Question: does a shared pointwise producer followed by separate output domains
retain a serial program-per-worker realization, and can dependency-checked
fusion remove that boundary without operator-name or shape dispatch?

The initial compiler is `81c2fac05`; benchmark-only additions provide sigmoid
and tanh-approximate GELU value/derivative pairs with separate native output
parameters. Both outputs are concatenated only after timing for complete FP64
validation. Torch preallocates both outputs; sigmoid uses three eager out
operations, GELU uses forward/backward out operations with a preallocated
unit gradient. This is not a compiled fused-Torch comparison.

First retain the old automatic mapper at 1x127 and 37x1537, then compare it to
the explicit reference mapper. After compiler correctness checks, freeze the
existing 1x256 element block and automatic thread policy, with no score fit or
parameter search. Replay both activation graphs at 1x127, 37x1537,
1024x4096 and 4096x4096. Keep single-output Add/GELU and row-reduction controls
separate; their unchanged source is not a new compiler speedup.

Performance acceptance requires balanced implementation/case order, fresh
capture/JIT per replay, complete outputs, raw samples, source and loaded
artifact hashes, and both GPU and E2E measurements. No-counter GPU
command-buffer intervals include dispatch/gaps; compute-pass counters are
separately instrumented diagnostics. Desktop activity is not isolated.

Semantic tests must cover interleaved producers/stores, more than two outputs,
nonzero minima, ragged/negative input coordinates, conditional stores and
reordered output coordinates. Same-buffer overlapping stores, a written
buffer later read by another domain, explicit resource/binding constraints,
neighbor reads and differing local domains must retain the reference path.
The noalias parameter contract remains mandatory. Reduction and CPU
correctness controls must still pass; this change alone makes no claim of
reduction, native-MPP or direct-XIR performance parity.
