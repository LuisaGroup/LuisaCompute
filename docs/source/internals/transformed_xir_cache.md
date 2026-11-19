# Transformed XIR attached to AST functions

Coroutine splitting and the shared AutoDiff-to-AST helper can retain their
normalized XIR alongside the AST they already produce. AST-only backends keep
using the AST. XIR backends keep their existing `ast_to_xir_translate` entry
point: its recursive function discovery imports a cached continuation even
when the entry kernel and scheduler wrappers are newly recorded ASTs.

## Ownership and isolation

`FunctionBuilder` owns an opaque, immutable cache handle. The AST library only
stores a shared pointer and a fingerprint; it does not link against XIR.
`XIR2ASTConfig::cache_transformed_functions` opts a producer into retention and
requires its input module to remain immutable until translation finalization.
The producer copies the reachable functions and their globals into one shared
private snapshot per translation context/batch. Shared dependencies are copied
once. The original transformed module may then be destroyed.

Each AST-to-XIR context selectively clones the requested closure into its own
module. Function, argument, block, instruction, constant, undefined-value and
special-register operands are all remapped; metadata is copied. Repeated
imports in one context reuse definitions. Different contexts never share
mutable IR. Backend-specific optimization, inlining, fast-math and lowering
still run on the consumer's private module, not the snapshot. There is no
backend-specific or disk-cache state in the retained representation.

The cloner is intentionally restricted to the verified, PHI-free XIR-to-AST
boundary. It is not a general cloning API for arbitrary intermediate SSA.

## Conservative misses and identity

The cache is attached before the AST builder is hashed and published. Its
identity includes a fingerprint of the immutable source module (computed once
per batch) and the selected function's ordinal, in addition to the AST hash.
This prevents metadata-rich preserved XIR from aliasing an AST-only generated
shader. Uncached functions retain their existing hash algorithm and values.
Including the whole source module is conservative: unrelated source changes
can cause an extra cache miss, never reuse a different transformed function.

Bound-resource argument sorting/deduplication can change the XIR signature.
This initial implementation deliberately does not attach snapshots when any
`XIR2ASTConfig::bound_arguments` entry is bound, or when the emitted AST's
argument types/reference modes differ from the source. These functions use the
ordinary AST path. Coroutine continuations use explicit arguments; their
scheduler/capture wrappers remain AST and can still import the cached bodies.
Unbound AutoDiff kernels retain XIR; bound AutoDiff kernels fall back. The
classic Metal/DX AutoDiff helper remains an AST consumer, so retention there
is not itself a claim of immediate backend speedup.

Legacy inline-ray-query translation and raster-stage configurations bypass
snapshots. AST rewrites/duplication do not automatically copy a cache to a
changed body. Normal AST lowering remains the correctness fallback.

## Diagnostics and tests

`AST2XIRConfig::use_transformed_function_cache = false` is an explicit oracle.
Optional `AST2XIRTranslationStatistics` counts AST body translations, cached
imports and same-context reuse without shared global counters.

* `test_xir_cache_clone`: selective dependency closure, callback-only callees,
  forward references, globals/metadata, source lifetime and mutation isolation.
* `test_ast_xir_cache`: source destruction, zero AST lowering on hits, repeated
  and concurrent imports, hash identity, configuration misses, bound-argument
  sorting/dedup and FunctionDuplicator fallback, and the real AutoDiff-to-AST
  handoff.
* `test_coro_xir_cache`: the actual coroutine compiler with two suspends, three
  continuation bodies, fresh scheduler wrappers and an ordinary callable.

`LUISA_XIR_CACHE_BENCHMARK=1 test_coro_xir_cache` additionally prints paired
30-iteration AST-to-XIR timings and counters. Every iteration owns a fresh
module; no shader disk cache participates. This measures the consumer handoff,
not total coroutine production, GPU compilation or device execution. Snapshot
creation adds one producer-side clone, so end-to-end savings are workload-
dependent and should not be inferred from the bypass counter alone.

When the fallback backend is available, `test_coro_xir_cache fallback` (also
registered as `test_coro_xir_cache_fallback` in CTest) compiles and executes both
cached and duplicated/uncached scheduler kernels with shader disk caching off.
It checks coroutine and ordinary callable numerical results against a host
oracle, then compares the two output buffers exactly.

`test_ast_xir_cache fallback` similarly executes the shared AutoDiff-to-AST
handoff with cached and uncached kernels, checking the derivative of `x*x`
against `2*x` for five scalar inputs.
