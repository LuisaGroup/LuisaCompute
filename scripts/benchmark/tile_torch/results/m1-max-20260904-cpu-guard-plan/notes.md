# Retained negative experiment: legal views scalarize ragged GEMM

This is a single-pass diagnostic run, **not a repeated causal speedup claim**.
Adding the missing Boolean bounds implication allowed the two ragged GEMMs
to forward their immutable A/B inputs. All eight Tile and eight Torch full
outputs passed the FP64 oracle, with maximum absolute error zero, but the
resulting ragged kernels were slow:

| M×N×K | Tile µs | Scalar FP32 load sites | Vector FP32 load sites | Scalar/vector FMA sites |
|---|---:|---:|---:|---:|
| 127×193×61 | 182.324 | 83 | 0 | 79 / 0 |
| 513×257×129 | 1252.341 | 173 | 0 | 142 / 0 |

Counts describe static sites in the archived LLVM, not dynamic instruction
counts or hardware profiling. See the [127×193×61 source](sources/5d06c585fcb5b30f0d187da0be7b45cf5213d0239b0af4556a40aad7ad9f16f8.ll)
and [513×257×129 source](sources/81182dea554706e4e9a0aba0f57e9b50c32fe748c055a9961e1ae99c47155566.ll).
Both sources have scalar branches and scalar FMA calls in the contraction;
removing storage alone did not preserve the contiguous SIMD realization.

The subsequent [full-vector prototype](../m1-max-20260904-cpu-vector-guard-pilot/results.md)
restored vector loads/FMAs and measured 30.070/193.871 µs. Those measurements
were separate single passes; their ratio is not a controlled estimate of the
optimization's benefit. Use the later frozen-binary replay for that purpose.

Both pilots use 4×16×32 tiles, worker binding, pipeline window 2, 8192-byte
stack budget, 64 logical SIMD-pack lanes, input forwarding, and eight CPU
threads. Five samples target 20 ms each after 100 ms warmup. All times are
synchronized device-resident host wall time including dispatch, with JIT and
transfers excluded. No build, test, or profiler ran alongside timing.
Full per-case timings, flags, correctness, and identities are in
[results.json](results.json). The incomplete-guard-proof reference in the
[earlier four-round report](../m1-max-20260904-cpu-views-replay/notes.md)
is a different implementation; its historical times are not paired with this
pilot. This failed realization is retained as evidence that legality and
materialization profitability must be separate decisions.
