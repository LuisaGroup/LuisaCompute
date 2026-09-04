# Full-vector guarded views versus Torch and Accelerate

The guard repair does **not** close the CPU library gap. At 1024³ the current
Tile implementation measures **5.919 ms**, versus **1.021 ms Torch** and
**1.028 ms direct Accelerate BLAS**; paired median ratios are 5.769× and
5.747×. All eight cases remain slower than both libraries.

| M×N×K | Tile µs | Torch µs | BLAS µs | Paired Tile/Torch |
|---|---:|---:|---:|---:|
| 32³ | 4.721 | 0.876 | 0.348 | 5.366 |
| 128³ | 17.899 | 4.826 | 4.182 | 3.725 |
| 512³ | 733.980 | 146.263 | 138.193 | 5.090 |
| 1024³ | 5919.062 | 1020.527 | 1027.681 | 5.769 |
| 256×1024×128 | 234.142 | 71.485 | 67.019 | 3.339 |
| 1024×128×256 | 172.539 | 65.562 | 62.591 | 2.621 |
| 127×193×61 | 38.110 | 6.645 | 5.908 | 5.780 |
| 513×257×129 | 282.882 | 45.868 | 43.100 | 6.203 |

Times are medians of six per-round p50s; ratios are paired medians. These
fresh library comparisons do not replace the separate
[frozen-binary A/B](../m1-max-20260905-cpu-guards-replay/notes.md), which measures
the actual lowering change. Differences from historical report medians are
not treated as causal speedups or regressions.

## Method and evidence

- Six rounds cover all six Tile/Torch/BLAS orders for each shape, with rotating
  shape order. No parameter search, minimum-of-round selection, or discarded
  slow round. Seven samples target 30 ms after 200 ms warmup, eight requested
  CPU threads. Thread requests are not measurements of library worker counts.
- All Tile cases use 4×16×32 worker tiles, window 2, input forwarding,
  automatic vectorization, 8192-byte stack budget and 64 logical pack lanes.
  The same frozen plan supplies parameters only; its old scores are unused.
- Compact FP32 row-major C=A×B, alpha=1/beta=0, no transposes, caller-side
  prepacking or hidden library replacement in Tile. Warm device-resident
  host-wall timing includes dispatch and internal temporary handling; JIT,
  allocation/upload and cold-call phases are separate. No concurrent build,
  test or profiler.
- **144/144 complete outputs valid**, 33,798,528 checked elements, maximum
  absolute error zero against the FP64 oracle on deterministic dyadic inputs.
  Non-dyadic and ordered-math coverage is in the C++ regression suite.
- All **20** executable/library paths matched before/after hashes and were
  independently rehashed after the run. All **48** raw LLVM hashes and all
  six implementation-order permutations per case were independently checked.
- Torch 2.14.0 reports `BLAS_INFO=accelerate`; the independent system binary
  calls classic LP64 `cblas_sgemm`. These identify APIs/build configuration,
  not an undocumented internal instruction set or microkernel.
- Apple M1 Max/macOS 26.6.2. Native commands, compiler/runtime identities,
  flags, raw samples and paired BLAS ranges are in [results.json](results.json)
  and [results.md](results.md). Final dual-build CTest logs and the two
  pre-existing Metal fence assertion failures are retained in the A/B report.

Next structural checks include CPU target-specific machine code, register
realization, cache reuse across logical programs and task granularity. The
archived LLVM currently says `target-cpu="generic"`; this is an observed
configuration, **not proof** that selecting a CPU model closes the gap. The
`target`/`host` distinction in the bridge must be tested before attributing
performance to a requested ISA. General materialization choices still need
measured cost evidence, not inference from a forwarding flag alone.
