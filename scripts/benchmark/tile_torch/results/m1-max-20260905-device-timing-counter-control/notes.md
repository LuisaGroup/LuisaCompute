# GPU timing observer audit and integration checkpoint

September 5, 2026; Apple M1 Max; macOS 26.6.2; FP32; PyTorch 2.14.0.
**Assessment: share as a measurement-method audit, not as a stable performance
ranking or new cost-model calibration.**

## Outcome

The initial compute-pass helper is numerically transparent, but its timing
probe is not performance-transparent across frameworks. The six-case matched
probe/control run measures these median paired GPU command-buffer ratios:

| Torch operator | Shape | Counter / no-counter GPU duration |
|---|---|---:|
| softmax | 17×257 | 5.854× |
| softmax | 64×4096 | 3.079× |
| RMSNorm | 17×257 | 0.957× |
| RMSNorm | 64×4096 | 1.076× |

These ratios compare seven alternating-order sample pairs at the **same 64
repetitions**, not different host/device batch sizes. They compare the full
probe path with a commit-only observer; they do not isolate the physical cost
of a timestamp write from other scheduling effects. No subtraction or scaling
is used to fabricate a corrected kernel time. Ratios below one and outliers
also show that noise/contention has not disappeared.

There is substantial background-load variability: a post-run process snapshot
showed WindowServer at about 65% CPU and another desktop application around
44% CPU. That is not a simultaneous GPU-utilization measurement and cannot
attribute individual outliers. No user application was stopped. This cohort
does not justify a claim that native beats Torch/MPS or a change to planner
coefficients. Low-load, position-balanced performance replay remains pending.

## Three distinct scopes

| Measurement | Instrumentation | Includes | Does not establish |
|---|---|---|---|
| Host E2E | None | Host binding/encoding/submission and final synchronization | Pure GPU/kernel time |
| GPU command-buffer control | Commit-only observer; no encoder hooks or counter attachments | GPU start/end interval of every completed buffer, including GPU work, blits and in-buffer gaps | Individual kernel duration or absence of GPU contention |
| Compute-pass probe | Public encoder factory + commit observer, pass-boundary counters | Instrumented GPU compute-pass intervals | Uninstrumented performance or per-dispatch timing inside a multi-dispatch pass |

The helper ABI is now 2; the C result layout remains unchanged. A new
`luisa_metal_timing_begin_control()` entry point uses completed
`GPUStartTime`/`GPUEndTime` only. It never hooks an encoder factory. Both paths
restore their method implementations on end/failure. The same real framework
dispatches and buffers are used, without source replay, private framework ABI,
Runtime changes or replacement of the pinned TVMx installation.

Native/PyTorch device phases alternate control/probe order for both batch and
single-call measurements. Raw JSON retains all control samples and derived
paired ratios. Main reports show the no-counter control alongside E2E, with
instrumented pass times as a separate diagnostic table. Frozen replay reports
require complete control pairs before showing GPU statistics and use medians
of paired round ratios, not the ratio of displayed medians or lucky minima.
JIT selection continues to use the documented host-wall objective.

## Integration coverage

- Initial wider [reduction probe](../m1-max-20260905-device-timing-reduction-cohort/results.json):
  24 shapes/operators, 48 complete native/Torch outputs valid. This ABI-1 run
  exposed the anomaly and has no no-counter control. Its timings are **not**
  a speed ranking.
- [Current six-case control](results.json): sum, softmax and RMSNorm at 17×257
  and 64×4096; **12/12 complete outputs valid**. Seven samples, 30 ms host batch
  target, 200 ms warmup. Both counter and control phases use 64 repetitions.
- [Controlled GEMM integration](../m1-max-20260905-device-timing-gemm-control/results.json):
  32³, 128³ and 127×193×61; TIRx SIMD-group, eager Torch and direct MPS;
  **9/9 complete outputs valid**. The generated report includes MPS GPU/E2E
  control times; it does not substitute MPSGraph for MPSMatrixMultiplication.
- [Native-MPP control](../m1-max-20260905-device-timing-native-control/results.json):
  the same three GEMM shapes through TileIR → Metal backend → MPP → Luisa
  Runtime; **3/3 complete FP64-oracle-valid outputs**, all fingerprinted
  artifacts unchanged. Fixed 32×32 tile, four-subgroup operation/group,
  dynamic K, inline tensors and fast math disabled. This independently run
  smoke is not a position-balanced comparison against the preceding run.
- An [explicit TIRx→MPP attempt](../m1-max-20260905-device-timing-gemm-smoke/results.json)
  with the standard unpatched TVMx build was
  rejected for missing MPP memory-contract v2, before timing. The historical
  MPP performance report used the separately patched compiler documented in
  `src/tile/bridge/tirx/patches/README.md`. Neither installation was replaced.
  Native MPP does not depend on that TVMx extension.

All host timings exclude setup/transfers/JIT, retain each framework's documented
output-allocation policy, and use device-resident inputs. Full result buffers,
not just samples, are validated. The raw `valid` flag means correctness and
measurement-schema coverage, **not** that environmental noise is acceptable.

## Reproduction and tests

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output NEW_EMPTY_DIRECTORY --backends metal \
  --operations sum,softmax,rmsnorm --row-shapes 17x257,64x4096 \
  --metal-subgroup-reductions --samples 7 --sample-ms 30 --warmup-ms 200 \
  --metal-device-timing cmake-build-tirx/bin/libluisa-benchmark-metal-timing.dylib \
  --capture-sources
```

For the controlled GEMM run, use `--operations gemm --quick --execution-scope
group --cooperative-matrix --matrix-realization simdgroup --gemm-block 32,32,32
--system-baseline cmake-build-tirx/bin/benchmark_tile_system`, omitting the
reduction/row flags. Native-MPP commands and schedules are retained verbatim
in its raw report; that smoke reused `compare_lowerings.measure` and its full
`compare_mpp.oracle`/`validate_output` checks.

After the ABI/control changes:

- Full selected CMake tree build passed; the subsequent affected/focused
  [CTest run](tests.log) passed **7/7**: timing helper, native Runtime,
  system CPU/Metal, TIRx execution CPU/Metal, and planner.
- Python benchmark contracts passed **80/80**, including no-counter scope,
  missing/invalid control rejection, MPS report coverage, paired-ratio math
  and withholding incomplete replay statistics.
- Project clangd checks passed for all five affected benchmark/test/helper
  translation units; formatting and maintained-source diff checks passed.
- The Sphinx HTML update passed with two existing unrelated toctree warnings.
  Independent recomputation from raw samples agrees with every published
  probe/control ratio; complete output counts and scope flags were checked.
- The earlier complete Tile regression remains the **33/33** submitted-source
  checkpoint in [the preceding validation note](../m1-max-20260905-dual-timing-validation/notes.md).
  No whole-repository pass is claimed. The user's unowned `mem_flags(2)` edit
  remains untouched throughout this observer-audit follow-up. These diagnostic
  binaries were built from that dirty worktree, including the local barrier
  value, not from a byte-for-byte clean submitted checkout. The edit is not
  staged; raw reports record the worktree state and actual binary hashes.

No coefficient, candidate winner or kernel-source change was selected from
these noisy diagnostics. The next optimization target remains reduction
execution mapping, with GPU and E2E evidence kept separate.
