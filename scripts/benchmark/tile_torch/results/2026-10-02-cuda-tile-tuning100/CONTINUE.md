# Shutdown handoff — 2026-10-02

The user requested that existing results be committed before shutdown, with further tuning deferred until tomorrow. All compiler/GPU jobs have finished. Measured production implementation is `bca9b9c9de7bc20fc2e5601ba325644fb783f788`; this subsequent commit records results and qualifies Tile Function/Type names to fix XMake PCH compilation. MSVC syntax-only compilation with the exact CUDA PCH header reproduced the ambiguity before the fix and passed afterward. The qualification has not been followed by a full rebuild or GPU run; existing build receipts still identify the measured implementation. Do not treat the ignored drafts below as integrated or device-validated.

## Completed and retained locally

- Full MSVC CMake builds of both LLVM 22/23, previous native/TIRX/SIMD/alignment/memcheck/PTX/OptiX regression checks, and the exact repaired-fixture/U32 recheck. Combined evidence: `.deps/oct01-tile-bundle3fix-resolved-v1/results.json`. Historical full-native runs used the earlier DLL; the composite explicitly separates them from the later focused checks.
- All eight production cohorts passed: `.deps/oct01-tile-bundle3-{pointwise12,packed14,sort3,alignment18-off,alignment18-on,embedding24,matrix9,mha2}-v2`. Queue journal: `.deps/oct01-tile-bundle3-matrix-queue-v1.json`.
- Worker8 `.deps/oct01-aligned-entry-worker8-runtime-v2/results.json`: correct but slower. Its v2 validation plan/parser preserves the old offline outputs. Do not adopt this hint based only on lower register count.
- CuTe `.deps/oct01-cute-gemm-{offline,runtime}-v2`: NVCC and full paired GPU validation passed. Source/plan prep: `.deps/cute-gemm-diagnostic-v2-prep`; helper/runtime in `.deps/cute-gemm-diagnostic-prep`. NVCC flags and pinned headers are in the plan.
- `.deps/cute-nvrtc-probe-v1`: compiled the same CuTe template with only cuda::std type-traits changes; original options retain device-as-default-execution-space. CUBIN/PTX produced, not GPU-tested. v2 removed that flag and failed with 62 host-function-in-JIT errors. Keep both records. Future intended production route is existing CUDACompiler NVRTC→PTX→Driver, so test that actual PTX route before integration.
- Lightweight decision notes: `.deps/bundle3-performance-review-prep/decision-note.md`.

## Frozen, unapplied candidates

| Candidate | Ignored path | State |
| --- | --- | --- |
| BF16 UINT32 stable packed sorting | `.deps/packed-bf16-prep/integration.patch` | SHA256 `d6c9e4f0514ab7ed6e782031d8a72a56e812c8915da57466cc915d3ab21e7bbc`; two reviews and 86 Python tests passed, no C++ build/GPU |
| Residual RMSNorm and FP32 GEMM+bias | `.deps/tile-fused-epilogues-prep/integration.patch` | SHA256 `62e4754ebc9d40d96f2ca551c32fe085195ed099a4207ac0f1d39c10e05fd69c`; reviewed, 88 host tests, no C++ build/GPU |
| Sort chunks1024/2048 | `.deps/chunked-sort-large-chunks-prep/integration.patch` | SHA256 `ddd019f1ac3b74d3daf7bbedec04ad48df3395328d2678c3f5027d95d8a18c73`; reviewed, 90 host tests, no C++ build/GPU |
| Strict scan chunks1024/2048, FP32 carry | `.deps/native-tile-chunked-scan-prep/integration.patch` | SHA256 `12b89cf121fc918c482758dd0bca478eae3bf5e700d2c3f2d923f86ff269073c`; 88 host tests and CPU models passed; final review/device work pending |
| Canonical CuTe SSA matcher/source generator | `.deps/cute-production-codegen-prep/integration.patch` | SHA256 `1b5acc66108183a3cca107b2662934111ab6adf532c28592ad4694e8819ebcb4`; unfinished, unreviewed/uncompiled draft with three new private/test files |

The first four patches overlap in benchmark whitelists/helpers. Merge focused hunks; do not copy whole shadow files over each other. Preserve every existing oracle and default. Proposed new benchmark inventories and negative/typed tests are in each prep directory.

CuTe design: `.deps/canonical-cute-route-prep/DESIGN.md`. Runtime/compiler/CMake integration has not been implemented. A CuTe entry requires block128, while native Tile uses logical block1: select a complete launch record (function/grid/block/shared) consistently in live launch and graph creation/update. Keep original Tile module and fallback, check actual pointers/ranges/alias, and expose the actual selected variant for tests rather than relabeling fallback source. Root owns runtime/CMake work; tile_probe only prepared matcher/source/test drafts.

## Next sequence

1. Validate NVRTC-produced PTX with the original typed Driver packets/protocol. The adapter has not been written; inspect `.deps/cutout-measurement-shutdown-handoff/README.md` before continuing. Do not claim NVCC results validate NVRTC output.
2. Optional extra pointwise schedule inventory: `.deps/pointwise-next12-prep/cases12.json` (not run). Refresh both full builds and receipts after the PCH qualification before invoking project executables.
3. Optionally inspect actual production code with `.deps/production-tile-offline-audit-prep/audit.py`; prepared/reviewed, never executed. It recompiles artifacts offline and must not be described as capturing the runtime cubin.
4. Review/merge selected candidates, then configure and run full `cmake --build` for BOTH `build-msvc-llvm` and `build-msvc-llvm23` before invoking project executables. Use MSVC and system CUDA; never call Ninja directly. Existing build script `.deps/build-tile-bundle3fix-root-v1.ps1` is a template; use new log names and refresh markers only after successful full builds.
5. Run exact new native groups, portable BF16 tests on SIMD/TIRX, memcheck, and appropriate graph/PTX/OptiX sentinels. Fused focused runner is `.deps/fused-epilogues-validation-prep/Invoke-Fused.ps1`. Then fresh production comparisons, retaining failed experiments and per-contract boundaries.
6. Run `python scripts/check_cpp_no_throw.py`, inspect/stage the exact diff, and commit/push promptly. User already authorized pushing to remote. GitHub API tree/commit/ref workflow is available when SSH push is blocked; verify remote tree before a soft synchronization.

Root serializes GPU work and heavy compilation. Keep existing PTX/OptiX behavior unaffected. Do not change clocks, power or overclock settings. The task's awake helper is stopped during shutdown cleanup; restart only when continuing actual work.
