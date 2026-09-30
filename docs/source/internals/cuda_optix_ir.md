# Experimental CUDA LLVM to OptiX IR

The CUDA backend has an opt-in path from its optimized LLVM module directly to
binary OptiX IR. Ray-tracing shaders can use this path; ordinary CUDA compute
shaders continue through LLVM's PTX emitter. Building the feature does not change
the default shader compiler.

The pipeline is:

```text
DSL / AST -> shared XIR normalization -> LLVM 22 optimization
          -> OptiX IR compatibility lowering
          -> NVVM kernel annotation transplant
          -> in-tree LLVM 7 bitcode writer -> level-2 OptiX IR container
          -> OptiX moduleCreate
```

This is an in-process container writer. It does not invoke CICC, NVRTC, or
libNVVM for this ray-tracing path. The existing AST/NVRTC and LLVM/PTX paths
remain available.

## Build and select the path

Configure with both experimental CMake options enabled. Both default to `OFF`:

```powershell
# Run from an MSVC developer PowerShell. Replace LLVM_DIR with your LLVM 22 package.
cmake -S . -B build-msvc-llvm -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CXX_COMPILER=cl `
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded `
  -DLUISA_COMPUTE_ENABLE_CUDA=ON `
  -DLUISA_COMPUTE_ENABLE_EXPERIMENTAL_CUDA_LLVM_CODEGEN=ON `
  -DLUISA_COMPUTE_ENABLE_EXPERIMENTAL_CUDA_OPTIX_IR=ON `
  -DLLVM_DIR="C:/deps/llvm-22/lib/cmake/llvm"
cmake --build build-msvc-llvm
```

The pinned in-tree downgrader requires **LLVM 22**. It compiles as a static helper
against the same LLVM package used by the CUDA backend; installing LLVM 7 or a
second host LLVM is unnecessary. On Windows, match the LLVM package's CRT and
RTTI configuration. The MSVC package used for bring-up requires `/MT`; CMake
uses `/FI` for the legacy writer shim and matches LLVM's disabled RTTI.
The existing Apple LLVM 14/AIR serialization path retains its own writer.

Set both environment variables **before starting the application**:

```powershell
$env:LUISA_EXPERIMENTAL_LLVM_CODEGEN = "1"
$env:LUISA_CUDA_LLVM_OPTIX_IR = "1"
# Launch the application with its normal CUDA backend arguments.
```

These runtime switches also default to off. They are read when the backend is
loaded, so changing them after device creation is not a supported switch between
formats in one process.

| Runtime selection | Ray-tracing shader | Ordinary compute shader |
|---|---|---|
| Both switches unset | AST/NVRTC to PTX | AST/NVRTC to PTX |
| LLVM codegen `1`, OptiX IR unset | LLVM to PTX | LLVM to PTX |
| Both switches `1` | LLVM to OptiX IR | LLVM to PTX |

The OptiX IR switch alone does not enable LLVM generation. Existing kernels
that require LLVM, such as autodiff kernels, retain their automatic LLVM
selection. Requesting OptiX IR for an LLVM ray-tracing kernel without the CMake
feature produces an explicit error.

## Cache and AOT artifacts

OptiX IR is binary. Preserve its explicit length, including embedded zero bytes;
do not pass it through a text writer or append a PTX terminator. The OptiX API
receives the byte buffer and its exact size. An OptiX IR compilation error is
reported directly and never retried with PTX text-version patching.

Generated cache entries separate PTX from OptiX IR by format and extension.
Metadata serializes `CODE_FORMAT PTX` or `CODE_FORMAT OPTIX_IR`, includes the
format in equality, and validates the code envelope before loading. A legacy
sidecar without `CODE_FORMAT` means PTX. OptiX IR is accepted only with
`RAY_TRACING` metadata.

For AOT generation, use the usual `ShaderOption::compile_only` flow and give
`ShaderOption::name` an explicit `.optixir` extension. Keep the matching
`.optixir.metadata` sidecar. Use an explicit `.ptx` or `.optixir` name when loading
artifacts with the same base name. For backward compatibility, an extensionless
load checks the `.ptx` artifact first and uses `.optixir` only if that PTX file is
absent. Loading follows the saved metadata; the runtime environment switches
select generation, not the interpretation of an existing artifact.

## Compatibility boundary

The container follows the supplied proof of concept's observed CUDA 13.3
level-2 layout. Its format fields and NVIDIA's
`nvvm.annotations_transplanted` marker are not a public cross-version contract.
Success on one driver/CUDA/OptiX combination does not establish compatibility
with another. LLVM 7-compatible bitcode serialization alone also does not prove
that the target OptiX compiler accepts every intrinsic in that module.

The adapter transfers recognized kernel annotation meaning before adding the
consumed marker and keeps the PTX kernel calling convention. Unsupported NVVM
annotation semantics are rejected. Optimization finishes before typed-pointer
reconstruction; running another optimizer between reconstruction and writing
would invalidate the legacy writer's no-op casts.

The compatibility lowering runs after the normal host optimization pipeline
(O2/O3 according to the selected level), only when emitting OptiX IR. LLVM/PTX
does not run these adaptations:

- Floating-point atomic add, subtract, minimum, and maximum become integer
  compare-exchange loops. They retain the original address space, alignment,
  ordering, synchronization scope, volatility, and returned old value. Retry
  compares integer bits, so a stored NaN cannot cause an endless floating-point
  equality retry. The replacement arithmetic has no newly introduced fast-math
  permissions; minimum/maximum retain `minnum`/`maxnum` semantics. Exchange uses
  an integer atomic exchange with bitcasts. The adapter handles 32- and 64-bit
  IR, but the public DSL and current runtime regression cover float32 only.
- Generic `rint` uses rounding-to-nearest-even instructions without flushing
  subnormals; half values pass through float. `fabs` clears the sign bit.
  Signed and unsigned integer min/max use comparisons and selects. Floating
  vector reductions preserve the explicit start value and lane order, carrying
  only the original operation's fast-math permissions.
- Packed boolean mask comparisons with zero or all ones become scalar bitwise
  all/any reductions. This avoids the observed OptiX compiler crash for
  `<3 x i1>` bitcast to `i3`, while retaining poison propagation from each lane.
- 2D/3D zero-boundary surface loads and stores use inline surface instructions
  with explicit memory effects. The adapter handles scalar, two-lane, and
  four-lane operations with 8-, 16-, or 32-bit components. This works around an
  observed compiler failure in the validation environment: a native NVRTC
  OptiX IR probe without ray queries also produced misaligned local/shared
  accesses, while its PTX counterpart passed. The runtime regression validates
  four-channel BYTE4, HALF4, and FLOAT4 images and volumes, including half signed
  zero and exactly representable finite values. Scalar/two-channel execution is
  not covered by that regression.

Before legacy serialization, fast-math flags are also removed from calls that
return aggregates: LLVM 7 does not encode those permissions on aggregate calls.
This removal does not change the call's defined result. The wrapper still
rejects `freeze`, scalable vectors, bfloat16, AMX, and target-extension types;
other unsupported instructions or intrinsics remain an explicit compatibility
boundary. These adaptations and limits apply to OptiX IR, not to LLVM/PTX.

## Initial adapter validation (5b28304a3, LLVM cache v12)

This record describes the initial OptiX IR adapter at commit `5b28304a3`, using
LLVM cache revision **v12**. Its PTX byte-identity and compilation measurements
predate the later shared fast-math FTZ and ray-query payload changes; they are
not byte-identity or performance claims for those subsequent revisions.

The following results were collected on Windows with MSVC/Ninja, LLVM **22.1.8**,
CUDA **13.4.92**, OptiX **9.0**, NVIDIA driver **617.14**, and an **RTX 4060 Laptop
GPU**. This work validates that environment; it does not establish compatibility
with other LLVM versions or change either default-off switch.

| Check | Evidence / result |
|---|---|
| Full MSVC/Ninja build with the opt-in gate | Passed; `.deps/optixir-final-build.log` |
| Focused regression suite | **22/22 passed**: 11 LLVM/PTX tests, 10 OptiX IR tests, and one host metadata test; `.deps/optixir-surface-full.log`, `build-msvc-llvm/logs/optixir-surface-full.xml` |
| Metadata, legacy sidecar, and malformed-envelope tests | Passed: 136 assertions in 16 host cases |
| Generated OptiX IR accepted and executed | Passed; IR tests require an actual `moduleCreate: format=OPTIX_IR` and reject PTX ray-tracing module fallback |
| Ray-query and tracing correctness | Passed: triangle/procedural hits, misses, visibility and build modes, curves, motion, world/object rays, termination, captured state, and mixed filter/general handlers |
| Additional reader compatibility regressions | Passed: float32 atomic results/old values/concurrency, all bool2/3/4 patterns and negations (97 assertions per route), and 2D/3D texture read/write bits (18 assertions per route) |
| Explicit and extensionless AOT load behavior | Passed for PTX and OptiX IR artifacts in the payload-boundary fixture, with the same strict readback oracle as the compiled shader |
| PTX isolation | Passed: PTX tests reject IR generation/consumption; the ordinary-compute cache fixture passes through PTX |
| Cold cutout LLVM/PTX output | Byte-identical to the pre-OptiX-IR baseline: 33,699 bytes, SHA-256 `7e589135663b4b0b67f12a362e18de40d7275cf134a50184378d04fd8e3494e6` |
| Cold cutout compile-time comparison | Completed; OptiX IR has a faster LLVM generation stage but a slower total shader compilation in this run; medians below |
| Cutout image comparison at 64 spp | 99.28455% of pixels identical; mean absolute channel error 0.0139923 on a 0–255 scale. Outputs are not bit-identical across the two compiler paths |
| Images Compute Sanitizer | Passed: exit 0, all 18 image/volume assertions, `ERROR SUMMARY: 0 errors`; `.deps/optixir-images-memcheck-fixed/{stdout,sanitizer}.log` |
| Cutout Compute Sanitizer | Passed: OptiX IR, 64 spp, exit 0 with a saved PNG and `ERROR SUMMARY: 0 errors`; `build-msvc-llvm/test-results/cutout-llvm-offline-cutout-query-memcheck-20260930-010618-347` |
| Interactive rendering on the final adapter | Passed: cutout and procedural, 4,096 spp each, exit 0 with saved PNGs; both logs confirm `moduleCreate: format=OPTIX_IR` |
| Additional Compute Sanitizer workloads | The ray-query integration fixture passed on both PTX and OptiX IR after payload sizing; see the v14 results below |
| Large captured-state Compute Sanitizer | Passed: 65,536 rays, exit 0, `ERROR SUMMARY: 0 errors`; `build-msvc-llvm/test-results/ray-memcheck-v7-large-20260930-010841/` |
| GPU execution-time comparison | Not measured by the compile-time experiment below |

The cold PTX comparison uses
`.deps/optixir-compile-ab-20260930-005337/0-ptx/.cache/kernel_ed984c0d76c0a790.llvm-v12.ptx`.
It proves unchanged output for this cutout kernel, not byte identity for every
possible shader. The original AOT checks loaded artifacts emitted by named
compilation. The subsequent v15 tests also exercise a `compile_only` producer
before creating any consumer for that kernel.

Interactive results are recorded under `build-msvc-llvm/test-results/` in
`cutout-llvm-gui-cutout-query-normal-20260930-010110-105` and
`clean-procedural-llvm-20260930-010152`. Each directory contains the process
result, rendering log, and PNG. These completion checks are separate from the
subsequent GPU throughput comparisons below.

The cold-cache cutout run records three processes per route in
`.deps/optixir-compile-ab-20260930-005337/results.json`. Its medians are:

| Compilation stage | LLVM/PTX | LLVM/OptiX IR |
|---|---:|---:|
| Host LLVM generation | 81.5722 ms | 72.9058 ms |
| OptiX `moduleCreate` | 164.089 ms | 217.289 ms |
| Total shader compilation | 270.549 ms | 318.592 ms |

The LLVM generation stage is approximately **10.6% faster**, but total shader
compilation is approximately **17.8% slower** with OptiX IR in this sample. The
first PTX `moduleCreate` took 1,330 ms; comparing against that initialization
outlier would give a misleading speedup. The table reports route medians,
including that process in the sample. These results support retaining the
explicit opt-in rather than changing the default compiler path.

`Generated OptiX IR with CUDA LLVM CodeGen` identifies an uncached generation.
`OptiX moduleCreate: format=..., bytes=..., attempt=..., duration=... ms` measures
the CPU module-creation call. Neither log is a GPU execution-time measurement;
record device, driver, shader/cache state, workload, and separate dispatch timing
when comparing performance.

## Subsequent shared ray-query payload sizing (LLVM cache v14)

LLVM/PTX and LLVM/OptiX IR now declare the actual maximum ray-query payload
capacity needed by the shader, within **2–32 words**. The existing layout is
unchanged: two words carry the query pointer, the third carries the pipeline ID,
and direct captures occupy up to 29 further words. A capture-free cutout shader
therefore uses **3 words**. A shader using only the generic context fallback
needs **5 words**, including its context pointer; the direct-capture boundary
still uses **32 words**. Mixed pipelines use one common maximum. Legacy AST
artifacts remain at 2 words, as do LLVM modules with no surviving query pipeline.

Every query call's active count and the host OptiX payload declaration agree
with the saved `RAY_QUERY_PAYLOAD_COUNT`. The private traversal intrinsic keeps
its fixed 32-word signature; arguments beyond the active count are undefined
padding. Query and captured-reference addresses retain their original identity.
Lazy compilation resolves the count before serialization, while validated cache
hits and AOT loads recover it from their sidecars. An internal unknown-count
sentinel is never saved or passed to OptiX. Cache revision **v14** separates this
ABI from earlier fixed-capacity artifacts.

The full MSVC/Ninja build and **22/22 focused tests passed** with this change;
the test record is `.deps/dynamic-ray-payload-tests.log`. Coverage includes
payload counts 3, 5 and 32 with mixed handlers, cache reuse and AOT loading, plus
both LLVM/PTX and OptiX IR correctness regressions. Both routes also passed the
query Compute Sanitizer run with `ERROR SUMMARY: 0 errors`, including those
cache/AOT paths and the 2,048-byte captured-state workload (257 rays, five
dispatches); see `.deps/dynamic-payload-memcheck-{0,1}/sanitizer.log`.

A controlled LLVM/PTX cutout comparison used 4,096 spp, three iterations per
fresh process, at most 64 spp per dispatch, and two ABBA groups (eight processes).
The median of process medians was **558.2515 spp/s** for the preceding FTZ-enabled
fixed-32 baseline and **699.1330 spp/s** with dynamic payload sizing, an observed
**25.24% improvement**. All output PNG hashes matched. OptiX reported raygen
registers falling from **102 to 70** and any-hit registers from **94 to 65**;
continuation stack **96 bytes** and continuation spills **236 bytes** were
unchanged. This laptop showed clock and timing variation across runs, so the
sample does not promise a fixed speedup for other workloads or operating states.
Commands, images, clocks and all samples are recorded in
`build-msvc-llvm/test-results/cutout-runtime-abba-20260930-013300-900350/experiment.json`.

A subsequent AST/NVRTC versus LLVM/PTX comparison used the same final runtime
and workload, again with two ABBA groups and four processes per route. Median
throughput was **623.3573 spp/s for AST** and **654.8742 spp/s for LLVM**.
Clock variation and overlapping samples make this evidence of comparable
performance on this workload, not a reliable claim that LLVM is faster.
Each route reproduced its own pre-optimization PNG exactly. The existing
cross-route difference was unchanged: RGB mean absolute error **0.0137313/255**,
maximum channel error **5/255**, and **96.6382%** identical pixels.
All samples and images are in
`build-msvc-llvm/test-results/cutout-ast-llvm-abba-20260930-013936-516204/`.

A final cold-cache compilation check after v14 repeated three processes per
route. Median LLVM generation was **79.7033 ms for PTX** versus **67.6546 ms
for OptiX IR**; module creation was **399.992 ms** versus **468.629 ms**, and
total shader compilation was **500.3611 ms** versus **555.4053 ms**. Thus direct
IR emission saved about 15% in its host generation stage, while total compilation
remained about 11% slower in this run. This does not demonstrate an end-to-end
compile-time win; the OptiX IR route remains experimental and opt-in. All six
renders completed and reproduced their respective earlier 64-spp PNG hashes.
The record is `.deps/optixir-compile-ab-20260930-014548/results.json`.

## General-query state follow-up

Cache v15 replaces load/select/store of the previous committed hit with a
conditional update. Valid distances write the new hit directly; invalid ones
preserve every field, including an earlier accepted hit. Ordered bounds checks
remain in place. The full MSVC build and all 22 focused tests passed, including
compile-only artifact production and invalid replacement preservation.

An eight-process ABBA comparison used the five-query mixed surface/procedural
benchmark with mutable captures: 65,536 rays, 512 warmup dispatches, 512 dispatches
per sample and nine samples. Median throughput was **2,173.43 Mqueries/s for v14**
and **2,196.79 Mqueries/s for v15** (+1.07%). The process ranges overlap; this
does not establish a repeatable speedup. PTX input shrank from **33,546 to 30,736
bytes**. Results are in
`build-msvc-llvm/test-results/ray-query-guarded-commit-v15-20260930-020109/`.

The corresponding eight-process cutout comparison produced identical PNGs and
all five generated PTX files were byte-identical. Registers remained raygen 70,
any-hit/intersection 65, with raygen continuation stack 96 bytes and spills
236 bytes. Median throughput was 638.9738 versus 641.7760 spp/s; the two ABBA
groups had inconsistent directions. This establishes unchanged cutout code,
resources and images for this workload, without a throughput improvement claim.
Results are in
`build-msvc-llvm/test-results/cutout-runtime-abba-20260930-020205-349878/`.

Cache v16 further reduces the private query state from **112 to 88 bytes**:
it retains the acceleration handle and original instance-table pointer, removes
the unused binding count/padding and query flags, and preserves public resource
bindings and captured-reference identity. Full build and all 22 regressions
passed. In another eight-process ABBA comparison, median mixed-query throughput
increased from **2,205.40 to 2,289.92 Mqueries/s (+3.83%)**; both groups improved
(+2.73% and +5.14%). PTX input decreased from **30,736 to 29,913 bytes**. Every
numerical check passed. GPU clock states still differed, including 7,001/8,001
MHz memory states, so these are workload-specific observations rather than a
guaranteed speedup. The record is
`build-msvc-llvm/test-results/ray-query-compact-query-v16-20260930-021321/`.

The v16 cold cutout renders retained the earlier route-specific image hashes.
All five default PTX artifacts remained byte-identical, including the 35,366-byte
ray-query shader (`b7c9478e16a16506addbc6ec2b5c81ec7b2456d6ce4d7484b54fcd34d8472d9c`).
The record is `.deps/optixir-compile-ab-20260930-021409/`.

A final four-process AST/LLVM ABBA comparison of the mixed-query benchmark
measured **2,520.10 Mqueries/s for AST** and **2,240.775 Mqueries/s for LLVM/PTX**
(median of each route's process medians). LLVM remains about **11.1% behind**
AST on this workload. The cutout and mixed-query results should not be generalized
to each other. All four numerical checks passed; the record is
`build-msvc-llvm/test-results/ray-query-v16-final-ast-llvm-20260930-021937/`.

## Direct traversal operands (OptiX IR revision 9)

Direct IR now passes its nine native LLVM float operands directly to the
OptiX traversal intrinsic, matching the SDK form. The existing typed-register
workaround remains in the PTX path. The fixed intrinsic signature, active payload
count and memory clobber are unchanged; only the OptiX IR cache revision changes.

Eight cold-cache ABBA processes preserved the image hash in every run and reduced
the cutout module from **21,516 to 20,664 bytes**. Median host generation was
75.454 to 69.605 ms, driver module creation 315.879 to 259.686 ms, and total
compilation 417.772 to 349.289 ms. The groups moved in opposite directions, so
these aggregate medians do **not** establish a stable compile-time speedup.
Registers, stack and spills were unchanged. The record is
`.deps/optixir-runtime-abba-20260930-022011-1932967/`.

All 11 OptiX IR/metadata regressions passed. Both LLVM/PTX and OptiX IR also
passed Compute Sanitizer with **38 assertions and zero memory errors**, covering
the ray-query integration fixture, including large captures, compile-only
production, cache reuse and AOT loading. Logs are in
`.deps/final-query-memcheck-v16-optixir9-{0,1}/`.

## Final build-gate validation

The selected MSVC/Ninja tree was reconfigured with the OptiX IR CMake option
**OFF**, fully rebuilt using `cmake --build`, and passed **12/12 PTX/metadata
tests**. Inspection of the actual CUDA compile/link rules confirmed that the
IR encoder, legalizer, LLVM 7 writer and enable macro were absent. CUDA, Fallback,
SIMD, Remote, GUI and LLVM remained enabled. The option was then restored **ON**,
the full tree rebuilt again, and **22/22 combined tests passed**. The final local
build includes OptiX IR, while runtime selection remains explicitly opt-in.

Records are `.deps/optixir-cmake-gate-{off,on}.json`,
`.deps/optixir-gate-{off,on}-build.log`, `.deps/optixir-gate-off-tests.log` and
`.deps/optixir-final-all-tests.log`. All five cutout PTX shaders and the PTX PNG
were byte-identical between the OFF build and the final ON build; final OptiX IR
artifacts and images also matched the earlier revision-9 comparison exactly.

The final six-process cold-cache check (three per route) measured these medians:

| Compilation stage | LLVM/PTX | LLVM/OptiX IR |
|---|---:|---:|
| Host LLVM generation | 80.9134 ms | 70.3120 ms |
| OptiX `moduleCreate` | 166.917 ms | 181.499 ms |
| Total shader compilation | 302.3605 ms | 272.3857 ms |

Individual-stage medians need not sum to the median total. Host generation was
13.1% faster in this sample; total compilation also favored IR here, but earlier
comparisons reversed that result. These measurements do not establish a stable
end-to-end compile-time improvement. All six renders succeeded and reproduced
their route-specific hashes. The record is
`.deps/optixir-compile-ab-20260930-022827/results.json`.

## Reconstructing resource captures (cache v17)

The shared XIR `analyze_unique_resource_origins` analysis proves which resource
arguments forward an unchanged kernel argument through every ordinary-call and
ray-query callback edge. Kernel roots map to themselves. Multiple roots,
computed descriptors, unknown function uses and cyclic dependencies remain
unproven. The analysis does not assume that resource contents are immutable.

CUDA LLVM excludes proven resource descriptors from both register captures and
scratch contexts. AH and IS reload the complete descriptor from the corresponding
constant launch-parameter field, including buffer view offset and length.
Bound texture storage assumptions use the original kernel argument index.
Ordinary captured values retain their snapshot semantics, and captured references
retain their addresses. Explicit per-capture field mappings preserve callback
argument order when resource and non-resource captures are interleaved.

The 32-word budget is computed after removing these descriptors. General queries
still reserve two words for the query pointer and one for the pipeline ID;
captures beyond the remaining 29 words use private scratch and a two-word context
pointer, for five payload words total. This change does not yet decompose the
general query's mutable state into bidirectional payloads.

The full MSVC build and **23/23 focused tests passed**, including the new shared
analysis test and AH/IS resource fixtures with offset views, live lengths,
conflicting callable roots, and both direct and oversized captures. Both output
routes passed Compute Sanitizer with **42 assertions and zero memory errors**.
The records are `.deps/rq-resource-v17-tests.log` and
`.deps/final-query-memcheck-resource-v17-{0,1}/`.

The mixed-query benchmark's active payload decreased from **12 to 8 words**.
OptiX reported registers falling from **74 to 70** in raygen, **83 to 78** in AH
and **84 to 80** in IS. Raygen continuation stack/spills remained **160/12 bytes**;
IS direct stack remained **8 bytes**. PTX input shrank from **29,913 to 26,197
bytes**. Optimized LLVM IR confirms descriptor reloads from `@params` in the
callbacks; see `.deps/rq-resource-v17-dump/`. These code-size and resource-count
changes do not by themselves establish a throughput improvement.

The subsequent eight-process v16/v17 LLVM comparison changed only the CUDA DLL.
Raw process-median throughput was **1,718.810 versus 1,113.680 Mqueries/s**;
the two ABBA groups moved **+5.63% and -35.21%**, respectively. GPU-utilization
samples at or above 90% had per-process median graphics clocks ranging from
825 to 1,665 MHz, and the CSVs do not isolate measured phases. All eight numerical
oracles passed. This unstable result does not establish a throughput improvement;
no slow runs were removed. The complete record is
`build-msvc-llvm/test-results/ray-query-resource-v17-long-20260930-093030-124997/`.

A longer v16 AST/LLVM baseline used 1,048,576 rays, 1,000 warmup dispatches,
512 dispatches per sample and nine samples across eight ABBA processes. Identical
LLVM code ranged from 921.91 to 2,627.28 Mqueries/s. Active GPU clock states varied
substantially; the clock CSVs include compilation and validation, without exact
sample boundaries. These results do not establish a stable relative throughput
or a precise cause of the variation. All eight runs and numerical checks remain
in `build-msvc-llvm/test-results/ray-query-v16-ast-llvm-long-20260930-085618-997821/`.

## Recovering native traversal results (cache v18)

Qualified AH/IS callbacks now use the implicit OptiX hit object instead of
passing the caller's query address through traversal. AH supports candidate-hit
reads, triangle commits, and termination after a commit in the same block.
IS supports candidate-hit/ray reads and procedural commits. Invocation-local
state preserves bound tightening and invalid-then-valid or valid-then-invalid
commit behavior; LLVM inlining/SROA removes unused fields. Final hit kind,
distance, instance, primitive and barycentrics are decoded after traversal,
with guarded miss/triangle/curve/custom getters and zero custom barycentrics.

Intermediate committed-hit/state observations, query aliases, AH ray-bound
observations and procedural termination retain the general stateful path.
This is a conservative qualification, not a claim that every query needs no
software state. The native-result layout still reserves two null query-pointer
words in v18; removing that padding is a separate ABI change.

The full MSVC build and **23/23 focused tests passed**. Added fixtures exercise
accepted surface termination and procedural commit recovery with exact bounds,
including NaN/Inf in precise mode. Both PTX and OptiX IR passed query integration
under Compute Sanitizer with **43 assertions and zero errors**. Evidence:
`.deps/rq-hardware-v18-{build,tests,memcheck}.log`.

For `test_procedural`, optimized LLVM no longer has a query alloca. PTX input
decreased **16,305 -> 14,366 bytes**, raygen continuation stack **128 -> 32 bytes**,
and AH registers **73 -> 67**; raygen stayed at **67 registers** and IS increased
**72 -> 74**, with zero spills. The dumps are in
`.deps/procedural-hardware-v18-dump/`.

The initial 1024-spp comparison correctly stopped on differing PNG hashes.
All **1,207 changed pixels** were black in v17 and colored in v18; AST, Fallback
and SIMD were nonblack at all those positions. Fallback, SIMD and v18 each
passed the existing gallery reference. Inspection found the old fast-math
IS range checks can accept a NaN into software hit state even when native
reporting does not accept it. The renderer can produce that NaN when subtracting
nearly equal squared distances before a square root. This is not a finite-input
query regression; precise-mode NaN rejection remains covered separately.
The failed exact comparison and subsequent CPU image analyses are preserved in
`procedural-hardware-v18-abba-20260930-094553-000023/` under the test-results
directory and `.deps/procedural-hardware-v18-*-image-analysis.json`.

A separate eight-process v17/v18 ABBA run required every process to pass the
unmodified gallery reference and exact PNG reproducibility within each version.
Median throughput was **1,141.27 -> 1,189.42 spp/s (+4.22%)**, with group ratios
**+3.91% and +4.18%**. It includes blit and final readback; laptop clocks were
unlocked, so this is a local observation rather than a general speed guarantee.
The v18 AST/LLVM four-process comparison measured **1,211.28 / 1,205.53 spp/s**.
Full records are `procedural-hardware-v18-reference-abba-20260930-095720-629103/`
and `procedural-hardware-repeat-abba-20260930-095053-182962/` under test-results.

A single-instruction diagnostic on a private v17 PTX cache changed only the
perpendicular-distance rejection from `setp.gt` to `setp.gtu`, rejecting NaN.
It removed all 1,207 reported black pixels (matching v18 there exactly) and
43 additional bright-region pinholes still present in v18. Every originally
nonblack pixel was unchanged. Thus native-result recovery is not by itself a
complete fix for NaN capture side effects. The untouched baseline, exact patch,
cache hashes and output analysis are preserved in
`.deps/procedural-v17-nan-diagnostic/`; no reference image was regenerated.
