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
| Additional Compute Sanitizer workloads on the final adapter | Pending final results |
| Large captured-state Compute Sanitizer | Passed: 65,536 rays, exit 0, `ERROR SUMMARY: 0 errors`; `build-msvc-llvm/test-results/ray-memcheck-v7-large-20260930-010841/` |
| GPU execution-time comparison | Not measured by the compile-time experiment below |

The cold PTX comparison uses
`.deps/optixir-compile-ab-20260930-005337/0-ptx/.cache/kernel_ed984c0d76c0a790.llvm-v12.ptx`.
It proves unchanged output for this cutout kernel, not byte identity for every
possible shader. The AOT checks load artifacts emitted by named compilation;
they do not separately exercise a `compile_only` producer process.

Interactive results are recorded under `build-msvc-llvm/test-results/` in
`cutout-llvm-gui-cutout-query-normal-20260930-010110-105` and
`clean-procedural-llvm-20260930-010152`. Each directory contains the process
result, rendering log, and PNG. These completion checks are separate from the
pending GPU throughput comparison.

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
