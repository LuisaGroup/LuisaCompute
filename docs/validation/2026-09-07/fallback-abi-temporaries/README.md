# Fallback native-call temporary alignment

## Root cause and invariant

The LLVM representation of a Luisa value does not always retain the source
ABI's alignment. In particular, a matrix is represented by nested arrays of
float: its LLVM ABI alignment is four bytes, while `float2x2` requires eight
and `float3x3`/`float4x4` require sixteen bytes. The embedded device library is
compiled against those source types and may use aligned vector loads/stores.

The required invariant for a compiler-created temporary passed to that library
is `allocation_alignment >= max(LLVM_alignment, source_ABI_alignment)`. This
applies to input pointers and output pointers, regardless of whether SROA,
inlining or instruction selection happens to eliminate their storage.

Before the fix, matrix operations and resource-result temporaries used plain
`CreateAlloca(mapped_type)`. The real failing module contained, after inlining:

```llvm
%tmp = alloca [4 x [4 x float]], align 4
store <4 x float> %column, ptr %tmp, align 16
```

The observed machine instruction was `vmovaps %xmm3, 0x2138(%rsp)` with
`rsp % 16 == 0`; the destination was only eight-byte aligned. This is an
invalid backend ABI promise, not a reason to disable vectorization or blame
the material operation which made the kernel large enough to expose it.

## Minimal permanent regression and fix

`test_fallback_llvm_abi_alignment` generates independent XIR operations for
transpose, inverse, determinant, matrix-matrix and matrix-vector products,
both outer-product forms, and an acceleration-structure instance transform.
It inspects all 53 temporary pointer contracts before optimization. It does
not rely on a lucky stack offset or on a crash being reproducible at one
optimization level. The old code fails for matrix inputs and outputs at
dimensions two, three and four, including the instance-transform result.

`_create_abi_temporary` now implements the invariant at the native-call
boundary. Matrix, texture, bindless, acceleration-structure and ray-query
result lowering use it. Existing explicitly aligned locals, aggregate
extraction/insertion, bitwise-cast storage and packed capture/print storage
retain their existing rules. No initialization or zero-fill semantics change.
The relocatable shader cache ABI is incremented so old objects are not reused.

## Validation

Evidence: `/var/tmp/psycles-fallback-codegen-policy-io0puc`.

- Original complete 311,835-instruction Psycles material kernel: SIGSEGV
  before, completes its 16x16, 4-sample staged render after the fix. The
  application binary and shader graph are unchanged; only the backend library
  changes. It still takes the large-function minimal-IR pipeline.
- Exact 53-contract LLVM regression: red before, green after.
- All 166 Psycles fallback CTests pass after rebuilding with 32 threads.
- Psycles HIP suite: 164/164, and a strict native XIR-to-SPIR-V whole-film
  Vulkan canary passes. This fix changes no HIP/Vulkan lowering.

The original-module reproduction, LLVM IR, machine assembly and logs are
retained with the evidence. Validation of the isolated next candidate uses
its codegen and public headers with the existing compatible runtime libraries;
it is not a claim that an unrelated dirty SDK tree has been committed.
