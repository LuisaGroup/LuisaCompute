# Metal AIR LLVM Code Generation: Intrinsics, ABI, and Translation Conventions

This document describes the independent LuisaCompute `metal4` backend's
compute and raster paths from XIR to Apple Metal IR (AIR), the AIR-facing LLVM
conventions used by the emitter, and the reconstructed `.metallib` container
written by the backend. The original `metal` backend remains a separate,
source-MSL implementation; sharing `metal-cpp` and runtime concepts does not
make its shader code-generation path depend on this experimental backend.

The authoritative implementation is:

- [metal_codegen_llvm.cpp](metal_codegen_llvm.cpp),
  [metal_codegen_llvm.h](metal_codegen_llvm.h), and the focused
  `metal_codegen_llvm_{type,function,access,resource,atomic,arithmetic,metadata,preflight}.cpp`
  translation units for XIR-to-LLVM/AIR lowering.
- [metal_xir_pipeline.cpp](../metal_xir_pipeline.cpp) for AST-to-XIR lowering
  and XIR optimization.
- [metal_air_pipeline.cpp](../metal_air_pipeline.cpp) for LLVM optimization,
  LLVM 14 bitcode downgrade, and library assembly.
- [metal_metallib.cpp](../metal_metallib.cpp) for the reconstructed MTLB
  container.
- [metal_device.cpp](../metal_device.cpp) and
  [metal_compiler.cpp](../metal_compiler.cpp) for loading,
  caching, and pipeline-state creation.
- [metal_raster_ext.cpp](../metal_raster_ext.cpp) and
  [metal_raster_shader.cpp](../metal_raster_shader.cpp) for paired
  vertex/fragment creation, render pipeline states, binding, and draw encoding.

## 1. Status and stability

Apple publicly documents the high-level flow
`MSL -> Metal IR -> metallib -> MTLLibrary -> pipeline state`, but does not
publish the LLVM-level AIR ABI, intrinsic catalogue, metadata schema, or MTLB
binary format as a stable interface.

Accordingly, this document separates three kinds of convention:

1. **Luisa conventions** are project-owned contracts such as argument packing,
   XIR pass order, and the fail-closed support boundary.
2. **LLVM conventions** are ordinary LLVM IR semantics and data-layout rules.
3. **Observed AIR conventions** are intrinsic names, metadata tuples, target
   triples, numeric flags, and container records inferred from Apple-generated
   artifacts and other open-source implementations.

The third category is empirical. The runtime-covered subset recorded in
Section 16 passes LLVM verification, the Metal framework loader, Apple
`metallib --app-store-validate`, `metal-readobj`, `metal-objdump`, and its
listed runtime tests. Oracle-only conventions are not runtime claims. Both
categories must be revalidated when Xcode, macOS, AIR, Metal language, or LLVM
versions change.

## 2. End-to-end pipeline

~~~text
Luisa DSL / AST
    |
    v
AST -> XIR
    |
    +-- basic XIR optimization
    +-- if autodiff scopes remain:
    |     lower ray-query loops and switches
    |     destructure CFG -> immediately inline callables
    |     cleanup -> simplify CFG -> reg2mem -> restructure CFG
    |     verify -> lower autodiff -> reg2mem -> verify
    +-- lower ray-query loops and switches
    +-- destructure CFG -> immediately inline all eligible call sites
    +-- mem2reg -> SSA optimization
    +-- remove unused callables
    +-- simplify CFG
    +-- verify all blocks are reachable
    |
    v
XIR support preflight
    |
    +-- unsupported compute or raster: explicit shader-creation error
    |
    v
LLVM 21 IR with AIR target, intrinsics, and metadata
    |
    +-- LLVM verification
    +-- LLVM default O2 module pipeline
    +-- LLVM verification
    |
    v
JuliaLLVM downgrade writer -> LLVM 14-compatible bitcode
    |
    +-- compute: kernel_main + kernel_main_indirect
    +-- raster: vertex_main + fragment_main
    |
    v
Deterministic MTLB container
    |
    v
MTLDevice::newLibrary(data)
    |
    +-- MTLFunction kernel_main
    +-- MTLFunction kernel_main_indirect
    |
    v
MTL4::Compiler -> MTL4 compute/render pipeline state
~~~

The `metal4` module contains no MSL shader code generator. Stable resource
objects and command encoding still use the established Metal runtime API,
while every user compute and raster shader is generated through XIR/LLVM/AIR.
The original `metal` backend remains the separate source-MSL implementation.

### 2.1 XIR pass order

`metal_translate_ast_to_xir` performs the following order. Pass failure reports
from ray-query-loop lowering, switch lowering, CFG destructuring, and CFG
restructuring are fatal rather than being treated as a partially lowered
module. Inlining likewise rejects a nonzero malformed-call count instead of
silently carrying malformed call sites into LLVM lowering.

1. Translate the traced AST with `ast_to_xir_translate`.
2. Verify the translated module.
3. Run `create_basic_optimization_pipeline`, forwarding
   `ShaderOption::enable_fast_math`.
4. If the module contains an autodiff scope, run this pre-autodiff
   normalization:
   - `lower_ray_query_loop_to_loop`
   - `lower_switch`
   - `destructure_cfg`
   - `inline_all` with autodiff scopes allowed in callers
   - one post-inline cleanup iteration
   - `simplify_cfg`
   - `reg2mem`
   - `restructure_cfg`
   - verify that PHIs are absent and merge blocks are unique
   - lower autodiff scopes and intrinsics
   - `reg2mem` again
   - verify the same no-PHI and unique-merge invariants again
5. Run the main lowering pipeline:
   - `lower_ray_query_loop_to_loop`
   - `lower_switch`
   - `destructure_cfg`
   - `inline_all`
   - `mem2reg`
6. Run `create_ssa_optimization_pipeline`.
7. Run cleanup:
   - `unused_callable_removal`
   - `simplify_cfg`
8. Verify the final module and require every block to be reachable. The AIR
   pipeline repeats this reachable-block verification immediately before
   XIR-to-LLVM translation.

Inlining is deliberately scheduled immediately after CFG destructuring in
both normalization paths. Multi-block inlining rejects structured caller or
callee CFG; for functions made plain by
`destructure_cfg`, its block splitting and branch insertion preserve the
unstructured form expected by LLVM lowering. Recursive callables and functions
with structured operations deliberately preserved by that pass can remain
uninlined. In the autodiff path, `reg2mem` removes PHIs and cross-block SSA
values before CFG restructuring and again after differentiation. The main
path then uses `mem2reg` to promote both original and inliner-created local
temporaries before the ordinary SSA optimizations.

Reverse autodiff is therefore an XIR transform, not an AIR intrinsic family.
By the LLVM handoff, autodiff scopes, gradient markers, and `BACKWARD` have
been replaced by ordinary XIR control flow, arithmetic, and memory operations.

`create_basic_optimization_pipeline` currently expands to:

1. LICM
2. DCE
3. local store forwarding
4. local load elimination
5. DCE
6. algebraic simplification
7. constant folding
8. DCE
9. reference-argument promotion
10. SROA
11. dead-store elimination
12. DCE

`create_ssa_optimization_pipeline` currently expands to:

1. LICM
2. algebraic simplification
3. constant folding
4. SCCP
5. GVN
6. if conversion
7. PHI cleanup
8. DCE
9. local store forwarding
10. local load elimination
11. dead-store elimination
12. DCE

These factory expansions live in the shared XIR pass pipeline and can change
independently of the Metal driver. The short factory names and the expanded
order are both relevant when bisecting an AIR miscompile.

`LUISA_DUMP_XIR=1` writes both the initial and optimized textual XIR forms.
The file names are based on the kernel hash.

### 2.2 LLVM and downgrade stages

Direct and indirect entries are generated in separate LLVM modules. Each
module is verified, optimized with LLVM's default per-module O2 pipeline,
verified again, and then passed to the in-tree JuliaLLVM downgrade writer:

~~~cpp
llvm::BitcodeWriter140::prepareModule(*module);
llvm::WriteBitcode140ToFile(*module, stream);
~~~

The downgrade code is compiled against the same LLVM 21 installation as the
Metal backend. It rewrites/serializes the module in the LLVM 14 bitcode dialect
accepted by the tested Apple Metal toolchain. The resulting bytes begin with
the LLVM bitcode magic `42 43 c0 de`; the backend does not invoke the
`metal` command-line compiler at runtime.

Each downgraded module becomes one compute-function record in the in-tree MTLB
writer. The writer orders the direct entry before the indirect entry, derives
UUID material from the SHA-256 hash of their concatenated AIR bytes, and emits
the same container bytes for identical inputs. No Apple `metallib` link step is
needed at runtime; Apple's loader consumes the reconstructed container
directly. Section 14 documents the container records and validator.

The submodule is [JuliaLLVM/llvm-downgrade](https://github.com/JuliaLLVM/llvm-downgrade),
whose upstream base is pinned by this branch at
`1e04ee99aff7c059606502ee86d03eb7d1c5d781`. Luisa builds only its LLVM 14
writer and the required pointer/module rewriters, not its older 5.0 or 7.0
writers. The submodule checkout remains pristine. LLVM 21 compatibility and
the AIR typed-pointer recovery extensions described below live in the parent-
owned `llvm-downgrade-llvm21-air.patch`, which CMake applies only to a build-
tree mirror.

CMake verifies the submodule commit and cleanliness when Git metadata is
available, and always verifies an exact SHA-256 source manifest. The source
manifest and overlay hash form a second fingerprint for the patched mirror, so
an unchanged reconfigure reuses its object files while any source or patch
change recreates and reapplies the mirror. Patch application clears inherited
Git work-tree variables and is bounded by the build directory. These checks
make the same pinned input reproducible without modifying the third-party
checkout. The imported writer sources retain the upstream Apache-2.0-with-
LLVM-exception licensing.

`LUISA_DUMP_LLVM_IR=1` dumps the post-O2 textual modules as
`<source>.direct.air.ll` and `<source>.indirect.air.ll`.

### 2.3 Typed-pointer recovery directives

LLVM 21 represents every pointer as opaque `ptr addrspace(N)`, but the LLVM 14
bitcode consumed by the AIR loader still encodes pointee types. Those pointee
types are semantically significant for textures, samplers, argument buffers,
and resources nested inside structures. A remaining unannotated opaque pointer
is serialized by the downgrade writer as a pointer to an empty structure, so
resource pointers must carry an explicit recovery directive.

Function parameters and returns use two function-metadata attachments:

~~~llvm
declare !arg_eltypes !0 !ret_eltype !1
    ptr addrspace(2) @example(ptr addrspace(1))

!0 = !{i32 0, %argument.element.type undef}
!1 = !{%return.element.type undef}
~~~

- `arg_eltypes` is a sequence of `(parameter index, element-typed undef)`
  pairs. It was already understood by the JuliaLLVM pointer rewriter.
- `ret_eltype` is the corresponding single element-typed undef for an opaque
  pointer return. Luisa's in-tree extension uses it for functions such as
  `air.get_read_sampler`, whose downgraded return must be
  `%struct._sampler_t addrspace(2)*`.

During `prepareModule`, the pointer rewriter reconstructs a legacy typed
function signature from both attachments and inserts no-op bitcasts around
call operands and pointer results where the opaque LLVM 21 values meet that
signature. The writer's pointer map then serializes the recovered typed uses.

Opaque pointer fields nested in named structures use a separate module-level
directive:

~~~llvm
!llvm.struct_eltypes = !{!2, !3}
!2 = !{!"luisa.bindless.array", i32 0,
       %luisa.bindless.item undef}
!3 = !{!"luisa.air.texture.2d", i32 0,
       %struct._texture_2d_t undef}
~~~

Each record names the LLVM structure followed by one or more
`(physical field index, element-typed undef)` pairs. The LLVM 14 writer builds
a per-structure override map, uses it both while enumerating subtypes and while
emitting the legacy structure type record, and omits
`llvm.struct_eltypes` itself from serialized named metadata. It is a downgrade
instruction, not AIR runtime metadata.

The current emitter applies these directives recursively to:

- buffer data fields as `i8 addrspace(1)*`;
- the bindless-array field as `LCBindlessItem addrspace(1)*`;
- direct and wrapped 2D/3D texture fields as pointers to the corresponding
  opaque `_texture_2d_t` or `_texture_3d_t` handle;
- the dynamic sampler wrapper as a pointer to the AIR-version-specific sampler
  state record.

This recovery protocol is part of the AIR ABI boundary. Ordinary LLVM type
correctness and reflection metadata alone do not reconstruct these legacy
pointee identities.

## 3. AIR module target contract

### 3.1 Target triple

The architecture component always carries the selected AIR major/minor pair:

~~~text
air64_v<air-major><air-minor>-apple-macosx<major>.<minor>.<patch>
~~~

The digits after `v` are concatenated, so AIR 2.8 uses `air64_v28`. For
example, the validated macOS 26.3 configuration emits:

~~~text
air64_v28-apple-macosx26.3.0
~~~

The macOS version in this triple currently comes from the **runtime host**, not
`CMAKE_OSX_DEPLOYMENT_TARGET`. The same host version selects the AIR and Metal
language versions and the MTLB platform/version header. The separately
detected command-line-tools SDK version appears only in the LLVM `SDK Version`
module flag described in Section 7. CMake obtains it from
`xcrun --sdk macosx --show-sdk-version`; if that query is unavailable, the
emitter falls back to the host version for the flag.

This makes the current generated AIR and raster AOT archives host-targeted.
An archive generated on macOS 26, for example, is not promised to load on the
project's macOS 13 deployment floor even when the host C++ binary was built
with a macOS 13 deployment target. Producing a genuinely portable archive
requires a separately validated older AIR/Metal target and corresponding
toolchain contract; merely substituting the CMake deployment number caused
Apple's runtime compiler service to reject the generated library during the
current investigation.

### 3.2 Data layout

The module uses the following hard-coded layout:

~~~text
e-p:64:64:64
-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64
-f32:32:32-f64:64:64
-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64
-v96:128:128-v128:128:128-v192:256:256-v256:256:256
-v512:512:512-v1024:1024:1024
-n8:16:32
~~~

This is little-endian, uses 64-bit pointers, stores `i1` in one byte, and
defines explicit vector ABI alignments. The emitter additionally owns the
memory representation described below rather than relying on LLVM vector
layout alone.

### 3.3 Address spaces

| LLVM address space | AIR/MSL meaning | Current uses |
|---:|---|---|
| 0 | generic/private | ordinary allocas, spill temporaries, callable reference parameters |
| 1 | device | buffer data, raw device addresses, indirect dispatch record |
| 2 | constant | root argument block and direct dispatch-size record |
| 3 | threadgroup | XIR shared allocations |

Address-space numbers are part of the observed AIR ABI and should not be
renumbered casually.

## 4. Type lowering

The core type convention is a two-form representation:

- `reg_type`: compact SSA form used for arithmetic, PHIs, function values,
  callable arguments, and returns.
- `mem_type`: ABI-preserving form used by argument blocks, buffers, allocas,
  loads, and stores.

Loads convert `mem_type -> reg_type`; stores convert
`reg_type -> mem_type`. This is essential for padded vectors and structures.

### 4.1 Scalars

| XIR/Luisa type | LLVM type | Signedness handling |
|---|---|---|
| `bool` | `i1` | logical/integer predicates |
| `int8`, `uint8` | `i8` | carried by XIR type |
| `int16`, `uint16` | `i16` | carried by XIR type |
| `int32`, `uint32` | `i32` | carried by XIR type |
| `int64`, `uint64` | `i64` | carried by XIR type |
| `half` | `half` | floating point |
| `float` | `float` | floating point |

LLVM integers are signless. Division, remainder, right shift, comparisons,
integer extension, and integer/float conversions consult the original XIR
type to choose signed or unsigned instructions.

Although `bool` is represented as `i1`, the AIR data layout specifies
`i1:8:8`. Its store and allocation size is therefore one byte. LLVM structure
fields are not bit-packed: `{ i1, i1, i1, i1 }` has allocation size four and
member offsets 0, 1, 2, and 3 under this layout.

`double`, `float8_e4m3`, and `float8_e5m2` are rejected.

### 4.2 Vectors

For a logical `vector<T, N>`:

~~~text
reg_type = <N x T.reg_type>
mem_type = [align(N, 2) x T.mem_type]
~~~

Thus a three-component vector has three SSA lanes but four memory slots.
Register-to-memory conversion starts from a zero aggregate and writes the
logical lanes, so padded lanes have deterministic zero bytes. Memory-to-
register conversion ignores padding. The same zero-initialization rule applies
to arrays and structures and is required by `PACK`, which observes the complete
object representation rather than only logical members.

This separation is especially important for booleans. LLVM can store a
register vector `<4 x i1>` in one byte, but that type is never used as the
Luisa memory representation. A Luisa `bool4` is stored as `[4 x i1]`, which
occupies four bytes under `i1:8:8`; `bool3` is also four bytes because its
memory array is padded to four elements. By comparison, `<4 x i8>` itself is
already four bytes, and its Luisa memory form `[4 x i8]` is also four bytes.

This convention mirrors Luisa/Metal aggregate layout and avoids depending on
LLVM's native three-vector memory behavior.

### 4.3 Matrices

Matrices are column-major:

~~~text
reg_type = [N x <N x scalar>]
mem_type = [N x vector-memory-type]
~~~

Matrix literals, extraction, transpose, multiplication, determinant, and
inverse all use column indexing. No named AIR matrix intrinsic is emitted;
matrix operations are expanded into ordinary LLVM vector/scalar operations.

### 4.4 Arrays

Arrays preserve their logical element count in both forms:

~~~text
reg_type = [N x element.reg_type]
mem_type = [N x element.mem_type]
~~~

Conversions recurse element by element.

### 4.5 Structures

Register structures are dense logical LLVM structures. Memory structures
contain explicit `[N x i8]` gaps and tail padding so that every member begins
at the offset prescribed by the Luisa `Type`.

For example, a host/Luisa structure containing four scalar booleans occupies
four bytes. Its memory form may print as `{ i1, i1, i1, i1 }`, but each `i1`
occupies a byte because of the module data layout; it is not an `i4` or a
one-byte packed bit field.

The type cache records:

- the physical LLVM member index for each logical member;
- the byte offset for each logical member.

Register/memory conversion recursively skips or supplies padding. Structure
GEP uses the cached byte offset rather than assuming the logical member number
is also the physical LLVM member number.

### 4.6 Buffers

Every typed buffer and byte buffer is internally:

~~~llvm
%luisa.buffer.N = type { ptr addrspace(1), i64 }
~~~

The fields are a device address and a byte length. Element type is a compile-
time XIR property, not part of the opaque LLVM pointer type. The
`llvm.struct_eltypes` directive marks the first field as `i8`, so the LLVM 14
form is `{ i8 addrspace(1)*, i64 }` rather than a pointer to an anonymous empty
type.

### 4.7 Textures and samplers

A direct 2D or 3D texture is represented in LLVM 21 as an opaque pointer in
address space 1. The root-argument structure field and every texture intrinsic
parameter are additionally marked for downgrade to a pointer to the opaque
named handle `%struct._texture_2d_t` or `%struct._texture_3d_t`. AIR reflection
adds the element and access information that neither handle type carries:

~~~text
texture2d<float|int|uint, read|write|read_write|sample>
texture3d<float|int|uint, read|write|read_write|sample>
~~~

The emitter derives the access qualifier from the normalized direct uses of
the XIR kernel argument. Size queries do not change the qualifier. Sampling
may not be mixed with read or write access on the same direct argument, and
the current sampling path accepts only `float` textures. Integer and unsigned
integer textures remain valid for direct read, write, and size operations.

Sampler values are constant-address-space pointers. They are not kernel
arguments: the backend constructs a module-local table of Apple sampler-state
records when a sampling operation is emitted. Section 8.7 records the version
split and exact table mapping.

### 4.8 Bindless ABI reconstructed from Apple AIR

The existing Metal host and MSL paths use this ABI:

~~~text
LCBindlessArray (8 bytes)
  +0  device LCBindlessItem *items

LCBindlessItem (32 bytes, alignment 16)
  +0  device void *buffer
  +8  uint64 packed
 +16  texture2d resource token
 +24  texture3d resource token

packed & 0x0000ffffffffffff  = buffer byte size
(packed >> 48) & 0xff       = 2D sampler code
(packed >> 56) & 0xff       = 3D sampler code
~~~

The LLVM 21 representation deliberately uses one-field texture wrappers for
resources nested inside the slot:

~~~llvm
%struct._texture_2d_t   = type opaque
%struct._texture_3d_t   = type opaque
%luisa.air.texture.2d   = type { ptr addrspace(1) }
%luisa.air.texture.3d   = type { ptr addrspace(1) }
%luisa.bindless.array = type { ptr addrspace(1) }
%luisa.bindless.item  = type { ptr addrspace(1), i64,
                               %luisa.air.texture.2d,
                               %luisa.air.texture.3d }
~~~

After applying `llvm.struct_eltypes`, the serialized LLVM 14 types are
equivalent to:

~~~llvm
%luisa.air.texture.2d = type {
    %struct._texture_2d_t addrspace(1)* }
%luisa.air.texture.3d = type {
    %struct._texture_3d_t addrspace(1)* }
%luisa.bindless.array = type {
    %luisa.bindless.item addrspace(1)* }
%luisa.bindless.item  = type { i8 addrspace(1)*, i64,
                               %luisa.air.texture.2d,
                               %luisa.air.texture.3d }
~~~

The wrappers are not cosmetic: a texture resource embedded in another AIR
structure must retain both its single eight-byte storage slot and the nested
typed resource identity expected by the legacy consumer. The XIR AIR emitter
models these types, emits the reflection in Section 6.1, and accepts the
implemented bindless operations in preflight. Strict bindless texture and
buffer tests now compile, create pipeline states, dispatch, and pass; see
Section 16.1.

### 4.9 GPU-written indirect-dispatch records

`LC_IndirectDispatchBuffer` is the one supported backend-owned custom type.
The public AST presents it as a value-like resource handle, while XIR must
represent opaque custom arguments as references. The AST-to-XIR translator
therefore creates an opaque reference argument; the Metal internal LLVM
function ABI carries the recovered 16-byte binding value, and the kernel entry
loads it from the root argument block.

The host/device binding and record layout are:

~~~text
LCIndirectDispatchBuffer (16 bytes)
  +0  device void *buffer
  +8  uint offset
 +12  uint capacity

dispatch allocation
  +0  16-byte-aligned uint count header
 +16  first 32-byte dispatch slot

slot[index] (32 bytes, alignment 16)
  +0  uint3 block_size
 +16  uint4(dispatch_size, kernel_id)
~~~

`INDIRECT_DISPATCH_SET_COUNT` performs an ordinary 16-byte-aligned `i32`
device store at the header. `INDIRECT_DISPATCH_SET_KERNEL` computes the
absolute 32-bit index as `offset + local_index`, compares it to `capacity`
with unsigned `<`, begins the slot array at byte 16, zero-extends the accepted
index to 64 bits, and stores `<3 x i32>` and `<4 x i32>` with 16-byte
alignment. The vector-3 ABI alignment gives the slot a 32-byte stride.

Apple reflection names the wrapper `LCIndirectDispatchBuffer` (without
Luisa's type-registry underscore) and expands it into three physical AIR
locations: a read-write device `air.buffer`, followed by two
`air.indirect_constant` uints for offset and capacity. The nested `void *`
resource record has no `air.buffer_size`, `air.arg_type_size`, or
`air.arg_type_align_size`; its eight-byte footprint is carried by the parent
field descriptor. A following root argument begins at physical location three.

### 4.10 Instanced acceleration structures

The Metal host ABI supplies a Luisa acceleration argument as a 16-byte
`LCAccel`, not as the nominal eight-byte AST handle alone:

~~~text
LCAccel (16 bytes, natural alignment 8; root slot alignment 16)
  +0  instance_acceleration_structure handle
  +8  device LCInstance *instances

LCInstance (72 bytes, alignment 8)
  +0  float transform[12]
 +48  uint options
 +52  uint mask
 +56  uint intersection_function_offset / Luisa user ID
 +60  uint mesh_index
 +64  MTLResourceID acceleration_structure_id
~~~

The transform contains the first three rows of each of four column-major
`float4x4` columns. Querying it reconstructs row three as `(0, 0, 0, 1)`;
writing it stores only those twelve affine elements. Field 56 is named
`intersection_function_offset` in Apple's reflected `LCInstance` layout and
is also the field used by Luisa's instance-user-ID API.

The LLVM 21 representation retains both wrapper storage and legacy AIR
pointee identity:

~~~llvm
%struct._instance_acceleration_structure_t = type opaque
%luisa.air.accel.handle = type { ptr addrspace(1) }
%luisa.accel.instance = type { [12 x float], i32, i32, i32, i32, i64 }
%luisa.accel = type { %luisa.air.accel.handle, ptr addrspace(1) }
~~~

`llvm.struct_eltypes` identifies the wrapper field as
`%struct._instance_acceleration_structure_t` and the second `LCAccel` field as
`%luisa.accel.instance`. These directives are required so the LLVM 14 writer
recovers the typed acceleration handle and `LCInstance addrspace(1)*` instead
of anonymous empty-structure pointers. The root reflection consumes two
physical locations: one acceleration-structure resource and one writable
device buffer for instance records.

Apple's source compiler gives the equivalent one-field handle wrapper a
`metal::raytracing::_acceleration_structure`-style name and wraps the twelve
floats in its `metal::array` spelling. Luisa's named LLVM structures are
structurally equivalent and are accepted after typed-pointer downgrade; the
names themselves are not reflection-facing ABI strings.

Instance access zero-extends the Luisa `uint` index to `i64` and performs an
in-bounds GEP over `LCInstance`; no runtime instance-count check is generated.
User-ID and visibility queries/updates are aligned `i32` loads/stores at
offsets 56 and 52. Opacity update preserves unrelated option bits, clears bits
`0x4 | 0x8`, then sets `0x4` for opaque or `0x8` for non-opaque. Transform
queries and writes use the twelve-float affine convention above.

### 4.11 Native Metal shader logging

`PrintInst` does not add a root argument, staging allocation, atomic cursor,
or command-completion parser. Metal4 lowers printing to Apple's shader-log
ABI and attaches one `MTL::LogState` to every `MTL4::CommandBuffer` through
`MTL4::CommandBufferOptions::setLogState`.

Each translated module contains an always-inline variadic helper:

~~~llvm
define linkonce_odr void @__luisa_metal_shader_log(
    ptr addrspace(2) %format, i64 %argument_size, ...)

declare void @air.os_log(
    ptr addrspace(2) %subsystem,
    ptr addrspace(2) %category,
    i32 %level,
    ptr addrspace(2) %format,
    ptr %arguments,
    i64 %argument_size)
~~~

The helper creates an LLVM `va_list`, passes its argument pointer to
`air.os_log`, and closes the list. The constant subsystem and category are
`org.luisa.compute` and `shader`; the current log-level operand is `1`.
Consequently, internal callables need no hidden printer state and direct and
indirect entry points use the same native logging convention.

### 4.12 Native stateful intersection-query objects

XIR represents the two Luisa query modes with opaque custom types named
`LC_RayQueryAll` and `LC_RayQueryAny`. They have no byte layout and are not
ordinary LLVM aggregate values. After the Metal pipeline has inlined
callables, each local XIR query allocation becomes one native, generic/private
address-space-zero pointer:

~~~llvm
%struct._intersection_query_t = type opaque
~~~

Both Luisa types use this same native pointee. Their behavioral difference is
the `accept_any` reset flag described in Section 8.9, not a different LLVM
type. A query allocation calls AIR's native allocator; the single XIR store
that initializes the matching local is an alias-preserving no-op rather than
a memory store. Native query objects are deallocated in reverse allocation
order immediately before every lowered return or unreachable exit.

Each of the four XIR constructors (`RAY_TRACING_QUERY_ALL`,
`RAY_TRACING_QUERY_ANY`, and their motion-blur forms) denotes fresh mutable
traversal state, even when its operands are identical to those of another
constructor. Shared XIR memory-effect analysis therefore classifies
constructors as reads of global state. Early CSE and GVN do not common them,
and LICM does not hoist them out of their original control-flow location. DCE
may still remove an unused constructor. This memory-effect rule also applies
to the motion-blur forms. The current Metal 4 `intersection_query` API has no
motion-AS tag or ray-time operand, so those forms are rejected by AIR
preflight before allocation or reset emission.

This object lifecycle is deliberately local to one lowered function. Query
objects are not root arguments, reflected resources, byte-addressable
storage, shared allocations, or values that may escape through a call. The
preflight and lowering pipeline rely on `inline_all` immediately after CFG
destructuring so no query object must cross an AIR function boundary.

## 5. Kernel argument ABI

### 5.1 Root argument block

The host and AIR emitter share these rules:

- Every logical argument begins at the next 16-byte boundary.
- The root block is at least 16 bytes.
- Its final size is rounded up to 16 bytes.
- Explicit byte padding is present in the LLVM structure.
- A buffer binding occupies 16 bytes: 64-bit GPU address plus 64-bit byte size.
- An acceleration binding occupies 16 bytes: an instanced acceleration-
  structure handle plus a device pointer to 72-byte `LCInstance` records.
- A direct texture occupies eight bytes in the LLVM root structure, while the
  host still starts the following logical argument at the next 16-byte boundary.
- Arrays and structures use their Luisa memory representation and may be
  passed as top-level uniform arguments.

This matches `MetalShader::launch`, which aligns every encoded argument to
16 bytes before copying it into the root argument block.

The 16-byte rule above is the **AIR object layout**, not the allocation
granularity of a Metal staging buffer. At dispatch/draw time, root blocks of
at most 4096 bytes use Metal's inline-byte APIs: `setBytes` for compute and
the matching `setVertexBytes`/`setFragmentBytes` pair for raster. Larger root
blocks, up to the current 65536-byte host scratch/archive limit, are copied
into the stream upload pool and bound with `setBuffer`; raster binds the same
staged allocation at buffer index 0 in both stages. Upload-pool suballocations
have 256-byte-aligned offsets (an oversized standalone allocation has offset
zero), satisfying Metal's buffer-offset contract without changing the
16-byte AIR layout or the reflected root size. The allocation is retained by
the command-buffer callback list until GPU completion.

Top-level reference arguments are rejected except for XIR's opaque-reference
representation of `LC_IndirectDispatchBuffer`, which is decoded as the
value-like binding in Section 4.9. Aggregate uniform reflection is recursive
and flattens arrays and structure leaves into AIR indirect-constant locations;
this is distinct from the byte layout of the root block itself.

### 5.2 Public direct entry

Conceptually:

~~~llvm
define void @kernel_main(
    ptr addrspace(2) %args,
    ptr addrspace(2) %dispatch_size,
    <3 x i32> %thread_id,
    <3 x i32> %block_id,
    <3 x i32> %dispatch_id,
    <3 x i32> %block_size,
    i32 %warp_size,
    i32 %warp_lane_id)
~~~

Buffer index 0 contains the root argument block. Buffer index 1 contains a
16-byte-aligned `uint3` dispatch size. Direct mode synthesizes kernel ID 0.

### 5.3 Public indirect entry

Conceptually:

~~~llvm
define void @kernel_main_indirect(
    ptr addrspace(2) %args,
    ptr addrspace(1) %dispatch_size_and_kernel_id,
    <3 x i32> %thread_id,
    <3 x i32> %block_id,
    <3 x i32> %dispatch_id,
    <3 x i32> %block_size,
    i32 %warp_size,
    i32 %warp_lane_id)
~~~

Buffer index 1 points to a device `uint4`:

- XYZ: dispatch size.
- W: Luisa kernel ID.

The built-in `prepare_indirect_dispatches` MSL kernel binds the same root
argument block at index 0 and the slot's `dispatch_size_and_kernel_id` at
index 1 before encoding the indirect command.

For a root larger than 4096 bytes, the prepare pass receives the upload-pool
buffer and writes that buffer reference into each indirect command. The later
encoder that executes the indirect-command buffer therefore declares the
staged root buffer with `useResource(..., MTL::ResourceUsageRead)`: binding it
only to the prepare kernel is insufficient residency information for the
resources referenced by the encoded commands. Inline roots continue to use
Metal-managed `setBytes` storage. The staged allocation's callback lifetime
spans preparation and indirect execution in the same command buffer; Metal
owns the corresponding lifetime for inline storage.

### 5.4 Internal implementation and callable ABI

The private `kernel_main_impl` receives decoded logical arguments followed
by this hidden state tail:

~~~text
uint3 dispatch_size
uint  kernel_id
uint3 thread_id
uint3 block_id
uint3 dispatch_id
uint3 block_size
uint  warp_size
uint  warp_lane_id
~~~

Every callable receives and forwards the same tail. This makes special
registers available inside nested callables without globals or special AIR
calls. Ordinary callable reference parameters are generic address-space
pointers; the opaque indirect-dispatch reference uses its value-like 16-byte
binding ABI after inlining.

Print instructions do not extend this tail. They lower directly to the
module-local `__luisa_metal_shader_log` helper and Apple's `air.os_log` ABI;
there is no printer root argument or callable state to forward.

Kernel implementation and callable definitions are private, `alwaysinline`,
and `convergent`. Calls to them are also marked convergent.

### 5.5 Entry guard

Every public entry computes:

~~~text
component_mask = dispatch_size ugt dispatch_id
in_bounds      = air.all.v3i1(component_mask)
~~~

The implementation is called only if all three coordinates are in range.
This guard is generated by the backend and is not present in XIR.

## 6. AIR kernel reflection metadata

The public function has `arg_eltypes` metadata describing the pointee types
of the opaque root-argument and dispatch pointers. Texture and sampler AIR
declarations use the same attachment for resource parameters;
`air.get_read_sampler` uses `ret_eltype` for its pointer return; and named
argument/resource wrappers use `llvm.struct_eltypes` as described in
Section 2.3. These typed-pointer directives are separate from the reflection
records below.

The module's `air.kernel` record contains:

1. the public function;
2. an empty stage-info node;
3. an ordered argument-info node.

The public entry argument semantics are:

| Entry index | AIR semantic | Reflected type/name |
|---:|---|---|
| 0 | `air.indirect_buffer` | `Arguments args` at buffer location 0 |
| 1 | `air.buffer` | direct `uint3 dispatch_size` or indirect `uint4 dispatch_size_and_kernel_id` at location 1 |
| 2 | `air.thread_position_in_threadgroup` | `uint3 thread_id` |
| 3 | `air.threadgroup_position_in_grid` | `uint3 block_id` |
| 4 | `air.thread_position_in_grid` | `uint3 dispatch_id` |
| 5 | `air.threads_per_threadgroup` | `uint3 block_size` |
| 6 | `air.threads_per_simdgroup` | `uint warp_size` |
| 7 | `air.thread_index_in_simdgroup` | `uint warp_lane_id` |

The root argument block is read-only constant memory. The direct dispatch
record is reflected as read-only constant memory. The indirect dispatch record
is in device address space and reflected as read-write, although the current
LLVM pointer parameter itself is marked `readonly`.

Inside the root argument block:

- A scalar, vector, or matrix uniform consumes one physical
  `air.indirect_constant` location.
- Each buffer consumes two physical indices:
  - device pointer: `air.buffer`, address space 1, currently
    `air.read_write`;
  - byte size: `air.indirect_constant` of type `ulong`.
- A structured buffer element gets recursive `air.struct_type_info`.
- A texture consumes one physical location and is reflected as `air.texture`
  with `air.read`, `air.write`, `air.read_write`, or `air.sample` plus the
  matching `texture2d`/`texture3d` type name.
- An acceleration argument consumes two physical locations: a read-only
  `air.instance_acceleration_structure` handle and a read-write
  `air.buffer` of 72-byte, 8-byte-aligned `LCInstance` elements.
- A top-level array contributes one location for every recursively flattened
  element. A top-level structure contributes the sum of its members' location
  counts. Nested `air.struct_type_info` records preserve byte offsets, base
  sizes, array counts, type names, and member names while leaf details carry
  the flattened physical location.
- Offsets, sizes, alignments, type names, and argument names are recorded.

All buffers are currently reflected as read-write; XIR argument-usage analysis
does not yet refine this metadata.

The logical XIR argument ordinal and physical AIR location are deliberately
tracked separately. A texture, nested aggregate, buffer, and trailing scalar
can therefore be interleaved without assigning the trailing fields to stale
logical ordinals. The strict ABI test covers nested structures containing four
byte-sized booleans, `byte4`, a nested array, a top-level array, a texture, a
buffer, and a trailing scalar.

### 6.1 Bindless reflection oracle

Apple reflects `LCBindlessArray` as a structure wrapper that occupies eight
bytes and consumes **one root physical location**, even though its nested slot
description contains six locations:

~~~text
LCBindlessArray.items
  offset 0, size 8
  air.indirect_buffer, location 0, air.read, address space 1
  element type LCBindlessItem, size 32, alignment 16

LCBindlessItem
  offset  0, size 8: void buffer
      air.buffer, location 0, air.read_write, address space 1
  offset  8, size 8: ulong buffer_size
      air.indirect_constant, location 1
  offset 14, reported size 4: uint sampler2d
      air.indirect_constant, location 2
  offset 15, reported size 4: uint sampler3d
      air.indirect_constant, location 3
  offset 16, size 8: texture2d<float, sample>
      air.texture, location 4, air.sample
  offset 24, size 8: texture3d<float, sample>
      air.texture, location 5, air.sample
~~~

The overlapping four-byte sampler metadata sizes are what Apple emits for the
two eight-bit fields packed into the `i64`; they are not independent four-byte
storage fields. Bindless texture tokens are reflected as `sample` even when a
particular kernel only performs an integer-coordinate read. This layout is an
observed oracle and must remain byte-for-byte compatible with
`MetalBindlessArray::Slot` and `LCBindlessItem`.

### 6.2 Acceleration reflection oracle

An `LCAccel` root member has size 16, consumes two physical locations, and
contains this recursive reflection shape:

~~~text
LCAccel.handle
  offset 0, size 8
  air.instance_acceleration_structure, location 0, air.read
  type acceleration_structure<instancing>

LCAccel.instances
  offset 8, size 8
  air.buffer, location 1, air.read_write, address space 1
  element type LCInstance, size 72, alignment 8

LCInstance
  offset  0, array count 12: float transform
  offset 48: uint options
  offset 52: uint mask
  offset 56: uint intersection_function_offset
  offset 60: uint mesh_index
  offset 64: ulong acceleration_structure_id
~~~

The nested locations are relative to the `LCAccel` wrapper; the outer root
argument descriptor records the current physical base. A following logical
argument therefore begins two physical locations later.

### 6.3 Shader logging has no root reflection

`PrintInst` contributes no root member and consumes no physical argument
location. Its format strings are constant-address-space globals used by the
module-local variadic logging helper. At runtime, `MetalStream` attaches an
`MTL::LogState` through `MTL4::CommandBufferOptions::setLogState`; Metal owns
the message transport and invokes the configured callback. Root reflection is
therefore identical for otherwise equivalent kernels with and without print
instructions.

## 7. Module metadata and function attributes

The emitter records:

- module flag `SDK Version = [SDK major, SDK minor]`, detected independently
  from the runtime-host macOS target encoded in the AIR triple;
- `wchar_size = 4`;
- frame-pointer mode 2;
- maximum device buffers: 31;
- maximum constant buffers: 31;
- maximum threadgroup buffers: 31;
- maximum textures: 128;
- maximum read-write textures: 8;
- maximum samplers: 16;
- `llvm.ident = "LuisaCompute XIR Metal AIR LLVM codegen"`;
- `air.version`;
- `air.language_version = ("Metal", major, minor, patch)`;
- `air.compile.denorms_disable`;
- either `air.compile.fast_math_enable` or
  `air.compile.fast_math_disable`;
- `air.compile.framebuffer_fetch_enable`;
- optional `air.source_file_name`.

The public entry is initially marked `mustprogress`, `nofree`, `nounwind`,
`willreturn`, `convergent`, `no-builtins`, frame-pointer `all`, and
no-trapping-math. It is deliberately not marked `nosync`, because inlining
may expose a workgroup barrier.

Both pointer parameters are `noundef`, aligned, dereferenceable, read-only
with respect to the argument/dispatch record itself, and carry
`air-buffer-no-alias`. Built-in parameters are `noundef`.

LLVM O2 may infer, remove, or canonicalize attributes. The source emitter's
attributes are the project-owned input contract; optimizer-inferred output is
not a stable AIR convention.

## 8. Exhaustive explicitly emitted AIR intrinsics

The hand-written emitter now declares bounds, barrier, fence, integer-bit,
atomic, SIMD, texture, sampler, acceleration-intersection/query, and scalar-
math AIR functions. Their names are assembled from the XIR operation,
signedness, dimension, address space, and scalar/vector type. The tables below
describe the complete emitted families; only the overloads reached by a module
are declared in that module.

XIR `ASSUME` additionally uses LLVM's `llvm.assume`. LLVM O2 may introduce
other optimizer-owned LLVM intrinsics; those are not part of the hand-written
AIR ABI.

### 8.1 Bounds reduction

~~~llvm
declare i1 @air.all.v3i1(<3 x i1>)
~~~

It is pure/no-memory, no-sync, no-throw, no-free, and will-return. It is used
only by the generated dispatch bounds guard.

### 8.2 Workgroup barrier

~~~llvm
declare void @air.wg.barrier(i32, i32)
call void @air.wg.barrier(i32 2, i32 1)
~~~

This is the supported workgroup-wide barrier operation and lowers
`SYNCHRONIZE_BLOCK`. Declaration and call are convergent. SIMD/subgroup
operations use the separate families in Section 8.6.

The literal pair `(2, 1)` is an observed ABI convention. In the current
mapping, `2` is the threadgroup-memory fence flag corresponding to MSL
`mem_threadgroup`, and `1` is the workgroup barrier scope/control value.
Because Apple does not document this LLVM ABI, recheck it against an
MSL-compiled sample when updating the toolchain.

### 8.3 Volatile device-memory fence

~~~llvm
declare void @air.atomic.fence(i32, i32, i32)
call void @air.atomic.fence(i32 1, i32 5, i32 2)
~~~

Apple-generated AIR for
`atomic_thread_fence(mem_device, memory_order_seq_cst)` uses the operand tuple
`(1, 5, 2)`. The observed memory flag bits are device `1`, threadgroup `2`,
and texture `4`; the current emitter uses only device `1` here.

Luisa volatile buffer and byte-buffer operations combine this fence with an
LLVM volatile memory operation:

- volatile read: device fence, then volatile load;
- volatile write: volatile store, then device fence.

The position of the fence is part of the lowering contract. An LLVM volatile
load/store by itself does not provide Metal device-scope ordering.

### 8.4 Integer bit intrinsics

`CLZ`, `CTZ`, `POPCOUNT`, and `REVERSE` use:

~~~text
air.clz.[vN]iW(value, i1 zero_is_undefined)
air.ctz.[vN]iW(value, i1 zero_is_undefined)
air.popcount.[vN]iW(value)
air.reverse_bits.[vN]iW(value)
~~~

`N` is present only for vectors and `W` is the scalar bit width. Luisa's CLZ
and CTZ pass `false`, defining the zero input. Declarations are pure and
no-sync. The same helper is reused internally for SIMD ballot masks.

### 8.5 Atomic intrinsics and conventions

Atomics are supported for 32-bit `int`, `uint`, and `float`, on device-buffer
or fixed-size threadgroup-shared storage. The prefix encodes the address
space:

~~~text
device memory:      air.atomic.global.*   scope operand 2
threadgroup memory: air.atomic.local.*    scope operand 1
~~~

The emitted integer families are:

~~~text
air.atomic.{global|local}.load.i32
air.atomic.{global|local}.xchg.i32
air.atomic.{global|local}.{add|sub|and|or|xor|min|max}.{s|u}.i32
air.atomic.{global|local}.cmpxchg.weak.i32
~~~

An integer RMW call has the observed shape
`(pointer, value, i32 0, scope, i1 true)`. Compare-exchange spills the expected
value to a four-byte-aligned private temporary and passes
`(pointer, expected-pointer, desired, i32 0, i32 0, scope, i1 true)`.

Device float exchange, add, and subtract use the native `.f32` global
intrinsics, and device float compare-exchange uses
`air.atomic.global.cmpxchg.weak.f32`. Threadgroup float exchange and compare-
exchange bitcast through `i32`. Float add/subtract/min/max operations not
covered by a native intrinsic use a load plus weak-compare-exchange loop over
the bit pattern. Floating bitwise atomics are rejected by preflight.

All atomic call declarations are must-progress, no-throw, and will-return, but
are not declared pure or no-sync. The literal ordering/control operands above
are empirical AIR ABI values; do not reinterpret them as ordinary LLVM
`AtomicOrdering` enumerators.

### 8.6 SIMD/subgroup intrinsics

The emitter uses convergent calls for all subgroup operations. Generic numeric
overloads follow this grammar:

~~~text
air.<operation>.{s|u}.[vN]iW   signed/unsigned integer
air.<operation>.[vN]fW         half/float
~~~

The currently emitted operation bases are:

- `simd_and`, `simd_or`, `simd_xor`;
- `simd_min`, `simd_max`, `simd_sum`, `simd_product`;
- `simd_prefix_exclusive_sum`, `simd_prefix_exclusive_product`;
- `simd_shuffle`, with an `i16` lane operand;
- `simd_broadcast_first`.

Reductions and prefixes accept 8/16/32-bit signed or unsigned integer and
16/32-bit floating scalar/vector values. `WARP_READ_LANE` and
`WARP_READ_FIRST_ACTIVE_LANE` additionally support booleans, 64-bit integers,
matrices, arrays, and structures by legalization:

- boolean lanes are widened to `i32`, shuffled, then truncated;
- 64-bit integer lanes are split into low/high `i32` halves and reassembled;
- vectors with 64-bit elements and all larger aggregates recurse through
  their logical elements or members.

The special boolean/mask calls are:

~~~text
air.simd_is_first() -> i1
air.simd_active_threads_mask.i64() -> i64
air.simd_ballot.i64(i1) -> i64
air.simd_all(i1) -> i1
air.simd_any(i1) -> i1
~~~

`WARP_FIRST_ACTIVE_LANE` counts trailing zeros in the active-thread mask.
Count-bits uses ballot plus popcount; the public `uint4` active mask stores the
low/high 32-bit halves of the 64-bit ballot in X/Y and leaves Z/W zero; prefix
count widens the predicate to `i16` and uses exclusive prefix sum. All-equal
broadcasts the first active value, compares it, and applies `air.simd_all` to
each comparison component. Shader execution reorder
matches the existing Metal source backend and is currently a no-op. Raster
quad derivatives are not subgroup operations in AIR and remain rejected in a
compute entry.

### 8.7 Direct textures and sampler states

AIR texture access codes used consistently by intrinsic operands and
reflection are:

| Code | Meaning | Reflection |
|---:|---|---|
| 1 | read | `air.read` |
| 2 | write | `air.write` |
| 3 | read-write | `air.read_write` |
| 4 | sample | `air.sample` |

For dimension `D = 2 or 3`, integer-coordinate reads return a data/status
structure and the emitter extracts field zero:

~~~llvm
declare { <4 x T>, i8 } @air.read_texture_Dd.<suffix>(
    ptr addrspace(1) texture,
    ptr addrspace(2) read_sampler,
    <D x i32> coordinate,
    <D x i32> offset,
    i32 level,
    i32 access)
~~~

`T` is float or i32. The suffix is `v4f32`, `s.v4i32`, or `u.v4i32`.
Direct reads obtain the sampler with `air.get_read_sampler()`, pass a zero
offset and level zero, and use the derived access code. Width, height, and
depth use `air.get_<component>_texture_Dd(texture, level)`; direct size queries
currently request level zero.

Writes use:

~~~llvm
declare void @air.write_texture_Dd.<suffix>(
    ptr addrspace(1) texture, <D x i32> coordinate,
    <4 x T> value, i32 level, i32 access)
~~~

A scalar Luisa write is splatted to four lanes. Direct writes use level zero.
When the argument was inferred as read-write, the emitter follows the write
with `air.fence_texture_Dd(texture)`.

Float sampling has two families:

~~~llvm
declare { <4 x float>, i8 } @air.sample_texture_Dd.v4f32(
    texture, sampler, <D x float> coord,
    i1 true, <D x i32> zero_offset,
    i1 has_explicit_lod, float lod, float 0.0, i32 0)

declare { <4 x float>, i8 } @air.sample_texture_Dd_grad.v4f32(
    texture, sampler, <D x float> coord,
    <D x float> ddx, <D x float> ddy,
    float min_lod_clamp,
    i1 true, <D x i32> zero_offset, i32 0)
~~~

This covers implicit LOD, explicit LOD, explicit gradients, and gradients with
a minimum-LOD clamp for both 2D and 3D. The non-gradient intrinsic is observed
as convergent; the gradient intrinsic is not. Both return `{data, status}` and
the status byte is ignored. Sampling and reflection require a float texture
with sample-only usage.

Dynamic Luisa sampler selection computes:

~~~text
code = ((filter << 2) | address) & 15
~~~

with filter order `POINT`, `LINEAR_POINT`, `LINEAR_LINEAR`, `ANISOTROPIC` and
address order `EDGE`, `REPEAT`, `MIRROR`, `ZERO`. The module contains 13 unique
descriptors and a 16-entry address-space-2 table of one-field sampler wrappers:

| Codes | Descriptors |
|---|---|
| 0..3 | `0x007bff0000080049`, `0x007bff0000080092`, `0x007bff00000800db`, `0x007bff0000080000` |
| 4..7 | `0x007bff0000082a49`, `0x007bff0000082a92`, `0x007bff0000082adb`, `0x007bff0000082a00` |
| 8..11 | `0x007bff0000084a49`, `0x007bff0000084a92`, `0x007bff0000084adb`, `0x007bff0000084a00` |
| 12 | `0x007bff0000f84a49` |
| 13..15 | aliases of descriptors for codes 9..11 |

The last three aliases deliberately match the MSL path: anisotropic repeat,
mirror, and zero reuse the corresponding linear-linear states.

Codes 4..7 use linear minification/magnification with nearest-mip selection;
they must not use the superficially similar no-mipmap descriptor. The latter
always samples mip zero and breaks Luisa's `LINEAR_POINT` contract. The four
nearest-mip constants above were regenerated with Apple Metal 32023.883 using
an MSL 4.0 differential oracle for edge, repeat, mirrored-repeat, and zero
address modes.

Sampler-state storage changed with AIR:

| AIR version | Constant representation |
|---|---|
| 2.5 / 2.6 | scalar `i64`, descriptor OR bit 63 |
| 2.7 and later | `[2 x i64] [descriptor, 0]` |

Every unique state is an aligned internal constant in address space 2 and is
listed in `!air.sampler_states` as
`!{!"air.sampler_state", <global>}`. Dynamic selection must preserve this
wrapper form:

~~~llvm
%luisa.air.sampler = type { ptr addrspace(2) }
@luisa.air.samplers = internal addrspace(2) constant
    [16 x %luisa.air.sampler] [...]
~~~

The code masks the dynamic sampler code, indexes the wrapper array, and loads
the wrapper's only field. It does not model the table as `[16 x ptr]`, even
though both layouts occupy the same bytes. `llvm.struct_eltypes` assigns that
field the actual sampler-state pointee (`i64` for AIR 2.5/2.6 or `[2 x i64]`
for AIR 2.7+), while `arg_eltypes` assigns the sampling intrinsic parameter the
opaque `%struct._sampler_t` pointee. The downgrade rewriter inserts the legacy
typed-pointer bitcast at their boundary. This nested-wrapper rule is required
for dynamic selection and is strict-runtime-validated by the bindless mip
test.

The table is created lazily, so a module without sampling has neither the
table nor sampler-state metadata.

### 8.8 Instanced triangle, curve, and motion intersection

Static triangle closest-hit and any-hit tracing use two observed AIR calls.
First the module obtains a typed null intersection-function table:

~~~llvm
declare ptr addrspace(1) @air.get_null_intersection_function_table()
~~~

`ret_eltype` recovers the return as
`%struct._intersection_function_table_t addrspace(1)*` for LLVM 14. The current
emitter follows the modern Apple oracle and declares it
`inaccessiblememonly mustprogress nofree nounwind readonly willreturn`. It is
not `readnone` and does not carry `nosync`.

The triangle-only intersection result is:

~~~llvm
%luisa.air.intersection.result = type {
    i32, float, i32, i32, ptr addrspace(1),
    i32, i32, <2 x float>, i1
}
~~~

Enabling curve data selects a distinct ten-field result whose final member is
the curve parameter:

~~~llvm
%luisa.air.curve.intersection.result = type {
    i32, float, i32, i32, ptr addrspace(1),
    i32, i32, <2 x float>, i1, float
}
~~~

The pointer at field 4 is recovered as `i8 addrspace(1)*`. The current emitter
uses four suffix families:

~~~text
.instancing.triangle_data
.instancing.triangle_data.curve_data
.instancing.triangle_data.primitive_motion.instance_motion
.instancing.triangle_data.curve_data.primitive_motion.instance_motion
~~~

The direct-call grammar is:

~~~text
air.intersect<suffix>(
    float3 origin, float3 direction, float t_min, float t_max,
    instance_acceleration_structure handle, uint mask,
    [float time for primitive/instance motion],
    intersection_function_table table,
    generic byte* payload, ulong payload_size,
    uint winding, uint triangle_culling, uint geometry_culling,
    uint opacity_culling, uint force_opaque, uint triangle_geometry,
    uint curve_basis, uint curve_type, uint curve_control_points,
    bool assume_identity_transforms, bool accept_any)
~~~

The exact control tail emitted by both Luisa and the modern Apple MSL oracle
is:

~~~text
payload = null, payload_size = 0,
winding = 0, triangle_culling = 0, geometry_culling = 0,
opacity_culling = 0, force_opaque = 1, geometry_type = 1,
curve_basis = 0, curve_type = 0, curve_control_points = 0,
assume_identity_transforms = false,
accept_any = false for closest / true for any
~~~

In raw order after the zero payload size, the nine `i32` controls and two
booleans are:

~~~text
i32 0, i32 0, i32 0, i32 0, i32 1, i32 1,
i32 0, i32 0, i32 0, i1 false, i1 accept_any
~~~

The first boolean is `assume_identity_transforms`, not a motion toggle. Motion
is encoded by the intrinsic suffix and by inserting `float time` immediately
after the mask. Geometry bits are `triangle = 1`, `bounding_box = 2`, and
`curve = 4`, so triangle+curve is `5` and all three are `7`.

`CurveBasisMD` controls the remaining curve tail:

| Luisa basis set | AIR basis | segment control points |
|---|---:|---:|
| piecewise linear | `2` | `2` |
| cubic B-spline | `0` | `4` |
| Catmull--Rom | `1` | `4` |
| cubic Bezier | `3` | `4` |
| multiple possible bases | `0xffffffff` | `0` |

Round curves use curve type `0`. For a stateful query local, lowering follows
its one initialization store back to the constructor to recover this metadata.

The acceleration handle, intersection-function table, and payload parameters
carry `arg_eltypes` for their respective legacy pointees. Matching the modern
Apple oracle, the intersection declaration carries only
`mustprogress nounwind willreturn`; it is not marked read-only, no-free, or
convergent. The emitted call carries `nounwind willreturn` and is likewise not
convergent.

Luisa's `Ray` stores origin and direction as three-element aggregate values.
The emitter explicitly builds AIR `<3 x float>` arguments rather than
bitcasting their memory representation. It treats result field 0 as the hit
kind (`!= 0` means hit). Closest-hit translation maps result fields as follows:

| Luisa `TriangleHit` field | AIR result field | Miss value |
|---|---:|---|
| instance ID | 5 | `~0u` |
| primitive ID | 2 | `~0u` |
| barycentrics | 7 | `(0, 0)` |
| distance | 1 | `0.0` |

Any-hit returns only the field-0 hit predicate. Native intersection type `3`
is a curve surface. For a closest curve hit, Luisa barycentrics become
`(curve_parameter, -1)` using result field 9; instance ID, primitive ID, and
distance keep the same field mapping. Static triangle/curve traversal and
direct primitive/instance-motion triangle/curve traversal are runtime-tested.
Direct procedural closest/any remains outside this family because Luisa's
procedural callbacks require the stateful query API below.

### 8.9 Stateful intersection-query lifecycle and intrinsics

#### 8.9.1 Native lifecycle and exact declarations

The native query pointee is the opaque type from Section 4.12. Each local XIR
query object produces one `allocate` call, one `reset`, any number of
`next`/getter/commit/abort calls, and one `deallocate` on each function exit.
The declaration spellings and LLVM 21 signatures are:

~~~llvm
declare ptr @air.allocate_intersection_query<query-suffix>()

declare void @air.deallocate_intersection_query<query-suffix>(
    ptr %query)

declare void @air.reset_intersection_query<query-suffix>(
    ptr %query,
    <3 x float> %origin, <3 x float> %direction,
    float %t_min, float %t_max,
    ptr addrspace(1) readonly %accel, i32 %mask,
    i32 %winding, i32 %triangle_culling,
    i32 %geometry_culling, i32 %opacity_culling,
    i32 %forced_opacity, i32 %geometry_type,
    i32 %curve_basis, i32 %curve_type,
    i32 %curve_control_point_count,
    i1 %assume_identity_transforms, i1 %accept_any)

declare i1 @air.next_intersection_query<query-suffix>(
    ptr %query)

declare void
@air.commit_triangle_intersection_intersection_query<query-suffix>(
    ptr %query)

declare void
@air.commit_bounding_box_intersection_intersection_query<query-suffix>(
    ptr %query, float %distance)

declare void @air.abort_intersection_query<query-suffix>(
    ptr %query)
~~~

`<query-suffix>` is `.instancing.triangle_data` without curve metadata and
`.instancing.triangle_data.curve_data` when `CurveBasisMD` is nonempty. The
stateful Metal 4 query API does not accept the `.primitive_motion` or
`.instance_motion` tags and has no reset-time ray-time parameter. Motion query
constructors therefore fail preflight instead of selecting a guessed suffix.

The repeated `intersection_query` in the two commit names is intentional and
matches the Apple oracle. Allocate, reset, next, both commits, abort, and
deallocate are declared exactly `mustprogress nounwind willreturn`. They are
not declared read-only or no-free. Reset's acceleration-pointer parameter is
individually `readonly`; that parameter attribute does not make the reset
operation read-only because it mutates the query object.

All getters have one query argument and use the name grammar
`air.<getter>_intersection_query<query-suffix>`:

| `<getter>` | Return type |
|---|---|
| `get_candidate_intersection_type` | `i32` |
| `get_candidate_instance_id` | `i32` |
| `get_candidate_primitive_id` | `i32` |
| `get_candidate_triangle_barycentric_coord` | `<2 x float>` |
| `get_candidate_triangle_distance` | `float` |
| `get_candidate_curve_parameter` | `float` |
| `get_candidate_curve_distance` | `float` |
| `get_world_space_ray_origin` | `<3 x float>` |
| `get_world_space_ray_direction` | `<3 x float>` |
| `get_candidate_ray_origin` | `<3 x float>` |
| `get_candidate_ray_direction` | `<3 x float>` |
| `get_ray_min_distance` | `float` |
| `get_committed_distance` | `float` |
| `get_committed_intersection_type` | `i32` |
| `get_committed_instance_id` | `i32` |
| `get_committed_primitive_id` | `i32` |
| `get_committed_triangle_barycentric_coord` | `<2 x float>` |
| `get_committed_curve_parameter` | `float` |

Getter declarations carry
`argmemonly mustprogress nofree nounwind readonly willreturn`; argument zero is
`nocapture readonly`. These are query-state reads, not `readnone` functions.

Opaque pointers do not preserve the AIR pointee types by themselves. Before
the LLVM 14 writer runs, the declarations carry these element-type metadata:

- allocate: `ret_eltype = %struct._intersection_query_t`;
- every non-allocate intrinsic: argument zero in `arg_eltypes` is
  `%struct._intersection_query_t`;
- reset: argument five in `arg_eltypes` is additionally
  `%struct._instance_acceleration_structure_t`.

The downgrade therefore reconstructs
`%struct._intersection_query_t*` in generic address space zero and the reset
acceleration argument as
`%struct._instance_acceleration_structure_t addrspace(1)*`, rather than
inventing byte pointers.

#### 8.9.2 Reset controls

Reset receives the Luisa ray as explicit `<3 x float>` origin and direction,
followed by `t_min`, `t_max`, the instanced-acceleration handle, and the
zero-extended 32-bit visibility mask. The remaining controls are constants.
In exact order after the mask they are:

~~~text
triangle-only query:
  0, 0, 0, 0, 0, 1, 0, 0, 0, false, accept_any

query with a procedural-bounding-box branch:
  0, 0, 0, 0, 0, 3, 0, 0, 0, false, accept_any

triangle plus curve query:
  0, 0, 0, 0, 0, 5, basis, 0, control_points, false, accept_any

triangle plus procedural-bounding-box plus curve query:
  0, 0, 0, 0, 0, 7, basis, 0, control_points, false, accept_any
~~~

The nine integer positions mean winding, triangle culling, geometry culling,
opacity culling, forced opacity, geometry type, curve basis, curve type, and
curve control-point count. Geometry type bits are the same as direct tracing:
`1` triangles, `2` bounding boxes, and `4` curves. The first boolean is
`assume_identity_transforms` and remains false. `LC_RayQueryAll` uses
`accept_any = false`; `LC_RayQueryAny` uses `accept_any = true`.

The geometry flag is selected from the normalized XIR query uses. Reading a
procedural candidate, testing whether the candidate is procedural, or
committing a procedural distance requires geometry type `3`. No intersection-
function table or payload argument participates in this stateful family.

#### 8.9.3 XIR operation semantics

The ray-query-loop lowering runs before CFG destructuring. By the AIR boundary
the callback-shaped AST construct has become ordinary control flow containing
`RayQueryObjectReadInst` and `RayQueryObjectWriteInst` operations on the local
opaque query pointer.

The direct AST `RAY_QUERY_PROCEED(query)` expression follows the same object
protocol. AST-to-XIR emits a `RAY_QUERY_OBJECT_PROCEED` write whose operand is
the query lvalue, then reads `RAY_QUERY_OBJECT_IS_TERMINATED` from that same
lvalue, and returns its boolean bitwise inverse. Thus the public `proceed()`
result is true when a candidate is available; the internal terminated flag is
true when native `next` reported no candidate.

- `PROCEED` calls native `next` exactly once and caches its `i1` result for
  that query.
- `IS_TERMINATED` is the logical inverse of the cached result. It does not
  call `next` again and is invalid before a corresponding `PROCEED`.
- Candidate kind compares the native candidate-intersection type with `1` for
  triangle, `2` for procedural/bounding-box, or `3` for curve. Luisa's
  triangle/surface predicate accepts both `1` and `3` when curve metadata is
  present.
- `WORLD_SPACE_RAY` reads native origin, direction, and minimum distance. Its
  Luisa `t_max` is the current committed distance, so it tightens when a hit is
  committed.
- `CANDIDATE_OBJECT_SPACE_RAY` reads
  `get_candidate_ray_origin_intersection_query` and
  `get_candidate_ray_direction_intersection_query`. It uses the same native
  minimum-distance and current committed-distance getters for `t_min` and
  `t_max`. These exact names and signatures were confirmed with a Metal 4.0
  source oracle emitted by the installed Apple compiler, rather than inferred
  by renaming the world-space getters.
- Triangle/surface commit calls the native curve-commit intrinsic for native
  type `3` and the triangle-commit intrinsic otherwise.
- Procedural commit calls the bounding-box commit only when the supplied
  distance is ordered and lies in the inclusive interval
  `[ray_min_distance, committed_distance]`. NaN and out-of-range distances are
  ignored.
- `TERMINATE` calls native abort.

Candidate and committed-hit values are rebuilt in Luisa register form:

| XIR read | Luisa fields | Native sources |
|---|---|---|
| triangle candidate | instance, primitive, barycentrics, distance | candidate instance ID, primitive ID, triangle barycentrics, triangle distance |
| curve candidate | instance, primitive, `(parameter, -1)`, distance | candidate instance ID, primitive ID, curve parameter, curve distance |
| procedural candidate | instance, primitive | candidate instance ID, primitive ID |
| committed hit | instance, primitive, barycentrics, kind, distance | committed instance ID, primitive ID, triangle barycentrics or `(curve parameter, -1)`, mapped type, committed distance |

Committed native type `0` maps to Luisa kind `0` (none), type `1` maps to kind
`1` (surface/triangle), type `3` also maps to kind `1` (surface/curve), and
every other supported nonzero type maps to kind `2` (procedural). On a miss,
the instance ID is explicitly normalized to
`~0u`; the other fields retain their native getter values and must be ignored
when kind is zero.

#### 8.9.4 Preflight boundary

AIR accepts only the normalized, compiler-owned lifecycle above:

- the custom type description must be exactly `LC_RayQueryAll` or
  `LC_RayQueryAny`;
- each query must be a local, non-shared `AllocaInst` in the function-entry
  block, initialized exactly once by the matching `RAY_TRACING_QUERY_ALL` or
  `RAY_TRACING_QUERY_ANY` resource query;
- construction must have acceleration, Luisa `Ray`, and `uint` mask operands,
  and its only materialization store must target that same allocation;
- query objects may be used only by the recognized read/write operations;
  they may not be loaded as ordinary data, indexed, passed to another
  function, returned, placed in a resource, or otherwise escape;
- the supported reads are termination, candidate-kind, world ray, candidate
  object-space ray, triangle candidate, procedural candidate, and committed
  hit; the supported writes are proceed, triangle commit, procedural commit,
  and terminate;
- `IS_TERMINATED` must immediately follow `PROCEED` for the same query in the
  normalized traversal, and procedural operations select the bounding-box
  geometry flag;
- static curve controls are taken from `CurveBasisMD`; motion query
  constructors fail with the explicit diagnostic that Metal 4
  `intersection_query` accepts neither motion acceleration structures nor a
  ray-time operand.

These constraints are checked before LLVM construction in addition to the
emitter's internal assertions. The intrinsic signatures and translations in
this section are oracle-derived compatibility requirements; runtime-validation
status is tracked separately in Section 16.

### 8.10 Native shader-log format and vararg translation

Each unique `(Luisa format string, record type)` is still assigned a stable
token during metadata traversal, but the token is used only to deduplicate a
constant native format string. No record is written to device memory. Direct
and indirect entries regenerate the table independently and require identical
results so AOT/cache metadata remains deterministic.

Luisa `{}` placeholders become native scalar format specifiers recursively:

| Luisa leaf | Native format | Vararg representation |
|---|---|---|
| `bool` | marked `%d` | zero-extended `i32` |
| `int8`/`int16`/`int32` | `%d` | narrow values sign-extended to `i32` |
| `uint8`/`uint16`/`uint32` | `%u` | narrow values zero-extended to `i32` |
| `int64`/`uint64` | `%ld`/`%lu` | `i64` |
| `half`/`float` | `%g` | extended `double` |

Vectors use `(a, b)`, arrays use `[a, b]`, matrices use `<column, ...>`, and
structures use `{member, ...}`. Literal `{{` and `}}` become braces and a
literal percent is doubled for the native formatter. Resources, opaque custom
values, `double`, and a native format of 1024 bytes or more fail preflight or
emission.

The emitter flattens every aggregate into scalar varargs and computes the
packed argument byte count with four-byte alignment for 32-bit values and
eight-byte alignment for 64-bit/double values. Boolean output uses a private
marker around `%d`; the `MTL::LogState` handler maps marker values back to
`false`/`true` before invoking `DeviceInterface::StreamLogCallback`. Messages
outside Luisa's subsystem/category are ignored. With no user callback, the
same normalized message goes to the ordinary Luisa logger.

### 8.11 Math symbol grammar

All emitted math declarations are homogeneous scalar functions:

~~~text
half  air.<op>.f16(half...)
float air.<op>.f32(float...)
float air.fast_<op>.f32(float...)
~~~

There are no emitted vector AIR math overloads. Vector XIR operations are
scalarized lane by lane:

1. create a poison result vector;
2. extract each input lane;
3. call the scalar AIR function;
4. insert each result lane.

Every math declaration is marked:

- `mustprogress`;
- `nofree`;
- `nosync`;
- `nounwind`;
- `willreturn`;
- no memory access;
- `speculatable`.

### 8.12 Direct AIR math mapping

| XIR operation | AIR base name | Arity |
|---|---|---:|
| floating `ABS` | `fabs` | 1 |
| `SATURATE` | `saturate` | 1 |
| `ACOS` | `acos` | 1 |
| `ACOSH` | `acosh` | 1 |
| `ASIN` | `asin` | 1 |
| `ASINH` | `asinh` | 1 |
| `ATAN` | `atan` | 1 |
| `ATAN2` | `atan2` | 2 |
| `ATANH` | `atanh` | 1 |
| `COS` | `cos` | 1 |
| `COSH` | `cosh` | 1 |
| `SIN` | `sin` | 1 |
| `SINH` | `sinh` | 1 |
| `TAN` | `tan` | 1 |
| `TANH` | `tanh` | 1 |
| `EXP` | `exp` | 1 |
| `EXP2` | `exp2` | 1 |
| `EXP10` | `exp10` | 1 |
| `LOG` | `log` | 1 |
| `LOG2` | `log2` | 1 |
| `LOG10` | `log10` | 1 |
| `POW` | `pow` | 2 |
| `SQRT` | `sqrt` | 1 |
| `RSQRT` | `rsqrt` | 1 |
| `CEIL` | `ceil` | 1 |
| `FLOOR` | `floor` | 1 |
| `FRACT` | `fract` | 1 |
| `TRUNC` | `trunc` | 1 |
| `ROUND` | `round` | 1 |
| `RINT` | `rint` | 1 |
| `FMA` | `fma` | 3 |
| floating `MIN` | `fmin` | 2 |
| floating `MAX` | `fmax` | 2 |
| floating `CLAMP` | `clamp` | 3 |

Compound mappings:

| XIR operation | AIR function used inside expansion |
|---|---|
| `POW_INT` | signed integer exponent converted to the base FP type, then `pow` |
| `SMOOTHSTEP` | `clamp`, followed by the cubic polynomial |
| `LENGTH` | ordinary dot expansion, then `sqrt` |
| `NORMALIZE` | ordinary dot expansion, then `rsqrt` |
| floating `REDUCE_MIN` | repeated scalar `fmin` |
| floating `REDUCE_MAX` | repeated scalar `fmax` |

### 8.13 Fast versus non-fast names

The `fast_` prefix is selected only when:

1. `enable_fast_math` is true;
2. the scalar type is `float`, not `half`;
3. the base name is not in this exclusion set:
   `fma`, `fabs`, `copysign`, `fmin`, `fmax`, `clamp`,
   `saturate`.

Consequences:

- Half-precision calls always use unprefixed names.
- Accurate float mode uses unprefixed names.
- Fast float transcendental and rounding calls use
  `air.fast_<op>.f32`.
- `fma`, `fabs`, `fmin`, `fmax`, `clamp`, and `saturate`
  remain unprefixed.
- `copysign` is reserved in the exclusion policy but is currently expanded
  with integer bit operations, so no `air.copysign.*` call is emitted.

Fast math is also expressed by entry/implementation function attributes and
`air.compile.fast_math_enable`. Ordinary floating-point instructions receive
LLVM `fast` flags when this mode is enabled, matching `metalfe`; accurate mode
leaves those flags unset. The implementation functions carry the same
attributes as the public entry so LLVM inlining cannot conservatively downgrade
the entry after it absorbs the kernel or raster-stage body.

### 8.14 Device debug trap

An Apple Metal 3.2 oracle with `__builtin_debugtrap()` behind a device-buffer
condition emits the ordinary LLVM debug-trap intrinsic:

~~~llvm
tail call void @llvm.debugtrap()

; Function Attrs: nounwind
declare void @llvm.debugtrap()
~~~

The `tail` marker is a consequence of the oracle's trap block immediately
joining its return block, not part of the intrinsic ABI. `llvm.debugtrap` has
no result, overload type, or operands. Luisa therefore lowers `DebugBreakInst`
through LLVM's `debugtrap` intrinsic without forwarding its watch values or
callback. Those fields remain XIR/debugger-side state, matching the CUDA and
HIP LLVM lowering convention rather than becoming AIR call arguments.

The XIR verifier still requires every watch to be an ordinary data rvalue, and
AIR preflight validates each watch's type before accepting the instruction.
This matters even though the intrinsic itself has no operands: the watched
expressions remain part of the source-level diagnostic operation. The strict
runtime regression puts the debug break behind a zero-valued device-buffer
condition. Because the condition is not a compile-time constant, the intrinsic
survives LLVM optimization and pipeline creation, while dispatch does not
execute the trap.

## 9. Arithmetic translation conventions

### 9.1 Ordinary LLVM operations

These operations intentionally do not use named AIR intrinsics:

- integer add, subtract, multiply, divide, remainder, bitwise operations,
  shifts, comparisons, and rotates;
- floating add, subtract, multiply, divide, and remainder;
- comparisons and select;
- integer `min`, `max`, and `clamp`;
- `lerp`, `step`, `dot`, `cross`, `reflect`, and
  `faceforward`;
- `isinf`, `isnan`, and `copysign`;
- sum/product reductions;
- all matrix operations;
- aggregate construction, shuffle, extract, and insert.

Rotates are expanded as masked shifts and ORs. The emitter does not explicitly
request `llvm.fshl` or `llvm.fshr`.

### 9.2 Comparisons and NaN

Floating comparisons use:

| XIR comparison | LLVM predicate |
|---|---|
| `<` | ordered less-than |
| `>` | ordered greater-than |
| `<=` | ordered less-or-equal |
| `>=` | ordered greater-or-equal |
| `==` | ordered equal |
| `!=` | unordered not-equal |

The unordered `!=` is deliberate: `NaN != NaN` evaluates true.

Non-fast Metal semantics are preserved by:

- `MIN` / `MAX`: AIR `fmin` / `fmax`;
- `CLAMP`: AIR `clamp`;
- `SATURATE`: AIR `saturate`;
- `SMOOTHSTEP`: AIR clamp before the polynomial;
- `STEP`: ordered `x >= edge`, therefore NaN returns zero;
- float reduce-min/max: AIR `fmin` / `fmax` folds.

The strict runtime test currently asserts:

~~~text
min(1, NaN)              = 1
min(NaN, 1)              = 1
max(1, NaN)              = 1
max(NaN, 1)              = 1
clamp(NaN, 0, 1)         = 0
saturate(NaN)            = 0
step(0, NaN)             = 0
reduce_min(NaN,2,3,4)    = 2
reduce_max(NaN,2,3,4)    = 4
smoothstep(0,1,NaN)      = 0
~~~

`isinf` and `isnan` use explicit IEEE-754 masks for f16/f32.
`copysign` clears and replaces the sign bit with integer operations.

### 9.3 Reductions and vector geometry

- `ALL` and `ANY` are left folds with boolean AND/OR.
- Integer sum/product reductions use integer zero/one identities and
  `add`/`mul`.
- Floating sum/product reductions use floating zero/one and
  `fadd`/`fmul`.
- Min/max reductions start at lane zero and fold the remaining lanes.
- `DOT` is multiply followed by a scalar sum.
- `LENGTH_SQUARED` is a dot with self.
- `LENGTH` applies AIR `sqrt`.
- `NORMALIZE` applies AIR `rsqrt` and splats the result.
- `CROSS`, `FACEFORWARD`, and `REFLECT` are explicit vector formulas.

Reduction order is deterministic and sequential in the emitted IR unless LLVM
can legally reassociate it under the active floating-point policy.

### 9.4 Matrices

Matrix component operations iterate over columns. Linear-algebra multiply
forms each result column as a weighted sum of the left matrix's columns.
Transpose explicitly exchanges row/column indices.

Determinant enumerates permutations. Inverse computes cofactors, transposes
them while constructing the result, and divides by the determinant. This is
compact and dimension-generic, but it is not a numerically specialized or
high-performance matrix intrinsic implementation.

### 9.5 Casts

Static casts use:

- vector splat for scalar-to-vector;
- compare/select for conversions involving booleans;
- `fpext`/`fptrunc`-style FP casts;
- signed or unsigned FP/integer conversions based on XIR type;
- signed or unsigned integer extension/truncation based on XIR type.

Bitwise casts use direct LLVM `bitcast` for non-boolean scalar/vector pairs.
Other cases spill through memory and reload using the target memory type.

## 10. CFG and SSA translation

Each translated function starts with:

~~~text
alloca -> entry -> translated XIR body
~~~

All LLVM basic blocks are created before instruction translation. The emitter
walks the XIR dominance tree so definitions are normally produced before
uses. PHI instructions are created on the first pass, while incoming
`(value, predecessor)` pairs are attached after all blocks and values exist.

Structured XIR operations lower to ordinary LLVM CFG:

| XIR | LLVM |
|---|---|
| `IfInst`, conditional branch | `condbr` |
| `SwitchInst` | `switch` |
| loop/simple loop | branch to prepare/body |
| break/continue | branch to explicit target |
| branch | `br` |
| return | `ret` |
| unreachable | conservative `ret void` or `ret poison` |

Every translated XIR block must end with an LLVM terminator.

XIR `ASSUME` becomes LLVM `llvm.assume`. The hand-written emitter does not
explicitly request other LLVM intrinsics.

## 11. Memory and resource operations

### 11.1 Loads, stores, and allocas

Loads and stores carry explicit Luisa type alignment and perform recursive
register/memory conversion. Explicit XIR local allocations become address-
space-0 allocas. Both XIR locals and compiler-created reinterpretation/CAS
temporaries are created in the synthetic function-entry alloca block. They are
not emitted at the current translated control-flow position; this avoids
loop-local stacksave/stackrestore behavior in Apple AIR.

XIR shared allocations become fixed-size internal globals in address space 3,
initialized with poison and explicitly aligned. Dynamic threadgroup memory is
not part of the current ABI.

### 11.2 GEP and aggregate access

- Vector, matrix, and array access use element-memory GEP.
- Structure indices must be constants and use cached byte offsets.
- Vector aggregate access stays in SSA.
- Constant one-level structure/array extraction uses LLVM
  `extractvalue`/`insertvalue`.
- Dynamic or nested aggregate access spills to memory, applies the normal
  access chain, and reloads.

### 11.3 Buffer addressing

Buffer access:

1. extracts the address-space-1 pointer;
2. converts the index to `i64`;
3. multiplies by a byte stride;
4. performs an in-bounds `i8` GEP;
5. loads/stores the requested type.

| XIR resource operation | Convention |
|---|---|
| typed buffer size | byte size divided by element size |
| byte-buffer size | raw byte size |
| buffer device address | `ptrtoint` |
| typed read/write | index multiplied by `sizeof(T)` |
| byte-buffer read/write | index already is a byte offset |
| device-address read/write | integer converted to address-space-1 pointer |
| volatile typed/byte read | device fence, then LLVM volatile load |
| volatile typed/byte write | LLVM volatile store, then device fence |

No resource bounds checks are generated. Byte-buffer offsets and raw device
addresses must satisfy the requested type's alignment. Because the emitter
uses in-bounds GEP and aligned memory operations, violating either contract
can produce LLVM undefined behavior rather than a recoverable GPU fault.

### 11.4 Direct texture translation summary

Direct texture arguments are normalized to immediate resource-query/read/write
uses by the XIR pipeline and inline-all boundary. Access inference scans those
uses before type-name and reflection generation. A size-only texture defaults
to read access. Sample plus read/write is rejected rather than assigned an
invalid combined AIR qualifier.

AIR access qualifiers and Metal runtime residency flags are related but not
identical. For both compute and raster, a direct texture whose Luisa argument
usage includes `READ` is conservatively declared with
`MTL::ResourceUsageRead | MTL::ResourceUsageSample`; write usage additionally
contributes `MTL::ResourceUsageWrite`. This covers a root argument that is
used by integer-coordinate reads, filtered sampling, or both even though the
host-side `Usage` mask does not preserve that distinction. Raster declarations
also carry the argument's vertex/fragment stage mask. The extra `Sample` bit
does not change the AIR reflection access code listed in Section 8.7.

`DepthBuffer::to_img()` follows the public read-only policy: a Metal depth
texture may be passed through a direct `Image<float>`/`ImageView<float>` root
only when its usage contains no write bit, and only mip level zero exists.
`MetalDepthBuffer` and ordinary `MetalTexture` share the polymorphic
`MetalTextureBase` handle/binding interface, so the image alias retains a
valid base pointer instead of relying on an invalid downcast to the color-
texture class. The base carries an explicit `TEXTURE`/`DEPTH` kind used by
destruction, naming, copies, presentation, color/depth attachment setup, and
shader binding to keep those operations type-safe. A depth object is created
with render-target plus shader-read usage; aliasing it does not make shader
writes, color attachment use, presentation, or nonzero mips legal. General
texture blit/copy commands resolve through the same base interface and retain
the underlying depth-texture identity.

Texture reads and samples return `{vector, i8}`; the emitter extracts the
vector and discards the status byte. Writes have no result. Component size
queries build a Luisa `uint2` or `uint3` one scalar AIR call at a time. See
Section 8.7 for the exact function signatures, access operands, sampler table,
and AIR-version split.

### 11.5 Bindless translation

AST-to-XIR treats the `TYPED_BINDLESS_*` and
`TYPED_UNIFORM_BINDLESS_*` call families as frontend aliases for the existing
bindless XIR operations. The normalization covers 2D/3D sampling in all four
LOD/gradient forms with either slot or explicit samplers, texture size and
integer-coordinate reads with optional mip levels, typed-buffer size, device
address, read and write, and byte-buffer read. The typed and uniform prefixes
do not survive as distinct XIR opcodes: element type, result type, and operand
shape retain the information needed by AIR reflection and lowering. The
removed bindless-buffer-type query and cooperative operations are not part of
this alias normalization.

At runtime, traversing a bindless array's tracked texture resources likewise
declares `MTL::ResourceUsageRead | MTL::ResourceUsageSample` for compute or
the selected raster stages. The buffer tracker retains the argument's ordinary
read/write mask. This conservative split is necessary because a bindless slot
does not expose whether a later AIR access will be an integer read or a sample
to the host encoder.

The emitter implements the reconstructed ABI with the following lowering:

- compute `items + index` with a 32-byte slot stride;
- load the address-space-1 buffer pointer from offset 0;
- mask the low 48 bits of the packed `i64` at offset 8 for byte size;
- extract sampler codes from bits 48..55 and 56..63;
- load the 2D/3D one-field texture wrapper at offset 16/24 and extract its
  typed resource pointer;
- multiply typed-buffer indices by the requested value type's Luisa byte size;
- leave byte-buffer offsets byte-addressed;
- divide masked byte size by the explicit stride for typed size queries;
- use dynamic mip levels for bindless texture size/read operations;
- use the slot sampler code unless the XIR `_SAMPLER` operation supplies an
  explicit filter/address pair.

Bindless integer-coordinate texture reads still use the texture reflected as
`sample`, obtain `air.get_read_sampler`, and pass final AIR access operand
zero in Apple-generated oracles. Those details differ from direct read-only
texture arguments and must not be copied from the direct path blindly. The
LLVM emission, support preflight, typed-pointer downgrade, Metal pipeline-state
creation, and representative strict runtime dispatches are all enabled and
validated as recorded in Sections 4.8 and 16.1.

### 11.6 `PACK` and `UNPACK` object representation

AST-to-XIR lowers packing through a synthetic one-member Luisa structure:

~~~text
storage_type = struct { T value }   // alignment is at least 4
word_count   = sizeof(storage_type) / 4
packed_type  = array<uint, word_count>
~~~

`PACK(value, buffer<uint>, offset)` constructs the wrapper, bitwise-casts it
to `packed_type`, and writes consecutive words. `UNPACK` reads the same words,
bitwise-casts them back to the wrapper, and extracts member zero. Resources and
opaque custom values are not packable. Both AST validation and XIR lowering
require a `uint` word buffer and `uint` offset.

XIR cast validation has a narrow storage-aggregate rule for this operation: a
bitwise cast between structures/arrays is valid only when the two Luisa
`Type::size()` values are equal. This does not weaken ordinary value bitcasts;
direct boolean scalar/vector bitcasts remain invalid, and non-boolean scalar/
vector bitcasts still require equal logical register widths. AIR lowers an
aggregate storage cast through a private temporary using the greater source/
target alignment rather than pretending it is a native aggregate LLVM
`bitcast`.

The wrapper is deliberate. Scalar `bool`, `byte`, and `short` values are
rounded to one four-byte word, while naturally wider values retain their
Luisa alignment and tail padding. A `float3` therefore occupies four words,
not three. `{bool, bool, bool, bool}` and `byte4` each occupy one word, with one
byte per logical element.

Register-to-memory conversion zero-initializes the complete destination before
inserting logical members, and compiler-created spill/reload storage uses the
maximum of source and target alignment. Padding bytes are consequently stable
zeros rather than LLVM poison. On little-endian Metal, current strict tests
observe these exact words:

| Value | First packed word |
|---|---:|
| `{true, false, true, true}` as four scalar bool fields | `0x01010001` |
| `byte4{1, 2, 3, 4}` | `0x04030201` |
| scalar `true` | `0x00000001` |
| scalar byte `0x5a` | `0x0000005a` |
| scalar short `0x1234` | `0x00001234` |

The AST usage contract is asymmetric: the value and offset are `READ`, the
`PACK` destination buffer is `WRITE`, and `UNPACK` follows the default
read-only argument rule. This matters because resource reflection and backend
normalization consume the propagated usage.

## 12. Supported subset and fail-closed boundary

The AIR preflight requires exactly one kernel for compute modules.
The allowlist below describes what the emitter accepts; it is not, by itself,
a claim that every path has passed strict Metal runtime validation.

Supported types:

- bool;
- signed and unsigned 8/16/32/64-bit integers;
- half and float;
- recursive vectors, matrices, arrays, structures, and buffers;
- direct 2D/3D textures of float, int, or uint, subject to the access rules in
  Section 4.7;
- bindless arrays with the 8-byte wrapper and 32-byte slot ABI in Section 4.8;
- instanced triangle/curve acceleration structures with the 16-byte `LCAccel`
  and 72-byte indirect `LCInstance` ABI in Section 4.10;
- the `LC_IndirectDispatchBuffer` custom handle described in Section 4.9;
- local, compiler-owned `LC_RayQueryAll` and `LC_RayQueryAny` native query
  objects with the lifecycle in Sections 4.12 and 8.9.

Supported instruction families:

- core CFG, PHI, alloca, load/store, GEP, cast, and assume;
- calls to internal XIR callables;
- the arithmetic operations explicitly listed by the emitter;
- block synchronization and the SIMD/subgroup operations in Section 8.6;
- 32-bit device and threadgroup atomics described in Section 8.5;
- typed-buffer, byte-buffer, and raw-device-address query/read/write, including
  fenced volatile typed/byte-buffer access;
- direct texture size/read/write for 2D and 3D float/int/uint textures;
- direct 2D/3D float sampling with implicit LOD, explicit LOD, gradients, or
  gradients plus minimum-LOD clamp;
- bindless typed/byte-buffer size, device address, read, and write; bindless
  2D/3D float-texture size and integer-coordinate read with optional mip level;
  and bindless 2D/3D float sampling in every direct-sampling mode, using either
  the sampler stored in the slot or an explicit filter/address pair;
- GPU-written indirect-dispatch count and kernel records, including the
  capacity guard and host binding offset;
- static and direct primitive/instance-motion triangle/curve closest-hit and
  any-hit tracing; instance
  transform, user-ID, and visibility-mask queries; and instance transform,
  visibility-mask, opacity, and user-ID writes;
- stateful `LC_RayQueryAll`/`LC_RayQueryAny` traversal over static triangle,
  curve, and procedural-bounding-box geometry, including candidate inspection,
  world-space and candidate object-space ray inspection,
  triangle/curve/procedural commit, committed-hit inspection, and abort;
- AST autodiff scopes, gradient markers, detach, and `BACKWARD`, after the XIR
  autodiff pipeline has replaced them with ordinary supported operations;
- AST `PACK`/`UNPACK` lowered to ordinary bitwise casts and uint-buffer
  operations as described in Section 11.6;
- XIR device printing for plain supported scalar/aggregate arguments through
  `air.os_log`, `MTL::LogState`, and the stream callback described in
  Sections 4.11 and 8.10;
- XIR debug break through the zero-operand `llvm.debugtrap` intrinsic described
  in Section 8.14; watch values and callbacks are not AIR call operands;
- XIR assertions with the same current semantics as the MSL backend: accepted
  as no-ops until a shared Metal device-side assertion ABI is defined.
- external function declarations and calls whose exact definitions are
  supplied by `ShaderOption::native_include` as target-compatible LLVM IR or
  bitcode, linked before LLVM optimization and downgrade;
- cooperative-vector load/store, workgroup load/store, splat, scalar cast,
  and 32-bit integer/float accumulation using the XIR byte-offset ABI;
- vertex and fragment raster stages with fixed Luisa `AppData` reconstruction,
  scalar/vector varyings, color render targets, per-stage root arguments,
  vertex/instance/base-instance/primitive/object IDs, floating barycentrics,
  fragment front-facing state, selectable center/centroid/sample and
  perspective/no-perspective interpolation, derivatives, and discard as
  described in Section 12.2.

Not currently supported in AIR:

- direct procedural closest/any tracing; procedural support requires explicit
  stateful bounding-box candidate traversal and commit;
- stateful motion-blur ray queries, because Metal 4 `intersection_query`
  accepts neither a motion-acceleration-structure tag nor a ray-time operand;
- clock and unresolved external functions or calls;
- double, float8, cooperative-matrix operations, and custom types other than the
  indirect-dispatch handle and the two local ray-query types;
- extended acceleration-limit shaders;
- raster debug-break state, stencil, and conservative rasterization.

Apple's Metal 3.0 and 3.2 frontends accept `__builtin_readcyclecounter()` and
emit `llvm.readcyclecounter`, and `metallib` accepts the resulting AIR. However,
compute-pipeline creation for both the Luisa-generated module and an otherwise
standalone Apple-frontend oracle failed with an XPC compiler-service
interruption on the tested Apple M1 Max/macOS 26 system. `ClockInst` therefore
remains rejected by AIR preflight until the supported device/OS boundary is
known; emitting the accepted IR alone is not sufficient evidence of runtime
support.

Unsupported compute and raster features fail shader creation; Metal4 never
dispatches an unsupported kernel through another code generator. The
compute-rendering CTests in Section 16.1 therefore validate the AIR path by
construction. Raster shader creation uses the paired XIR/AIR vertex/fragment
path and rejects unsupported raster features rather than silently compiling
one stage through a different ABI. Direct compute texture operations, static
and motion triangle/curve traces, instance access, the local static
triangle/curve/procedural-bounding-box query subset above, and the raster
subset in Section 12.2 are AIR-native. Stateful motion queries and extended
acceleration-limit shaders remain outside that boundary.
The documented bindless buffer, texture-read, mip-query, and sampling paths
also pass preflight and have dedicated strict AIR runtime coverage.

Reverse autodiff is normalized entirely in the XIR pipeline before LLVM
emission. A raw AST `BACKWARD` operation must not reach LLVM CodeGen; failed
normalization or unsupported output reports an error instead of changing
backend or IR pipeline.

### 12.1 External AST-to-XIR lowering and native LLVM linkage

The shared AST-to-XIR translator preserves external declarations and call
sites, and Metal AIR emits matching LLVM declarations.
Declarations are deduplicated by the AST external-function hash, keep their
source name and return type (including `void`), and derive argument form from
the AST type and `Usage`:

| AST argument | XIR external argument/call operand |
|---|---|
| non-resource `READ` or `NONE` | value |
| non-resource `WRITE` or `READ_WRITE` | reference; call operand must be an lvalue |
| resource | resource value, retaining reference-like resource semantics |
| opaque custom type | reference; call operand must be an lvalue |

The same opaque-custom rule applies to ordinary custom callables, and AST
usage propagation consults the callee's recorded usage for those arguments.
`TypeIDExpr` currently lowers to `uint64(0)` for parity with the source Metal
and CUDA paths until a stable cross-backend type-ID ABI exists. `StringIDExpr`
lowers to the 64-bit Luisa hash of its string contents.

For each used declaration, `ShaderOption::native_include` must provide a
definition in textual LLVM IR or bitcode. Metal4 validates the target triple,
data layout, function type, address space, calling convention, ABI attributes,
and reference alignment, then links only the needed definitions before O2 and
LLVM-14 downgrade. Value arguments use their register ABI, reference arguments
use generic address-space pointers, and no Luisa internal state parameters are
appended. Missing or incompatible definitions fail shader creation.

### 12.2 Raster AIR conventions and implemented ABI

Vertex and fragment AIR are distinct program kinds, not compute kernels with a
few additional built-ins. Apple-generated modules register them in
`!air.vertex` and `!air.fragment`, respectively. Each entry record still has
the familiar shape `(function, return-info, argument-info)`, but stage I/O is
encoded in the latter two metadata nodes.

Observed vertex conventions:

- the function returns a scalar or aggregate containing all stage outputs;
- clip-space position is reflected with `air.position`;
- interpolants use `air.vertex_output`, `user(locnN)`;
- attribute parameters use `air.vertex_input`, `air.location_index`, a
  location, and a count;
- built-ins include `air.vertex_id`, `air.instance_id`, and
  `air.base_instance`.

An observed vertex return/argument description is equivalent to:

~~~text
return:   air.position float4
          air.vertex_output user(locn0) float3
          air.vertex_output user(locn1) float2
arguments: air.vertex_input location 0 float3
           air.vertex_input location 1 float3
           air.vertex_input location 3 float2
           air.vertex_id uint
           air.instance_id uint
           air.base_instance uint
~~~

Observed fragment conventions:

- position input is `air.position`, `air.center`, `air.no_perspective`;
- user varying inputs are `air.fragment_input`, `user(locnN)` or an Apple
  generated name, plus interpolation metadata;
- floating varyings use one location token (`air.center`, `air.centroid`, or
  `air.sample`) followed by one correction token (`air.perspective` or
  `air.no_perspective`); integer varyings require the single `air.flat` token;
- fragment built-ins include `air.primitive_id`, `air.front_facing`, and
  `air.barycentric_coord`; the barycentric value is floating point (the
  observed Metal value is a three-component floating vector), not an object
  ID or integer payload;
- color return components use `air.render_target`, target index, and a fixed
  secondary index of zero;
- depth is another returned component reflected with `air.depth`, followed by
  `air.depth_qualifier` and `air.any`, `air.greater`, or `air.less`.

The Apple-frontend oracle for `bool front_facing [[front_facing]]` emits a
physical `i1 noundef` fragment-entry parameter; it is not an intrinsic. Its
argument metadata is exactly the entry index followed by `air.front_facing`,
`air.arg_type_name`, `bool`, `air.arg_name`, and `front_facing`. This is a
register ABI: the use of `i1` here does not change Luisa's byte-addressable
memory ABI, where a scalar bool and each member of `{bool, bool, bool, bool}`
occupy one byte. The convention was recovered with `xcrun metal -std=metal3.2
-S -emit-llvm -c` and then validated through Luisa's LLVM-21 emission,
LLVM-14 downgrade, `metallib`, pipeline creation, and GPU execution.

The selectable interpolation spellings were recovered from a single
Metal-3.2 fragment oracle containing `center_perspective`,
`center_no_perspective`, `centroid_perspective`,
`centroid_no_perspective`, `sample_perspective`,
`sample_no_perspective`, and `flat` inputs:

~~~sh
xcrun metal -std=metal3.2 -S -emit-llvm -c interpolation_oracle.metal \
  -o interpolation_oracle.ll
~~~

The emitted argument-metadata pairs are respectively
`air.center + air.perspective`, `air.center + air.no_perspective`,
`air.centroid + air.perspective`, `air.centroid + air.no_perspective`,
`air.sample + air.perspective`, `air.sample + air.no_perspective`, and the
single `air.flat` token. Metadata order is significant: the location token
precedes the correction token. These conventions were recovered with Apple
metalfe 32023.883 and are strict-runtime tested after Luisa's LLVM-21 emission
and in-tree LLVM-14 downgrade.

Fragment derivatives are ordinary AIR calls such as
`air.dfdx.f32`, `air.dfdy.f32`, `air.dfdx.v2f32`, and
`air.dfdy.v2f16`; overload suffixes follow scalar/vector type. Fragment
discard uses `air.discard_fragment`. Depth is not written by an intrinsic: it
is part of the returned value and return metadata.

The current raster path emits these conventions directly. The frontend marks
each `RasterStageKernel` with an explicit `xir::RasterStage` role and produces
one `RasterStageFunction` module per stage. Both modules run the ordinary XIR
optimization sequence. In the plain-CFG lowering phase, `destructure-cfg` is
followed immediately by `inline-all`, then `mem2reg`; this ordering is shared
with compute and is important for callables containing stage built-ins or
discard. The paired AIR handoff verifies both XIR modules and the equality of
the vertex return and fragment payload types before generating either module.

Shader-written depth follows the same AST-to-XIR route as other fragment-only
operations. `raster_set_z_depth`, `raster_set_z_depth_greater_equal`, and
`raster_set_z_depth_less_equal` become the corresponding `ThreadGroupOp`
values and must be inlined into the fragment entry before AIR preflight. Each
operation takes exactly one `f32`, returns void, and is rejected in compute,
vertex, or residual callable code. A fragment stage may use one qualifier kind
but may write it more than once; mixing `any`, `greater`, and `less` promises in
one stage fails closed.

With color outputs, the internal fragment implementation returns a packed pair
of its logical color value and `float` depth. The external AIR entry flattens
the logical color value into the normal render-target components and appends
depth as the last physical return component. A void fragment that writes depth
instead returns Apple's packed singleton `<{ float }>` ABI, and its output
metadata contains only depth. Depth does not increment the runtime color
attachment count. Its return metadata is exactly `air.depth`,
`air.depth_qualifier`, then `air.any`, `air.greater`, or `air.less`; there is no
depth-write intrinsic. Every non-discard return path must execute a matching
depth write when a shader-depth operation is present, just as an MSL fragment
output field must be initialized before return. A void fragment without a
shader-depth write remains invalid because it has no physical output.

The vertex wrapper is named `vertex_main`. Its physical parameters are:

1. vertex attributes, flattened in mesh-stream order and then attribute order;
2. `air.vertex_id`, `air.instance_id`, and `air.base_instance`;
3. the shared root argument structure as an `air.indirect_buffer` at constant
   buffer location 0;
4. a read-only, four-byte object-ID constant buffer at location 1.

Runtime vertex streams therefore begin at Metal buffer index 2. Attribute
locations in AIR metadata use the same flattened order. Signed/unsigned
8/16/32-bit vertex formats arrive as `intN`/`uintN`, half formats as `halfN`,
and normalized, packed R10G10B10A2 normalized, and float32 formats as
`floatN`. Integer and half lanes are converted to float while reconstructing
the fixed Luisa `AppData` structure. Position/normal are truncated or
zero-padded to three lanes, tangent/color to four, and UV0--UV3 to two. Missing
semantics remain zero. Mesh formats are limited to four non-empty vertex
streams. Duplicate semantics and block-compressed formats fail preflight.
`R10G10B10A2UInt` and `RGBA8SRGB` also fail because Metal has no
vertex-descriptor format that preserves their Luisa semantics.
`R11G11B10F` is deliberately rejected as well: its Metal vertex format starts
at macOS 14, while this backend keeps a macOS 13 deployment floor. These checks
run before JIT generation, archive serialization/loading, and every fixed-size
runtime stride table.

The vertex stage returns either `float4` position alone or a structure whose
first member is `float4` position. Remaining scalar/vector `half`, `float`,
`int32`, or `uint32` members (up to four lanes) become
`air.vertex_output user(locnN)` values. A reflected host structure can declare
one interpolation mode for each member after position with
`LUISA_RASTER_VARYING_INTERPOLATION(...)`. The available enumerators are
`CENTER_PERSPECTIVE`, `CENTER_NO_PERSPECTIVE`, `CENTROID_PERSPECTIVE`,
`CENTROID_NO_PERSPECTIVE`, `SAMPLE_PERSPECTIVE`,
`SAMPLE_NO_PERSPECTIVE`, and `FLAT`. The declaration is serialized into the
structure `Type` description as member attributes, so it survives AST-to-XIR,
optimization, JIT, and raster archive creation/loading without a parallel
side table. The fragment wrapper is named
`fragment_main`; it receives the flattened position/varyings followed by
`air.primitive_id`, a perspective-center `air.barycentric_coord float3`, an
`air.front_facing bool`, the same root structure, and the object-ID buffer.
Unannotated floating varyings retain center-perspective interpolation and
unannotated non-floating varyings retain flat interpolation. Perspective,
centroid, and sample modes on a non-floating varying fail preflight; unknown
attribute values, a missing first-position declaration, or a second position
member also fail closed. A
scalar/vector fragment return is render target 0; a structure is flattened to
consecutive `air.render_target` indices. The fragment return is capped at eight
targets. Its exact target count is carried from LLVM emission through the
paired AIR result, raster archive, loaded `MetalRasterShader`, pipeline cache,
and draw encoder; a draw must bind exactly that many color attachments. The
count may be zero for a depth-only fragment, in which case the draw binds a DSV
and no RTVs.

Root fields from the two AST functions occupy one 16-byte-aligned structure in
vertex-argument order followed by fragment-argument order. Each stage wrapper
loads only its own range, while both AIR modules reflect the identical full
layout. This reuses the compute root ABI, including byte-sized bool/byte memory
layout, typed buffers, textures, samplers, bindless arrays, acceleration
structures, and uniform aggregates. `raster_object_id()` is loaded from the
location-1 constant buffer in both stages. In the fragment stage,
`raster_barycentrics()` is the native float3 builtin and the ordinary XIR
kernel-ID special register maps to primitive ID. `raster_is_front_face()` is
an AST bool builtin translated to
`DerivedSpecialRegisterTag::RASTER_FRONT_FACING`/`SPR_FrontFacing`; the
fragment entry passes its physical `i1` through the hidden raster-state ABI
used by the stage implementation and generated callables. The vertex wrapper
supplies `false` only to keep that internal signature uniform, while AIR
preflight rejects actual front-facing use in vertex, compute, or otherwise
invalid stage code.

In the vertex stage, `raster_base_instance()` maps to
`DerivedSpecialRegisterTag::RASTER_BASE_INSTANCE`/`SPR_BaseInstance` and the
physical `i32 noundef` parameter reflected by `air.base_instance`. The value is
passed through the same hidden callable state; fragment and compute use are
rejected. `RasterMesh::base_instance()` carries the draw-time value, defaulting
to zero for source compatibility. Metal4 forwards it to both indexed and
non-indexed MTL4 draw calls; the DX12 and Vulkan raster encoders likewise use
it as `StartInstanceLocation`/`firstInstance` so ordinary instance-ID semantics
remain consistent across the runtime.

Fragment `ddx`/`ddy` accept scalar or vector f16/f32 values and lower to the
suffix-matched convergent `air.dfdx.*`/`air.dfdy.*` calls. `raster_discard()`
emits `air.discard_fragment` and terminates the translated stage path. These
operations, fragment-only barycentrics, and other raster special registers are
stage-checked by preflight. Raster printing uses the same `air.os_log` ABI and
command-buffer `MTL::LogState` as compute, so it adds no raster root binding;
raster debug-break state remains rejected with an explicit diagnostic.

After LLVM O2 and the in-tree LLVM-14 downgrade, the paired packager writes a
single metallib containing `vertex_main` as program type 0 and `fragment_main`
as program type 1. The Metal raster runtime loads both binary functions,
creates render/depth pipeline states, binds the root block and object ID at
indices 0 and 1, binds vertex streams starting at 2, and encodes indexed or
non-indexed instanced draws. Actual vertex-buffer strides are part of the
pipeline-state cache key and are installed as static vertex-layout strides;
buffers use the legacy `setVertexBuffer:offset:atIndex:` binding. This preserves
the project's macOS 13 deployment floor instead of depending on dynamic vertex
strides and `setVertexBuffer:offset:attributeStride:atIndex:`, which start at
macOS 14. The strict raster integration test checks the
post-O2 metadata and intrinsic spellings, then renders and reads back a
triangle whose visibility/color depend on vertex and fragment object ID,
barycentrics, base instance, front-facing state, root constants, derivatives,
and discard. A nonzero base instance is passed through an integer flat varying;
changing only the draw-time value makes every fragment discard, across both
indexed and non-indexed draws and both JIT and archive-loaded AOT shaders.
With culling disabled, the test inverts `RasterState::front_counter_clockwise`
and requires the covered pixel to switch between two alpha values; it repeats
both winding cases through the archive-loaded AOT shader. The same draw binds
a D32 depth buffer through its read-only `to_img()` alias and uses a uniform
array that pushes the shared root block above 4 KiB, covering both the common
depth/texture binding interface and the staged vertex-plus-fragment root path.
The fragment also writes depth 0.375 with the greater-equal qualifier. A
follow-up compute dispatch reads the D32 attachment and requires that exact
value for both the JIT draw and the archive-loaded AOT draw; the IR checks also
require `air.depth`, `air.depth_qualifier`, and `air.greater`. A separate void
fragment returns only depth 0.4375 through the packed singleton ABI, binds no
color attachment, and must reproduce that value through both JIT and
archive-loaded AOT execution.

The same strict test compiles a second varying payload containing every
selectable interpolation mode. It requires the exact AIR metadata token pairs,
then renders a triangle whose top vertex has clip-space W=4 while preserving
its screen-space position. At the center pixel, the perspective-correct value
must be near 0.2 while the no-perspective value must be near 0.5; this proves
GPU interpolation semantics rather than metadata presence alone. Centroid and
sample values are consumed by the fragment stage, and an archive-loaded AOT
draw must exactly match the JIT image.

The implemented slice does not yet expose stencil, conservative
rasterization, tessellation, or mesh shaders. These are separate extensions to
the stage ABI rather than aliases for the currently emitted metadata.

## 13. Fast-math policy

`ShaderOption::enable_fast_math` affects three layers:

1. XIR optimization options.
2. AIR math symbol selection, ordinary-instruction fast-math flags, and
   entry/implementation function attributes.
3. `air.compile.fast_math_enable` versus
   `air.compile.fast_math_disable`.

The module always emits `air.compile.denorms_disable`. Therefore "accurate"
mode is not a promise to preserve denormals; it means no fast AIR symbol prefix
and no unsafe entry-level fast-math attributes.

The LLVM O2 pipeline deliberately disables full loop unrolling. Apple
`metalfe` preserves ordinary constant-trip loops in AIR, and handing a compact
loop to the final GPU compiler avoids inflated AIR and register pressure while
retaining loop/interleave/vector optimizations other than unrolling.

## 14. Reconstructed MTLB container

The library writer is adapted from Metal.jl's open-source implementation. It
does not call Apple's `metallib` linker. Integers are little-endian.

### 14.1 Target version mapping

macOS major versions 16 through 25 are normalized by adding 10, accommodating
the tested compatibility/version-number transition.

| Normalized macOS | MTLB file format | AIR | Metal language |
|---:|---|---|---|
| 13 | 1.2.7 | 2.5 | 3.0 |
| 14 | 1.2.7 | 2.6 | 3.1 |
| 15 | 1.2.8 | 2.7 | 3.2 |
| 26 | 1.2.9 | 2.8 | 4.0 |
| 27+ | 1.2.9 | 2.9 | 4.1 |

This mapping is empirical and must be extended deliberately for future OS
versions. It describes the versions the writer can encode, not a cross-target
portability guarantee: the current AIR pipeline selects the row for the
runtime host, as described in Section 3.1.

### 14.2 Fixed header

The header is 88 bytes:

| Offset | Size | Meaning |
|---:|---:|---|
| 0 | 4 | ASCII `MTLB` |
| 4 | 2 | file major with bit 15 set |
| 6 | 2 | file minor |
| 8 | 2 | file patch |
| 10 | 1 | file type, currently 0 |
| 11 | 1 | platform type, currently `0x81` |
| 12 | 2 | platform major |
| 14 | 1 | platform minor |
| 15 | 1 | platform patch |
| 16 | 8 | total file size |
| 24 | 8 | function-list offset |
| 32 | 8 | function-list size, excluding count |
| 40 | 8 | public-metadata offset |
| 48 | 8 | public-metadata size |
| 56 | 8 | private-metadata offset |
| 64 | 8 | private-metadata size |
| 72 | 8 | module-payload offset |
| 80 | 8 | module-payload size |

### 14.3 Function records

The function list begins with a 32-bit count. Each function is a length-
prefixed tagged group:

| Tag | Payload |
|---|---|
| `NAME` | NUL-terminated function name |
| `TYPE` | one-byte program type; compute kernel is 2 |
| `HASH` | SHA-256 of this AIR module |
| `OFFT` | three 64-bit offsets: public metadata, private metadata, module |
| `VERS` | AIR major/minor and Metal major/minor as four 16-bit values |
| `MDSZ` | 64-bit AIR module byte size |
| `ENDT` | group terminator |

Program-type values understood by the writer are:

| Value | Program type |
|---:|---|
| 0 | vertex |
| 1 | fragment |
| 2 | kernel |
| 3 | unqualified |
| 4 | visible |
| 5 | extern |
| 6 | intersection |

Compute libraries use type 2. Paired raster libraries use type 0 for
`vertex_main` and type 1 for `fragment_main`.

After all function records, the function-list extension contains a 16-byte
`UUID` tag and `ENDT`.

### 14.4 UUID, metadata sections, and payloads

The writer concatenates all AIR modules, hashes that byte string with SHA-256,
uses the first 16 bytes as deterministic UUID material, sets RFC 4122
version-4/variant bits, and serializes each eight-byte half in reverse order.

Public and private metadata currently contain one empty eight-byte
`ENDT` group per function. AIR modules are concatenated without padding in
function order. A compute library contains:

1. `kernel_main` AIR;
2. `kernel_main_indirect` AIR.

A raster library contains:

1. `vertex_main` AIR;
2. `fragment_main` AIR.

The structural validator checks:

- header magic, supported versions, platform/type fields, and exact file size;
- ordered, non-overlapping section ranges;
- required function tags, expected entry-point order, and optional expected
  program-type order (vertex/fragment for raster AOT loading);
- per-function metadata and payload offsets;
- complete consumption with no trailing bytes;
- SHA-256 for each AIR payload.

It does **not** parse LLVM bitcode, validate AIR metadata semantics, or prove
that Apple accepts the library. It also skips rather than authenticates the
UUID header-extension value. Apple-tool and runtime validation remain separate
gates.

## 15. Runtime loading and fail-closed behavior

### 15.1 Build-time switch

In-process **generation** of XIR, LLVM IR, downgraded AIR, and a new MTLB is
compiled only when:

~~~text
APPLE
CMAKE_CXX_COMPILER_ID matches Clang
LUISA_COMPUTE_ENABLE_METAL4=ON
LLVM_VERSION_MAJOR=21
~~~

The CMake option is experimental and defaults to OFF. Enabling the `metal4`
module requires its complete XIR/LLVM/AIR generator, the pinned in-tree
`llvm-downgrade` source, and LLVM 21; there is no partially built Metal4 module
that contains only the archive loader. The original `metal` backend is selected
independently with `LUISA_COMPUTE_ENABLE_METAL`.

The Metal4 module is currently integrated through CMake/Ninja only. Its
LLVM-21 discovery, pinned llvm-downgrade validation, disposable patched-source
mirror, and backend link graph have no xmake equivalent yet; no placeholder
xmake target is provided under `backends/metal4`.

At runtime the module requires macOS 26 or newer, a device reporting
`MTL::GPUFamilyMetal4`, and successful creation of an `MTL4::Compiler`.
Unsupported hosts expose no `metal4` device rather than silently behaving like
the original Metal backend.

### 15.2 Unconditional AIR path

Selecting the `metal4` device unconditionally selects XIR/LLVM/AIR generation.
There is no `LUISA_METAL4_AIR` or compatibility mode variable, and the backend
does not compile or link `MetalCodegenAST`. Unsupported preflight, LLVM
verification, downgrade, metallib construction/loading, or PSO creation fails
the requested shader operation. Compute and raster AOT loaders consume their
validated archives directly and likewise have no source-code fallback.

### 15.3 Device-gated acceleration-structure build path

Metal 4 itself starts at Apple7, but Apple's feature table lists
**address-driven acceleration-structure builds** as Apple9. The runtime checks
`MTLDevice::supportsFamily(MTL::GPUFamilyApple9)` rather than assuming that
successful creation of an `MTL4::CommandQueue` implies support for every MTL4
encoder operation.

- Apple9 and newer use `MTL4::PrimitiveAccelerationStructureDescriptor` or
  `MTL4::InstanceAccelerationStructureDescriptor` and encode build, refit,
  compact-size query, and compaction on an `MTL4::ComputeCommandEncoder`.
- Apple7/Apple8 create one isolated `MTL::CommandQueue` per Luisa stream for
  AS build/refit/compact only. The runtime flushes and waits for pending MTL4
  work, performs the legacy descriptor/encoder operation synchronously,
  validates `MTLCommandBuffer.status` and `error`, then resumes ordinary MTL4
  command encoding.

This is a device-feature bridge, not shader fallback. On both branches user
shaders remain XIR/LLVM/AIR, pipeline creation remains `MTL4::Compiler`, and
compute/render dispatch remains `MTL4::CommandBuffer` with MTL4 compute/render
encoders and argument tables. The pre-Apple9 branch exists because an M1 Max
can create the Metal4 compiler and queue and execute AIR ray tracing, while
Metal Validation reports that the device cannot execute the MTL4 AS build; an
unchecked build otherwise completes with an empty acceleration structure.
Validation now fails the command explicitly instead of accepting such output.

The implementation is split between
[metal_stream.cpp](../metal_stream.cpp), which owns the capability decision
and compatibility queue, [metal_acceleration_structure_build.cpp](../metal_acceleration_structure_build.cpp),
which enforces the synchronization/error boundary, and the mesh, curve,
procedural-primitive, and instance-AS classes, which maintain matching MTL4
and compatibility descriptors. See Apple's
[Metal feature tables](https://developer.apple.com/metal/capabilities/) for
the Apple9 boundary.

### 15.4 `MotionInstance` descriptor translation

The public `MotionInstance` is a host-side transform resource rather than a
second Metal acceleration-structure allocation. Its build command captures a
built mesh, curve, or procedural child plus exactly the configured number of
64-byte Luisa keyframes. Matrix keyframes must be finite affine matrices; SRT
keyframes must have finite components and a nonzero quaternion. A containing
`Accel::build()` snapshots those values and constructs the native motion TLAS
inputs.

Two layouts deliberately coexist:

- Shader-visible `LCAccel` remains a 16-byte pair of the native instanced-AS
  handle and a device pointer to 72-byte `LCInstance` records. AIR instance
  transform, mask, opacity, user-ID reads/writes, and the MTLResourceID tail
  therefore keep the same ABI for static and motion TLASes.
- Native TLAS construction uses one 48-byte
  `MTL::IndirectAccelerationStructureMotionInstanceDescriptor` per instance
  plus a separate motion-transform buffer. Its fields are packed as options,
  mask, Luisa user ID in `intersectionFunctionTableOffset`, Luisa instance
  index in `userID`, child `MTLResourceID`, transform start/count, border
  modes, and time range. Static children placed in a motion TLAS receive one
  clamped keyframe, so static and motion resources can coexist.

For `AccelMotionMode::MATRIX`, keyframes become 48-byte
`MTL::PackedFloat4x3` values in column-major layout. The containing Accel
transform is composed as `outer * keyframe`, matching the fallback and SIMD
runtime semantics; it is not discarded or applied in the opposite order.
Matrix MotionInstance resources, ordinary static instances, motion BLASes,
and matrix MotionInstance refits execute on the Apple7 M1 Max through the
compatibility build bridge in Section 15.3.

For `AccelMotionMode::SRT`, the 64-byte Luisa value maps field-for-field to
`MTL::ComponentTransform` (`scale`, `shear`, `pivot`, quaternion rotation, and
translation); nonzero input quaternions are normalized before upload, as the
Metal contract requires. The SDK defines scale, shear, and pivot as the
pre-rotation components, quaternion rotation next, and translation last; with
a zero pivot this is exactly the upper-triangular scale/shear plus rotation
decomposition used for static transforms. Apple's feature table places
per-component motion interpolation at Apple9, so creation fails explicitly on
Apple7/Apple8. Metal
selects one transform representation for the entire TLAS: mixing MATRIX and SRT
MotionInstance resources is rejected. Static transforms in an SRT TLAS are
decomposed into component form. An outer translation can be added without
changing component interpolation, but an outer rotation, scale, or shear is
rejected; callers needing that composition must use MATRIX motion.

Changing keyframes does not mutate a native object in place. Submit
`motion_instance.build()` and then `accel.build()`: the latter uploads fresh
motion descriptors/transforms and refits when `AccelOption::allow_update`
permits it. A descriptor-mode change, instance-count change, buffer growth, or
forced build recreates the TLAS. Device queries expose the chosen boundaries:

| Query | Meaning |
|---|---|
| `metal_motion_blur` | matrix motion is accepted by the device |
| `metal4_component_motion` | Apple9 component/SRT interpolation is available |
| `metal4_address_driven_acceleration_structures` | the native MTL4 AS-build path is available |

The focused acceleration test executes non-commuting outer/keyframe matrix
composition, mixed static/motion instances, metadata, any/closest motion
traces, and rebuild/refit. `test_metal4_motion_instance_render` additionally
renders a scanline-time image and checks hit coverage, motion extent, and
upper/lower hit centroids; on the M1 Max both tests pass with Metal API and GPU
Validation enabled.

### 15.5 Loading and PSO creation

The generated bytes are wrapped in `dispatch_data_t` and passed directly to
`MTLDevice::newLibrary`. No temporary file is required. Function descriptors
are then handed to the device's `MTL4::Compiler`; compute and raster pipeline
creation both use Metal 4 descriptors. Descriptor properties with copy
semantics must be populated before assignment: setting a function name after
assigning its descriptor leaves the copied descriptor unnamed.

The compute compiler loads both names:

- `kernel_main`, with indirect-command-buffer support disabled;
- `kernel_main_indirect`, with indirect-command-buffer support enabled.

Each function is requested with `MTLFunctionOptionCompileToBinary` and used
to create an MTL4 compute pipeline. The pipeline descriptor records the Luisa
block size and marks it as a multiple of thread execution width.

The raster extension instead loads `vertex_main` and `fragment_main` from the
same paired library with `MTLFunctionOptionCompileToBinary`. It creates render
pipeline states lazily for the color/depth format and `RasterState` key, plus a
matching depth/stencil state. Its shader archive stores the paired metallib,
mesh format, reflected stage arguments, root-block size, and exact fragment
output count.

Raster archive version 3 appends a seeded 64-bit `luisa::hash64` checksum over
every preceding serialized byte, including the header, mesh/argument metadata,
and complete MTLB payload. Deserialization verifies this checksum before
parsing, then rejects unknown versions, malformed stream/argument/output
counts, unsupported macOS-13 vertex formats, invalid usage/stage metadata,
invalid or oversized root layouts, truncation, empty libraries, and trailing
bytes. This checksum detects accidental or stale metadata changes but is not
an authentication mechanism; the inner MTLB validator independently verifies
each AIR payload's recorded SHA-256 as described in Section 14.4.

The raster AOT loader then applies ABI checks that do not trust the archived
size alone. It requires exact argument count and type-description order and
recomputes the 16-byte-slotted root size from the requested `Type` list:
ordinary values use `Type::size()`, buffers/acceleration structures and
`LC_IndirectDispatchBuffer` use 16 bytes, and textures/bindless arrays use
eight bytes before the next 16-byte slot. Unsupported/null types, overflow
beyond 65536 bytes, or a size different from the archive cause an invalid
load. Structural MTLB validation requires both names **and** program types in
order: `vertex_main`/vertex followed by `fragment_main`/fragment.

For AOT resource residency, non-texture resources are conservatively restored
as read-write and visible to both raster stages. Texture usage preserves the
archived access mask so a read-only `DepthBuffer::to_img()` binding is not
incorrectly upgraded to shader-write access; it is still made visible to both
stages. Finally, failure of `newLibrary` or either binary function lookup marks
the `MetalRasterShader` invalid. `load_raster_shader` warns, destroys the
partial object, and returns an invalid resource rather than aborting on an
archive produced for an incompatible host/device.

Memory and disk cache hashes include the metallib bytes, shader options, and a
backend namespace discriminator.

The in-memory PSO cache is a 64-entry LRU. Disk caching is enabled by an
explicit shader name or `ShaderOption::enable_cache`. The disk artifact is a
Metal binary archive plus serialized Luisa metadata, not merely the raw MTLB
blob. Debug information suppresses loading an existing disk archive. Raw MTLB
dumping is a separate diagnostic path and only occurs when a configured shader
I/O destination is available.

## 16. Validation and reverse-engineering workflow

The patched JuliaLLVM tree retains its standalone FileCheck suite. Its
`ret_eltype` and `struct_eltypes` regressions generate LLVM 14 bitcode with the
LLVM 21-built downgrader, disassemble it with an actual LLVM 14 `llvm-dis`, and
verify the reconstructed return, argument, nested-structure pointer types, and
removal of the private `llvm.struct_eltypes` directive. Both tests pass on the
reference setup; tests for LLVM 5 or 7 remain independently skippable when
those legacy disassemblers are unavailable.

### 16.1 Runtime tests

The Metal4 integration test runs the unconditional AIR backend and exercises
typed buffers, byte buffers, uniforms, dispatch state, shared
memory, the workgroup barrier, integer reductions, non-fast NaN behavior,
device/threadgroup atomics, integer-bit intrinsics, SIMD reductions/prefixes/
ballot/shuffle, direct 2D/3D float/int/uint texture size/read/write, recursive
aggregate uniforms, volatile typed/byte buffers with AIR fences, and direct
2D/3D float sampling. The sample cases cover implicit LOD, explicit LOD,
gradients, gradients plus minimum LOD, all four filter enums, and all four
address enums while writing results to a separate buffer so the sampled
resource remains sample-only. Its ABI cases include both a four-field scalar-
bool structure and Luisa `bool4`: each has a four-byte host/device stride, and
the test checks every field or lane across multiple elements after a GPU round
trip. A typed-buffer `byte4` case similarly reverses all four byte lanes across
multiple elements, proving that its LLVM memory stride is four bytes. A raw
byte-buffer case reads false and true bool values from adjacent byte offsets,
so an erroneous packed-i1 load cannot pass. The aggregate case interleaves a
texture, nested structure, top-level array, buffer, and scalar so logical-
versus-physical reflection indices are tested. The same strict executable
checks exact `PACK` words and `UNPACK`
round trips for a four-bool structure, `float3`, `byte4`, scalar bool, byte,
and short.

The same kernel places a debug break with watch values behind a dynamically
loaded, zero-valued buffer condition. This keeps `llvm.debugtrap` present
through AIR compilation and pipeline-state creation, then proves the untaken
path can dispatch without firing the trap.

The public C++ DSL currently exposes sampling through bindless texture
wrappers, not direct `Image`/`Volume` variables. The strict direct-sampling
case therefore constructs the corresponding direct texture sample calls with
`FunctionBuilder`; this tests the XIR/AIR contract even before a convenience
DSL method is added.

Sampler descriptors, direct sample signatures, the old/new AIR state storage,
bindless slot reflection, and raster stage metadata were derived with small
Apple MSL differential oracles. Direct sampling is additionally strict-runtime
tested on the active AIR 2.8 toolchain; the AIR 2.5/2.6 sampler-state form is
still an oracle-derived compatibility path. A convention observed only in an
oracle is not treated as runtime-complete until a strict Luisa test executes
that path; keep that distinction when updating this snapshot.

Dedicated bindless validation now passes on Apple M1 Max with the active AIR
2.8 toolchain:

- `test_bindless_mip metal4` exits successfully through AIR. It checks four
  mip levels, dynamic mip-size queries,
  integer-coordinate reads, explicit-LOD sampling, and the sampler code stored
  in the bindless slot.
- `test_bindless_buffer metal4 --offline` exits
  successfully after compiling and dispatching a bindless typed-buffer read
  into an output image.
- The strict `test_metal_xir_air metal4` regression also passes after the typed-
  resource and packing changes. That run may reuse Metal cache artifacts, so
  the dedicated strict bindless executables are the primary validation of the
  reconstructed pointer and wrapper ABI. It additionally sends a 1024-element
  `uint` array plus a destination buffer through both direct dispatch and a
  GPU-prepared indirect-command buffer, covering staged roots above 4 KiB and
  the indirect execution encoder's explicit residency declaration.

Typed-bindless frontend coverage is split deliberately. The
`test_ast_typed_bindless_lowering` unit test exhaustively checks the ordinary
typed and typed-uniform aliases: 44 query aliases, 12 read aliases, and two
write aliases. The strict `test_metal_xir_air_typed_bindless` runtime test then
executes representative typed and typed-uniform buffer reads/writes, byte-
buffer reads, and 2D texture reads/size queries through AIR. Its buffer cases
include four-byte-stride scalar-bool structures and `byte4` values, with GPU
writes copied back to the host to verify every field or lane.

The failure mode before typed-pointer recovery was Metal pipeline-state
creation ending with `XPC_ERROR_CONNECTION_INTERRUPTED`. Restoring the sampler
return pointee, nested texture-wrapper pointees, bindless-item pointer, and
dynamic sampler wrapper table eliminates that failure and permits dispatch.

GPU-written indirect dispatch is also runtime validated on Apple M1 Max.
`test_indirect metal4` passes through the unconditional AIR path. The test
sets the count, fills records, launches individual offset views and the full
indirect range, and deliberately dispatches one excess producer lane to check
the capacity guard.

Static acceleration lowering is strict-runtime validated by
`test_metal_xir_air_accel`. On Apple M1 Max it covers triangle closest hit and
miss, any-hit visibility-mask acceptance and
rejection, instance and primitive IDs, barycentrics, distance, user ID,
visibility mask, and reconstruction of the identity instance transform. The
Metal4 target has no MSL generator, so another code path cannot hide an AIR
rejection or pipeline-state failure.

Primitive and instance motion are covered by the same strict executable. It
executes triangle and curve motion BLASes plus a standalone matrix
`MotionInstance`, mixes that resource with a static instance, preserves
instance/user metadata, composes a 90-degree outer rotation with a translating
keyframe, and refreshes changed keyframes through a TLAS refit. The companion
`test_metal4_motion_instance_render` produces a 160 by 120 PNG using scanline
ray time and checks 1,431 hit pixels, more than one-third-frame horizontal
motion extent, and separated upper/lower hit centroids. Both pass Metal API
and GPU Validation on the Apple M1 Max compatibility-build path; the shader
and traversal code remain Metal4 AIR.

Stateful triangle and procedural-bounding-box traversal is strict-runtime
validated by the focused `test_metal_xir_air_ray_query` CTest. It covers
triangle `QueryAll` rejection and selective commit,
`QueryAny` commit plus termination, visibility-filtered miss behavior,
candidate and committed-hit payloads, procedural bounding-box commit, and the
world-ray getters. The strict `test_metal4_air_world_and_object_ray` regression
also exercises candidate object-space origin/direction under a non-identity
instance transform, before and after commit, and compares it with the immutable
world-space ray. It verifies that an out-of-range procedural distance is
ignored instead of reaching the native bounding-box commit intrinsic. Because
the backend is AIR-only, none of those assertions can pass through an MSL
fallback.

True vertex/fragment execution is strict-runtime validated by
`test_metal_xir_air_raster`. Before drawing, the test enables the post-O2 LLVM
dump and checks `!air.vertex`, `!air.fragment`, vertex inputs and location
indices, vertex/instance/base-instance/primitive/barycentric/front-facing built-ins, root indirect-buffer
metadata, the object-ID buffer in both stages, render-target and shader-depth
metadata, every center/centroid/sample perspective/no-perspective varying
combination, `air.dfdx.f32`, `air.dfdy.f32`, and `air.discard_fragment`. It then
creates the paired metallib and render PSO, draws an instanced triangle, and
reads back the target. Vertex visibility depends on the object ID, a vertex
root constant, and a nonzero draw-time base instance carried through a flat
integer varying; an alternate base value must discard every fragment.
Fragment coverage/color also depends on object ID, a fragment root constant,
barycentrics, front-facing state, derivatives, and discard. The test
flips the front-winding state with culling disabled and requires both JIT and
AOT pixels to invert their encoded facing value. A D32 depth texture is read
through `DepthBuffer::to_img()`, and a 1024-float uniform pushes the shared
raster root above 4 KiB. The fragment writes depth 0.375 with
`raster_set_z_depth_greater_equal`; a compute readback requires 0.375 after both
the JIT draw and archive-loaded AOT draw. A second void fragment binds only the
D32 attachment, emits no `air.render_target`, and requires depth 0.4375 through
both JIT and AOT. The test then writes raster archives, reloads them through the
AOT boundary, redraws, and requires exact agreement with the JIT outputs.
The interpolation draw independently requires perspective-correct and
screen-linear center values to diverge under nonuniform clip W and requires
its archive-loaded image to exactly equal the JIT image.
Separate fragment modules require Apple to accept the `air.any` and `air.less`
metadata variants as well. Both the ordinary strict registration and the
validation-layer registration pass on the Apple M1 Max reference machine.

Reverse autodiff is registered as the strict
`test_metal_xir_air_autodiff` CTest. It exercises basic products,
trigonometric derivatives, custom gradients, chain-rule composition,
addition, a callable that is inlined after CFG destructuring, and branch-
sensitive piecewise control flow. Finite-result checks prevent an AIR compile
or dispatch success from masking invalid gradient values.

A compute-rendering smoke sweep is registered as fifteen CTests. Each is
labelled `integration`, `integration_metal4`, and `rendering`, runs offline,
and has a finite timeout. These are finite
compute-only renderers: they create no window or swapchain and do not exercise
the vertex/fragment raster architecture described in Section 12.2. The set
passes all fifteen mirrored executables with no timeouts: `test_sdf_renderer`,
`test_voxel_raytracer`, `test_path_tracing`,
`test_path_tracing_nested_callable`,
`test_path_tracing_ray_masks`, `test_photon_mapping`, `test_path_tracing_hdr`,
`test_path_tracing_spectrum`, `test_path_tracing_camera`,
`test_path_tracing_cutout`, `test_blackhole`, `test_procedural`,
`test_shader_toy`, `test_shader_toy_spacex`, and
`test_shader_visuals_present`. In particular,
`test_path_tracing_cutout metal4 --offline --spp 1`
exits successfully through AIR.

The cutout renderer's offline animation seed is fixed so same-build backend
comparisons use the same per-frame instance transforms. A previous
deterministic reference-image snapshot recorded **45.292483 dB PSNR** for
strict AIR against same-build MSL, above its 30 dB threshold. That snapshot
also recorded black hole at **37.487781 dB block PSNR** (**31.458710 dB raw**),
ShaderToy at **96.875522 dB**, ShaderToy SpaceX at **65.625341 dB**, and
shader-visuals at **51.499800 dB** against their respective committed or
same-build references. These numbers are retained as historical translation
evidence; the 2026-08-26 closure run below re-executes all fourteen renderers
but does not recompute every reference PSNR.

XIR unit regressions cover frontend conventions independently of Metal.
`test_ast_pack_usage` checks `PACK`/`UNPACK` usage and instruction shapes plus
opaque-custom usage propagation. `test_ast_external_lowering` checks external
declaration and call argument forms, `void`, TypeID zero, and StringID hash
lowering. The ray-query-loop regression also checks the direct proceed write,
same-lvalue terminated read, inversion, instruction order, and verifier
acceptance; optimizer regressions ensure query constructors are neither
commoned nor loop-hoisted.

Native shader logging is strict-runtime validated by the CTest mirrors
`test_metal4_air_printer` and `test_metal4_air_printer_callback`. The callback
regression dispatches `128 x 128` threads and observes exactly 128 verbose,
128 info, and 128 warning messages through the stream callback. This covers
`air.os_log`, integer/float/bool and aggregate formatting, outlined callables,
`MTL::LogState` command-buffer attachment, message normalization, and callback
ownership; no legacy printer buffer or host record parser participates.

The MTLB unit test covers deterministic two-function generation, version
mapping, expected entry order, truncation, corruption, trailing data, bad
hashes, duplicate names, invalid program types, and invalid target versions.
`test_metal_raster_archive` separately covers the version-3 wrapper round
trip, every truncated prefix, trailing data, malformed mesh/argument/output
metadata, and checksum-protected mutations that would otherwise remain
structurally valid. The strict raster integration test exercises the complete
compile-only/write/load/draw AOT boundary rather than stopping at archive
deserialization.

### 16.2 Validation and benchmark snapshot (2026-08-26)

The closure configuration uses CMake/Ninja Release builds with legacy Metal,
Metal4, and fallback enabled, Homebrew LLVM 21.1.8, Apple metalfe 32023.883,
and the macOS 26.4 SDK. After rebasing onto `origin/next` at `a5d69e492`, a
fresh full build and complete configured CTest run pass **155/155** in 111.91
seconds: **33/33** `integration_metal4` tests,
**15/15** offline graphics/rendering executables, and **11/11** tutorials are
included. The standalone matrix-motion acceleration and image regressions also
pass Metal API and GPU Validation with 43 and 8 assertions respectively; the
Metal4 timeline-event regression passes the same validation layers with 11
assertions, including host-callback ordering and `UINT64_MAX` fences. The same
fourteen renderers passed on the restored legacy `metal` backend in the
preceding 2026-08-25 comparison run. These counts test actual compilation and
GPU execution; the separate Metal4 backend prevents an unsupported or invalid
AIR module from silently succeeding through MSL.

MTL4 commit feedback publishes a submission's host-visible completion only
after callbacks, resource release, profiling, and the in-flight decrement have
finished. This prevents a synchronous waiter from destroying `MetalStream`
while its feedback handler still accesses the stream. The autodiff integration
test passes 32 consecutive fresh-process stress runs and a Metal API/GPU
Validation run after this lifetime fix.

`benchmark_metal4` compares the same arithmetic kernel and resource/dispatch
path on the two backends. Each process disables the Luisa shader cache and uses
a unique source variant, because otherwise the system `MTLCompilerService`
cache turns later fresh processes into MSL cache-hit measurements. Nine paired
fresh-process samples alternate backend order. Each sample operates on
1,048,576 `float4` elements, performs 64 arithmetic rounds, warms up with 256
dispatches, measures nine batches of 64 dispatches, and verifies the same
`14.498938` checksum.

| Metric | Legacy `metal` (MSL) | `metal4` (XIR/LLVM/AIR) | Paired result |
|---|---:|---:|---:|
| Cold end-to-end JIT median | 389.976 ms | 69.979 ms | **5.59x faster**, 82.1% less time |
| Backend codegen median | 0.199 ms | 9.730 ms | AIR spends more time in XIR/LLVM lowering |
| Apple compile/load median | 389.742 ms | 60.693 ms | AIR avoids the expensive source-MSL compile |
| Batched wall time per dispatch | 0.148398 ms | 0.196728 ms | Metal4 is **1.307x slower** by paired median |
| Median process p25/p75 | 0.146452/0.150892 ms | 0.184237/0.204446 ms | distributions do not overlap |

These numbers are the fresh 2026-08-26 rerun using variants 82701 through
82709 after the selectable-interpolation change. The measured gain remains a
substantial real cold-JIT improvement, but the current small steady-state
workload is not at parity: Metal4 takes about 30.7% more host wall time per
dispatch. The `runtime_ms` field includes command encoding, submission, queue
execution, and synchronization amortized over a 64-dispatch batch; it is not a
pure GPU-kernel timestamp.

A separately profiled pair (variant 82801, 833 labelled dispatches including
validation and warmup) reports average command-buffer GPU time of 0.128286 ms
for legacy Metal and 0.135542 ms for Metal4, a much smaller 5.7% difference,
while its batched wall medians are 0.147023 ms and 0.186022 ms. This indicates
that most of the current regression is outside shader arithmetic. The MTL4
path currently creates a command allocator, command buffer, log-state options,
all-stage barrier, residency set, and commit-feedback handler for each small
submission; pooling/batching that state is the next performance target. This
microbenchmark magnifies fixed submission cost, so longer rendering kernels
need a separate end-to-end comparison before generalizing the 30.7% number.

Reproduce one pair with distinct variants as follows:

~~~sh
LUISA_METAL_SHADER_INFO=1 ./bin/benchmark_metal4 metal 64 1101
LUISA_METAL_SHADER_INFO=1 ./bin/benchmark_metal4 metal4 64 1101
~~~

### 16.3 Useful commands

Dump XIR and optimized LLVM:

~~~sh
LUISA_DUMP_XIR=1 \
LUISA_DUMP_LLVM_IR=1 \
<test-or-example> metal4
~~~

Validate and inspect a generated library:

~~~sh
xcrun metallib --app-store-validate shader.metallib
xcrun metal-readobj --symbols shader.metallib
xcrun metal-objdump --metallib --air-version shader.metallib
~~~

The validated two-entry output should report:

~~~text
kernel_main
kernel_main_indirect
~~~

and an AIR version matching the module metadata/container target.

The final toolchain snapshot for this implementation used:

~~~text
macOS product version:  26.6.2
Apple metal:           32023.883
LLVM emitter/writer:   21.1.8
AIR reported by tools: 2.8
~~~

Apple's normal source flow remains useful as a differential oracle:

~~~sh
xcrun metal -std=metal3.2 -c oracle.metal -o oracle.air
xcrun metallib oracle.air -o oracle.metallib
~~~

Compare target triples, declarations, function attributes, `air.kernel`
metadata, module flags, record tags, version fields, and Apple-tool behavior.
Do not assume byte-for-byte identity: Apple compiler passes, debug metadata,
UUIDs, and container metadata can differ while preserving the same contract.

## 17. Extension checklist

When adding an AIR feature:

1. Add the XIR type/instruction to preflight only after emission exists.
2. Match MSL semantics, especially NaN, overflow, bounds, memory ordering, and
   convergence.
3. Obtain a minimal MSL oracle and inspect its AIR output.
4. Record exact intrinsic signature, address spaces, attributes, metadata, and
   numeric flags in this document.
5. Preserve register/memory conversion at every load/store boundary.
6. Mark barriers, atomics, and calls with correct convergence and memory
   effects; never add `nosync` transitively across a barrier.
7. Add an executing Metal4 runtime test for every new lowering.
8. Add a negative test for unsupported behavior and require a clear error.
9. Verify pre-O2 and post-O2 LLVM modules.
10. Validate every generated metallib with the structural validator, Apple
    tools, `MTLDevice::newLibrary`, and an executing GPU test.
11. Re-run the graphics/rendering suite to validate AIR behavior and caches.

For atomics or volatile operations, define the Metal memory scope and ordering
first. LLVM `volatile` is not a substitute for device coherence, and a
correct atomic lowering requires an explicit AIR intrinsic/ABI convention.

## 18. Known risks and deliberate limitations

- AIR intrinsic names and metadata are private Apple implementation details.
- The macOS/AIR/Metal/file-format table is hand-maintained and empirical.
- AIR triples, AIR/Metal language versions, and MTLB platform fields currently
  target the runtime host. A raster archive generated on a newer macOS release
  is not guaranteed to load at the project's older C++ deployment floor.
- The MTLB writer uses empty public/private metadata groups; more advanced
  function kinds may require real records.
- The internal structural validator is not a semantic AIR validator.
- Raster archive v3's outer 64-bit checksum and the MTLB payload hashes detect
  corruption; they do not authenticate an archive or prove that archived AIR
  is semantically compatible with the current Metal runtime.
- All buffer arguments are reflected read-write.
- Resource indexing has no generated bounds checks.
- Byte-buffer/raw-address operations assume correct alignment.
- The support preflight is an allowlist, but a few operand invariants remain
  emitter assertions rather than explicit preflight diagnostics.
- Explicit XIR `unreachable` becomes a conservative return rather than LLVM
  `unreachable`.
- Matrix determinant/inverse lowering favors simplicity over numerical
  specialization.
- Sampler descriptor bit patterns, texture/atomic/SIMD intrinsic signatures,
  and numeric control operands are private Apple conventions and need
  differential revalidation on each toolchain/AIR-version update.
- Direct compute textures, 32-bit atomics, volatile device-buffer access, the
  documented subgroup subset, the documented bindless buffer/texture subset,
  GPU-written indirect dispatch records, and the static instanced-triangle
  trace/query/write subset are AIR-native and strict-runtime-tested. Static
  curve and direct primitive/instance-motion triangle/curve tracing are also
  oracle-matched and strict-runtime-tested. The local stateful static
  triangle/curve/procedural-bounding-box query subset in Section 8.9 is
  likewise covered. The fixed-AppData raster subset in Section 12.2 is also
  AIR-native and strict-runtime-tested. Direct procedural traces and stateful
  motion queries remain fail-closed.
- Luisa's standalone runtime `MotionInstance` is implemented with separate
  shader-visible 72-byte instance records and native 48-byte indirect-motion
  descriptors. Matrix motion and refit are strict-runtime-tested on Apple7;
  component/SRT interpolation remains Apple9-only and must not be approximated
  on Apple7/Apple8. A TLAS has one native transform representation, so mixed
  MATRIX/SRT resources fail closed, as does a non-translation outer transform
  around SRT motion. The Apple9 SRT path is layout/oracle matched but cannot be
  executed on the local M1 Max and still needs Apple9 hardware validation.
- Typed resource identities are carried by emitter-only metadata until the
  LLVM 14 writer serializes them. Dropping `ret_eltype`, flattening a nested
  texture or sampler wrapper to a bare pointer, or failing to consume
  `llvm.struct_eltypes` can yield structurally valid LLVM that Metal rejects at
  pipeline-state creation.
- Raster stage metadata and render encoding are implemented, but every new
  stage semantic still requires an Apple oracle, frontend/XIR modeling,
  preflight rules, emitter metadata, runtime binding, and a strict draw test.
  Shader-written `any`/`greater`/`less` depth and selectable varying
  interpolation now have that full chain and strict JIT/AOT readback coverage.
  Do not infer stencil or conservative-raster support from them.

## 19. Local source index

| Subject | Source |
|---|---|
| Core AIR emitter orchestration | [metal_codegen_llvm.cpp](metal_codegen_llvm.cpp), [metal_codegen_llvm_impl.cpp](metal_codegen_llvm_impl.cpp) |
| Type/ABI and native shader logging | [metal_codegen_llvm_type.cpp](metal_codegen_llvm_type.cpp) |
| Raster varying interpolation validation | [metal_codegen_llvm_raster.cpp](metal_codegen_llvm_raster.cpp), [raster_interpolation.h](../../../../include/luisa/dsl/raster/raster_interpolation.h) |
| Resource, curve, motion, and query lowering | [metal_codegen_llvm_resource.cpp](metal_codegen_llvm_resource.cpp), [metal_codegen_llvm_access.cpp](metal_codegen_llvm_access.cpp) |
| Fail-closed support preflight | [metal_codegen_llvm_preflight.cpp](metal_codegen_llvm_preflight.cpp) |
| Public codegen configuration/result | [metal_codegen_llvm.h](metal_codegen_llvm.h) |
| AST-to-XIR orchestration | [metal_xir_pipeline.cpp](../metal_xir_pipeline.cpp) |
| Shared XIR pipeline factories | [pass_pipeline.cpp](../../../xir/passes/pass_pipeline.cpp) |
| LLVM O2, version selection, dual entries, packaging | [metal_air_pipeline.cpp](../metal_air_pipeline.cpp) |
| MTLB writer and validator | [metal_metallib.cpp](../metal_metallib.cpp) |
| Raster extension and paired AIR creation | [metal_raster_ext.cpp](../metal_raster_ext.cpp) |
| Raster PSO, root/object binding, and draw encoding | [metal_raster_shader.cpp](../metal_raster_shader.cpp) |
| Raster archive format | [metal_raster_archive.cpp](../metal_raster_archive.cpp) |
| Shared color/depth texture binding and depth storage | [metal_texture.h](../metal_texture.h), [metal_depth_buffer.cpp](../metal_depth_buffer.cpp) |
| MotionInstance capture and native motion-TLAS packing | [metal_motion_instance.cpp](../metal_motion_instance.cpp), [metal_accel_motion.cpp](../metal_accel_motion.cpp) |
| Apple9 MTL4 versus Apple7/Apple8 compatibility AS build | [metal_acceleration_structure_build.cpp](../metal_acceleration_structure_build.cpp), [metal_stream.cpp](../metal_stream.cpp) |
| JuliaLLVM wrapper | [llvm_downgrade.cpp](../../../ext/llvm_downgrade.cpp) |
| Parent-owned LLVM 21/AIR downgrade overlay and regressions | [llvm-downgrade-llvm21-air.patch](../../../ext/llvm-downgrade-llvm21-air.patch) |
| Pinned upstream typed-pointer and LLVM 14 writer base | [PointerRewriter.cpp](../../../ext/llvm-downgrade/src/PointerRewriter.cpp), [ValueEnumerator140.cpp](../../../ext/llvm-downgrade/src/ValueEnumerator140.cpp), [BitcodeWriter140.cpp](../../../ext/llvm-downgrade/src/BitcodeWriter140.cpp) |
| AIR-only shader creation | [metal_device.cpp](../metal_device.cpp) |
| In-memory MTLB loading and PSO creation | [metal_compiler.cpp](../metal_compiler.cpp) |
| Host argument packing and dispatch | [metal_shader.cpp](../metal_shader.cpp) |
| 256-byte-aligned root/upload staging allocations | [metal_stage_buffer_pool.cpp](../metal_stage_buffer_pool.cpp) |
| Native `MTL::LogState` creation, normalization, and callback ownership | [metal_stream.cpp](../metal_stream.cpp), [metal_stream.h](../metal_stream.h) |
| Apple9 AS guard and synchronized pre-Apple9 bridge | [metal_acceleration_structure_build.cpp](../metal_acceleration_structure_build.cpp), [metal_stream.cpp](../metal_stream.cpp) |
| Indirect-command preparation ABI | [metal_builtin_kernels.metal](../metal_builtin/metal_builtin_kernels.metal) |
| GPU-written indirect-dispatch regression | [test_indirect.cpp](../../../tests/integration/runtime/test_indirect.cpp) |
| Strict AIR runtime semantics | [test_metal_xir_air.cpp](../../../tests/integration/runtime/test_metal_xir_air.cpp) |
| Strict AIR acceleration semantics | [test_metal_xir_air_accel.cpp](../../../tests/integration/runtime/test_metal_xir_air_accel.cpp) |
| Stateful AIR ray-query semantics | [test_metal_xir_air_ray_query.cpp](../../../tests/integration/runtime/test_metal_xir_air_ray_query.cpp) |
| Strict AIR typed-bindless semantics | [test_metal_xir_air_typed_bindless.cpp](../../../tests/integration/runtime/test_metal_xir_air_typed_bindless.cpp) |
| Strict AIR vertex/fragment metadata and draw semantics | [test_metal_xir_air_raster.cpp](../../../tests/integration/runtime/test_metal_xir_air_raster.cpp) |
| Raster archive round-trip and corruption handling | [test_metal_raster_archive.cpp](../../../tests/unit/ext/test_metal_raster_archive.cpp) |
| Typed-bindless AST alias normalization | [test_xir_translators.cpp](../../../tests/unit/xir/test_xir_translators.cpp) |
| XIR storage-cast validation | [test_xir_verifier.cpp](../../../tests/unit/xir/test_xir_verifier.cpp) |
| Direct ray-query proceed normalization | [test_xir_pass_reconstruct_ray_query_loop.cpp](../../../tests/unit/xir/test_xir_pass_reconstruct_ray_query_loop.cpp) |
| Query-constructor CSE/LICM constraints | [test_xir_pass_licm.cpp](../../../tests/unit/xir/test_xir_pass_licm.cpp) |
| Strict AIR autodiff semantics | [test_autodiff.cpp](../../../tests/unit/dsl/test_autodiff.cpp) |
| AST pack/unpack usage and XIR shape | [test_ast_pack_usage.cpp](../../../tests/unit/xir/test_ast_pack_usage.cpp) |
| External AST-to-XIR lowering | [test_ast_external_lowering.cpp](../../../tests/unit/xir/test_ast_external_lowering.cpp) |
| Native shader-log dispatch | [test_printer.cpp](../../../tests/unit/runtime/test_printer.cpp) |
| Native shader-log stream callback | [test_printer_custom_callback.cpp](../../../tests/unit/runtime/test_printer_custom_callback.cpp) |
| Bindless mip/read/sample semantics | [test_bindless_mip.cpp](../../../tests/unit/runtime/test_bindless_mip.cpp) |
| Bindless typed-buffer dispatch | [test_bindless_buffer.cpp](../../../tests/integration/runtime/test_bindless_buffer.cpp) |
| MTLB format/corruption tests | [test_metal_metallib.cpp](../../../tests/unit/ext/test_metal_metallib.cpp) |
| Legacy Metal versus Metal4 JIT/runtime benchmark | [benchmark_metal4.cpp](../../../tests/benchmark/benchmark_metal4.cpp) |

## 20. External references

- Apple, [Building a shader library by precompiling source files](https://developer.apple.com/documentation/metal/building-a-shader-library-by-precompiling-source-files)
- Apple, [Metal libraries](https://developer.apple.com/documentation/metal/metal-libraries)
- Apple, [Metal command-line tools](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Dev-Technique/Dev-Technique.html)
- Apple, [Metal feature tables](https://developer.apple.com/metal/capabilities/)
- Apple, [Logging shader debug messages](https://developer.apple.com/documentation/metal/logging-shader-debug-messages)
- Apple, [`MTL4CommandBufferOptions.logState`](https://developer.apple.com/documentation/metal/mtl4commandbufferoptions/logstate?language=objc)
- JuliaGPU, [Metal.jl library container implementation](https://github.com/JuliaGPU/Metal.jl/blob/main/src/compiler/library.jl)
- JuliaLLVM, [llvm-downgrade](https://github.com/JuliaLLVM/llvm-downgrade)

The Apple references define the supported public workflow. They do not make
the internal AIR LLVM or MTLB details in this document a public ABI.
