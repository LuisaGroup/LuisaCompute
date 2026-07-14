---
name: rust_workspace
description: Rust IR, compiler transforms, CPU backend, FFI, and crate structure.
---

# LuisaCompute Rust Workspace

`src/rust/` — legacy IR implementation, compiler passes, CPU/Remote backends. Integrates with C++ via FFI (`cbindgen` headers + static libs).

## Workspace Structure

```
src/rust/
├── Cargo.toml                         # workspace root
├── luisa_compute_api_types/           # shared C++/Rust FFI types (staticlib+rlib)
├── luisa_compute_ir/                  # core IR data structures & transforms (rlib)
├── luisa_compute_ir_staticlib/        # static lib wrapper for C++ linking (staticlib)
│                                     # produces library luisa_compute_ir_static
├── luisa_compute_ir_v2/               # IR v2 C API bindings (libloading)
├── luisa_compute_cpu_kernel_defs/     # CPU kernel runtime types
├── luisa_compute_backend/             # Backend trait definitions
└── luisa_compute_backend_impl/        # CPU + Remote backends (cdylib)
```

### Dependency Graph

```
luisa_compute_api_types
  ├── luisa_compute_ir
  │     ├── luisa_compute_ir_staticlib  (lib name: luisa_compute_ir_static)
  │     ├── luisa_compute_ir_v2
  │     └── luisa_compute_backend
  │           └── luisa_compute_backend_impl
  ├── luisa_compute_backend
  └── luisa_compute_cpu_kernel_defs
        └── luisa_compute_backend_impl
```

`luisa_compute_backend_impl` also depends directly on `luisa_compute_ir_v2` and the four crates above.

## Crate Responsibilities

### `luisa_compute_api_types`
FFI types shared between C++ and Rust. Build script uses `cbindgen` → `api_types.hpp`/`api_types.h`.
- Resource handles: `Buffer`, `Texture`, `Stream`, `Device`, `Shader`, `Accel`, `Mesh`, `Curve`, `ProceduralPrimitive`, `BindlessArray`, `Event`, `Swapchain`, `IrModule`, `NodeRef`
- Commands: `BufferUploadCommand`, `ShaderDispatchCommand`, `AccelBuildCommand`, `MeshBuildCommand`, `CurveBuildCommand`, `ProceduralPrimitiveBuildCommand`, `BindlessArrayUpdateCommand`, etc.
- Pixel formats: `PixelStorage` (28 variants incl. BC compression), `PixelFormat`
- Ray tracing: `AccelOption`, `AccelBuildModification`, `CurveBasis`
- `DeviceInterface` / `LibInterface` (vtable of function pointers), denoiser extension types, pinned-memory extension

### `luisa_compute_ir`
Core IR (rlib). Key deps: `half`, `serde`, `bincode`, `indexmap`, `parking_lot`, `smallvec`, `bitflags`, plus `luisa_compute_api_types`.

**Data structures** (`src/ir.rs`):
```rust
pub enum Type {
    Void,
    UserData,
    Primitive(Primitive),
    Vector(VectorType),
    Matrix(MatrixType),
    Struct(StructType),
    Array(ArrayType),
    Opaque(CBoxedSlice<u8>),
}

pub enum Primitive {
    Bool, Int8, Int16, Int32, Int64,
    Uint8, Uint16, Uint32, Uint64,
    Float16, Float32, Float64,
}

pub struct Node {
    pub type_: CArc<Type>,
    pub next: NodeRef,
    pub prev: NodeRef,
    pub instruction: CArc<Instruction>,
}
// NodeRef is a pool-index handle: pub struct NodeRef(pub usize)

pub enum Instruction {
    Buffer, Bindless, Texture2D, Texture3D, Accel, Shared, Uniform,
    Local { init: NodeRef },
    Argument { by_value: bool },
    UserData(CArc<UserData>),
    Const(Const),
    Update { var: NodeRef, value: NodeRef },
    Call(Func, CBoxedSlice<NodeRef>),
    Phi(CBoxedSlice<PhiIncoming>),
    Return(NodeRef),
    Loop { body: Pooled<BasicBlock>, cond: NodeRef },
    GenericLoop { prepare, cond, body, update: Pooled<BasicBlock> },
    Break, Continue,
    If { cond, true_branch, false_branch: Pooled<BasicBlock> },
    Switch { value, default, cases: CBoxedSlice<SwitchCase> },
    AdScope { body, forward, n_forward_grads },
    RayQuery { ray_query, on_triangle_hit, on_procedural_hit },
    Print { fmt: CBoxedSlice<u8>, args: CBoxedSlice<NodeRef> },
    AdDetach(Pooled<BasicBlock>),
    Comment(CBoxedSlice<u8>),
    Invalid,
}

pub struct Module {
    pub kind: ModuleKind,        // Block | Function | Kernel
    pub entry: Pooled<BasicBlock>,
    pub flags: ModuleFlags,      // REQUIRES_REV_AD_TRANSFORM, REQUIRES_FWD_AD_TRANSFORM
    pub curve_basis_set: CurveBasisSet,
    pub pools: CArc<ModulePools>,
}
```

`Func` enum: ~180 builtins — math (`Add`, `Mul`, `Sin`, `Cos`, `Exp`, `Log`, `Sqrt`), vector/matrix (`Cross`, `Dot`, `Determinant`, `Inverse`, `Transpose`), memory (`BufferRead`/`Write`, `Texture2dRead`), atomic (`AtomicExchange`, `AtomicFetchAdd`), warp (`WarpActiveSum`, `WarpPrefixSum`), ray tracing (`RayTracingTraceClosest`, `RayQueryCommitTriangle`), AD (`RequiresGradient`, `Backward`, `PropagateGrad`, `OutputGrad`), indirect dispatch, raster discard, shader execution reorder, etc.

**Memory**: `CArc<T>` (atomic refcount), `CBox<T>` / `CBoxedSlice<T>` (C-compat boxes), `Pool<T>` (chunked pool), `ModulePools` (separate pools for nodes/blocks).

### `luisa_compute_ir_staticlib` / `luisa_compute_ir_v2`
- **Staticlib**: re-exports `luisa_compute_ir` symbols so C++ can link the static library `luisa_compute_ir_static` (crate folder name ≠ lib name).
- **IR v2**: `libloading`-based Rust wrapper around the IR v2 C API. Loaded via `lc_ir_v2_binding_table` and consumed by `luisa_compute_backend_impl` through `IrV2BindingTable`.

### `luisa_compute_cpu_kernel_defs`
Runtime types passed to CPU kernels: `KernelFnArgs`, `KernelFnArg`, `BufferView`, `Texture`, `BindlessArray`, `Accel`, `Ray`, `Hit`/`TriangleHit`/`ProceduralHit`/`CommittedHit`/`HitType`, `RayQuery`, `CpuCustomOp`, `Aabb`, `Mat4`.

### `luisa_compute_backend`
- `Backend` trait (20+ methods) implemented by concrete backends.
- `Context` loads the C++ `luisa-api` shared library (`luisa-api.dll` / `libluisa-api.so` / `libluisa-api.dylib`) that exports `luisa_compute_lib_interface()`.
- `ProxyBackend` dynamic-dispatches through the C `DeviceInterface` vtable.

### `luisa_compute_backend_impl`
Concrete backend cdylib. Features: `cpu` (enables `embree_sys`), `remote` (stub).
- Exports `luisa_compute_lib_interface()` and `luisa_compute_set_ir_v2_binding(...)`.
- **CPU** (`cpu/`): `RustBackend` with Rayon thread pool, warp size = 1.
  - `shader.rs` — kernel compilation orchestration & cache.
  - `codegen/cpp.rs`, `codegen/cpp_v2.rs` — IR → C++ source.
  - `llvm.rs` — loads `libLLVM` at runtime, parses bitcode, runs `LLJIT`.
  - `accel.rs` — Embree ray tracing.
  - `stream.rs`, `texture.rs`, `resource.rs`.
- **Remote**: network-distributed backend placeholder.

## C-Compatible Pointers (`src/ffi.rs`)

```rust
pub struct CArc<T> { inner: *mut CArcSharedBlock<T> }
pub struct CArcSharedBlock<T> {
    pub(crate) ptr: *mut T,
    ref_count: AtomicUsize,
    destructor: extern "C" fn(*mut CArcSharedBlock<T>),
}

pub struct CBox<T> {
    ptr: *mut T,
    destructor: unsafe extern "C" fn(*mut T),
}

pub struct CBoxedSlice<T> {
    ptr: *mut T,
    len: usize,
    destructor: Option<unsafe extern "C" fn(*mut T, usize)>,
}

pub struct CSlice<'a, T> { ptr: *const T, len: usize, phantom: PhantomData<&'a T> }
pub struct CSliceMut<'a, T> { ptr: *mut T, len: usize, phantom: PhantomData<&'a T> }
```

All are `#[repr(C)]` and designed for zero-cost crossing with C++.

## Transforms (`src/transform/`)

| Transform | Purpose | Pipeline name |
|---|---|---|
| `ssa::ToSSA` | `Local`/`Update` → SSA with `Phi` nodes | `ssa` |
| `autodiff::Autodiff` | Reverse-mode AD | `autodiff` |
| `fwd_autodiff::FwdAutodiff` | Forward-mode AD | *(used by `transform_auto` only)* |
| `dce::Dce` | Dead code elimination | *(struct exists; not registered in pipeline)* |
| `inliner::inline_callable` | Function inlining helper | *(utility, not pipeline-registered)* |
| `canonicalize_control_flow::CanonicalizeControlFlow` | Normalize control flow | `canonicalize_control_flow` |
| `ref2ret::Ref2Ret` | Reference returns → value returns | `ref2ret` |
| `reg2mem::Reg2Mem` | Register → memory conversion | `reg2mem` |

`TransformPipeline` is created from C++ via:

```rust
luisa_compute_ir_transform_pipeline_new() -> *mut TransformPipeline
luisa_compute_ir_transform_pipeline_add_transform(pipeline, name)
luisa_compute_ir_transform_pipeline_transform(pipeline, module) -> Module
luisa_compute_ir_transform_pipeline_destroy(pipeline)
luisa_compute_ir_transform_auto(module) -> Module
```

### Autodiff (`autodiff.rs`)
Reverse-mode: forward sweep marks gradient-requiring nodes, backward sweep accumulates via chain rule. Supports arithmetic, vector (`dot`, `cross`, `length`, `normalize`), matrix (`matmul`, `determinant`, `inverse`, `transpose`), math (`exp`, `log`, `sin`, `cos`, `sqrt`, `pow`, trig), selection (`min`, `max`, `select`, `clamp`).

### SSA (`ssa.rs`)
Promotes `Local`→SSA values, tracks current value in `stored` map, inserts `Phi` at merge points (if/else, loops), supports `GetElementPtr` → `ExtractElement`/`InsertElement`.

### DCE (`dce.rs`)
UseDef analysis, removes pure nodes with no side effects, preserves memory ops and control flow.

## CPU Backend

- **Thread Pool**: Rayon parallel.
- **Warp Size**: 1 (scalar).
- **Shader Pipeline**: IR → C++ source → `clang++ -emit-llvm` → `.bc` bitcode → `libLLVM` C API (`LLJIT`) → native code.
- **Codegen**: `cpu/codegen/cpp.rs` / `cpp_v2.rs`.
- **Ray Tracing**: Embree integration (`accel.rs`).
- **Resources**: `BufferImpl` (aligned host memory), `TextureImpl` (mipmapped), `BindlessArrayImpl`, `AccelImpl` (Embree scene).
- **Swapchain**: CPU backend optionally loads a platform helper DLL (`luisa-backend-cpu.dll` / `.so`) exposing `luisa_compute_create_cpu_swapchain` etc.

## FFI Integration

### Header Generation
`cbindgen` in `build.rs`:
- `luisa_compute_ir` → `include/luisa/rust/ir.hpp`
- `luisa_compute_api_types` → `include/luisa/rust/api_types.hpp` (C++) and `include/luisa/rust/api_types.h` (C)
- `luisa_compute_cpu_kernel_defs` → `cpu_kernel_defs.h` (only when `LC_RS_GENERATE_BINDINGS=1`)

### Key FFI Functions
```rust
// IR transform pipeline
luisa_compute_ir_transform_pipeline_new() -> *mut TransformPipeline
luisa_compute_ir_transform_pipeline_add_transform(pipeline, name)
luisa_compute_ir_transform_pipeline_transform(pipeline, module) -> Module
luisa_compute_ir_transform_pipeline_destroy(pipeline)
luisa_compute_ir_transform_auto(module) -> Module

// Backend loader interface
luisa_compute_lib_interface() -> LibInterface
luisa_compute_set_ir_v2_binding(table: *const IrV2BindingTable)
```

### Conventions
- All FFI types are `#[repr(C)]`.
- Handle types are newtype wrappers around `u64` (e.g., `pub struct Buffer(pub u64)`).
- Callbacks use `extern "C" fn` pointers; `DeviceInterface`/`LibInterface` are fn-pointer vtable structs.
- Static libs for C++ linking: `luisa_compute_api_types`, `luisa_compute_ir_static`.
- Shared backend lib: `luisa_compute_backend_impl` (exports `luisa_compute_lib_interface`).

## CMake Integration

**File**: `src/rust/CMakeLists.txt`.

Custom commands invoke `cargo build`:
- Profile: `dev` in Debug, `release` in Release.
- Features: controlled by CMake options `LUISA_COMPUTE_ENABLE_CPU` (`cpu`) and `LUISA_COMPUTE_ENABLE_REMOTE` (`remote`); passed as `--no-default-features --features <list>`.
- `CARGO_TARGET_DIR` is redirected to the CMake binary dir.

Targets produced:
- `luisa_compute_rust_build` — builds all Rust artifacts.
- `luisa-compute-rust-meta` (INTERFACE) — links the static Rust libs + Windows system libs.
- `luisa_compute_backend_impl` (INTERFACE) — links the shared Rust backend.

Platform handling:
- Windows: copies `.dll`, `.lib`, `.pdb`.
- macOS: `install_name_tool` for rpath/id.
- Linux: `patchelf --set-rpath $ORIGIN`.

Embree (CPU only):
- CMake forwards `LUISA_COMPUTE_EMBREE_ZIP_PATH` or the `EMBREE_ZIP_FILE`/`EMBREE_ZIP_PATH` environment variables to the Rust build.
- The Rust `embree_sys` build script downloads/builds Embree and copies shared libraries to the output directory.

## XMake Integration

`src/rust/xmake.lua` defines target `lc-rust` with the `build_cargo` rule. It sets `LC_RS_DO_NOT_GENERATE_BINDINGS=1` before building to skip cbindgen header generation.

## Common Workflows

### Running Rust checks
```bash
cd src/rust

# Check default workspace (no backend features)
cargo check

# Check with CPU and remote features
# CPU requires the Embree dependency to be available.
cargo check -p luisa_compute_backend_impl --features cpu,remote

# Run tests
cargo test

# Formatting & lints
cargo fmt --check
cargo clippy --workspace -- -D warnings
```

### Adding a transform
1. Create `luisa_compute_ir/src/transform/my_transform.rs`.
2. Implement the `Transform` trait (`fn transform(&self, module: ir::Module) -> ir::Module`).
3. `pub mod my_transform;` in `luisa_compute_ir/src/transform/mod.rs`.
4. If it should be pipeline-selectable from C++, add a match arm in `luisa_compute_ir_transform_pipeline_add_transform`.

### Exposing a new FFI function
1. Add `#[no_mangle] pub extern "C" fn luisa_compute_ir_...` in the appropriate crate (usually `luisa_compute_ir` or `luisa_compute_backend_impl`).
2. Ensure argument/return types are `#[repr(C)]`.
3. Rebuild; `cbindgen` will emit the declaration into `ir.hpp` / `api_types.hpp`.

### Debugging cbindgen output
- Set `LC_RS_DO_NOT_GENERATE_BINDINGS=1` to skip header generation and speed up `cargo check`.
- Force regeneration by unsetting it and running `cargo build` for `luisa_compute_ir` or `luisa_compute_api_types`.
- Verify output under `include/luisa/rust/`.

## Key Design Decisions

1. **Intrusive linked-list IR** with pool allocation for cache efficiency.
2. **Global type registry** with structural equality (`context.rs`).
3. **`CArc`/`CBox`/`CBoxedSlice`** custom smart pointers with C-compatible destructor callbacks.
4. **Pipeline-based transforms**: modular passes (SSA, autodiff, DCE).
5. **CPU JIT**: C++ source → LLVM bitcode → runtime-loaded `libLLVM` / `LLJIT`.
6. **Bidirectional FFI**: `cbindgen` (Rust → C++) + staticlib (C++ → Rust) + cdylib backend loader.
