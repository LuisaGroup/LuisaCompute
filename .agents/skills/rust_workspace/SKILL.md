---
name: rust_workspace
description: Rust workspace — IR data structures, compiler transforms, CPU backend, FFI with C++, crate dependencies
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
├── luisa_compute_ir_v2/               # IR v2 bindings
├── luisa_compute_cpu_kernel_defs/     # CPU kernel runtime types
├── luisa_compute_backend/             # Backend trait definitions
└── luisa_compute_backend_impl/        # CPU + Remote backends (cdylib)
```

### Dependency Graph
```
luisa_compute_api_types
  ├── luisa_compute_ir
  │     ├── luisa_compute_ir_staticlib
  │     ├── luisa_compute_ir_v2
  │     └── luisa_compute_backend
  │           └── luisa_compute_backend_impl (uses luisa_compute_cpu_kernel_defs)
  └── luisa_compute_cpu_kernel_defs
```

## Crate Responsibilities

### `luisa_compute_api_types`
FFI types shared between C++ and Rust. Build script uses `cbindgen` → `api_types.hpp`/`api_types.h`.
- Resource handles: `Buffer`, `Texture`, `Stream`, `Device`, `Shader`, `Accel`, `Mesh`
- Commands: `BufferUploadCommand`, `ShaderDispatchCommand`, `AccelBuildCommand`
- Pixel formats: `PixelStorage`, `PixelFormat` (28 variants incl. BC compression)
- Ray tracing: `AccelOption`, `AccelBuildModification`, `CurveBasis`
- `DeviceInterface` (vtable of function pointers), denoiser extension types

### `luisa_compute_ir`
Core IR (rlib). Deps: `half`, `serde`, `bincode`, `indexmap`.

**Data structures** (`src/ir.rs`):
```rust
pub enum Type { Void, Primitive(Primitive), Vector(VectorType), Matrix(MatrixType),
                Struct(StructType), Array(ArrayType), Opaque(CBoxedSlice<u8>) }
pub enum Primitive { Bool, Int8/16/32/64, Uint8/16/32/64, Float16/32/64 }

pub struct Node { pub type_: CArc<Type>, pub next/prev: NodeRef, pub instruction: CArc<Instruction> }

pub enum Instruction {
    Local { init: NodeRef }, Update { var, value: NodeRef },
    Call(Func, CBoxedSlice<NodeRef>), Phi(CBoxedSlice<PhiIncoming>),
    If { cond, true_branch, false_branch: Pooled<BasicBlock> },
    Loop { body: Pooled<BasicBlock>, cond: NodeRef },
    GenericLoop { prepare, cond, body, update: Pooled<BasicBlock> },
    Switch { value, default, cases: CBoxedSlice<SwitchCase> },
    AdScope { body, forward, n_forward_grads },
    RayQuery { ray_query, on_triangle_hit, on_procedural_hit },
    // ... 20+ more
}
```

`Func` enum: 180+ builtins — math (Add, Mul, Sin, Cos, Exp, Log, Sqrt), vector/matrix (Cross, Dot, Determinant, Inverse, Transpose), memory (BufferRead/Write, Texture2dRead), atomic (AtomicExchange, AtomicFetchAdd), warp (WarpActiveSum, WarpPrefixSum), ray tracing (RayTracingTraceClosest, RayQueryCommitTriangle), AD (RequiresGradient, Backward, PropagateGrad).

**Memory**: `CArc<T>` (atomic refcount), `CBoxedSlice<T>` (C-compat boxed slice), `Pool<T>` (chunked pool), `ModulePools` (separate pools for nodes/blocks).

### `luisa_compute_ir_staticlib` / `luisa_compute_ir_v2`
Staticlib: exports all `luisa_compute_ir` symbols for C++ linking. IR v2: `libloading`-based dynamic loading.

### `luisa_compute_cpu_kernel_defs`
Runtime types: `KernelFnArgs`, `BufferView`, `Texture`, `Accel`, `Ray`, `Hit`, `RayQuery`, `CpuCustomOp`.

### `luisa_compute_backend`
`Backend` trait (20+ methods), `Context` (loads backend DLLs), `ProxyBackend` (dynamic dispatch).

### `luisa_compute_backend_impl`
Concrete backends (cdylib, features: `cpu`, `remote`).
- **CPU** (`cpu/`): `RustBackend` with Rayon thread pool, warp size=1. `shader.rs` — JIT via Clang/LLVM. `codegen/cpp.rs` — IR→C++→LLVM IR→machine code. `accel.rs` — Embree ray tracing. `stream.rs`, `texture.rs`, `resource.rs`.
- **Remote**: network-distributed backend.

## C-Compatible Pointers (`src/ffi.rs`)

```rust
pub struct CArc<T> { pub ptr: *mut T, pub inner: *mut CArcInner }
pub struct CBox<T> { pub ptr: *mut T, pub drop: extern "C" fn(*mut T) }
pub struct CBoxedSlice<T> { pub ptr: *mut T, pub len: usize, pub drop: extern "C" fn(*mut T, usize) }
```

## Transforms (`src/transform/`)

| Transform | Purpose |
|---|---|
| `ssa::ToSSA` | Update→SSA with Phi nodes |
| `autodiff::Autodiff` | Reverse-mode AD |
| `fwd_autodiff::FwdAutodiff` | Forward-mode AD |
| `dce::Dce` | Dead code elimination |
| `inliner::inline_callable` | Function inlining |
| `canonicalize_control_flow` | Normalize control flow |
| `ref2ret::Ref2Ret` | Reference returns → value returns |
| `reg2mem::Reg2Mem` | Register→memory conversion |

### Autodiff (`autodiff.rs`)
Reverse-mode: forward sweep marks gradient-requiring nodes, backward sweep accumulates via chain rule. Supports: arithmetic, vector (dot, cross, length, normalize), matrix (matmul, determinant, inverse, transpose), math (exp, log, sin, cos, sqrt, pow, trig), selection (min, max, select, clamp).

### SSA (`ssa.rs`)
Promotes `Local`→SSA values, tracks current value in `stored` map, inserts `Phi` at merge points (if/else, loops), supports GEP→ExtractElement/InsertElement.

### DCE (`dce.rs`)
UseDef analysis, removes pure nodes with no side effects, preserves memory ops and control flow.

## CPU Backend

- **Thread Pool**: Rayon parallel
- **Warp Size**: 1 (scalar)
- **Shader**: JIT via Clang/LLVM → shared lib
- **Codegen**: IR → C++ → LLVM IR → machine code
- **Ray Tracing**: Embree integration
- **Resources**: `BufferImpl` (aligned host memory), `TextureImpl` (mipmapped), `BindlessArrayImpl`, `AccelImpl` (Embree scene)

## FFI Integration

### Header Generation
`cbindgen` in `build.rs`: parse Rust → generate `ir.hpp`, `api_types.hpp`, `api_types.h` → `include/luisa/rust/`.

### Key FFI Functions
```rust
luisa_compute_ir_transform_pipeline_new() -> *mut TransformPipeline
luisa_compute_ir_transform_pipeline_add_transform(pipeline, name)
luisa_compute_ir_transform_pipeline_transform(pipeline, module) -> Module
luisa_compute_ir_transform_auto(module) -> Module
luisa_compute_lib_interface() -> LibInterface
```

### Conventions
- All types `#[repr(C)]`; handle types are `u64` wrappers (`Buffer(u64)`, `Texture(u64)`)
- Callbacks use `extern "C" fn` pointers; `DeviceInterface` is a fn-pointer vtable struct

## CMake Integration

**File**: `src/rust/CMakeLists.txt`. Custom command invokes `cargo build` (profile: `dev`/`release`, features: `cpu`, `remote`).

- `luisa-compute-rust-meta` (INTERFACE): static Rust libs + Windows system libs
- `luisa_compute_backend_impl` (INTERFACE): shared Rust backend
- Platform: macOS `install_name_tool`, Linux `patchelf`, Windows DLL/lib/pdb copy
- Embree: `EMBREE_ZIP_FILE` env var, downloads/builds via Rust build script, copies DLLs

## Key Design Decisions

1. **Intrusive linked-list IR** with pool allocation for cache efficiency
2. **Global type registry** with structural equality (`context.rs`)
3. **`CArc`** custom smart pointers with C-compatible destructor callbacks
4. **Pipeline-based transforms**: modular passes (SSA, autodiff, DCE)
5. **CPU JIT**: LLVM/Clang compilation to native code
6. **Bidirectional FFI**: cbindgen (Rust→C++) + staticlib (C++→Rust)
