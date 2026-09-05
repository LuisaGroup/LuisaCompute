#pragma once

#include <array>
#include <cstdint>
#include <utility>

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/optional.h>
#include <tvm/ir/module.h>
#include <tvm/tirx/function.h>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/tile/bridge/tirx/planner.h>

namespace luisa::compute::tile::bridge::tirx {

enum class PipelineKind : uint8_t {
    // Ordinary native TIRx statements and expressions.
    STANDARD,
    // Native TIRx TilePrimitive calls that require LowerTIRx.
    TILE
};

enum class CpuMatrixBackend : uint8_t {
    // Keep the portable loop/SIMD realization emitted by the TIRx bridge.
    REFERENCE,
    // Realize a proved whole-kernel FP32 GEMM contract as the registered
    // tvm.contrib.cblas.matmul runtime atom. Availability is checked at
    // compile time and unsupported contracts fail closed.
    CBLAS
};

enum class CpuMathBackend : uint8_t {
    // Keep target LLVM/libm scalar semantics after ordinary vectorization.
    REFERENCE,
    // Realize proved, compiler-materialized FP32 exp maps with Apple's
    // synchronous array provider and exact add/max/min reduction contracts
    // with vDSP. This explicit policy permits provider reduction order and
    // vForce's documented denormal/exception differences. It is a target
    // choice, never a Tile DSL operation or execution-hierarchy change.
    ACCELERATE
};

struct CompileOptions {
    // A target kind or native TVMx JSON target configuration. For standalone
    // CPU programs this selects the ISA of both compute and its packed entry.
    luisa::string target{"llvm"};
    // CPU code for GPU launch wrappers. A standalone CPU target is its own
    // effective host; this option must not replace its CPU/features/ABI.
    luisa::string host{"llvm"};
    PipelineKind pipeline{PipelineKind::STANDARD};
    // Disabling vectorization rejects an explicit vector execution constraint
    // instead of silently replacing the requested mapping with a serial loop.
    bool vectorize{true};
    // Opt-in CPU packing of inferred independent-element domains. It preserves
    // inner serial/reduction order, but is not uniformly profitable yet.
    // Explicit vector execution bindings only require vectorize, not this.
    bool auto_vectorize{false};
    CpuMatrixBackend cpu_matrix_backend{CpuMatrixBackend::REFERENCE};
    CpuMathBackend cpu_math_backend{CpuMathBackend::REFERENCE};
    bool eliminate_common_subexpressions{true};
    // This is a caller contract, not something the bridge can infer from raw
    // PrimFunc parameters. Keep the conservative default when buffers may
    // alias, including when compiling directly lowered TileIR today.
    bool noalias{false};
    // Opt-in capability contract for cross-compilation: the selected device
    // supports native FP32 cooperative matrices. Metal additionally requires
    // thread_warp_size=32 (Apple GPU family 7+). Generic "metal" is not enough
    // to infer this feature. Ineligible shapes, scopes, or MMA policies keep
    // the reference realization; no input precision is reduced.
    bool cooperative_matrix{false};
    // Experimental Metal 4 / MPP memory-input realization through TVM's own
    // codegen, not the native backend. Requires the versioned TVM extension;
    // unsupported installations fail closed. The existing solver selects the
    // same legal geometry using SIMD-group reference costs, not MPP timings.
    bool metal_mpp{false};
    // Optional snapshot-to-view forwarding before resource planning. Requires
    // noalias and proves immutable input, complete initialization, lexical
    // dominance, non-escape, and bounds. Ordinary CPU/GPU consumers retain
    // guarded/zero-fill read expressions; Metal MPP requires fully in-bounds
    // memory-input views.
    // No materialization policy is inferred from an external buffer's scope.
    bool forward_readonly_tile_loads{false};
    PlannerOptions planner;
};

class LUISA_TILE_TIRX_BRIDGE_API CompilationResult final {

private:
    tvm::ffi::Optional<tvm::ffi::Module> _module;
    luisa::string _error;
    luisa::vector<GroupPlan> _plans;

public:
    CompilationResult() noexcept = default;
    explicit CompilationResult(tvm::ffi::Module module, luisa::vector<GroupPlan> plans = {}) noexcept
        : _module{std::move(module)}, _plans{std::move(plans)} {}
    explicit CompilationResult(luisa::string error) noexcept
        : _error{std::move(error)} {}

    [[nodiscard]] bool ok() const noexcept { return _module.has_value() && _error.empty(); }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
    [[nodiscard]] const tvm::ffi::Optional<tvm::ffi::Module> &module() const noexcept { return _module; }
    [[nodiscard]] luisa::string_view error() const noexcept { return _error; }
    [[nodiscard]] luisa::span<const GroupPlan> plans() const noexcept { return _plans; }
};

// A single, statically launched device entry. ABI and launch information come
// from typed TIRx, never from parsing the generated source. The original host
// parameter order is preserved by buffer_arguments[device_binding_index].
struct DeviceArtifact {
    enum class Format : uint8_t { METAL_SOURCE };
    Format format{Format::METAL_SOURCE};
    tvm::tirx::PrimFunc function;
    luisa::string entry;
    luisa::string source;
    std::array<uint32_t, 3u> grid{1u, 1u, 1u};
    std::array<uint32_t, 3u> block{1u, 1u, 1u};
    luisa::vector<uint32_t> buffer_arguments;
    bool requires_metal4{false};
};

struct DeviceCompilationResult {
    DeviceArtifact artifact;
    luisa::vector<GroupPlan> plans;
    luisa::string error;
    [[nodiscard]] explicit operator bool() const noexcept {
        return error.empty() && artifact.function.defined() && !artifact.source.empty();
    }
};

// Shares execution mapping and device passes with compile(), but does not
// generate a packed host wrapper or execute through TVM's device runtime.
// Initially accepts Metal, buffer-only ABI, one unconditional static launch.
// Host loops/effects, multiple launches, scalar arguments, and dynamic launch
// resources are rejected; silently dropping host work is never permitted.
[[nodiscard]] LUISA_TILE_TIRX_BRIDGE_API DeviceCompilationResult compile_device(
    tvm::tirx::PrimFunc function, luisa::string_view name,
    const CompileOptions &options) noexcept;

// Compiles a native TIRx module through the C++ pass and target registries.
// No Python runtime, source generation, or TVMScript parsing participates in
// this path. Device modules are finalized independently and imported into the
// host module, matching TVM's native module model.
[[nodiscard]] LUISA_TILE_TIRX_BRIDGE_API CompilationResult compile(
    tvm::IRModule module,
    const CompileOptions &options = {}) noexcept;

// Convenience overload for a single public PrimFunc. The supplied name is
// both the IRModule symbol and the unprefixed packed-function lookup name.
[[nodiscard]] LUISA_TILE_TIRX_BRIDGE_API CompilationResult compile(
    tvm::tirx::PrimFunc function,
    luisa::string_view name,
    const CompileOptions &options = {}) noexcept;

}// namespace luisa::compute::tile::bridge::tirx
