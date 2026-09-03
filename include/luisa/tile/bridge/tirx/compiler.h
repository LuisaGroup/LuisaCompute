#pragma once

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

struct CompileOptions {
    luisa::string target{"llvm"};
    luisa::string host{"llvm"};
    PipelineKind pipeline{PipelineKind::STANDARD};
    // Disabling vectorization rejects an explicit vector execution constraint
    // instead of silently replacing the requested mapping with a serial loop.
    bool vectorize{true};
    // Opt-in CPU packing of inferred independent-element domains. It preserves
    // inner serial/reduction order, but is not uniformly profitable yet.
    // Explicit vector execution bindings only require vectorize, not this.
    bool auto_vectorize{false};
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
