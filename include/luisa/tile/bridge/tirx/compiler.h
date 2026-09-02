#pragma once

#include <cstdint>
#include <utility>

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/optional.h>
#include <tvm/ir/module.h>
#include <tvm/tirx/function.h>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>

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
    bool vectorize{true};
    bool eliminate_common_subexpressions{true};
    // This is a caller contract, not something the bridge can infer from raw
    // PrimFunc parameters. Keep the conservative default when buffers may
    // alias, including when compiling directly lowered TileIR today.
    bool noalias{false};
};

class LUISA_TILE_TIRX_BRIDGE_API CompilationResult final {

private:
    tvm::ffi::Optional<tvm::ffi::Module> _module;
    luisa::string _error;

public:
    CompilationResult() noexcept = default;
    explicit CompilationResult(tvm::ffi::Module module) noexcept
        : _module{std::move(module)} {}
    explicit CompilationResult(luisa::string error) noexcept
        : _error{std::move(error)} {}

    [[nodiscard]] bool ok() const noexcept { return _module.has_value() && _error.empty(); }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
    [[nodiscard]] const tvm::ffi::Optional<tvm::ffi::Module> &module() const noexcept { return _module; }
    [[nodiscard]] luisa::string_view error() const noexcept { return _error; }
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
