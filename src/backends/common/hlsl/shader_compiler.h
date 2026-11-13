#pragma once

#include <filesystem>
#include <luisa/core/dynamic_module.h>
#include <luisa/core/stl/memory.h>
#include <luisa/vstl/common.h>
#include <luisa/core/platform.h>
#include <luisa/core/stl/filesystem.h>

#include "dxcapi.h"
#include "hlsl_codegen.h"

namespace lc::hlsl {
template <typename T>
struct ComDeleter {
    inline void operator()(T* blob) const {
        blob->Release();
    }
};
class ShaderCompilerModule : public vstd::IOperatorNewBase {
public:
    luisa::DynamicModule dxil;
    luisa::DynamicModule dxcCompiler;
    IDxcCompiler3 *comp{nullptr};
    IDxcLibrary *library{nullptr};
    IDxcUtils *utils{nullptr};

    ShaderCompilerModule(luisa::filesystem::path const &path, bool is_spirv);
    ~ShaderCompilerModule();
};
template <typename T>
using ComUniquePtr = luisa::unique_ptr<T, ComDeleter<T>>;
using CompileResult = vstd::variant<
    ComUniquePtr<IDxcBlob>,
    vstd::string>;
struct RasterBin {
    CompileResult vertex;
    CompileResult pixel;
};
class ShaderCompiler final : public vstd::IOperatorNewBase {
    ShaderCompilerModule compiler_module;
    // Shared body of `compile_compute` and `compile_graphics_stage`: the two
    // differ only in the profile (target) string, so every HLSL stage flag
    // (`-spirv`, `/DSPV`, `-fspv-target-env`, `-HV 2021`, ...) lives in exactly
    // one place.
    // `spirv_capabilities` names the optional shader-capability extensions the
    // target device negotiated (`SpirvShaderCapabilities`).  It only selects the
    // `-fspv-extension=` flags of the SPIR-V route: dxc rejects an extension name it
    // does not know, so a capability is only enabled when the caller asks for it, and
    // the default (nothing) passes exactly the pre-feature flag set.
    CompileResult _compile_with_profile(
        vstd::string_view code,
        vstd::wstring_view profile,
        bool optimize,
        uint shaderModel,
        bool enableUnsafeMath,
        bool spirv,
        bool debug,
        // Entry point to compile. Empty keeps the historical behaviour: DXC's
        // default entry point, i.e. a function named "main".
        vstd::string_view entry_point,
        // Directories searched for `#include`d headers, in addition to the
        // raw include spelling itself. Empty keeps the historical behaviour:
        // no include handler is registered.
        luisa::span<const luisa::filesystem::path> include_dirs,
        SpirvShaderCapabilities spirv_capabilities = {}) const;

public:
    // Optional include handler for `#include` resolution. nullptr keeps
    // the historical behaviour: no includes are resolved at all.
    CompileResult compile(
        vstd::string_view code,
        vstd::span<LPCWSTR> args,
        IDxcIncludeHandler *include_handler = nullptr) const;
    IDxcCompiler3 *compiler() const;
    IDxcUtils *utils() const;
    IDxcLibrary *library() const;

    ShaderCompiler(luisa::filesystem::path const &path, bool is_spirv);
    ~ShaderCompiler();
    CompileResult compile_compute(
        vstd::string_view code,
        bool optimize,
        uint shaderModel,
        bool enableUnsafeMath,
        bool spirv,
        bool debug,
        // Entry point to compile. Empty keeps the historical behaviour: DXC's
        // default entry point, i.e. a function named "main". Native shaders
        // routinely declare a differently named entry point (CSMain, ...) and
        // are compiled through this overload.
        vstd::string_view entry_point = {},
        // Directories searched for `#include`d headers, in addition to the
        // raw include spelling itself. Empty keeps the historical behaviour:
        // no include handler is registered.
        luisa::span<const luisa::filesystem::path> include_dirs = {},
        SpirvShaderCapabilities spirv_capabilities = {}) const;
    // Compiles *one* stage of a graphics pipeline: profile `vs_<sm>` when
    // `vertex` is true, `ps_<sm>` otherwise. Behaviour is otherwise identical
    // to `compile_compute` (same SPIR-V flags, same entry-point/include
    // handling).
    //
    // This is deliberately not `compile_raster`: that one compiles both stages
    // of a *single* translation unit and selects them with `/DVS`/`/DPS`, which
    // is the DSL raster codegen's contract, not the native raster one, where the
    // application hands in one source per stage.
    CompileResult compile_graphics_stage(
        vstd::string_view code,
        bool vertex,
        bool optimize,
        uint shaderModel,
        bool enableUnsafeMath,
        bool spirv,
        bool debug,
        vstd::string_view entry_point = {},
        luisa::span<const luisa::filesystem::path> include_dirs = {},
        SpirvShaderCapabilities spirv_capabilities = {}) const;
    RasterBin compile_raster(
        vstd::string_view code,
        bool optimize,
        uint shaderModel,
        bool enableUnsafeMath,
        bool spirv,
        bool debug,
        SpirvShaderCapabilities spirv_capabilities = {}) const;
    CompileResult compile_raytracing(
        vstd::string_view code,
        bool optimize,
        uint shaderModel,
        bool enableUnsafeMath,
        bool spirv,
        bool debug,
        SpirvShaderCapabilities spirv_capabilities = {}) const;
};
}// namespace lc::hlsl
