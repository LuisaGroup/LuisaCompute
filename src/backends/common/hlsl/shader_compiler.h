#pragma once

#include <filesystem>
#include <luisa/core/dynamic_module.h>
#include <luisa/core/stl/memory.h>
#include <luisa/vstl/common.h>
#include <luisa/core/platform.h>
#include <luisa/core/stl/filesystem.h>

#include "dxcapi.h"

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
        luisa::span<const luisa::filesystem::path> include_dirs = {}) const;
    RasterBin compile_raster(
        vstd::string_view code,
        bool optimize,
        uint shaderModel,
        bool enableUnsafeMath,
        bool spirv,
        bool debug) const;
    CompileResult compile_raytracing(
        vstd::string_view code,
        bool optimize,
        uint shaderModel,
        bool enableUnsafeMath,
        bool spirv,
        bool debug) const;
};
}// namespace lc::hlsl
