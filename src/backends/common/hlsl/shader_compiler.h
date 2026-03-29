#pragma once

#include <filesystem>
#include <luisa/core/dynamic_module.h>
#include <luisa/vstl/common.h>
#include <luisa/core/platform.h>

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

    ShaderCompilerModule(std::filesystem::path const &path, bool is_spirv);
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
    CompileResult compile(
        vstd::string_view code,
        vstd::span<LPCWSTR> args) const;
    IDxcCompiler3 *compiler() const;
    IDxcUtils *utils() const;
    IDxcLibrary *library() const;

    ShaderCompiler(std::filesystem::path const &path, bool is_spirv);
    ~ShaderCompiler();
    CompileResult compile_compute(
        vstd::string_view code,
        bool optimize,
        uint shaderModel,
        bool enableUnsafeMath,
        bool spirv,
        bool debug) const;
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
    [[nodiscard]] CompileResult compile_work_graph(
        vstd::string_view code,
        bool optimize,
        uint shaderModel,
        bool enableUnsafeMath,
        bool debug) const;
};
}// namespace lc::hlsl
