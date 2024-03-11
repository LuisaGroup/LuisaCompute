#pragma once
#include <filesystem>
#include <luisa/core/dynamic_module.h>
#include <wrl/client.h>
#include "dxcapi.h"
#include <luisa/vstl/common.h>
#include <luisa/core/platform.h>

namespace lc::hlsl {
class ShaderCompilerModule : public vstd::IOperatorNewBase {
public:
    luisa::DynamicModule dxil;
    luisa::DynamicModule dxcCompiler;
    IDxcCompiler3 *comp;
    IDxcLibrary *library;
    IDxcUtils *utils;

    ShaderCompilerModule(std::filesystem::path const &path);
    ~ShaderCompilerModule();
};
using Microsoft::WRL::ComPtr;
using CompileResult = vstd::variant<
    ComPtr<IDxcBlob>,
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
        vstd::span<LPCWSTR> args);
    IDxcCompiler3 *compiler();
    IDxcUtils *utils();
    IDxcLibrary *library();

    ShaderCompiler(std::filesystem::path const &path);
    ~ShaderCompiler();
    CompileResult compile_compute(
        vstd::string_view code,
        bool optimize,
        uint shaderModel,
        bool enableUnsafeMath,
        bool spirv);
    RasterBin compile_raster(
        vstd::string_view code,
        bool optimize,
        uint shaderModel,
        bool enableUnsafeMath,
        bool spirv);
    /*CompileResult CompileRayTracing(
        vstd::string_view code,
        bool optimize,
        uint shaderModel = 63);*/
};
}// namespace lc::hlsl
