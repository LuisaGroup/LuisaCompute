#pragma once
#include <filesystem>
#include <luisa/core/dynamic_module.h>
#include <wrl/client.h>
#include "dxcapi.h"
#include <luisa/vstl/common.h>
#include <luisa/core/platform.h>

namespace lc::hlsl {
using Microsoft::WRL::ComPtr;
class DxcByteBlob final : public vstd::IOperatorNewBase {
private:
    ComPtr<IDxcBlob> blob;

public:
    DxcByteBlob(Microsoft::WRL::ComPtr<IDxcBlob> &&b);
    std::byte *data() const;
    size_t size() const;
};

using CompileResult = vstd::variant<
    vstd::unique_ptr<DxcByteBlob>,
    vstd::string>;
struct RasterBin {
    CompileResult vertex;
    CompileResult pixel;
};
class ShaderCompilerModule {
public:
    luisa::DynamicModule dxil;
    luisa::DynamicModule dxcCompiler;
    ComPtr<IDxcCompiler3> comp;
    ShaderCompilerModule(std::filesystem::path const &path);
    ~ShaderCompilerModule();
};
class ShaderCompiler final : public vstd::IOperatorNewBase {
private:
    vstd::optional<ShaderCompilerModule> module;
    std::mutex moduleInstantiateMtx;
    std::filesystem::path path;
    CompileResult compile(
        vstd::string_view code,
        vstd::span<LPCWSTR> args);
    IDxcCompiler3 *compiler();

public:
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
