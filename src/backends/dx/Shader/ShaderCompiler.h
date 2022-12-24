#pragma once
#include <filesystem>
#include <core/dynamic_module.h>
#include <DXRuntime/Device.h>
#include <dxc/dxcapi.h>
namespace toolhub::directx {

class DXByteBlob final : public vstd::IOperatorNewBase {
private:
    ComPtr<IDxcBlob> blob;
    ComPtr<IDxcResult> comRes;

public:
    DXByteBlob(
        ComPtr<IDxcBlob> &&b,
        ComPtr<IDxcResult> &&rr);
    std::byte *GetBufferPtr() const;
    size_t GetBufferSize() const;
};

using CompileResult = vstd::variant<
    vstd::unique_ptr<DXByteBlob>,
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
    ShaderCompilerModule(std::filesystem::path const& path);
    ~ShaderCompilerModule();
};
class ShaderCompiler final : public vstd::IOperatorNewBase {
private:
    vstd::optional<ShaderCompilerModule> module;
    std::mutex moduleInstantiateMtx;
    std::filesystem::path path;
    CompileResult Compile(
        vstd::string_view code,
        vstd::span<LPCWSTR> args);

public:
    IDxcCompiler3* Compiler();
    ShaderCompiler(std::filesystem::path const &path);
    ~ShaderCompiler();
    CompileResult CompileCompute(
        vstd::string_view code,
        bool optimize,
        uint shaderModel);
    RasterBin CompileRaster(
        vstd::string_view code,
        bool optimize,
        uint shaderModel);
#ifdef SHADER_COMPILER_TEST
    CompileResult CustomCompile(
        vstd::string_view code,
        vstd::span<vstd::string const> args);
#endif
    /*CompileResult CompileRayTracing(
        vstd::string_view code,
        bool optimize,
        uint shaderModel = 63);*/
};
}// namespace toolhub::directx