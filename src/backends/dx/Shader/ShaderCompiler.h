#pragma once
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
    vbyte *GetBufferPtr() const;
    size_t GetBufferSize() const;
};
using CompileResult = vstd::variant<
    vstd::unique_ptr<DXByteBlob>,
    vstd::string>;
class DXShaderCompiler final : public vstd::IOperatorNewBase {
private:
    ComPtr<IDxcCompiler3> comp;
    CompileResult Compile(
        vstd::string_view code,
        vstd::span<LPCWSTR> args);

public:
    DXShaderCompiler();
    ~DXShaderCompiler();
    CompileResult CompileCompute(
        vstd::string_view code,
        bool optimize,
        uint shaderModel = 63);
    CompileResult CompileRayTracing(
        vstd::string_view code,
        bool optimize,
        uint shaderModel = 63);
};
}// namespace toolhub::directx