
#include <Shader/BuiltinKernel.h>
#include "HLSL/dx_codegen.h"
#include <core/stl/filesystem.h>
namespace toolhub::directx {
ComputeShader *BuiltinKernel::LoadAccelSetKernel(Device *device, luisa::BinaryIO const *ctx) {
    auto func = [&] {
        CodegenResult code;
        code.bdlsBufferCount = 0;
        code.result = CodegenUtility::ReadInternalHLSLFile("accel_process", ctx);
        code.properties.resize(3);
        auto &Global = code.properties[0];
        Global.arrSize = 0;
        Global.registerIndex = 0;
        Global.spaceIndex = 0;
        Global.type = ShaderVariableType::ConstantBuffer;
        auto &SetBuffer = code.properties[1];
        SetBuffer.arrSize = 0;
        SetBuffer.registerIndex = 0;
        SetBuffer.spaceIndex = 0;
        SetBuffer.type = ShaderVariableType::StructuredBuffer;
        auto &InstBuffer = code.properties[2];
        InstBuffer.arrSize = 0;
        InstBuffer.registerIndex = 0;
        InstBuffer.spaceIndex = 0;
        InstBuffer.type = ShaderVariableType::RWStructuredBuffer;
        return code;
    };
    return ComputeShader::CompileCompute(
        device->fileIo,
        device,
        {},
        func,
        {},
        uint3(64, 1, 1),
        60,
        "set_accel_kernel.dxil"sv,
        true);
}
namespace detail {
static ComputeShader *LoadBCKernel(
    Device *device,
    vstd::function<vstd::string_view()> const &includeCode,
    vstd::function<vstd::vector<char>()> const &kernelCode,
    vstd::string_view codePath) {
    auto func = [&] {
        CodegenResult code;
        auto incCode = includeCode();
        auto kerCode = kernelCode();
        code.result.reserve(incCode.size() + kerCode.size());
        code.result << incCode << vstd::string_view{kerCode.data(), kerCode.size()};
        code.bdlsBufferCount = 0;
        code.properties.resize(4);
        auto &globalBuffer = code.properties[0];
        globalBuffer.arrSize = 0;
        globalBuffer.registerIndex = 0;
        globalBuffer.spaceIndex = 0;
        globalBuffer.type = ShaderVariableType::ConstantBuffer;

        auto &gInput = code.properties[1];
        gInput.arrSize = 0;
        gInput.registerIndex = 0;
        gInput.spaceIndex = 0;
        gInput.type = ShaderVariableType::SRVDescriptorHeap;

        auto &gInBuff = code.properties[2];
        gInBuff.arrSize = 0;
        gInBuff.registerIndex = 1;
        gInBuff.spaceIndex = 0;
        gInBuff.type = ShaderVariableType::StructuredBuffer;

        auto &gOutBuff = code.properties[3];
        gOutBuff.arrSize = 0;
        gOutBuff.registerIndex = 0;
        gOutBuff.spaceIndex = 0;
        gOutBuff.type = ShaderVariableType::RWStructuredBuffer;
        return code;
    };
    vstd::string fileName;
    vstd::string_view extName = ".dxil"sv;
    fileName.reserve(codePath.size() + extName.size());
    fileName << codePath << extName;
    return ComputeShader::CompileCompute(
        device->fileIo,
        device,
        {},
        func,
        {},
        uint3(1, 1, 1),
        60,
        fileName,
        true);
}
static vstd::string_view Bc6Header(luisa::BinaryIO const *ctx) {
    static auto bc6Header = CodegenUtility::ReadInternalHLSLFileByte("bc6_header", ctx);
    return {bc6Header.data(), bc6Header.size()};
}
static vstd::string_view Bc7Header(luisa::BinaryIO const *ctx) {
    static auto bc7Header = CodegenUtility::ReadInternalHLSLFileByte("bc7_header", ctx);
    return {bc7Header.data(), bc7Header.size()};
}

static vstd::string bc7Header;
}// namespace detail

ComputeShader *BuiltinKernel::LoadBC6TryModeG10CSKernel(Device *device, luisa::BinaryIO const *ctx) {
    return detail::LoadBCKernel(
        device,
        [&] { return detail::Bc6Header(ctx); },
        [&] { return CodegenUtility::ReadInternalHLSLFileByte("bc6_trymode_g10cs", ctx); },
        "bc6_trymodeg10"sv);
}
ComputeShader *BuiltinKernel::LoadBC6TryModeLE10CSKernel(Device *device, luisa::BinaryIO const *ctx) {
    return detail::LoadBCKernel(
        device,
        [&] { return detail::Bc6Header(ctx); },
        [&] { return CodegenUtility::ReadInternalHLSLFileByte("bc6_trymode_le10cs", ctx); },
        "bc6_trymodele10"sv);
}
ComputeShader *BuiltinKernel::LoadBC6EncodeBlockCSKernel(Device *device, luisa::BinaryIO const *ctx) {
    return detail::LoadBCKernel(
        device,
        [&] { return detail::Bc6Header(ctx); },
        [&] { return CodegenUtility::ReadInternalHLSLFileByte("bc6_encode_block", ctx); },
        "bc6_encodeblock"sv);
}
ComputeShader *BuiltinKernel::LoadBC7TryMode456CSKernel(Device *device, luisa::BinaryIO const *ctx) {
    return detail::LoadBCKernel(
        device,
        [&] { return detail::Bc7Header(ctx); },
        [&] { return CodegenUtility::ReadInternalHLSLFileByte("bc7_trymode_456cs", ctx); },
        "bc7_trymode456"sv);
}
ComputeShader *BuiltinKernel::LoadBC7TryMode137CSKernel(Device *device, luisa::BinaryIO const *ctx) {
    return detail::LoadBCKernel(
        device,
        [&] { return detail::Bc7Header(ctx); },
        [&] { return CodegenUtility::ReadInternalHLSLFileByte("bc7_trymode_137cs", ctx); },
        "bc7_trymode137"sv);
}
ComputeShader *BuiltinKernel::LoadBC7TryMode02CSKernel(Device *device, luisa::BinaryIO const *ctx) {
    return detail::LoadBCKernel(
        device,
        [&] { return detail::Bc7Header(ctx); },
        [&] { return CodegenUtility::ReadInternalHLSLFileByte("bc7_trymode_02cs", ctx); },
        "bc7_trymode02"sv);
}
ComputeShader *BuiltinKernel::LoadBC7EncodeBlockCSKernel(Device *device, luisa::BinaryIO const *ctx) {
    return detail::LoadBCKernel(
        device,
        [&] { return detail::Bc7Header(ctx); },
        [&] { return CodegenUtility::ReadInternalHLSLFileByte("bc7_encode_block", ctx); },
        "bc7_encodeblock"sv);
}
}// namespace toolhub::directx