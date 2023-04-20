
#include <Shader/BuiltinKernel.h>
#include "backends/common/hlsl/hlsl_codegen.h"
#include <core/stl/filesystem.h>
namespace lc::dx {
ComputeShader *BuiltinKernel::LoadAccelSetKernel(Device *device, luisa::BinaryIO const *ctx) {
    auto func = [&] {
        hlsl::CodegenResult code;
        code.useBufferBindless = false;
        code.useTex2DBindless = false;
        code.useTex3DBindless = false;
        code.result = hlsl::CodegenUtility::ReadInternalHLSLFile("accel_process", ctx);
        code.properties.resize(3);
        auto &Global = code.properties[0];
        Global.arrSize = 1;
        Global.registerIndex = 0;
        Global.spaceIndex = 0;
        Global.type = hlsl::ShaderVariableType::ConstantBuffer;
        auto &SetBuffer = code.properties[1];
        SetBuffer.arrSize = 1;
        SetBuffer.registerIndex = 0;
        SetBuffer.spaceIndex = 0;
        SetBuffer.type = hlsl::ShaderVariableType::StructuredBuffer;
        auto &InstBuffer = code.properties[2];
        InstBuffer.arrSize = 1;
        InstBuffer.registerIndex = 0;
        InstBuffer.spaceIndex = 0;
        InstBuffer.type = hlsl::ShaderVariableType::RWStructuredBuffer;
        return code;
    };
    return ComputeShader::CompileCompute(
        device->fileIo,
        device,
        {},
        func,
        {},
        {},
        uint3(64, 1, 1),
        62,
        "set_accel_kernel.dxil"sv,
        CacheType::Internal, true);
}
namespace detail {
static ComputeShader *LoadBCKernel(
    Device *device,
    vstd::function<vstd::string_view()> const &includeCode,
    vstd::function<vstd::vector<char>()> const &kernelCode,
    vstd::string_view codePath) {
    auto func = [&] {
        hlsl::CodegenResult code;
        auto incCode = includeCode();
        auto kerCode = kernelCode();
        code.result.reserve(incCode.size() + kerCode.size());
        code.result << incCode << vstd::string_view{kerCode.data(), kerCode.size()};
        code.useBufferBindless = false;
        code.useTex2DBindless = false;
        code.useTex3DBindless = false;
        code.properties.resize(4);
        auto &globalBuffer = code.properties[0];
        globalBuffer.arrSize = 1;
        globalBuffer.registerIndex = 0;
        globalBuffer.spaceIndex = 0;
        globalBuffer.type = hlsl::ShaderVariableType::ConstantBuffer;

        auto &gInput = code.properties[1];
        gInput.arrSize = 1;
        gInput.registerIndex = 0;
        gInput.spaceIndex = 0;
        gInput.type = hlsl::ShaderVariableType::SRVTextureHeap;

        auto &gInBuff = code.properties[2];
        gInBuff.arrSize = 1;
        gInBuff.registerIndex = 1;
        gInBuff.spaceIndex = 0;
        gInBuff.type = hlsl::ShaderVariableType::StructuredBuffer;

        auto &gOutBuff = code.properties[3];
        gOutBuff.arrSize = 1;
        gOutBuff.registerIndex = 0;
        gOutBuff.spaceIndex = 0;
        gOutBuff.type = hlsl::ShaderVariableType::RWStructuredBuffer;
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
        {},
        uint3(1, 1, 1),
        62,
        fileName,
        CacheType::Internal, true);
}
static vstd::string_view Bc6Header(luisa::BinaryIO const *ctx) {
    static auto bc6Header = hlsl::CodegenUtility::ReadInternalHLSLFileByte("bc6_header", ctx);
    return {bc6Header.data(), bc6Header.size()};
}
static vstd::string_view Bc7Header(luisa::BinaryIO const *ctx) {
    static auto bc7Header = hlsl::CodegenUtility::ReadInternalHLSLFileByte("bc7_header", ctx);
    return {bc7Header.data(), bc7Header.size()};
}

static vstd::string bc7Header;
}// namespace detail

ComputeShader *BuiltinKernel::LoadBC6TryModeG10CSKernel(Device *device, luisa::BinaryIO const *ctx) {
    return detail::LoadBCKernel(
        device,
        [&] { return detail::Bc6Header(ctx); },
        [&] { return hlsl::CodegenUtility::ReadInternalHLSLFileByte("bc6_trymode_g10cs", ctx); },
        "bc6_trymodeg10"sv);
}
ComputeShader *BuiltinKernel::LoadBC6TryModeLE10CSKernel(Device *device, luisa::BinaryIO const *ctx) {
    return detail::LoadBCKernel(
        device,
        [&] { return detail::Bc6Header(ctx); },
        [&] { return hlsl::CodegenUtility::ReadInternalHLSLFileByte("bc6_trymode_le10cs", ctx); },
        "bc6_trymodele10"sv);
}
ComputeShader *BuiltinKernel::LoadBC6EncodeBlockCSKernel(Device *device, luisa::BinaryIO const *ctx) {
    return detail::LoadBCKernel(
        device,
        [&] { return detail::Bc6Header(ctx); },
        [&] { return hlsl::CodegenUtility::ReadInternalHLSLFileByte("bc6_encode_block", ctx); },
        "bc6_encodeblock"sv);
}
ComputeShader *BuiltinKernel::LoadBC7TryMode456CSKernel(Device *device, luisa::BinaryIO const *ctx) {
    return detail::LoadBCKernel(
        device,
        [&] { return detail::Bc7Header(ctx); },
        [&] { return hlsl::CodegenUtility::ReadInternalHLSLFileByte("bc7_trymode_456cs", ctx); },
        "bc7_trymode456"sv);
}
ComputeShader *BuiltinKernel::LoadBC7TryMode137CSKernel(Device *device, luisa::BinaryIO const *ctx) {
    return detail::LoadBCKernel(
        device,
        [&] { return detail::Bc7Header(ctx); },
        [&] { return hlsl::CodegenUtility::ReadInternalHLSLFileByte("bc7_trymode_137cs", ctx); },
        "bc7_trymode137"sv);
}
ComputeShader *BuiltinKernel::LoadBC7TryMode02CSKernel(Device *device, luisa::BinaryIO const *ctx) {
    return detail::LoadBCKernel(
        device,
        [&] { return detail::Bc7Header(ctx); },
        [&] { return hlsl::CodegenUtility::ReadInternalHLSLFileByte("bc7_trymode_02cs", ctx); },
        "bc7_trymode02"sv);
}
ComputeShader *BuiltinKernel::LoadBC7EncodeBlockCSKernel(Device *device, luisa::BinaryIO const *ctx) {
    return detail::LoadBCKernel(
        device,
        [&] { return detail::Bc7Header(ctx); },
        [&] { return hlsl::CodegenUtility::ReadInternalHLSLFileByte("bc7_encode_block", ctx); },
        "bc7_encodeblock"sv);
}
}// namespace lc::dx