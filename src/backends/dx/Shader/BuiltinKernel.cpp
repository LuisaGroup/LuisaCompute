#include <Shader/BuiltinKernel.h>
#include <luisa/core/stl/filesystem.h>
#include "../common/hlsl/hlsl_codegen.h"
namespace lc::dx {
ComputeShader *BuiltinKernel::LoadAccelSetKernel(Device *device, luisa::BinaryIO const *ctx) {
    auto func = [&] {
        hlsl::CodegenResult code;
        code.useBufferBindless = false;
        code.useTex2DBindless = false;
        code.useTex3DBindless = false;
        code.result << hlsl::CodegenUtility::ReadInternalHLSLFile("accel_process", ctx);
        code.properties.resize(3);
        auto &Global = code.properties[0];
        Global.array_size = 1;
        Global.register_index = 0;
        Global.space_index = 0;
        Global.type = hlsl::ShaderVariableType::ConstantBuffer;
        auto &SetBuffer = code.properties[1];
        SetBuffer.array_size = 1;
        SetBuffer.register_index = 0;
        SetBuffer.space_index = 0;
        SetBuffer.type = hlsl::ShaderVariableType::StructuredBuffer;
        auto &InstBuffer = code.properties[2];
        InstBuffer.array_size = 1;
        InstBuffer.register_index = 0;
        InstBuffer.space_index = 0;
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
        uint3(256, 1, 1),
        62,
        "set_accel2.dxil"sv,
        CacheType::Internal, true);
}
ComputeShader *BuiltinKernel::LoadBindlessSetKernel(Device *device, luisa::BinaryIO const *ctx) {
    auto func = [&] {
        hlsl::CodegenResult code;
        code.useBufferBindless = false;
        code.useTex2DBindless = false;
        code.useTex3DBindless = false;
        code.result << hlsl::CodegenUtility::ReadInternalHLSLFile("bindless_upload", ctx);
        code.properties.resize(3);
        auto &Global = code.properties[0];
        Global.array_size = 1;
        Global.register_index = 0;
        Global.space_index = 0;
        Global.type = hlsl::ShaderVariableType::ConstantBuffer;
        auto &SetBuffer = code.properties[1];
        SetBuffer.array_size = 1;
        SetBuffer.register_index = 0;
        SetBuffer.space_index = 0;
        SetBuffer.type = hlsl::ShaderVariableType::StructuredBuffer;
        auto &InstBuffer = code.properties[2];
        InstBuffer.array_size = 1;
        InstBuffer.register_index = 0;
        InstBuffer.space_index = 0;
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
        uint3(256, 1, 1),
        62,
        "load_bdls.dxil"sv,
        CacheType::Internal, true);
}
namespace detail {
static ComputeShader *LoadBCKernel(
    Device *device,
    vstd::function<vstd::string_view()> const &includeCode,
    vstd::function<vstd::string_view()> const &kernelCode,
    vstd::string_view codePath) {
    auto func = [&] {
        hlsl::CodegenResult code;
        auto incCode = includeCode();
        auto kerCode = kernelCode();
        code.result.reserve(incCode.size() + kerCode.size());
        code.result << incCode << kerCode;
        code.useBufferBindless = false;
        code.useTex2DBindless = false;
        code.useTex3DBindless = false;
        code.properties.resize(4);
        auto &globalBuffer = code.properties[0];
        globalBuffer.array_size = 1;
        globalBuffer.register_index = 0;
        globalBuffer.space_index = 0;
        globalBuffer.type = hlsl::ShaderVariableType::ConstantBuffer;

        auto &gInput = code.properties[1];
        gInput.array_size = 1;
        gInput.register_index = 0;
        gInput.space_index = 0;
        gInput.type = hlsl::ShaderVariableType::SRVTextureHeap;

        auto &gInBuff = code.properties[2];
        gInBuff.array_size = 1;
        gInBuff.register_index = 1;
        gInBuff.space_index = 0;
        gInBuff.type = hlsl::ShaderVariableType::StructuredBuffer;

        auto &gOutBuff = code.properties[3];
        gOutBuff.array_size = 1;
        gOutBuff.register_index = 0;
        gOutBuff.space_index = 0;
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
    static auto bc6Header = hlsl::CodegenUtility::ReadInternalHLSLFile("bc6_header", ctx);
    return {bc6Header.data(), bc6Header.size()};
}
static vstd::string_view Bc7Header(luisa::BinaryIO const *ctx) {
    static auto bc7Header = hlsl::CodegenUtility::ReadInternalHLSLFile("bc7_header", ctx);
    return {bc7Header.data(), bc7Header.size()};
}

static vstd::string bc7Header;
}// namespace detail

ComputeShader *BuiltinKernel::LoadBC6TryModeG10CSKernel(Device *device, luisa::BinaryIO const *ctx) {
    return detail::LoadBCKernel(
        device,
        [&] { return detail::Bc6Header(ctx); },
        [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc6_trymode_g10cs", ctx); },
        "bc6_trymodeg10"sv);
}
ComputeShader *BuiltinKernel::LoadBC6TryModeLE10CSKernel(Device *device, luisa::BinaryIO const *ctx) {
    return detail::LoadBCKernel(
        device,
        [&] { return detail::Bc6Header(ctx); },
        [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc6_trymode_le10cs", ctx); },
        "bc6_trymodele10"sv);
}
ComputeShader *BuiltinKernel::LoadBC6EncodeBlockCSKernel(Device *device, luisa::BinaryIO const *ctx) {
    return detail::LoadBCKernel(
        device,
        [&] { return detail::Bc6Header(ctx); },
        [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc6_encode_block", ctx); },
        "bc6_encodeblock"sv);
}
ComputeShader *BuiltinKernel::LoadBC7TryMode456CSKernel(Device *device, luisa::BinaryIO const *ctx) {
    return detail::LoadBCKernel(
        device,
        [&] { return detail::Bc7Header(ctx); },
        [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc7_trymode_456cs", ctx); },
        "bc7_trymode456"sv);
}
ComputeShader *BuiltinKernel::LoadBC7TryMode137CSKernel(Device *device, luisa::BinaryIO const *ctx) {
    return detail::LoadBCKernel(
        device,
        [&] { return detail::Bc7Header(ctx); },
        [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc7_trymode_137cs", ctx); },
        "bc7_trymode137"sv);
}
ComputeShader *BuiltinKernel::LoadBC7TryMode02CSKernel(Device *device, luisa::BinaryIO const *ctx) {
    return detail::LoadBCKernel(
        device,
        [&] { return detail::Bc7Header(ctx); },
        [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc7_trymode_02cs", ctx); },
        "bc7_trymode02"sv);
}
ComputeShader *BuiltinKernel::LoadBC7EncodeBlockCSKernel(Device *device, luisa::BinaryIO const *ctx) {
    return detail::LoadBCKernel(
        device,
        [&] { return detail::Bc7Header(ctx); },
        [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc7_encode_block", ctx); },
        "bc7_encodeblock"sv);
}
}// namespace lc::dx
