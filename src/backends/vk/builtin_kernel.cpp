#include "builtin_kernel.h"
#include <luisa/core/stl/filesystem.h>
#include "../common/hlsl/hlsl_codegen.h"
#include "device.h"
namespace lc::vk {
ComputeShader *BuiltinKernel::load_accel_set_kernel(Device *device) {
    auto func = [&] {
        hlsl::CodegenResult code;
        code.useBufferBindless = false;
        code.useTex2DBindless = false;
        code.useTex3DBindless = false;
        code.result << hlsl::CodegenUtility::ReadInternalHLSLFile("accel_process_vk.bytes");
        code.properties.resize(2);
        auto &set_buffer = code.properties[0];
        set_buffer.array_size = 1;
        set_buffer.register_index = 0;
        set_buffer.space_index = 0;
        set_buffer.type = hlsl::ShaderVariableType::StructuredBuffer;
        auto &inst_buffer = code.properties[1];
        inst_buffer.array_size = 1;
        inst_buffer.register_index = 1;
        inst_buffer.space_index = 0;
        inst_buffer.type = hlsl::ShaderVariableType::RWStructuredBuffer;
        return code;
    };
    vstd::vector<SavedArgument> saved_args;
    return ComputeShader::compile_builtin_hlsl_to_spirv(
        device->binary_io(),
        device,
        std::move(saved_args),
        std::move(func),
        vstd::MD5{"accel_process_vk"sv},
        {},
        uint3(256, 1, 1),
        "accel_process_vk.dxil"sv,
        SerdeType::kBuiltin,
        62, true);
}
ComputeShader *BuiltinKernel::load_bindless_set_kernel(Device *device) {
    auto func = [&] {
        hlsl::CodegenResult code;
        code.useBufferBindless = false;
        code.useTex2DBindless = false;
        code.useTex3DBindless = false;
        code.result << hlsl::CodegenUtility::ReadInternalHLSLFile("bindless_upload_vk.bytes");
        code.properties.resize(2);
        auto &set_buffer = code.properties[0];
        set_buffer.array_size = 1;
        set_buffer.register_index = 0;
        set_buffer.space_index = 0;
        set_buffer.type = hlsl::ShaderVariableType::StructuredBuffer;
        auto &inst_buffer = code.properties[1];
        inst_buffer.array_size = 1;
        inst_buffer.register_index = 1;
        inst_buffer.space_index = 0;
        inst_buffer.type = hlsl::ShaderVariableType::RWStructuredBuffer;
        return code;
    };
    vstd::vector<SavedArgument> saved_args;
    return ComputeShader::compile_builtin_hlsl_to_spirv(
        device->binary_io(),
        device,
        std::move(saved_args),
        std::move(func),
        vstd::MD5{"bindless_upload_vk"sv},
        {},
        uint3(256, 1, 1),
        "load_bdls_vk.dxil"sv,
        SerdeType::kBuiltin,
        62, true);
}
// namespace detail {
// static ComputeShader *LoadBCKernel(
//     Device *device,
//     vstd::function<vstd::string_view()> const &includeCode,
//     vstd::function<vstd::string_view()> const &kernelCode,
//     vstd::string_view codePath) {
//     auto func = [&] {
//         hlsl::CodegenResult code;
//         auto incCode = includeCode();
//         auto kerCode = kernelCode();
//         code.result.reserve(incCode.size() + kerCode.size());
//         code.result << incCode << kerCode;
//         code.useBufferBindless = false;
//         code.useTex2DBindless = false;
//         code.useTex3DBindless = false;
//         code.properties.resize(4);
//         auto &globalBuffer = code.properties[0];
//         globalBuffer.array_size = 1;
//         globalBuffer.register_index = 0;
//         globalBuffer.space_index = 0;
//         globalBuffer.type = hlsl::ShaderVariableType::ConstantBuffer;

//         auto &gInput = code.properties[1];
//         gInput.array_size = 1;
//         gInput.register_index = 0;
//         gInput.space_index = 0;
//         gInput.type = hlsl::ShaderVariableType::SRVTextureHeap;

//         auto &gInBuff = code.properties[2];
//         gInBuff.array_size = 1;
//         gInBuff.register_index = 1;
//         gInBuff.space_index = 0;
//         gInBuff.type = hlsl::ShaderVariableType::StructuredBuffer;

//         auto &gOutBuff = code.properties[3];
//         gOutBuff.array_size = 1;
//         gOutBuff.register_index = 0;
//         gOutBuff.space_index = 0;
//         gOutBuff.type = hlsl::ShaderVariableType::RWStructuredBuffer;
//         return code;
//     };
//     vstd::string fileName;
//     vstd::string_view extName = "2.dxil"sv;
//     fileName.reserve(codePath.size() + extName.size());
//     fileName << codePath << extName;
//     return ComputeShader::CompileCompute(
//         device->fileIo,
//         device->profiler,
//         device,
//         {},
//         func,
//         {},
//         {},
//         uint3(1, 1, 1),
//         62,
//         fileName,
//         CacheType::Internal, true, false);
// }
// static vstd::string_view Bc6Header() {
//     static auto bc6Header = hlsl::CodegenUtility::ReadInternalHLSLFile("bc6_header");
//     return {bc6Header.data(), bc6Header.size()};
// }
// static vstd::string_view Bc7Header() {
//     static auto bc7Header = hlsl::CodegenUtility::ReadInternalHLSLFile("bc7_header");
//     return {bc7Header.data(), bc7Header.size()};
// }

// static vstd::string bc7Header;
// }// namespace detail

// ComputeShader *BuiltinKernel::LoadBC6TryModeG10CSKernel(Device *device) {
//     return detail::LoadBCKernel(
//         device,
//         [&] { return detail::Bc6Header(); },
//         [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc6_trymode_g10cs"); },
//         "bc6_trymodeg10"sv);
// }
// ComputeShader *BuiltinKernel::LoadBC6TryModeLE10CSKernel(Device *device) {
//     return detail::LoadBCKernel(
//         device,
//         [&] { return detail::Bc6Header(); },
//         [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc6_trymode_le10cs"); },
//         "bc6_trymodele10"sv);
// }
// ComputeShader *BuiltinKernel::LoadBC6EncodeBlockCSKernel(Device *device) {
//     return detail::LoadBCKernel(
//         device,
//         [&] { return detail::Bc6Header(); },
//         [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc6_encode_block"); },
//         "bc6_encodeblock"sv);
// }
// ComputeShader *BuiltinKernel::LoadBC7TryMode456CSKernel(Device *device) {
//     return detail::LoadBCKernel(
//         device,
//         [&] { return detail::Bc7Header(); },
//         [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc7_trymode_456cs"); },
//         "bc7_trymode456"sv);
// }
// ComputeShader *BuiltinKernel::LoadBC7TryMode137CSKernel(Device *device) {
//     return detail::LoadBCKernel(
//         device,
//         [&] { return detail::Bc7Header(); },
//         [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc7_trymode_137cs"); },
//         "bc7_trymode137"sv);
// }
// ComputeShader *BuiltinKernel::LoadBC7TryMode02CSKernel(Device *device) {
//     return detail::LoadBCKernel(
//         device,
//         [&] { return detail::Bc7Header(); },
//         [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc7_trymode_02cs"); },
//         "bc7_trymode02"sv);
// }
// ComputeShader *BuiltinKernel::LoadBC7EncodeBlockCSKernel(Device *device) {
//     return detail::LoadBCKernel(
//         device,
//         [&] { return detail::Bc7Header(); },
//         [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc7_encode_block"); },
//         "bc7_encodeblock"sv);
// }
}// namespace lc::vk
