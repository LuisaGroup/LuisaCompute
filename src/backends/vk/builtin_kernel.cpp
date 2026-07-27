#include "builtin_kernel.h"
#include <luisa/core/stl/filesystem.h>
#include "../common/hlsl/hlsl_codegen.h"
#include "../common/indirect_dispatch_layout.h"
#include "device.h"
#include "indirect_prepare_shader.h"
namespace lc::vk {
ComputeShader *BuiltinKernel::load_indirect_prepare_kernel(Device *device) {
    auto func = [] {
        hlsl::CodegenResult code;
        code.useBufferBindless = false;
        code.useTex2DBindless = false;
        code.useTex3DBindless = false;
        code.result = indirect_prepare_hlsl_layout_definitions();
        code.result << R"(
struct IndirectPrepareConstants {
    uint command_count;
    uint source_record_offset;
    uint target_block_size_x;
    uint target_block_size_y;
    uint target_block_size_z;
    uint max_group_count_x;
    uint max_group_count_y;
    uint max_group_count_z;
    uint command_base;
    uint reserved_0;
    uint reserved_1;
    uint reserved_2;
};
[[vk::push_constant]] ConstantBuffer<IndirectPrepareConstants> pc;
StructuredBuffer<uint> source_records : register(t0);
RWStructuredBuffer<uint> commands : register(u1);

uint ceil_div(uint value, uint divisor) {
    return value / divisor + (value % divisor != 0u);
}

[numthreads(LC_INDIRECT_PREPARE_BLOCK_SIZE, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    uint remaining = pc.command_count - pc.command_base;
    if (dispatch_id.x >= remaining) { return; }
    uint command_index = pc.command_base + dispatch_id.x;
    uint3 group_count = uint3(0, 0, 0);
    uint source_index = pc.source_record_offset + command_index;
    if (command_index < source_records[0]) {
        uint logical_word = LC_INDIRECT_HEADER_WORDS +
                            source_index * LC_INDIRECT_RECORD_WORDS +
                            LC_INDIRECT_LOGICAL_WORD;
        uint3 logical_size = uint3(
            source_records[logical_word],
            source_records[logical_word + 1],
            source_records[logical_word + 2]);
        uint3 block_size = uint3(
            pc.target_block_size_x,
            pc.target_block_size_y,
            pc.target_block_size_z);
        uint group_word = LC_INDIRECT_HEADER_WORDS +
                          source_index * LC_INDIRECT_RECORD_WORDS +
                          LC_INDIRECT_GROUP_WORD;
        uint3 authored_group_count = uint3(
            source_records[group_word],
            source_records[group_word + 1],
            source_records[group_word + 2]);
        if (all(authored_group_count != uint3(0, 0, 0))) {
            group_count = uint3(
                ceil_div(logical_size.x, block_size.x),
                ceil_div(logical_size.y, block_size.y),
                ceil_div(logical_size.z, block_size.z));
        }
        if (group_count.x > pc.max_group_count_x ||
            group_count.y > pc.max_group_count_y ||
            group_count.z > pc.max_group_count_z) {
            group_count = uint3(0, 0, 0);
        }
    }
    uint command_word = command_index * LC_INDIRECT_COMMAND_WORDS;
    commands[command_word] = group_count.x;
    commands[command_word + 1] = group_count.y;
    commands[command_word + 2] = group_count.z;
}
)";
        code.properties.resize(3u);
        code.properties[0] = hlsl::Property{
            hlsl::ShaderVariableType::StructuredBuffer, 0u, 0u, 1u};
        code.properties[1] = hlsl::Property{
            hlsl::ShaderVariableType::RWStructuredBuffer, 0u, 1u, 1u};
        code.properties[2] = hlsl::Property{
            hlsl::ShaderVariableType::SamplerHeap, 1u, 0u, 16u};
        return code;
    };
    vstd::vector<SavedArgument> saved_args;
    return ComputeShader::compile_builtin_hlsl_to_spirv(
        device->binary_io(), device, std::move(saved_args), std::move(func),
        vstd::MD5{"indirect_prepare_vk_v9"sv}, {},
        uint3{IndirectDispatchLayout::prepare_block_size, 1u, 1u},
        "indirect_prepare_vk.spv"sv, SerdeType::kBuiltin, 62u, true,
        0u, luisa::nullopt, false,
        sizeof(IndirectDispatchPrepareConstants));
}

ComputeShader *BuiltinKernel::load_accel_set_kernel(Device *device) {
    auto func = [&] {
        hlsl::CodegenResult code;
        code.useBufferBindless = false;
        code.useTex2DBindless = false;
        code.useTex3DBindless = false;
        code.result << hlsl::CodegenUtility::ReadInternalHLSLFile("accel_process_vk.bytes");
        code.properties.resize(3);
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
        code.properties[2] = hlsl::Property{
            hlsl::ShaderVariableType::SamplerHeap, 1u, 0u, 16u};
        return code;
    };
    vstd::vector<SavedArgument> saved_args;
    return ComputeShader::compile_builtin_hlsl_to_spirv(
        device->binary_io(),
        device,
        std::move(saved_args),
        std::move(func),
        vstd::MD5{"accel_process_vk_v2"sv},
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
        code.properties.resize(3);
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
        code.properties[2] = hlsl::Property{
            hlsl::ShaderVariableType::SamplerHeap, 1u, 0u, 16u};
        return code;
    };
    vstd::vector<SavedArgument> saved_args;
    return ComputeShader::compile_builtin_hlsl_to_spirv(
        device->binary_io(),
        device,
        std::move(saved_args),
        std::move(func),
        vstd::MD5{"bindless_upload_vk_v2"sv},
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
