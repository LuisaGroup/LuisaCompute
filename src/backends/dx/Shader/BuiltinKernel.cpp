#include <Shader/BuiltinKernel.h>
#include <luisa/core/stl/filesystem.h>
#include "../common/hlsl/hlsl_codegen.h"
namespace lc::dx {
ComputeShader *BuiltinKernel::load_accel_set_kernel(Device *device) {
    auto func = [&] {
        hlsl::CodegenResult code;
        code.useBufferBindless = false;
        code.useTex2DBindless = false;
        code.useTex3DBindless = false;
        code.result << hlsl::CodegenUtility::ReadInternalHLSLFile("accel_process.bytes");
        code.properties.resize(3);
        auto &global = code.properties[0];
        global.array_size = 1;
        global.register_index = 0;
        global.space_index = 0;
        global.type = hlsl::ShaderVariableType::ConstantBuffer;
        auto &set_buffer = code.properties[1];
        set_buffer.array_size = 1;
        set_buffer.register_index = 0;
        set_buffer.space_index = 0;
        set_buffer.type = hlsl::ShaderVariableType::StructuredBuffer;
        auto &inst_buffer = code.properties[2];
        inst_buffer.array_size = 1;
        inst_buffer.register_index = 0;
        inst_buffer.space_index = 0;
        inst_buffer.type = hlsl::ShaderVariableType::RWStructuredBuffer;
        return code;
    };
    return ComputeShader::compile_compute(
        device->file_io,
        device->profiler,
        device,
        {},
        func,
        {},
        {},
        uint3(256, 1, 1),
        62,
        "set_accel4.dxil"sv,
        CacheType::Internal, true, false,
        0);
}
ComputeShader *BuiltinKernel::load_bindless_set_kernel(Device *device) {
    auto func = [&] {
        hlsl::CodegenResult code;
        code.useBufferBindless = false;
        code.useTex2DBindless = false;
        code.useTex3DBindless = false;
        code.result << hlsl::CodegenUtility::ReadInternalHLSLFile("bindless_upload.bytes");
        code.properties.resize(3);
        auto &global = code.properties[0];
        global.array_size = 1;
        global.register_index = 0;
        global.space_index = 0;
        global.type = hlsl::ShaderVariableType::ConstantBuffer;
        auto &set_buffer = code.properties[1];
        set_buffer.array_size = 1;
        set_buffer.register_index = 0;
        set_buffer.space_index = 0;
        set_buffer.type = hlsl::ShaderVariableType::StructuredBuffer;
        auto &inst_buffer = code.properties[2];
        inst_buffer.array_size = 1;
        inst_buffer.register_index = 0;
        inst_buffer.space_index = 0;
        inst_buffer.type = hlsl::ShaderVariableType::RWStructuredBuffer;
        return code;
    };
    return ComputeShader::compile_compute(
        device->file_io,
        device->profiler,
        device,
        {},
        func,
        {},
        {},
        uint3(256, 1, 1),
        62,
        "load_bdls.dxil"sv,
        CacheType::Internal, true, false,
        0);
}
namespace detail {
static ComputeShader *_load_bc_kernel(
    Device *device,
    vstd::function<vstd::string_view()> const &include_code,
    vstd::function<vstd::string_view()> const &kernel_code,
    vstd::string_view code_path) {
    auto func = [&] {
        hlsl::CodegenResult code;
        auto inc_code = include_code();
        auto ker_code = kernel_code();
        code.result.reserve(inc_code.size() + ker_code.size());
        code.result << inc_code << ker_code;
        code.useBufferBindless = false;
        code.useTex2DBindless = false;
        code.useTex3DBindless = false;
        code.properties.resize(4);
        auto &global_buffer = code.properties[0];
        global_buffer.array_size = 1;
        global_buffer.register_index = 0;
        global_buffer.space_index = 0;
        global_buffer.type = hlsl::ShaderVariableType::ConstantBuffer;

        auto &g_input = code.properties[1];
        g_input.array_size = 1;
        g_input.register_index = 0;
        g_input.space_index = 0;
        g_input.type = hlsl::ShaderVariableType::SRVTextureHeap;

        auto &g_in_buff = code.properties[2];
        g_in_buff.array_size = 1;
        g_in_buff.register_index = 1;
        g_in_buff.space_index = 0;
        g_in_buff.type = hlsl::ShaderVariableType::StructuredBuffer;

        auto &g_out_buff = code.properties[3];
        g_out_buff.array_size = 1;
        g_out_buff.register_index = 0;
        g_out_buff.space_index = 0;
        g_out_buff.type = hlsl::ShaderVariableType::RWStructuredBuffer;
        return code;
    };
    vstd::string file_name;
    vstd::string_view ext_name = ".dxil"sv;
    file_name.reserve(code_path.size() + ext_name.size());
    file_name << code_path << ext_name;
    return ComputeShader::compile_compute(
        device->file_io,
        device->profiler,
        device,
        {},
        func,
        {},
        {},
        uint3(1, 1, 1),
        62,
        file_name,
        CacheType::Internal, true, false,
        0);
}
static vstd::string_view _bc6_header() {
    static auto bc6_header = hlsl::CodegenUtility::ReadInternalHLSLFile("bc6_header");
    return {bc6_header.data(), bc6_header.size()};
}
static vstd::string_view _bc7_header() {
    static auto bc7_header = hlsl::CodegenUtility::ReadInternalHLSLFile("bc7_header");
    return {bc7_header.data(), bc7_header.size()};
}

static vstd::string _bc7_header_str;
}// namespace detail

ComputeShader *BuiltinKernel::load_bc6_try_mode_g10cs_kernel(Device *device) {
    return detail::_load_bc_kernel(
        device,
        [&] { return detail::_bc6_header(); },
        [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc6_trymode_g10cs"); },
        "bc6_trymodeg10"sv);
}
ComputeShader *BuiltinKernel::load_bc6_try_mode_le10cs_kernel(Device *device) {
    return detail::_load_bc_kernel(
        device,
        [&] { return detail::_bc6_header(); },
        [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc6_trymode_le10cs"); },
        "bc6_trymodele10"sv);
}
ComputeShader *BuiltinKernel::load_bc6_encode_block_cs_kernel(Device *device) {
    return detail::_load_bc_kernel(
        device,
        [&] { return detail::_bc6_header(); },
        [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc6_encode_block"); },
        "bc6_encodeblock"sv);
}
ComputeShader *BuiltinKernel::load_bc7_try_mode_456cs_kernel(Device *device) {
    return detail::_load_bc_kernel(
        device,
        [&] { return detail::_bc7_header(); },
        [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc7_trymode_456cs"); },
        "bc7_trymode456"sv);
}
ComputeShader *BuiltinKernel::load_bc7_try_mode_137cs_kernel(Device *device) {
    return detail::_load_bc_kernel(
        device,
        [&] { return detail::_bc7_header(); },
        [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc7_trymode_137cs"); },
        "bc7_trymode137"sv);
}
ComputeShader *BuiltinKernel::load_bc7_try_mode_02cs_kernel(Device *device) {
    return detail::_load_bc_kernel(
        device,
        [&] { return detail::_bc7_header(); },
        [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc7_trymode_02cs"); },
        "bc7_trymode02"sv);
}
ComputeShader *BuiltinKernel::load_bc7_encode_block_cs_kernel(Device *device) {
    return detail::_load_bc_kernel(
        device,
        [&] { return detail::_bc7_header(); },
        [&] { return hlsl::CodegenUtility::ReadInternalHLSLFile("bc7_encode_block"); },
        "bc7_encodeblock"sv);
}
}// namespace lc::dx
