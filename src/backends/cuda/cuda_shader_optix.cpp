#include "cuda_error.h"
#include "cuda_device.h"
#include "cuda_stream.h"
#include "cuda_buffer.h"
#include "cuda_accel.h"
#include "cuda_texture.h"
#include "cuda_bindless_array.h"
#include "cuda_command_encoder.h"
#include "cuda_shader_optix.h"

namespace luisa::compute::cuda {

struct alignas(optix::SBT_RECORD_ALIGNMENT) OptiXSBTRecord {
    std::byte data[optix::SBT_RECORD_HEADER_SIZE];
};

/// Retrieves direct and continuation stack sizes for each program in the program group and accumulates the upper bounds
/// in the correponding output variables based on the semantic type of the program. Before the first invocation of this
/// function with a given instance of #OptixStackSizes, the members of that instance should be set to 0.
inline void accumulate_stack_sizes(optix::StackSizes &sizes, optix::ProgramGroup group, optix::Pipeline pipeline) noexcept {
    optix::StackSizes local{};
    LUISA_CHECK_OPTIX(optix::api().programGroupGetStackSize(group, &local, pipeline));
    LUISA_VERBOSE("OptiX program group stack sizes: "
                  "CSS_RG = {}, CSS_MS = {}, CSS_CH = {}, CSS_AH = {}, "
                  "CSS_IS = {}, CSS_CC = {}, DSS_DC = {}.",
                  local.cssRG, local.cssMS, local.cssCH, local.cssAH,
                  local.cssIS, local.cssCC, local.dssDC);
    sizes.cssRG = std::max(sizes.cssRG, local.cssRG);
    sizes.cssMS = std::max(sizes.cssMS, local.cssMS);
    sizes.cssCH = std::max(sizes.cssCH, local.cssCH);
    sizes.cssAH = std::max(sizes.cssAH, local.cssAH);
    sizes.cssIS = std::max(sizes.cssIS, local.cssIS);
    sizes.cssCC = std::max(sizes.cssCC, local.cssCC);
    sizes.dssDC = std::max(sizes.dssDC, local.dssDC);
    LUISA_VERBOSE("Accumulated OptiX stack sizes: "
                  "CSS_RG = {}, CSS_MS = {}, CSS_CH = {}, CSS_AH = {}, "
                  "CSS_IS = {}, CSS_CC = {}, DSS_DC = {}.",
                  sizes.cssRG, sizes.cssMS, sizes.cssCH, sizes.cssAH,
                  sizes.cssIS, sizes.cssCC, sizes.dssDC);
}

[[nodiscard]] inline uint compute_continuation_stack_size(optix::StackSizes ss) noexcept {
    auto size = ss.cssRG + std::max(std::max(ss.cssCH, ss.cssMS), ss.cssIS + ss.cssAH);
    LUISA_VERBOSE("Computed OptiX continuation stack size: {}.", size);
    return size;
}

CUDAShaderOptiX::CUDAShaderOptiX(optix::DeviceContext optix_ctx,
                                 const char *ptx, size_t ptx_size, const char *entry,
                                 const CUDAShaderMetadata &metadata,
                                 luisa::vector<ShaderDispatchCommand::Argument> bound_arguments) noexcept
    : CUDAShader{metadata.argument_usages},
      _bound_arguments{std::move(bound_arguments)} {

    // compute argument buffer size
    _argument_buffer_size = 0u;
    for (auto &&arg : metadata.argument_types) {
        auto type = Type::from(arg);
        switch (type->tag()) {
            case Type::Tag::BUFFER:
                _argument_buffer_size += sizeof(CUDABuffer::Binding);
                break;
            case Type::Tag::TEXTURE:
                _argument_buffer_size += sizeof(CUDATexture::Binding);
                break;
            case Type::Tag::BINDLESS_ARRAY:
                _argument_buffer_size += sizeof(CUDABindlessArray::Binding);
                break;
            case Type::Tag::ACCEL:
                _argument_buffer_size += sizeof(CUDAAccel::Binding);
                break;
            case Type::Tag::CUSTOM:
                LUISA_ERROR_WITH_LOCATION(
                    "Invalid custom type '{}' for OptiX shader argument.",
                    type->description());
            default:
                _argument_buffer_size += type->size();
                break;
        }
        _argument_buffer_size = luisa::align(_argument_buffer_size, 16u);
    }
    // for dispatch size and kernel id
    _argument_buffer_size += 16u;
    LUISA_VERBOSE_WITH_LOCATION(
        "Argument buffer size for {}: {}.",
        entry, _argument_buffer_size);

    // create module
    static constexpr std::array ray_trace_payload_semantics{
        optix::PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | optix::PAYLOAD_SEMANTICS_CH_READ,
    };
    static constexpr std::array ray_query_payload_semantics{
        optix::PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | optix::PAYLOAD_SEMANTICS_IS_READ | optix::PAYLOAD_SEMANTICS_AH_READ,
        optix::PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | optix::PAYLOAD_SEMANTICS_IS_READ | optix::PAYLOAD_SEMANTICS_AH_READ,
    };

    std::array<optix::PayloadType, 2u> payload_types{};
    payload_types[0].numPayloadValues = ray_trace_payload_semantics.size();
    payload_types[0].payloadSemantics = ray_trace_payload_semantics.data();
    payload_types[1].numPayloadValues = ray_query_payload_semantics.size();
    payload_types[1].payloadSemantics = ray_query_payload_semantics.data();

    optix::ModuleCompileOptions module_compile_options{};
    module_compile_options.maxRegisterCount = optix::COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.debugLevel = metadata.enable_debug ? optix::COMPILE_DEBUG_LEVEL_MINIMAL :
                                                                optix::COMPILE_DEBUG_LEVEL_NONE;
    module_compile_options.optLevel = metadata.enable_debug ?
                                          optix::COMPILE_OPTIMIZATION_LEVEL_3 :
                                          optix::COMPILE_OPTIMIZATION_LEVEL_3;
    module_compile_options.numPayloadTypes = payload_types.size();
    module_compile_options.payloadTypes = payload_types.data();

    optix::PipelineCompileOptions pipeline_compile_options{};
    pipeline_compile_options.exceptionFlags = optix::EXCEPTION_FLAG_NONE;
    pipeline_compile_options.traversableGraphFlags = optix::TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_compile_options.numPayloadValues = 0u;
    auto primitive_flags = metadata.requires_ray_query ? (optix::PRIMITIVE_TYPE_FLAGS_CUSTOM |
                                                          optix::PRIMITIVE_TYPE_FLAGS_TRIANGLE) :
                                                         optix::PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    pipeline_compile_options.usesPrimitiveTypeFlags = primitive_flags;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    char log[2048];// For error reporting from OptiX creation functions
    size_t log_size;
    LUISA_CHECK_OPTIX_WITH_LOG(
        log, log_size,
        optix::api().moduleCreate(
            optix_ctx, &module_compile_options,
            &pipeline_compile_options, ptx, ptx_size,
            log, &log_size, &_module));

    // create program groups
    luisa::fixed_vector<optix::ProgramGroup, 2u> program_groups;

    // raygen
    optix::ProgramGroupOptions program_group_options_rg{};
    program_group_options_rg.payloadType = &payload_types[metadata.requires_ray_query ? 1 : 0];
    optix::ProgramGroupDesc program_group_desc_rg{};
    program_group_desc_rg.kind = optix::PROGRAM_GROUP_KIND_RAYGEN;
    program_group_desc_rg.raygen.module = _module;
    program_group_desc_rg.raygen.entryFunctionName = entry;
    LUISA_CHECK_OPTIX_WITH_LOG(
        log, log_size,
        optix::api().programGroupCreate(
            optix_ctx,
            &program_group_desc_rg, 1u,
            &program_group_options_rg,
            log, &log_size, &_program_group_rg));
    program_groups.emplace_back(_program_group_rg);

    if (metadata.requires_ray_query) {
        optix::ProgramGroupOptions program_group_options_ch_query{};
        program_group_options_ch_query.payloadType = &payload_types[1];
        optix::ProgramGroupDesc program_group_desc_ch_query{};
        program_group_desc_ch_query.kind = optix::PROGRAM_GROUP_KIND_HITGROUP;
        program_group_desc_ch_query.hitgroup.moduleAH = _module;
        program_group_desc_ch_query.hitgroup.entryFunctionNameAH = "__anyhit__ray_query";
        program_group_desc_ch_query.hitgroup.moduleIS = _module;
        program_group_desc_ch_query.hitgroup.entryFunctionNameIS = "__intersection__ray_query";
        LUISA_CHECK_OPTIX_WITH_LOG(
            log, log_size,
            optix::api().programGroupCreate(
                optix_ctx,
                &program_group_desc_ch_query, 1u,
                &program_group_options_ch_query,
                log, &log_size, &_program_group_ray_query));
        program_groups.emplace_back(_program_group_ray_query);
    }

    // create pipeline
    optix::PipelineLinkOptions pipeline_link_options{};
    pipeline_link_options.maxTraceDepth = 1u;
    LUISA_CHECK_OPTIX_WITH_LOG(
        log, log_size,
        optix::api().pipelineCreate(
            optix_ctx, &pipeline_compile_options, &pipeline_link_options,
            program_groups.data(), program_groups.size(), log, &log_size, &_pipeline));

    // compute stack sizes
    optix::StackSizes stack_sizes{};
    for (auto pg : program_groups) { accumulate_stack_sizes(stack_sizes, pg, _pipeline); }
    auto continuation_stack_size = compute_continuation_stack_size(stack_sizes);
    LUISA_CHECK_OPTIX(optix::api().pipelineSetStackSize(
        _pipeline, 0u, 0u, continuation_stack_size, 2u));

    // create shader binding table
    std::array<OptiXSBTRecord, 2u> sbt_records{};
    if (_program_group_rg) { LUISA_CHECK_OPTIX(optix::api().sbtRecordPackHeader(_program_group_rg, &sbt_records[0])); }
    if (_program_group_ray_query) { LUISA_CHECK_OPTIX(optix::api().sbtRecordPackHeader(_program_group_ray_query, &sbt_records[1])); }
    LUISA_CHECK_CUDA(cuMemAlloc(&_sbt_buffer, sbt_records.size() * sizeof(OptiXSBTRecord)));
    LUISA_CHECK_CUDA(cuMemcpyHtoD(_sbt_buffer, sbt_records.data(), sbt_records.size() * sizeof(OptiXSBTRecord)));
}

CUDAShaderOptiX::~CUDAShaderOptiX() noexcept {
    LUISA_CHECK_CUDA(cuMemFree(_sbt_buffer));
    LUISA_CHECK_OPTIX(optix::api().pipelineDestroy(_pipeline));
    if (_program_group_rg) { LUISA_CHECK_OPTIX(optix::api().programGroupDestroy(_program_group_rg)); }
    if (_program_group_ray_query) { LUISA_CHECK_OPTIX(optix::api().programGroupDestroy(_program_group_ray_query)); }
    LUISA_CHECK_OPTIX(optix::api().moduleDestroy(_module));
}

void CUDAShaderOptiX::_launch(CUDACommandEncoder &encoder, ShaderDispatchCommand *command) const noexcept {
    luisa::vector<std::byte> indirect_dispatches_host;
    const IndirectParameters *indirect_dispatches_device = nullptr;
    if (command->is_indirect()) {
        // read back dispatch buffer
        auto indirect = command->indirect_dispatch();
        auto buffer = reinterpret_cast<CUDAIndirectDispatchBuffer *>(indirect.handle);
        indirect_dispatches_host.resize(buffer->size_bytes());
        encoder.with_download_pool_no_fallback(buffer->size_bytes(), [&](auto temp) noexcept {
            auto stream = encoder.stream()->handle();
            if (temp) {
                LUISA_CHECK_CUDA(cuMemcpyDtoHAsync(temp->address(), buffer->handle(),
                                                   buffer->size_bytes(), stream));
                LUISA_CHECK_CUDA(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(indirect_dispatches_host.data()),
                                               reinterpret_cast<CUdeviceptr>(temp->address()),
                                               buffer->size_bytes(), stream));
            } else {
                LUISA_CHECK_CUDA(cuMemcpyDtoHAsync(indirect_dispatches_host.data(), buffer->handle(),
                                                   buffer->size_bytes(), stream));
            }
        });
        indirect_dispatches_device = reinterpret_cast<const IndirectParameters *>(buffer->handle());
    }

    // encode arguments
    encoder.with_upload_buffer(_argument_buffer_size, [&](CUDAHostBufferPool::View *argument_buffer) noexcept {
        auto argument_buffer_offset = static_cast<size_t>(0u);
        auto allocate_argument = [&](size_t bytes) noexcept {
            static constexpr auto alignment = 16u;
            auto offset = (argument_buffer_offset + alignment - 1u) / alignment * alignment;
            LUISA_ASSERT(offset + bytes <= _argument_buffer_size,
                         "Too many arguments in ShaderDispatchCommand");
            argument_buffer_offset = offset + bytes;
            return argument_buffer->address() + offset;
        };

        auto encode_argument = [&allocate_argument, command](const auto &arg) noexcept {
            using Tag = ShaderDispatchCommand::Argument::Tag;
            switch (arg.tag) {
                case Tag::BUFFER: {
                    auto buffer = reinterpret_cast<const CUDABuffer *>(arg.buffer.handle);
                    auto binding = buffer->binding(arg.buffer.offset, arg.buffer.size);
                    auto ptr = allocate_argument(sizeof(binding));
                    std::memcpy(ptr, &binding, sizeof(binding));
                    break;
                }
                case Tag::TEXTURE: {
                    auto texture = reinterpret_cast<const CUDATexture *>(arg.texture.handle);
                    auto binding = texture->binding(arg.texture.level);
                    auto ptr = allocate_argument(sizeof(binding));
                    std::memcpy(ptr, &binding, sizeof(binding));
                    break;
                }
                case Tag::UNIFORM: {
                    auto uniform = command->uniform(arg.uniform);
                    auto ptr = allocate_argument(uniform.size_bytes());
                    std::memcpy(ptr, uniform.data(), uniform.size_bytes());
                    break;
                }
                case Tag::BINDLESS_ARRAY: {
                    auto array = reinterpret_cast<const CUDABindlessArray *>(arg.bindless_array.handle);
                    auto binding = array->binding();
                    auto ptr = allocate_argument(sizeof(binding));
                    std::memcpy(ptr, &binding, sizeof(binding));
                    break;
                }
                case Tag::ACCEL: {
                    auto accel = reinterpret_cast<const CUDAAccel *>(arg.accel.handle);
                    auto binding = accel->binding();
                    auto ptr = allocate_argument(sizeof(binding));
                    std::memcpy(ptr, &binding, sizeof(binding));
                    break;
                }
            }
        };
        // TODO: optimize this
        for (auto &&arg : _bound_arguments) { encode_argument(arg); }
        for (auto &&arg : command->arguments()) { encode_argument(arg); }
        auto s = command->dispatch_size();
        auto ds_and_kid = make_uint4(s, 0u);
        auto ptr = allocate_argument(sizeof(ds_and_kid));
        std::memcpy(ptr, &ds_and_kid, sizeof(ds_and_kid));
        auto cuda_stream = encoder.stream()->handle();
        if (argument_buffer->is_pooled()) [[likely]] {
            // for (direct) dispatches, if the argument buffer
            // is pooled, we can use the device pointer directly
            auto device_argument_buffer = 0ull;
            LUISA_CHECK_CUDA(cuMemHostGetDevicePointer(
                &device_argument_buffer, argument_buffer->address(), 0u));
            if (!indirect_dispatches_host.empty()) {
                auto indirect = command->indirect_dispatch();
                _do_launch_indirect(cuda_stream, device_argument_buffer,
                                    indirect.offset, indirect.max_dispatch_size,
                                    indirect_dispatches_device,
                                    reinterpret_cast<IndirectParameters *>(indirect_dispatches_host.data()));
            } else {
                _do_launch(cuda_stream, device_argument_buffer, s);
            }
        } else {// otherwise, we need to copy the argument buffer to the device
            auto device_argument_buffer = 0ull;
            LUISA_CHECK_CUDA(cuMemAllocAsync(
                &device_argument_buffer, _argument_buffer_size, cuda_stream));
            LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(
                device_argument_buffer, argument_buffer->address(),
                _argument_buffer_size, cuda_stream));
            if (!indirect_dispatches_host.empty()) {
                auto indirect = command->indirect_dispatch();
                _do_launch_indirect(cuda_stream, device_argument_buffer,
                                    indirect.offset, indirect.max_dispatch_size,
                                    indirect_dispatches_device,
                                    reinterpret_cast<IndirectParameters *>(indirect_dispatches_host.data()));
            } else {
                _do_launch(cuda_stream, device_argument_buffer, s);
            }
            LUISA_CHECK_CUDA(cuMemFreeAsync(device_argument_buffer, cuda_stream));
        }
    });
}

inline optix::ShaderBindingTable CUDAShaderOptiX::_make_sbt() const noexcept {
    optix::ShaderBindingTable sbt{};
    sbt.raygenRecord = _sbt_buffer;
    sbt.hitgroupRecordBase = _sbt_buffer + sizeof(OptiXSBTRecord);
    sbt.hitgroupRecordCount = 1u;
    sbt.hitgroupRecordStrideInBytes = sizeof(OptiXSBTRecord);
    sbt.missRecordBase = _sbt_buffer + sizeof(OptiXSBTRecord) * 2u;
    sbt.missRecordCount = 1u;// FIXME: we are not using miss shaders but it's mandatory to set this to 1
    sbt.missRecordStrideInBytes = sizeof(OptiXSBTRecord);
    return sbt;
}

void CUDAShaderOptiX::_do_launch(CUstream stream, CUdeviceptr argument_buffer, uint3 dispatch_size) const noexcept {
    auto sbt = _make_sbt();
    LUISA_CHECK_OPTIX(optix::api().launch(
        _pipeline, stream, argument_buffer, _argument_buffer_size,
        &sbt, dispatch_size.x, dispatch_size.y, dispatch_size.z));
}

void CUDAShaderOptiX::_do_launch_indirect(CUstream stream, CUdeviceptr argument_buffer,
                                          size_t dispatch_offset, size_t dispatch_count,
                                          const CUDAShaderOptiX::IndirectParameters *indirect_buffer_device,
                                          const CUDAShaderOptiX::IndirectParameters *indirect_params_readback) const noexcept {
    auto sbt = _make_sbt();
    auto p = argument_buffer + _argument_buffer_size - sizeof(uint4);
    LUISA_CHECK_CUDA(cuStreamSynchronize(stream));
    auto n = std::min<size_t>(indirect_params_readback->header.size, dispatch_count);
    for (auto i = 0u; i < n; i++) {
        auto dispatch = indirect_params_readback->dispatches[i + dispatch_offset].dispatch_size_and_kernel_id;
        auto dispatch_size = dispatch.xyz();
        auto d = reinterpret_cast<CUdeviceptr>(&indirect_buffer_device->dispatches[i + dispatch_offset]);
        LUISA_CHECK_CUDA(cuMemcpyAsync(p, d, sizeof(uint4), stream));
        LUISA_CHECK_OPTIX(optix::api().launch(
            _pipeline, stream, argument_buffer, _argument_buffer_size,
            &sbt, dispatch_size.x, dispatch_size.y, dispatch_size.z));
    }
}

}// namespace luisa::compute::cuda
