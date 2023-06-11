//
// Created by Mike on 3/18/2023.
//

#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_device.h>
#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_buffer.h>
#include <backends/cuda/cuda_accel.h>
#include <backends/cuda/cuda_texture.h>
#include <backends/cuda/cuda_bindless_array.h>
#include <backends/cuda/cuda_command_encoder.h>
#include <backends/cuda/cuda_shader_optix.h>

namespace luisa::compute::cuda {

struct alignas(optix::SBT_RECORD_ALIGNMENT) OptiXSBTRecord {
    std::byte data[optix::SBT_RECORD_HEADER_SIZE];
};

/// Retrieves direct and continuation stack sizes for each program in the program group and accumulates the upper bounds
/// in the correponding output variables based on the semantic type of the program. Before the first invocation of this
/// function with a given instance of #OptixStackSizes, the members of that instance should be set to 0.
inline void accumulate_stack_sizes(optix::StackSizes &sizes, optix::ProgramGroup group) noexcept {
    optix::StackSizes local{};
    LUISA_CHECK_OPTIX(optix::api().programGroupGetStackSize(group, &local));
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
    LUISA_INFO("Computed OptiX continuation stack size: {}.", size);
    return size;
}

CUDAShaderOptiX::CUDAShaderOptiX(optix::DeviceContext optix_ctx,
                                 const char *ptx, size_t ptx_size,
                                 const char *entry, bool enable_debug,
                                 luisa::vector<Usage> argument_usages,
                                 luisa::vector<ShaderDispatchCommand::Argument> bound_arguments) noexcept
    : CUDAShader{std::move(argument_usages)},
      _bound_arguments{std::move(bound_arguments)} {

    // create argument buffer
    static constexpr auto pattern = "params[";
    auto ptr = strstr(ptx, pattern);
    if (ptr == nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Cannot find global symbol 'params' in PTX for {}.",
            entry);
    }
    ptr += std::string_view{pattern}.size();
    char *end = nullptr;
    _argument_buffer_size = strtoull(ptr, &end, 10);
    if (_argument_buffer_size == 0u) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to parse argument buffer size for {}.",
            entry);
    }
    LUISA_VERBOSE_WITH_LOCATION(
        "Argument buffer size for {}: {}.",
        entry, _argument_buffer_size);

    // reflect ray tracing calls
    auto detect_rtx_entry = [ptx = luisa::string_view{ptx, ptx_size}](luisa::string_view name) noexcept {
        auto pattern = luisa::format(".visible .entry __miss__{}()", name);
        return ptx.find(pattern) != luisa::string_view::npos;
    };
    auto uses_trace_closest = detect_rtx_entry("trace_closest");
    auto uses_trace_any = detect_rtx_entry("trace_any");
    auto uses_ray_query = detect_rtx_entry("ray_query");

    // create module
    static constexpr std::array trace_closest_payload_semantics{
        optix::PAYLOAD_SEMANTICS_TRACE_CALLER_READ | optix::PAYLOAD_SEMANTICS_CH_WRITE | optix::PAYLOAD_SEMANTICS_MS_WRITE,
        optix::PAYLOAD_SEMANTICS_TRACE_CALLER_READ | optix::PAYLOAD_SEMANTICS_CH_WRITE,
        optix::PAYLOAD_SEMANTICS_TRACE_CALLER_READ | optix::PAYLOAD_SEMANTICS_CH_WRITE,
        optix::PAYLOAD_SEMANTICS_TRACE_CALLER_READ | optix::PAYLOAD_SEMANTICS_CH_WRITE,
        optix::PAYLOAD_SEMANTICS_TRACE_CALLER_READ | optix::PAYLOAD_SEMANTICS_CH_WRITE};

    static constexpr std::array trace_any_payload_semantics{
        optix::PAYLOAD_SEMANTICS_TRACE_CALLER_READ | optix::PAYLOAD_SEMANTICS_MS_WRITE};

    static constexpr std::array ray_query_payload_semantics{
        optix::PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | optix::PAYLOAD_SEMANTICS_IS_READ | optix::PAYLOAD_SEMANTICS_AH_READ | optix::PAYLOAD_SEMANTICS_CH_WRITE | optix::PAYLOAD_SEMANTICS_MS_WRITE,
        optix::PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | optix::PAYLOAD_SEMANTICS_IS_READ | optix::PAYLOAD_SEMANTICS_AH_READ | optix::PAYLOAD_SEMANTICS_CH_WRITE,
        optix::PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | optix::PAYLOAD_SEMANTICS_IS_READ | optix::PAYLOAD_SEMANTICS_AH_READ | optix::PAYLOAD_SEMANTICS_CH_WRITE,
        optix::PAYLOAD_SEMANTICS_TRACE_CALLER_READ | optix::PAYLOAD_SEMANTICS_CH_WRITE,
        optix::PAYLOAD_SEMANTICS_TRACE_CALLER_READ | optix::PAYLOAD_SEMANTICS_CH_WRITE};

    std::array<optix::PayloadType, 3u> payload_types{};
    payload_types[0].numPayloadValues = trace_closest_payload_semantics.size();
    payload_types[0].payloadSemantics = trace_closest_payload_semantics.data();
    payload_types[1].numPayloadValues = trace_any_payload_semantics.size();
    payload_types[1].payloadSemantics = trace_any_payload_semantics.data();
    payload_types[2].numPayloadValues = ray_query_payload_semantics.size();
    payload_types[2].payloadSemantics = ray_query_payload_semantics.data();

    optix::ModuleCompileOptions module_compile_options{};
    module_compile_options.maxRegisterCount = optix::COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.debugLevel = enable_debug ? optix::COMPILE_DEBUG_LEVEL_MINIMAL :
                                                       optix::COMPILE_DEBUG_LEVEL_NONE;
    module_compile_options.optLevel = optix::COMPILE_OPTIMIZATION_LEVEL_3;
    module_compile_options.numPayloadTypes = payload_types.size();
    module_compile_options.payloadTypes = payload_types.data();

    optix::PipelineCompileOptions pipeline_compile_options{};
    pipeline_compile_options.exceptionFlags = optix::EXCEPTION_FLAG_NONE;
    pipeline_compile_options.traversableGraphFlags = optix::TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_compile_options.numPayloadValues = 0u;
    auto primitive_flags = uses_ray_query ? (optix::PRIMITIVE_TYPE_FLAGS_CUSTOM |
                                             optix::PRIMITIVE_TYPE_FLAGS_TRIANGLE) :
                                            optix::PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    pipeline_compile_options.usesPrimitiveTypeFlags = primitive_flags;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    char log[2048];// For error reporting from OptiX creation functions
    size_t log_size;
    LUISA_CHECK_OPTIX_WITH_LOG(
        log, log_size,
        optix::api().moduleCreateFromPTX(
            optix_ctx, &module_compile_options,
            &pipeline_compile_options, ptx, ptx_size,
            log, &log_size, &_module));

    // create program groups
    luisa::fixed_vector<optix::ProgramGroup, 6u> program_groups;

    // raygen
    optix::ProgramGroupOptions program_group_options_rg{};
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

    if (uses_trace_closest) {
        optix::ProgramGroupOptions program_group_options_ch_closest{};
        program_group_options_ch_closest.payloadType = &payload_types[0];
        optix::ProgramGroupDesc program_group_desc_ch_closest{};
        program_group_desc_ch_closest.kind = optix::PROGRAM_GROUP_KIND_HITGROUP;
        program_group_desc_ch_closest.hitgroup.moduleCH = _module;
        program_group_desc_ch_closest.hitgroup.entryFunctionNameCH = "__closesthit__trace_closest";
        LUISA_CHECK_OPTIX_WITH_LOG(
            log, log_size,
            optix::api().programGroupCreate(
                optix_ctx,
                &program_group_desc_ch_closest, 1u,
                &program_group_options_ch_closest,
                log, &log_size, &_program_group_ch_closest));

        optix::ProgramGroupOptions program_group_options_miss_closest{};
        program_group_options_miss_closest.payloadType = &payload_types[0];
        optix::ProgramGroupDesc program_group_desc_miss_closest{};
        program_group_desc_miss_closest.kind = optix::PROGRAM_GROUP_KIND_MISS;
        program_group_desc_miss_closest.miss.module = _module;
        program_group_desc_miss_closest.miss.entryFunctionName = "__miss__trace_closest";
        LUISA_CHECK_OPTIX_WITH_LOG(
            log, log_size,
            optix::api().programGroupCreate(
                optix_ctx,
                &program_group_desc_miss_closest, 1u,
                &program_group_options_miss_closest,
                log, &log_size, &_program_group_miss_closest));

        program_groups.emplace_back(_program_group_ch_closest);
        program_groups.emplace_back(_program_group_miss_closest);
    }

    if (uses_ray_query) {
        optix::ProgramGroupOptions program_group_options_ch_query{};
        program_group_options_ch_query.payloadType = &payload_types[2];
        optix::ProgramGroupDesc program_group_desc_ch_query{};
        program_group_desc_ch_query.kind = optix::PROGRAM_GROUP_KIND_HITGROUP;
        program_group_desc_ch_query.hitgroup.moduleCH = _module;
        program_group_desc_ch_query.hitgroup.entryFunctionNameCH = "__closesthit__ray_query";
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
                log, &log_size, &_program_group_ch_query));

        optix::ProgramGroupOptions program_group_options_ray_query{};
        program_group_options_ray_query.payloadType = &payload_types[2];
        optix::ProgramGroupDesc program_group_desc_ray_query{};
        program_group_desc_ray_query.kind = optix::PROGRAM_GROUP_KIND_MISS;
        program_group_desc_ray_query.miss.module = _module;
        program_group_desc_ray_query.miss.entryFunctionName = "__miss__ray_query";
        LUISA_CHECK_OPTIX_WITH_LOG(
            log, log_size,
            optix::api().programGroupCreate(
                optix_ctx,
                &program_group_desc_ray_query, 1u,
                &program_group_options_ray_query,
                log, &log_size, &_program_group_miss_query));

        program_groups.emplace_back(_program_group_ch_query);
        program_groups.emplace_back(_program_group_miss_query);
    }

    if (uses_trace_any) {
        optix::ProgramGroupOptions program_group_options_miss_any{};
        program_group_options_miss_any.payloadType = &payload_types[1];
        optix::ProgramGroupDesc program_group_desc_miss_any{};
        program_group_desc_miss_any.kind = optix::PROGRAM_GROUP_KIND_MISS;
        program_group_desc_miss_any.miss.module = _module;
        program_group_desc_miss_any.miss.entryFunctionName = "__miss__trace_any";
        LUISA_CHECK_OPTIX_WITH_LOG(
            log, log_size,
            optix::api().programGroupCreate(
                optix_ctx,
                &program_group_desc_miss_any, 1u,
                &program_group_options_miss_any,
                log, &log_size, &_program_group_miss_any));

        program_groups.emplace_back(_program_group_miss_any);
    }

    // create pipeline
    optix::PipelineLinkOptions pipeline_link_options{};
    pipeline_link_options.debugLevel = enable_debug ? optix::COMPILE_DEBUG_LEVEL_MINIMAL :
                                                      optix::COMPILE_DEBUG_LEVEL_NONE;
    pipeline_link_options.maxTraceDepth = 1u;
    LUISA_CHECK_OPTIX_WITH_LOG(
        log, log_size,
        optix::api().pipelineCreate(
            optix_ctx, &pipeline_compile_options, &pipeline_link_options,
            program_groups.data(), program_groups.size(), log, &log_size, &_pipeline));

    // compute stack sizes
    optix::StackSizes stack_sizes{};
    for (auto pg : program_groups) { accumulate_stack_sizes(stack_sizes, pg); }
    auto continuation_stack_size = compute_continuation_stack_size(stack_sizes);
    LUISA_CHECK_OPTIX(optix::api().pipelineSetStackSize(
        _pipeline, 0u, 0u, continuation_stack_size, 2u));

    // create shader binding table
    std::array<OptiXSBTRecord, 6u> sbt_records{};
    if (_program_group_rg) { LUISA_CHECK_OPTIX(optix::api().sbtRecordPackHeader(_program_group_rg, &sbt_records[0])); }
    if (_program_group_ch_closest) { LUISA_CHECK_OPTIX(optix::api().sbtRecordPackHeader(_program_group_ch_closest, &sbt_records[1])); }
    if (_program_group_ch_query) { LUISA_CHECK_OPTIX(optix::api().sbtRecordPackHeader(_program_group_ch_query, &sbt_records[2])); }
    if (_program_group_miss_closest) { LUISA_CHECK_OPTIX(optix::api().sbtRecordPackHeader(_program_group_miss_closest, &sbt_records[3])); }
    if (_program_group_miss_query) { LUISA_CHECK_OPTIX(optix::api().sbtRecordPackHeader(_program_group_miss_query, &sbt_records[4])); }
    if (_program_group_miss_any) { LUISA_CHECK_OPTIX(optix::api().sbtRecordPackHeader(_program_group_miss_any, &sbt_records[5])); }
    LUISA_CHECK_CUDA(cuMemAlloc(&_sbt_buffer, sbt_records.size() * sizeof(OptiXSBTRecord)));
    LUISA_CHECK_CUDA(cuMemcpyHtoD(_sbt_buffer, sbt_records.data(), sbt_records.size() * sizeof(OptiXSBTRecord)));
}

CUDAShaderOptiX::~CUDAShaderOptiX() noexcept {
    LUISA_CHECK_CUDA(cuMemFree(_sbt_buffer));
    LUISA_CHECK_OPTIX(optix::api().pipelineDestroy(_pipeline));
    if (_program_group_rg) { LUISA_CHECK_OPTIX(optix::api().programGroupDestroy(_program_group_rg)); }
    if (_program_group_ch_closest) { LUISA_CHECK_OPTIX(optix::api().programGroupDestroy(_program_group_ch_closest)); }
    if (_program_group_ch_query) { LUISA_CHECK_OPTIX(optix::api().programGroupDestroy(_program_group_ch_query)); }
    if (_program_group_miss_closest) { LUISA_CHECK_OPTIX(optix::api().programGroupDestroy(_program_group_miss_closest)); }
    if (_program_group_miss_any) { LUISA_CHECK_OPTIX(optix::api().programGroupDestroy(_program_group_miss_any)); }
    if (_program_group_miss_query) { LUISA_CHECK_OPTIX(optix::api().programGroupDestroy(_program_group_miss_query)); }
    LUISA_CHECK_OPTIX(optix::api().moduleDestroy(_module));
}

class CUDAIndirectDispatchOptiX final : public CUDAIndirectDispatchStream::Task {

public:
    struct DispatchBuffer {
        uint size;
        [[no_unique_address]] CUDAIndirectDispatchBuffer::Dispatch dispatches[];
    };

private:
    const CUDAShaderOptiX *_shader;
    CUdeviceptr _device_argument_buffer;
    CUdeviceptr _device_dispatch_buffer;
    luisa::vector<std::byte> _downloaded_dispatch_buffer;

private:
    [[nodiscard]] static auto &_pool() noexcept {
        static Pool<CUDAIndirectDispatchOptiX, true> pool;
        return pool;
    }

public:
    CUDAIndirectDispatchOptiX(const CUDAShaderOptiX *shader,
                              CUdeviceptr device_argument_buffer,
                              CUdeviceptr device_dispatch_buffer,
                              luisa::vector<std::byte> &&downloaded_dispatch_buffer) noexcept
        : _shader{shader},
          _device_argument_buffer{device_argument_buffer},
          _device_dispatch_buffer{device_dispatch_buffer},
          _downloaded_dispatch_buffer{std::move(downloaded_dispatch_buffer)} {}

    void execute(CUstream stream) noexcept override {
        auto count = reinterpret_cast<DispatchBuffer *>(_downloaded_dispatch_buffer.data())->size;
        LUISA_INFO("Dispatching {} tasks...", count);
        auto dispatch_size_in_argument_buffer = _device_argument_buffer +
                                                _shader->_argument_buffer_size - sizeof(uint4);
        auto device_dispatch_buffer = reinterpret_cast<DispatchBuffer *>(_device_dispatch_buffer);
        auto sbt = _shader->_make_sbt();
        for (auto i = 0u; i < count; i++) {
            auto dispatch_size_in_dispatch_buffer = reinterpret_cast<CUdeviceptr>(
                &(device_dispatch_buffer->dispatches[i].dispatch_size_and_kernel_id));
            auto &dispatch_size = reinterpret_cast<DispatchBuffer *>(_downloaded_dispatch_buffer.data())
                                     ->dispatches[i]
                                     .dispatch_size_and_kernel_id;
            LUISA_INFO("Dispatch #{}: ({}, {}, {})", i, dispatch_size.x, dispatch_size.y, dispatch_size.z);
            LUISA_INFO("Copy from 0x{:016x} to 0x{:016x}",
                       dispatch_size_in_dispatch_buffer,
                       dispatch_size_in_argument_buffer);
            LUISA_CHECK_CUDA(cuMemcpyDtoDAsync(dispatch_size_in_argument_buffer,
                                               dispatch_size_in_dispatch_buffer,
                                               sizeof(uint4), stream));
            LUISA_CHECK_OPTIX(optix::api().launch(
                _shader->_pipeline, stream, _device_argument_buffer,
                _shader->_argument_buffer_size, &sbt,
                dispatch_size.x, dispatch_size.y, dispatch_size.z));
        }
        _pool().destroy(this);
    }
    [[nodiscard]] static auto create(const CUDAShaderOptiX *shader,
                                     CUdeviceptr device_argument_buffer,
                                     CUdeviceptr device_dispatch_buffer,
                                     luisa::vector<std::byte> &&downloaded_dispatch_buffer) noexcept {
        return _pool().create(shader,
                              device_argument_buffer,
                              device_dispatch_buffer,
                              std::move(downloaded_dispatch_buffer));
    }
};

void CUDAShaderOptiX::_launch(CUDACommandEncoder &encoder, ShaderDispatchCommand *command) const noexcept {

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
        if (!command->is_indirect() && argument_buffer->is_pooled()) [[likely]] {
            // for (direct) dispatches, if the argument buffer
            // is pooled, we can use the device pointer directly
            auto device_argument_buffer = 0ull;
            LUISA_CHECK_CUDA(cuMemHostGetDevicePointer(
                &device_argument_buffer, argument_buffer->address(), 0u));
            _do_launch(cuda_stream, device_argument_buffer, s);
        } else {// otherwise, we need to copy the argument buffer to the device
            auto device_argument_buffer = 0ull;
            LUISA_CHECK_CUDA(cuMemAllocAsync(
                &device_argument_buffer, _argument_buffer_size, cuda_stream));
            LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(
                device_argument_buffer, argument_buffer->address(),
                _argument_buffer_size, cuda_stream));
            if (command->is_indirect()) {
                auto indirect_buffer = reinterpret_cast<CUDAIndirectDispatchBuffer *>(
                    command->indirect_dispatch().handle);
                luisa::vector<std::byte> downloaded_dispatch_buffer(indirect_buffer->size_bytes());
                LUISA_CHECK_CUDA(cuMemcpyDtoHAsync(
                    downloaded_dispatch_buffer.data(), indirect_buffer->handle(),
                    downloaded_dispatch_buffer.size(), cuda_stream));
                encoder.stream()->indirect().enqueue(CUDAIndirectDispatchOptiX::create(
                    this, device_argument_buffer, indirect_buffer->handle(),
                    std::move(downloaded_dispatch_buffer)));
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
    sbt.hitgroupRecordCount = 2u;
    sbt.hitgroupRecordStrideInBytes = sizeof(OptiXSBTRecord);
    sbt.missRecordBase = _sbt_buffer + sizeof(OptiXSBTRecord) * 3u;
    sbt.missRecordCount = 3u;
    sbt.missRecordStrideInBytes = sizeof(OptiXSBTRecord);
    return sbt;
}

void CUDAShaderOptiX::_do_launch(CUstream stream, CUdeviceptr argument_buffer, uint3 dispatch_size) const noexcept {
    auto sbt = _make_sbt();
    LUISA_CHECK_OPTIX(optix::api().launch(
        _pipeline, stream, argument_buffer, _argument_buffer_size,
        &sbt, dispatch_size.x, dispatch_size.y, dispatch_size.z));
}

}// namespace luisa::compute::cuda

