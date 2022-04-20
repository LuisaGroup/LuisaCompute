//
// Created by Mike on 2021/12/4.
//

#include <cuda.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <ast/function.h>
#include <runtime/command.h>
#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_device.h>
#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_accel.h>
#include <backends/cuda/cuda_mipmap_array.h>
#include <backends/cuda/cuda_bindless_array.h>
#include <backends/cuda/cuda_shader.h>

namespace luisa::compute::cuda {

class CUDAShaderNative final : public CUDAShader {

private:
    CUmodule _module{};
    CUfunction _function{};
    luisa::string _entry;

public:
    CUDAShaderNative(const char *ptx, const char *entry) noexcept
        : _entry{entry} {
        LUISA_CHECK_CUDA(cuModuleLoadData(&_module, ptx));
        LUISA_CHECK_CUDA(cuModuleGetFunction(&_function, _module, entry));
    }

    ~CUDAShaderNative() noexcept override {
        LUISA_CHECK_CUDA(cuModuleUnload(_module));
    }

    void launch(CUDAStream *stream, const ShaderDispatchCommand *command) const noexcept override {
        static thread_local std::array<std::byte, 65536u> argument_buffer;// should be enough...
        static thread_local std::vector<void *> arguments;
        auto argument_buffer_offset = static_cast<size_t>(0u);
        auto allocate_argument = [&](size_t bytes) noexcept {
            static constexpr auto alignment = 16u;
            auto offset = (argument_buffer_offset + alignment - 1u) / alignment * alignment;
            argument_buffer_offset = offset + bytes;
            if (argument_buffer_offset > argument_buffer.size()) {
                LUISA_ERROR_WITH_LOCATION(
                    "Too many arguments in ShaderDispatchCommand");
            }
            return arguments.emplace_back(argument_buffer.data() + offset);
        };
        arguments.clear();
        arguments.reserve(32u);
        command->decode([&](auto argument) noexcept -> void {
            using T = decltype(argument);
            if constexpr (std::is_same_v<T, ShaderDispatchCommand::BufferArgument>) {
                auto ptr = allocate_argument(sizeof(CUdeviceptr));
                auto buffer = argument.handle + argument.offset;
                std::memcpy(ptr, &buffer, sizeof(CUdeviceptr));
            } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::TextureArgument>) {
                auto mipmap_array = reinterpret_cast<CUDAMipmapArray *>(argument.handle);
                auto surface = mipmap_array->surface(argument.level);
                auto ptr = allocate_argument(sizeof(CUDASurface));
                std::memcpy(ptr, &surface, sizeof(CUDASurface));
            } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::BindlessArrayArgument>) {
                auto ptr = allocate_argument(sizeof(CUDABindlessArray::SlotSOA));
                auto array = reinterpret_cast<CUDABindlessArray *>(argument.handle)->handle();
                std::memcpy(ptr, &array, sizeof(CUDABindlessArray::SlotSOA));
            } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::AccelArgument>) {
                auto ptr = allocate_argument(sizeof(CUDAAccel::Binding));
                auto accel = reinterpret_cast<CUDAAccel *>(argument.handle);
                CUDAAccel::Binding binding{.handle = accel->handle(), .instances = accel->instance_buffer()};
                std::memcpy(ptr, &binding, sizeof(CUDAAccel::Binding));
            } else {// uniform
                static_assert(std::same_as<T, ShaderDispatchCommand::UniformArgument>);
                auto ptr = allocate_argument(argument.size);
                std::memcpy(ptr, argument.data, argument.size);
            }
        });
        // the last one is always the launch size
        auto launch_size = command->dispatch_size();
        auto ptr = allocate_argument(sizeof(luisa::uint3));
        std::memcpy(ptr, &launch_size, sizeof(luisa::uint3));
        auto block_size = command->kernel().block_size();
        auto blocks = (launch_size + block_size - 1u) / block_size;
        LUISA_VERBOSE_WITH_LOCATION(
            "Dispatching native shader #{} ({}) with {} argument(s) "
            "in ({}, {}, {}) blocks of size ({}, {}, {}).",
            command->handle(), _entry, arguments.size(),
            blocks.x, blocks.y, blocks.z,
            block_size.x, block_size.y, block_size.z);
        auto cuda_stream = stream->handle();
        LUISA_CHECK_CUDA(cuLaunchKernel(
            _function,
            blocks.x, blocks.y, blocks.z,
            block_size.x, block_size.y, block_size.z,
            0u, cuda_stream,
            arguments.data(), nullptr));
    }
};

// TODO...
class CUDAShaderOptiX final : public CUDAShader {

public:
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SBTRecord {
        std::byte data[OPTIX_SBT_RECORD_HEADER_SIZE];
    };

private:
    CUDADevice *_device;
    CUdeviceptr _sbt_buffer{};
    size_t _argument_buffer_size{};
    OptixModule _module{};
    OptixProgramGroup _program_group_rg{};
    OptixProgramGroup _program_group_ch_closest{};
    OptixProgramGroup _program_group_ch_any{};
    OptixProgramGroup _program_group_miss{};
    OptixPipeline _pipeline{};
    luisa::string _entry;
    mutable OptixShaderBindingTable _sbt{};

public:
    CUDAShaderOptiX(CUDADevice *device, const char *ptx, size_t ptx_size, const char *entry) noexcept
        : _device{device}, _entry{entry} {

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
        LUISA_CHECK_CUDA(cuMemAlloc(&_sbt_buffer, sizeof(SBTRecord) * 4u));

        // create module
        OptixModuleCompileOptions module_compile_options{};
        module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
        OptixPipelineCompileOptions pipeline_compile_options{};
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        pipeline_compile_options.numPayloadValues = 4u;
        pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        char log[2048];// For error reporting from OptiX creation functions
        size_t log_size;
        LUISA_CHECK_OPTIX_WITH_LOG(
            log, log_size,
            optixModuleCreateFromPTX(
                device->handle().optix_context(), &module_compile_options,
                &pipeline_compile_options, ptx, ptx_size,
                log, &log_size, &_module));

        // create program groups
        OptixProgramGroupOptions program_group_options{};
        OptixProgramGroupDesc program_group_desc_rg{};
        program_group_desc_rg.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        program_group_desc_rg.raygen.module = _module;
        program_group_desc_rg.raygen.entryFunctionName = entry;
        LUISA_CHECK_OPTIX_WITH_LOG(
            log, log_size,
            optixProgramGroupCreate(
                device->handle().optix_context(),
                &program_group_desc_rg, 1u,
                &program_group_options,
                log, &log_size, &_program_group_rg));
        OptixProgramGroupDesc program_group_desc_ch_closest{};
        program_group_desc_ch_closest.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        program_group_desc_ch_closest.hitgroup.moduleCH = _module;
        program_group_desc_ch_closest.hitgroup.entryFunctionNameCH = "__closesthit__trace_closest";
        LUISA_CHECK_OPTIX_WITH_LOG(
            log, log_size,
            optixProgramGroupCreate(
                device->handle().optix_context(),
                &program_group_desc_ch_closest, 1u,
                &program_group_options,
                log, &log_size, &_program_group_ch_closest));
        OptixProgramGroupDesc program_group_desc_ch_any{};
        program_group_desc_ch_any.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        program_group_desc_ch_any.hitgroup.moduleCH = _module;
        program_group_desc_ch_any.hitgroup.entryFunctionNameCH = "__closesthit__trace_any";
        LUISA_CHECK_OPTIX_WITH_LOG(
            log, log_size,
            optixProgramGroupCreate(
                device->handle().optix_context(),
                &program_group_desc_ch_any, 1u,
                &program_group_options,
                log, &log_size, &_program_group_ch_any));
        OptixProgramGroupDesc program_group_desc_miss{};
        program_group_desc_miss.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;

        LUISA_CHECK_OPTIX_WITH_LOG(
            log, log_size,
            optixProgramGroupCreate(
                device->handle().optix_context(),
                &program_group_desc_miss, 1u,
                &program_group_options,
                log, &log_size, &_program_group_miss));

        // create pipeline
        OptixProgramGroup program_groups[]{_program_group_rg, _program_group_ch_closest, _program_group_ch_any};
        OptixPipelineLinkOptions pipeline_link_options{};
        pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        pipeline_link_options.maxTraceDepth = 1u;
        LUISA_CHECK_OPTIX_WITH_LOG(
            log, log_size,
            optixPipelineCreate(
                device->handle().optix_context(),
                &pipeline_compile_options,
                &pipeline_link_options,
                program_groups, 2u,
                log, &log_size,
                &_pipeline));

        // compute stack sizes
        OptixStackSizes stack_sizes{};
        for (auto pg : program_groups) {
            LUISA_CHECK_OPTIX(optixUtilAccumulateStackSizes(pg, &stack_sizes));
        }
        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        LUISA_CHECK_OPTIX(optixUtilComputeStackSizes(
            &stack_sizes, 1u, 0u, 0u,
            &direct_callable_stack_size_from_traversal,
            &direct_callable_stack_size_from_state,
            &continuation_stack_size));
        LUISA_CHECK_OPTIX(optixPipelineSetStackSize(
            _pipeline,
            direct_callable_stack_size_from_traversal,
            direct_callable_stack_size_from_state,
            continuation_stack_size, 2u));
    }

    ~CUDAShaderOptiX() noexcept override {
        LUISA_CHECK_CUDA(cuMemFree(_sbt_buffer));
        LUISA_CHECK_OPTIX(optixPipelineDestroy(_pipeline));
        LUISA_CHECK_OPTIX(optixProgramGroupDestroy(_program_group_rg));
        LUISA_CHECK_OPTIX(optixProgramGroupDestroy(_program_group_ch_any));
        LUISA_CHECK_OPTIX(optixProgramGroupDestroy(_program_group_ch_closest));
        LUISA_CHECK_OPTIX(optixProgramGroupDestroy(_program_group_miss));
        LUISA_CHECK_OPTIX(optixModuleDestroy(_module));
    }

    void launch(CUDAStream *stream, const ShaderDispatchCommand *command) const noexcept override {
        auto cuda_stream = stream->handle();
        auto argument_buffer = 0ull;
        LUISA_CHECK_CUDA(cuMemAllocAsync(&argument_buffer, _argument_buffer_size, cuda_stream));
        if (_sbt.raygenRecord == 0u) {// create shader binding table if not present
            constexpr auto sbt_buffer_size = sizeof(SBTRecord) * 4u;
            auto sbt_record_buffer = stream->upload_pool()->allocate(sbt_buffer_size);
            auto sbt_records = reinterpret_cast<SBTRecord *>(sbt_record_buffer->address());
            LUISA_CHECK_OPTIX(optixSbtRecordPackHeader(_program_group_rg, &sbt_records[0]));
            LUISA_CHECK_OPTIX(optixSbtRecordPackHeader(_program_group_ch_closest, &sbt_records[1]));
            LUISA_CHECK_OPTIX(optixSbtRecordPackHeader(_program_group_ch_any, &sbt_records[2]));
            LUISA_CHECK_OPTIX(optixSbtRecordPackHeader(_program_group_miss, &sbt_records[3]));
            LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(
                _sbt_buffer, sbt_record_buffer->address(),
                sbt_buffer_size, cuda_stream));
            stream->emplace_callback(sbt_record_buffer);
            _sbt.raygenRecord = _sbt_buffer;
            _sbt.hitgroupRecordBase = _sbt_buffer + sizeof(SBTRecord);
            _sbt.hitgroupRecordCount = 2u;
            _sbt.hitgroupRecordStrideInBytes = sizeof(SBTRecord);
            _sbt.missRecordBase = _sbt_buffer + sizeof(SBTRecord) * 3u;
            _sbt.missRecordCount = 1u;
            _sbt.missRecordStrideInBytes = sizeof(SBTRecord);
        }
        // encode arguments
        auto host_argument_buffer = stream->upload_pool()->allocate(_argument_buffer_size);
        auto argument_buffer_offset = static_cast<size_t>(0u);
        auto allocate_argument = [&](size_t bytes) noexcept {
            static constexpr auto alignment = 16u;
            auto offset = (argument_buffer_offset + alignment - 1u) / alignment * alignment;
            argument_buffer_offset = offset + bytes;
            if (argument_buffer_offset > _argument_buffer_size) {
                LUISA_ERROR_WITH_LOCATION(
                    "Too many arguments in ShaderDispatchCommand");
            }
            return host_argument_buffer->address() + offset;
        };
        command->decode([&](auto argument) noexcept -> void {
            using T = decltype(argument);
            if constexpr (std::is_same_v<T, ShaderDispatchCommand::BufferArgument>) {
                auto ptr = allocate_argument(sizeof(CUdeviceptr));
                auto buffer = argument.handle + argument.offset;
                std::memcpy(ptr, &buffer, sizeof(CUdeviceptr));
            } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::TextureArgument>) {
                auto mipmap_array = reinterpret_cast<CUDAMipmapArray *>(argument.handle);
                auto surface = mipmap_array->surface(argument.level);
                auto ptr = allocate_argument(sizeof(CUDASurface));
                std::memcpy(ptr, &surface, sizeof(CUDASurface));
            } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::BindlessArrayArgument>) {
                auto ptr = allocate_argument(sizeof(CUDABindlessArray::SlotSOA));
                auto array = reinterpret_cast<CUDABindlessArray *>(argument.handle)->handle();
                std::memcpy(ptr, &array, sizeof(CUDABindlessArray::SlotSOA));
            } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::AccelArgument>) {
                auto ptr = allocate_argument(sizeof(CUDAAccel::Binding));
                auto accel = reinterpret_cast<CUDAAccel *>(argument.handle);
                CUDAAccel::Binding binding{.handle = accel->handle(), .instances = accel->instance_buffer()};
                std::memcpy(ptr, &binding, sizeof(CUDAAccel::Binding));
            } else {// uniform
                static_assert(std::same_as<T, ShaderDispatchCommand::UniformArgument>);
                auto ptr = allocate_argument(argument.size);
                std::memcpy(ptr, argument.data, argument.size);
            }
        });
        LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(
            argument_buffer, host_argument_buffer->address(),
            _argument_buffer_size, cuda_stream));
        stream->emplace_callback(host_argument_buffer);
        auto s = command->dispatch_size();
        LUISA_CHECK_OPTIX(optixLaunch(
            _pipeline, cuda_stream, argument_buffer,
            _argument_buffer_size, &_sbt, s.x, s.y, s.z));
        LUISA_CHECK_CUDA(cuMemFreeAsync(argument_buffer, cuda_stream));
    }
};

CUDAShader *CUDAShader::create(CUDADevice *device, const char *ptx, size_t ptx_size, const char *entry, bool is_raytracing) noexcept {
    return is_raytracing ?
               static_cast<CUDAShader *>(new_with_allocator<CUDAShaderOptiX>(device, ptx, ptx_size, entry)) :
               static_cast<CUDAShader *>(new_with_allocator<CUDAShaderNative>(ptx, entry));
}

void CUDAShader::destroy(CUDAShader *shader) noexcept {
    delete_with_allocator(shader);
}

}// namespace luisa::compute::cuda
