//
// Created by Mike on 2021/12/4.
//

#include <algorithm>
#include <mutex>

#include <cuda.h>

#include <core/hash.h>
#include <core/spin_mutex.h>
#include <ast/function.h>
#include <runtime/command.h>
#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_device.h>
#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_accel.h>
#include <backends/cuda/cuda_mipmap_array.h>
#include <backends/cuda/cuda_bindless_array.h>
#include <backends/cuda/cuda_shader.h>
#include <backends/cuda/optix_api.h>

namespace luisa::compute::cuda {

class CUDAShaderNative final : public CUDAShader {

private:
    CUmodule _module{};
    CUfunction _function{};
    luisa::string _entry;

public:
    CUDAShaderNative(const char *ptx, size_t ptx_size, const char *entry) noexcept
        : _entry{entry} {
        auto ret = cuModuleLoadData(&_module, ptx);
        if (ret == CUDA_ERROR_UNSUPPORTED_PTX_VERSION) {
            // For users with newer CUDA and older driver,
            // the generated PTX might be reported invalid.
            // We have to patch the ".version 7.x" instruction.
            using namespace std::string_view_literals;
            luisa::string s{ptx, ptx_size};
            auto pattern = ".version 7."sv;
            if (auto p = s.find(pattern); p != luisa::string_view::npos) {
                auto begin = p + pattern.size();
                auto end = begin;
                for (; isdigit(s[end]); end++) {}
                s.replace(begin, end - begin, "0");
            }
            ret = cuModuleLoadData(&_module, s.c_str());
        }
        LUISA_CHECK_CUDA(ret);
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

/// Retrieves direct and continuation stack sizes for each program in the program group and accumulates the upper bounds
/// in the correponding output variables based on the semantic type of the program. Before the first invocation of this
/// function with a given instance of #OptixStackSizes, the members of that instance should be set to 0.
inline void accumulate_stack_sizes(optix::StackSizes &sizes, optix::ProgramGroup programGroup) noexcept {
    optix::StackSizes local{};
    LUISA_CHECK_OPTIX(optix::api().programGroupGetStackSize(programGroup, &local));
    sizes.cssRG = std::max(sizes.cssRG, local.cssRG);
    sizes.cssMS = std::max(sizes.cssMS, local.cssMS);
    sizes.cssCH = std::max(sizes.cssCH, local.cssCH);
    sizes.cssAH = std::max(sizes.cssAH, local.cssAH);
    sizes.cssIS = std::max(sizes.cssIS, local.cssIS);
    sizes.cssCC = std::max(sizes.cssCC, local.cssCC);
    sizes.dssDC = std::max(sizes.dssDC, local.dssDC);
}

[[nodiscard]] inline uint compute_continuation_stack_size(optix::StackSizes ss) noexcept {
    return ss.cssRG + std::max(std::max(ss.cssCH, ss.cssMS), ss.cssIS + ss.cssAH);
}

class CUDAShaderOptiX final : public CUDAShader {

public:
    struct alignas(optix::SBT_RECORD_ALIGNMENT) SBTRecord {
        std::byte data[optix::SBT_RECORD_HEADER_SIZE];
    };

private:
    CUDADevice *_device;
    size_t _argument_buffer_size{};
    optix::Module _module{};
    optix::ProgramGroup _program_group_rg{};
    optix::ProgramGroup _program_group_ch_closest{};
    optix::ProgramGroup _program_group_ch_any{};
    optix::ProgramGroup _program_group_miss{};
    optix::Pipeline _pipeline{};
    luisa::string _entry;
    mutable CUdeviceptr _sbt_buffer{};
    mutable luisa::spin_mutex _mutex;
    mutable CUevent _sbt_event{};
    mutable optix::ShaderBindingTable _sbt{};
    mutable luisa::unordered_set<CUstream> _sbt_recoreded_streams;

public:
    CUDAShaderOptiX(CUDADevice *device, const char *ptx, size_t ptx_size, const char *entry) noexcept
        : _device{device}, _entry{entry} {

        // create SBT event
        LUISA_CHECK_CUDA(cuEventCreate(&_sbt_event, CU_EVENT_DISABLE_TIMING));

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

        // create module
        optix::ModuleCompileOptions module_compile_options{};
        module_compile_options.maxRegisterCount = 80u;
        module_compile_options.debugLevel = optix::COMPILE_DEBUG_LEVEL_NONE;
        module_compile_options.optLevel = optix::COMPILE_OPTIMIZATION_LEVEL_3;
        optix::PipelineCompileOptions pipeline_compile_options{};
        pipeline_compile_options.exceptionFlags = optix::EXCEPTION_FLAG_NONE;
        pipeline_compile_options.traversableGraphFlags = optix::TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        pipeline_compile_options.numPayloadValues = 4u;
        pipeline_compile_options.usesPrimitiveTypeFlags = optix::PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        char log[2048];// For error reporting from OptiX creation functions
        size_t log_size;
        LUISA_CHECK_OPTIX_WITH_LOG(
            log, log_size,
            optix::api().moduleCreateFromPTX(
                device->handle().optix_context(), &module_compile_options,
                &pipeline_compile_options, ptx, ptx_size,
                log, &log_size, &_module));

        // create program groups
        optix::ProgramGroupOptions program_group_options{};
        optix::ProgramGroupDesc program_group_desc_rg{};
        program_group_desc_rg.kind = optix::PROGRAM_GROUP_KIND_RAYGEN;
        program_group_desc_rg.raygen.module = _module;
        program_group_desc_rg.raygen.entryFunctionName = entry;
        LUISA_CHECK_OPTIX_WITH_LOG(
            log, log_size,
            optix::api().programGroupCreate(
                device->handle().optix_context(),
                &program_group_desc_rg, 1u,
                &program_group_options,
                log, &log_size, &_program_group_rg));
        optix::ProgramGroupDesc program_group_desc_ch_closest{};
        program_group_desc_ch_closest.kind = optix::PROGRAM_GROUP_KIND_HITGROUP;
        program_group_desc_ch_closest.hitgroup.moduleCH = _module;
        program_group_desc_ch_closest.hitgroup.entryFunctionNameCH = "__closesthit__trace_closest";
        LUISA_CHECK_OPTIX_WITH_LOG(
            log, log_size,
            optix::api().programGroupCreate(
                device->handle().optix_context(),
                &program_group_desc_ch_closest, 1u,
                &program_group_options,
                log, &log_size, &_program_group_ch_closest));
        optix::ProgramGroupDesc program_group_desc_ch_any{};
        program_group_desc_ch_any.kind = optix::PROGRAM_GROUP_KIND_HITGROUP;
        program_group_desc_ch_any.hitgroup.moduleCH = _module;
        program_group_desc_ch_any.hitgroup.entryFunctionNameCH = "__closesthit__trace_any";
        LUISA_CHECK_OPTIX_WITH_LOG(
            log, log_size,
            optix::api().programGroupCreate(
                device->handle().optix_context(),
                &program_group_desc_ch_any, 1u,
                &program_group_options,
                log, &log_size, &_program_group_ch_any));
        optix::ProgramGroupDesc program_group_desc_miss{};
        program_group_desc_miss.kind = optix::PROGRAM_GROUP_KIND_MISS;

        LUISA_CHECK_OPTIX_WITH_LOG(
            log, log_size,
            optix::api().programGroupCreate(
                device->handle().optix_context(),
                &program_group_desc_miss, 1u,
                &program_group_options,
                log, &log_size, &_program_group_miss));

        // create pipeline
        optix::ProgramGroup program_groups[]{
            _program_group_rg, _program_group_ch_closest, _program_group_ch_any};
        optix::PipelineLinkOptions pipeline_link_options{};
        pipeline_link_options.debugLevel = optix::COMPILE_DEBUG_LEVEL_NONE;
        pipeline_link_options.maxTraceDepth = 1u;
        LUISA_CHECK_OPTIX_WITH_LOG(
            log, log_size,
            optix::api().pipelineCreate(
                device->handle().optix_context(),
                &pipeline_compile_options,
                &pipeline_link_options,
                program_groups, 2u,
                log, &log_size,
                &_pipeline));

        // compute stack sizes
        optix::StackSizes stack_sizes{};
        for (auto pg : program_groups) {
            accumulate_stack_sizes(stack_sizes, pg);
        }
        auto continuation_stack_size = compute_continuation_stack_size(stack_sizes);
        LUISA_CHECK_OPTIX(optix::api().pipelineSetStackSize(
            _pipeline, 0u, 0u, continuation_stack_size, 2u));
    }

    ~CUDAShaderOptiX() noexcept override {
        LUISA_CHECK_CUDA(cuMemFree(_sbt_buffer));
        LUISA_CHECK_CUDA(cuEventDestroy(_sbt_event));
        LUISA_CHECK_OPTIX(optix::api().pipelineDestroy(_pipeline));
        LUISA_CHECK_OPTIX(optix::api().programGroupDestroy(_program_group_rg));
        LUISA_CHECK_OPTIX(optix::api().programGroupDestroy(_program_group_ch_any));
        LUISA_CHECK_OPTIX(optix::api().programGroupDestroy(_program_group_ch_closest));
        LUISA_CHECK_OPTIX(optix::api().programGroupDestroy(_program_group_miss));
        LUISA_CHECK_OPTIX(optix::api().moduleDestroy(_module));
    }

private:
    void _prepare_sbt(CUDAStream *stream, CUstream cuda_stream) const noexcept {
        std::scoped_lock lock{_mutex};
        if (_sbt.raygenRecord == 0u) {// create shader binding table if not present
            constexpr auto sbt_buffer_size = sizeof(SBTRecord) * 4u;
            LUISA_CHECK_CUDA(cuMemAllocAsync(&_sbt_buffer, sbt_buffer_size, cuda_stream));
            auto sbt_record_buffer = stream->upload_pool()->allocate(sbt_buffer_size);
            auto sbt_records = reinterpret_cast<SBTRecord *>(sbt_record_buffer->address());
            LUISA_CHECK_OPTIX(optix::api().sbtRecordPackHeader(_program_group_rg, &sbt_records[0]));
            LUISA_CHECK_OPTIX(optix::api().sbtRecordPackHeader(_program_group_ch_closest, &sbt_records[1]));
            LUISA_CHECK_OPTIX(optix::api().sbtRecordPackHeader(_program_group_ch_any, &sbt_records[2]));
            LUISA_CHECK_OPTIX(optix::api().sbtRecordPackHeader(_program_group_miss, &sbt_records[3]));
            LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(
                _sbt_buffer, sbt_record_buffer->address(),
                sbt_buffer_size, cuda_stream));
            LUISA_CHECK_CUDA(cuEventRecord(_sbt_event, cuda_stream));
            stream->emplace_callback(sbt_record_buffer);
            _sbt.raygenRecord = _sbt_buffer;
            _sbt.hitgroupRecordBase = _sbt_buffer + sizeof(SBTRecord);
            _sbt.hitgroupRecordCount = 2u;
            _sbt.hitgroupRecordStrideInBytes = sizeof(SBTRecord);
            _sbt.missRecordBase = _sbt_buffer + sizeof(SBTRecord) * 3u;
            _sbt.missRecordCount = 1u;
            _sbt.missRecordStrideInBytes = sizeof(SBTRecord);
            _sbt_recoreded_streams.emplace(cuda_stream);
        } else {
            if (_sbt_recoreded_streams.emplace(cuda_stream).second) {
                LUISA_CHECK_CUDA(cuStreamWaitEvent(cuda_stream, _sbt_event, 0u));
            }
        }
    }

public:
    void launch(CUDAStream *stream, const ShaderDispatchCommand *command) const noexcept override {
        auto cuda_stream = stream->handle();
        _prepare_sbt(stream, cuda_stream);

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
        auto s = command->dispatch_size();
        if (host_argument_buffer->is_pooled()) {
            auto argument_buffer = 0ull;
            LUISA_CHECK_CUDA(cuMemHostGetDevicePointer(
                &argument_buffer, host_argument_buffer->address(), 0u));
            LUISA_CHECK_OPTIX(optix::api().launch(
                _pipeline, cuda_stream, argument_buffer,
                _argument_buffer_size, &_sbt, s.x, s.y, s.z));
        } else {
            auto argument_buffer = 0ull;
            LUISA_CHECK_CUDA(cuMemAllocAsync(
                &argument_buffer, _argument_buffer_size, cuda_stream));
            LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(
                argument_buffer, host_argument_buffer->address(),
                _argument_buffer_size, cuda_stream));
            LUISA_CHECK_OPTIX(optix::api().launch(
                _pipeline, cuda_stream, argument_buffer,
                _argument_buffer_size, &_sbt, s.x, s.y, s.z));
            LUISA_CHECK_CUDA(cuMemFreeAsync(argument_buffer, cuda_stream));
        }
        stream->emplace_callback(host_argument_buffer);
    }
};

CUDAShader *CUDAShader::create(CUDADevice *device, const char *ptx, size_t ptx_size,
                               const char *entry, bool is_raytracing) noexcept {
    return is_raytracing ?
               static_cast<CUDAShader *>(new_with_allocator<CUDAShaderOptiX>(device, ptx, ptx_size, entry)) :
               static_cast<CUDAShader *>(new_with_allocator<CUDAShaderNative>(ptx, ptx_size, entry));
}

void CUDAShader::destroy(CUDAShader *shader) noexcept {
    delete_with_allocator(shader);
}

}// namespace luisa::compute::cuda
