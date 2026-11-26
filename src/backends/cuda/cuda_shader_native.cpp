#include <luisa/runtime/rhi/command.h>
#include <luisa/core/binary_io.h>
#include <luisa/core/stl/hash.h>
#include <luisa/core/stl/string.h>

#include "cuda_error.h"
#include "cuda_device.h"
#include "cuda_buffer.h"
#include "cuda_accel.h"
#include "cuda_texture.h"
#include "cuda_bindless_array.h"
#include "cuda_command_encoder.h"
#include "cuda_pdl_launch.h"
#include "cuda_shader_native.h"
#include "cuda_shader_printer.h"

namespace luisa::compute::cuda {

namespace cuda_shader_native_detail {

[[nodiscard]] inline CUresult launch_kernel(CUfunction function,
                                            uint3 grid_size, uint3 block_size,
                                            unsigned int shared_memory_bytes,
                                            CUstream stream, void **arguments,
                                            bool relax_stream_serialization) noexcept {
#if CUDA_VERSION >= 12000
    if (relax_stream_serialization) {
        // Zero-initialise the whole aggregate: the attribute carries internal
        // padding that must not be left undefined.
        CUlaunchAttribute attribute{};
        attribute.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        attribute.value.programmaticStreamSerializationAllowed = 1;
        CUlaunchConfig config{};
        config.gridDimX = grid_size.x;
        config.gridDimY = grid_size.y;
        config.gridDimZ = grid_size.z;
        config.blockDimX = block_size.x;
        config.blockDimY = block_size.y;
        config.blockDimZ = block_size.z;
        config.sharedMemBytes = shared_memory_bytes;
        config.hStream = stream;
        config.attrs = &attribute;
        config.numAttrs = 1u;
        return cuLaunchKernelEx(&config, function, arguments, nullptr);
    }
#else
    static_cast<void>(relax_stream_serialization);// the attribute needs CUDA >= 12.0
#endif
    // The unchanged classic launch (numAttrs = 0, attrs = nullptr).
    return cuLaunchKernel(
        function,
        grid_size.x, grid_size.y, grid_size.z,
        block_size.x, block_size.y, block_size.z,
        shared_memory_bytes, stream, arguments, nullptr);
}

}// namespace cuda_shader_native_detail

CUDAShaderNative::CUDAShaderNative(CUDADevice *device, luisa::vector<std::byte> ptx,
                                   const char *entry, const CUDAShaderMetadata &metadata,
                                   luisa::vector<ShaderDispatchCommand::Argument> bound_arguments,
                                   bool enable_cache) noexcept
    : CUDAShader{CUDAShaderPrinter::create(metadata.format_types),
                 metadata.argument_usages,
                 metadata.texture_sampler_masks},
      _entry{entry},
      _block_size{metadata.block_size.x,
                  metadata.block_size.y,
                  metadata.block_size.z},
      _bound_arguments{std::move(bound_arguments)},
      _uses_grid_dependency_synchronize{metadata.requires_grid_dependency_synchronize} {

    // A generated PTX module only needs the CUDA device runtime (cudadevrt)
    // when it references one of the entry points that live there. Everything
    // else keeps the plain-PTX load path, which also keeps the loaded module
    // image usable for cross-backend import.
    auto requires_device_runtime = [](luisa::string_view ptx) noexcept {
        constexpr luisa::string_view device_runtime_symbols[]{
            "kernel_launcher",               // indirect dispatch
            "cudaGraphSetConditional",       // device-side graph control
            "cudaGraphLaunch",               // device-side graph launch
            "cudaGraphKernelNodeSetParam",   // device-side graph node mutation
            "cudaGraphKernelNodeSetEnabled", // device-side graph node mutation
            "cudaGraphKernelNodeSetGridDim", // device-side graph node mutation
        };
        for (auto symbol : device_runtime_symbols) {
            if (ptx.find(symbol) != luisa::string_view::npos) { return true; }
        }
        return false;
    };
    auto load_ptx = [&](const void *ptx, size_t ptx_size) noexcept {
        auto devrt = device->cudadevrt_library();
        auto requires_devrt = requires_device_runtime(
            luisa::string_view{static_cast<const char *>(ptx), ptx_size});
        if (!requires_devrt) {
            if (auto ret = cuModuleLoadData(&_module, ptx); ret != CUDA_SUCCESS) { return ret; }
            LUISA_CHECK_CUDA(cuModuleGetFunction(&_function, _module, entry));
            return CUDA_SUCCESS;
        }
        if (devrt.empty()) {
            LUISA_WARNING_WITH_LOCATION(
                "CUDA Device Runtime library is not linked. "
                "Indirect dispatch and device-side graph control will not be "
                "available for this kernel.");
            if (auto ret = cuModuleLoadData(&_module, ptx); ret != CUDA_SUCCESS) { return ret; }
            LUISA_CHECK_CUDA(cuModuleGetFunction(&_function, _module, entry));
            return CUDA_SUCCESS;
        }
        auto bind_module_functions = [&]() noexcept {
            LUISA_CHECK_CUDA(cuModuleGetFunction(&_function, _module, entry));
            if (cuModuleGetFunction(&_indirect_function, _module, "kernel_launcher") != CUDA_SUCCESS) {
                LUISA_WARNING_WITH_LOCATION(
                    "Failed to find kernel_launcher() in the PTX module. "
                    "Indirect dispatch will not be available for this kernel.");
                _indirect_function = nullptr;
            }
        };
        auto cubin_cache_name = [&]() -> luisa::string {
            if (!enable_cache || device->io() == nullptr) { return {}; }
            static constexpr auto cache_tag = luisa::string_view{"luisa-cuda-linked-cubin-v1"};
            auto ptx_hash = luisa::hash64(ptx, ptx_size, luisa::hash64_default_seed);
            auto devrt_hash = luisa::hash64(devrt.data(), devrt.size(), luisa::hash64_default_seed);
            auto cc = static_cast<uint64_t>(device->handle().compute_capability());
            auto drv = static_cast<uint64_t>(device->handle().driver_version());
            auto key = luisa::hash_combine({
                luisa::hash_value(cache_tag),
                ptx_hash,
                devrt_hash,
                cc,
                drv,
            });
            return luisa::format("cubin_{:016x}.sm{}.drv{}.bin", key, cc, drv);
        }();
        if (!cubin_cache_name.empty()) {
            if (auto stream = device->io()->read_shader_cache(cubin_cache_name);
                stream != nullptr && stream->length() > 0u) {
                luisa::vector<std::byte> cached_cubin(stream->length());
                stream->read(cached_cubin);
                if (cuModuleLoadData(&_module, cached_cubin.data()) == CUDA_SUCCESS) {
                    _module_image = std::move(cached_cubin);
                    bind_module_functions();
                    return CUDA_SUCCESS;
                }
                LUISA_WARNING_WITH_LOCATION(
                    "Cached linked CUDA cubin '{}' failed to load; relinking.",
                    cubin_cache_name);
            }
        }
        CUlinkState link_state{};
        size_t cubin_size;
        void *cubin = nullptr;
        LUISA_CHECK_CUDA(cuLinkCreate(0u, nullptr, nullptr, &link_state));
        if (!devrt.empty()) {
            LUISA_CHECK_CUDA(cuLinkAddData(link_state, CU_JIT_INPUT_LIBRARY,
                                           const_cast<char *>(device->cudadevrt_library().data()),
                                           device->cudadevrt_library().size(),
                                           "cudadevrt", 0u, nullptr, nullptr));
        }
        auto ret = cuLinkAddData(link_state, CU_JIT_INPUT_PTX,
                                 const_cast<void *>(ptx), ptx_size,
                                 "my_kernel.ptx", 0u, nullptr, nullptr);
        if (ret != CUDA_SUCCESS) {
            LUISA_CHECK_CUDA(cuLinkDestroy(link_state));
            return ret;
        }
        LUISA_CHECK_CUDA(cuLinkComplete(link_state, &cubin, &cubin_size));
        LUISA_CHECK_CUDA(cuModuleLoadData(&_module, cubin));
        // Retain the linked cubin as the module image for cross-backend
        // import: the pre-link PTX references cudadevrt symbols (through
        // kernel_launcher) and would fail a standalone (re)load.
        auto image = static_cast<const std::byte *>(cubin);
        _module_image.assign(image, image + cubin_size);
        bind_module_functions();
        LUISA_CHECK_CUDA(cuLinkDestroy(link_state));
        if (!cubin_cache_name.empty()) {
            static_cast<void>(device->io()->write_shader_cache(
                cubin_cache_name,
                luisa::span<const std::byte>{_module_image.data(), _module_image.size()}));
        }
        return CUDA_SUCCESS;
    };

    auto ret = load_ptx(ptx.data(), ptx.size());
    if (ret == CUDA_ERROR_UNSUPPORTED_PTX_VERSION) {
        CUDAShader::_patch_ptx_version(ptx);
        ret = load_ptx(ptx.data(), ptx.size());
    }
    LUISA_CHECK_CUDA(ret);
    if (_module_image.empty()) {
        // Plain-PTX load path (no cudadevrt link): retain the loaded (and
        // possibly version-patched) PTX text for cross-backend import.
        _module_image = std::move(ptx);
    }
}

CUDAShaderNative::~CUDAShaderNative() noexcept {
    LUISA_CHECK_CUDA(cuModuleUnload(_module));
}

void CUDAShaderNative::_launch(CUDACommandEncoder &encoder, ShaderDispatchCommand *command) const noexcept {

    static thread_local std::array<std::byte, 65536u> argument_buffer;// should be enough

    auto argument_buffer_offset = static_cast<size_t>(0u);
    auto allocate_argument = [&](size_t bytes) noexcept {
        static constexpr auto cuda_shader_native_alignment = 16u;
        auto offset = (argument_buffer_offset + cuda_shader_native_alignment - 1u) / cuda_shader_native_alignment * cuda_shader_native_alignment;
        LUISA_ASSERT(offset + bytes <= argument_buffer.size(),
                     "Too many arguments in ShaderDispatchCommand");
        argument_buffer_offset = offset + bytes;
        return argument_buffer.data() + offset;
    };

    size_t arg_index = 0u;
    auto encode_argument = [this, &arg_index, &allocate_argument, &encoder, command](const auto &arg) noexcept {
        using Tag = ShaderDispatchCommand::Argument::Tag;
        const auto current_arg_index = arg_index++;
        switch (arg.tag) {
            case Tag::BUFFER: {
                if (reinterpret_cast<const CUDABufferBase *>(arg.buffer.handle)->is_indirect()) {
                    auto buffer = reinterpret_cast<const CUDAIndirectDispatchBuffer *>(arg.buffer.handle);
                    auto binding = buffer->binding(arg.buffer.offset, arg.buffer.size);
                    auto ptr = allocate_argument(sizeof(binding));
                    std::memcpy(ptr, &binding, sizeof(binding));
                } else {
                    auto buffer = reinterpret_cast<const CUDABuffer *>(arg.buffer.handle);
                    auto binding = buffer->binding(arg.buffer.offset, arg.buffer.size);
                    auto ptr = allocate_argument(sizeof(binding));
                    std::memcpy(ptr, &binding, sizeof(binding));
                }
                break;
            }
            case Tag::TEXTURE: {
                auto texture = reinterpret_cast<const CUDATexture *>(arg.texture.handle);
                const auto meta_mask = texture_sampler_mask(current_arg_index);
                const auto sampler_mask = static_cast<uint16_t>(meta_mask & CUDAShaderMetadata::texture_sampler_mask_bits);
                const auto needs_surface = (meta_mask & CUDAShaderMetadata::texture_uses_surface_bit) != 0u;
                auto binding = texture->binding(arg.texture.level, sampler_mask, needs_surface, encoder.stream()->handle());
                auto ptr = allocate_argument(sizeof(binding));
                std::memcpy(ptr, &binding, sizeof(binding));
                break;
            }
            case Tag::UNIFORM: {
                auto uniform = command->uniform(arg.uniform);
                LUISA_ASSERT(arg.uniform.alignment <= 16u, "Invalid uniform alignment {}.", arg.uniform.alignment);
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
                  // A *fallback* accel argument carries the fallback's own heap
                  // ABI (`LCFallbackAccel`, cuda_builtin/
                  // cuda_device_fallback_rtx.h): the bindless heap of the TLAS
                  // - whose slots hold the region buffers its traversal resolves
                  // - the heap slot of the tree's own region, and the device
                  // address of the tree's instance slice.  They are resolved
                  // here, on every dispatch, because the fallback storage may
                  // have grown and the heap may have been replaced since the
                  // last one.
                  if (encoder.stream()->device()->owns_fallback_accel(arg.accel.handle)) {
                      struct alignas(16) FallbackAccelArgument {
                          uint64_t heap_slots;
                          uint64_t heap_capacity;
                          uint64_t instances;
                          uint32_t region_slot;
                          uint32_t pad;
                      };
                      static_assert(sizeof(FallbackAccelArgument) == 32u,
                                    "the fallback accel argument is 32 bytes");
                      auto binding = encoder.stream()->device()->fallback_rtx()->binding(arg.accel.handle);
                      // Not built yet: bind nulls, which the traversal reports as
                      // a miss (see `lc_fallback_trace_closest`).
                      FallbackAccelArgument argument{0ull, 0ull, 0ull, 0u, 0u};
                      if (binding.valid()) {
                          auto *heap = reinterpret_cast<const CUDABindlessArray *>(binding.accel_heap);
                          auto heap_binding = heap->binding();
                          argument.heap_slots = heap_binding.slots;
                          argument.heap_capacity = heap_binding.capacity;
                          argument.instances =
                              reinterpret_cast<const CUDABuffer *>(binding.instance_buffer)->device_address() +
                              binding.instance_offset_bytes;
                          argument.region_slot = binding.accel_slot;
                      }
                      auto ptr = allocate_argument(sizeof(argument));
                      std::memcpy(ptr, &argument, sizeof(argument));
                      break;
                  }
                auto accel = reinterpret_cast<const CUDAAccel *>(arg.accel.handle);
                auto binding = accel->binding();
                auto ptr = allocate_argument(sizeof(binding));
                std::memcpy(ptr, &binding, sizeof(binding));
                break;
            }
        }
    };
    for (auto &&arg : _bound_arguments) { encode_argument(arg); }
    for (auto &&arg : command->arguments()) { encode_argument(arg); }
    // printer
    if (printer()) {
        auto b = printer()->encode(encoder);
        auto ptr = allocate_argument(sizeof(b));
        std::memcpy(ptr, &b, sizeof(b));
    }
    // launch
    auto cuda_stream = encoder.stream()->handle();
    auto *device = encoder.stream()->device();
    auto &device_handle = device->handle();
    if (command->is_indirect()) {
        LUISA_ASSERT(_indirect_function != nullptr,
                     "Indirect dispatch is not supported by this shader.");
        auto indirect = command->indirect_dispatch();
        auto indirect_buffer = reinterpret_cast<const CUDAIndirectDispatchBuffer *>(indirect.handle);
        auto indirect_binding = indirect_buffer->binding(indirect.offset, indirect.max_dispatch_size);
        void *arguments[] = {argument_buffer.data(), &indirect_binding};
        static constexpr auto block_size = 64u;
        auto block_count = (indirect_binding.capacity - indirect_binding.offset + block_size - 1u) / block_size;
        // This launch stays a plain `cuLaunchKernel` and is never relaxed: the
        // grid is a cudadevrt launcher that does not wait for a programmatic
        // dependency, and the indirect buffer it reads is normally written by the
        // predecessor, so starting it early would read an unlaunched dispatch
        // descriptor. It is counted as a strict launch for the same reason, and
        // it does not become a usable predecessor either: a child grid launched
        // from the device may still be running when the launcher retires, so
        // `note_pdl_grid_launched()` is deliberately not called here.
        LUISA_CHECK_CUDA(cuLaunchKernel(
            _indirect_function,
            block_count, 1u, 1u,
            block_size, 1u, 1u,
            0u, cuda_stream, arguments, nullptr));
        device->note_pdl_launch(false);
    } else {
        // the last argument is the launch size
        auto ptr = allocate_argument(sizeof(uint4));
        auto single_dispatch_size = make_uint3(0u);
        luisa::span<const uint3> dispatch_sizes;
        if (command->is_multiple_dispatch()) {
            dispatch_sizes = command->dispatch_sizes();
        } else {
            single_dispatch_size = command->dispatch_size();
            dispatch_sizes = luisa::span{&single_dispatch_size, 1u};
        }
        auto kernel_id = 0u;
        for (auto dispatch_size : dispatch_sizes) {
            if (any(dispatch_size == make_uint3(0u))) {
                ++kernel_id;
                continue;
            }
            auto launch_size_and_kernel_id = make_uint4(dispatch_size, kernel_id);
            std::memcpy(ptr, &launch_size_and_kernel_id, sizeof(launch_size_and_kernel_id));
            // launch configuration
            auto block_size = make_uint3(_block_size[0], _block_size[1], _block_size[2]);
            auto blocks = (dispatch_size + block_size - 1u) / block_size;
            // `kernelParams` is the array of pointers to the parameter values
            // itself (one entry per kernel parameter); a DSL kernel has exactly
            // one (the packed argument buffer), just like the indirect path
            // above.  Passing the address of a local that holds the array base
            // instead would make the driver read every parameter after the
            // first out of unrelated stack memory.
            void *arguments[] = {argument_buffer.data()};
            auto relax_stream_serialization =
                cuda_pdl_detail::should_relax_stream_serialization({
                    .shader_waits_for_dependency = uses_grid_dependency_synchronize(),
                    .predecessor_is_kernel = encoder.pdl_predecessor_is_kernel(),
                    .stream_policy_enabled = device->pdl_stream_enabled(),
                    .driver_accepts_attribute = _pdl_launch_enabled.load(std::memory_order_relaxed),
                    .compute_capability = device_handle.compute_capability(),
                    .driver_version = device_handle.driver_version(),
                });
            auto launch = [&](bool relax) noexcept {
                return cuda_shader_native_detail::launch_kernel(
                    _function, blocks, block_size, 0u, cuda_stream, arguments, relax);
            };
            auto error = launch(relax_stream_serialization);
            // The launch the driver actually accepted. A refused attribute is
            // re-issued through the classic entry point, so it must be counted (and
            // reported) as a strict launch even though the predicate said relax.
            auto relaxed = relax_stream_serialization;
            if (error != CUDA_SUCCESS && relax_stream_serialization &&
                (error == CUDA_ERROR_INVALID_VALUE || error == CUDA_ERROR_NOT_SUPPORTED)) {
                // The driver refused the attribute, so nothing was enqueued and the
                // identical launch can be re-issued through the unchanged path.
                // Latch the attribute off for this shader only, so the fallback
                // costs one failed call in total. Every other error (in particular
                // CUDA_ERROR_LAUNCH_FAILED) goes through LUISA_CHECK_CUDA below,
                // because the grid may already have run.
                _pdl_launch_enabled.store(false, std::memory_order_relaxed);
                device->note_pdl_driver_fallback();
                LUISA_WARNING_WITH_LOCATION(
                    "Driver rejected CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION; "
                    "falling back to cuLaunchKernel for this shader.");
                relaxed = false;
                error = launch(false);
            }
            LUISA_CHECK_CUDA(error);
            // The grid was enqueued: a later launch of this submission may be
            // relaxed against it. The counter distinguishes the two entry points
            // for the A/B comparison.
            encoder.note_pdl_grid_launched();
            device->note_pdl_launch(relaxed);
            ++kernel_id;
        }
    }
}

}// namespace luisa::compute::cuda
