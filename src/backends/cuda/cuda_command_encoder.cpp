//
// Created by Mike on 8/1/2021.
//

#include <core/allocator.h>
#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_mipmap_array.h>
#include <backends/cuda/cuda_bindless_array.h>
#include <backends/cuda/cuda_command_encoder.h>

namespace luisa::compute::cuda {

struct RingBufferRecycleContext {
    std::span<std::byte> buffer;
    CUDAStream *stream{nullptr};
    RingBufferRecycleContext(std::span<std::byte> b, CUDAStream *s) noexcept
        : buffer{b}, stream{s} {}
};

[[nodiscard]] decltype(auto) ring_buffer_recycle_context_pool() noexcept {
    static Pool<RingBufferRecycleContext> pool;
    return (pool);
}

template<typename F>
inline void CUDACommandEncoder::with_upload_buffer(size_t size, F &&f) noexcept {
    auto upload_buffer = _stream->upload_pool().allocate(size);
    auto upload_stream = _stream;
    if (upload_buffer.empty()) {
        auto temp = luisa::detail::allocator_allocate(size, 16u);
        upload_buffer = std::span{static_cast<std::byte *>(temp), size};
        upload_stream = nullptr;
    }
    f(upload_buffer);
    LUISA_CHECK_CUDA(cuLaunchHostFunc(
        _stream->handle(), [](void *user_data) noexcept {
            auto context = static_cast<RingBufferRecycleContext *>(user_data);
            if (auto stream = context->stream) {// from stream->upload_pool()
                context->stream->upload_pool().recycle(context->buffer);
            } else {// temporary memory
                luisa::detail::allocator_deallocate(context->buffer.data(), 16u);
            }
            ring_buffer_recycle_context_pool().recycle(context);
        },
        ring_buffer_recycle_context_pool().create(upload_buffer, upload_stream)));
}

void CUDACommandEncoder::visit(const BufferUploadCommand *command) noexcept {
    auto buffer = command->handle() + command->offset();
    auto data = command->data();
    auto size = command->size();
    with_upload_buffer(size, [&](std::span<std::byte> upload_buffer) noexcept {
        std::memcpy(upload_buffer.data(), data, size);
        LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(buffer, upload_buffer.data(), size, _stream->handle()));
    });
}

void CUDACommandEncoder::visit(const BufferDownloadCommand *command) noexcept {
    auto buffer = command->handle() + command->offset();
    auto data = command->data();
    auto size = command->size();
    LUISA_CHECK_CUDA(cuMemcpyDtoHAsync(data, buffer, size, _stream->handle()));
}

void CUDACommandEncoder::visit(const BufferCopyCommand *command) noexcept {
    auto src_buffer = command->src_handle() + command->src_offset();
    auto dst_buffer = command->dst_handle() + command->dst_offset();
    auto size = command->size();
    LUISA_CHECK_CUDA(cuMemcpyDtoDAsync(dst_buffer, src_buffer, size, _stream->handle()));
}

void CUDACommandEncoder::visit(const BufferToTextureCopyCommand *command) noexcept {
    auto mipmap_array = reinterpret_cast<CUDAMipmapArray *>(command->texture());
    auto array = mipmap_array->level(command->level());
    CUDA_MEMCPY3D copy{};
    auto pixel_size = pixel_storage_size(command->storage());
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = command->buffer() + command->buffer_offset();
    copy.srcPitch = pixel_size * command->size().x;
    copy.srcHeight = command->size().y;
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = array;
    LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
}

void CUDACommandEncoder::visit(const ShaderDispatchCommand *command) noexcept {
    auto kernel = reinterpret_cast<CUfunction>(command->handle());
    auto launch_size = command->dispatch_size();
    auto block_size = command->kernel().block_size();
    auto blocks = (launch_size + block_size - 1u) / block_size;
    static thread_local std::array<std::byte, 4096u> argument_buffer;
    static thread_local std::vector<void *> arguments;
    auto argument_buffer_offset = static_cast<size_t>(0u);
    auto allocate_argument = [&](size_t bytes) noexcept {
        static constexpr auto alignment = 16u;
        auto offset = (argument_buffer_offset + alignment - 1u) / alignment * alignment;
        argument_buffer_offset = offset + bytes;
        return arguments.emplace_back(argument_buffer.data() + offset);
    };
    arguments.clear();
    arguments.reserve(32u);
    command->decode([&](auto, auto argument) noexcept -> void {
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
            auto ptr = allocate_argument(sizeof(CUdeviceptr));
            auto array = reinterpret_cast<CUDABindlessArray *>(argument.handle);
            auto desc_buffer = array->handle();
            std::memcpy(ptr, &desc_buffer, sizeof(CUdeviceptr));
        } else if constexpr (std::is_same_v<T, ShaderDispatchCommand::AccelArgument>) {
            // TODO...
        } else {// uniform
            static_assert(std::same_as<T, std::span<const std::byte>>);
            auto ptr = allocate_argument(argument.size_bytes());
            std::memcpy(ptr, argument.data(), argument.size_bytes());
        }
    });
    // the last one is always the launch size
    auto ptr = allocate_argument(sizeof(luisa::uint3));
    std::memcpy(ptr, &launch_size, sizeof(luisa::uint3));
    LUISA_VERBOSE_WITH_LOCATION(
        "Dispatching shader #{} with {} argument(s) "
        "in ({}, {}, {}) blocks of size ({}, {}, {}).",
        command->handle(), arguments.size(),
        blocks.x, blocks.y, blocks.z,
        block_size.x, block_size.y, block_size.z);
    LUISA_CHECK_CUDA(cuLaunchKernel(
        kernel,
        blocks.x, blocks.y, blocks.z,
        block_size.x, block_size.y, block_size.z,
        0u, _stream->handle(),
        arguments.data(), nullptr));
}

void CUDACommandEncoder::visit(const TextureUploadCommand *command) noexcept {
    auto mipmap_array = reinterpret_cast<CUDAMipmapArray *>(command->handle());
    auto array = mipmap_array->level(command->level());
    CUDA_MEMCPY3D copy{};
    auto pixel_size = pixel_storage_size(command->storage());
    auto data = command->data();
    auto size_bytes = command->size().x * command->size().y * command->size().z * pixel_size;
    with_upload_buffer(size_bytes, [&](std::span<std::byte> upload_buffer) noexcept {
        std::memcpy(upload_buffer.data(), data, size_bytes);
        copy.srcMemoryType = CU_MEMORYTYPE_HOST;
        copy.srcHost = upload_buffer.data();
        copy.srcPitch = pixel_size * command->size().x;
        copy.srcHeight = command->size().y;
        copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copy.dstArray = array;
        LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
    });
}

void CUDACommandEncoder::visit(const TextureDownloadCommand *command) noexcept {
    auto mipmap_array = reinterpret_cast<CUDAMipmapArray *>(command->handle());
    auto array = mipmap_array->level(command->level());
    CUDA_MEMCPY3D copy{};
    auto pixel_size = pixel_storage_size(command->storage());
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = array;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = command->data();
    copy.dstPitch = pixel_size * command->size().x;
    copy.dstHeight = command->size().y;
    LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
}

void CUDACommandEncoder::visit(const TextureCopyCommand *command) noexcept {
    auto src_mipmap_array = reinterpret_cast<CUDAMipmapArray *>(command->src_handle());
    auto dst_mipmap_array = reinterpret_cast<CUDAMipmapArray *>(command->dst_handle());
    auto src_array = src_mipmap_array->level(command->src_level());
    auto dst_array = dst_mipmap_array->level(command->dst_level());
    CUDA_MEMCPY3D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = src_array;
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = dst_array;
    LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
}

void CUDACommandEncoder::visit(const TextureToBufferCopyCommand *command) noexcept {
    auto mipmap_array = reinterpret_cast<CUDAMipmapArray *>(command->texture());
    auto array = mipmap_array->level(command->level());
    CUDA_MEMCPY3D copy{};
    auto pixel_size = pixel_storage_size(command->storage());
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = array;
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstDevice = command->buffer() + command->buffer_offset();
    copy.dstPitch = pixel_size * command->size().x;
    copy.dstHeight = command->size().y;
    LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
}

void CUDACommandEncoder::visit(const AccelUpdateCommand *command) noexcept {
}
void CUDACommandEncoder::visit(const AccelBuildCommand *command) noexcept {
}
void CUDACommandEncoder::visit(const MeshUpdateCommand *command) noexcept {
}
void CUDACommandEncoder::visit(const MeshBuildCommand *command) noexcept {
}

void CUDACommandEncoder::visit(const BindlessArrayUpdateCommand *command) noexcept {
    auto array = reinterpret_cast<CUDABindlessArray *>(command->handle());
    auto size_bytes = sizeof(CUDABindlessArray::Item) * command->count();
    auto offset_bytes = sizeof(CUDABindlessArray::Item) * command->offset();
    with_upload_buffer(size_bytes, [&](std::span<std::byte> upload_buffer) noexcept {
        std::memcpy(upload_buffer.data(), array->slots().data() + command->offset(), size_bytes);
        LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(array->handle() + offset_bytes, upload_buffer.data(), size_bytes, _stream->handle()));
    });
}

}// namespace luisa::compute::cuda
