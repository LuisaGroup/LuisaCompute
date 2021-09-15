//
// Created by Mike on 8/1/2021.
//

#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_buffer.h>
#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_texture.h>
#include <backends/cuda/cuda_command_encoder.h>

namespace luisa::compute::cuda {

struct RingBufferRecycleContext {
    std::span<std::byte> buffer;
    CUDAStream *stream{nullptr};
    RingBufferRecycleContext(std::span<std::byte> b, CUDAStream *s) noexcept
        : buffer{b}, stream{s} {}
};

[[nodiscard]] decltype(auto) ring_buffer_recycle_context_pool() noexcept {
    static ArenaPool<RingBufferRecycleContext> pool{Arena::global()};
    return (pool);
}

void CUDACommandEncoder::visit(const BufferUploadCommand *command) noexcept {
    auto buffer = reinterpret_cast<CUDABuffer *>(command->handle())->handle() + command->offset();
    auto data = command->data();
    auto size = command->size();
    auto upload_buffer = _stream->upload_pool().allocate(size);
    std::memcpy(upload_buffer.data(), data, size);
    LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(buffer, upload_buffer.data(), size, _stream->handle()));
    LUISA_CHECK_CUDA(cuLaunchHostFunc(
        _stream->handle(), [](void *user_data) noexcept {
            auto context = static_cast<RingBufferRecycleContext *>(user_data);
            context->stream->upload_pool().recycle(context->buffer);
            ring_buffer_recycle_context_pool().recycle(context);
        },
        ring_buffer_recycle_context_pool().create(upload_buffer, _stream)));
}

void CUDACommandEncoder::visit(const BufferDownloadCommand *command) noexcept {
    auto buffer = reinterpret_cast<CUDABuffer *>(command->handle())->handle() + command->offset();
    auto data = command->data();
    auto size = command->size();
    LUISA_CHECK_CUDA(cuMemcpyDtoHAsync(data, buffer, size, _stream->handle()));
}

void CUDACommandEncoder::visit(const BufferCopyCommand *command) noexcept {
    auto src_buffer = reinterpret_cast<CUDABuffer *>(command->src_handle())->handle() + command->src_offset();
    auto dst_buffer = reinterpret_cast<CUDABuffer *>(command->dst_handle())->handle() + command->dst_offset();
    auto size = command->size();
    LUISA_CHECK_CUDA(cuMemcpyDtoDAsync(dst_buffer, src_buffer, size, _stream->handle()));
}

void CUDACommandEncoder::visit(const BufferToTextureCopyCommand *command) noexcept {
    auto buffer = reinterpret_cast<CUDABuffer *>(command->buffer());
    auto texture = reinterpret_cast<CUDATexture *>(command->texture());

}

void CUDACommandEncoder::visit(const ShaderDispatchCommand *command) noexcept {
}
void CUDACommandEncoder::visit(const TextureUploadCommand *command) noexcept {
}
void CUDACommandEncoder::visit(const TextureDownloadCommand *command) noexcept {
}
void CUDACommandEncoder::visit(const TextureCopyCommand *command) noexcept {
}
void CUDACommandEncoder::visit(const TextureToBufferCopyCommand *command) noexcept {
}
void CUDACommandEncoder::visit(const AccelUpdateCommand *command) noexcept {
}
void CUDACommandEncoder::visit(const AccelBuildCommand *command) noexcept {
}
void CUDACommandEncoder::visit(const MeshUpdateCommand *command) noexcept {
}
void CUDACommandEncoder::visit(const MeshBuildCommand *command) noexcept {
}

}// namespace luisa::compute::cuda
