//
// Created by Mike on 8/1/2021.
//

#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_buffer.h>
#include <backends/cuda/cuda_command_encoder.h>

namespace luisa::compute::cuda {

void CUDACommandEncoder::visit(const BufferUploadCommand *command) noexcept {
    auto buffer = reinterpret_cast<CUDABuffer *>(command->handle())->handle() + command->offset();
    auto data = command->data();
    auto size = command->size();
    LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(buffer, data, size, _stream));
}

void CUDACommandEncoder::visit(const BufferDownloadCommand *command) noexcept {
    auto buffer = reinterpret_cast<CUDABuffer *>(command->handle())->handle() + command->offset();
    auto data = command->data();
    auto size = command->size();
    LUISA_CHECK_CUDA(cuMemcpyDtoHAsync(data, buffer, size, _stream));
}

void CUDACommandEncoder::visit(const BufferCopyCommand *command) noexcept {
    auto src_buffer = reinterpret_cast<CUDABuffer *>(command->src_handle())->handle() + command->src_offset();
    auto dst_buffer = reinterpret_cast<CUDABuffer *>(command->dst_handle())->handle() + command->dst_offset();
    auto size = command->size();
    LUISA_CHECK_CUDA(cuMemcpyDtoDAsync(dst_buffer, src_buffer, size, _stream));
}

void CUDACommandEncoder::visit(const BufferToTextureCopyCommand *command) noexcept {
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
