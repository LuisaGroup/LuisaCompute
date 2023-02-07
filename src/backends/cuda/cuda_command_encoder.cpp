//
// Created by Mike on 8/1/2021.
//

#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_mesh.h>
#include <backends/cuda/cuda_accel.h>
#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_device.h>
#include <backends/cuda/cuda_shader.h>
#include <backends/cuda/cuda_host_buffer_pool.h>
#include <backends/cuda/cuda_mipmap_array.h>
#include <backends/cuda/cuda_bindless_array.h>
#include <backends/cuda/cuda_command_encoder.h>

namespace luisa::compute::cuda {

template<typename F>
inline void CUDACommandEncoder::with_upload_buffer(size_t size, F &&f) noexcept {
    auto upload_buffer = _stream->upload_pool()->allocate(size);
    f(upload_buffer);
    _stream->emplace_callback(upload_buffer);
}

void CUDACommandEncoder::visit(const BufferUploadCommand *command) noexcept {
    auto buffer = command->handle() + command->offset();
    auto data = command->data();
    auto size = command->size();
    with_upload_buffer(size, [&](auto upload_buffer) noexcept {
        std::memcpy(upload_buffer->address(), data, size);
        LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(
            buffer, upload_buffer->address(),
            size, _stream->handle()));
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
    auto pitch = pixel_size * command->size().x;
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = command->buffer() + command->buffer_offset();
    copy.srcPitch = pitch;
    copy.srcHeight = command->size().y;
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = array;
    copy.WidthInBytes = pitch;
    copy.Height = command->size().y;
    copy.Depth = command->size().z;
    LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
}

void CUDACommandEncoder::visit(const ShaderDispatchCommand *command) noexcept {
    reinterpret_cast<CUDAShader *>(command->handle())->launch(_stream, command);
}

void CUDACommandEncoder::visit(const TextureUploadCommand *command) noexcept {
    auto mipmap_array = reinterpret_cast<CUDAMipmapArray *>(command->handle());
    auto array = mipmap_array->level(command->level());
    CUDA_MEMCPY3D copy{};
    auto pixel_size = pixel_storage_size(command->storage());
    auto pitch = pixel_size * command->size().x;
    auto data = command->data();
    auto size_bytes = pitch * command->size().y * command->size().z;
    with_upload_buffer(size_bytes, [&](auto upload_buffer) noexcept {
        std::memcpy(upload_buffer->address(), data, size_bytes);
        copy.srcMemoryType = CU_MEMORYTYPE_HOST;
        copy.srcHost = upload_buffer->address();
        copy.srcPitch = pitch;
        copy.srcHeight = command->size().y;
        copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copy.dstArray = array;
        copy.WidthInBytes = pitch;
        copy.Height = command->size().y;
        copy.Depth = command->size().z;
        LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
    });
}

void CUDACommandEncoder::visit(const TextureDownloadCommand *command) noexcept {
    auto mipmap_array = reinterpret_cast<CUDAMipmapArray *>(command->handle());
    auto array = mipmap_array->level(command->level());
    CUDA_MEMCPY3D copy{};
    auto pixel_size = pixel_storage_size(command->storage());
    auto pitch = pixel_size * command->size().x;
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = array;
    copy.WidthInBytes = pitch;
    copy.Height = command->size().y;
    copy.Depth = command->size().z;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = command->data();
    copy.dstPitch = pitch;
    copy.dstHeight = command->size().y;
    LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
}

void CUDACommandEncoder::visit(const TextureCopyCommand *command) noexcept {
    auto src_mipmap_array = reinterpret_cast<CUDAMipmapArray *>(command->src_handle());
    auto dst_mipmap_array = reinterpret_cast<CUDAMipmapArray *>(command->dst_handle());
    auto src_array = src_mipmap_array->level(command->src_level());
    auto dst_array = dst_mipmap_array->level(command->dst_level());
    auto pixel_size = pixel_format_size(src_mipmap_array->format());
    CUDA_MEMCPY3D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = src_array;
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = dst_array;
    copy.WidthInBytes = command->size().x * pixel_size;
    copy.Height = command->size().y;
    copy.Depth = command->size().z;
    LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
}

void CUDACommandEncoder::visit(const TextureToBufferCopyCommand *command) noexcept {
    auto mipmap_array = reinterpret_cast<CUDAMipmapArray *>(command->texture());
    auto array = mipmap_array->level(command->level());
    CUDA_MEMCPY3D copy{};
    auto pixel_size = pixel_storage_size(command->storage());
    auto pitch = pixel_size * command->size().x;
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = array;
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstDevice = command->buffer() + command->buffer_offset();
    copy.dstPitch = pitch;
    copy.dstHeight = command->size().y;
    copy.WidthInBytes = pitch;
    copy.Height = command->size().y;
    copy.Depth = command->size().z;
    LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
}

void CUDACommandEncoder::visit(const AccelBuildCommand *command) noexcept {
    auto accel = reinterpret_cast<CUDAAccel *>(command->handle());
    accel->build(_device, _stream, command);
}

void CUDACommandEncoder::visit(const MeshBuildCommand *command) noexcept {
    auto mesh = reinterpret_cast<CUDAMesh *>(command->handle());
    mesh->build(_device, _stream, command);
}

void CUDACommandEncoder::visit(const BindlessArrayUpdateCommand *command) noexcept {
    auto bindless_array = reinterpret_cast<CUDABindlessArray *>(command->handle());
    bindless_array->upload(_stream);
}

}// namespace luisa::compute::cuda
