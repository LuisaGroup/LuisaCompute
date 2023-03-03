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

void CUDACommandEncoder::visit(const BufferUploadCommand *command) noexcept {
    auto buffer = command->handle() + command->offset();
    auto data = command->data();
    auto size = command->size();
    _stream->with_upload_buffer(size, [&](auto upload_buffer) noexcept {
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

void CUDACommandEncoder::visit(const ShaderDispatchCommand *command) noexcept {
    reinterpret_cast<CUDAShader *>(command->handle())->launch(_stream, command);
}

void CUDACommandEncoder::visit(const BufferToTextureCopyCommand *command) noexcept {
    auto mipmap_array = reinterpret_cast<CUDAMipmapArray *>(command->texture());
    auto array = mipmap_array->level(command->level());
    CUDA_MEMCPY3D copy{};
    auto pitch = pixel_storage_size(command->storage(), make_uint3(command->size().x, 1u, 1u));
    auto height = pixel_storage_size(command->storage(), make_uint3(command->size().xy(), 1u)) / pitch;
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = command->buffer() + command->buffer_offset();
    copy.srcPitch = pitch;
    copy.srcHeight = height;
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = array;
    copy.WidthInBytes = pitch;
    copy.Height = height;
    copy.Depth = command->size().z;
    LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
}

void CUDACommandEncoder::visit(const TextureUploadCommand *command) noexcept {
    auto mipmap_array = reinterpret_cast<CUDAMipmapArray *>(command->handle());
    auto array = mipmap_array->level(command->level());
    CUDA_MEMCPY3D copy{};
    auto pitch = pixel_storage_size(command->storage(), make_uint3(command->size().x, 1u, 1u));
    auto height = pixel_storage_size(command->storage(), make_uint3(command->size().xy(), 1u)) / pitch;
    auto size_bytes = pixel_storage_size(command->storage(), command->size());
    auto data = command->data();
    _stream->with_upload_buffer(size_bytes, [&](auto upload_buffer) noexcept {
        std::memcpy(upload_buffer->address(), data, size_bytes);
        copy.srcMemoryType = CU_MEMORYTYPE_HOST;
        copy.srcHost = upload_buffer->address();
        copy.srcPitch = pitch;
        copy.srcHeight = height;
        copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copy.dstArray = array;
        copy.WidthInBytes = pitch;
        copy.Height = height;
        copy.Depth = command->size().z;
        LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
    });
}

void CUDACommandEncoder::visit(const TextureDownloadCommand *command) noexcept {
    auto mipmap_array = reinterpret_cast<CUDAMipmapArray *>(command->handle());
    auto array = mipmap_array->level(command->level());
    CUDA_MEMCPY3D copy{};
    auto pitch = pixel_storage_size(command->storage(), make_uint3(command->size().x, 1u, 1u));
    auto height = pixel_storage_size(command->storage(), make_uint3(command->size().xy(), 1u)) / pitch;
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = array;
    copy.WidthInBytes = pitch;
    copy.Height = height;
    copy.Depth = command->size().z;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = command->data();
    copy.dstPitch = pitch;
    copy.dstHeight = height;
    LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
}

void CUDACommandEncoder::visit(const TextureCopyCommand *command) noexcept {
    auto src_mipmap_array = reinterpret_cast<CUDAMipmapArray *>(command->src_handle());
    auto dst_mipmap_array = reinterpret_cast<CUDAMipmapArray *>(command->dst_handle());
    auto src_array = src_mipmap_array->level(command->src_level());
    auto dst_array = dst_mipmap_array->level(command->dst_level());
    auto pitch = pixel_storage_size(command->storage(), make_uint3(command->size().x, 1u, 1u));
    auto height = pixel_storage_size(command->storage(), make_uint3(command->size().xy(), 1u)) / pitch;
    CUDA_MEMCPY3D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = src_array;
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = dst_array;
    copy.WidthInBytes = pitch;
    copy.Height = height;
    copy.Depth = command->size().z;
    LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
}

void CUDACommandEncoder::visit(const TextureToBufferCopyCommand *command) noexcept {
    auto mipmap_array = reinterpret_cast<CUDAMipmapArray *>(command->texture());
    auto array = mipmap_array->level(command->level());
    CUDA_MEMCPY3D copy{};
    auto pitch = pixel_storage_size(command->storage(), make_uint3(command->size().x, 1u, 1u));
    auto height = pixel_storage_size(command->storage(), make_uint3(command->size().xy(), 1u)) / pitch;
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = array;
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstDevice = command->buffer() + command->buffer_offset();
    copy.dstPitch = pitch;
    copy.dstHeight = height;
    copy.WidthInBytes = pitch;
    copy.Height = height;
    copy.Depth = command->size().z;
    LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
}

void CUDACommandEncoder::visit(const AccelBuildCommand *command) noexcept {
    auto accel = reinterpret_cast<CUDAAccel *>(command->handle());
    accel->build(_stream, command);
}

void CUDACommandEncoder::visit(const MeshBuildCommand *command) noexcept {
    auto mesh = reinterpret_cast<CUDAMesh *>(command->handle());
    mesh->build(_stream, command);
}

void CUDACommandEncoder::visit(const BindlessArrayUpdateCommand *command) noexcept {
    auto bindless_array = reinterpret_cast<CUDABindlessArray *>(command->handle());
    bindless_array->update(_stream, command->modifications());
}

void CUDACommandEncoder::visit(const ProceduralPrimitiveBuildCommand *command) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

void CUDACommandEncoder::visit(const CustomCommand *command) noexcept {
    LUISA_ERROR_WITH_LOCATION("Custom command '{}' is not supported on CUDA.",
                              command->name());
}

void CUDACommandEncoder::visit(const DrawRasterSceneCommand *command) noexcept {
    LUISA_ERROR_WITH_LOCATION("Rasterization is not supported on CUDA.");
}

void CUDACommandEncoder::visit(const ClearDepthCommand *command) noexcept {
    LUISA_ERROR_WITH_LOCATION("Depth clear is not supported on CUDA.");
}

}// namespace luisa::compute::cuda
