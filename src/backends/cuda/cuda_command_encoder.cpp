//
// Created by Mike on 8/1/2021.
//

#include <runtime/command_list.h>
#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_buffer.h>
#include <backends/cuda/cuda_mesh.h>
#include <backends/cuda/cuda_procedural_primitive.h>
#include <backends/cuda/cuda_accel.h>
#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_device.h>
#include <backends/cuda/cuda_shader.h>
#include <backends/cuda/cuda_host_buffer_pool.h>
#include <backends/cuda/cuda_texture.h>
#include <backends/cuda/cuda_bindless_array.h>
#include <backends/cuda/cuda_dstorage.h>
#include <backends/cuda/cuda_command_encoder.h>

namespace luisa::compute::cuda {

class UserCallbackContext : public CUDACallbackContext {

public:
    using CallbackContainer = CommandList::CallbackContainer;

private:
    CallbackContainer _functions;

private:
    [[nodiscard]] static auto &_object_pool() noexcept {
        static Pool<UserCallbackContext, true> pool;
        return pool;
    }

public:
    explicit UserCallbackContext(CallbackContainer &&cbs) noexcept
        : _functions{std::move(cbs)} {}

    [[nodiscard]] static auto create(CallbackContainer &&cbs) noexcept {
        return _object_pool().create(std::move(cbs));
    }

    void recycle() noexcept override {
        for (auto &&f : _functions) { f(); }
        _object_pool().destroy(this);
    }
};

class MemcpyCallbackContext : public CUDACallbackContext {

private:
    void *_dst;
    const void *_src;
    size_t _size;

private:
    [[nodiscard]] static auto &_object_pool() noexcept {
        static Pool<MemcpyCallbackContext, true> pool;
        return pool;
    }

public:
    MemcpyCallbackContext(void *dst, const void *src, size_t size) noexcept
        : _dst{dst}, _src{src}, _size{size} {}

    [[nodiscard]] static auto create(void *dst, const void *src, size_t size) noexcept {
        return _object_pool().create(dst, src, size);
    }

    void recycle() noexcept override {
        std::memcpy(_dst, _src, _size);
        _object_pool().destroy(this);
    }
};

void CUDACommandEncoder::commit(CommandList::CallbackContainer &&user_callbacks) noexcept {
    if (!user_callbacks.empty()) {
        _callbacks.emplace_back(
            UserCallbackContext::create(
                std::move(user_callbacks)));
    }
    if (auto callbacks = std::move(_callbacks); !callbacks.empty()) {
        _stream->callback(std::move(callbacks));
    }
}

void CUDACommandEncoder::visit(BufferUploadCommand *command) noexcept {
    auto buffer = reinterpret_cast<const CUDABuffer *>(command->handle());
    auto address = buffer->handle() + command->offset();
    auto data = command->data();
    auto size = command->size();
    with_upload_buffer(size, [&](auto upload_buffer) noexcept {
        std::memcpy(upload_buffer->address(), data, size);
        LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(
            address, upload_buffer->address(),
            size, _stream->handle()));
    });
}

void CUDACommandEncoder::visit(BufferDownloadCommand *command) noexcept {
    auto buffer = reinterpret_cast<const CUDABuffer *>(command->handle());
    auto address = buffer->handle() + command->offset();
    auto data = command->data();
    auto size = command->size();
    LUISA_CHECK_CUDA(cuMemcpyDtoHAsync(data, address, size, _stream->handle()));
}

void CUDACommandEncoder::visit(BufferCopyCommand *command) noexcept {
    auto src_buffer = reinterpret_cast<const CUDABuffer *>(command->src_handle())->handle() +
                      command->src_offset();
    auto dst_buffer = reinterpret_cast<const CUDABuffer *>(command->dst_handle())->handle() +
                      command->dst_offset();
    auto size = command->size();
    LUISA_CHECK_CUDA(cuMemcpyDtoDAsync(dst_buffer, src_buffer, size, _stream->handle()));
}

void CUDACommandEncoder::visit(ShaderDispatchCommand *command) noexcept {
    reinterpret_cast<CUDAShader *>(command->handle())->launch(*this, command);
}

void CUDACommandEncoder::visit(BufferToTextureCopyCommand *command) noexcept {
    auto mipmap_array = reinterpret_cast<CUDATexture *>(command->texture());
    auto array = mipmap_array->level(command->level());
    CUDA_MEMCPY3D copy{};
    auto pitch = pixel_storage_size(command->storage(), make_uint3(command->size().x, 1u, 1u));
    auto height = pixel_storage_size(command->storage(), make_uint3(command->size().xy(), 1u)) / pitch;
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = reinterpret_cast<const CUDABuffer *>(command->buffer())->handle() +
                     command->buffer_offset();
    copy.srcPitch = pitch;
    copy.srcHeight = height;
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = array;
    copy.WidthInBytes = pitch;
    copy.Height = height;
    copy.Depth = command->size().z;
    LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
}

void CUDACommandEncoder::visit(TextureUploadCommand *command) noexcept {
    auto mipmap_array = reinterpret_cast<CUDATexture *>(command->handle());
    auto array = mipmap_array->level(command->level());
    CUDA_MEMCPY3D copy{};
    auto pitch = pixel_storage_size(command->storage(), make_uint3(command->size().x, 1u, 1u));
    auto height = pixel_storage_size(command->storage(), make_uint3(command->size().xy(), 1u)) / pitch;
    auto size_bytes = pixel_storage_size(command->storage(), command->size());
    auto data = command->data();
    with_upload_buffer(size_bytes, [&](auto upload_buffer) noexcept {
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

void CUDACommandEncoder::visit(TextureDownloadCommand *command) noexcept {
    auto mipmap_array = reinterpret_cast<CUDATexture *>(command->handle());
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

void CUDACommandEncoder::visit(TextureCopyCommand *command) noexcept {
    auto src_mipmap_array = reinterpret_cast<CUDATexture *>(command->src_handle());
    auto dst_mipmap_array = reinterpret_cast<CUDATexture *>(command->dst_handle());
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

void CUDACommandEncoder::visit(TextureToBufferCopyCommand *command) noexcept {
    auto mipmap_array = reinterpret_cast<CUDATexture *>(command->texture());
    auto array = mipmap_array->level(command->level());
    CUDA_MEMCPY3D copy{};
    auto pitch = pixel_storage_size(command->storage(), make_uint3(command->size().x, 1u, 1u));
    auto height = pixel_storage_size(command->storage(), make_uint3(command->size().xy(), 1u)) / pitch;
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = array;
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstDevice = reinterpret_cast<const CUDABuffer *>(command->buffer())->handle() +
                     command->buffer_offset();
    copy.dstPitch = pitch;
    copy.dstHeight = height;
    copy.WidthInBytes = pitch;
    copy.Height = height;
    copy.Depth = command->size().z;
    LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
}

void CUDACommandEncoder::visit(AccelBuildCommand *command) noexcept {
    auto accel = reinterpret_cast<CUDAAccel *>(command->handle());
    accel->build(*this, command);
}

void CUDACommandEncoder::visit(MeshBuildCommand *command) noexcept {
    auto mesh = reinterpret_cast<CUDAMesh *>(command->handle());
    mesh->build(*this, command);
}

void CUDACommandEncoder::visit(ProceduralPrimitiveBuildCommand *command) noexcept {
    auto primitive = reinterpret_cast<CUDAProceduralPrimitive *>(command->handle());
    primitive->build(*this, command);
}

void CUDACommandEncoder::visit(BindlessArrayUpdateCommand *command) noexcept {
    auto bindless_array = reinterpret_cast<CUDABindlessArray *>(command->handle());
    bindless_array->update(*this, command);
}

void CUDACommandEncoder::visit(CustomCommand *command) noexcept {
    switch (command->uuid()) {
        case to_underlying(CustomCommandUUID::DSTORAGE_READ):
            visit(static_cast<DStorageReadCommand *>(command));
            break;
        default:
            LUISA_ERROR_WITH_LOCATION("Custom command (UUID = 0x{:04x}) "
                                      "is not supported on CUDA.",
                                      command->uuid());
    }
}

void CUDACommandEncoder::visit(DStorageReadCommand *command) noexcept {
    LUISA_ASSERT(command->compression() == DStorageCompression::None,
                 "DStorageReadCommand does not support compression.");

    auto ret = luisa::visit(
        [](auto src) noexcept {
            using T = std::remove_cvref_t<decltype(src)>;
            auto request_offset_bytes = src.offset_bytes;
            auto request_size_bytes = src.size_bytes;
            if (std::is_same_v<T, DStorageReadCommand::FileSource>) {
                auto source = reinterpret_cast<CUDAMappedFile *>(src.handle);
                LUISA_ASSERT(request_offset_bytes < source->size_bytes() &&
                                 request_size_bytes <= source->size_bytes() - request_offset_bytes,
                             "DStorageReadCommand out of range.");
                return std::make_tuple(source->device_address() + request_offset_bytes,
                                       static_cast<const std::byte *>(source->mapped_pointer()) + request_offset_bytes,
                                       request_size_bytes);
            } else if (std::is_same_v<T, DStorageReadCommand::MemorySource>) {
                auto source = reinterpret_cast<CUDAPinnedMemory *>(src.handle);
                LUISA_ASSERT(request_offset_bytes < source->size_bytes() &&
                                 request_size_bytes <= source->size_bytes() - request_offset_bytes,
                             "DStorageReadCommand out of range.");
                return std::make_tuple(source->device_address() + request_offset_bytes,
                                       static_cast<const std::byte *>(source->host_pointer()) + request_offset_bytes,
                                       request_size_bytes);
            } else {
                LUISA_ERROR_WITH_LOCATION("Unreachable.");
            }
        },
        command->source());

    // clang doesn't like structured bindings in lambda capture
    auto device_ptr = std::get<0>(ret);
    auto host_ptr = std::get<1>(ret);
    auto size_bytes = std::get<2>(ret);

    luisa::visit(
        [size_bytes, device_ptr, host_ptr, this](auto dst) noexcept {
            using T = std::remove_cvref_t<decltype(dst)>;
            if constexpr (std::is_same_v<T, DStorageReadCommand::BufferRequest>) {
                auto buffer = reinterpret_cast<const CUDABuffer *>(dst.handle);
                LUISA_ASSERT(dst.offset_bytes < buffer->size() &&
                                 size_bytes <= buffer->size() - dst.offset_bytes,
                             "DStorageReadCommand out of range.");
                auto dst_addr = buffer->handle() + dst.offset_bytes;
                if (dst.size_bytes != size_bytes) {
                    LUISA_WARNING_WITH_LOCATION(
                        "DStorageReadCommand size mismatch: "
                        "request size = {}, "
                        "actual size = {}.",
                        dst.size_bytes, size_bytes);
                }
                auto valid_size = std::min(dst.size_bytes, size_bytes);
                LUISA_CHECK_CUDA(cuMemcpyDtoDAsync(dst_addr, device_ptr, valid_size, _stream->handle()));
            } else if constexpr (std::is_same_v<T, DStorageReadCommand::TextureRequest>) {
                auto texture = reinterpret_cast<const CUDATexture *>(dst.handle);
                auto size = make_uint3(dst.size[0], dst.size[1], dst.size[2]);
                LUISA_ASSERT(all(size == max(texture->size() >> dst.level, 1u)),
                             "DStorageReadCommand size mismatch.");
                auto array = texture->level(dst.level);
                CUDA_MEMCPY3D copy{};
                auto pitch = pixel_storage_size(texture->storage(), make_uint3(size.x, 1u, 1u));
                auto height = pixel_storage_size(texture->storage(), make_uint3(size.xy(), 1u)) / pitch;
                copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                copy.srcDevice = device_ptr;
                copy.srcPitch = pitch;
                copy.srcHeight = height;
                copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
                copy.dstArray = array;
                copy.WidthInBytes = pitch;
                copy.Height = height;
                copy.Depth = size.z;
                LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
            } else if constexpr (std::is_same_v<T, DStorageReadCommand::MemoryRequest>) {
                auto p = reinterpret_cast<std::byte *>(dst.data);
                if (dst.size_bytes != size_bytes) {
                    LUISA_WARNING_WITH_LOCATION(
                        "DStorageReadCommand size mismatch: "
                        "request size = {}, "
                        "actual size = {}.",
                        dst.size_bytes, size_bytes);
                }
                auto valid_size = std::min(dst.size_bytes, size_bytes);
                add_callback(MemcpyCallbackContext::create(p, host_ptr, valid_size));
            } else {
                LUISA_ERROR_WITH_LOCATION("Unreachable.");
            }
        },
        command->request());
}

}// namespace luisa::compute::cuda
