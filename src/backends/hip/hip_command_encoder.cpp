//
// Created by mike on 1/9/26.
//

#include <luisa/core/pool.h>

#include "hip_check.h"
#include "hip_buffer.h"
#include "hip_texture.h"
#include "hip_bindless_array.h"
#include "hip_command_encoder.h"
#include "hip_mesh.h"
#include "hip_curve.h"
#include "hip_motion_instance.h"
#include "hip_procedural_primitive.h"
#include "hip_accel.h"
#include "hip_shader.h"

#include "luisa/core/logging.h"

namespace luisa::compute::hip {

class UserCallbackContext : public HIPCallbackContext {

public:
    using CallbackContainer = CommandList::CallbackContainer;

private:
    CallbackContainer _functions;

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

void HIPCommandEncoder::commit(CommandList::CallbackContainer &&user_callbacks) noexcept {
    if (!user_callbacks.empty()) {
        _callbacks.emplace_back(
            UserCallbackContext::create(
                std::move(user_callbacks)));
    }
    if (auto callbacks = std::move(_callbacks); !callbacks.empty()) {
        _stream->callback(std::move(callbacks));
    }
}

void HIPCommandEncoder::visit(BufferUploadCommand *command) noexcept {
    LUISA_DEBUG_ASSERT(command->handle() != invalid_resource_handle);
    auto buffer = reinterpret_cast<const HIPBuffer *>(command->handle());
    auto binding = buffer->binding(command->offset(), command->size());
    auto data = command->data();
    auto size = command->size();
    with_upload_buffer(size, [&](auto upload_buffer) noexcept {
        std::memcpy(upload_buffer->address(), data, size);
        LUISA_CHECK_HIP(hipMemcpyHtoDAsync(
            binding.ptr, upload_buffer->address(),
            size, _stream->handle()));
    });
}

void HIPCommandEncoder::visit(BufferDownloadCommand *command) noexcept {
    LUISA_DEBUG_ASSERT(command->handle() != invalid_resource_handle);
    auto buffer = reinterpret_cast<const HIPBuffer *>(command->handle());
    auto binding = buffer->binding(command->offset(), command->size());
    auto data = command->data();
    auto size = command->size();
    with_download_pool_no_fallback(size, [&](auto download_buffer) noexcept {
        if (download_buffer) {
            LUISA_CHECK_HIP(hipMemcpyDtoHAsync(
                download_buffer->address(), binding.ptr,
                size, _stream->handle()));
            LUISA_CHECK_HIP(hipMemcpyAsync(
                data, download_buffer->address(),
                size, hipMemcpyHostToHost, _stream->handle()));
        } else {
            LUISA_CHECK_HIP(hipMemcpyDtoHAsync(
                data, binding.ptr, size, _stream->handle()));
        }
    });
}

void HIPCommandEncoder::visit(BufferCopyCommand *command) noexcept {
    LUISA_DEBUG_ASSERT(command->src_handle() != invalid_resource_handle);
    LUISA_DEBUG_ASSERT(command->dst_handle() != invalid_resource_handle);
    auto src_buffer = reinterpret_cast<const HIPBuffer *>(command->src_handle());
    auto dst_buffer = reinterpret_cast<const HIPBuffer *>(command->dst_handle());
    auto size = command->size();
    auto src_binding = src_buffer->binding(command->src_offset(), size);
    auto dst_binding = dst_buffer->binding(command->dst_offset(), size);
    LUISA_CHECK_HIP(hipMemcpyDtoDAsync(dst_binding.ptr, src_binding.ptr, size, _stream->handle()));
}

namespace {

void memcpy_buffer_to_texture(HIPBuffer::Binding buffer,
                              hipArray_t array, PixelStorage array_storage,
                              uint3 array_size, hipStream_t stream) noexcept {
    HIP_MEMCPY3D copy{};
    auto pitch = pixel_storage_size(array_storage, make_uint3(array_size.x, 1u, 1u));
    auto height = pixel_storage_size(array_storage, make_uint3(array_size.xy(), 1u)) / pitch;
    copy.srcMemoryType = hipMemoryTypeDevice;
    copy.srcDevice = buffer.ptr;
    copy.srcPitch = pitch;
    copy.srcHeight = height;
    copy.dstMemoryType = hipMemoryTypeArray;
    copy.dstArray = array;
    copy.WidthInBytes = pitch;
    copy.Height = height;
    copy.Depth = array_size.z;
    LUISA_CHECK_HIP(hipDrvMemcpy3DAsync(&copy, stream));
}

}// namespace

void HIPCommandEncoder::visit(BufferToTextureCopyCommand *command) noexcept {
    LUISA_ASSERT(command->texture() != invalid_resource_handle);
    LUISA_ASSERT(command->buffer() != invalid_resource_handle);
    auto texture = reinterpret_cast<HIPTexture *>(command->texture());
    auto buffer = reinterpret_cast<const HIPBuffer *>(command->buffer());
    auto copy_size_bytes = pixel_storage_size(command->storage(), command->size());
    auto buffer_binding = buffer->binding(command->buffer_offset(), copy_size_bytes);
    auto array = texture->level(command->level());
    memcpy_buffer_to_texture(buffer_binding, array, command->storage(),
                             command->size(), _stream->handle());
}

void HIPCommandEncoder::visit(ShaderDispatchCommand *command) noexcept {
    reinterpret_cast<HIPShader *>(command->handle())->launch(*this, command);
}

void HIPCommandEncoder::visit(TextureUploadCommand *command) noexcept {
    LUISA_ASSERT(command->handle() != invalid_resource_handle);
    auto texture = reinterpret_cast<HIPTexture *>(command->handle());
    auto array = texture->level(command->level());
    HIP_MEMCPY3D copy{};
    auto pitch = pixel_storage_size(command->storage(), make_uint3(command->size().x, 1u, 1u));
    auto height = pixel_storage_size(command->storage(), make_uint3(command->size().xy(), 1u)) / pitch;
    auto size_bytes = pixel_storage_size(command->storage(), command->size());
    auto data = command->data();
    with_upload_buffer(size_bytes, [&](auto upload_buffer) noexcept {
        std::memcpy(upload_buffer->address(), data, size_bytes);
        copy.srcMemoryType = hipMemoryTypeHost;
        copy.srcHost = upload_buffer->address();
        copy.srcPitch = pitch;
        copy.srcHeight = height;
        copy.dstMemoryType = hipMemoryTypeArray;
        copy.dstArray = array;
        copy.WidthInBytes = pitch;
        copy.Height = height;
        copy.Depth = command->size().z;
        LUISA_CHECK_HIP(hipDrvMemcpy3DAsync(&copy, _stream->handle()));
    });
}

void HIPCommandEncoder::visit(TextureDownloadCommand *command) noexcept {
    LUISA_ASSERT(command->handle() != invalid_resource_handle);
    auto texture = reinterpret_cast<HIPTexture *>(command->handle());
    auto array = texture->level(command->level());
    HIP_MEMCPY3D copy{};
    auto pitch = pixel_storage_size(command->storage(), make_uint3(command->size().x, 1u, 1u));
    auto height = pixel_storage_size(command->storage(), make_uint3(command->size().xy(), 1u)) / pitch;
    auto size_bytes = pixel_storage_size(command->storage(), command->size());
    copy.srcMemoryType = hipMemoryTypeArray;
    copy.srcArray = array;
    copy.WidthInBytes = pitch;
    copy.Height = height;
    copy.Depth = command->size().z;
    copy.dstMemoryType = hipMemoryTypeHost;
    copy.dstPitch = pitch;
    copy.dstHeight = height;
    with_download_pool_no_fallback(size_bytes, [&](auto download_buffer) noexcept {
        if (download_buffer) {
            copy.dstHost = download_buffer->address();
            LUISA_CHECK_HIP(hipDrvMemcpy3DAsync(&copy, _stream->handle()));
            LUISA_CHECK_HIP(hipMemcpyAsync(
                command->data(), download_buffer->address(),
                size_bytes, hipMemcpyHostToHost, _stream->handle()));
        } else {
            copy.dstHost = command->data();
            LUISA_CHECK_HIP(hipDrvMemcpy3DAsync(&copy, _stream->handle()));
        }
    });
}

void HIPCommandEncoder::visit(TextureCopyCommand *command) noexcept {
    LUISA_ASSERT(command->src_handle() != invalid_resource_handle);
    LUISA_ASSERT(command->dst_handle() != invalid_resource_handle);
    auto src_texture = reinterpret_cast<HIPTexture *>(command->src_handle());
    auto dst_texture = reinterpret_cast<HIPTexture *>(command->dst_handle());
    auto src_array = src_texture->level(command->src_level());
    auto dst_array = dst_texture->level(command->dst_level());
    HIP_MEMCPY3D copy{};
    auto pitch = pixel_storage_size(command->storage(), make_uint3(command->size().x, 1u, 1u));
    auto height = pixel_storage_size(command->storage(), make_uint3(command->size().xy(), 1u)) / pitch;
    copy.srcMemoryType = hipMemoryTypeArray;
    copy.srcArray = src_array;
    copy.dstMemoryType = hipMemoryTypeArray;
    copy.dstArray = dst_array;
    copy.WidthInBytes = pitch;
    copy.Height = height;
    copy.Depth = command->size().z;
    LUISA_CHECK_HIP(hipDrvMemcpy3DAsync(&copy, _stream->handle()));
}

void HIPCommandEncoder::visit(TextureToBufferCopyCommand *command) noexcept {
    LUISA_ASSERT(command->texture() != invalid_resource_handle);
    LUISA_ASSERT(command->buffer() != invalid_resource_handle);
    auto texture = reinterpret_cast<HIPTexture *>(command->texture());
    auto buffer = reinterpret_cast<const HIPBuffer *>(command->buffer());
    auto copy_size_bytes = pixel_storage_size(command->storage(), command->size());
    auto buffer_binding = buffer->binding(command->buffer_offset(), copy_size_bytes);
    auto array = texture->level(command->level());
    HIP_MEMCPY3D copy{};
    auto pitch = pixel_storage_size(command->storage(), make_uint3(command->size().x, 1u, 1u));
    auto height = pixel_storage_size(command->storage(), make_uint3(command->size().xy(), 1u)) / pitch;
    copy.srcMemoryType = hipMemoryTypeArray;
    copy.srcArray = array;
    copy.dstMemoryType = hipMemoryTypeDevice;
    copy.dstDevice = buffer_binding.ptr;
    copy.dstPitch = pitch;
    copy.dstHeight = height;
    copy.WidthInBytes = pitch;
    copy.Height = height;
    copy.Depth = command->size().z;
    LUISA_CHECK_HIP(hipDrvMemcpy3DAsync(&copy, _stream->handle()));
}

void HIPCommandEncoder::visit(AccelBuildCommand *command) noexcept {
    auto accel = reinterpret_cast<HIPAccel *>(command->handle());
    accel->build(*this, command);
    if (!command->update_instance_buffer_only()) {
        _stream->mark_hiprt_build_submitted();
    }
}

void HIPCommandEncoder::visit(MeshBuildCommand *command) noexcept {
    auto mesh = reinterpret_cast<HIPMesh *>(command->handle());
    mesh->build(*this, command);
    _stream->mark_hiprt_build_submitted();
}

void HIPCommandEncoder::visit(CurveBuildCommand *command) noexcept {
    auto curve = reinterpret_cast<HIPCurve *>(command->handle());
    curve->build(*this, command);
    _stream->mark_hiprt_build_submitted();
}

void HIPCommandEncoder::visit(ProceduralPrimitiveBuildCommand *command) noexcept {
    auto prim = reinterpret_cast<HIPProceduralPrimitive *>(command->handle());
    prim->build(*this, command);
    _stream->mark_hiprt_build_submitted();
}

void HIPCommandEncoder::visit(MotionInstanceBuildCommand *command) noexcept {
    auto instance = reinterpret_cast<HIPMotionInstance *>(command->handle());
    instance->build(*this, command);
    _stream->mark_hiprt_build_submitted();
}

void HIPCommandEncoder::visit(BindlessArrayUpdateCommand *command) noexcept {
    auto array = reinterpret_cast<HIPBindlessArray *>(command->handle());
    array->update(*this, command);
}

void HIPCommandEncoder::visit(CustomCommand *) noexcept {
}

}// namespace luisa::compute::hip
