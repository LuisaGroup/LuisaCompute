#include <luisa/core/pool.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/rhi/pixel.h>
#include "metal_buffer.h"
#include "metal_texture.h"
#include "metal_accel.h"
#include "metal_mesh.h"
#include "metal_curve.h"
#include "metal_shader.h"
#include "metal_procedural_primitive.h"
#include "metal_bindless_array.h"
#include "metal_command_encoder.h"

namespace luisa::compute::metal {

class UserCallbackContext : public MetalCallbackContext {

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

MetalCommandEncoder::MetalCommandEncoder(MetalStream *stream) noexcept
    : _stream{stream} {}

void MetalCommandEncoder::_prepare_command_buffer() noexcept {
    if (_command_buffer == nullptr) {
        auto desc = MTL::CommandBufferDescriptor::alloc()->init();
        desc->setRetainedReferences(false);
#ifndef NDEBUG
        desc->setErrorOptions(MTL::CommandBufferErrorOptionEncoderExecutionStatus);
#else
        desc->setErrorOptions(MTL::CommandBufferErrorOptionNone);
#endif
        _command_buffer = _stream->queue()->commandBuffer(desc);
        desc->release();
    }
}

void MetalCommandEncoder::add_callback(MetalCallbackContext *cb) noexcept {
    _callbacks.emplace_back(cb);
}

MTL::CommandBuffer *MetalCommandEncoder::command_buffer() noexcept {
    _prepare_command_buffer();
    return _command_buffer;
}

MTL::CommandBuffer *MetalCommandEncoder::submit(CommandList::CallbackContainer &&user_callbacks) noexcept {
    if (!user_callbacks.empty()) {
        add_callback(UserCallbackContext::create(
            std::move(user_callbacks)));
    }
    auto callbacks = std::exchange(_callbacks, {});
    if (!callbacks.empty()) { _prepare_command_buffer(); }
    auto command_buffer = std::exchange(_command_buffer, nullptr);
    if (command_buffer != nullptr) {
        _stream->submit(command_buffer, std::move(callbacks));
    }
    return command_buffer;
}

void MetalCommandEncoder::visit(BufferUploadCommand *command) noexcept {
    _prepare_command_buffer();
    auto buffer = reinterpret_cast<const MetalBuffer *>(command->handle())->handle();
    auto offset = command->offset();
    auto size = command->size();
    auto data = command->data();
    with_upload_buffer(size, [&](MetalStageBufferPool::Allocation *upload_buffer) noexcept {
        auto p = static_cast<std::byte *>(upload_buffer->buffer()->contents()) +
                 upload_buffer->offset();
        std::memcpy(p, data, size);
        auto encoder = _command_buffer->blitCommandEncoder();
        encoder->copyFromBuffer(upload_buffer->buffer(),
                                upload_buffer->offset(),
                                buffer, offset, size);
        encoder->endEncoding();
    });
}

void MetalCommandEncoder::visit(BufferDownloadCommand *command) noexcept {
    _prepare_command_buffer();
    auto buffer = reinterpret_cast<const MetalBuffer *>(command->handle())->handle();
    auto offset = command->offset();
    auto size = command->size();
    auto data = command->data();
    with_download_buffer(size, [&](MetalStageBufferPool::Allocation *download_buffer) noexcept {
        auto encoder = _command_buffer->blitCommandEncoder();
        encoder->copyFromBuffer(buffer, offset,
                                download_buffer->buffer(),
                                download_buffer->offset(), size);
        encoder->endEncoding();
        // copy from download buffer to user buffer
        // TODO: use a better way to pass data back to CPU
        add_callback(FunctionCallbackContext::create([download_buffer, data, size] {
            std::memcpy(data, download_buffer->data(), size);
        }));
    });
}

void MetalCommandEncoder::visit(BufferCopyCommand *command) noexcept {
    _prepare_command_buffer();
    auto src_buffer = reinterpret_cast<const MetalBuffer *>(command->src_handle())->handle();
    auto dst_buffer = reinterpret_cast<const MetalBuffer *>(command->dst_handle())->handle();
    auto src_offset = command->src_offset();
    auto dst_offset = command->dst_offset();
    auto size = command->size();
    auto encoder = _command_buffer->blitCommandEncoder();
    encoder->copyFromBuffer(src_buffer, src_offset, dst_buffer, dst_offset, size);
    encoder->endEncoding();
}

void MetalCommandEncoder::visit(BufferToTextureCopyCommand *command) noexcept {
    _prepare_command_buffer();
    auto buffer = reinterpret_cast<const MetalBuffer *>(command->buffer())->handle();
    auto buffer_offset = command->buffer_offset();
    auto texture = reinterpret_cast<const MetalTexture *>(command->texture())->handle();
    auto texture_level = command->level();
    auto size = command->size();
    auto pitch_size = pixel_storage_size(command->storage(), make_uint3(size.x, 1u, 1u));
    auto image_size = pixel_storage_size(command->storage(), make_uint3(size.xy(), 1u));
    auto encoder = _command_buffer->blitCommandEncoder();
    encoder->copyFromBuffer(buffer, buffer_offset, pitch_size, image_size,
                            MTL::Size{size.x, size.y, size.z},
                            texture, 0u, texture_level,
                            MTL::Origin{0u, 0u, 0u});
    encoder->endEncoding();
}

void MetalCommandEncoder::visit(ShaderDispatchCommand *command) noexcept {
    _prepare_command_buffer();
    auto shader = reinterpret_cast<const MetalShader *>(command->handle());
    shader->launch(*this, command);
}

void MetalCommandEncoder::visit(TextureUploadCommand *command) noexcept {
    _prepare_command_buffer();
    auto texture = reinterpret_cast<const MetalTexture *>(command->handle())->handle();
    auto level = command->level();
    auto size = command->size();
    auto data = command->data();
    auto storage = command->storage();
    auto pitch_size = pixel_storage_size(command->storage(), make_uint3(size.x, 1u, 1u));
    auto image_size = pixel_storage_size(command->storage(), make_uint3(size.xy(), 1u));
    auto total_size = image_size * size.z;
    with_upload_buffer(total_size, [&](MetalStageBufferPool::Allocation *upload_buffer) noexcept {
        auto p = static_cast<std::byte *>(upload_buffer->buffer()->contents()) +
                 upload_buffer->offset();
        std::memcpy(p, data, total_size);
        auto encoder = _command_buffer->blitCommandEncoder();
        encoder->copyFromBuffer(upload_buffer->buffer(), upload_buffer->offset(),
                                pitch_size, image_size, MTL::Size{size.x, size.y, size.z},
                                texture, 0u, level, MTL::Origin{0u, 0u, 0u});
        encoder->endEncoding();
    });
}

void MetalCommandEncoder::visit(TextureDownloadCommand *command) noexcept {
    _prepare_command_buffer();
    auto texture = reinterpret_cast<const MetalTexture *>(command->handle())->handle();
    auto level = command->level();
    auto size = command->size();
    auto data = command->data();
    auto storage = command->storage();
    auto pitch_size = pixel_storage_size(command->storage(), make_uint3(size.x, 1u, 1u));
    auto image_size = pixel_storage_size(command->storage(), make_uint3(size.xy(), 1u));
    auto total_size = image_size * size.z;
    with_download_buffer(total_size, [&](MetalStageBufferPool::Allocation *download_buffer) noexcept {
        auto encoder = _command_buffer->blitCommandEncoder();
        encoder->copyFromTexture(texture, 0u, level,
                                 MTL::Origin{0u, 0u, 0u},
                                 MTL::Size{size.x, size.y, size.z},
                                 download_buffer->buffer(),
                                 download_buffer->offset(),
                                 pitch_size, image_size);
        encoder->endEncoding();
        // copy from download buffer to user buffer
        // TODO: use a better way to pass data back to CPU
        add_callback(FunctionCallbackContext::create([download_buffer, data, total_size] {
            std::memcpy(data, download_buffer->data(), total_size);
        }));
    });
}

void MetalCommandEncoder::visit(TextureCopyCommand *command) noexcept {
    _prepare_command_buffer();
    auto src_texture = reinterpret_cast<const MetalTexture *>(command->src_handle())->handle();
    auto dst_texture = reinterpret_cast<const MetalTexture *>(command->dst_handle())->handle();
    auto src_level = command->src_level();
    auto dst_level = command->dst_level();
    auto storage = command->storage();
    auto size = command->size();
    auto encoder = _command_buffer->blitCommandEncoder();
    encoder->copyFromTexture(src_texture, 0u, src_level,
                             MTL::Origin{0u, 0u, 0u},
                             MTL::Size{size.x, size.y, size.z},
                             dst_texture, 0u, dst_level,
                             MTL::Origin{0u, 0u, 0u});
    encoder->endEncoding();
}

void MetalCommandEncoder::visit(TextureToBufferCopyCommand *command) noexcept {
    _prepare_command_buffer();
    auto texture = reinterpret_cast<const MetalTexture *>(command->texture())->handle();
    auto texture_level = command->level();
    auto buffer = reinterpret_cast<const MetalBuffer *>(command->buffer())->handle();
    auto buffer_offset = command->buffer_offset();
    auto size = command->size();
    auto pitch_size = pixel_storage_size(command->storage(), make_uint3(size.x, 1u, 1u));
    auto image_size = pixel_storage_size(command->storage(), make_uint3(size.xy(), 1u));
    auto encoder = _command_buffer->blitCommandEncoder();
    encoder->copyFromTexture(texture, 0u, texture_level,
                             MTL::Origin{0u, 0u, 0u},
                             MTL::Size{size.x, size.y, size.z},
                             buffer, buffer_offset,
                             pitch_size, image_size);
    encoder->endEncoding();
}

void MetalCommandEncoder::visit(AccelBuildCommand *command) noexcept {
    _prepare_command_buffer();
    auto accel = reinterpret_cast<MetalAccel *>(command->handle());
    accel->build(*this, command);
}

void MetalCommandEncoder::visit(MeshBuildCommand *command) noexcept {
    _prepare_command_buffer();
    auto mesh = reinterpret_cast<MetalMesh *>(command->handle());
    mesh->build(*this, command);
}

void MetalCommandEncoder::visit(CurveBuildCommand *command) noexcept {
    _prepare_command_buffer();
    auto curve = reinterpret_cast<MetalCurve *>(command->handle());
    curve->build(*this, command);
}

void MetalCommandEncoder::visit(ProceduralPrimitiveBuildCommand *command) noexcept {
    _prepare_command_buffer();
    auto prim = reinterpret_cast<MetalProceduralPrimitive *>(command->handle());
    prim->build(*this, command);
}

void MetalCommandEncoder::visit(MotionInstanceBuildCommand *command) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void MetalCommandEncoder::visit(BindlessArrayUpdateCommand *command) noexcept {
    _prepare_command_buffer();
    auto bindless_array = reinterpret_cast<MetalBindlessArray *>(command->handle());
    bindless_array->update(*this, command);
}

void MetalCommandEncoder::visit(CustomCommand *command) noexcept {
    _prepare_command_buffer();
    LUISA_ERROR_WITH_LOCATION(
        "Custom command (uuid = 0x{:04x}) is not "
        "supported in Metal backend.",
        command->uuid());
}

}// namespace luisa::compute::metal

